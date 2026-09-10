"""Gradio front end for the document agent."""

import os
import time
from pathlib import Path

import gradio as gr

from doc_agent import config, embeddings, jobs
from doc_agent.agent import DocumentAgent
from doc_agent.embeddings import MODEL_NAME
from doc_agent.ingest import ingest_pdf
from doc_agent.providers import AllProvidersFailed, ProviderChain
from doc_agent.store import VectorStore

STORE = VectorStore(path=config.QDRANT_PATH)
CHAIN = ProviderChain()


def warm_start() -> None:
    """Load the embedding model and index the sample before serving traffic.

    A host that scales to zero holds the visitor's first request while the
    container starts, so work done here is paid during a wait the visitor is
    already in. Deferring it instead makes their first click slow, which reads
    as a broken app rather than a cold one.
    """
    if os.environ.get("WARM_ON_START", "1") != "1":
        return

    try:
        embeddings.model()

        if config.SAMPLE_PDF.is_file() and STORE.count() == 0:
            ingest_pdf(
                config.SAMPLE_PDF,
                STORE,
                max_tokens=config.CHUNK_TOKENS,
                overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
            )
    except Exception as exc:
        # A failed warm-up must not stop the app booting; the UI still works.
        print(f"warm start skipped: {exc}")

INTRO = """# Document agent

Ask questions about a PDF. The agent searches the document itself, reads what
comes back, searches again when the first attempt misses, and answers only from
what it found, citing pages.

Token counts are shown throughout, because chunk size and context budget are
what decide whether retrieval works.
"""


def status_line() -> str:
    """Describe the current state: what is loaded, and how ingestion runs."""
    providers = ", ".join(CHAIN.available) or "none configured"
    mode = "queued (Redis + RQ)" if jobs.queue_enabled() else "inline"

    return (
        f"**Chunks indexed:** {STORE.count()} | **Providers:** {providers} | "
        f"**Embeddings:** {MODEL_NAME} | **Ingestion:** {mode}"
    )


def load_sample():
    """Index the bundled PDF."""
    if not config.SAMPLE_PDF.is_file():
        yield "No sample document is bundled with this deployment.", status_line()
        return

    yield from _ingest(config.SAMPLE_PDF)


def load_upload(file_obj):
    """Index a PDF the user uploaded."""
    if file_obj is None:
        yield "Choose a PDF first.", status_line()
        return

    path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb > config.MAX_UPLOAD_MB:
        yield (
            f"{path.name} is {size_mb:.1f}MB; this demo accepts up to "
            f"{config.MAX_UPLOAD_MB}MB.",
            status_line(),
        )
        return

    yield from _ingest(path)


def _describe(report) -> str:
    """Render an ingestion report."""
    return (
        f"**{report['source']}** indexed.\n\n"
        f"- {report['pages']} pages split into {report['chunks']} chunks\n"
        f"- {report['total_tokens']:,} tokens total, {report['average_chunk_tokens']} per chunk "
        f"(target {config.CHUNK_TOKENS}, overlap {config.CHUNK_OVERLAP_TOKENS})"
    )


def _ingest_inline(path: Path):
    """Ingest in this process, as a single-container deployment does."""
    try:
        report = ingest_pdf(
            path,
            STORE,
            max_tokens=config.CHUNK_TOKENS,
            overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        )
    except ValueError as exc:
        return str(exc), status_line()

    return (
        _describe(
            {
                "source": report.source,
                "pages": report.pages,
                "chunks": report.chunks,
                "total_tokens": report.total_tokens,
                "average_chunk_tokens": report.average_chunk_tokens,
            }
        ),
        status_line(),
    )


def _ingest_queued(path: Path):
    """Hand the document to a worker and report progress until it finishes."""
    staged = config.UPLOAD_DIR / path.name
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if staged.resolve() != path.resolve():
        staged.write_bytes(path.read_bytes())

    try:
        job_id = jobs.enqueue_ingest(staged)
    except Exception as exc:
        yield f"Could not reach the queue: {exc}", status_line()
        return

    yield f"Queued as `{job_id}`. Waiting for a worker.", status_line()

    deadline = time.time() + config.JOB_TIMEOUT

    while time.time() < deadline:
        time.sleep(1.0)
        status = jobs.job_status(job_id)

        if status.state == "finished" and status.result:
            yield _describe(status.result), status_line()
            return

        if status.state == "failed":
            yield f"Ingestion failed: {status.detail}", status_line()
            return

        yield f"Job `{job_id}` is {status.detail}.", status_line()

    yield f"Job `{job_id}` did not finish within {config.JOB_TIMEOUT}s.", status_line()


def _ingest(path: Path):
    """Ingest through the queue when one is configured, otherwise inline."""
    if jobs.queue_enabled():
        yield from _ingest_queued(path)
    else:
        yield _ingest_inline(path)


def format_sources(hits) -> str:
    """Render the retrieved excerpts with page and similarity."""
    if not hits:
        return "_No excerpts were retrieved._"

    return "\n\n".join(
        f"**page {hit.page}** · similarity {hit.score:.3f}\n\n> {hit.text.strip()[:400]}"
        for hit in hits
    )


def format_trace(trace) -> str:
    """Render the agent's reasoning and the token accounting behind it."""
    lines = [f"**Provider used:** {trace.provider or 'none'}", ""]

    for step in trace.steps:
        lines.append(f"- *{step['kind']}* — {step['detail']}")

    lines += [
        "",
        f"**Searches:** {trace.searches}",
        f"**Chunks used:** {trace.chunks_seen} (dropped for budget: {trace.chunks_dropped})",
        f"**Context tokens:** {trace.context_tokens:,} of "
        f"{config.budget().available_for_context:,} available",
        f"**Model tokens:** {trace.prompt_tokens:,} in, {trace.completion_tokens:,} out",
    ]

    return "\n".join(lines)


def ask(question: str):
    """Answer one question, returning the answer, sources, and trace."""
    if not question.strip():
        return "Ask a question first.", "", ""

    if STORE.count() == 0:
        return "Load a document first, using the buttons above.", "", ""

    agent = DocumentAgent(STORE, CHAIN, config.budget())

    try:
        result = agent.ask(question)
    except AllProvidersFailed as exc:
        return f"Every language model provider failed: {exc}", "", ""

    return result.text, format_sources(result.hits), format_trace(result.trace)


with gr.Blocks(title="Document agent") as demo:
    gr.Markdown(INTRO)
    status = gr.Markdown(status_line())

    with gr.Row():
        upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
        with gr.Column():
            sample_button = gr.Button("Use the bundled sample document")
            index_button = gr.Button("Index the uploaded PDF", variant="primary")

    ingest_output = gr.Markdown()

    question = gr.Textbox(
        label="Question",
        placeholder="What problems does RAG solve, and how does chunking relate to it?",
        lines=2,
    )
    ask_button = gr.Button("Ask", variant="primary")

    answer = gr.Markdown(label="Answer")

    with gr.Accordion("Retrieved excerpts", open=False):
        sources = gr.Markdown()

    with gr.Accordion("Agent trace and token accounting", open=False):
        trace = gr.Markdown()

    sample_button.click(load_sample, outputs=[ingest_output, status])
    index_button.click(load_upload, inputs=upload, outputs=[ingest_output, status])
    ask_button.click(ask, inputs=question, outputs=[answer, sources, trace])
    question.submit(ask, inputs=question, outputs=[answer, sources, trace])


if __name__ == "__main__":
    warm_start()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
