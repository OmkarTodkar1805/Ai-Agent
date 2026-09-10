"""Background ingestion through Redis and RQ.

Reading a large PDF, splitting it, and embedding every chunk takes seconds to
minutes. Holding an HTTP connection open for that is fragile, so the work is
handed to a queue and a worker process consumes it.

The queue is optional. Without REDIS_URL the app ingests inline and stays a
single process, which is what keeps a free single-container deployment viable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config
from .ingest import ingest_pdf
from .store import VectorStore

QUEUE_NAME = "ingestion"


@dataclass
class JobStatus:
    """A queued job's state, in terms the UI can render."""

    id: str
    state: str
    detail: str = ""
    result: Optional[dict] = None

    @property
    def finished(self) -> bool:
        return self.state in ("finished", "failed")


def queue_enabled() -> bool:
    """True when a Redis URL is configured."""
    return bool(config.REDIS_URL)


def connection():
    """Open a Redis connection. Imported lazily so redis stays optional."""
    from redis import Redis

    return Redis.from_url(config.REDIS_URL)


def get_queue():
    """Return the ingestion queue."""
    from rq import Queue

    return Queue(QUEUE_NAME, connection=connection())


def ingest_document(path: str, chunk_tokens: int, overlap_tokens: int) -> dict:
    """Ingest one PDF. Runs inside the worker process.

    The worker builds its own VectorStore against the Qdrant service, because
    the web process's store is not reachable from here.
    """
    store = VectorStore(path=config.QDRANT_PATH, url=config.QDRANT_URL)
    report = ingest_pdf(
        Path(path), store, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens
    )

    return {
        "source": report.source,
        "pages": report.pages,
        "chunks": report.chunks,
        "total_tokens": report.total_tokens,
        "average_chunk_tokens": report.average_chunk_tokens,
    }


def enqueue_ingest(path: Path) -> str:
    """Queue a document for ingestion and return the job id."""
    job = get_queue().enqueue(
        ingest_document,
        str(path),
        config.CHUNK_TOKENS,
        config.CHUNK_OVERLAP_TOKENS,
        job_timeout=config.JOB_TIMEOUT,
    )

    return job.id


def job_status(job_id: str) -> JobStatus:
    """Look up a job, translating RQ's states into something displayable."""
    from rq.exceptions import NoSuchJobError
    from rq.job import Job

    try:
        job = Job.fetch(job_id, connection=connection())
    except NoSuchJobError:
        return JobStatus(id=job_id, state="failed", detail="The job has expired or was never queued.")

    # RQ 2.x returns a JobStatus enum. It compares equal to its string value
    # but hashes by member, so a dict lookup misses and str() renders it as
    # "JobStatus.QUEUED". Take the value.
    raw_state = job.get_status()
    state = getattr(raw_state, "value", raw_state)

    if state == "finished":
        return JobStatus(id=job_id, state=state, result=job.return_value())

    if state == "failed":
        return JobStatus(
            id=job_id,
            state=state,
            detail=(job.latest_result().exc_string if job.latest_result() else "")
            or "The worker raised an error.",
        )

    positions = {"queued": "waiting for a worker", "started": "being processed"}

    return JobStatus(id=job_id, state=state, detail=positions.get(state, state))
