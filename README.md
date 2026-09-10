---
title: Document Agent
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Document agent

Ask questions about a PDF and get answers grounded in it, with page citations.
The agent searches the document itself, reads what comes back, searches again
when the first attempt misses, and answers only from what it found.

Runs entirely on free tiers: embeddings execute in-process, the vector database
is embedded in the container, and the language model comes from Groq's free tier
with Gemini as a fallback.

## What it does

```
PDF ─▶ token-aware chunking ─▶ local embeddings ─▶ Qdrant (in-process)
                                                        │
question ─▶ agent ──search──▶ ────────────────────────┘
              │  ▲                    │
              │  └──── excerpts ──────┘
              │       (repeat if the first search missed)
              ▼
           answer with (p. N) citations
```

A single retrieval pass answers questions whose wording already matches the
document. It fails when the answer needs two lookups, or when the user's words
differ from the document's. Letting the model run its own searches and reconsider
covers both. Asked "what problems does RAG solve, and how does chunking relate to
it?", the agent searches twice, then cites eight pages across both halves of the
question.

## Why it is built this way

**Embeddings run locally.** FastEmbed's ONNX runtime embeds in-process, so there
is no per-request quota to exhaust and no key to leak. It also keeps the image
small: `sentence-transformers` would pull roughly 2GB of PyTorch for the same job.

**The vector database is embedded.** Qdrant runs inside the app process against a
local path, so the deployment is one container with no external service to
provision. The same client talks to a hosted cluster by passing a URL instead of
a path, so moving to managed Qdrant is a configuration change.

**Two LLM providers, tried in order.** Groq's free tier survives public traffic;
Gemini's free tier allows 20 requests per day per model, which a shared link
exhausts almost immediately. Rate limits, server errors, and stale keys all fall
through to the next provider rather than failing the request.

**Tokens are the unit of measurement.** Chunk sizes, context budgets, and limits
are counted in tokens, not characters. A 1000-character chunk of dense technical
prose and one of sparse prose differ by roughly a factor of two in tokens, so
character limits either waste context or overflow it. The UI shows the counts.

## Context budgeting

Each request divides the model's context window explicitly:

```
context window ─ answer reservation ─ prompt reservation = available for excerpts
```

Retrieved chunks are packed into that remainder in relevance order. A chunk that
does not fit is skipped rather than truncated, because half an excerpt cited as a
source is worse than one fewer source. The UI reports what was used and what was
dropped.

## Running it locally

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then put your keys in .env
python app.py
```

Then open http://localhost:7860. At least one provider key must be set.

Keys are read from `.env` in this directory, which `.gitignore` excludes. An
environment variable set in the shell wins over the file, which is what lets a
host inject secrets without a file being present.

## Deploying to Hugging Face Spaces

1. Create a Space with **Docker** as the SDK.
2. Push this repository to it.
3. In Settings → Variables and secrets, add `GROQ_API_KEY` (and `GEMINI_API_KEY`
   if you want the fallback).

The YAML block at the top of this file is what tells Spaces to build the
Dockerfile and serve port 7860. Embedding weights are downloaded during the image
build, so the first visitor does not wait for them.

## Two shapes, one codebase

Ingestion is slow relative to a web request, so the app can hand it to a queue.
The queue is optional, and which shape runs is decided by whether `REDIS_URL` is
set.

**Inline.** One container. The web process reads, chunks, embeds, and stores the
document itself, against an in-process Qdrant. This is what runs on a free
single-service host.

```
browser ──▶ app (Gradio + embeddings + Qdrant in-process)
```

**Queued.** Four services. The web process stages the upload on a shared volume
and enqueues a job; a worker embeds it and writes to a Qdrant service both can
reach. The UI streams job status while it runs.

```
browser ──▶ app ──enqueue──▶ redis ──▶ worker
                │                        │
                └────▶ qdrant ◀──────────┘
```

The second shape is not a refactor of the first, it is a consequence of it. Once
a worker runs in its own process it cannot see an index held in the web
process's memory, so the vector store has to become a service. `worker.py`
refuses to start without `QDRANT_URL` for exactly this reason.

Run the queued shape locally:

```
docker compose up --build
docker compose up --scale worker=3     # a worker pool
```

Qdrant is published on host port 6335 so the stack can run beside another
Qdrant already holding 6333. Services address each other on 6333 over the
compose network.

## Deploying to Render

The free plan gives 512MB of memory and sleeps a service after about fifteen
minutes of inactivity. [render.yaml](render.yaml) declares the settings that
keep the app inside that envelope.

1. Push this repository to GitHub.
2. On Render, create a new Blueprint from the repository. It reads
   `render.yaml` and builds the Dockerfile.
3. Add `GROQ_API_KEY` in the service's Environment tab.

### What a sleeping service does to a visitor

The URL does not go down. Render holds the incoming request, starts the
container, and serves the page once it is up, so a visitor sees a slow load
rather than an error. Measured on a laptop:

| Phase | Time |
|---|---|
| Boot to first response, warm start enabled | 20s |
| Boot to first response, warm start disabled | 11s |
| First document, when not warmed at boot | 6s |
| Every query after that | 0.02s |

Render's shared CPU is slower than a laptop, so budget 30 to 40 seconds for a
cold visit.

`WARM_ON_START=1` loads the embedding model and indexes the sample document
before the server accepts traffic. That lengthens boot but means the visitor
waits once and lands on an app that answers immediately, instead of waiting
again on their first click. Set it to `0` to get content on screen sooner and
pay the cost later.

[.github/workflows/keepalive.yml](.github/workflows/keepalive.yml) pings the
app every ten minutes so it rarely sleeps. It is disabled until you set the
`APP_URL` repository variable. Two honest caveats: a continuously pinged service
consumes the plan's monthly instance hours steadily rather than on demand, and
GitHub delays scheduled workflows when its queues are busy, so it reduces cold
starts without eliminating them.

If cold starts matter more than avoiding a card on file, Google Cloud Run with
1GB of memory removes both the memory pressure and most of the wait, and stays
inside its free monthly allowance at this traffic.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | unset | Primary provider |
| `GEMINI_API_KEY` | unset | Fallback provider |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model name |
| `CHUNK_TOKENS` | `400` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap between chunks |
| `CONTEXT_WINDOW` | `32768` | Assumed model context |
| `QDRANT_PATH` | unset | On-disk vector storage; in-memory when unset |
| `MAX_UPLOAD_MB` | `10` | Upload size cap |
| `EMBED_BATCH_SIZE` | `4` | Chunks embedded per batch; the main lever on peak memory |
| `WARM_ON_START` | `1` | Load the model and index the sample before serving |
| `REDIS_URL` | unset | Queue ingestion when set; ingest inline when not |
| `QDRANT_URL` | unset | Qdrant service address; required when a worker runs |
| `UPLOAD_DIR` | `/tmp/doc-agent-uploads` | Shared staging directory for queued uploads |
| `JOB_TIMEOUT` | `600` | Seconds before a queued job is abandoned |

## Tests

```
pip install -r requirements-dev.txt
pytest
```

Retrieval tests use the real embedding model, so they verify that semantically
distinct passages actually separate rather than trusting a mock. The agent and
provider tests use stubs and need no API key.

## Scaling notes

What is true of this deployment: the app is stateless apart from the vector
store, the container is reproducible, and both rate limits and provider outages
are handled rather than surfaced as tracebacks.

What is not: in the inline shape, each replica holds its own index, so more than
one replica requires the queued shape with a shared Qdrant. Uploaded documents
live only as long as the container. Neither is a problem at demo scale.

Measured on this machine: the inline container runs in 380MB at boot and 199MB
steady inside a 512MB limit, and ingesting the sample takes about six seconds.
Peak memory is dominated by `EMBED_BATCH_SIZE`, which is why it defaults to 4:
the same document costs 397MB of RSS at batch 4 and 705MB at batch 64.

## Limitations

- Text-based PDFs only. Scanned documents need OCR first.
- One document at a time; a new upload replaces the previous index.
- Token counts use `cl100k_base`, which is exact for GPT-class models and an
  estimate within about ten percent for Llama and Gemini.
- No conversation memory. Each question is answered independently.
