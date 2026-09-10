"""Runtime configuration, read from the environment."""

import os
from pathlib import Path

from .tokens import Budget

# Reservations are deliberate: the answer needs room to be written, and the
# system prompt plus the question occupy space before any excerpt is added.
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", "32768"))
RESERVED_FOR_ANSWER = int(os.environ.get("RESERVED_FOR_ANSWER", "1500"))
RESERVED_FOR_PROMPT = int(os.environ.get("RESERVED_FOR_PROMPT", "800"))

CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "400"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "60"))

# Spaces gives a writable /data only on paid tiers, so default to memory and
# let a path be supplied when one exists.
QDRANT_PATH = os.environ.get("QDRANT_PATH") or None
QDRANT_URL = os.environ.get("QDRANT_URL") or None

# Ingestion is slow relative to a web request, so it is pushed onto a queue when
# one is configured. Without REDIS_URL the app runs ingestion inline, which
# keeps a single-container deployment to a single container.
REDIS_URL = os.environ.get("REDIS_URL") or None
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/doc-agent-uploads"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "600"))
SAMPLE_PDF = Path(__file__).resolve().parents[2] / "samples" / "retrieval_systems_primer.pdf"

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))


def budget() -> Budget:
    """The context budget for one request."""
    return Budget(
        context_window=CONTEXT_WINDOW,
        reserved_for_answer=RESERVED_FOR_ANSWER,
        reserved_for_prompt=RESERVED_FOR_PROMPT,
    )
