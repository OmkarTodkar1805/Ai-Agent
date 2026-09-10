"""Turn a PDF into stored, searchable chunks."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .chunking import chunk_pages
from .store import VectorStore

MAX_PAGES = 200


@dataclass
class IngestReport:
    """What a single ingestion produced, for display in the UI."""

    source: str
    pages: int
    chunks: int
    total_tokens: int

    @property
    def average_chunk_tokens(self) -> int:
        return round(self.total_tokens / self.chunks) if self.chunks else 0


def read_pdf(path: Path) -> list[str]:
    """Return the text of each page, in order."""
    reader = PdfReader(str(path))

    if len(reader.pages) > MAX_PAGES:
        raise ValueError(
            f"{path.name} has {len(reader.pages)} pages; this demo accepts up to {MAX_PAGES}."
        )

    return [page.extract_text() or "" for page in reader.pages]


def ingest_pdf(
    path: Path,
    store: VectorStore,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
    replace: bool = True,
) -> IngestReport:
    """Read, chunk, embed, and store a PDF.

    Replacing by default means a new upload answers about the new document
    rather than a mixture of both, which is what someone trying the demo
    expects.
    """
    pages = read_pdf(path)

    if not any(page.strip() for page in pages):
        raise ValueError(
            f"No text could be extracted from {path.name}. Scanned PDFs need OCR first."
        )

    chunks = chunk_pages(pages, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    if replace:
        store.reset()

    store.add(chunks, source=path.name)

    return IngestReport(
        source=path.name,
        pages=len(pages),
        chunks=len(chunks),
        total_tokens=sum(chunk.tokens for chunk in chunks),
    )
