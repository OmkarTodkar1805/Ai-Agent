"""Token-aware splitting of documents into retrievable chunks."""

import re
from dataclasses import dataclass, field

from .tokens import count_tokens, encoder

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    """One retrievable passage, with the provenance needed to cite it."""

    text: str
    page: int
    index: int
    tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = count_tokens(self.text)


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Cut a passage that exceeds max_tokens on token boundaries."""
    tokens = encoder().encode(text)

    return [
        encoder().decode(tokens[start : start + max_tokens])
        for start in range(0, len(tokens), max_tokens)
    ]


def split_page(text: str, page: int, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split one page into passages of at most max_tokens, preferring paragraph breaks.

    Paragraphs are accumulated until the next one would overflow the budget.
    Overlap is taken from the tail of the previous passage so a sentence spanning
    a boundary is still retrievable from both sides.
    """
    paragraphs = [p.strip() for p in PARAGRAPH_BREAK.split(text) if p.strip()]

    if not paragraphs:
        return []

    passages: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        cost = count_tokens(paragraph)

        if cost > max_tokens:
            if current:
                passages.append("\n\n".join(current))
                current, current_tokens = [], 0

            passages.extend(_split_oversized(paragraph, max_tokens))
            continue

        if current_tokens + cost > max_tokens and current:
            passages.append("\n\n".join(current))

            tail = encoder().encode(passages[-1])[-overlap_tokens:] if overlap_tokens else []
            carry = encoder().decode(tail) if tail else ""
            current = [carry] if carry else []
            current_tokens = len(tail)

        current.append(paragraph)
        current_tokens += cost

    if current:
        passages.append("\n\n".join(current))

    return passages


def chunk_pages(
    pages: list[str],
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Turn a list of page texts into numbered chunks.

    Pages are one-indexed in the output because that is how a reader refers to
    them, whatever the loader used internally.
    """
    chunks: list[Chunk] = []

    for page_number, page_text in enumerate(pages, start=1):
        for passage in split_page(page_text, page_number, max_tokens, overlap_tokens):
            chunks.append(Chunk(text=passage, page=page_number, index=len(chunks)))

    return chunks
