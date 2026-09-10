"""Token-aware chunking."""

from doc_agent.chunking import chunk_pages, split_page
from doc_agent.tokens import count_tokens

PARAGRAPH = "This is a sentence about retrieval augmented generation systems. " * 8


def test_every_chunk_stays_within_the_token_limit():
    pages = ["\n\n".join([PARAGRAPH] * 6)]

    chunks = chunk_pages(pages, max_tokens=200, overlap_tokens=20)

    assert chunks
    assert all(chunk.tokens <= 200 for chunk in chunks)


def test_a_single_oversized_paragraph_is_split_not_dropped():
    giant = "token " * 2000

    passages = split_page(giant, page=1, max_tokens=100, overlap_tokens=0)

    assert len(passages) > 1
    assert all(count_tokens(p) <= 100 for p in passages)


def test_pages_are_numbered_from_one():
    chunks = chunk_pages(["first page text", "second page text"], max_tokens=100)

    assert [c.page for c in chunks] == [1, 2]


def test_chunk_indexes_are_sequential_across_pages():
    chunks = chunk_pages([PARAGRAPH, PARAGRAPH], max_tokens=60, overlap_tokens=10)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_text_between_chunks():
    pages = ["\n\n".join([PARAGRAPH] * 4)]

    with_overlap = chunk_pages(pages, max_tokens=150, overlap_tokens=40)
    without_overlap = chunk_pages(pages, max_tokens=150, overlap_tokens=0)

    assert sum(c.tokens for c in with_overlap) > sum(c.tokens for c in without_overlap)


def test_empty_pages_produce_no_chunks():
    assert chunk_pages(["", "   ", "\n\n"]) == []


def test_token_count_is_recorded_on_each_chunk():
    chunks = chunk_pages([PARAGRAPH], max_tokens=100)

    assert all(chunk.tokens == count_tokens(chunk.text) for chunk in chunks)
