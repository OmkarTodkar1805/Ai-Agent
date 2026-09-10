"""Vector storage and retrieval, using the real local embedding model."""

import pytest

from doc_agent.chunking import Chunk
from doc_agent.store import VectorStore


@pytest.fixture(scope="module")
def store():
    """An in-memory store holding three clearly distinct passages."""
    store = VectorStore()
    store.add(
        [
            Chunk(
                text=(
                    "Retrieval augmented generation combines a search step with a "
                    "language model, so answers are grounded in retrieved documents."
                ),
                page=1,
                index=0,
            ),
            Chunk(
                text=(
                    "The mitochondrion is the organelle responsible for producing "
                    "adenosine triphosphate inside eukaryotic cells."
                ),
                page=2,
                index=1,
            ),
            Chunk(
                text=(
                    "Sourdough bread relies on wild yeast and lactic acid bacteria "
                    "to leaven the dough over many hours."
                ),
                page=3,
                index=2,
            ),
        ],
        source="test.pdf",
    )
    return store


def test_all_chunks_are_stored(store):
    assert store.count() == 3


def test_search_ranks_the_relevant_chunk_first(store):
    hits = store.search("how does grounding answers in documents work?", limit=3)

    assert hits[0].page == 1
    assert "retrieval" in hits[0].text.lower()


def test_search_discriminates_between_unrelated_topics(store):
    biology = store.search("what makes ATP in a cell?", limit=1)
    baking = store.search("how is bread leavened?", limit=1)

    assert biology[0].page == 2
    assert baking[0].page == 3


def test_scores_are_ordered_best_first(store):
    hits = store.search("retrieval augmented generation", limit=3)

    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_provenance_survives_the_round_trip(store):
    hit = store.search("retrieval augmented generation", limit=1)[0]

    assert hit.source == "test.pdf"
    assert hit.page == 1


def test_empty_store_returns_no_hits():
    assert VectorStore().search("anything") == []


def test_reset_clears_the_collection():
    store = VectorStore()
    store.add([Chunk(text="temporary content", page=1, index=0)], source="x.pdf")
    assert store.count() == 1

    store.reset()

    assert store.count() == 0


def test_small_batches_store_every_chunk():
    """Batch size is a memory lever, so it must not change what gets stored."""
    chunks = [
        Chunk(text=f"passage number {i} about retrieval systems", page=i + 1, index=i)
        for i in range(9)
    ]
    store = VectorStore()

    store.add(chunks, source="batched.pdf", batch_size=4)

    assert store.count() == 9
