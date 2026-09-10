"""Agent loop: step parsing, multi-search behaviour, and accounting."""

import json
import types

import pytest

from doc_agent.agent import DocumentAgent, parse_step
from doc_agent.chunking import Chunk
from doc_agent.providers import Completion
from doc_agent.store import VectorStore
from doc_agent.tokens import Budget


class StubChain:
    """Replays scripted model replies and records what it was sent."""

    def __init__(self, replies):
        self._replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls = []

    def complete(self, messages, temperature=0.0):
        self.calls.append([dict(m) for m in messages])
        return Completion(
            text=self._replies.pop(0), provider="stub", model="stub-1",
            prompt_tokens=10, completion_tokens=5,
        )


@pytest.fixture
def store():
    store = VectorStore()
    store.add(
        [
            Chunk(text="Chunking splits documents into passages sized for retrieval.",
                  page=2, index=0),
            Chunk(text="Embeddings map text to vectors so similar meanings sit close together.",
                  page=5, index=1),
        ],
        source="doc.pdf",
    )
    return store


@pytest.fixture
def budget():
    return Budget(context_window=8000, reserved_for_answer=1000, reserved_for_prompt=500)


def test_parses_a_plain_json_object():
    assert parse_step('{"action": "answer", "answer": "hi"}')["action"] == "answer"


def test_parses_json_wrapped_in_code_fences():
    """Models routinely wrap JSON in markdown despite instructions."""
    raw = '```json\n{"action": "search", "query": "chunking"}\n```'

    assert parse_step(raw)["query"] == "chunking"


def test_parses_json_embedded_in_prose():
    raw = 'Here is my step:\n{"action": "answer", "answer": "ok"}\nHope that helps.'

    assert parse_step(raw)["action"] == "answer"


def test_unparseable_text_yields_an_empty_step():
    assert parse_step("no json at all") == {}


def test_agent_searches_then_answers(store, budget):
    chain = StubChain([
        {"action": "search", "reasoning": "need the definition", "query": "chunking"},
        {"action": "answer", "reasoning": "excerpt covers it", "answer": "Chunking splits text (p. 2)."},
    ])
    agent = DocumentAgent(store, chain, budget)

    result = agent.ask("what is chunking?")

    assert result.text == "Chunking splits text (p. 2)."
    assert result.trace.searches == 1
    assert result.hits


def test_agent_can_search_more_than_once(store, budget):
    """A multi-part question should be allowed a second lookup."""
    chain = StubChain([
        {"action": "search", "reasoning": "part one", "query": "chunking"},
        {"action": "search", "reasoning": "part two", "query": "embeddings"},
        {"action": "answer", "reasoning": "both covered", "answer": "Both (p. 2, p. 5)."},
    ])
    agent = DocumentAgent(store, chain, budget)

    result = agent.ask("how do chunking and embeddings relate?")

    assert result.trace.searches == 2
    assert {hit.page for hit in result.hits} == {2, 5}


def test_excerpts_are_sent_back_to_the_model(store, budget):
    chain = StubChain([
        {"action": "search", "query": "embeddings"},
        {"action": "answer", "answer": "done"},
    ])
    DocumentAgent(store, chain, budget).ask("q")

    last_user_message = chain.calls[-1][-1]["content"]
    assert "page 5" in last_user_message


def test_invalid_json_is_corrected_rather_than_crashing(store, budget):
    chain = StubChain([
        "this is not json",
        {"action": "answer", "reasoning": "recovered", "answer": "fine"},
    ])
    agent = DocumentAgent(store, chain, budget)

    assert agent.ask("q").text == "fine"


def test_step_limit_stops_a_model_that_never_answers(store, budget):
    chain = StubChain([{"action": "search", "query": "chunking"}] * 3)
    agent = DocumentAgent(store, chain, budget, max_steps=3)

    result = agent.ask("q")

    assert "could not settle on an answer" in result.text
    assert result.trace.searches == 3


def test_token_usage_accumulates_across_steps(store, budget):
    chain = StubChain([
        {"action": "search", "query": "chunking"},
        {"action": "answer", "answer": "done"},
    ])

    trace = DocumentAgent(store, chain, budget).ask("q").trace

    assert trace.prompt_tokens == 20
    assert trace.completion_tokens == 10
    assert trace.provider == "stub"


def test_repeated_hits_are_deduplicated(store, budget):
    """Two searches returning the same chunk should cite it once."""
    chain = StubChain([
        {"action": "search", "query": "chunking"},
        {"action": "search", "query": "chunking"},
        {"action": "answer", "answer": "done"},
    ])

    result = DocumentAgent(store, chain, budget).ask("q")

    keys = [(hit.page, hit.text[:80]) for hit in result.hits]
    assert len(keys) == len(set(keys))


def test_a_tiny_budget_drops_excerpts_and_records_it(store):
    tiny = Budget(context_window=100, reserved_for_answer=50, reserved_for_prompt=40)
    chain = StubChain([
        {"action": "search", "query": "chunking"},
        {"action": "answer", "answer": "done"},
    ])

    trace = DocumentAgent(store, chain, tiny).ask("q").trace

    assert trace.chunks_dropped > 0
    assert trace.context_tokens <= tiny.available_for_context
