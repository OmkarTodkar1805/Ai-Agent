"""The agent loop: decide, search, reconsider, answer.

A single retrieval pass answers a question whose wording already matches the
document. It fails on questions needing two lookups, or where the user's words
differ from the document's. Letting the model run its own searches, see what
came back, and search again covers both, at the cost of more requests.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .providers import Completion, ProviderChain
from .store import Hit, VectorStore
from .tokens import Budget, count_tokens, pack_to_budget

MAX_STEPS = 6
RETRIEVE_LIMIT = 6
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

SYSTEM_PROMPT = """You answer questions about one document by searching it.

Reply with a single JSON object and nothing else. One of:

{"action": "search", "reasoning": "<why this query>", "query": "<search terms>"}
{"action": "answer", "reasoning": "<how the excerpts support this>", "answer": "<your answer>"}

How to work:

- Start by searching. The user's wording often differs from the document's, so
  search for the concept, not their exact phrase.
- Read what comes back. If the excerpts do not cover the question, search again
  with different terms rather than guessing. Two or three searches is normal for
  a question with several parts.
- Answer only from the excerpts you were shown. Cite pages as (p. N).
- If the document does not contain the answer, say so plainly and name what is
  missing. Do not fall back on general knowledge.
"""


@dataclass
class Trace:
    """A record of what the agent did, shown in the UI so the work is visible."""

    steps: list[dict] = field(default_factory=list)
    searches: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_tokens: int = 0
    chunks_seen: int = 0
    chunks_dropped: int = 0
    provider: str = ""

    def record(self, kind: str, detail: str) -> None:
        self.steps.append({"kind": kind, "detail": detail})


@dataclass
class Answer:
    """The final answer plus the evidence and accounting behind it."""

    text: str
    hits: list[Hit]
    trace: Trace


def parse_step(raw: str) -> dict:
    """Parse the model's JSON step, tolerating code fences and stray prose."""
    cleaned = FENCE.sub("", raw.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}


def format_excerpts(hits: list[Hit]) -> str:
    """Lay out excerpts with page headers the model can cite."""
    return "\n\n---\n\n".join(f"[page {hit.page}]\n{hit.text.strip()}" for hit in hits)


class DocumentAgent:
    """Runs the search-and-answer loop over one document collection."""

    def __init__(
        self,
        store: VectorStore,
        chain: ProviderChain,
        budget: Budget,
        max_steps: int = MAX_STEPS,
        on_step: Callable[[str, str], None] = None,
    ) -> None:
        self.store = store
        self.chain = chain
        self.budget = budget
        self.max_steps = max_steps
        self._on_step = on_step or (lambda kind, detail: None)

    def _complete(self, messages: list[dict], trace: Trace) -> Completion:
        result = self.chain.complete(messages)
        trace.prompt_tokens += result.prompt_tokens
        trace.completion_tokens += result.completion_tokens
        trace.provider = result.provider
        return result

    def ask(self, question: str) -> Answer:
        """Run the loop until the model answers or the step budget runs out."""
        trace = Trace()
        collected: list[Hit] = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]

        for _ in range(self.max_steps):
            raw = self._complete(messages, trace).text
            step = parse_step(raw)
            messages.append({"role": "assistant", "content": raw})

            action = step.get("action")
            reasoning = str(step.get("reasoning", "")).strip()

            if not action:
                messages.append(
                    {"role": "user", "content": "That was not valid JSON. Send one JSON object."}
                )
                continue

            if reasoning:
                trace.record(action, reasoning)
                self._on_step(action, reasoning)

            if action == "search":
                query = str(step.get("query", "")).strip() or question
                trace.searches += 1
                self._on_step("query", query)

                hits = self.store.search(query, limit=RETRIEVE_LIMIT)
                packed = pack_to_budget(
                    hits, self.budget.available_for_context, text_of=lambda hit: hit.text
                )

                trace.chunks_seen += len(packed.included)
                trace.chunks_dropped += packed.dropped
                trace.context_tokens += packed.used_tokens
                collected.extend(packed.included)

                body = (
                    format_excerpts(packed.included)
                    if packed.included
                    else "No excerpts matched that query. Try different search terms."
                )
                messages.append({"role": "user", "content": f"Excerpts for {query!r}:\n\n{body}"})
                continue

            if action == "answer":
                return Answer(
                    text=str(step.get("answer", "")).strip(),
                    hits=_dedupe(collected),
                    trace=trace,
                )

            messages.append(
                {"role": "user", "content": 'Unknown action. Use "search" or "answer".'}
            )

        return Answer(
            text=(
                f"I could not settle on an answer within {self.max_steps} steps. "
                f"Try asking about one thing at a time."
            ),
            hits=_dedupe(collected),
            trace=trace,
        )


def _dedupe(hits: list[Hit]) -> list[Hit]:
    """Collapse repeated hits across searches, keeping the best score for each."""
    best: dict[tuple[int, str], Hit] = {}

    for hit in hits:
        key = (hit.page, hit.text[:80])

        if key not in best or hit.score > best[key].score:
            best[key] = hit

    return sorted(best.values(), key=lambda hit: hit.score, reverse=True)
