"""Token accounting: counting, budgeting, and packing text into a context window.

Chunk sizes and context limits are measured in tokens, not characters, because
that is the unit models actually charge and truncate on. A 1000-character chunk
of dense technical prose and one of sparse prose differ by a factor of two in
tokens, so character-based limits either waste budget or overflow it.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import tiktoken

# cl100k_base is the encoding used by GPT-4-class models. Llama and Gemini
# tokenise differently, so counts here are an estimate for those, accurate to
# roughly ten percent. That is close enough for budgeting and far better than
# counting characters.
ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def encoder() -> tiktoken.Encoding:
    """Return the shared tokeniser, loaded once."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in text."""
    return len(encoder().encode(text))


def truncate_to_tokens(text: str, limit: int) -> str:
    """Cut text to at most limit tokens, on a token boundary."""
    tokens = encoder().encode(text)

    if len(tokens) <= limit:
        return text

    return encoder().decode(tokens[:limit])


@dataclass(frozen=True)
class Budget:
    """How a request's context window is divided up."""

    context_window: int
    reserved_for_answer: int
    reserved_for_prompt: int

    @property
    def available_for_context(self) -> int:
        """Tokens left for retrieved excerpts once everything else is accounted for."""
        return max(0, self.context_window - self.reserved_for_answer - self.reserved_for_prompt)


@dataclass
class PackResult:
    """The outcome of fitting excerpts into a budget."""

    included: list
    used_tokens: int
    dropped: int

    @property
    def total(self) -> int:
        return len(self.included) + self.dropped


def pack_to_budget(items: Iterable, budget: int, text_of=lambda item: item) -> PackResult:
    """Take items in order until the token budget is spent.

    Items arrive ranked by relevance, so stopping at the budget keeps the best
    ones. An item that does not fit is skipped rather than truncated: half an
    excerpt cited as a source is worse than one fewer source.
    """
    included = []
    used = 0
    dropped = 0

    for item in items:
        cost = count_tokens(text_of(item))

        if used + cost > budget:
            dropped += 1
            continue

        included.append(item)
        used += cost

    return PackResult(included=included, used_tokens=used, dropped=dropped)
