"""LLM providers, tried in order so one outage does not take the demo down.

Groq and Gemini both expose OpenAI-compatible endpoints, so a single client
class serves both and switching is a matter of base URL, key, and model name.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Failures worth trying the next provider for. An authentication error is
# included because a missing or stale key on one provider should not take the
# app down when another is configured.
FALLBACK_ERRORS = (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    AuthenticationError,
)


class AllProvidersFailed(RuntimeError):
    """Raised when every configured provider refused the request."""


@dataclass(frozen=True)
class ProviderConfig:
    """One LLM backend."""

    name: str
    base_url: str
    api_key_env: str
    model: str
    context_window: int

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


GROQ = ProviderConfig(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    context_window=32768,
)

GEMINI = ProviderConfig(
    name="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key_env="GEMINI_API_KEY",
    model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    context_window=32768,
)

DEFAULT_CHAIN = (GROQ, GEMINI)


@dataclass
class Completion:
    """An answer plus which provider produced it."""

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ProviderChain:
    """Calls providers in order, falling through on recoverable failures."""

    def __init__(
        self,
        configs=DEFAULT_CHAIN,
        client_factory: Callable[[ProviderConfig], OpenAI] = None,
    ) -> None:
        self.configs = [c for c in configs if c.configured]
        self._factory = client_factory or (
            lambda config: OpenAI(api_key=config.api_key, base_url=config.base_url)
        )

    @property
    def available(self) -> list[str]:
        """Names of providers that have a key configured."""
        return [config.name for config in self.configs]

    def complete(self, messages: list[dict], temperature: float = 0.0) -> Completion:
        """Return the first successful completion, or raise if all fail."""
        if not self.configs:
            raise AllProvidersFailed(
                "No provider is configured. Set GROQ_API_KEY or GEMINI_API_KEY."
            )

        failures = []

        for config in self.configs:
            try:
                response = self._factory(config).chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=temperature,
                )
            except FALLBACK_ERRORS as exc:
                failures.append(f"{config.name}: {type(exc).__name__}")
                continue
            except APIStatusError as exc:
                # A 404 for a retired model name is worth falling through for.
                failures.append(f"{config.name}: HTTP {exc.status_code}")
                continue

            usage = getattr(response, "usage", None)

            return Completion(
                text=response.choices[0].message.content or "",
                provider=config.name,
                model=config.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )

        raise AllProvidersFailed("; ".join(failures))
