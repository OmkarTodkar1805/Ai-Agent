"""Provider selection and fallback."""

import types

import httpx2 as httpx
import pytest
from openai import AuthenticationError, RateLimitError

from doc_agent.providers import (
    AllProvidersFailed,
    ProviderChain,
    ProviderConfig,
)


def config(name: str, env: str) -> ProviderConfig:
    return ProviderConfig(
        name=name, base_url="http://stub", api_key_env=env, model=f"{name}-model",
        context_window=8000,
    )


def error(kind, status):
    return kind(
        "boom",
        response=httpx.Response(status, request=httpx.Request("POST", "http://stub")),
        body=None,
    )


def client_returning(text: str):
    def factory(_cfg):
        completions = types.SimpleNamespace(
            create=lambda **kw: types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))],
                usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )
        )
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

    return factory


def client_raising(*errors):
    """A factory whose clients raise the given errors, one per provider in order."""
    queue = list(errors)

    def factory(_cfg):
        item = queue.pop(0)

        def create(**kw):
            if isinstance(item, Exception):
                raise item
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=item))],
                usage=None,
            )

        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )

    return factory


@pytest.fixture
def two_providers(monkeypatch):
    monkeypatch.setenv("PRIMARY_KEY", "x")
    monkeypatch.setenv("SECONDARY_KEY", "y")
    return [config("primary", "PRIMARY_KEY"), config("secondary", "SECONDARY_KEY")]


def test_primary_is_used_when_it_works(two_providers):
    chain = ProviderChain(two_providers, client_factory=client_returning("answer"))

    result = chain.complete([{"role": "user", "content": "hi"}])

    assert result.provider == "primary"
    assert result.text == "answer"


def test_usage_is_reported_when_the_api_supplies_it(two_providers):
    chain = ProviderChain(two_providers, client_factory=client_returning("answer"))

    result = chain.complete([{"role": "user", "content": "hi"}])

    assert (result.prompt_tokens, result.completion_tokens) == (11, 7)


def test_rate_limited_primary_falls_through_to_secondary(two_providers):
    factory = client_raising(error(RateLimitError, 429), "from secondary")
    chain = ProviderChain(two_providers, client_factory=factory)

    result = chain.complete([{"role": "user", "content": "hi"}])

    assert result.provider == "secondary"
    assert result.text == "from secondary"


def test_bad_key_on_primary_falls_through(two_providers):
    """A stale key on one provider must not take the app down."""
    factory = client_raising(error(AuthenticationError, 401), "from secondary")
    chain = ProviderChain(two_providers, client_factory=factory)

    assert chain.complete([{"role": "user", "content": "hi"}]).provider == "secondary"


def test_all_failing_raises_with_every_reason(two_providers):
    factory = client_raising(error(RateLimitError, 429), error(RateLimitError, 429))
    chain = ProviderChain(two_providers, client_factory=factory)

    with pytest.raises(AllProvidersFailed) as caught:
        chain.complete([{"role": "user", "content": "hi"}])

    assert "primary" in str(caught.value) and "secondary" in str(caught.value)


def test_providers_without_a_key_are_skipped(monkeypatch):
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.setenv("SECONDARY_KEY", "y")
    configs = [config("primary", "PRIMARY_KEY"), config("secondary", "SECONDARY_KEY")]

    chain = ProviderChain(configs, client_factory=client_returning("ok"))

    assert chain.available == ["secondary"]


def test_no_configured_provider_is_an_actionable_error(monkeypatch):
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.delenv("SECONDARY_KEY", raising=False)
    chain = ProviderChain([config("primary", "PRIMARY_KEY")], client_factory=client_returning("x"))

    with pytest.raises(AllProvidersFailed, match="GROQ_API_KEY"):
        chain.complete([{"role": "user", "content": "hi"}])
