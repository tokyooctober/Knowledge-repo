"""Tests for llm_provider.py — factory dispatch, singleton caching, backend translation.

Every test patches `anthropic.Anthropic` and `openai.OpenAI` with fakes; no real client
is constructed and no network call is made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import llm_provider as lp
from models import ConfigError, VisionNotSupportedError


@pytest.fixture(autouse=True)
def _fresh_singletons():
    lp._reset_providers_for_tests()
    yield
    lp._reset_providers_for_tests()


@pytest.fixture
def fake_anthropic(monkeypatch):
    """anthropic.Anthropic() -> a MagicMock whose messages.create returns a Claude-shaped
    response."""
    import anthropic

    msg = MagicMock()
    msg.content = [MagicMock(text="hello from claude")]
    msg.usage = MagicMock(input_tokens=11, output_tokens=7)
    client = MagicMock()
    client.messages.create.return_value = msg
    ctor = MagicMock(return_value=client)
    monkeypatch.setattr(anthropic, "Anthropic", ctor)
    return ctor, client


@pytest.fixture
def fake_openai(monkeypatch):
    """openai.OpenAI(...) -> a MagicMock with chat.completions and embeddings."""
    import openai

    chat_resp = MagicMock()
    chat_resp.choices = [MagicMock(message=MagicMock(content="hello from gpt"))]
    chat_resp.usage = MagicMock(prompt_tokens=13, completion_tokens=5)

    emb_resp = MagicMock()
    emb_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    client = MagicMock()
    client.chat.completions.create.return_value = chat_resp
    client.embeddings.create.return_value = emb_resp
    ctor = MagicMock(return_value=client)
    monkeypatch.setattr(openai, "OpenAI", ctor)
    return ctor, client


# ── text factory dispatch ───────────────────────────────────────────────────


def test_anthropic_backend_builds_anthropic_provider(monkeypatch, fake_anthropic):
    monkeypatch.setattr(lp, "LLM_BACKEND", "anthropic")
    provider = lp.get_text_provider()
    assert isinstance(provider, lp.AnthropicTextProvider)


@pytest.mark.parametrize("backend", ["openai", "openai_compat"])
def test_openai_backends_both_build_the_compat_provider(monkeypatch, fake_openai, backend):
    monkeypatch.setattr(lp, "LLM_BACKEND", backend)
    provider = lp.get_text_provider()
    assert isinstance(provider, lp.OpenAICompatTextProvider)


def test_unknown_text_backend_raises_config_error(monkeypatch):
    monkeypatch.setattr(lp, "LLM_BACKEND", "banana")
    with pytest.raises(ConfigError, match="banana"):
        lp.get_text_provider()


def test_text_provider_is_a_cached_singleton(monkeypatch, fake_anthropic):
    monkeypatch.setattr(lp, "LLM_BACKEND", "anthropic")
    ctor, _ = fake_anthropic
    a = lp.get_text_provider()
    b = lp.get_text_provider()
    assert a is b
    assert ctor.call_count == 1  # __init__ ran once


# ── text translation + token counts ─────────────────────────────────────────


def test_anthropic_lifts_system_message_to_top_level(monkeypatch, fake_anthropic):
    monkeypatch.setattr(lp, "LLM_BACKEND", "anthropic")
    _, client = fake_anthropic
    provider = lp.get_text_provider()
    resp = provider.complete(
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "be terse"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert resp.input_tokens == 11 and resp.output_tokens == 7
    assert resp.content == "hello from claude"


def test_openai_compat_reports_token_counts(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "LLM_BACKEND", "openai_compat")
    provider = lp.get_text_provider()
    resp = provider.complete([{"role": "user", "content": "hi"}])
    assert resp.input_tokens == 13 and resp.output_tokens == 5
    assert resp.content == "hello from gpt"


def test_openai_compat_missing_usage_defaults_to_zero(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "LLM_BACKEND", "openai_compat")
    _, client = fake_openai
    client.chat.completions.create.return_value.usage = None
    provider = lp.get_text_provider()
    resp = provider.complete([{"role": "user", "content": "hi"}])
    assert resp.input_tokens == 0 and resp.output_tokens == 0


def test_openai_compat_passes_base_url(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "LLM_BACKEND", "openai_compat")
    monkeypatch.setattr(lp, "LLM_BASE_URL", "http://localhost:11434/v1")
    ctor, _ = fake_openai
    lp.get_text_provider()
    assert ctor.call_args.kwargs["base_url"] == "http://localhost:11434/v1"


def test_openai_backend_defaults_base_url_to_none(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "LLM_BACKEND", "openai")
    monkeypatch.setattr(lp, "LLM_BASE_URL", None)
    ctor, _ = fake_openai
    lp.get_text_provider()
    assert ctor.call_args.kwargs["base_url"] is None


# ── vision ──────────────────────────────────────────────────────────────────


def test_vision_anthropic_is_supported(monkeypatch, fake_anthropic):
    monkeypatch.setattr(lp, "VISION_BACKEND", "anthropic")
    provider = lp.get_vision_provider()
    assert provider.supports_vision is True
    assert isinstance(provider, lp.AnthropicVisionProvider)


def test_vision_compat_supports_a_vision_model(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "VISION_BACKEND", "openai_compat")
    monkeypatch.setattr(lp, "VISION_MODEL", "qwen2.5vl:72b")
    provider = lp.get_vision_provider()
    assert provider.supports_vision is True


def test_vision_compat_rejects_a_text_only_model(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "VISION_BACKEND", "openai_compat")
    monkeypatch.setattr(lp, "VISION_MODEL", "qwen2.5:72b")  # text-only
    with pytest.raises(VisionNotSupportedError, match="qwen2.5:72b"):
        lp.get_vision_provider()


def test_vision_compat_reads_vision_constants_not_llm(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "VISION_BACKEND", "openai_compat")
    monkeypatch.setattr(lp, "VISION_MODEL", "llava:13b")
    monkeypatch.setattr(lp, "VISION_BASE_URL", "http://vision:9/v1")
    monkeypatch.setattr(lp, "LLM_MODEL", "some-text-model")
    monkeypatch.setattr(lp, "LLM_BASE_URL", "http://text:8/v1")
    ctor, _ = fake_openai
    provider = lp.get_vision_provider()
    assert provider.model_name == "llava:13b"
    assert ctor.call_args.kwargs["base_url"] == "http://vision:9/v1"


def test_unknown_vision_backend_raises(monkeypatch):
    monkeypatch.setattr(lp, "VISION_BACKEND", "banana")
    with pytest.raises(ConfigError):
        lp.get_vision_provider()


def test_anthropic_vision_sends_a_base64_image_block(monkeypatch, fake_anthropic):
    monkeypatch.setattr(lp, "VISION_BACKEND", "anthropic")
    _, client = fake_anthropic
    provider = lp.get_vision_provider()
    provider.complete_with_image(b"\x89PNG...", "image/png", "transcribe this")
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1] == {"type": "text", "text": "transcribe this"}


# ── embedding ───────────────────────────────────────────────────────────────


def test_embedding_local_backend(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = [_np([0.1, 0.2]), _np([0.3, 0.4])]

    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", MagicMock(return_value=fake_model)
    )
    monkeypatch.setattr(lp, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(lp, "EMBEDDING_QUERY_PREFIX", "Represent this: ")

    provider = lp.get_embedding_provider()
    assert isinstance(provider, lp.LocalEmbeddingProvider)
    assert provider.query_prefix == "Represent this: "  # only Local carries a prefix
    assert provider.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_openai_compat_has_no_query_prefix(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "EMBEDDING_BACKEND", "openai_compat")
    provider = lp.get_embedding_provider()
    assert isinstance(provider, lp.OpenAICompatEmbeddingProvider)
    assert provider.query_prefix == ""
    assert provider.embed(["x"]) == [[0.1, 0.2, 0.3]]


def test_unknown_embedding_backend_raises(monkeypatch):
    monkeypatch.setattr(lp, "EMBEDDING_BACKEND", "banana")
    with pytest.raises(ConfigError):
        lp.get_embedding_provider()


# ── _api_key resolution ─────────────────────────────────────────────────────


def test_api_key_prefers_primary_then_fallback_then_ollama(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    assert lp._api_key("OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY") == "ollama"

    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-key")
    assert lp._api_key("OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY") == "compat-key"

    monkeypatch.setenv("OPENAI_API_KEY", "primary-key")
    assert lp._api_key("OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY") == "primary-key"


class _np(list):
    """Stand-in for a numpy row: supports .tolist()."""

    def tolist(self):
        return list(self)


# ── openai-compat error translation ─────────────────────────────────────────


def test_compat_connection_error_becomes_provider_connection_error(monkeypatch, fake_openai):
    import openai

    monkeypatch.setattr(lp, "LLM_BACKEND", "openai_compat")
    monkeypatch.setattr(lp, "LLM_BASE_URL", "http://down:9/v1")
    _, client = fake_openai
    client.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())
    provider = lp.get_text_provider()
    with pytest.raises(lp.ProviderConnectionError, match="down:9"):
        provider.complete([{"role": "user", "content": "hi"}])


def test_compat_not_found_becomes_model_not_found(monkeypatch, fake_openai):
    import openai

    monkeypatch.setattr(lp, "LLM_BACKEND", "openai_compat")
    monkeypatch.setattr(lp, "LLM_MODEL", "ghost-model")
    _, client = fake_openai
    client.chat.completions.create.side_effect = openai.NotFoundError(
        "nope", response=MagicMock(status_code=404), body=None
    )
    provider = lp.get_text_provider()
    with pytest.raises(lp.ModelNotFoundError, match="ghost-model"):
        provider.complete([{"role": "user", "content": "hi"}])


# ── compat vision call shape ────────────────────────────────────────────────


def test_compat_vision_sends_a_data_uri(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "VISION_BACKEND", "openai_compat")
    monkeypatch.setattr(lp, "VISION_MODEL", "llava:13b")
    _, client = fake_openai
    client.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=20, completion_tokens=8
    )
    provider = lp.get_vision_provider()
    resp = provider.complete_with_image(b"PNGBYTES", "image/png", "read the chart")
    content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert resp.input_tokens == 20 and resp.output_tokens == 8


# ── retry predicate + token coercion ───────────────────────────────────────


def test_rate_limit_is_retried(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "LLM_BACKEND", "openai_compat")

    class RateLimitError(Exception):
        pass

    _, client = fake_openai
    good = client.chat.completions.create.return_value
    client.chat.completions.create.side_effect = [RateLimitError("slow down"), good]
    provider = lp.get_text_provider()
    resp = provider.complete([{"role": "user", "content": "hi"}])  # retried, then ok
    assert resp.content == "hello from gpt"
    assert client.chat.completions.create.call_count == 2


def test_non_int_token_count_coerces_to_zero(monkeypatch, fake_openai):
    monkeypatch.setattr(lp, "LLM_BACKEND", "openai_compat")
    _, client = fake_openai
    client.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens="not-a-number", completion_tokens=None
    )
    provider = lp.get_text_provider()
    resp = provider.complete([{"role": "user", "content": "hi"}])
    assert resp.input_tokens == 0 and resp.output_tokens == 0


# ── vision / embedding singletons ──────────────────────────────────────────


def test_vision_and_embedding_providers_are_cached(monkeypatch, fake_anthropic, fake_openai):
    monkeypatch.setattr(lp, "VISION_BACKEND", "anthropic")
    monkeypatch.setattr(lp, "EMBEDDING_BACKEND", "openai_compat")
    assert lp.get_vision_provider() is lp.get_vision_provider()
    assert lp.get_embedding_provider() is lp.get_embedding_provider()
