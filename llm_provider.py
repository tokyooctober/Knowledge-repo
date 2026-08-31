"""A thin, swappable layer between the app and any language model — commercial API or a
self-hosted open-weights model behind an OpenAI-compatible endpoint.

Changing the active model is a `config.py` edit only; no other module imports an SDK.

Three interfaces, one factory each, each returning a cached singleton:
    get_text_provider()      -> TextProvider       (answerer.py)
    get_vision_provider()    -> VisionProvider      (image_transcriber.py)
    get_embedding_provider() -> EmbeddingProvider   (embedder.py, retriever.py)
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Protocol

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import (
    EMBEDDING_BACKEND,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_QUERY_PREFIX,
    LLM_BACKEND,
    LLM_BASE_URL,
    LLM_MODEL,
    LOCAL_EMBEDDING_MODEL,
    NORMALIZE_EMBEDDINGS,
    OPENAI_EMBEDDING_MODEL,
    VISION_BACKEND,
    VISION_BASE_URL,
    VISION_MODEL,
)
from logger import get_logger
from models import ConfigError, VisionNotSupportedError

log = get_logger(__name__)


# ── module exceptions (single producer: this module) ─────────────────────────


class ProviderConnectionError(Exception):
    """An OpenAI-compatible endpoint could not be reached. Carries the base_url."""


class ModelNotFoundError(Exception):
    """The configured model is not available at the endpoint."""


# ── interfaces ──────────────────────────────────────────────────────────────


@dataclass
class TextResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class TextProvider(Protocol):
    model_name: str

    def complete(
        self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0
    ) -> TextResponse: ...


class VisionProvider(Protocol):
    model_name: str
    supports_vision: bool

    def complete_with_image(
        self, image_bytes: bytes, media_type: str, text_prompt: str, max_tokens: int = 400
    ) -> TextResponse: ...


class EmbeddingProvider(Protocol):
    model_name: str
    embedding_dim: int
    query_prefix: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ── retry: rate limits only, both SDKs, no import-time coupling ──────────────


def _is_rate_limit(exc: BaseException) -> bool:
    return type(exc).__name__ == "RateLimitError"


_retry_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


def _api_key(primary: str, fallback: str | None = None) -> str:
    """Resolve the API key for an OpenAI-shaped backend.

    The "openai" and "openai_compat" backends share the provider classes but not their
    credentials. Returns "ollama" only when nothing is set — what local backends expect
    and what remote ones reject loudly.
    """
    for name in (primary, fallback):
        if name and os.environ.get(name):
            return os.environ[name]
    return "ollama"


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        log.warning("Non-standard token count — defaulting to 0", extra={"value": repr(value)})
        return 0


# ── Anthropic ───────────────────────────────────────────────────────────────


class AnthropicTextProvider:
    """Anthropic SDK directly. LLM_BACKEND = 'anthropic'."""

    def __init__(self) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model_name = LLM_MODEL

    @_retry_rate_limit
    def complete(
        self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0
    ) -> TextResponse:
        # The Anthropic Messages API takes the system prompt as a top-level parameter,
        # not a role inside messages. Callers pass the OpenAI-style shape; translating it
        # is this adapter's job.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or self._anthropic.NOT_GIVEN,
            messages=turns,
        )
        return TextResponse(
            content=resp.content[0].text,
            model=self.model_name,
            input_tokens=_int(resp.usage.input_tokens),
            output_tokens=_int(resp.usage.output_tokens),
        )


class AnthropicVisionProvider:
    """Claude vision via the Anthropic SDK. VISION_BACKEND = 'anthropic'."""

    supports_vision = True

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self.model_name = VISION_MODEL  # VISION_MODEL, never LLM_MODEL

    @_retry_rate_limit
    def complete_with_image(
        self, image_bytes: bytes, media_type: str, text_prompt: str, max_tokens: int = 400
    ) -> TextResponse:
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_bytes).decode(),
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
        )
        return TextResponse(
            content=resp.content[0].text,
            model=self.model_name,
            input_tokens=_int(resp.usage.input_tokens),
            output_tokens=_int(resp.usage.output_tokens),
        )


# ── OpenAI-compatible (covers "openai" and "openai_compat") ──────────────────


def _openai_client(base_url: str | None, backend: str, compat_key_env: str):
    import openai

    key = _api_key("OPENAI_API_KEY" if backend == "openai" else compat_key_env)
    return openai, openai.OpenAI(api_key=key, base_url=base_url)


class OpenAICompatTextProvider:
    """openai library with an overridden base_url. Covers OpenAI, Ollama, vLLM,
    Together.ai, Groq, Fireworks, Mistral. LLM_BACKEND in ('openai', 'openai_compat')."""

    def __init__(self) -> None:
        self._openai, self._client = _openai_client(
            LLM_BASE_URL, LLM_BACKEND, "OPENAI_COMPAT_API_KEY"
        )
        self.model_name = LLM_MODEL

    @_retry_rate_limit
    def complete(
        self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0
    ) -> TextResponse:
        resp = self._call(messages, max_tokens, temperature)
        usage = resp.usage
        return TextResponse(
            content=resp.choices[0].message.content,
            model=self.model_name,
            input_tokens=_int(getattr(usage, "prompt_tokens", 0)),
            output_tokens=_int(getattr(usage, "completion_tokens", 0)),
        )

    def _call(self, messages, max_tokens, temperature):
        try:
            return self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except self._openai.APIConnectionError as exc:
            log.critical("Endpoint unreachable", extra={"base_url": LLM_BASE_URL}, exc_info=True)
            raise ProviderConnectionError(str(LLM_BASE_URL)) from exc
        except self._openai.NotFoundError as exc:
            log.error(
                "Model not found at endpoint",
                extra={"model": self.model_name, "base_url": LLM_BASE_URL},
            )
            raise ModelNotFoundError(f"{self.model_name} @ {LLM_BASE_URL}") from exc


_KNOWN_VISION_MODELS = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-4-turbo",
    "gpt-5",
    "o1",
    "o3",
    "llama3.2-vision",
    "llama-3.2-90b-vision",
    "llama-3.2-11b-vision",
    "qwen2.5vl",
    "qwen2.5-vl",
    "qwen2-vl",
    "minicpm-v",
    "llava",
    "pixtral",
)


class OpenAICompatVisionProvider:
    """Vision via an OpenAI-compatible endpoint. VISION_BACKEND in ('openai',
    'openai_compat'). Reads VISION_MODEL / VISION_BASE_URL, never the LLM_* constants."""

    def __init__(self) -> None:
        self._openai, self._client = _openai_client(
            VISION_BASE_URL, VISION_BACKEND, "OPENAI_COMPAT_API_KEY"
        )
        self.model_name = VISION_MODEL

    @property
    def supports_vision(self) -> bool:
        m = self.model_name.lower()
        return any(v in m for v in _KNOWN_VISION_MODELS)

    @_retry_rate_limit
    def complete_with_image(
        self, image_bytes: bytes, media_type: str, text_prompt: str, max_tokens: int = 400
    ) -> TextResponse:
        data_uri = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"
        resp = self._client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
        )
        usage = resp.usage
        return TextResponse(
            content=resp.choices[0].message.content,
            model=self.model_name,
            input_tokens=_int(getattr(usage, "prompt_tokens", 0)),
            output_tokens=_int(getattr(usage, "completion_tokens", 0)),
        )


# ── Embedding ───────────────────────────────────────────────────────────────


class LocalEmbeddingProvider:
    """sentence-transformers, in-process. EMBEDDING_BACKEND = 'local'."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        self.model_name = LOCAL_EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM
        self.query_prefix = EMBEDDING_QUERY_PREFIX

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=NORMALIZE_EMBEDDINGS)
        return [v.tolist() for v in vectors]


class OpenAICompatEmbeddingProvider:
    """OpenAI embeddings API or any compatible endpoint (Ollama nomic-embed, etc.).
    EMBEDDING_BACKEND in ('openai', 'openai_compat')."""

    def __init__(self) -> None:
        import openai

        key = _api_key(
            "OPENAI_API_KEY" if EMBEDDING_BACKEND == "openai" else "EMBEDDING_API_KEY",
            fallback="OPENAI_COMPAT_API_KEY",
        )
        self._client = openai.OpenAI(api_key=key, base_url=EMBEDDING_BASE_URL)
        self.model_name = OPENAI_EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM
        self.query_prefix = ""  # OpenAI-style models do not use instruction prefixes

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in resp.data]


# ── factories ───────────────────────────────────────────────────────────────

_text_provider: TextProvider | None = None
_vision_provider: VisionProvider | None = None
_embedding_provider: EmbeddingProvider | None = None


def get_text_provider() -> TextProvider:
    global _text_provider
    if _text_provider is None:
        if LLM_BACKEND == "anthropic":
            _text_provider = AnthropicTextProvider()
        elif LLM_BACKEND in ("openai", "openai_compat"):
            _text_provider = OpenAICompatTextProvider()
        else:
            raise ConfigError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}")
        log.info(
            "Text provider initialised",
            extra={"backend": LLM_BACKEND, "model": _text_provider.model_name},
        )
    return _text_provider


def get_vision_provider() -> VisionProvider:
    global _vision_provider
    if _vision_provider is None:
        if VISION_BACKEND == "anthropic":
            _vision_provider = AnthropicVisionProvider()
        elif VISION_BACKEND in ("openai", "openai_compat"):
            _vision_provider = OpenAICompatVisionProvider()
        else:
            raise ConfigError(f"Unknown VISION_BACKEND: {VISION_BACKEND!r}")
        if not _vision_provider.supports_vision:
            raise VisionNotSupportedError(
                f"Model {_vision_provider.model_name!r} does not support vision input. "
                f"Check VISION_MODEL in config."
            )
        log.info(
            "Vision provider initialised",
            extra={"backend": VISION_BACKEND, "model": _vision_provider.model_name},
        )
    return _vision_provider


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        if EMBEDDING_BACKEND == "local":
            _embedding_provider = LocalEmbeddingProvider()
        elif EMBEDDING_BACKEND in ("openai", "openai_compat"):
            _embedding_provider = OpenAICompatEmbeddingProvider()
        else:
            raise ConfigError(f"Unknown EMBEDDING_BACKEND: {EMBEDDING_BACKEND!r}")
        log.info(
            "Embedding provider initialised",
            extra={
                "backend": EMBEDDING_BACKEND,
                "model": _embedding_provider.model_name,
                "dim": _embedding_provider.embedding_dim,
            },
        )
    return _embedding_provider


def _reset_providers_for_tests() -> None:
    """Clear the singletons. Test-only — production initialises once per process."""
    global _text_provider, _vision_provider, _embedding_provider
    _text_provider = _vision_provider = _embedding_provider = None
