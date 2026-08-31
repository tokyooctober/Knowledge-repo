# `llm_provider.py` — LLM Provider Abstraction

---
```
module:     llm_provider.py
spec:       SPEC_llm_provider.md
layer:      Shared foundation
depends_on: config.py · logger.py
used_by:    ingestion/embedder.py
            ingestion/image_transcriber.py
            query/answerer.py
            query/retriever.py
services:   Anthropic API · OpenAI API · Ollama · vLLM · any OpenAI-compat endpoint
```
---

## Purpose
Define a thin, swappable interface layer between the application and any underlying language model — whether a commercial API (Anthropic, OpenAI) or a self-hosted open-weights model served via an OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, Together.ai, Groq, Fireworks). Changing the active model requires editing config only; no application code changes.

---

## Design Principle: OpenAI-compatible as the universal adapter

The OpenAI Chat Completions API (`POST /v1/chat/completions`) has become the de facto standard interface for language model serving. Nearly every open-weights serving framework implements it. This means the `openai` Python library — with its `base_url` parameter overridden — can reach Ollama, vLLM, LM Studio, or any cloud provider that is OpenAI-compatible. A single client implementation covers all of them.

```
BACKEND = "anthropic"     → Anthropic SDK directly (claude-*)
BACKEND = "openai"        → OpenAI API (gpt-*, o1-*)
BACKEND = "openai_compat" → openai library + custom base_url
                             covers: Ollama · vLLM · LM Studio · Together.ai
                                     Groq · Fireworks · Mistral API · any local server
```

---

## Three Provider Interfaces

### 1. `TextProvider` — chat completion (used by `answerer.py`)

```python
from typing import Protocol

class TextProvider(Protocol):
    """Synchronous chat completion interface."""

    model_name: str

    def complete(
        self,
        messages: list[dict],          # [{"role": "system"|"user"|"assistant", "content": str}]
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> TextResponse: ...

@dataclass
class TextResponse:
    content:       str
    model:         str
    input_tokens:  int
    output_tokens: int
```

### 2. `VisionProvider` — image + text completion (used by `image_transcriber.py`)

```python
class VisionProvider(Protocol):
    """Chat completion with image input."""

    model_name: str
    supports_vision: bool    # checked at startup; False raises ConfigError

    def complete_with_image(
        self,
        image_bytes: bytes,
        media_type: str,               # "image/png" | "image/jpeg" | "image/webp"
        text_prompt: str,
        max_tokens: int = 400,
    ) -> TextResponse: ...
```

### 3. `EmbeddingProvider` — dense vector embedding (used by `embedder.py`)

```python
class EmbeddingProvider(Protocol):
    """Text embedding interface."""

    model_name:    str
    embedding_dim: int
    query_prefix:  str    # instruction prefix applied at query time only ("" if none)

    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

---

## Concrete Implementations

### `AnthropicTextProvider`
```python
import anthropic
from anthropic import NOT_GIVEN            # sentinel for "omit this parameter"

class AnthropicTextProvider:
    """Wraps the Anthropic SDK. Used when LLM_BACKEND = 'anthropic'."""

    def __init__(self):
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self.model_name = LLM_MODEL           # e.g. "claude-sonnet-4-20250514"

    def complete(self, messages, max_tokens=1024, temperature=0.0) -> TextResponse:
        # The Anthropic Messages API rejects role="system" inside messages — the
        # system prompt is a top-level parameter. Callers pass the OpenAI-style
        # shape (system as the first message); translating it is this adapter's
        # job, exactly like every other backend difference.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns  = [m for m in messages if m["role"] != "system"]
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or NOT_GIVEN,
            messages=turns,
        )
        return TextResponse(
            content=response.content[0].text,
            model=self.model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
```

### `AnthropicVisionProvider`
```python
class AnthropicVisionProvider:
    """Claude vision via Anthropic SDK. Used when VISION_BACKEND = 'anthropic'."""

    supports_vision = True

    def __init__(self):
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self.model_name = VISION_MODEL         # VISION_MODEL, never LLM_MODEL

    def complete_with_image(self, image_bytes, media_type, text_prompt, max_tokens=400):
        import base64
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type,
                                "data": base64.b64encode(image_bytes).decode()}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )
        return TextResponse(
            content=response.content[0].text,
            model=self.model_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
```

### `OpenAICompatTextProvider`
```python
def _api_key(primary: str, fallback: str | None = None) -> str:
    """Resolve the API key for an OpenAI-shaped backend.

    The "openai" and "openai_compat" backends share these provider classes but
    NOT their credentials: a user who sets LLM_BACKEND="openai" fills in
    OPENAI_API_KEY per the .env template, and reading OPENAI_COMPAT_API_KEY
    instead sends the literal string "ollama" to api.openai.com — a 401 that
    points at nothing. Hence the explicit primary/fallback.

    Returns "ollama" only when nothing is set, which is what local backends
    (Ollama, vLLM, LM Studio) expect and what remote ones will reject loudly.
    """
    for name in (primary, fallback):
        if name and os.environ.get(name):
            return os.environ[name]
    return "ollama"


class OpenAICompatTextProvider:
    """OpenAI library with overridden base_url.
    Covers Ollama, vLLM, LM Studio, Together.ai, Groq, Fireworks, Mistral, etc.
    Used when LLM_BACKEND = 'openai_compat'.
    """

    def __init__(self):
        self.client = openai.OpenAI(
            api_key=_api_key("OPENAI_API_KEY" if LLM_BACKEND == "openai"
                                 else "OPENAI_COMPAT_API_KEY"),  # "ollama" for local
            base_url=LLM_BASE_URL,    # e.g. "http://localhost:11434/v1"
        )
        self.model_name = LLM_MODEL   # e.g. "llama3.3:70b", "qwen2.5:72b"

    def complete(self, messages, max_tokens=1024, temperature=0.0) -> TextResponse:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return TextResponse(
            content=response.choices[0].message.content,
            model=self.model_name,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
```

### `OpenAICompatVisionProvider`
```python
class OpenAICompatVisionProvider:
    """Vision via OpenAI-compatible endpoint (also VISION_BACKEND = 'openai').
    Requires the serving backend and model to support vision inputs.
    Validated at startup via the supports_vision check.
    """

    # Models known to accept image input on an OpenAI-shaped endpoint. Extend as needed;
    # a model not on this list makes supports_vision False and get_vision_provider() raise.
    _VISION_MODELS = (
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4-turbo",
        "llama3.2-vision", "llama-3.2-90b-vision", "qwen2.5vl", "qwen2.5-vl",
        "minicpm-v", "llava",
    )

    def __init__(self):
        self.client = openai.OpenAI(
            api_key=_api_key("OPENAI_API_KEY" if VISION_BACKEND == "openai"
                                 else "OPENAI_COMPAT_API_KEY"),   # "ollama" for local
            base_url=VISION_BASE_URL,    # VISION_BASE_URL, never LLM_BASE_URL
        )
        self.model_name = VISION_MODEL   # VISION_MODEL, never LLM_MODEL

    @property
    def supports_vision(self) -> bool:
        m = self.model_name.lower()
        return any(m.startswith(v) or v in m for v in self._VISION_MODELS)

    def complete_with_image(self, image_bytes, media_type, text_prompt, max_tokens=400):
        import base64
        b64 = base64.b64encode(image_bytes).decode()
        data_uri = f"data:{media_type};base64,{b64}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )
        return TextResponse(
            content=response.choices[0].message.content,
            model=self.model_name,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
```

### `LocalEmbeddingProvider`
```python
class LocalEmbeddingProvider:
    """sentence-transformers. Backend = 'local'."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        self.model_name   = LOCAL_EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM
        self.query_prefix  = EMBEDDING_QUERY_PREFIX   # BGE prefix or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=NORMALIZE_EMBEDDINGS)
        return [v.tolist() for v in vectors]
```

### `OpenAICompatEmbeddingProvider`
```python
class OpenAICompatEmbeddingProvider:
    """OpenAI embeddings API or any compatible endpoint (Ollama nomic-embed, etc.)."""

    def __init__(self):
        self.client = openai.OpenAI(
            api_key=_api_key("OPENAI_API_KEY" if EMBEDDING_BACKEND == "openai"
                                 else "EMBEDDING_API_KEY",
                                 fallback="OPENAI_COMPAT_API_KEY"),
            base_url=EMBEDDING_BASE_URL,    # None → OpenAI; "http://localhost:11434/v1" → Ollama
        )
        self.model_name    = OPENAI_EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM
        self.query_prefix  = ""    # OpenAI-style models do not use instruction prefixes

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]
```

---

## Factory Functions

```python
# Module-level singletons (created once, reused across calls)
_text_provider:      TextProvider | None = None
_vision_provider:    VisionProvider | None = None
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
        log.info("Text provider initialised",
                 extra={"backend": LLM_BACKEND, "model": _text_provider.model_name})
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
        # Both vision classes read VISION_MODEL and VISION_BASE_URL in their
        # __init__ — never LLM_MODEL / LLM_BASE_URL. Sharing the text constants
        # silently collapses the documented "cheap cloud text + separate cloud
        # vision" setups onto a single backend and model.
        # Both also expose supports_vision as a bool property: Anthropic returns
        # True; OpenAICompat checks VISION_MODEL against its known-vision list.
        # Every VisionProvider must expose supports_vision (a bool property).
        # AnthropicVisionProvider returns True; OpenAICompatVisionProvider checks
        # VISION_MODEL against its known-vision-model list and returns False for
        # a text-only model. A provider without the attribute is a bug, not a
        # reason to skip the check — do not getattr(..., True) around it.
        if not _vision_provider.supports_vision:
            raise VisionNotSupportedError(
                f"Model {_vision_provider.model_name!r} does not support vision input. "
                f"Check VISION_MODEL in config."
            )
        log.info("Vision provider initialised",
                 extra={"backend": VISION_BACKEND, "model": _vision_provider.model_name})
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
        log.info("Embedding provider initialised",
                 extra={"backend": EMBEDDING_BACKEND,
                        "model":   _embedding_provider.model_name,
                        "dim":     _embedding_provider.embedding_dim})
    return _embedding_provider
```

---

## Supported Model Reference

### Text generation models

| Backend | `LLM_BACKEND` | `LLM_BASE_URL` | Recommended `LLM_MODEL` |
|---|---|---|---|
| Anthropic (cloud) | `anthropic` | — | `claude-sonnet-4-20250514` |
| OpenAI (cloud) | `openai` | — | `gpt-4.1` |
| Ollama (local) | `openai_compat` | `http://localhost:11434/v1` | `llama3.3:70b` · `qwen2.5:72b` · `mistral-small3.1` |
| vLLM (local/server) | `openai_compat` | `http://localhost:8000/v1` | `meta-llama/Llama-3.3-70B-Instruct` · `Qwen/Qwen2.5-72B-Instruct` |
| Together.ai (cloud) | `openai_compat` | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Groq (cloud) | `openai_compat` | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Fireworks (cloud) | `openai_compat` | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |

### Vision models

| Backend | `VISION_BACKEND` | `VISION_BASE_URL` | Recommended `VISION_MODEL` |
|---|---|---|---|
| Anthropic (cloud) | `anthropic` | — | `claude-sonnet-4-20250514` |
| OpenAI (cloud) | `openai` | — | `gpt-4o` · `gpt-4.1` |
| Ollama (local) | `openai_compat` | `http://localhost:11434/v1` | `llama3.2-vision:90b` · `qwen2.5vl:72b` · `minicpm-v` |
| vLLM (local/server) | `openai_compat` | `http://localhost:8000/v1` | `meta-llama/Llama-3.2-90B-Vision-Instruct` · `Qwen/Qwen2.5-VL-72B-Instruct` |
| Together.ai (cloud) | `openai_compat` | `https://api.together.xyz/v1` | `meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo` |

> **Vision capability note**: not all Ollama models support vision even if pulled. Use `ollama show <model>` to confirm. If `complete_with_image` receives a non-vision model, the API returns an error — the provider raises `VisionNotSupportedError`.

### Embedding models

| Backend | `EMBEDDING_BACKEND` | `EMBEDDING_BASE_URL` | `LOCAL_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_MODEL` | Dim |
|---|---|---|---|---|
| Local sentence-transformers | `local` | — | `BAAI/bge-large-en-v1.5` | 1024 |
| Local sentence-transformers | `local` | — | `nomic-ai/nomic-embed-text-v1.5` | 768 |
| OpenAI (cloud) | `openai` | — | `text-embedding-3-small` | 1536 |
| Ollama (local) | `openai_compat` | `http://localhost:11434/v1` | `nomic-embed-text` | 768 |
| Ollama (local) | `openai_compat` | `http://localhost:11434/v1` | `mxbai-embed-large` | 1024 |

> **Dimension consistency rule**: changing the embedding model invalidates the entire Qdrant collection. Record `model_name` and `embedding_dim` in the collection metadata. At startup, raise `ModelMismatchError` if the configured model does not match what was used to build the collection.

---

## Config Constants

```python
# ── Text LLM ─────────────────────────────────────────────────────────────────
LLM_BACKEND   = "anthropic"                       # "anthropic" | "openai" | "openai_compat"
LLM_MODEL     = "claude-sonnet-4-20250514"        # model identifier for the chosen backend
LLM_BASE_URL  = None                              # None for anthropic/openai; URL for openai_compat

# ── Vision LLM ────────────────────────────────────────────────────────────────
VISION_BACKEND   = "anthropic"                    # "anthropic" | "openai" | "openai_compat"
VISION_MODEL     = "claude-sonnet-4-20250514"
VISION_BASE_URL  = None                           # None for anthropic/openai; URL for openai_compat

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_BACKEND      = "local"                  # "local" | "openai" | "openai_compat"
LOCAL_EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BASE_URL     = None                     # None for openai; URL for openai_compat
EMBEDDING_DIM          = 1024                     # must match the active model
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
                         # Set to "" for non-BGE models
```

---

## Example: Switch Everything to Ollama (full local, no cloud APIs)

```python
# config.py
LLM_BACKEND      = "openai_compat"
LLM_MODEL        = "qwen2.5:72b"
LLM_BASE_URL     = "http://localhost:11434/v1"

VISION_BACKEND   = "openai_compat"
VISION_MODEL     = "qwen2.5vl:72b"
VISION_BASE_URL  = "http://localhost:11434/v1"

EMBEDDING_BACKEND     = "openai_compat"
OPENAI_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_BASE_URL    = "http://localhost:11434/v1"
EMBEDDING_DIM         = 768
EMBEDDING_QUERY_PREFIX = "search_query: "   # nomic-embed-text uses a different prefix
```

```bash
# Pull the required models once
ollama pull qwen2.5:72b
ollama pull qwen2.5vl:72b
ollama pull nomic-embed-text
```

## Example: Mix backends (local embeddings + cloud generation)

```python
# config.py — cheapest practical setup
EMBEDDING_BACKEND     = "local"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"   # free, runs on CPU

LLM_BACKEND  = "openai_compat"
LLM_MODEL    = "llama-3.3-70b-versatile"
LLM_BASE_URL = "https://api.groq.com/openai/v1"     # fast, generous free tier

VISION_BACKEND  = "openai_compat"
VISION_MODEL    = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
VISION_BASE_URL = "https://api.together.xyz/v1"
```

---

## `.env` additions

```
# Required for anthropic backend
ANTHROPIC_API_KEY=sk-ant-...

# Required for openai backend
OPENAI_API_KEY=sk-...

# Required for openai_compat cloud backends
OPENAI_COMPAT_API_KEY=...       # Together.ai / Groq / Fireworks key
EMBEDDING_API_KEY=...           # separate key if embedding uses a different provider

# Local backends (Ollama, vLLM): no API key needed
# Set OPENAI_COMPAT_API_KEY=ollama (literal string) for Ollama
```

---

## Error Handling

`ConfigError` and `VisionNotSupportedError` are defined in `models.py` (see *Shared
exceptions* in [SPEC.md](SPEC.md)) — they are raised by more than one module.
`ProviderConnectionError` and `ModelNotFoundError` are defined **here**, in
`llm_provider.py`, and imported by any module that wants to catch them.

| Scenario | Behaviour | Log level |
|---|---|---|
| Unknown `LLM_BACKEND` value | Raise `ConfigError` at startup | CRITICAL |
| Unknown `VISION_BACKEND` value | Raise `ConfigError` at startup | CRITICAL |
| Model does not support vision | Raise `VisionNotSupportedError` at startup | CRITICAL |
| API key missing for chosen backend | Raise `ConfigError`; never log key | CRITICAL |
| OpenAI-compat endpoint unreachable | Raise `ProviderConnectionError` with `base_url` | CRITICAL |
| Model not found at endpoint | Raise `ModelNotFoundError` with model name and endpoint | ERROR |
| Rate limit (any backend) | Retry with exponential backoff via `tenacity` (max 3) | WARNING |
| Non-standard token count fields | Default to 0; log at WARNING | WARNING |

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.llm_provider"
```

| Event | Level | Extra fields |
|---|---|---|
| Text provider initialised | INFO | `backend`, `model`, `base_url` (masked if contains key) |
| Vision provider initialised | INFO | `backend`, `model`, `base_url` |
| Embedding provider initialised | INFO | `backend`, `model`, `dim` |
| API call started | DEBUG | `provider`, `model`, `max_tokens` |
| API call complete | DEBUG | `provider`, `model`, `input_tokens`, `output_tokens`, `elapsed_ms` |
| Rate limit hit — retrying | WARNING | `provider`, `attempt`, `retry_after_s` |
| Model not found at endpoint | ERROR | `model`, `base_url`, `error_type` |
| Endpoint unreachable | CRITICAL | `base_url`, `error_type` |
| Vision not supported by model | CRITICAL | `model`, `backend` |

---

## Key Dependencies
- `anthropic` — Anthropic SDK (only if `LLM_BACKEND="anthropic"` or `VISION_BACKEND="anthropic"`)
- `openai` — OpenAI library used for all `openai` and `openai_compat` backends
- `sentence-transformers` — local embedding (only if `EMBEDDING_BACKEND="local"`)
- `tenacity` — retry logic

---

## Testing Notes
- Assert `get_text_provider()` returns `AnthropicTextProvider` when `LLM_BACKEND="anthropic"`
- Assert `get_text_provider()` returns `OpenAICompatTextProvider` for both `"openai"` and `"openai_compat"`
- Assert factory returns cached singleton on repeated calls (no re-initialisation)
- Assert unknown backend raises `ConfigError`
- Assert `get_vision_provider()` raises `VisionNotSupportedError` if `supports_vision=False`
- Assert `OpenAICompatVisionProvider.__init__` reads `VISION_MODEL` / `VISION_BASE_URL`,
  not `LLM_MODEL` / `LLM_BASE_URL` (set them to different values in the test and check)
- Assert `OpenAICompatVisionProvider.supports_vision` is `True` for `qwen2.5vl:72b` and
  `False` for `qwen2.5:72b` (text-only)
- Assert `OpenAICompatTextProvider` sets `base_url` correctly from `LLM_BASE_URL`
- Assert `OpenAICompatTextProvider` with `LLM_BASE_URL=None` defaults to OpenAI endpoint
- Mock both `anthropic.Anthropic()` and `openai.OpenAI()` in all tests — never call real APIs
- Assert `TextResponse.input_tokens` is populated for both Anthropic and OpenAI-compat responses
- Assert BGE query prefix applied only by `LocalEmbeddingProvider`, not `OpenAICompatEmbeddingProvider`
