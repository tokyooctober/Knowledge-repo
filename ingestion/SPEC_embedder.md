# `ingestion/embedder.py` — Sentence Embedder

---
```
module:     ingestion/embedder.py
spec:       ingestion/SPEC_embedder.md
layer:      Ingestion — embedding
depends_on: config.py · logger.py · llm_provider.py (EmbeddingProvider)
            models.py  (Chunk → EmbeddedChunk)
used_by:    scheduler/monthly_job.py
            query/retriever.py  (embed_query reuses same provider)
input:      list[Chunk]  (from ingestion/chunker.py)
output:     list[EmbeddedChunk]  →  passed to storage/vector_store.py
services:   embedding model  (via llm_provider.py: local or API)
```
---

## Purpose
Convert a list of `Chunk` objects into dense vector embeddings. Produces `EmbeddedChunk` objects that are ready to be upserted into the vector database.

---

## Responsibilities
- Load and cache the embedding model (local or API-based)
- Embed chunk text in batches for efficiency
- Attach the embedding vector to each chunk
- Work with whichever backend `llm_provider.get_embedding_provider()` returns — local
  (`sentence-transformers`), OpenAI, or any OpenAI-compatible endpoint — with no
  SDK imports or model names of its own
- Ensure the same model is always used for both indexing and query-time embedding

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `chunks` | `List[Chunk]` from chunker | Chunks to embed |
| `EMBEDDING_BACKEND` | config | `"local"` \| `"openai"` \| `"openai_compat"` (resolved by `llm_provider`) |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `List[EmbeddedChunk]` | Python list | Chunks with `.embedding` vector attached |

### `EmbeddedChunk` dataclass
```python
@dataclass
class EmbeddedChunk:
    chunk: Chunk               # original Chunk, all fields preserved
    embedding: list[float]     # dense vector, length = EMBEDDING_DIM
    model_name: str            # e.g. "BAAI/bge-large-en-v1.5"
```

---

## Embedding Model Options

All embedding backends are configured and instantiated through `llm_provider.get_embedding_provider()`. The embedder itself contains no SDK imports and no model-specific logic — it calls the provider interface only. Refer to `SPEC_llm_provider.md` for the full backend reference table.

### Option A: Local — `sentence-transformers` (default, recommended)
Set `EMBEDDING_BACKEND = "local"`. The model runs in-process on CPU or GPU.

Recommended models:

| Model | Dim | Notes |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | 1024 | Best retrieval quality; requires BGE query prefix |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Strong alternative; uses `search_query:` prefix |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Lightweight; lower quality but fast on CPU |

All download automatically on first use from Hugging Face Hub. No API key needed.

### Option B: OpenAI-compatible API endpoint
Set `EMBEDDING_BACKEND = "openai_compat"` and configure `EMBEDDING_BASE_URL`.

Works with: OpenAI (`base_url=None`), Ollama (`http://localhost:11434/v1`), vLLM, and any OpenAI-compatible server.

| Serving | Model | Dim |
|---|---|---|
| OpenAI (cloud) | `text-embedding-3-small` | 1536 |
| Ollama (local) | `nomic-embed-text` | 768 |
| Ollama (local) | `mxbai-embed-large` | 1024 |

---

## Configuration Constants
```python
# Embedding backend and model are configured in config.py and resolved by llm_provider.py.
# embedder.py reads the active provider via get_embedding_provider() — no model names here.
BATCH_SIZE           = 64      # chunks per embedding batch call
NORMALIZE_EMBEDDINGS = True    # L2 normalise vectors for cosine similarity
```

The active model's `embedding_dim` and `query_prefix` are read from the provider object at runtime:
```python
provider = get_embedding_provider()
dim    = provider.embedding_dim   # e.g. 1024 for bge-large, 768 for nomic-embed
prefix = provider.query_prefix    # applied at query time only, not at indexing time
```

---

## Core Logic

```
provider = llm_provider.get_embedding_provider()   # singleton, loaded once

1. BATCH EMBEDDING
   For batch in chunks_batched(chunks, BATCH_SIZE):
     texts   = [chunk.text for chunk in batch]
     vectors = provider.embed(texts)               # normalisation handled by provider
     For chunk, vector in zip(batch, vectors):
       yield EmbeddedChunk(
         chunk=chunk,
         embedding=vector,
         model_name=provider.model_name,
       )

2. RETURN list of EmbeddedChunk
```

---

## Critical Rule: Model Consistency
The provider used for indexing **must be identical** to the provider used at query time. Enforce this by:
- Writing `provider.model_name` into each `EmbeddedChunk` and storing it in the Qdrant payload
- At query time, `retriever.py` reads `model_name` from the collection metadata and raises `ModelMismatchError` if it doesn't match the currently configured provider
- Never change `EMBEDDING_BACKEND`, `LOCAL_EMBEDDING_MODEL`, or `OPENAI_EMBEDDING_MODEL` on an existing collection without re-indexing from scratch

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Empty `chunks` list | Return `[]` immediately |
| Provider fails to initialise (model missing, endpoint down) | `llm_provider.py` raises at startup — propagate |
| Embedding API rate limit | Handled by `llm_provider.py` with retry; propagated if retries exhausted |
| Embedding dimension mismatch (provider vs collection) | Raise `ModelMismatchError` with both dimensions and re-index instructions |

---

## Key Dependencies
- `llm_provider.py` — provides `EmbeddingProvider` (no direct SDK imports in embedder.py)

---

## Public Interface
```python
def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Embed a list of chunks using the configured embedding provider.

    Provider (local sentence-transformers, OpenAI, Ollama, etc.) is determined
    by EMBEDDING_BACKEND in config. Processes in batches. Returns EmbeddedChunk
    objects with .embedding and .model_name populated.
    """

def embed_query(query_text: str) -> list[float]:
    """Embed a single query string for retrieval.

    Applies the provider's query_prefix automatically (e.g. BGE instruction prefix).
    Uses the same provider as embed_chunks.
    """
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.ingestion.embedder"
```

| Event | Level | Extra fields |
|---|---|---|
| Provider acquired | DEBUG | `backend`, `model_name`, `embedding_dim` |
| Embedding batch started | DEBUG | `batch_index`, `batch_size`, `backend` |
| Embedding batch complete | DEBUG | `batch_index`, `batch_size`, `elapsed_ms` |
| API error (rate limit, auth) | delegated to `llm_provider.py` logging | — |
| Embedding dimension mismatch | CRITICAL | `expected_dim`, `actual_dim`, `model_name` |
| All chunks embedded | INFO | `total_chunks`, `backend`, `model_name`, `elapsed_ms` |
| embed_query: query_prefix applied | DEBUG | `prefix`, `query_length_tokens` |
| embed_query: query truncated | WARNING | `original_tokens`, `max_tokens`, `model_name` |

---

## Testing Notes
- Mock `llm_provider.get_embedding_provider()` to return a `MockEmbeddingProvider`
- Assert `embed_chunks` output length equals input length
- Assert `EmbeddedChunk.model_name` matches `provider.model_name`
- Assert `embed_query` applies `provider.query_prefix` to the text before embedding
- Assert empty input returns empty output
- Assert provider is called once per batch, not once per chunk
- Swap mock for a real local provider in integration tests; assert embedding_dim matches config
