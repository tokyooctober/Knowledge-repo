# `storage/vector_store.py` — Vector Store (Qdrant)

---
```
module:     storage/vector_store.py
spec:       storage/SPEC_vector_store.md
layer:      Storage
depends_on: config.py · logger.py
            models.py  (EmbeddedChunk → SearchResult)
used_by:    scheduler/monthly_job.py  (upsert, delete_by_url)
            query/retriever.py  (search)
services:   Qdrant  (vector database, localhost:6333 or configured host)
```
---

## Purpose
Manage all interactions with the Qdrant vector database: collection creation, upserting embedded chunks, vector similarity search, and deletion. Acts as the sole gateway between the application and Qdrant.

---

## Responsibilities
- Create and configure Qdrant collections on first run
- Upsert `EmbeddedChunk` objects as Qdrant points (vector + payload)
- Perform top-k approximate nearest-neighbour search by query vector
- Support metadata filtering (by tag, date range, URL) at search time
- Delete points by article URL (for update/removal workflows)
- Report collection statistics (point count, disk usage)

---

## Inputs & Outputs by Operation

### `upsert`
| Input | Type | Description |
|---|---|---|
| `embedded_chunks` | `List[EmbeddedChunk]` | Chunks with vectors and metadata |

Returns: `int` — number of points successfully upserted

### `search`
| Input | Type | Description |
|---|---|---|
| `query_vector` | `list[float]` | Embedded query from `embedder.embed_query()` |
| `top_k` | `int` | Number of results to return (default: 6) |
| `filters` | `dict` | Optional metadata filters (tags, date range) |

Returns: `List[SearchResult]`

### `delete_by_url`
| Input | Type | Description |
|---|---|---|
| `article_url` | `str` | URL whose points should be deleted |

Returns: `int` — number of points deleted

---

## Data Model

### Qdrant Point structure
```
id:      chunk_id (str → UUID5 derived from chunk_id for Qdrant compat)
vector:  embedding (list[float], length = EMBEDDING_DIM)
payload: {
    "chunk_id":      str,
    "article_url":   str,
    "article_title": str,
    "published_at":  ISO8601 str | null,
    "tags":          list[str],
    "text":          str,
    "content_type":  str,          # "body" | "table" | "image_transcription"
    "chunk_index":   int,
    "total_chunks":  int,
    "model_name":    str
}
```

### `SearchResult` dataclass
```python
@dataclass
class SearchResult:
    score:         float       # cosine similarity (0–1; higher = more relevant)
    text:          str         # chunk text
    chunk_id:      str         # from the payload — the stable id chunker.py assigned
    article_url:   str
    article_title: str
    published_at:  datetime | None
    tags:          list[str]
    content_type:  str         # "body" | "table" | "image_transcription"
    chunk_index:   int
```

`chunk_id` is carried through so `retriever.py` can log which chunk it kept or capped and
`answerer.py` / `app.py` can key the debug excerpt panel on something stable. It is in the
Qdrant payload already (see *Qdrant Point structure*); `search` just needs to copy it out.
This dataclass is defined once in `models.py` — the copy here is for reference.

`content_type` is written into the Qdrant payload by `upsert` and read back by `search`.
`query/retriever.py` filters on it, and `app.py` uses it to label a source card as coming
from a chart transcription rather than prose. It was previously absent from this dataclass
while `chunker.py` and `retriever.py` both assumed it — that gap is closed here.

---

## Qdrant Collection Configuration

`COLLECTION_NAME`, `EMBEDDING_DIM`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_IN_MEMORY` and
`UPSERT_BATCH_SIZE` come from `config.py`. The four index-tuning values below are
**module-level constants in `vector_store.py`, not `config.py`** — `DISTANCE_METRIC` is a
`qdrant_client` enum and putting it in `config` would drag that dependency into every
module's `import config`, and none of the four is a knob a user should turn.

```python
# config.py
COLLECTION_NAME   = "knowledge_repo"
EMBEDDING_DIM     = 1024                    # must match embedder

# vector_store.py (module constants)
DISTANCE_METRIC   = Distance.COSINE         # cosine for normalised vectors
ON_DISK_PAYLOAD   = True                    # persist payload to disk
HNSW_M            = 16                      # HNSW graph connectivity
HNSW_EF_CONSTRUCT = 100                     # build-time accuracy
```

**Why HNSW?** Qdrant's default index. At the expected scale (< 100k chunks), HNSW gives sub-millisecond search with excellent recall. No tuning needed until > 1M points.

---

## Core Logic

### Collection initialisation
```
On __init__, check if collection exists.
If not: create with COSINE distance, EMBEDDING_DIM, HNSW config.
If it exists: compare its vector size to EMBEDDING_DIM; raise ModelMismatchError on a
             mismatch (a changed embedding model is a drop-and-reindex, not a live edit).
```

The embedding **model name** is not known at `__init__` — it arrives on the
`EmbeddedChunk`s. So it is recorded lazily: the first `upsert` writes it onto a sentinel
point (Qdrant has no first-class collection metadata), and every later `upsert` compares
`embedded_chunks[0].model_name` against it and raises `ModelMismatchError` on a mismatch.
`recorded_model() -> str | None` exposes it so `query/retriever.py` can guard query-time
embedding against the same value. The sentinel is filtered out of `count()` and `search()`.

### Collection drop (`--reset` only)
```
client.delete_collection(COLLECTION_NAME)
then re-run collection initialisation, so the object is usable afterwards.
```
Drops every point and the collection itself, then recreates it empty. The recorded model
name goes with it (the sentinel is dropped), so the next `upsert` records whatever model
it is given — this is what makes a drop the sanctioned way past a `ModelMismatchError`.
Recreating is part of the operation, not a separate step: a half-reset that leaves no
collection turns the next run's first `upsert` into an obscure Qdrant 404 instead of a
clean first-run creation.

This is the only method that removes points it was not given a URL for. It exists solely
for `monthly_job.py --reset` and must never be called from an ingestion path — a "clear
and rebuild" that runs automatically is how an interrupted run turns into an empty index.

Dropping the collection is also the *correct* way to change embedding models: the stored
`model_name` check exists to refuse a mixed-model collection, and the resolution is a drop
plus a full re-ingest, not a bypass of the check.

### Upsert
```
1. Convert chunk_id → UUID5 for Qdrant point ID
2. Batch into UPSERT_BATCH_SIZE groups
3. For each batch: client.upsert(collection_name, points=batch, wait=True)
4. Return total count upserted
```

### Search
```
1. Build optional Qdrant Filter from filters dict
   Supported filter keys:
     "tags"       → must contain any of given tags (MatchAny)
     "date_after" → published_at >= value (Range)
     "date_before"→ published_at <= value (Range)
     "content_type" → exact match (MatchValue): "body" | "table" | "image_transcription"
2. client.search(collection_name, query_vector, limit=top_k, query_filter=filter)
3. Map ScoredPoint → SearchResult
4. Return list, already sorted by score descending
```

---

## Configuration Constants
```python
QDRANT_HOST        = "localhost"
QDRANT_PORT        = 6333
QDRANT_IN_MEMORY   = False   # True for tests / local dev without Docker
UPSERT_BATCH_SIZE  = 100     # points per upsert call
DEFAULT_TOP_K      = 6
```

`search` returns Qdrant's raw top-k by score, unfiltered. `MIN_SCORE_THRESHOLD` and
`MAX_CHUNKS_PER_ARTICLE` are **not** this module's concern — `query/retriever.py` over-fetches
(`top_k * 2`) and applies both. This module has no opinion about what score is "good enough".

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Qdrant server unreachable | Raise `VectorStoreConnectionError` with host:port |
| Collection model_name mismatch | Raise `ModelMismatchError` with both names and re-index instructions |
| Upsert partial failure | Log failed point IDs; raise after batch if any failed |
| Search returns 0 results | Return `[]` (not an error; handled upstream in retriever) |
| Duplicate `chunk_id` on upsert | Qdrant overwrites; this is the intended behaviour for updates |

---

## Key Dependencies
- `qdrant-client` — official Python client (sync + async)
- `uuid` — UUID5 generation for point IDs (stdlib)

---

## Public Interface
```python
class VectorStore:
    def __init__(self): ...         # connect and initialise collection

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int: ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        filters: dict | None = None,
    ) -> list[SearchResult]: ...

    def delete_by_url(self, article_url: str) -> int: ...

    def drop_collection(self) -> None: ...   # --reset only; drops and recreates empty

    def recorded_model(self) -> str | None: ...  # model that built the collection, or None

    def count(self) -> int: ...     # real points (the model sentinel is excluded)

    def stats(self) -> dict: ...    # {collection_name, points, model_name, dim, status}
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.storage.vector_store"
```

| Event | Level | Extra fields |
|---|---|---|
| Connecting to Qdrant | INFO | `host`, `port` |
| Collection already exists | DEBUG | `collection_name`, `point_count` |
| Creating new collection | INFO | `collection_name`, `dim`, `distance_metric` |
| Model mismatch detected | CRITICAL | `collection_name`, `stored_model`, `configured_model` |
| Qdrant unreachable | CRITICAL | `host`, `port`, `error_type` |
| Upsert batch started | DEBUG | `batch_index`, `batch_size`, `collection_name` |
| Upsert batch complete | DEBUG | `batch_index`, `upserted_count` |
| Upsert partial failure | ERROR | `failed_point_ids`, `error_type` |
| Search executed | DEBUG | `query_vector_dim`, `top_k`, `filters`, `result_count` |
| Search returned 0 results | DEBUG | `top_k`, `filters` — not a warning here; the retriever decides whether 0 usable results is a problem |
| Delete by URL started | DEBUG | `url`, `collection_name` |
| Delete by URL complete | INFO | `url`, `deleted_count` |
| Delete matched 0 points | WARNING | `url` — URL may already have been removed |

---

## Testing Notes
- Use `QDRANT_IN_MEMORY = True` in tests (no Docker required)
- Assert upserted count matches input length
- Assert `search` returns raw top-k ordered by score, with **no** score-threshold or
  per-article filtering applied (that is the retriever's job)
- Assert every `SearchResult` carries the `chunk_id` from its payload
- Assert search with tag filter only returns chunks matching that tag
- Assert `delete_by_url` removes all points for that URL and no others
- Assert model mismatch raises `ModelMismatchError`
- Assert duplicate upsert (same chunk_id) overwrites, not duplicates
- Assert `drop_collection` leaves `count() == 0` and a working collection: an `upsert`
  immediately afterwards succeeds without re-instantiating `VectorStore`
- Assert `drop_collection` followed by init with a *different* model does not raise —
  the drop is the sanctioned way past `ModelMismatchError`
