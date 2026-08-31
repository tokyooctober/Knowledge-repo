# `query/retriever.py` — Retriever

---
```
module:     query/retriever.py
spec:       query/SPEC_retriever.md
layer:      Query
depends_on: config.py · logger.py
            llm_provider.py  (EmbeddingProvider, via embed_query)
            ingestion/embedder.py  (embed_query function)
            storage/vector_store.py  (VectorStore.search)
used_by:    app.py
input:      query str  (from user)
output:     list[SearchResult]  →  passed to query/answerer.py
services:   embedding model  (via llm_provider.py)
            Qdrant  (via storage/vector_store.py)
```
---

## Purpose
Given a natural language query, embed it and retrieve the most semantically relevant chunks from the vector store. Returns a ranked list of `SearchResult` objects for the context builder.

---

## Responsibilities
- Embed the user's raw query text using the correct model and backend
- Apply the BGE instruction prefix when using the local embedding model
- Execute vector similarity search against Qdrant
- Apply optional metadata filters (tags, date range)
- Post-process results: deduplicate by article, enforce minimum score threshold
- Return a coherent, ranked list of `SearchResult` objects

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `query` | User / `app.py` | Raw natural language question |
| `top_k` | Caller / config | Number of chunks to retrieve (default: 6) |
| `filters` | Caller / optional | Dict of metadata filters (`tags`, `date_after`, `date_before`, `content_type`) |

### `content_type` filtering

Every chunk carries `content_type` — `"body"`, `"table"`, or `"image_transcription"` —
written into the Qdrant payload by `chunker.py`. Passing it in `filters` restricts the
search to one kind of content:

```python
retrieve("what did the yield curve chart show?",
         filters={"content_type": "image_transcription"})
retrieve("the CPI breakdown table", filters={"content_type": "table"})
```

Unfiltered search covers all three, which is the default and the right behaviour for most
questions. The filter exists for the case where the user is explicitly asking about a chart
or a table, and for debugging whether vision transcription is pulling its weight.

`content_type` is also returned on every `SearchResult`, so `app.py` can label a source as
coming from a chart rather than prose without a second lookup.

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `List[SearchResult]` | From `vector_store` | Ranked by score descending; may be empty |

---

## Core Logic

```
1. EMBED QUERY
   query_vector = embedder.embed_query(query)
   (embed_query handles BGE instruction prefix internally)

2. VECTOR SEARCH
   results = vector_store.search(
       query_vector=query_vector,
       top_k=top_k * 2,        # over-fetch for post-filtering
       filters=filters,
   )

3. POST-FILTERING
   a. Discard results with score < MIN_SCORE_THRESHOLD (0.35 default)
   b. Optional: MAX_CHUNKS_PER_ARTICLE — cap results from the same article
      (prevents a very long article from dominating all top_k slots)
   c. Trim to top_k

4. RETURN results (sorted by score, descending)
```

---

## Query Rewriting (optional enhancement)
Before embedding, optionally expand the query to improve recall:

```
If ENABLE_QUERY_REWRITING:
  Pass raw query to a lightweight Claude call:
  "Rewrite this question as a declarative statement suitable for document retrieval. Return only the rewritten query."
  Embed the rewritten query instead.
```

Enable via `ENABLE_QUERY_REWRITING = True` in config. Adds ~200ms latency but improves recall for short or ambiguous questions.

---

## Hybrid Search (optional enhancement)
If the collection is small (< 5000 chunks) and precision matters more than speed, combine vector search with BM25 keyword search:

```
vector_results = vector_store.search(query_vector, top_k=top_k*2)
keyword_results = bm25_index.search(query, top_k=top_k*2)
merged = reciprocal_rank_fusion(vector_results, keyword_results)
```

`rank_bm25` library handles the keyword side. Not implemented by default — enable via `ENABLE_HYBRID_SEARCH = True`. Requires a separately maintained BM25 index (built from chunk texts at ingestion time).

---

## Configuration Constants
```python
DEFAULT_TOP_K            = 6
MAX_CHUNKS_PER_ARTICLE   = 3       # max results from a single article
MIN_SCORE_THRESHOLD      = 0.35    # below this = not relevant
ENABLE_QUERY_REWRITING   = False
ENABLE_HYBRID_SEARCH     = False
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Empty query string | Raise `ValueError("Query must not be empty")` |
| Query > 512 tokens (local model limit) | Truncate to 512 tokens; log at WARNING |
| Vector store returns 0 results | Return `[]`; caller handles "no results" message |
| All results below threshold | Return `[]` |
| Vector store connection error | Propagate `VectorStoreConnectionError` to caller |

---

## Key Dependencies
- `embedder.py` — `embed_query()` function
- `vector_store.py` — `VectorStore.search()`
- `tiktoken` — query token count check

---

## Public Interface
```python
def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[SearchResult]:
    """Embed query and retrieve top-k relevant chunks.
    
    Applies score threshold and per-article caps.
    Returns empty list if no relevant chunks found.
    """
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.query.retriever"
```

| Event | Level | Extra fields |
|---|---|---|
| Empty query rejected | ERROR | `error_type` |
| Query truncated (over token limit) | WARNING | `original_tokens`, `max_tokens`, `model_name` |
| Query embedded | DEBUG | `query_length_chars`, `embedding_dim`, `backend` |
| Query rewriting enabled — calling Claude | DEBUG | `original_query` |
| Rewritten query produced | DEBUG | `original_query`, `rewritten_query` |
| Query rewriting failed — using original | WARNING | `original_query`, `error_type` |
| Vector search executed | DEBUG | `top_k_requested`, `over_fetch_k`, `filters` |
| Results before filtering | DEBUG | `raw_result_count`, `min_score`, `max_score` |
| Results below score threshold discarded | DEBUG | `discarded_count`, `min_score_threshold` |
| Per-article cap applied | DEBUG | `url`, `kept`, `discarded`, `cap` |
| No results after filtering | WARNING | `query`, `min_score_threshold`, `filters` |
| Retrieval complete | INFO | `query`, `result_count`, `top_score`, `bottom_score` |

---

## Testing Notes
- Mock `embed_query` to return a fixed vector; assert it's passed to `vector_store.search`
- Assert results are sorted by score descending
- Assert results below `MIN_SCORE_THRESHOLD` are excluded
- Assert `MAX_CHUNKS_PER_ARTICLE` cap: no more than N results from the same article URL
- Assert empty query raises `ValueError`
- Assert over-long query is truncated (check token count of input to `embed_query`)
- Integration test: seed Qdrant in-memory with known chunks; assert correct chunk retrieved for matching query
