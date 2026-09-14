# `query/retriever.py` — Retriever

---
```
module:     query/retriever.py
spec:       query/SPEC_retriever.md
layer:      Query
depends_on: config.py · logger.py
            llm_provider.py  (EmbeddingProvider, via embed_query)
            llm_provider.py  (TextProvider, via get_text_provider — query decomposition only)
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
1. DECOMPOSE (optional — see Query Decomposition below)
   subqueries = decompose(query) if ENABLE_QUERY_DECOMPOSITION else [query]
   (decompose() returns [query] unchanged whenever decomposition is off, the
   query is short, the LLM call fails, or its output is malformed/degenerate —
   so this step is a no-op in every case except a genuine multi-topic split)

2. EMBED + SEARCH, once per subquery
   for sq in subqueries:
       query_vector = embedder.embed_query(sq)
       (embed_query handles BGE instruction prefix internally)
       results = vector_store.search(
           query_vector=query_vector,
           top_k=top_k * 2,        # over-fetch for post-filtering
           filters=filters,
       )

3. FUSE (only when len(subqueries) > 1 — see Query Decomposition below)
   pooled = fuse(per_subquery_results)   # dedupe by chunk_id, keep max score
   (with one subquery, pooled is just that subquery's results — identical to
   today's single-query behaviour)

4. POST-FILTERING, once, on the pooled set
   a. Discard results with score < MIN_SCORE_THRESHOLD (0.35 default)
   b. Optional: MAX_CHUNKS_PER_ARTICLE — cap results from the same article
      (prevents a very long article — or a dominant article across several
      subqueries — from dominating all top_k slots)
   c. Trim to top_k

5. RETURN results (sorted by score, descending)
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

Still issues exactly **one** embed against **one** rewritten query string — it does not
fan out across articles and does not address multi-source coverage. See **Query
Decomposition** below for that.

---

## Query Decomposition (optional enhancement)

Targets a specific, measured gap: two-source questions retrieve at roughly half the
`recall@k` of single-source ones, while ranking quality (MRR) barely changes between the
two groups — the retriever finds the right article and ranks it well when it finds it at
all; it just doesn't reach a second article a single dense-vector query embedding didn't
surface. Decomposition fans a question out into several subqueries so a second (or third)
article gets a real chance to enter the candidate pool before filtering runs.

```
If ENABLE_QUERY_DECOMPOSITION and len(query.split()) >= MIN_DECOMPOSITION_WORDS:
  Pass raw query to a text-provider call:
    "Decide whether this question is one topic or genuinely combines multiple
     distinct topics. If one topic, return it unchanged as the only line.
     If multiple, return each as its own self-contained search query, one per
     line. Return at most MAX_SUBQUERIES lines."
  Parse the response into lines: strip bullets/numbering, drop blanks, dedupe
  case/whitespace-insensitively, cap at MAX_SUBQUERIES.
  If 0 or 1 distinct lines survive -> fall back to [query] (not a real split).
  Otherwise -> subqueries = the parsed lines.

Embed and search once per subquery (see Core Logic step 2), then fuse:
  Dedupe by chunk_id across all subqueries' results, keeping the MAX score seen
  for each chunk; sort by that score descending.
```

**Why max-score fusion, not reciprocal rank fusion (RRF).** RRF exists to combine
*incomparable* scoring systems (e.g. BM25 vs. cosine — the actual case in Hybrid Search
below). Here every subquery is embedded by the same model into the same vector space, so
cosine scores are directly comparable across subqueries; RRF would discard real relevance
magnitude and produce rank-only scores incompatible with `MIN_SCORE_THRESHOLD` and every
current reader of `SearchResult.score`. Max-score dedupe keeps `.score` meaning "cosine
similarity," unchanged whether decomposition ran or not.

**Why filtering happens once, after fusion, not per subquery.** `MIN_SCORE_THRESHOLD` is
equivalent either way under max-score fusion, so applying it once is simply simpler.
`MAX_CHUNKS_PER_ARTICLE` is *not* equivalent: applying it per subquery would let a
dominant article place up to the cap from *each* subquery (e.g. 3 + 3 = 6), reintroducing
the single-article-dominance problem the cap exists to prevent, just laundered through
fusion. It must run once, globally, on the fused pool.

**Why the LLM decides subquery count instead of a fixed N.** A fixed split forces an
artificial second sub-question onto every single-topic question and under-serves the rare
question needing three or more articles. The decomposition prompt's line count *is* the
complexity judgment — there is no separate classifier call.

**Fallback — always to `[query]`, never to empty results:** a `ProviderConnectionError`
from the decomposition call, unparseable/empty output, or output that collapses to 0 or 1
distinct lines all fall back to exactly today's single-query path. Decomposition is
strictly additive when it works and a no-op when it doesn't.

Enable via `ENABLE_QUERY_DECOMPOSITION = True` in config. Distinct from `ENABLE_QUERY_
REWRITING` — a separate flag, not a redefinition of it, since that flag already has a
different documented behaviour (see above). Worst-case added cost per query: one text
completion (decomposition) plus up to `MAX_SUBQUERIES` extra embed + search calls;
`MIN_DECOMPOSITION_WORDS` skips the LLM call entirely for short queries where a split is
implausible.

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
DEFAULT_TOP_K              = 6
MAX_CHUNKS_PER_ARTICLE     = 3       # max results from a single article
MIN_SCORE_THRESHOLD        = 0.35    # below this = not relevant
ENABLE_QUERY_REWRITING     = False
ENABLE_HYBRID_SEARCH       = False
ENABLE_QUERY_DECOMPOSITION = False
MAX_SUBQUERIES             = 4       # cap on LLM-decided subquery fan-out
MIN_DECOMPOSITION_WORDS    = 6       # below this word count, skip decomposition (cost gate)
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
| Decomposition call fails (`ProviderConnectionError`) | Log WARNING; fall back to `[query]` — no exception raised |
| Decomposition output malformed (0 usable lines) | Log WARNING; fall back to `[query]` |
| Decomposition output degenerate (0 or 1 distinct lines after dedup) | Log DEBUG; fall back to `[query]` |

---

## Key Dependencies
- `embedder.py` — `embed_query()` function
- `vector_store.py` — `VectorStore.search()`
- `llm_provider.py` — `get_text_provider()` (query decomposition only)
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
| Query decomposition skipped (below `MIN_DECOMPOSITION_WORDS`) | DEBUG | `query` |
| Query decomposed | INFO | `query`, `subquery_count` |
| Query decomposition call failed — using original query | WARNING | `query` |
| Decomposition output malformed — using original query | WARNING | `query` |
| Decomposition collapsed to a single query — using original query | DEBUG | `query` |
| Vector search executed | DEBUG | `top_k_requested`, `over_fetch_k`, `filters` |
| Results before filtering | DEBUG | `raw_result_count`, `min_score`, `max_score`, `subquery_count` |
| Results below score threshold discarded | DEBUG | `discarded_count`, `min_score_threshold` |
| Per-article cap applied | DEBUG | `url`, `kept`, `discarded`, `cap` |
| No results after filtering | WARNING | `query`, `min_score_threshold`, `filters` |
| Retrieval complete | INFO | `query`, `result_count`, `top_score`, `bottom_score`, `subquery_count`, `decomposition_triggered` |

---

## Testing Notes
- Mock `embed_query` to return a fixed vector; assert it's passed to `vector_store.search`
- Assert results are sorted by score descending
- Assert results below `MIN_SCORE_THRESHOLD` are excluded
- Assert `MAX_CHUNKS_PER_ARTICLE` cap: no more than N results from the same article URL
- Assert empty query raises `ValueError`
- Assert over-long query is truncated (check token count of input to `embed_query`)
- Integration test: seed Qdrant in-memory with known chunks; assert correct chunk retrieved for matching query

**Query decomposition (`ENABLE_QUERY_DECOMPOSITION = True`):**
- Unit-test `_fuse()` directly with hand-built result lists — no mocking needed: disjoint
  articles merge, duplicate `chunk_id` across subqueries keeps the max score, one empty
  subquery list doesn't break fusion, identical results from every subquery collapse to
  that same list unchanged (the regression guard for single-source questions)
- Mock `get_text_provider()` the same way `embed_query`/`get_embedding_provider` are
  mocked; assert a well-formed N-line response fans out to N `vector_store.search` calls
- Assert malformed output, a single line identical to the original, and a
  `ProviderConnectionError` from the decomposition call all fall back to exactly one
  search call using the original query — never to empty results and never propagating
- Assert a query below `MIN_DECOMPOSITION_WORDS`, and the flag being off, both skip the
  decomposition call entirely (`get_text_provider().complete` never invoked)
- Assert a response longer than `MAX_SUBQUERIES` lines is capped, and near-duplicate lines
  (case/whitespace) collapse to one subquery
- Component test: script a fake store to return different results per subquery text
  (`embed_query` mocked to identity so the "vector" is the query text); assert the fused,
  filtered, capped, trimmed output matches a single `store.search` call's contract
- Not a unit test — a manual regression gate: re-run `eval/score_ragas.py` on the frozen
  test set with the flag on vs. off; `recall@k`/`hit_rate` on multi-source questions must
  rise without regressing single-source questions (see `eval/METRICS.md` Finding 2)
