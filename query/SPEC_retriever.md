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
   subqueries = decompose(query) if ENABLE_QUERY_DECOMPOSITION else []
   (decompose() returns [] whenever decomposition is off, the query is short,
   the LLM call fails, or its output is malformed/degenerate — so this step is
   a no-op in every case except a genuine multi-topic split)

2. EMBED + SEARCH the ORIGINAL query, always
   holistic = vector_store.search(
       query_vector=embedder.embed_query(query),
       top_k=_pool_depth(top_k, decomposed),
       filters=filters,
   )
   (the original query is searched whether or not decomposition fired.
   _pool_depth() takes its floor from whoever consumes the pool: with
   ENABLE_RERANK on it is at least RERANK_POOL, because the reranker measured
   far worse at depth 12 than at 48 and that need has nothing to do with
   whether the query decomposed. Without a reranker it stays at top_k * 2.)

3. EMBED + SEARCH, once per subquery (only when subqueries is non-empty)
   for sq in subqueries:
       results = vector_store.search(
           query_vector=embedder.embed_query(sq),
           top_k=top_k * 2,        # over-fetch for post-filtering
           filters=filters,
       )

4. MERGE BY HOLISTIC RANK (only when subqueries is non-empty)
   pooled = merge_by_holistic_rank(holistic, per_subquery_results)
   (candidates in the holistic list keep their score against the ORIGINAL
   query and stay in that order; candidates only the subqueries found have no
   comparable score and sort after them, by best subquery rank. With no
   subqueries, pooled is just the holistic results — identical to the
   single-query behaviour)

5. CROSS-ENCODER RERANK (only when ENABLE_RERANK)
   pooled[:RERANK_POOL] = rerank(query, pooled[:RERANK_POOL])
   (a cross-encoder scores (query, chunk) JOINTLY, rather than comparing two
   independently computed embeddings, and reorders on that. It only reorders —
   SearchResult.score stays cosine similarity, because MIN_SCORE_THRESHOLD and
   every downstream reader assume that scale, and cross-encoder outputs are
   logits that can be negative. Any failure returns the vector order unchanged:
   reranking is an optimisation, never a hard dependency)

6. POST-FILTERING, once, on the pooled set
   a. Discard results with score < MIN_SCORE_THRESHOLD (0.35 default)
   b. Optional: MAX_CHUNKS_PER_ARTICLE — cap results from the same article
      (prevents a very long article — or a dominant article across several
      subqueries — from dominating all top_k slots)
   c. Trim to top_k

7. RETURN results (in pooled order — reranked if enabled, otherwise
   score-descending, with any subquery-only tail following)
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

Search the ORIGINAL query (wider — see Core Logic step 2) and once per subquery,
then merge by holistic rank:
  Candidates the holistic search returned keep their score against the original
  query and stay in that (score-descending) order. Candidates only a subquery
  found sort after all of those, ordered by their best subquery rank.
```

**Subqueries decide what is *considered*; the original query decides what *wins*.**
Decomposition must be strictly additive — it may widen the candidate pool, never remove a
chunk the plain query would have returned.

**Why not max-score fusion across subqueries.** An earlier version of this spec pooled every
subquery's results and sorted by raw cosine, reasoning that one embedding model means one
vector space means comparable scores. That is wrong, and it caused a measured regression
(`context_recall` 0.915 → 0.690, `hit_rate@k` 1.000 → 0.900 on the n=10 smoke set). A cosine
score is only meaningful *relative to the query vector it was measured against*: a short,
generic subquery sits in a denser region of the space and produces a different baseline
similarity than the long, specific original question. Observed directly — subquery matches
scored **0.7867** while the holistic matches they displaced scored **0.7743**, and the
displaced context was better. Sorting a mixed pool by these numbers compares two rulers with
different zero points. Only the holistic list is scored on what the user actually asked, so
only it may order the result.

**Why the holistic search is widened to `top_k * HOLISTIC_OVERFETCH`.** It serves two jobs:
it supplies the ranking yardstick for subquery-surfaced candidates (any that appear in this
list carry a real holistic score; any that do not are known to fall outside it and sort
below), and it reaches genuinely relevant chunks that the `top_k * 2` window cut off — for a
two-topic question the second topic's chunks typically sit around holistic rank 15-25, which
is where the multi-source benefit actually comes from once `MAX_CHUNKS_PER_ARTICLE` evicts
the dominant article's surplus.

**Why filtering happens once, after merging, not per subquery.** `MIN_SCORE_THRESHOLD` is
near-inert in practice (real scores run 0.64-0.83 against a 0.35 floor), so applying it once
is simply simpler. `MAX_CHUNKS_PER_ARTICLE` is *not* equivalent: applying it per subquery
would let a dominant article place up to the cap from *each* subquery (e.g. 3 + 3 = 6),
reintroducing the single-article-dominance problem the cap exists to prevent, just laundered
through merging. It must run once, globally, on the merged pool.

**Why the LLM decides subquery count instead of a fixed N.** A fixed split forces an
artificial second sub-question onto every single-topic question and under-serves the rare
question needing three or more articles. The decomposition prompt's line count *is* the
complexity judgment — there is no separate classifier call.

**Fallback — always to `[query]`, never to empty results:** a `ProviderConnectionError`
from the decomposition call, unparseable/empty output, or output that collapses to 0 or 1
distinct lines all fall back to exactly today's single-query path. Decomposition is
strictly additive when it works and a no-op when it doesn't.

Enable via `ENABLE_QUERY_DECOMPOSITION` in config. Distinct from `ENABLE_QUERY_REWRITING` —
a separate flag, not a redefinition of it, since that flag already has a different
documented behaviour (see above). Worst-case added cost per query: one text completion
(decomposition) plus up to `MAX_SUBQUERIES` extra embed + search calls;
`MIN_DECOMPOSITION_WORDS` skips the LLM call entirely for short queries where a split is
implausible.

> **Currently OFF, and the merge below is broken.** Measured across all 50 eval questions:
> decomposition changed the retrieved set on **zero** of them, because
> `_merge_by_holistic_rank` appends subquery-only finds *behind* a holistic list that already
> fills `RERANK_POOL` — so they are sliced off before the reranker ever scores them. The cost
> was ~2.2 s/query on 47 of 50 queries, for nothing.
>
> Giving those finds slots does work, but barely: RRF(k=60) and simply reserving 12 of the 48
> slots produce *identical* results, each gaining a net one question in fifty (2 better, 1
> worse) — a coin flip, not a result. **Fix the merge before re-enabling the flag**, and
> prefer slot reservation over RRF, which adds a tuning parameter for no measured gain.

---

## Hybrid Search (optional enhancement)
Dense search ranks chunks by what they are *about*. It struggles when one rare exact term is
the whole question. In q004, "Kirkland Lake Gold" is the only word that separates the right
chunk from five other "portfolio changes" chunks, all within 0.02 cosine. BM25 matches exact
terms, so hybrid search runs both and fuses them. Off by default: set
`ENABLE_HYBRID_SEARCH=true` (env or `.env`).

```
dense  = vector_store.search(query_vector, top_k=HYBRID_FETCH)       # cosine
sparse = vector_store.search_sparse(query_text, top_k=HYBRID_FETCH)  # BM25
fused  = RRF(dense, sparse)        # sum 1/(RRF_K + rank); dense instance kept on overlap
BM25-only chunks: .score = vector_store.score_by_ids(query_vector, ids)   # real cosine
ENABLE_RERANK: fused[:RERANK_POOL] -> cross-encoder   # RRF picks the pool; tail dropped
otherwise:     fused as is                            # RRF order is the final order
-> the usual MIN_SCORE_THRESHOLD / MAX_CHUNKS_PER_ARTICLE / top_k loop
```

- **The BM25 index is a Qdrant sparse vector** named `bm25` on every point, with IDF applied
  by Qdrant (`Modifier.IDF`), so it can't drift out of sync with the dense index. See
  `storage/SPEC_vector_store.md`. A collection built before hybrid search lacks it:
  `python -m scheduler.monthly_job --add-sparse` adds it once, and every later upsert writes
  both vectors.
- **Fuse by rank, never by score.** Cosine and BM25 scores are on unrelated scales. That
  mistake is what broke `_fuse()` in the decomposition work.
- **`.score` stays cosine.** RRF decides the order. A chunk only BM25 found is re-scored by
  cosine against the query vector, so `MIN_SCORE_THRESHOLD` still means what it says, and
  logged and eval scores stay comparable with dense-only runs.
- **`HYBRID_FETCH` is the same with or without the reranker**, so in an A/B the reranker is
  the only difference.
- **No BM25 index → dense only**, with one warning per process. A missing index must never
  take retrieval down, the same rule as the reranker fallback.
- **RRF runs client-side**, not through Qdrant's `FusionQuery`. That returns RRF scores in
  place of cosine and would break both rules above.
- Not combined with decomposition. If both are on, the fused list takes the holistic list's
  place and subquery finds append behind it, as before.

---

## Configuration Constants
```python
DEFAULT_TOP_K              = 6
MAX_CHUNKS_PER_ARTICLE     = 3       # max results from a single article
MIN_SCORE_THRESHOLD        = 0.35    # below this = not relevant
ENABLE_QUERY_REWRITING     = False
ENABLE_HYBRID_SEARCH       = False   # env-overridable; see Hybrid Search
HYBRID_FETCH               = 48      # candidates from EACH of dense and BM25
RRF_K                      = 60      # reciprocal rank fusion constant
ENABLE_QUERY_DECOMPOSITION = False   # see "Currently OFF" note above
MAX_SUBQUERIES             = 4       # cap on LLM-decided subquery fan-out
MIN_DECOMPOSITION_WORDS    = 6       # below this word count, skip decomposition (cost gate)
HOLISTIC_OVERFETCH         = 8       # x top_k for the original query's own search when
                                     # decomposing (see Query Decomposition)
ENABLE_RERANK              = True    # env-overridable (ENABLE_RERANK=false)
RERANK_MODEL               = "BAAI/bge-reranker-v2-m3"   # 8192-token context
RERANK_POOL                = 48      # candidates scored per query (~38 ms each)
RERANK_MAX_LENGTH          = 1024    # covers the ~610-token worst-case pair
```

**Why a long-context reranker.** Chunks are `CHUNK_SIZE` (512) tokens, so a (query, chunk)
pair runs ~550 tokens at the median and ~610 at the max. Every 512-limit cross-encoder —
`ms-marco-MiniLM`, `bge-reranker-base`, `bge-reranker-large` — therefore truncates the chunk
tail on ~69% of pairs, and measured, that truncation *demoted the correct article* on
single-source questions (recall 1.000 -> 0.750). `bge-reranker-v2-m3` has identical capacity
to `bge-reranker-large` (24 layers, 1024 hidden, ~558M params) but an 8192-token window, so
it scores whole chunks and holds single-source recall at 1.000.

**Why `RERANK_POOL = 48`.** Cost is linear at ~38 ms per candidate, so the pool size *is* the
latency budget: 24 costs ~0.97 s, 48 ~1.84 s, 200 ~7.4 s. Quality does not follow the same
curve — measured recall rises to 48 and then falls (deeper pools hand the reranker more
distractors than signal), so 48 is both the best measured depth and near the practical
latency ceiling.

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

**Query decomposition (currently disabled — see the note above):**
- Unit-test `_merge_by_holistic_rank()` directly with hand-built result lists — no mocking
  needed: holistic order is preserved ahead of subquery-only finds, a chunk found by both
  keeps its *holistic* score rather than the larger subquery one, the subquery-only tail
  orders by best rank across lists, and an empty subquery list is a no-op
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
