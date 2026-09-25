"""Embed a natural-language query and return the most relevant chunks.

Over-fetches `top_k * 2` from the vector store, then applies `MIN_SCORE_THRESHOLD` and the
`MAX_CHUNKS_PER_ARTICLE` cap before trimming to `top_k`. The store returns raw top-k; the
score and per-article policy live here.

When query decomposition fires, the original query is still searched (wider, at
`top_k * HOLISTIC_OVERFETCH`) and is what ranks the results — subquery searches only widen
the candidate pool. See `_merge_by_holistic_rank`.

With `ENABLE_HYBRID_SEARCH`, the candidate pool comes from two searches instead of one: dense
(cosine) and BM25 (sparse), `HYBRID_FETCH` each, fused by reciprocal rank fusion. RRF decides
the order; chunks only BM25 found are given their real cosine score, so `.score` is still
cosine. See `_hybrid_candidates`.

With `ENABLE_RERANK`, a cross-encoder then reorders the top `RERANK_POOL` candidates by
scoring (query, chunk) jointly. It only reorders: `.score` stays cosine similarity. With
hybrid on, those `RERANK_POOL` are RRF's best, and the RRF tail below them is dropped.
"""

from __future__ import annotations

import re
from dataclasses import replace

from config import (
    DEFAULT_TOP_K,
    ENABLE_HYBRID_SEARCH,
    ENABLE_QUERY_DECOMPOSITION,
    ENABLE_QUERY_REWRITING,
    ENABLE_RERANK,
    HOLISTIC_OVERFETCH,
    HYBRID_FETCH,
    MAX_CHUNKS_PER_ARTICLE,
    MAX_SUBQUERIES,
    MIN_DECOMPOSITION_WORDS,
    MIN_SCORE_THRESHOLD,
    RERANK_BATCH_SIZE,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RERANK_POOL,
    RRF_K,
)
from ingestion.embedder import embed_query
from llm_provider import ProviderConnectionError, get_embedding_provider, get_text_provider
from logger import get_logger
from models import ModelMismatchError, SearchResult
from storage.vector_store import VectorStore

log = get_logger(__name__)

_DECOMPOSITION_SYSTEM_PROMPT = "\n".join(
    [
        "Given a question, decide whether it can be answered by searching for one topic, "
        'or whether it genuinely combines multiple distinct topics that would be better '
        'searched separately (e.g. a comparison, a "both X and Y" question, or a '
        "multi-part question).",
        "",
        "If it's one topic, return the question unchanged as the only line.",
        "If it combines distinct topics, return each as its own line, phrased as a "
        "self-contained search query.",
        f"Return at most {MAX_SUBQUERIES} lines. No numbering, bullets, or commentary.",
    ]
)

_LEADING_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

_store: VectorStore | None = None
_reranker = None
_warned_no_sparse = False


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _reset_store_for_tests() -> None:
    global _store, _reranker, _warned_no_sparse
    _store = None
    _reranker = None
    _warned_no_sparse = False


def _get_reranker():
    """Cached cross-encoder. Loading costs ~7 s, so it happens once per process."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
    return _reranker


def _rerank(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
    """Reorder the first `RERANK_POOL` candidates by a cross-encoder's joint (query, chunk)
    score, leaving `SearchResult.score` untouched — it stays cosine similarity, which
    `MIN_SCORE_THRESHOLD` and every downstream reader still assume. Any failure returns the
    input order: a missing or broken reranker must never take retrieval down with it."""
    head, tail = candidates[:RERANK_POOL], candidates[RERANK_POOL:]
    if len(head) < 2:
        return candidates
    try:
        scores = _get_reranker().predict(
            [(query, c.text) for c in head],
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception:  # noqa: BLE001 - reranking is an optimisation, never a hard dependency
        log.warning(
            "Reranking failed — falling back to vector order",
            extra={"query": query, "candidate_count": len(head), "model": RERANK_MODEL},
            exc_info=True,
        )
        return candidates
    ordered = [c for _, c in sorted(zip(scores, head, strict=True), key=lambda pair: -pair[0])]
    return ordered + tail


def _parse_subqueries(raw: str, original_query: str) -> list[str]:
    """Lines of a decomposition response, deduped and capped at `MAX_SUBQUERIES`. Empty
    (malformed) or single-line (nothing to fan out) responses return `[]` — the caller
    falls back to `[original_query]` either way."""
    lines = [_LEADING_MARKER.sub("", ln).strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        log.warning(
            "Decomposition output malformed — using original query", extra={"query": original_query}
        )
        return []

    seen: set[str] = set()
    deduped: list[str] = []
    for ln in lines:
        key = " ".join(ln.lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ln)

    if len(deduped) <= 1:
        log.debug(
            "Decomposition collapsed to a single query — using original query",
            extra={"query": original_query},
        )
        return []

    return deduped[:MAX_SUBQUERIES]


def _decompose_query(query: str) -> list[str]:
    """Subqueries for `query`, or `[]` if decomposition is off, the query is too short to
    plausibly combine multiple topics, or the LLM call fails or degenerates. Callers fall
    back to `[query]` in every `[]` case — decomposition is strictly additive."""
    if not ENABLE_QUERY_DECOMPOSITION:
        return []
    if len(query.split()) < MIN_DECOMPOSITION_WORDS:
        log.debug(
            "Query decomposition skipped — below MIN_DECOMPOSITION_WORDS", extra={"query": query}
        )
        return []

    try:
        resp = get_text_provider().complete(
            [
                {"role": "system", "content": _DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0.0,
        )
    except ProviderConnectionError:
        log.warning(
            "Query decomposition call failed — using original query", extra={"query": query}
        )
        return []

    subqueries = _parse_subqueries(resp.content, query)
    if subqueries:
        log.info("Query decomposed", extra={"query": query, "subquery_count": len(subqueries)})
    return subqueries


def _merge_by_holistic_rank(
    holistic: list[SearchResult], per_fragment: list[list[SearchResult]]
) -> list[SearchResult]:
    """Rank every candidate by its score against the ORIGINAL query.

    Cosine scores are only comparable within one query vector, so a fragment's 0.66 and the
    full question's 0.67 measure different things and must never be sorted against each
    other. `holistic` is the only scored-on-the-real-question list, so it orders the result;
    fragments contribute recall only. Candidates they surface from outside the holistic
    window have no comparable score, so they sort after everything that does, by best
    fragment rank — filling slots the holistic list could not.
    """
    scored_ids = {r.chunk_id for r in holistic}
    best_rank: dict[str, tuple[int, SearchResult]] = {}
    for results in per_fragment:
        for rank, result in enumerate(results):
            if result.chunk_id in scored_ids:
                continue
            existing = best_rank.get(result.chunk_id)
            if existing is None or rank < existing[0]:
                best_rank[result.chunk_id] = (rank, result)

    tail = [result for _, result in sorted(best_rank.values(), key=lambda pair: pair[0])]
    return list(holistic) + tail


def _rrf(ranked_lists: list[list[SearchResult]]) -> list[SearchResult]:
    """Reciprocal rank fusion: each chunk scores sum 1/(RRF_K + rank) over the lists it
    appears in, rank starting at 1. Scores from different lists (cosine, BM25) are on
    different scales and are never compared — only ranks are. The first list's instance of a
    chunk is kept, so pass the dense list first to keep its cosine `.score`. Ties keep
    first-seen order."""
    fused: dict[str, float] = {}
    first: dict[str, SearchResult] = {}
    for results in ranked_lists:
        for rank, result in enumerate(results, 1):
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            first.setdefault(result.chunk_id, result)
    order = sorted(fused, key=lambda cid: -fused[cid])
    return [first[cid] for cid in order]


def _hybrid_candidates(
    store: VectorStore, query: str, query_vector: list[float], filters: dict | None
) -> list[SearchResult] | None:
    """Dense and BM25 top-`HYBRID_FETCH`, fused by RRF. `None` when the collection has no
    BM25 index, and the caller falls back to dense only — a missing index must never take
    retrieval down.

    Chunks only BM25 found come back carrying a BM25 score. They are re-scored by cosine
    against the query vector, so every result's `.score` means the same thing:
    `MIN_SCORE_THRESHOLD` applies to them, and logs and eval output stay comparable with
    dense-only runs."""
    global _warned_no_sparse
    if not store.has_sparse():
        if not _warned_no_sparse:
            log.warning(
                "Hybrid search is on but the collection has no BM25 index — using vector "
                "search. Run: python -m scheduler.monthly_job --add-sparse"
            )
            _warned_no_sparse = True
        return None

    dense = store.search(query_vector=query_vector, top_k=HYBRID_FETCH, filters=filters)
    sparse = store.search_sparse(query, top_k=HYBRID_FETCH, filters=filters)
    fused = _rrf([dense, sparse])

    dense_ids = {r.chunk_id for r in dense}
    bm25_only = [r.chunk_id for r in fused if r.chunk_id not in dense_ids]
    cosine = store.score_by_ids(query_vector, bm25_only)
    fused = [
        r if r.chunk_id in dense_ids else replace(r, score=cosine.get(r.chunk_id, 0.0))
        for r in fused
    ]
    log.info(
        "Hybrid fusion",
        extra={
            "dense_count": len(dense),
            "sparse_count": len(sparse),
            "overlap": len(dense_ids & {r.chunk_id for r in sparse}),
            "bm25_only_in_pool": sum(1 for r in fused[:RERANK_POOL] if r.chunk_id not in dense_ids),
            "bm25_only_in_top_k": sum(
                1 for r in fused[:DEFAULT_TOP_K] if r.chunk_id not in dense_ids
            ),
        },
    )
    return fused


def _pool_depth(top_k: int, decomposed: bool) -> int:
    """How many candidates to fetch. Whoever consumes the pool sets the floor: the reranker
    needs `RERANK_POOL` to work well (measured far worse at 12 than at 48), and that need is
    unrelated to whether the query happened to decompose — so decomposition must not be able
    to shrink the pool out from under it. Without this floor, a query under
    `MIN_DECOMPOSITION_WORDS`, one the LLM declines to split, or a failed decomposition call
    would all silently fall back to a depth the reranker does badly at."""
    depth = top_k * (HOLISTIC_OVERFETCH if decomposed else 2)
    return max(depth, RERANK_POOL) if ENABLE_RERANK else depth


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[SearchResult]:
    """Top-k relevant chunks for `query`, ranked by score. Empty list if nothing clears
    `MIN_SCORE_THRESHOLD`."""
    if not query or not query.strip():
        log.error("Empty query rejected", extra={"error_type": "ValueError"})
        raise ValueError("Query must not be empty")

    if ENABLE_QUERY_REWRITING:  # pragma: no cover - optional, off
        log.warning("Query rewriting is not implemented — using the query as given")

    store = _get_store()
    _guard_model(store)

    subqueries = _decompose_query(query)
    query_vector = embed_query(query)
    fused = (
        _hybrid_candidates(store, query, query_vector, filters) if ENABLE_HYBRID_SEARCH else None
    )
    if fused is not None:
        # RRF picks the reranker's pool; its tail below RERANK_POOL is dropped, so every
        # chunk the reranker can return is one RRF chose.
        holistic = fused[:RERANK_POOL] if ENABLE_RERANK else fused
    else:
        holistic = store.search(
            query_vector=query_vector,
            top_k=_pool_depth(top_k, bool(subqueries)),
            filters=filters,
        )
    if subqueries:
        per_fragment = [
            store.search(query_vector=embed_query(sq), top_k=top_k * 2, filters=filters)
            for sq in subqueries
        ]
        raw = _merge_by_holistic_rank(holistic, per_fragment)
    else:
        raw = holistic
    if ENABLE_RERANK:
        raw = _rerank(query, raw)
    log.debug(
        "Results before filtering",
        extra={
            "raw_result_count": len(raw),
            "max_score": raw[0].score if raw else None,
            "subquery_count": len(subqueries),
            "reranked": ENABLE_RERANK,
            "hybrid": fused is not None,
        },
    )

    kept: list[SearchResult] = []
    per_article: dict[str, int] = {}
    for result in raw:  # holistic-ranked, then any fragment-only tail
        if result.score < MIN_SCORE_THRESHOLD:
            continue
        if per_article.get(result.article_url, 0) >= MAX_CHUNKS_PER_ARTICLE:
            continue
        per_article[result.article_url] = per_article.get(result.article_url, 0) + 1
        kept.append(result)
        if len(kept) == top_k:
            break

    if not kept:
        log.warning(
            "No results after filtering",
            extra={"query": query, "min_score_threshold": MIN_SCORE_THRESHOLD, "filters": filters},
        )
    else:
        log.info(
            "Retrieval complete",
            extra={
                "query": query,
                "result_count": len(kept),
                "top_score": kept[0].score,
                "bottom_score": kept[-1].score,
                "subquery_count": len(subqueries),
                "decomposition_triggered": bool(subqueries),
            },
        )
    return kept


def _guard_model(store: VectorStore) -> None:
    recorded = store.recorded_model()
    configured = get_embedding_provider().model_name
    if recorded is not None and recorded != configured:
        log.critical(
            "Model mismatch detected",
            extra={"stored_model": recorded, "configured_model": configured},
        )
        raise ModelMismatchError(
            f"The index was built with {recorded!r} but the configured embedding model is "
            f"{configured!r}. Re-index from scratch (monthly_job.py --reset)."
        )
