"""Embed a natural-language query and return the most relevant chunks.

Over-fetches `top_k * 2` from the vector store, then applies `MIN_SCORE_THRESHOLD` and the
`MAX_CHUNKS_PER_ARTICLE` cap before trimming to `top_k`. The store returns raw top-k; the
score and per-article policy live here.

When query decomposition fires, the original query is still searched (wider, at
`top_k * HOLISTIC_OVERFETCH`) and is what ranks the results — subquery searches only widen
the candidate pool. See `_merge_by_holistic_rank`.
"""

from __future__ import annotations

import re

from config import (
    DEFAULT_TOP_K,
    ENABLE_HYBRID_SEARCH,
    ENABLE_QUERY_DECOMPOSITION,
    ENABLE_QUERY_REWRITING,
    HOLISTIC_OVERFETCH,
    MAX_CHUNKS_PER_ARTICLE,
    MAX_SUBQUERIES,
    MIN_DECOMPOSITION_WORDS,
    MIN_SCORE_THRESHOLD,
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


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _reset_store_for_tests() -> None:
    global _store
    _store = None


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

    if ENABLE_QUERY_REWRITING or ENABLE_HYBRID_SEARCH:  # pragma: no cover - optional, off
        log.warning("Query rewriting / hybrid search are not implemented — using vector search")

    store = _get_store()
    _guard_model(store)

    subqueries = _decompose_query(query)
    holistic = store.search(
        query_vector=embed_query(query),
        top_k=top_k * (HOLISTIC_OVERFETCH if subqueries else 2),
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
    log.debug(
        "Results before filtering",
        extra={
            "raw_result_count": len(raw),
            "max_score": raw[0].score if raw else None,
            "subquery_count": len(subqueries),
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
