"""Embed a natural-language query and return the most relevant chunks.

Over-fetches `top_k * 2` from the vector store, then applies `MIN_SCORE_THRESHOLD` and the
`MAX_CHUNKS_PER_ARTICLE` cap before trimming to `top_k`. The store returns raw top-k; the
score and per-article policy live here.
"""

from __future__ import annotations

from config import (
    DEFAULT_TOP_K,
    ENABLE_HYBRID_SEARCH,
    ENABLE_QUERY_REWRITING,
    MAX_CHUNKS_PER_ARTICLE,
    MIN_SCORE_THRESHOLD,
)
from ingestion.embedder import embed_query
from llm_provider import get_embedding_provider
from logger import get_logger
from models import ModelMismatchError, SearchResult
from storage.vector_store import VectorStore

log = get_logger(__name__)

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _reset_store_for_tests() -> None:
    global _store
    _store = None


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

    query_vector = embed_query(query)
    raw = store.search(query_vector=query_vector, top_k=top_k * 2, filters=filters)
    log.debug(
        "Results before filtering",
        extra={"raw_result_count": len(raw), "max_score": raw[0].score if raw else None},
    )

    kept: list[SearchResult] = []
    per_article: dict[str, int] = {}
    for result in raw:  # already score-descending
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
