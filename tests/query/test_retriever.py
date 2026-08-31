"""Tests for query/retriever.py — score threshold, per-article cap, trim to top_k, the
empty-query guard, and the embedding-model mismatch guard.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import query.retriever as rt
from models import ModelMismatchError, SearchResult


def _result(chunk_id, score, url="https://example.com/a") -> SearchResult:
    return SearchResult(
        score=score,
        text=f"text {chunk_id}",
        chunk_id=chunk_id,
        article_url=url,
        article_title="A",
        published_at=datetime(2021, 1, 1, tzinfo=UTC),
        tags=[],
        content_type="body",
        chunk_index=0,
    )


class FakeStore:
    def __init__(self, results, model="mock-embed"):
        self._results = results
        self._model = model
        self.search_args = None

    def search(self, query_vector, top_k, filters):
        self.search_args = {"top_k": top_k, "filters": filters}
        return list(self._results)

    def recorded_model(self):
        return self._model


@pytest.fixture
def wire(monkeypatch):
    """Patch embed_query and the embedding provider; return a helper to install a store."""
    monkeypatch.setattr(rt, "embed_query", lambda q: [0.1] * 8)
    monkeypatch.setattr(
        rt, "get_embedding_provider", lambda: type("P", (), {"model_name": "mock-embed"})()
    )

    def install(results, model="mock-embed"):
        store = FakeStore(results, model)
        monkeypatch.setattr(rt, "_get_store", lambda: store)
        return store

    return install


def test_empty_query_raises(wire):
    wire([])
    with pytest.raises(ValueError, match="empty"):
        rt.retrieve("   ")


def test_over_fetches_double_top_k(wire):
    store = wire([_result("a", 0.9)])
    rt.retrieve("q", top_k=6)
    assert store.search_args["top_k"] == 12


def test_results_are_score_descending_and_trimmed(wire):
    wire([_result(f"c{i}", 0.9 - i * 0.05, url=f"https://example.com/{i}") for i in range(10)])
    out = rt.retrieve("q", top_k=3)
    assert len(out) == 3
    assert [r.score for r in out] == sorted((r.score for r in out), reverse=True)


def test_below_threshold_is_excluded(wire, monkeypatch):
    monkeypatch.setattr(rt, "MIN_SCORE_THRESHOLD", 0.5)
    wire(
        [
            _result("hi", 0.8, url="https://example.com/1"),
            _result("lo", 0.2, url="https://example.com/2"),
        ]
    )
    out = rt.retrieve("q")
    assert [r.chunk_id for r in out] == ["hi"]


def test_all_below_threshold_returns_empty(wire, monkeypatch):
    monkeypatch.setattr(rt, "MIN_SCORE_THRESHOLD", 0.9)
    wire([_result("a", 0.4), _result("b", 0.3)])
    assert rt.retrieve("q") == []


def test_per_article_cap(wire, monkeypatch):
    monkeypatch.setattr(rt, "MAX_CHUNKS_PER_ARTICLE", 2)
    wire([_result(f"a{i}", 0.9 - i * 0.01, url="https://example.com/same") for i in range(6)])
    out = rt.retrieve("q", top_k=6)
    assert len(out) == 2  # capped despite 6 clearing the threshold


def test_empty_store_returns_empty(wire):
    wire([])
    assert rt.retrieve("q") == []


def test_model_mismatch_raises(wire):
    wire([_result("a", 0.9)], model="some-other-model")
    with pytest.raises(ModelMismatchError, match="some-other-model"):
        rt.retrieve("q")


def test_unrecorded_model_does_not_raise(wire):
    wire([_result("a", 0.9)], model=None)  # nothing upserted yet
    assert rt.retrieve("q")[0].chunk_id == "a"


def test_filters_are_passed_through(wire):
    store = wire([_result("a", 0.9)])
    rt.retrieve("q", filters={"content_type": "table"})
    assert store.search_args["filters"] == {"content_type": "table"}


def test_integration_seeded_store_returns_the_matching_chunk(monkeypatch):
    """The spec's integration check: seed an in-memory Qdrant, embed a query with the same
    deterministic mock, and get the right chunk back."""
    import storage.vector_store as vs
    from models import Chunk, EmbeddedChunk

    monkeypatch.setattr(vs, "QDRANT_IN_MEMORY", True)
    monkeypatch.setattr(vs, "EMBEDDING_DIM", 8)
    rt._reset_store_for_tests()
    monkeypatch.setattr(rt, "_reset_store_for_tests", rt._reset_store_for_tests)

    def vec(text: str) -> list[float]:
        import hashlib

        d = hashlib.sha256(text.encode()).digest()
        return [d[i] / 255 for i in range(8)]

    monkeypatch.setattr(rt, "embed_query", lambda q: vec(q))
    monkeypatch.setattr(
        rt, "get_embedding_provider", lambda: type("P", (), {"model_name": "mock-embed"})()
    )

    store = vs.VectorStore()
    monkeypatch.setattr(rt, "_get_store", lambda: store)

    def _chunk(cid, text):
        return Chunk(
            chunk_id=cid,
            article_url=f"https://example.com/{cid}",
            article_title=cid,
            published_at=None,
            tags=[],
            text=text,
            content_type="body",
            chunk_index=0,
            total_chunks=1,
            word_count=3,
        )

    store.upsert(
        [
            EmbeddedChunk(
                _chunk("m2", "money supply growth"), vec("money supply growth"), "mock-embed"
            ),
            EmbeddedChunk(
                _chunk("gold", "gold silver ratio"), vec("gold silver ratio"), "mock-embed"
            ),
        ]
    )

    out = rt.retrieve("money supply growth", top_k=1)
    assert out and out[0].chunk_id == "m2"
    rt._reset_store_for_tests()
