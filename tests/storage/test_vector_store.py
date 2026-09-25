"""Tests for storage/vector_store.py — in-memory Qdrant, small vectors.

`search` must return raw top-k (no score / per-article filtering — that is the
retriever's job); the model sentinel must stay out of `count()` and `search`; and
`drop_collection` must leave an immediately usable empty collection.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import storage.vector_store as vs_module
from models import Chunk, EmbeddedChunk, ModelMismatchError

_DIM = 4


@pytest.fixture(autouse=True)
def _in_memory(monkeypatch):
    monkeypatch.setattr(vs_module, "QDRANT_IN_MEMORY", True)
    monkeypatch.setattr(vs_module, "EMBEDDING_DIM", _DIM)


@pytest.fixture
def store():
    return vs_module.VectorStore()


def _chunk(
    chunk_id: str,
    *,
    url: str = "https://example.com/a",
    ctype: str = "body",
    tags: list[str] | None = None,
    text: str = "some chunk text",
    published: datetime | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        article_url=url,
        article_title="Article A",
        published_at=published,
        tags=tags or ["macro"],
        text=text,
        content_type=ctype,
        chunk_index=0,
        total_chunks=1,
        word_count=len(text.split()),
    )


def _emb(chunk: Chunk, vector: list[float], model: str = "bge-large") -> EmbeddedChunk:
    return EmbeddedChunk(chunk=chunk, embedding=vector, model_name=model)


# ── upsert ──────────────────────────────────────────────────────────────────


def test_upsert_returns_input_length_and_count_excludes_sentinel(store):
    n = store.upsert(
        [
            _emb(_chunk("a_b_0"), [1, 0, 0, 0]),
            _emb(_chunk("a_b_1"), [0, 1, 0, 0]),
        ]
    )
    assert n == 2
    assert store.count() == 2  # the model sentinel is NOT counted


def test_empty_upsert_is_a_noop(store):
    assert store.upsert([]) == 0
    assert store.count() == 0
    assert store.recorded_model() is None


def test_duplicate_chunk_id_overwrites(store):
    store.upsert([_emb(_chunk("a_b_0", text="v1"), [1, 0, 0, 0])])
    store.upsert([_emb(_chunk("a_b_0", text="v2"), [1, 0, 0, 0])])
    assert store.count() == 1
    hit = store.search([1, 0, 0, 0], top_k=1)[0]
    assert hit.text == "v2"


def test_model_recorded_on_first_upsert_then_enforced(store):
    store.upsert([_emb(_chunk("a_b_0"), [1, 0, 0, 0], model="bge-large")])
    assert store.recorded_model() == "bge-large"
    with pytest.raises(ModelMismatchError, match="bge-large"):
        store.upsert([_emb(_chunk("a_b_1"), [0, 1, 0, 0], model="nomic-embed")])


# ── search ──────────────────────────────────────────────────────────────────


class TestSearch:
    def test_orders_by_score_and_carries_chunk_id(self, store):
        store.upsert(
            [
                _emb(_chunk("near"), [1.0, 0.0, 0.0, 0.0]),
                _emb(_chunk("far"), [0.0, 0.0, 0.0, 1.0]),
            ]
        )
        results = store.search([1.0, 0.05, 0.0, 0.0], top_k=2)
        assert [r.chunk_id for r in results] == ["near", "far"]
        assert results[0].score >= results[1].score

    def test_returns_raw_top_k_no_threshold(self, store):
        """A weak match is still returned — the retriever applies MIN_SCORE_THRESHOLD."""
        store.upsert([_emb(_chunk("weak"), [0.0, 0.0, 0.0, 1.0])])
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=6)
        assert len(results) == 1  # not dropped for a low score

    def test_top_k_is_a_hard_limit(self, store):
        store.upsert([_emb(_chunk(f"c{i}"), [1.0, i * 0.01, 0.0, 0.0]) for i in range(10)])
        assert len(store.search([1.0, 0.0, 0.0, 0.0], top_k=3)) == 3

    def test_sentinel_never_appears(self, store):
        store.upsert([_emb(_chunk("real"), [0.0, 0.0, 0.0, 0.0])])  # zero vector, like sentinel
        results = store.search([0.0, 0.0, 0.0, 0.0], top_k=6)
        assert all(r.chunk_id != "" for r in results)
        assert {r.chunk_id for r in results} == {"real"}

    def test_tag_filter(self, store):
        store.upsert(
            [
                _emb(_chunk("m", tags=["macro"]), [1, 0, 0, 0]),
                _emb(_chunk("e", tags=["equities"]), [1, 0, 0, 0]),
            ]
        )
        results = store.search([1, 0, 0, 0], top_k=6, filters={"tags": ["macro"]})
        assert {r.chunk_id for r in results} == {"m"}

    def test_content_type_filter(self, store):
        store.upsert(
            [
                _emb(_chunk("b", ctype="body"), [1, 0, 0, 0]),
                _emb(_chunk("t", ctype="table"), [1, 0, 0, 0]),
            ]
        )
        results = store.search([1, 0, 0, 0], top_k=6, filters={"content_type": "table"})
        assert {r.chunk_id for r in results} == {"t"}

    def test_date_after_filter(self, store):
        old = datetime(2020, 1, 1, tzinfo=UTC)
        new = datetime(2024, 1, 1, tzinfo=UTC)
        store.upsert(
            [
                _emb(_chunk("old", published=old), [1, 0, 0, 0]),
                _emb(_chunk("new", published=new), [1, 0, 0, 0]),
            ]
        )
        results = store.search(
            [1, 0, 0, 0],
            top_k=6,
            filters={"date_after": datetime(2022, 1, 1, tzinfo=UTC)},
        )
        assert {r.chunk_id for r in results} == {"new"}

    def test_published_at_round_trips_to_datetime(self, store):
        when = datetime(2021, 6, 27, tzinfo=UTC)
        store.upsert([_emb(_chunk("a", published=when), [1, 0, 0, 0])])
        hit = store.search([1, 0, 0, 0], top_k=1)[0]
        assert hit.published_at == when

    def test_missing_published_at_is_none(self, store):
        store.upsert([_emb(_chunk("a", published=None), [1, 0, 0, 0])])
        assert store.search([1, 0, 0, 0], top_k=1)[0].published_at is None

    def test_date_before_filter(self, store):
        store.upsert(
            [
                _emb(_chunk("old", published=datetime(2020, 1, 1, tzinfo=UTC)), [1, 0, 0, 0]),
                _emb(_chunk("new", published=datetime(2024, 1, 1, tzinfo=UTC)), [1, 0, 0, 0]),
            ]
        )
        results = store.search(
            [1, 0, 0, 0],
            top_k=6,
            filters={"date_before": datetime(2022, 1, 1, tzinfo=UTC)},
        )
        assert {r.chunk_id for r in results} == {"old"}

    def test_empty_collection_returns_no_results(self, store):
        assert store.search([1, 0, 0, 0], top_k=6) == []


# ── delete_by_url ───────────────────────────────────────────────────────────


def test_delete_by_url_removes_only_that_url(store):
    store.upsert(
        [
            _emb(_chunk("a0", url="https://example.com/a"), [1, 0, 0, 0]),
            _emb(_chunk("a1", url="https://example.com/a"), [0, 1, 0, 0]),
            _emb(_chunk("b0", url="https://example.com/b"), [0, 0, 1, 0]),
        ]
    )
    removed = store.delete_by_url("https://example.com/a")
    assert removed == 2
    assert store.count() == 1
    remaining = store.search([0, 0, 1, 0], top_k=6)
    assert {r.article_url for r in remaining} == {"https://example.com/b"}


def test_delete_by_url_no_match_returns_zero(store):
    store.upsert([_emb(_chunk("a0"), [1, 0, 0, 0])])
    assert store.delete_by_url("https://example.com/missing") == 0


# ── drop_collection ─────────────────────────────────────────────────────────


def test_drop_collection_leaves_usable_empty_collection(store):
    store.upsert([_emb(_chunk("a0"), [1, 0, 0, 0], model="bge-large")])
    store.drop_collection()
    assert store.count() == 0
    # immediately usable without re-instantiating VectorStore
    assert store.upsert([_emb(_chunk("b0"), [0, 1, 0, 0], model="bge-large")]) == 1


def test_drop_collection_clears_the_model_so_a_new_model_does_not_raise(store):
    store.upsert([_emb(_chunk("a0"), [1, 0, 0, 0], model="bge-large")])
    store.drop_collection()
    assert store.recorded_model() is None
    store.upsert([_emb(_chunk("b0"), [0, 1, 0, 0], model="nomic-embed")])  # no ModelMismatchError
    assert store.recorded_model() == "nomic-embed"


# ── init-time dim guard ─────────────────────────────────────────────────────


def test_reconnecting_with_a_different_dim_raises(store, monkeypatch):
    store.upsert([_emb(_chunk("a0"), [1, 0, 0, 0])])
    # a second VectorStore against the same client would need a real server; instead
    # re-run _init_collection with EMBEDDING_DIM changed under it.
    monkeypatch.setattr(vs_module, "EMBEDDING_DIM", 8)
    with pytest.raises(ModelMismatchError, match="EMBEDDING_DIM"):
        store._init_collection()


def test_reinit_with_matching_dim_is_a_noop(store):
    store.upsert([_emb(_chunk("a0"), [1, 0, 0, 0])])
    store._init_collection()  # collection exists, dim matches → no raise, no change
    assert store.count() == 1


def test_stats_reports_points_and_model(store):
    store.upsert([_emb(_chunk("a0"), [1, 0, 0, 0], model="bge-large")])
    s = store.stats()
    assert s["points"] == 1
    assert s["model_name"] == "bge-large"
    assert s["dim"] == _DIM


# ── BM25 sparse vectors (hybrid search) ─────────────────────────────────────


def _make_pre_hybrid(store) -> None:
    """Recreate the collection the way it was before hybrid search: dense only."""
    from qdrant_client import models

    store.client.delete_collection(vs_module.COLLECTION_NAME)
    store.client.create_collection(
        vs_module.COLLECTION_NAME,
        vectors_config=models.VectorParams(size=_DIM, distance=models.Distance.COSINE),
    )


def _seed_three(store) -> None:
    rows = [
        ("gold", "Sold Kirkland Lake Gold shares", [1, 0, 0, 0]),
        ("port", "No current portfolio changes", [0.9, 0.1, 0, 0]),
        ("btc", "Bitcoin on-chain indicators", [0, 0, 1, 0]),
    ]
    store.upsert([_emb(_chunk(c, url=f"https://e.com/{c}", text=t), v) for c, t, v in rows])


def test_new_collection_has_the_bm25_sparse_vector(store):
    assert store.has_sparse()


def test_search_sparse_ranks_the_exact_term_chunk_first(store):
    _seed_three(store)
    results = store.search_sparse("Kirkland Lake Gold", top_k=3)
    assert results[0].chunk_id == "gold"
    assert all(r.chunk_id for r in results)  # the sentinel (no chunk_id) never appears


def test_search_sparse_applies_filters(store):
    _seed_three(store)
    chart = _chunk("gold2", url="https://e.com/g2", ctype="image", text="Kirkland Lake Gold chart")
    store.upsert([_emb(chart, [1, 0, 0, 0])])
    results = store.search_sparse("Kirkland Lake Gold", top_k=5, filters={"content_type": "image"})
    assert [r.chunk_id for r in results] == ["gold2"]


def test_score_by_ids_returns_cosine_for_just_those_chunks(store):
    _seed_three(store)
    scores = store.score_by_ids([1, 0, 0, 0], ["gold", "btc"])
    assert set(scores) == {"gold", "btc"}
    assert scores["gold"] == pytest.approx(1.0)
    assert scores["btc"] == pytest.approx(0.0, abs=1e-6)
    assert store.score_by_ids([1, 0, 0, 0], []) == {}


def test_upsert_into_a_pre_hybrid_collection_stays_dense_only(store):
    _make_pre_hybrid(store)
    store.upsert([_emb(_chunk("a0"), [1, 0, 0, 0])])
    assert not store.has_sparse()
    assert store.count() == 1


def test_add_sparse_migrates_a_pre_hybrid_collection_without_changing_dense_search(store):
    _make_pre_hybrid(store)
    _seed_three(store)
    before = [(r.chunk_id, r.score, r.text) for r in store.search([1, 0, 0, 0], top_k=3)]

    result = store.add_sparse_vectors()

    assert result["migrated"] is True
    assert result["sparse_vectors_written"] == 3
    assert result["points"] == 4  # three chunks + the model sentinel
    assert store.has_sparse()
    assert store.count() == 3
    assert store.recorded_model() == "bge-large"
    after = [(r.chunk_id, r.score, r.text) for r in store.search([1, 0, 0, 0], top_k=3)]
    assert after == before
    assert store.search_sparse("Kirkland Lake Gold", top_k=1)[0].chunk_id == "gold"


def test_add_sparse_is_idempotent(store):
    _seed_three(store)
    result = store.add_sparse_vectors()
    assert result["migrated"] is False
    assert store.count() == 3
    assert store.search_sparse("Bitcoin", top_k=1)[0].chunk_id == "btc"


def test_add_sparse_reports_the_measured_average_length(store):
    _seed_three(store)  # 5, 4 and 4 terms under the fake encoder ("on-chain" is two)
    assert store.add_sparse_vectors()["measured_avg_len"] == 4.3
