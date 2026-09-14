"""Tests for query/retriever.py — score threshold, per-article cap, trim to top_k, the
empty-query guard, and the embedding-model mismatch guard.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import query.retriever as rt
from llm_provider import ProviderConnectionError, TextResponse
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
    """`results` is either a fixed `list[SearchResult]` (every search call gets the same
    answer, the shape all pre-decomposition tests use) or a `dict[str, list[SearchResult]]`
    keyed by query text, for tests that script a different result set per subquery — relies
    on `wire` patching `embed_query` to the identity function so `query_vector` IS the query
    text.
    """

    def __init__(self, results, model="mock-embed"):
        self._results = results
        self._model = model
        self.search_args = None
        self.search_calls: list[dict] = []

    def search(self, query_vector, top_k, filters):
        self.search_args = {"top_k": top_k, "filters": filters}
        self.search_calls.append({"query_vector": query_vector, "top_k": top_k, "filters": filters})
        if isinstance(self._results, dict):
            return list(self._results.get(query_vector, []))
        return list(self._results)

    def recorded_model(self):
        return self._model


class FakeTextProvider:
    """Stands in for `get_text_provider()` in query-decomposition tests. Returns `content`
    verbatim as the decomposition response, or raises `error` if given."""

    def __init__(self, content: str = "", error: Exception | None = None):
        self._content = content
        self._error = error
        self.calls = 0

    def complete(self, messages, max_tokens=1024, temperature=0.0):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return TextResponse(
            content=self._content, model="mock-llm", input_tokens=1, output_tokens=1
        )


@pytest.fixture
def wire(monkeypatch):
    """Patch embed_query (identity — the "vector" IS the query text, so FakeStore can
    dispatch per-subquery) and the embedding provider; return a helper to install a store."""
    monkeypatch.setattr(rt, "embed_query", lambda q: q)
    monkeypatch.setattr(
        rt, "get_embedding_provider", lambda: type("P", (), {"model_name": "mock-embed"})()
    )

    def install(results, model="mock-embed"):
        store = FakeStore(results, model)
        monkeypatch.setattr(rt, "_get_store", lambda: store)
        return store

    return install


@pytest.fixture
def with_decomposition(monkeypatch):
    """Enable decomposition and install a `FakeTextProvider`; return a helper to configure it."""
    monkeypatch.setattr(rt, "ENABLE_QUERY_DECOMPOSITION", True)

    def install(content: str = "", error: Exception | None = None):
        provider = FakeTextProvider(content=content, error=error)
        monkeypatch.setattr(rt, "get_text_provider", lambda: provider)
        return provider

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


# ── Query decomposition + fusion ─────────────────────────────────────────────

_LONG_QUERY = "Compare the author's view on inflation and on interest rates"


# Layer 1 — `_fuse`, pure function, no mocking


def test_fuse_disjoint_articles_merge_all():
    a = [_result("a1", 0.9, url="https://example.com/a")]
    b = [_result("b1", 0.8, url="https://example.com/b")]
    out = rt._fuse([a, b])
    assert [r.chunk_id for r in out] == ["a1", "b1"]


def test_fuse_dedupes_same_chunk_id_keeping_max_score():
    a = [_result("x", 0.4, url="https://example.com/a")]
    b = [_result("x", 0.9, url="https://example.com/a")]
    out = rt._fuse([a, b])
    assert len(out) == 1
    assert out[0].score == 0.9


def test_fuse_handles_one_empty_subquery_list():
    a: list[SearchResult] = []
    b = [_result("b1", 0.7, url="https://example.com/b")]
    out = rt._fuse([a, b])
    assert [r.chunk_id for r in out] == ["b1"]


def test_fuse_all_same_article_collapses_to_single_query_shape():
    same = [_result(f"c{i}", 0.9 - i * 0.01, url="https://example.com/a") for i in range(3)]
    out = rt._fuse([same, same])
    assert [r.chunk_id for r in out] == [r.chunk_id for r in same]


# Layer 1b — `_decompose_query` / `_parse_subqueries`, mocked LLM call


def test_decompose_single_topic_mock_one_search_call(wire, with_decomposition):
    with_decomposition(content=_LONG_QUERY)  # LLM judges it single-topic: echoes it back
    store = wire([_result("a", 0.9)])
    rt.retrieve(_LONG_QUERY)
    assert len(store.search_calls) == 1


def test_decompose_two_topic_mock_two_search_calls(wire, with_decomposition):
    sub_a = "the author's view on inflation"
    sub_b = "the author's view on interest rates"
    with_decomposition(content=f"{sub_a}\n{sub_b}")
    store = wire({sub_a: [_result("a1", 0.9, url="https://example.com/a")],
                  sub_b: [_result("b1", 0.8, url="https://example.com/b")]})
    rt.retrieve(_LONG_QUERY)
    assert len(store.search_calls) == 2


def test_decompose_multi_entity_mock_three_search_calls(wire, with_decomposition):
    subs = [
        "the author's view on inflation",
        "the author's view on rates",
        "the author's view on gold",
    ]
    with_decomposition(content="\n".join(subs))
    store = wire(
        {sq: [_result(sq, 0.9, url=f"https://example.com/{i}")] for i, sq in enumerate(subs)}
    )
    rt.retrieve(_LONG_QUERY)
    assert len(store.search_calls) == 3


def test_decompose_malformed_output_falls_back_to_original_query(wire, with_decomposition):
    with_decomposition(content="")  # no usable lines
    store = wire([_result("a", 0.9)])
    out = rt.retrieve(_LONG_QUERY)
    assert len(store.search_calls) == 1
    assert store.search_calls[0]["query_vector"] == _LONG_QUERY
    assert out[0].chunk_id == "a"


def test_decompose_single_line_identical_to_original_falls_back(wire, with_decomposition):
    with_decomposition(content=f"  {_LONG_QUERY}  ")  # whitespace-different, same query
    store = wire([_result("a", 0.9)])
    rt.retrieve(_LONG_QUERY)
    assert len(store.search_calls) == 1
    assert store.search_calls[0]["query_vector"] == _LONG_QUERY


def test_decompose_provider_connection_error_falls_back_to_original_query(wire, with_decomposition):
    with_decomposition(error=ProviderConnectionError("unreachable"))
    store = wire([_result("a", 0.9)])
    out = rt.retrieve(_LONG_QUERY)
    assert len(store.search_calls) == 1
    assert out[0].chunk_id == "a"


def test_decomposition_skipped_below_min_words(wire, with_decomposition):
    provider = with_decomposition(content="irrelevant")
    store = wire([_result("a", 0.9)])
    rt.retrieve("What is CPI?")  # 3 words, below MIN_DECOMPOSITION_WORDS
    assert provider.calls == 0
    assert len(store.search_calls) == 1


def test_decomposition_disabled_by_default_flag_off(wire, monkeypatch):
    provider = FakeTextProvider(content="irrelevant")
    monkeypatch.setattr(rt, "get_text_provider", lambda: provider)
    store = wire([_result("a", 0.9)])
    rt.retrieve(_LONG_QUERY)  # ENABLE_QUERY_DECOMPOSITION defaults False — wire doesn't touch it
    assert provider.calls == 0
    assert len(store.search_calls) == 1


def test_decompose_caps_at_max_subqueries(wire, with_decomposition, monkeypatch):
    monkeypatch.setattr(rt, "MAX_SUBQUERIES", 2)
    subs = ["topic one about inflation", "topic two about rates", "topic three about gold"]
    with_decomposition(content="\n".join(subs))
    store = wire(
        {sq: [_result(sq, 0.9, url=f"https://example.com/{i}")] for i, sq in enumerate(subs)}
    )
    rt.retrieve(_LONG_QUERY)
    assert len(store.search_calls) == 2


def test_decompose_dedupes_case_insensitive_duplicate_lines(wire, with_decomposition):
    content = "Author's view on inflation\nauthor's   view on inflation\nAuthor's view on rates"
    with_decomposition(content=content)
    store = wire(
        {
            "Author's view on inflation": [_result("a", 0.9, url="https://example.com/a")],
            "Author's view on rates": [_result("b", 0.8, url="https://example.com/b")],
        }
    )
    rt.retrieve(_LONG_QUERY)
    # 3 raw lines, first two are near-duplicates -> 2 distinct subqueries, not 3
    assert len(store.search_calls) == 2


# Layer 2 — component: `retrieve()` end-to-end with the decomposition + fusion wired together


def test_decomposition_end_to_end_fuses_across_subqueries(wire, with_decomposition):
    sub_a = "the author's view on inflation"
    sub_b = "the author's view on interest rates"
    with_decomposition(content=f"{sub_a}\n{sub_b}")
    wire(
        {
            sub_a: [_result("a1", 0.9, url="https://example.com/article-a")],
            sub_b: [_result("b1", 0.85, url="https://example.com/article-b")],
        }
    )
    out = rt.retrieve(_LONG_QUERY, top_k=6)
    assert {r.article_url for r in out} == {
        "https://example.com/article-a",
        "https://example.com/article-b",
    }


def test_decomposition_respects_min_score_threshold_on_fused_set(wire, with_decomposition):
    sub_a = "the author's view on inflation"
    sub_b = "the author's view on interest rates"
    with_decomposition(content=f"{sub_a}\n{sub_b}")
    wire(
        {
            sub_a: [_result("weak", 0.2, url="https://example.com/weak")],
            sub_b: [_result("weak", 0.25, url="https://example.com/weak")],
        }
    )
    out = rt.retrieve(_LONG_QUERY, top_k=6)
    assert out == []  # max fused score (0.25) still below MIN_SCORE_THRESHOLD (0.35)


def test_decomposition_respects_max_chunks_per_article_across_subqueries(wire, with_decomposition):
    sub_a = "the author's view on inflation"
    sub_b = "the author's view on interest rates"
    with_decomposition(content=f"{sub_a}\n{sub_b}")
    dominant = "https://example.com/dominant"
    wire(
        {
            sub_a: [_result(f"a{i}", 0.9 - i * 0.01, url=dominant) for i in range(3)],
            sub_b: [_result(f"b{i}", 0.87 - i * 0.01, url=dominant) for i in range(3)],
        }
    )
    out = rt.retrieve(_LONG_QUERY, top_k=6)
    # 6 distinct chunks from one article across the two subqueries; cap (3) must apply to
    # the FUSED set, not per subquery, or the single-article-dominance bug sneaks back in.
    assert len(out) == 3
    assert [r.chunk_id for r in out] == ["a0", "a1", "a2"]  # the 3 highest-scoring


def test_decomposition_output_is_score_descending_and_trimmed_to_top_k(wire, with_decomposition):
    sub_a = "the author's view on inflation"
    sub_b = "the author's view on interest rates"
    with_decomposition(content=f"{sub_a}\n{sub_b}")
    wire(
        {
            sub_a: [
                _result(f"a{i}", 0.9 - i * 0.05, url=f"https://example.com/a{i}") for i in range(4)
            ],
            sub_b: [
                _result(f"b{i}", 0.85 - i * 0.05, url=f"https://example.com/b{i}")
                for i in range(4)
            ],
        }
    )
    out = rt.retrieve(_LONG_QUERY, top_k=3)
    assert len(out) == 3
    assert [r.score for r in out] == sorted((r.score for r in out), reverse=True)


def test_decomposition_off_is_identical_to_current_behavior(wire, monkeypatch):
    provider = FakeTextProvider(content="irrelevant")
    monkeypatch.setattr(rt, "get_text_provider", lambda: provider)
    store = wire([_result("a", 0.9)])
    out = rt.retrieve(_LONG_QUERY, top_k=6)
    assert len(store.search_calls) == 1
    assert store.search_calls[0]["query_vector"] == _LONG_QUERY
    assert provider.calls == 0
    assert out[0].chunk_id == "a"
