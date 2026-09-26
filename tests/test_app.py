"""Tests for app.py — the CLI surface. The Streamlit UI is exercised via `streamlit run`
and is not unit-tested here beyond the shared helpers (`run_query`, `is_web_url`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

import app
from models import Answer, Source


def _answer(response="The money supply grew sharply [1].", sources=None) -> Answer:
    return Answer(
        query="q",
        response=response,
        sources=sources
        if sources is not None
        else [
            Source(
                index=1,
                title="Macro Update",
                url="https://example.com/premium-2021-6-27",
                published_at=datetime(2021, 6, 27, tzinfo=UTC),
                score=0.82,
            )
        ],
        model="mock-text",
        input_tokens=100,
        output_tokens=40,
    )


@pytest.fixture
def stub_query(monkeypatch):
    holder = {"answer": _answer(), "calls": []}

    def fake(query, top_k, filters):
        holder["calls"].append({"query": query, "top_k": top_k, "filters": filters})
        return holder["answer"]

    monkeypatch.setattr(app, "run_query", fake)
    return holder


# ── CLI: query ─────────────────────────────────────────────────────────────


def test_one_shot_query_prints_answer_and_sources(stub_query, capsys):
    assert app.main(["What happened to M2?"]) == 0
    out = capsys.readouterr().out
    assert "The money supply grew sharply [1]." in out
    assert "Macro Update" in out and "score 0.82" in out


def test_json_output_round_trips_to_answer(stub_query, capsys):
    assert app.main(["q", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "mock-text"
    assert payload["sources"][0]["url"] == "https://example.com/premium-2021-6-27"
    assert payload["sources"][0]["published_at"].startswith("2021-06-27")


def test_no_citations_flag_hides_the_source_list(stub_query, capsys):
    app.main(["q", "--no-citations"])
    out = capsys.readouterr().out
    assert "Sources" not in out


def test_empty_query_message(capsys):
    assert app.main(["   "]) == 2
    assert "Please enter a question." in capsys.readouterr().err


def test_filters_are_parsed_and_forwarded(stub_query):
    app.main(["q", "--tags", "macro, rates", "--date-after", "2022-01-01", "--top-k", "3"])
    call = stub_query["calls"][0]
    assert call["top_k"] == 3
    assert call["filters"]["tags"] == ["macro", "rates"]
    assert call["filters"]["date_after"] == datetime(2022, 1, 1)


def test_unknown_tag_returns_a_graceful_empty_answer(stub_query, capsys):
    stub_query["answer"] = _answer(
        response="I couldn't find relevant content in the knowledge base for this question.",
        sources=[],
    )
    assert app.main(["q", "--tags", "no-such-tag"]) == 0
    assert "couldn't find" in capsys.readouterr().out


def test_local_id_source_is_rendered_as_plain_text(stub_query, capsys):
    stub_query["answer"] = _answer(
        sources=[
            Source(
                index=1,
                title="Untitled Export",
                url="local:2021-05-16",
                published_at=None,
                score=0.7,
            )
        ]
    )
    app.main(["q"])
    out = capsys.readouterr().out
    assert "local:2021-05-16" in out
    assert "fix frontmatter" in out


# ── CLI: other modes ───────────────────────────────────────────────────────


def test_check_email_needs_phase2_configured(monkeypatch, capsys):
    monkeypatch.setattr("config.phase2_configured", lambda: False)
    assert app.main(["--check-email"]) == 0
    assert "Phase 2 is not configured" in capsys.readouterr().err


def test_check_email_runs_when_configured(monkeypatch, capsys):
    monkeypatch.setattr("config.phase2_configured", lambda: True)

    async def fake_run(dry_run=False):
        return {"new": 1, "updated": 0, "skipped": 0, "failed": 0}

    monkeypatch.setattr("scheduler.monthly_job.run_email_triggered", fake_run)
    assert app.main(["--check-email"]) == 0
    assert "'new': 1" in capsys.readouterr().out


def test_stats_mode(monkeypatch, capsys):
    class _DB:
        def __init__(self, *a):
            pass

        def get_stats(self):
            return {
                "articles": {"total": 5, "active": 4, "archived": 1},
                "by_source": {"corpus": 4, "web": 1},
                "pipeline_versions": {1: 4},
                "last_run": {"corpus": {"started_at": "2026-01-01", "error_code": None}},
            }

        def close(self):
            pass

    monkeypatch.setattr("storage.metadata_db.MetadataDB", _DB)
    assert app.main(["--stats"]) == 0
    out = capsys.readouterr().out
    assert "5 total" in out and "corpus 4" in out


def test_sync_corpus_mode(monkeypatch, capsys):
    async def fake_sync(dry_run=False):
        return {"new": 2, "updated": 0, "skipped": 1, "failed": 0}

    monkeypatch.setattr("scheduler.monthly_job.run_corpus_sync", fake_sync)
    assert app.main(["--sync-corpus"]) == 0
    assert "'new': 2" in capsys.readouterr().out


# ── error translation ──────────────────────────────────────────────────────


def test_vector_store_down_message(monkeypatch, capsys):
    from storage.vector_store import VectorStoreConnectionError

    def boom(*a, **k):
        raise VectorStoreConnectionError("localhost:6333")

    monkeypatch.setattr(app, "run_query", boom)
    assert app.main(["q"]) == 1
    assert "Qdrant is running" in capsys.readouterr().err


def test_model_mismatch_message(monkeypatch, capsys):
    from models import ModelMismatchError

    def boom(*a, **k):
        raise ModelMismatchError("index built with bge, config says nomic")

    monkeypatch.setattr(app, "run_query", boom)
    assert app.main(["q"]) == 1
    assert "nomic" in capsys.readouterr().err


# ── helpers ────────────────────────────────────────────────────────────────


def test_is_web_url():
    assert app.is_web_url("https://example.com/x")
    assert app.is_web_url("http://example.com/x")
    assert not app.is_web_url("local:2021-05-16")


def test_run_query_wires_retrieve_into_answer(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "query.retriever.retrieve", lambda q, top_k, filters: seen.setdefault("retrieved", []) or []
    )
    monkeypatch.setattr("query.answerer.answer", lambda q, results: _answer(response="grounded"))
    out = app.run_query("q", 6, None)
    assert out.response == "grounded"


def test_run_query_reports_progress_phases_in_order(monkeypatch):
    """The UI's wait feedback is driven by this callback. Retrieval is no longer instant once
    reranking is on, so the phases must fire in order and name what is actually happening."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "query.retriever.retrieve", lambda q, top_k, filters: [SimpleNamespace(score=0.9)]
    )
    monkeypatch.setattr("query.answerer.answer", lambda q, results: _answer(response="ok"))
    seen: list[str] = []
    app.run_query("q", 6, None, progress=seen.append)
    assert len(seen) == 2
    assert "reranking" in seen[0].lower()  # retrieval phase names the slow part
    assert "1 sources" in seen[1] and "generating" in seen[1].lower()


def test_run_query_progress_is_optional(monkeypatch):
    """The CLI path passes no callback and must be unaffected."""
    monkeypatch.setattr("query.retriever.retrieve", lambda q, top_k, filters: [])
    monkeypatch.setattr("query.answerer.answer", lambda q, results: _answer(response="ok"))
    assert app.run_query("q", 6, None).response == "ok"


def test_warm_models_loads_embedder_store_and_reranker(monkeypatch):
    """Without this the first user question pays ~19 s of model loading and every later one
    pays none, which reads as the app being slow rather than as a one-time cost."""
    called = []
    monkeypatch.setattr("config.ENABLE_RERANK", True)
    monkeypatch.setattr("config.ENABLE_HYBRID_SEARCH", False)
    monkeypatch.setattr("ingestion.embedder.embed_query", lambda q: called.append("embed") or [0.1])
    monkeypatch.setattr(
        "query.retriever._get_store",
        lambda: type("S", (), {"search": lambda self, **kw: called.append("search") or []})(),
    )
    monkeypatch.setattr("query.retriever._get_reranker", lambda: called.append("rerank"))
    timings = app.warm_models()
    assert "embed" in called and "search" in called and "rerank" in called
    assert set(timings) == {"embedder", "vector_store", "reranker"}


def test_warm_models_loads_the_bm25_encoder_when_hybrid(monkeypatch):
    called = []
    monkeypatch.setattr("config.ENABLE_RERANK", False)
    monkeypatch.setattr("config.ENABLE_HYBRID_SEARCH", True)
    monkeypatch.setattr("ingestion.embedder.embed_query", lambda q: [0.1])
    monkeypatch.setattr(
        "query.retriever._get_store",
        lambda: type("S", (), {"search": lambda self, **kw: []})(),
    )
    monkeypatch.setattr(
        "ingestion.sparse_encoder.encode_query", lambda q: called.append("bm25") or None
    )
    timings = app.warm_models()
    assert called == ["bm25"]
    assert set(timings) == {"embedder", "vector_store", "bm25_encoder"}


def test_warm_models_skips_the_reranker_when_disabled(monkeypatch):
    called = []
    monkeypatch.setattr("config.ENABLE_RERANK", False)
    monkeypatch.setattr("ingestion.embedder.embed_query", lambda q: [0.1])
    monkeypatch.setattr(
        "query.retriever._get_store",
        lambda: type("S", (), {"search": lambda self, **kw: []})(),
    )
    monkeypatch.setattr("query.retriever._get_reranker", lambda: called.append("rerank"))
    timings = app.warm_models()
    assert called == []
    assert "reranker" not in timings


def test_warm_models_survives_a_failing_step(monkeypatch):
    """Warming is an optimisation: a broken step must not stop the app starting, because the
    lazy path still loads the model on first use."""
    monkeypatch.setattr("config.ENABLE_RERANK", True)
    monkeypatch.setattr("ingestion.embedder.embed_query", lambda q: [0.1])
    monkeypatch.setattr(
        "query.retriever._get_store",
        lambda: type("S", (), {"search": lambda self, **kw: []})(),
    )

    def boom():
        raise RuntimeError("no GPU")

    monkeypatch.setattr("query.retriever._get_reranker", boom)
    timings = app.warm_models()  # must not raise
    assert "reranker" not in timings
    assert "embedder" in timings


def test_cli_integration_seeded_index(monkeypatch, tmp_path, capsys):
    """The Success Criterion: a known question cites the right article."""
    import hashlib

    import storage.vector_store as vs
    from models import Chunk, EmbeddedChunk

    monkeypatch.setattr(vs, "QDRANT_IN_MEMORY", True)
    monkeypatch.setattr(vs, "EMBEDDING_DIM", 8)

    def vec(text: str):
        d = hashlib.sha256(text.encode()).digest()
        return [d[i] / 255 for i in range(8)]

    import query.retriever as rt

    rt._reset_store_for_tests()
    store = vs.VectorStore()
    store.upsert(
        [
            EmbeddedChunk(
                Chunk(
                    chunk_id="m2_b_0",
                    article_url="https://example.com/premium-x",
                    article_title="M2 Deep Dive",
                    published_at=datetime(2022, 3, 1, tzinfo=UTC),
                    tags=[],
                    text="the money supply surged forty percent",
                    content_type="body",
                    chunk_index=0,
                    total_chunks=1,
                    word_count=6,
                ),
                vec("money supply growth 2022"),
                "mock-embed",
            )
        ]
    )
    monkeypatch.setattr(rt, "_get_store", lambda: store)
    monkeypatch.setattr(rt, "embed_query", lambda q: vec("money supply growth 2022"))
    monkeypatch.setattr(
        rt, "get_embedding_provider", lambda: type("P", (), {"model_name": "mock-embed"})()
    )

    import query.answerer as an

    monkeypatch.setattr(an, "get_text_provider", lambda: _Prov())

    assert app.main(["What happened to the money supply in 2022?"]) == 0
    out = capsys.readouterr().out
    assert "https://example.com/premium-x" in out
    rt._reset_store_for_tests()


class _Prov:
    model_name = "mock-text"

    def complete(self, messages, max_tokens=1024, temperature=0.0):
        from llm_provider import TextResponse

        return TextResponse("The money supply surged in 2022 [1].", "mock-text", 100, 20)
