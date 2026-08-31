"""Tests for scheduler/monthly_job.py — corpus mode only (Milestone 2).

Integration level: a real MetadataDB (file in tmp_path, shared across runs), a real
in-memory VectorStore, mock embedding + vision providers, a fixture corpus on disk.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

import ingestion.embedder as embedder_mod
import ingestion.image_transcriber as it_mod
import scheduler.monthly_job as mj
import storage.vector_store as vs_mod
from llm_provider import TextResponse
from storage.metadata_db import MetadataDB


class _Vision:
    model_name = "mock-vision"
    supports_vision = True

    def complete_with_image(self, image_bytes, media_type, text_prompt, max_tokens=400):
        if "one word" in text_prompt:
            return TextResponse("chart", self.model_name, 10, 1)
        return TextResponse(
            "Line chart. X axis years. Y axis USD trillions. Grows then declines "
            "over the observed window with several notable inflection points recorded.",
            self.model_name,
            1500,
            200,
        )


class _Embed:
    model_name = "mock-embed"
    embedding_dim = 8
    query_prefix = ""

    def embed(self, texts):
        return [[float(len(t) % 7) / 7 + i * 0.01 for i in range(8)] for t in texts]


def _png() -> bytes:
    b = io.BytesIO()
    Image.effect_noise((200, 160), 128).convert("RGB").save(b, "PNG")
    return b.getvalue()


BODY = "The money supply and interest-rate path are the focus of this report. " * 20


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "meta.db"
    monkeypatch.setattr(mj, "MetadataDB", lambda *a, **k: MetadataDB(str(db_file)))
    monkeypatch.setattr(vs_mod, "QDRANT_IN_MEMORY", True)
    monkeypatch.setattr(vs_mod, "EMBEDDING_DIM", 8)
    monkeypatch.setattr(embedder_mod, "get_embedding_provider", _Embed)
    monkeypatch.setattr(it_mod, "get_vision_provider", _Vision)
    monkeypatch.setattr(it_mod, "IMAGE_CACHE_DB", str(tmp_path / "img.db"))
    monkeypatch.setattr(it_mod, "IMAGE_CACHE_DIR", str(tmp_path / "img_cache"))

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return type("Env", (), {"corpus": corpus, "db_file": db_file, "tmp_path": tmp_path})()


def _write(corpus, stem, *, body=BODY, url=None, image=False):
    import yaml

    fm = {"title": stem, "url": url or f"https://www.example.com/{stem}"}
    b = body
    if image:
        rel = f"images/{stem}/00-c.png"
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_bytes(_png())
        b = f"{body}\n\n![c]({rel})"
    (corpus / f"{stem}.md").write_text(
        f"---\n{yaml.safe_dump(fm).strip()}\n---\n{b}", encoding="utf-8"
    )


# ── corpus mode ────────────────────────────────────────────────────────────


async def test_new_corpus_is_ingested_then_a_second_run_is_all_skips(env):
    _write(env.corpus, "a", image=True)
    _write(env.corpus, "b")

    first = await mj.run_corpus_sync(str(env.corpus))
    assert first["new"] == 2 and first["updated"] == 0 and first["failed"] == 0

    # a fresh VectorStore for the second run; is_changed short-circuits before it is touched
    upserts = {"n": 0}
    real_upsert = vs_mod.VectorStore.upsert

    def counting_upsert(self, chunks):
        upserts["n"] += 1
        return real_upsert(self, chunks)

    import unittest.mock as m

    with m.patch.object(vs_mod.VectorStore, "upsert", counting_upsert):
        second = await mj.run_corpus_sync(str(env.corpus))

    assert second == {"new": 0, "updated": 0, "skipped": 2, "failed": 0}
    assert upserts["n"] == 0  # nothing re-embedded


async def test_edited_file_is_re_ingested_delete_before_upsert(env):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))

    _write(env.corpus, "a", body=BODY + " A new paragraph about liquidity conditions.")
    calls: list[str] = []
    import unittest.mock as m

    real_del = vs_mod.VectorStore.delete_by_url
    real_up = vs_mod.VectorStore.upsert
    with (
        m.patch.object(
            vs_mod.VectorStore,
            "delete_by_url",
            lambda self, u: (calls.append("delete"), real_del(self, u))[1],
        ),
        m.patch.object(
            vs_mod.VectorStore,
            "upsert",
            lambda self, c: (calls.append("upsert"), real_up(self, c))[1],
        ),
    ):
        stats = await mj.run_corpus_sync(str(env.corpus))

    assert stats["updated"] == 1
    assert calls == ["delete", "upsert"]  # delete first, exactly once each


async def test_a_new_file_added_later_is_picked_up(env):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))
    _write(env.corpus, "b")
    stats = await mj.run_corpus_sync(str(env.corpus))
    assert stats == {"new": 1, "updated": 0, "skipped": 1, "failed": 0}


async def test_unparseable_file_counts_as_failed_and_run_continues(env):
    _write(env.corpus, "good")
    (env.corpus / "bad.md").write_text("---\n: : :\n---\n", encoding="utf-8")
    stats = await mj.run_corpus_sync(str(env.corpus))
    assert stats["new"] == 1 and stats["failed"] == 1


async def test_stub_is_skipped_not_embedded(env):
    _write(env.corpus, "stub", body="only a few words, well under the minimum")
    stats = await mj.run_corpus_sync(str(env.corpus))
    assert stats == {"new": 0, "updated": 0, "skipped": 1, "failed": 0}


async def test_dry_run_writes_nothing(env):
    _write(env.corpus, "a", image=True)
    import unittest.mock as m

    with m.patch.object(vs_mod.VectorStore, "upsert", side_effect=AssertionError("wrote!")):
        stats = await mj.run_corpus_sync(str(env.corpus), dry_run=True)
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    db = MetadataDB(str(env.db_file))
    assert db.get_known_urls() == set()
    db.close()


async def test_corpus_not_found_aborts_with_error_code(env):
    stats = await mj.run_corpus_sync(str(env.tmp_path / "nope"))
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    db = MetadataDB(str(env.db_file))
    assert db.get_run_history()[0]["error_code"] == "corpus_not_found"
    db.close()


async def test_empty_corpus_aborts(env):
    assert await mj.run_corpus_sync(str(env.corpus)) == dict.fromkeys(
        ("new", "updated", "skipped", "failed"), 0
    )
    db = MetadataDB(str(env.db_file))
    assert db.get_run_history()[0]["error_code"] == "corpus_empty"
    db.close()


# ── selection / force ──────────────────────────────────────────────────────


async def test_only_processes_just_those_stems(env):
    _write(env.corpus, "a")
    _write(env.corpus, "b")
    _write(env.corpus, "c")
    stats = await mj.run_corpus_sync(str(env.corpus), only=["a", "c"])
    assert stats["new"] == 2
    db = MetadataDB(str(env.db_file))
    assert {u.rsplit("/", 1)[1] for u in db.get_known_urls()} == {"a", "c"}
    db.close()


async def test_only_with_an_unknown_stem_aborts(env):
    _write(env.corpus, "a")
    stats = await mj.run_corpus_sync(str(env.corpus), only=["a", "ghost"])
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    db = MetadataDB(str(env.db_file))
    assert db.get_run_history()[0]["error_code"] == "selection_not_found"
    assert db.get_known_urls() == set()  # nothing ingested
    db.close()


async def test_limit_applies_after_only(env):
    for s in ("a", "b", "c", "d"):
        _write(env.corpus, s)
    stats = await mj.run_corpus_sync(str(env.corpus), only=["a", "b", "c"], limit=2)
    assert stats["new"] == 2


async def test_force_re_ingests_an_unchanged_article(env):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))
    calls: list[str] = []
    import unittest.mock as m

    real_up = vs_mod.VectorStore.upsert
    with m.patch.object(
        vs_mod.VectorStore, "upsert", lambda self, c: (calls.append("upsert"), real_up(self, c))[1]
    ):
        stats = await mj.run_corpus_sync(str(env.corpus), force=True)
    assert stats["updated"] == 1 and calls == ["upsert"]


async def test_force_does_not_ingest_a_stub(env):
    _write(env.corpus, "s", body="tiny body")
    stats = await mj.run_corpus_sync(str(env.corpus), force=True)
    assert stats["skipped"] == 1 and stats["new"] == 0


# ── reset ──────────────────────────────────────────────────────────────────


async def test_reset_drops_vectors_before_rows_and_keeps_run_history(env, monkeypatch):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))

    order: list[str] = []
    import unittest.mock as m

    real_drop = vs_mod.VectorStore.drop_collection
    with (
        m.patch.object(
            vs_mod.VectorStore,
            "drop_collection",
            lambda self: (order.append("vectors"), real_drop(self))[1],
        ),
        m.patch(
            "storage.metadata_db.MetadataDB.drop_all_articles",
            autospec=True,
            side_effect=lambda self: order.append("rows") or 1,
        ),
    ):
        result = await mj.run_reset(assume_yes=True)

    assert order == ["vectors", "rows"]
    assert result["aborted"] is False
    db = MetadataDB(str(env.db_file))
    triggers = [r["trigger"] for r in db.get_run_history()]
    assert "reset" in triggers and "corpus" in triggers  # history intact
    db.close()


async def test_reset_refuses_without_tty_or_yes(env, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    result = await mj.run_reset(assume_yes=False)
    assert result["aborted"] is True


async def test_reset_does_not_touch_the_image_cache(env, monkeypatch):
    _write(env.corpus, "a", image=True)
    await mj.run_corpus_sync(str(env.corpus))
    import sqlite3

    before = (
        sqlite3.connect(str(env.tmp_path / "img.db"))
        .execute("SELECT COUNT(*) FROM image_cache")
        .fetchone()[0]
    )
    assert before >= 1

    await mj.run_reset(assume_yes=True)
    after = (
        sqlite3.connect(str(env.tmp_path / "img.db"))
        .execute("SELECT COUNT(*) FROM image_cache")
        .fetchone()[0]
    )
    assert after == before  # cache survives


# ── prune ──────────────────────────────────────────────────────────────────


async def test_prune_archives_a_corpus_article_whose_file_is_gone(env):
    _write(env.corpus, "a")
    _write(env.corpus, "b")
    await mj.run_corpus_sync(str(env.corpus))

    (env.corpus / "b.md").unlink()
    stats = await mj.run_prune()
    assert stats["archived"] == 1 and stats["skipped"] == 1 and stats["aborted"] is False

    db = MetadataDB(str(env.db_file))
    assert {u.rsplit("/", 1)[1] for u in db.get_known_urls()} == {"a"}  # b archived
    db.close()


async def test_prune_dry_run_writes_nothing(env):
    _write(env.corpus, "a")
    _write(env.corpus, "b")
    await mj.run_corpus_sync(str(env.corpus))
    (env.corpus / "b.md").unlink()

    stats = await mj.run_prune(dry_run=True)
    assert stats["archived"] == 0
    db = MetadataDB(str(env.db_file))
    assert len(db.get_known_urls()) == 2
    db.close()


async def test_prune_over_50pct_guard_aborts_without_force(env):
    for s in ("a", "b", "c", "d"):
        _write(env.corpus, s)
    await mj.run_corpus_sync(str(env.corpus))
    for s in ("b", "c", "d"):
        (env.corpus / f"{s}.md").unlink()  # 3 of 4 gone

    stats = await mj.run_prune()
    assert stats["aborted"] is True and stats["archived"] == 0

    forced = await mj.run_prune(force=True)
    assert forced["aborted"] is False and forced["archived"] == 3


async def test_prune_never_touches_web_articles(env):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))
    db = MetadataDB(str(env.db_file))
    # simulate a scraped article with a dangling source_path
    from datetime import UTC, datetime

    from models import Article

    web = Article(
        url="https://example.com/web-1",
        title="W",
        author="A",
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        tags=[],
        body_text="b",
        tables_md=[],
        images=[],
        word_count=1,
        content_hash="h" * 64,
        is_stub=False,
        source="web",
        source_path="/gone/x.html",
    )
    db.upsert_article(web, 1, "mock-embed")
    db.close()

    stats = await mj.run_prune(force=True)
    assert stats["archived"] == 0  # web row with a missing source_path is not a candidate


# ── concurrency guard ──────────────────────────────────────────────────────


async def test_open_run_blocks_a_new_corpus_sync(env):
    db = MetadataDB(str(env.db_file))
    db.start_run("corpus", corpus_dir=str(env.corpus))  # left open
    db.close()

    _write(env.corpus, "a")
    stats = await mj.run_corpus_sync(str(env.corpus))
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}


# ── milestone-3 stubs ──────────────────────────────────────────────────────


async def test_email_mode_is_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="Milestone 3"):
        await mj.run_email_triggered()
    with pytest.raises(NotImplementedError, match="Milestone 3"):
        mj.start_scheduler()


def test_cli_bare_invocation_points_at_corpus(capsys):
    assert mj.main([]) == 2
    assert "Milestone 3" in capsys.readouterr().err


def test_cli_stats_runs(env, capsys):
    assert mj.main(["--stats"]) == 0
    assert "articles" in capsys.readouterr().out


# ── inspect ────────────────────────────────────────────────────────────────


async def test_inspect_a_corpus_path_before_ingest(env, capsys):
    _write(env.corpus, "a", image=True)
    report = mj.run_inspect(str(env.corpus / "a.md"))
    assert report["loaded"] is True
    assert report["stored"] is False
    assert report["chunk_count"] >= 1
    assert len(report["images"]) == 1


async def test_inspect_reports_no_drift_after_a_clean_ingest(env):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))
    report = mj.run_inspect("https://example.com/a")
    assert report["stored"] is True and report["loaded"] is True
    assert report["drift"] is False
    assert report["content_hash"] == report["stored_hash"]


async def test_inspect_detects_drift_when_the_file_changed_underneath(env):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))
    _write(env.corpus, "a", body=BODY + " an extra sentence changing the hash")
    report = mj.run_inspect("https://example.com/a")
    assert report["drift"] is True


def test_inspect_an_unknown_url(env):
    report = mj.run_inspect("https://example.com/nope")
    assert report["loaded"] is False and report["stored"] is False


# ── CLI dispatch ───────────────────────────────────────────────────────────


def test_cli_corpus(env, capsys):
    _write(env.corpus, "a")
    assert mj.main(["--corpus", "--dir", str(env.corpus)]) == 0


def test_cli_corpus_with_only_and_dry_run(env):
    _write(env.corpus, "a")
    _write(env.corpus, "b")
    assert mj.main(["--corpus", "--dir", str(env.corpus), "--only", "a", "--dry-run"]) == 0


def test_cli_inspect(env, capsys):
    _write(env.corpus, "a")
    assert mj.main(["--inspect", str(env.corpus / "a.md")]) == 0
    assert "a" in capsys.readouterr().out


def test_cli_clear_lock(env, capsys):
    db = MetadataDB(str(env.db_file))
    db.start_run("corpus", corpus_dir="x")
    db.close()
    assert mj.main(["--clear-lock"]) == 0
    assert "cleared 1" in capsys.readouterr().out


def test_cli_prune(env):
    _write(env.corpus, "a")
    # nothing to prune, but the CLI path runs
    assert mj.main(["--prune", "--dry-run"]) == 0


def test_cli_reset_needs_yes_non_tty(env, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert mj.main(["--reset"]) == 0  # refused internally, still exits 0


# ── error paths ────────────────────────────────────────────────────────────


async def test_a_failure_inside_ingest_article_is_caught_and_counted(env, monkeypatch):
    _write(env.corpus, "a")
    _write(env.corpus, "b")

    async def boom(*a, **k):
        raise RuntimeError("embedding exploded")

    monkeypatch.setattr(mj, "ingest_article", boom)
    stats = await mj.run_corpus_sync(str(env.corpus))
    assert stats["failed"] == 2 and stats["new"] == 0  # both failed, run finished


async def test_a_top_level_crash_still_finishes_the_run_row_and_reraises(env, monkeypatch):
    _write(env.corpus, "a")

    def kaboom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(mj.md_loader, "get_known_urls", kaboom, raising=False)
    monkeypatch.setattr(mj, "VectorStore", kaboom)  # crash after start_run, before the loop

    with pytest.raises(KeyboardInterrupt):
        await mj.run_corpus_sync(str(env.corpus))

    db = MetadataDB(str(env.db_file))
    row = db.get_run_history()[0]
    assert row["completed_at"] is not None and row["error_code"] == "crashed"
    assert db.has_open_run() is None  # not wedged
    db.close()


async def test_prune_crash_finishes_the_run_row(env, monkeypatch):
    for s in ("a", "b", "c"):
        _write(env.corpus, s)
    await mj.run_corpus_sync(str(env.corpus))
    (env.corpus / "a.md").unlink()  # 1 of 3 — under the 50% guard

    def kaboom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(mj, "VectorStore", kaboom)
    with pytest.raises(KeyboardInterrupt):
        await mj.run_prune()
    db = MetadataDB(str(env.db_file))
    assert db.has_open_run() is None
    db.close()


async def test_reset_declined_at_the_prompt(env, monkeypatch):
    _write(env.corpus, "a")
    await mj.run_corpus_sync(str(env.corpus))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "no")
    result = await mj.run_reset(assume_yes=False)
    assert result["aborted"] is True
    db = MetadataDB(str(env.db_file))
    assert len(db.get_known_urls()) == 1  # nothing was dropped
    db.close()
