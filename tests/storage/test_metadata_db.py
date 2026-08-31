"""Tests for storage/metadata_db.py — all against an in-memory SQLite database.

Covers every bullet in SPEC_metadata_db.md § Testing Notes: change detection (new /
changed / unchanged / archived / pipeline-version), the archive-not-delete rule, the
run-history audit trail, and the pre-corpus-schema migration.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

import storage.metadata_db as mdb_module
from models import Article, ModelMismatchError
from storage.metadata_db import MetadataDB


def _article(
    url: str = "https://example.com/a",
    *,
    content_hash: str = "hash-1",
    source: str = "corpus",
    source_path: str | None = "/corpus/a.md",
    tags: list[str] | None = None,
    title: str = "Article A",
) -> Article:
    return Article(
        url=url,
        title=title,
        author="Writer",
        published_at=datetime(2021, 6, 27, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        tags=tags or ["macro"],
        body_text="body",
        tables_md=[],
        images=[],
        word_count=500,
        content_hash=content_hash,
        is_stub=False,
        source=source,
        source_path=source_path,
    )


@pytest.fixture
def db() -> MetadataDB:
    database = MetadataDB(":memory:")
    yield database
    database.close()


# ── known URLs / status filtering ───────────────────────────────────────────


def test_get_known_urls_only_active(db):
    db.upsert_article(_article("https://example.com/a"), 3, "bge")
    db.upsert_article(_article("https://example.com/b"), 3, "bge")
    db.archive_article("https://example.com/b")
    assert db.get_known_urls() == {"https://example.com/a"}


# ── is_changed ──────────────────────────────────────────────────────────────


class TestIsChanged:
    def test_new_url(self, db):
        assert db.is_changed("https://example.com/new", "any") is True

    def test_changed_hash(self, db):
        db.upsert_article(_article(content_hash="old"), 1, "bge")
        assert db.is_changed("https://example.com/a", "new") is True

    def test_same_hash(self, db):
        db.upsert_article(_article(content_hash="same"), 1, "bge")
        assert db.is_changed("https://example.com/a", "same") is False

    def test_archived_row_with_matching_hash(self, db):
        db.upsert_article(_article(content_hash="h"), 1, "bge")
        db.archive_article("https://example.com/a")
        assert db.is_changed("https://example.com/a", "h") is True

    def test_pipeline_version_bump_forces_change(self, db, monkeypatch):
        db.upsert_article(_article(content_hash="h"), 1, "bge")  # stamped current version
        monkeypatch.setattr(mdb_module, "PIPELINE_VERSION", mdb_module.PIPELINE_VERSION + 1)
        assert db.is_changed("https://example.com/a", "h") is True

    def test_pipeline_version_rollback_also_forces_change(self, db, monkeypatch):
        db.upsert_article(_article(content_hash="h"), 1, "bge")
        monkeypatch.setattr(mdb_module, "PIPELINE_VERSION", mdb_module.PIPELINE_VERSION - 1)
        assert db.is_changed("https://example.com/a", "h") is True


# ── upsert_article ──────────────────────────────────────────────────────────


class TestUpsert:
    def test_idempotent(self, db):
        art = _article(content_hash="h")
        db.upsert_article(art, 3, "bge")
        db.upsert_article(art, 3, "bge")
        assert db._count_articles() == 1

    def test_source_defaults_to_web(self, db):
        # a web article carries source_path=None
        db.upsert_article(_article(source="web", source_path=None), 2, "bge")
        row = db.get_article("https://example.com/a")
        assert row["source"] == "web"
        assert row["source_path"] is None

    def test_corpus_round_trips_with_source_path(self, db):
        db.upsert_article(_article(source="corpus", source_path="/c/a.md"), 2, "bge")
        row = db.get_article("https://example.com/a")
        assert row["source"] == "corpus"
        assert row["source_path"] == "/c/a.md"
        assert row["tags"] == ["macro"]

    def test_stamps_current_pipeline_version_not_the_rows(self, db, monkeypatch):
        db.upsert_article(_article(content_hash="v1"), 1, "bge")
        monkeypatch.setattr(mdb_module, "PIPELINE_VERSION", 7)
        db.upsert_article(_article(content_hash="v2"), 1, "bge")
        assert db.get_article("https://example.com/a")["pipeline_version"] == 7

    def test_revives_an_archived_row(self, db):
        db.upsert_article(_article(content_hash="h"), 1, "bge")
        db.archive_article("https://example.com/a")
        db.upsert_article(_article(content_hash="h2"), 1, "bge")
        assert db.get_article("https://example.com/a")["status"] == "active"

    def test_first_scraped_preserved_on_update(self, db):
        db.upsert_article(_article(content_hash="h1"), 1, "bge")
        first = db.get_article("https://example.com/a")["first_scraped"]
        db.upsert_article(_article(content_hash="h2"), 1, "bge")
        row = db.get_article("https://example.com/a")
        assert row["first_scraped"] == first
        assert row["last_scraped"] >= first

    def test_model_mismatch_raises(self, db):
        db.upsert_article(_article(content_hash="h"), 1, "bge-large")
        with pytest.raises(ModelMismatchError, match="bge-large"):
            db.upsert_article(_article(content_hash="h2"), 1, "nomic-embed")


# ── touch_source_path / update_last_scraped ─────────────────────────────────


def test_touch_source_path_leaves_hash_and_status(db):
    db.upsert_article(_article(content_hash="h", source_path="/c/old.md"), 1, "bge")
    db.touch_source_path("https://example.com/a", "/c/new.md")
    row = db.get_article("https://example.com/a")
    assert row["source_path"] == "/c/new.md"
    assert row["content_hash"] == "h"
    assert row["status"] == "active"


# ── archive / drop ─────────────────────────────────────────────────────────


def test_archive_does_not_delete(db):
    db.upsert_article(_article(content_hash="h"), 4, "bge")
    db.archive_article("https://example.com/a")
    row = db.get_article("https://example.com/a")
    assert row is not None
    assert row["status"] == "archived"
    assert row["chunk_count"] == 4  # history preserved


def test_no_hard_delete_method_exists():
    assert not hasattr(MetadataDB, "delete_article")


def test_drop_all_articles_keeps_ingestion_runs(db):
    db.upsert_article(_article("https://example.com/a"), 1, "bge")
    db.upsert_article(_article("https://example.com/b", source_path="/c/b.md"), 1, "bge")
    run_id = db.start_run("corpus", corpus_dir="/c")
    db.finish_run(run_id, {"new": 2, "updated": 0, "skipped": 0, "failed": 0})

    removed = db.drop_all_articles()
    assert removed == 2
    assert db._count_articles() == 0
    assert len(db.get_run_history()) == 1  # audit trail intact


# ── get_corpus_articles ────────────────────────────────────────────────────


def test_get_corpus_articles_filters(db):
    db.upsert_article(
        _article("https://example.com/c", source="corpus", source_path="/c/c.md"), 1, "bge"
    )
    db.upsert_article(
        _article("https://example.com/w", source="web", source_path="/leaked"), 1, "bge"
    )
    db.upsert_article(
        _article("https://example.com/x", source="corpus", source_path="/c/x.md"), 1, "bge"
    )
    db.archive_article("https://example.com/x")

    got = db.get_corpus_articles()
    urls = {row["url"] for row in got}
    assert urls == {"https://example.com/c"}  # web excluded, archived excluded


# ── ingestion runs ─────────────────────────────────────────────────────────


class TestRuns:
    def test_start_run_records_trigger_and_source_ref(self, db):
        run_id = db.start_run("corpus", corpus_dir="/data/corpus")
        row = db.get_run_history()[0]
        assert row["run_id"] == run_id
        assert row["trigger"] == "corpus"
        assert row["source_ref"] == "/data/corpus"
        assert row["completed_at"] is None

    def test_email_run_uses_email_uid_as_source_ref(self, db):
        db.start_run("email", email_uid="uid-42")
        assert db.get_run_history()[0]["source_ref"] == "uid-42"

    def test_finish_run_with_error(self, db):
        run_id = db.start_run("corpus", corpus_dir="/c")
        db.finish_run(
            run_id, {"new": 0, "updated": 0, "skipped": 0, "failed": 0}, error="corpus_not_found"
        )
        row = db.get_run_history()[0]
        assert row["error_code"] == "corpus_not_found"
        assert row["completed_at"] is not None

    def test_finish_run_ignores_extra_stats_keys(self, db):
        run_id = db.start_run("prune")
        db.finish_run(
            run_id,
            {
                "new": 0,
                "updated": 0,
                "skipped": 5,
                "failed": 0,
                "archived": 3,
                "chunks_removed": 40,
                "aborted": False,
            },
        )
        row = db.get_run_history()[0]
        assert row["articles_skipped"] == 5  # no crash on the extra keys

    def test_open_run_lifecycle(self, db):
        assert db.has_open_run() is None
        run_id = db.start_run("corpus", corpus_dir="/c")
        assert db.has_open_run() == run_id
        db.finish_run(run_id, {"new": 1, "updated": 0, "skipped": 0, "failed": 0})
        assert db.has_open_run() is None

    def test_clear_open_runs(self, db):
        db.start_run("corpus", corpus_dir="/c")
        db.start_run("email", email_uid="u")  # two stale rows
        assert db.clear_open_runs() == 2
        assert db.has_open_run() is None

    def test_get_stats_shape(self, db):
        db.upsert_article(
            _article("https://example.com/a", source="corpus", source_path="/c/a.md"), 1, "bge"
        )
        db.upsert_article(
            _article("https://example.com/b", source="web", source_path=None), 1, "bge"
        )
        db.archive_article("https://example.com/b")
        run_id = db.start_run("corpus", corpus_dir="/c")
        db.finish_run(run_id, {"new": 1, "updated": 0, "skipped": 0, "failed": 0})

        stats = db.get_stats()
        assert stats["articles"] == {"total": 2, "active": 1, "archived": 1}
        assert stats["by_source"]["corpus"] == 1
        assert stats["last_run"]["corpus"]["run_id"] == run_id
        assert stats["last_run"]["email"] is None


# ── migration from a pre-corpus schema ─────────────────────────────────────


def test_migration_adds_columns_without_data_loss(tmp_path):
    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE articles (
            url TEXT PRIMARY KEY, title TEXT NOT NULL, author TEXT, published_at TEXT,
            first_scraped TEXT NOT NULL, last_scraped TEXT NOT NULL,
            content_hash TEXT NOT NULL, word_count INTEGER, tags TEXT, chunk_count INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "https://example.com/legacy",
            "Legacy",
            "W",
            None,
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:00:00+00:00",
            "legacy-hash",
            900,
            '["old"]',
            5,
        ),
    )
    conn.commit()
    conn.close()

    db = MetadataDB(str(db_file))
    row = db.get_article("https://example.com/legacy")
    assert row is not None  # no data loss
    assert row["title"] == "Legacy"
    assert row["source"] == "web"  # migration default
    assert row["pipeline_version"] == 0
    # a version-0 row must report as changed so the fix that added the column re-ingests it
    assert db.is_changed("https://example.com/legacy", "legacy-hash") is True
    db.close()


def test_creates_parent_directory(tmp_path):
    nested = tmp_path / "data" / "sub" / "metadata.db"
    db = MetadataDB(str(nested))
    assert nested.exists()
    db.close()


# ── touch points that are easy to miss ─────────────────────────────────────


def test_update_last_scraped_touches_only_that_column(db):
    db.upsert_article(_article(content_hash="h"), 1, "bge")
    before = db.get_article("https://example.com/a")
    db.update_last_scraped("https://example.com/a")
    after = db.get_article("https://example.com/a")
    assert after["content_hash"] == before["content_hash"]
    assert after["last_scraped"] >= before["last_scraped"]


def test_get_known_urls_empty_is_a_set(db):
    assert db.get_known_urls() == set()


def test_clear_open_runs_with_nothing_open_returns_zero(db):
    run_id = db.start_run("corpus", corpus_dir="/c")
    db.finish_run(run_id, {"new": 0, "updated": 0, "skipped": 0, "failed": 0})
    assert db.clear_open_runs() == 0


# ── lock-retry ────────────────────────────────────────────────────────────
#
# sqlite3.Connection.execute is a read-only C attribute, so the retry is exercised
# through a thin proxy that fails `execute` a controlled number of times.


class _FlakyConn:
    def __init__(self, real: sqlite3.Connection, fail_times: int, exc: Exception):
        self._real = real
        self._remaining = fail_times
        self._exc = exc
        self.calls = 0

    def execute(self, sql, params=()):
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise self._exc
        return self._real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_execute_retries_a_locked_database_then_succeeds(db, monkeypatch):
    monkeypatch.setattr(mdb_module.time, "sleep", lambda _s: None)
    fake = _FlakyConn(db._conn, 2, sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(db, "_conn", fake)
    db._execute("UPDATE articles SET last_scraped = ? WHERE url = ?", ("t", "none"))
    assert fake.calls == 3  # two failures, one success


def test_execute_reraises_after_exhausting_retries(db, monkeypatch):
    monkeypatch.setattr(mdb_module.time, "sleep", lambda _s: None)
    fake = _FlakyConn(db._conn, 99, sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(db, "_conn", fake)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        db._execute("UPDATE articles SET last_scraped = ? WHERE url = ?", ("t", "x"))
    assert fake.calls == 3  # capped at _LOCK_RETRIES


def test_execute_does_not_retry_a_non_lock_error(db, monkeypatch):
    monkeypatch.setattr(mdb_module.time, "sleep", lambda _s: None)
    fake = _FlakyConn(db._conn, 99, sqlite3.OperationalError("no such table: nope"))
    monkeypatch.setattr(db, "_conn", fake)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        db._execute("SELECT 1")
    assert fake.calls == 1  # not retried
