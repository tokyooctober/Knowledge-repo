"""SQLite record of every indexed article and every ingestion run.

`url` is the primary key for the whole system. Both phases write here; the `source`
column records which one. This module stores what it is given and never re-normalises a
URL — callers pass values already through `models.canonical_url()`.

Nothing here hard-deletes an individual article row: `archive_article()` is the only
removal path. `drop_all_articles()` is the one blunt exception, for `--reset` only, and it
leaves `ingestion_runs` intact.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from config import DB_PATH, PIPELINE_VERSION
from logger import get_logger
from models import ModelMismatchError

if TYPE_CHECKING:
    from models import Article

log = get_logger(__name__)

_LOCK_RETRIES = 3
_LOCK_BACKOFF_S = 0.5

_ARTICLES_DDL = """
CREATE TABLE IF NOT EXISTS articles (
    url              TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    author           TEXT,
    published_at     TEXT,
    first_scraped    TEXT NOT NULL,
    last_scraped     TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    word_count       INTEGER,
    tags             TEXT,
    chunk_count      INTEGER,
    status           TEXT NOT NULL DEFAULT 'active',
    embedding_model  TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'web',
    source_path      TEXT,
    pipeline_version INTEGER NOT NULL DEFAULT 0
);
"""

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id           TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    completed_at     TEXT,
    articles_new     INTEGER DEFAULT 0,
    articles_updated INTEGER DEFAULT 0,
    articles_skipped INTEGER DEFAULT 0,
    articles_failed  INTEGER DEFAULT 0,
    trigger          TEXT NOT NULL DEFAULT 'manual',
    source_ref       TEXT,
    error_code       TEXT,
    error_log        TEXT
);
"""

# column -> "ADD COLUMN" clause, applied to a pre-existing `articles` table that predates
# the column. Order matters only for readability.
_ARTICLES_MIGRATIONS = {
    "status": "status TEXT NOT NULL DEFAULT 'active'",
    "embedding_model": "embedding_model TEXT NOT NULL DEFAULT ''",
    "source": "source TEXT NOT NULL DEFAULT 'web'",
    "source_path": "source_path TEXT",
    "pipeline_version": "pipeline_version INTEGER NOT NULL DEFAULT 0",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MetadataDB:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            first_run = not Path(db_path).exists()
        else:
            first_run = True
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        if first_run:
            log.info("Database file created (first run)", extra={"db_path": db_path})
        else:
            log.debug(
                "Connected to existing database",
                extra={"db_path": db_path, "article_count": self._count_articles()},
            )

    # ── schema ──────────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_ARTICLES_DDL)
        cur.execute(_RUNS_DDL)
        existing = {row["name"] for row in cur.execute("PRAGMA table_info(articles)")}
        for column, clause in _ARTICLES_MIGRATIONS.items():
            if column not in existing:
                cur.execute(f"ALTER TABLE articles ADD COLUMN {clause}")
                log.info(
                    "Schema migration applied",
                    extra={"migration": f"articles.{column}", "db_path": self.db_path},
                )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
        self._conn.commit()

    # ── write helper with the documented lock-retry ─────────────────────────

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        for attempt in range(1, _LOCK_RETRIES + 1):
            try:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == _LOCK_RETRIES:
                    if "locked" in str(exc).lower():
                        log.error(
                            "SQLite locked after max retries",
                            extra={
                                "db_path": self.db_path,
                                "attempts": attempt,
                                "error_type": type(exc).__name__,
                            },
                            exc_info=True,
                        )
                    raise
                log.warning(
                    "SQLite locked — retrying",
                    extra={"attempt": attempt, "retry_ms": int(_LOCK_BACKOFF_S * 1000)},
                )
                time.sleep(_LOCK_BACKOFF_S)
        raise AssertionError("unreachable")  # pragma: no cover

    def _count_articles(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    # ── articles: read ─────────────────────────────────────────────────────

    def get_known_urls(self) -> set[str]:
        rows = self._conn.execute("SELECT url FROM articles WHERE status = 'active'").fetchall()
        urls = {row["url"] for row in rows}
        if not urls:
            log.warning("get_known_urls returned empty set", extra={"db_path": self.db_path})
        return urls

    def get_article(self, url: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM articles WHERE url = ?", (url,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["tags"] = json.loads(record["tags"]) if record["tags"] else []
        return record

    def is_changed(self, url: str, new_hash: str) -> bool:
        row = self.get_article(url)
        if row is None:
            return True  # new article
        if row["status"] == "archived":
            return True  # revive: its vectors were removed
        if row["pipeline_version"] != PIPELINE_VERSION:
            return True  # our code changed, not the corpus
        return row["content_hash"] != new_hash

    def get_corpus_articles(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT url, source_path, title FROM articles "
            "WHERE source = 'corpus' AND status = 'active'"
        ).fetchall()
        return [dict(row) for row in rows]

    # ── articles: write ────────────────────────────────────────────────────

    def upsert_article(self, article: Article, chunk_count: int, model_name: str) -> None:
        existing = self.get_article(article.url)
        if existing is not None and existing["embedding_model"] not in ("", model_name):
            log.critical(
                "Model mismatch on upsert",
                extra={
                    "url": article.url,
                    "stored_model": existing["embedding_model"],
                    "new_model": model_name,
                },
            )
            raise ModelMismatchError(
                f"{article.url}: collection was built with {existing['embedding_model']!r}, "
                f"upsert used {model_name!r}. Re-index from scratch (--reset)."
            )

        now = _now()
        published = article.published_at.isoformat() if article.published_at is not None else None
        self._execute(
            """
            INSERT INTO articles (
                url, title, author, published_at, first_scraped, last_scraped,
                content_hash, word_count, tags, chunk_count, status, embedding_model,
                source, source_path, pipeline_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                author = excluded.author,
                published_at = excluded.published_at,
                last_scraped = excluded.last_scraped,
                content_hash = excluded.content_hash,
                word_count = excluded.word_count,
                tags = excluded.tags,
                chunk_count = excluded.chunk_count,
                embedding_model = excluded.embedding_model,
                source = excluded.source,
                source_path = excluded.source_path,
                pipeline_version = excluded.pipeline_version,
                status = 'active'
            """,
            (
                article.url,
                article.title,
                article.author,
                published,
                now,
                now,
                article.content_hash,
                article.word_count,
                json.dumps(article.tags),
                chunk_count,
                model_name,
                article.source,
                article.source_path,
                PIPELINE_VERSION,
            ),
        )
        verb = "updated (changed)" if existing else "inserted (new)"
        log.debug(
            f"Article {verb}",
            extra={
                "url": article.url,
                "chunk_count": chunk_count,
                "word_count": article.word_count,
            },
        )

    def update_last_scraped(self, url: str) -> None:
        self._execute("UPDATE articles SET last_scraped = ? WHERE url = ?", (_now(), url))
        log.debug("Article unchanged — last_scraped updated", extra={"url": url})

    def touch_source_path(self, url: str, source_path: str) -> None:
        """Record a new .md path for an unchanged corpus article.

        A rename leaves content_hash identical, so is_changed() is False and
        upsert_article never runs. Without this the stored source_path goes stale and the
        next --prune archives an article that is still in the corpus. No-op-safe for web
        articles (source_path is None there and the scheduler skips the call).
        """
        self._execute(
            "UPDATE articles SET source_path = ?, last_scraped = ? WHERE url = ?",
            (source_path, _now(), url),
        )

    def archive_article(self, url: str) -> None:
        row = self.get_article(url)
        self._execute("UPDATE articles SET status = 'archived' WHERE url = ?", (url,))
        log.info(
            "Article archived",
            extra={"url": url, "source_path": row["source_path"] if row else None},
        )

    def drop_all_articles(self) -> int:
        """`--reset` only. Empties `articles`; leaves `ingestion_runs` untouched."""
        count = self._count_articles()
        self._execute("DELETE FROM articles")
        return count

    # ── ingestion runs ────────────────────────────────────────────────────

    def start_run(self, trigger: str, **context: str) -> str:
        run_id = str(uuid.uuid4())
        source_ref = context.get("corpus_dir") or context.get("email_uid")
        self._execute(
            "INSERT INTO ingestion_runs (run_id, started_at, trigger, source_ref) "
            "VALUES (?, ?, ?, ?)",
            (run_id, _now(), trigger, source_ref),
        )
        log.info(
            "Ingestion run started",
            extra={"run_id": run_id, "trigger": trigger, "source_ref": source_ref},
        )
        return run_id

    def finish_run(self, run_id: str, stats: dict, error: str | None = None) -> None:
        """Close a run row. Reads only new/updated/skipped/failed from `stats`; any other
        key (a prune's `archived`, `chunks_removed`, …) is ignored so every mode can pass
        its own stats shape through unchanged.
        """
        self._execute(
            """
            UPDATE ingestion_runs SET
                completed_at = ?,
                articles_new = ?,
                articles_updated = ?,
                articles_skipped = ?,
                articles_failed = ?,
                error_code = ?
            WHERE run_id = ?
            """,
            (
                _now(),
                int(stats.get("new", 0)),
                int(stats.get("updated", 0)),
                int(stats.get("skipped", 0)),
                int(stats.get("failed", 0)),
                error,
                run_id,
            ),
        )
        level = log.warning if error else log.info
        level(
            "Ingestion run finished with error" if error else "Ingestion run completed",
            extra={
                "run_id": run_id,
                "error_code": error,
                **{k: int(stats.get(k, 0)) for k in ("new", "updated", "skipped", "failed")},
            },
        )

    def has_open_run(self) -> str | None:
        row = self._conn.execute(
            "SELECT run_id FROM ingestion_runs WHERE completed_at IS NULL "
            "ORDER BY started_at LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    def clear_open_runs(self) -> int:
        cur = self._execute(
            "UPDATE ingestion_runs SET completed_at = ?, error_code = 'crashed' "
            "WHERE completed_at IS NULL",
            (_now(),),
        )
        count = cur.rowcount
        if count:
            log.info("Stale open runs cleared", extra={"cleared_count": count})
        return count

    def get_run_history(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    # ── stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        conn = self._conn
        by_status = {
            row["status"]: row["n"]
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM articles GROUP BY status")
        }
        by_source = {
            row["source"]: row["n"]
            for row in conn.execute(
                "SELECT source, COUNT(*) AS n FROM articles WHERE status = 'active' GROUP BY source"
            )
        }
        pipeline_versions = {
            row["pipeline_version"]: row["n"]
            for row in conn.execute(
                "SELECT pipeline_version, COUNT(*) AS n FROM articles "
                "WHERE status = 'active' GROUP BY pipeline_version"
            )
        }
        last_run: dict[str, dict | None] = {}
        for trigger in ("corpus", "email", "manual", "prune", "reset"):
            row = conn.execute(
                "SELECT * FROM ingestion_runs WHERE trigger = ? ORDER BY started_at DESC LIMIT 1",
                (trigger,),
            ).fetchone()
            last_run[trigger] = dict(row) if row else None

        return {
            "articles": {
                "total": sum(by_status.values()),
                "active": by_status.get("active", 0),
                "archived": by_status.get("archived", 0),
            },
            "by_source": {"corpus": by_source.get("corpus", 0), "web": by_source.get("web", 0)},
            "pipeline_versions": pipeline_versions,
            "last_run": last_run,
        }

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
