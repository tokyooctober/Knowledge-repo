# `storage/metadata_db.py` — Metadata Store (SQLite)

---
```
module:     storage/metadata_db.py
spec:       storage/SPEC_metadata_db.md
layer:      Storage
depends_on: config.py · logger.py
            models.py  (Article)
used_by:    scheduler/monthly_job.py  (upsert, is_changed, start_run, finish_run, prune)
            scraper/crawler.py  (get_known_urls for is_new flagging)
            app.py  (get_stats, get_run_history for the sidebar)
services:   SQLite  (file at data/metadata.db)
files:      data/metadata.db  (articles table + ingestion_runs table)
```
---

## Purpose
Maintain a lightweight relational record of every article that has been indexed, from
either content source. Supports change detection (via content hash), delta discovery for
re-syncs and monthly updates, and audit logging of ingestion runs.

`url` is the primary key for the whole system. Both phases write here: a corpus article's
`url` comes from its markdown frontmatter, a web article's from the scraped page. Both
arrive already normalised by `models.canonical_url()` — this module stores what it is
given and never re-normalises, so a caller that skips it creates a second row for an
article that already exists. The `source` column records which phase wrote the row, which
is what makes a corpus-only prune safe.

---

## Responsibilities
- Store one row per article with URL, title, dates, tags, and content hash
- Look up articles by URL to check if they exist and whether they have changed
- Return the full set of known URLs for the crawler's deduplication step
- Record ingestion run history (start time, articles processed, errors)
- Mark articles as archived when they should stop being retrieved without losing their
  history (`archive_article`). This is the **only** removal path: `run_prune()` archives
  corpus articles whose file is gone, and web articles are never removed. Nothing in the
  system hard-deletes an article row.

---

## Schema

### Table: `articles`
```sql
CREATE TABLE IF NOT EXISTS articles (
    url             TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    author          TEXT,
    published_at    TEXT,             -- ISO8601 or NULL
    first_scraped   TEXT NOT NULL,    -- ISO8601 datetime
    last_scraped    TEXT NOT NULL,    -- ISO8601 datetime
    content_hash    TEXT NOT NULL,    -- models.content_hash(): body + tables + per-image
                                      -- identity (URL for web; URL+size+mtime for corpus)
    word_count      INTEGER,
    tags            TEXT,             -- JSON array: '["tag1","tag2"]'
    chunk_count     INTEGER,          -- number of chunks in vector DB
    status          TEXT NOT NULL DEFAULT 'active',  -- 'active' | 'archived'
    embedding_model TEXT NOT NULL,    -- model name used to embed
    source          TEXT NOT NULL DEFAULT 'web',     -- 'corpus' | 'web'
    source_path     TEXT,             -- absolute .md path for corpus rows; NULL for web
    pipeline_version INTEGER NOT NULL DEFAULT 0       -- config.PIPELINE_VERSION at ingest
);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
```

`source` and `source_path` mirror the fields on `Article`. They exist for two reasons:

- `run_prune()` must never touch a web-sourced article, and `WHERE source = 'corpus'` is
  the guard that guarantees it
- when a corpus and an email article collide on the same `url`, `source` is what tells you
  which phase wrote the surviving row

`content_hash` covers body text, tables, and image sources — not the file path or mtime,
so renaming a corpus file does not force a re-embed.

`pipeline_version` records `config.PIPELINE_VERSION` as it stood when the row was written.
It covers the axis `content_hash` deliberately does not: a change in *this repo's* code
rather than in the corpus. Fixing a bug in `md_loader`'s image resolution, changing chunk
boundaries, or rewriting the vision prompt leaves every `content_hash` identical, so
`is_changed()` returns False and every article in the index keeps its wrong vectors,
silently and permanently. Bumping `PIPELINE_VERSION` in the same commit as the fix makes
the next `--corpus` re-ingest everything the fix affects, with no wipe and no flag to
remember.

Widening `content_hash` to include a version number instead would work but is worse: the
hash is also the image cache's identity, and two orthogonal reasons to re-ingest are
easier to debug when they are two columns. The 0 default exists so rows written before
this column existed re-ingest exactly once on the first run after the migration.

### Table: `ingestion_runs`
```sql
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id          TEXT PRIMARY KEY,   -- UUID4
    started_at      TEXT NOT NULL,      -- ISO8601
    completed_at    TEXT,               -- NULL if still running / crashed
    articles_new    INTEGER DEFAULT 0,
    articles_updated INTEGER DEFAULT 0,
    articles_skipped INTEGER DEFAULT 0,
    articles_failed INTEGER DEFAULT 0,
    trigger         TEXT NOT NULL DEFAULT 'manual',  -- 'corpus'|'email'|'manual'|'prune'|'reset'
    source_ref      TEXT,               -- corpus_dir for corpus runs, email_uid for email runs
    error_code      TEXT,               -- 'corpus_not_found' | 'session_error' | … | NULL
    error_log       TEXT                -- JSON list of {url, error} dicts
);
```

`trigger` is what makes the log queries in [SPEC.md](../SPEC.md) work
(`grep '"trigger": "corpus"'`) and what lets `get_stats()` report when each phase last ran.

---

## Core Operations

### `get_known_urls() -> set[str]`
```sql
SELECT url FROM articles WHERE status = 'active';
```
Used by the crawler at startup to filter out already-indexed URLs, and by both scheduler
modes for the new-vs-known decision.

### `get_article(url: str) -> dict | None`
```sql
SELECT * FROM articles WHERE url = ?;
```
Returns full row as a dict, or `None` if not found.

### `is_changed(url: str, new_hash: str) -> bool`
```python
row = get_article(url)
if row is None:                       return True   # new article
if row["status"] == "archived":       return True   # revive: its vectors were removed
if row["pipeline_version"] != PIPELINE_VERSION:
                                      return True   # our code changed, not the corpus
return row["content_hash"] != new_hash
```
The archived check is load-bearing. Without it, restoring a pruned `.md` file unchanged
produces a hash match, the scheduler skips it, and `upsert_article` is never called — so
the article stays archived and unsearchable with no error anywhere.

The version check is the same class of bug one level up: the corpus is unchanged, our
handling of it is not. Note it compares `!=`, not `<`. Rolling `PIPELINE_VERSION` back to
reproduce an old result must also re-ingest, or the index silently mixes output from two
different pipelines — which is exactly the state that makes a retrieval bug impossible to
reason about.

`--force` (see [SPEC_monthly_job.md](../scheduler/SPEC_monthly_job.md)) short-circuits
this function entirely rather than adding a fourth condition to it. Keep the flag out of
this signature: change detection stays a pure function of stored state, and the decision
to ignore it stays in the caller that a human typed a flag at.

### `upsert_article(article: Article, chunk_count: int, model_name: str)`
```sql
INSERT INTO articles (...) VALUES (...)
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
    status = 'active';
```
Every column except `url` and `first_scraped` is refreshed. `source_path` in particular:
it is the only thing `run_prune()` checks, so a stale value gets a live article archived.

`pipeline_version` is written from `config.PIPELINE_VERSION` at the moment of the upsert —
never copied from the existing row, and never passed in by the caller. A row's version
means "this is the pipeline that produced the vectors currently in Qdrant for this
article", so it can only be stamped by the code that just produced them.

### `touch_source_path(url: str, source_path: str)`
```sql
UPDATE articles SET source_path = ?, last_scraped = ? WHERE url = ?;
```
The unchanged-content path needs this too. Renaming a corpus file leaves `content_hash`
identical, so `is_changed()` returns False and `upsert_article` is never called — the row
keeps pointing at the old path, and the next `--prune` archives an article that is still
in the corpus. The scheduler therefore calls `update_last_scraped(url)` **and**
`touch_source_path(url, article.source_path)` on every unchanged corpus article. (For web
articles `source_path` is None and this is a no-op.)

### `archive_article(url: str)`
```sql
UPDATE articles SET status = 'archived' WHERE url = ?;
```
Marks a row inactive without deleting it — `first_scraped`, `chunk_count` and the run
history are preserved, and `get_known_urls()` stops returning it. Called by `run_prune()`
after `vector_store.delete_by_url(url)` has removed the article's vectors.

An archived row is not a tombstone: `upsert_article` sets `status = 'active'` on conflict,
so restoring a `.md` file and re-running `--corpus` revives the article in place.

**Consequence for change detection.** `is_changed()` compares hashes and does not look at
`status`, so an archived article whose file returns unedited has a matching hash and would
be skipped — leaving an active row with no vectors. `is_changed()` must therefore return
`True` whenever the stored row has `status = 'archived'`, regardless of hash.

### `drop_all_articles() -> int`
```sql
DELETE FROM articles;
```
The one exception to "nothing hard-deletes an article", and the reason it is a separate
method with a blunt name rather than a parameter on `archive_article`. It exists only to
serve `monthly_job.py --reset`, which pairs it with `vector_store.drop_collection()` to
return the index to a first-run state. Returns the number of rows removed.

Two rules that make it safe to have in the codebase at all:

- It never runs implicitly. No code path reaches it except the `--reset` flag, and that
  flag prompts for confirmation before calling it.
- It does **not** touch `ingestion_runs`. The run history is the audit trail of what was
  ingested and when, including the runs whose articles this just deleted; wiping it is how
  you lose the ability to answer "was this ever ingested, and did it fail?" after a reset
  that turns out to have been a mistake. A `--reset` therefore leaves a database that
  knows it has been reset.

It also does **not** touch the image transcription cache, which lives in its own database
— see the reset section of [SPEC_monthly_job.md](../scheduler/SPEC_monthly_job.md).

### `get_corpus_articles() -> list[dict]`
```sql
SELECT url, source_path, title FROM articles
WHERE source = 'corpus' AND status = 'active';
```
Used by `run_prune()` to find indexed corpus articles whose `.md` file no longer exists.
Never returns web-sourced rows, so a prune cannot delete a scraped article.

### `start_run(trigger, **context) -> str` / `finish_run(run_id, stats, error=None)`
`finish_run` reads only `new`, `updated`, `skipped` and `failed` from `stats` and ignores
every other key, so `run_prune()` can pass its own extra counters (`archived`,
`chunks_removed`, `aborted`) through the same call without a special case.
Insert a row into `ingestion_runs` at start, recording `trigger` and a `source_ref`
(`corpus_dir` for a corpus run, `email_uid` for an email run); update `completed_at`,
counters, and `error_code` at the end. `finish_run` must be called on every exit path,
including aborts — an open row blocks the next run.

### `has_open_run() -> str | None` / `clear_open_runs() -> int`
```sql
SELECT run_id FROM ingestion_runs WHERE completed_at IS NULL ORDER BY started_at LIMIT 1;
```
The concurrency guard. A hard crash leaves `completed_at` NULL forever, so
`clear_open_runs()` backs the `--clear-lock` CLI flag. This is advisory only — it is not a
lock, and two processes started in the same second can both pass the check.

---

## Change Detection Flow (used by scheduler)
```
For each scraped article:
  1. extractor produces Article with content_hash
  2. metadata_db.is_changed(url, content_hash)
     → True  (new or changed): proceed to chunk → embed → upsert vector DB
                                then metadata_db.upsert_article(...)
     → False (unchanged): skip; metadata_db.update_last_scraped(url)
```
This prevents re-embedding articles that haven't changed, saving embedding cost and time.
It is identical for both phases — the only difference is whether the `Article` came from
`md_loader.load_article()` or `extractor.extract()`.

`update_last_scraped(url)` is the "seen but unchanged" touch. The name predates the corpus
source; it now means *last confirmed present at source*, whether that source was a scrape
or a file on disk.

---

## Configuration Constants
```python
DB_PATH = "data/metadata.db"
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| DB file path doesn't exist | Create directory and file on first connection |
| SQLite locked (concurrent write) | Retry up to 3 times with 500ms backoff (WAL mode enabled) |
| Schema out of date | Run `ALTER TABLE` migrations; log at INFO |
| `pipeline_version` column absent (pre-existing DB) | `ALTER TABLE articles ADD COLUMN pipeline_version INTEGER NOT NULL DEFAULT 0`; every existing row then re-ingests once on the next run, which is the intended behaviour — the column exists precisely because those rows were produced by unknown code |
| `upsert_article` called with mismatched model | Raise `ModelMismatchError` if stored `embedding_model` differs from new `model_name`. This is a backstop — `vector_store` makes the same check at startup against the collection, so in practice a mismatched run aborts before any write. If it does fire here, the scheduler has already called `vector_store.upsert`; recover with `--reset`. |

---

## Key Dependencies
- `sqlite3` — stdlib, no extra install
- `json` — tags serialisation (stdlib)
- `uuid` — run_id generation (stdlib)

---

## Public Interface
```python
class MetadataDB:
    def __init__(self, db_path: str = DB_PATH): ...   # connect + migrate schema

    def get_known_urls(self) -> set[str]: ...
    def get_article(self, url: str) -> dict | None: ...
    def is_changed(self, url: str, new_hash: str) -> bool: ...
    def upsert_article(self, article: Article, chunk_count: int, model_name: str) -> None: ...
    def update_last_scraped(self, url: str) -> None: ...
    def archive_article(self, url: str) -> None: ...  # the only removal path
    # No delete_article(). Nothing hard-deletes an *individual* article row.

    def drop_all_articles(self) -> int: ...   # --reset only; keeps ingestion_runs

    def get_corpus_articles(self) -> list[dict]: ...  # source='corpus' AND status='active'

    def start_run(self, trigger: str, **context) -> str: ...
        # trigger: 'corpus' | 'email' | 'manual' | 'prune' | 'reset'
        # context: corpus_dir=… or email_uid=… → stored as source_ref
    def finish_run(self, run_id: str, stats: dict, error: str | None = None) -> None: ...

    def has_open_run(self) -> str | None: ...   # run_id of an unfinished run, or None
    def clear_open_runs(self) -> int: ...       # close stale rows; returns count

    def get_run_history(self, limit: int = 10) -> list[dict]: ...
    def get_stats(self) -> dict: ...         # counts by status and by source, last run per
                                             # trigger, and the pipeline_version spread —
                                             # more than one live version in the index means
                                             # a re-ingest was started and not finished
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.storage.metadata_db"
```

| Event | Level | Extra fields |
|---|---|---|
| Database file created (first run) | INFO | `db_path` |
| Connected to existing database | DEBUG | `db_path`, `article_count` |
| Schema migration applied | INFO | `migration`, `db_path` |
| SQLite locked — retrying | WARNING | `attempt`, `retry_ms` |
| SQLite locked after max retries | ERROR | `db_path`, `attempts`, `error_type` |
| Article inserted (new) | DEBUG | `url`, `word_count`, `chunk_count` |
| Article updated (changed) | DEBUG | `url`, `old_hash`, `new_hash` |
| Article unchanged — last_scraped updated | DEBUG | `url` |
| Article archived | INFO | `url`, `source_path` |
| Archived article revived by re-ingest | INFO | `url` |
| Model mismatch on upsert | CRITICAL | `url`, `stored_model`, `new_model` |
| Ingestion run started | INFO | `run_id`, `trigger`, `source_ref` |
| Open run detected — new run blocked | WARNING | `run_id`, `existing_run_id`, `started_at` |
| Stale open runs cleared | INFO | `cleared_count` |
| Ingestion run completed | INFO | `run_id`, `trigger`, `new`, `updated`, `skipped`, `failed` |
| Ingestion run finished with error | WARNING | `run_id`, `trigger`, `error_code` |
| get_known_urls returned empty set | WARNING | `db_path` — possible first run or corrupted DB |

---

## Testing Notes
- Use an in-memory SQLite DB (`":memory:"`) for all tests
- Assert `get_known_urls` returns only `status='active'` rows
- Assert `is_changed` returns `True` for new URL and for changed hash
- Assert `is_changed` returns `False` for same hash
- Assert `upsert_article` is idempotent (calling twice with same data = one row)
- Assert `source` defaults to `'web'` and round-trips `'corpus'` with its `source_path`
- Assert `get_corpus_articles` returns only `source='corpus'` **and** `status='active'` rows
- Assert `get_corpus_articles` never returns a web row, even one with a non-NULL `source_path`
- Assert `touch_source_path` updates `source_path` without touching `content_hash` or `status`
- Assert `start_run('corpus', corpus_dir=…)` stores both `trigger` and `source_ref`
- Assert `finish_run(..., error='corpus_not_found')` sets `error_code` and `completed_at`
- Assert `has_open_run` returns the run_id of an unfinished row and `None` once finished
- Assert `clear_open_runs` closes every unfinished row and returns the count
- **Migration**: open a database written by the pre-corpus schema and assert the new
  columns are added with `source='web'` and no data loss
- Assert `archive_article` sets status to `'archived'` without deleting the row
- Assert `is_changed` returns `True` for an archived row whose hash matches
- Assert `is_changed` returns `True` when the stored `pipeline_version` differs from
  `config.PIPELINE_VERSION`, in **both** directions (stored higher and stored lower), with
  an identical `content_hash`
- Assert `upsert_article` stamps the current `PIPELINE_VERSION`, not the row's previous one
- Assert the `ALTER TABLE` migration gives pre-existing rows `pipeline_version = 0` and
  that those rows then report as changed
- Assert `drop_all_articles` empties `articles` and leaves `ingestion_runs` intact
- Assert `upsert_article` flips an archived row back to `'active'`
- Assert no method hard-deletes an article row
- Assert `start_run` + `finish_run` creates a complete row in `ingestion_runs`
