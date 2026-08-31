# Tasks

Derived from [SPEC.md](SPEC.md) § *Build and run order*. Work top to bottom — tasks are
dependency-ordered, not priority-ordered. Each task is one focused session, touches ≤ 5
files, and is not done until its **Verify** step passes.

Convention: check a box only when **Acceptance** holds *and* **Verify** is green *and* the
module spec's own "Testing Notes" are all covered by passing tests (see SPEC.md §
*Testing Strategy*).

---

## Milestone 1 — Shared foundation + storage

Delivers everything the rest of the system imports, with **no LLM API, no browser, no
mailbox**. Qdrant (Docker) is the only external service, and only for `vector_store`.

```
M1.1 ─► M1.2 ─┬─► M1.3 ─► M1.4 ─► M1.5 ─► M1.6 ─► M1.7
              └─ (M1.3 and M1.4 are a short chain, not parallel:
                  config imports ConfigError from models; logger imports LOG_* from config)
```

- [x] **M1.1 — Project scaffolding**
  - Acceptance: `requirements.txt` matches SPEC.md § *Dependencies* (add `pytest-cov`,
    `ruff`). `pyproject.toml` configures `ruff` (line length 100, format + check) and
    `pytest` (`testpaths = ["tests"]`, `asyncio_mode = "auto"`, `--cov` defaults).
    `.env.template` matches SPEC.md § *.env template*. `tests/` tree exists with
    `conftest.py` (empty for now) mirroring the source layout. Python 3.11 pinned in
    `pyproject.toml` (`requires-python = ">=3.11"`).
  - Verify: fresh venv → `pip install -r requirements.txt` succeeds; `ruff check .` exits
    0; `pytest` exits 0 or 5 (no tests yet); `python -c "import sys; assert sys.version_info >= (3,11)"`.
  - Files: `requirements.txt`, `pyproject.toml`, `.env.template`, `tests/conftest.py`,
    `tests/__init__.py` (+ package `__init__.py` under each source dir).

- [x] **M1.2 — `models.py`**
  - Acceptance: every dataclass in SPEC.md § *Shared data types* verbatim (field names,
    order, types), plus `content_hash()`, `canonical_url()`, and the shared exceptions
    `ConfigError`, `ModelMismatchError`, `VisionNotSupportedError`. `from __future__ import
    annotations` at top. No first-party imports — `models.py` is the root of the dep graph.
  - Verify: `tests/test_models.py` — `content_hash` is deterministic, unchanged by a
    `.md` path rename, changed by body text, changed when a referenced image file's
    size/mtime changes, identical for a web image across re-runs; the trailing `:` after
    `img.src` is present for both sources. `canonical_url` collapses `http`→`https`,
    strips `www.`, strips a trailing slash, drops query + fragment, and maps the corpus
    and email spellings of one article to the same string; `""` in → `""` out.
  - Files: `models.py`, `tests/test_models.py`.

- [x] **M1.3 — `config.py`**
  - Acceptance: every constant any module spec names is present (SPEC.md says this
    invariant is worth re-checking — grep the module specs). Phase 2 secrets and
    author-specific values use `os.environ.get(..., "")` / a generic default; `import
    config` must succeed with **no `.env`**. `require_phase2_config()` raises `ConfigError`
    naming each missing setting and runs the `SITE_DOMAIN` cross-check on `LOGIN_URL` /
    `HEALTH_CHECK_URL`. `phase2_configured()` returns a bool and never raises.
  - Verify: `tests/test_config.py` — `import config` with an empty environment does not
    raise; `require_phase2_config()` with nothing set raises `ConfigError` listing
    `LOGIN_URL, TRUSTED_SENDER, SITE_DOMAIN, HEALTH_CHECK_URL`; with all set but
    `LOGIN_URL` on a different host, it raises the cross-check error; with all consistent,
    it returns `None`. `phase2_configured()` is `False` then `True` across those.
  - Files: `config.py`, `tests/test_config.py`.

- [x] **M1.4 — `logger.py`**
  - Acceptance: matches `SPEC_logger.md` — `JsonFormatter` (one JSON object per line,
    `ts`/`level`/`logger`/`msg` + any `extra=` keys + an `exception` block when
    `exc_info`), `HUMAN_FMT` stderr handler, `configure_logging()` idempotent (second
    call is a no-op), `get_logger(name)` returns `knowledge_repo.<name>`. Logging never
    raises out to the caller.
  - Verify: `tests/test_logger.py` — a record with `extra={"url": ...}` serialises to
    JSON containing that key; `exc_info=True` attaches `exception.type` and a traceback
    list; calling `configure_logging()` twice adds no duplicate handlers; a handler that
    fails internally does not propagate.
  - Files: `logger.py`, `tests/test_logger.py`.

- [x] **M1.5 — `storage/metadata_db.py`**
  - Acceptance: `articles` + `ingestion_runs` schema from `SPEC_metadata_db.md` exactly
    (incl. `pipeline_version`, `source`, `source_path`, `status`, the `idx_articles_source`
    index). WAL mode; the lock-retry (3× / 500 ms); the `ALTER TABLE` migration path.
    Every method in the Public Interface, with the documented semantics —
    `is_changed()` returns `True` for a new URL, a changed hash, an archived row, and a
    `pipeline_version` mismatch in **either** direction; `drop_all_articles()` leaves
    `ingestion_runs` intact; `finish_run()` reads only `new/updated/skipped/failed`.
  - Verify: `tests/storage/test_metadata_db.py` against `:memory:` — **every bullet in
    `SPEC_metadata_db.md` § Testing Notes is a passing test**, including the migration
    test (open a pre-corpus-schema DB, assert columns added with `source='web'`, no data
    loss) and "no method hard-deletes an article row".
  - Files: `storage/metadata_db.py`, `storage/__init__.py`, `tests/storage/test_metadata_db.py`.

- [ ] **M1.6 — `storage/vector_store.py`**
  - Acceptance: `VectorStore` per `SPEC_vector_store.md` — collection init with COSINE /
    `EMBEDDING_DIM` / HNSW config, `model_name` recorded on a sentinel and checked on
    reconnect (`ModelMismatchError` on mismatch), `chunk_id`→UUID5 point ids, batched
    `upsert` (`UPSERT_BATCH_SIZE`), `search` returning **raw top-k** (no score or
    per-article filtering — that is the retriever's job) with `chunk_id` on every
    `SearchResult`, `delete_by_url`, `drop_collection` (drops **and recreates empty**),
    `count`, `stats`.
  - Verify: `tests/storage/test_vector_store.py` with `QDRANT_IN_MEMORY=True` — every
    bullet in `SPEC_vector_store.md` § Testing Notes, especially: `search` applies no
    threshold; `delete_by_url` removes exactly that URL's points; `drop_collection` leaves
    `count()==0` and an immediately usable collection; `drop_collection` then re-init with
    a *different* model does not raise.
  - Files: `storage/vector_store.py`, `tests/storage/test_vector_store.py`.

- [ ] **M1.7 — Milestone 1 gate**
  - Acceptance: `ruff check .` and `ruff format --check .` clean. `pytest` green.
    Coverage: `models.py` and `storage/metadata_db.py` ≥ 95 %; the milestone's modules
    ≥ 85 % overall. A one-paragraph note in the PR / commit on anything the specs left
    genuinely ambiguous that you had to decide (feed it back into the spec).
  - Verify: `ruff check . && ruff format --check . && pytest --cov=models --cov=config --cov=logger --cov=storage --cov-report=term-missing`.
  - Files: none (CI/gate only) — see `SPEC_ci` work in a later milestone for automating it.

---

## Milestone 2 — Phase 1 pipeline (corpus → queryable)

Build steps 5–12. Not broken into tasks yet — do that once Milestone 1 is green, since
the exact interfaces may shift when real code meets the specs.

Rough order (from SPEC.md § *Build order*): `llm_provider.py` (mocked) → `md_loader.py` →
`chunker.py` → `embedder.py` → `image_transcriber.py` (local-path first) →
`monthly_job.py` corpus mode → `retriever.py` → `answerer.py` → `app.py`.

Independent once `llm_provider` exists: `md_loader`, `chunker` (pure, no services).

Milestone 2 is done when SPEC.md § *Success Criteria* "Phase 1" holds.

---

## Milestone 3 — Phase 2 (email-triggered scrape)

Build steps 13–17: `email_reader.py` → `login.py` → `crawler.py` → `extractor.py` →
`monthly_job.py` email mode. Strictly additive; do not start until Milestone 2's
Success Criteria hold.

Milestone 3 is done when SPEC.md § *Success Criteria* "Phase 2" holds.

---

## Decisions log

Things the specs left ambiguous that were decided during implementation. Fold the
important ones back into SPEC.md.

- **M1.1** — `ruff` (≥ 0.14) also formats Markdown and would rewrite the deliberate
  column alignment in the spec code blocks. `pyproject.toml` excludes `*.md` / `*.rst`
  from ruff entirely. CI's "fails on a format diff" rule is Python-only.
- **M1.1** — flat layout (modules + packages at repo root) declared in `pyproject.toml`
  via `py-modules` + `packages`; `pip install -e .` makes `import config` etc. work from
  anywhere. Added `.gitattributes` (`eol=lf`) so the repo is byte-identical across
  platforms despite Windows checkouts.
- **M1.1** — toolchain (`ruff`, `pytest`, `pytest-cov`, `pytest-asyncio`) + M1 runtime
  deps (`python-dotenv`, `qdrant-client`) installed and verified. The full
  `requirements.txt` install (torch via `sentence-transformers`, playwright, streamlit —
  none needed before Milestone 2) runs separately; not a blocker for M1.
- **M1.2** — `content_hash` calls `Path.stat()`, so a corpus image that has vanished
  raises `FileNotFoundError` rather than skipping silently. `md_loader` resolves/drops
  missing images before calling it (SPEC_md_loader.md step 4), so this only bites if a
  file disappears mid-run — acceptable as a loud failure. Documented via a test.
- **M1.3** — `phase2_configured()` is implemented as `try require_phase2_config() except
  ConfigError: return False` — one source of truth for "is Phase 2 usable", instead of a
  parallel list of checks that could drift.
- **M1.1** — the full `requirements.txt` install succeeded (exit 0). The resolver pulled
  much newer majors than the specs assume (`openai` 3.x, `anthropic` 1.x, `pandas` 3.x,
  `langchain` 1.x, `httpx` → `httpx2`). No impact on Milestone 1 (stdlib + `sqlite3` +
  `python-dotenv` only), but **`SPEC_llm_provider.md` / `SPEC_extractor.md` code samples
  must be re-checked against the installed SDK versions at the start of Milestone 2.**
- **M1.5** — `MetadataDB` holds one `sqlite3` connection (`check_same_thread=False`, WAL).
  `touch_source_path` is in the spec's Core Operations but was missing from its Public
  Interface block — implemented, and it's the method `monthly_job` already calls. The
  `_execute` lock-retry only retries on an error whose message contains "locked"; any
  other `OperationalError` (bad SQL, missing table) raises immediately.
- **M1.4** — `SPEC_logger.md`'s `JsonFormatter` used `key not in logging.LogRecord.__dict__`
  to spot `extra=` keys — that is the *class* dict (methods only), so every built-in field
  (`funcName`, `lineno`, `process`, …) would leak into every JSON line. Fixed in both the
  code and the spec: build `_RESERVED_KEYS` from a real `LogRecord` instance. Also added
  `logging.raiseExceptions = False` in `configure_logging()` for the documented fail-safe.
- **M1.6 (pending)** — `SPEC_vector_store.md` lists `DISTANCE_METRIC`, `ON_DISK_PAYLOAD`,
  `HNSW_M`, `HNSW_EF_CONSTRUCT` in its config block, but they are **not** in SPEC.md §
  Shared config and `DISTANCE_METRIC = Distance.COSINE` would force `config.py` to import
  `qdrant_client` (which every module then pays for). Decision: keep these four as
  module-level constants in `vector_store.py`; fix `SPEC_vector_store.md` to say so.

