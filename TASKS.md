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

- [x] **M1.6 — `storage/vector_store.py`**
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

- [x] **M1.7 — Milestone 1 gate** — `ruff check` + `ruff format --check` clean, 99 tests
  green, coverage 99.8% overall (models & metadata_db 100%, vector_store 99%). Decisions
  fed back into the specs below.
  - Acceptance: `ruff check .` and `ruff format --check .` clean. `pytest` green.
    Coverage: `models.py` and `storage/metadata_db.py` ≥ 95 %; the milestone's modules
    ≥ 85 % overall. A one-paragraph note in the PR / commit on anything the specs left
    genuinely ambiguous that you had to decide (feed it back into the spec).
  - Verify: `ruff check . && ruff format --check . && pytest --cov=models --cov=config --cov=logger --cov=storage --cov-report=term-missing`.
  - Files: none (CI/gate only) — see `SPEC_ci` work in a later milestone for automating it.

---

## Milestone 2 — Phase 1 pipeline (corpus → queryable)

Build steps 5–12. Delivers a system that ingests the markdown corpus and answers cited
questions against it — **no browser, no mailbox, no site credentials**. Every LLM call is
mocked in the suite; real providers appear only in the integration checks the operator
runs.

```
M2.1 llm_provider ─┬─► M2.3 embedder ──┐
                   ├─► M2.4 image_transcriber (local-path only)
                   │                    ├─► M2.5 monthly_job (corpus mode)
M2.2 md_loader ────┤                    │
M2.3b chunker ─────┴────────────────────┘
                                        └─► M2.6 retriever ─► M2.7 answerer ─► M2.8 app
```
`md_loader` and `chunker` are pure and independent — buildable in parallel with
`llm_provider`. `mock providers` land in `tests/conftest.py` as part of M2.1.

> **Before M2.1**: the installed SDKs are newer majors than the specs' code samples
> (`anthropic` 1.2, `openai` 3.6, `qdrant-client` 1.19, `sentence-transformers` 6,
> `langchain-text-splitters` 1.1, `trafilatura` 2.2, `httpx`→`httpx2`). For each module,
> open the spec, then confirm the exact call against the installed package before writing
> it. Record any drift in the Decisions log and patch the spec.

- [x] **M2.1 — `llm_provider.py` + mock providers**
  - Acceptance: `TextProvider` / `VisionProvider` / `EmbeddingProvider` protocols and the
    concrete Anthropic + OpenAI-compat + local classes from `SPEC_llm_provider.md`, with
    the `__init__`s the review added (vision reads `VISION_*`, never `LLM_*`). Factories
    `get_text_provider` / `get_vision_provider` / `get_embedding_provider` cache a
    singleton and raise `ConfigError` on an unknown backend, `VisionNotSupportedError`
    when `supports_vision` is False. `ProviderConnectionError` / `ModelNotFoundError`
    defined here. `tests/conftest.py` gains `MockTextProvider`, `MockVisionProvider`,
    `MockEmbeddingProvider` (deterministic vectors from a hash of the text).
  - Verify: `tests/test_llm_provider.py` — every bullet in `SPEC_llm_provider.md` §
    Testing Notes, with `anthropic.Anthropic` and `openai.OpenAI` patched; never a real
    call. Factory returns the right class per backend; singleton is reused; `"openai"` and
    `"openai_compat"` both map to the OpenAI-compat class; vision `supports_vision` list
    check works.
  - Files: `llm_provider.py`, `tests/test_llm_provider.py`, `tests/conftest.py`.

- [x] **M2.2 — `ingestion/md_loader.py`**
  - Acceptance: `iter_article_paths`, `load_article`, `load_corpus` per
    `SPEC_md_loader.md` — frontmatter contract (empty string counts as absent;
    `author or AUTHOR_NAME`), whole-relative-path image resolution (never basename),
    dedup by resolved path, caption rejection (URL-as-title, bare-date italic),
    GFM-table extraction verbatim, `models.content_hash` / `models.canonical_url` (never
    reimplemented), stub → `Article(is_stub=True)` with a valid hash, unparseable → `None`.
    `CorpusNotFoundError` / `CorpusEmptyError`.
  - Verify: `tests/ingestion/test_md_loader.py` against `tests/fixtures/corpus/` — the
    full fixture list in the spec, **including two articles whose image subfolders each
    hold a `00-*.jpg` with different bytes** (the basename-resolution regression) and the
    parity test (same fixture as markdown and as HTML through a stub `extract()` produce
    the same `Article` field set).
  - Files: `ingestion/md_loader.py`, `tests/ingestion/test_md_loader.py`,
    `tests/fixtures/corpus/**`.

- [x] **M2.3 — `ingestion/chunker.py`**
  - Acceptance: `chunk_article(article, image_transcriptions=None)` per
    `SPEC_chunker.md` — body chunks via `RecursiveCharacterTextSplitter` (512/64,
    `tiktoken` length), one chunk per table (never split), one per non-skipped
    transcription, `MIN_CHUNK_WORDS` filter, `chunk_id = f"{url_hash}_{type[0]}_{i:04d}"`,
    `total_chunks` set on all, `[]` for a stub.
  - Verify: `tests/ingestion/test_chunker.py` — every § Testing Notes bullet: content-type
    prefixes, oversized-table WARNING-but-kept, `chunk_article(..., None)` yields only
    body+table, `total_chunks` == sum.
  - Files: `ingestion/chunker.py`, `tests/ingestion/test_chunker.py`.

- [x] **M2.4 — `ingestion/embedder.py`**
  - Acceptance: `embed_chunks(chunks) -> list[EmbeddedChunk]` (batched by `BATCH_SIZE`,
    `model_name` from the provider), `embed_query(text) -> list[float]` (applies
    `provider.query_prefix`, truncates over the token limit with a WARNING). No SDK
    imports — provider via `llm_provider.get_embedding_provider()` only. `[]` in → `[]`
    out.
  - Verify: `tests/ingestion/test_embedder.py` with `MockEmbeddingProvider` — output
    length == input, `model_name` matches, provider called once per batch not per chunk,
    `embed_query` applies the prefix.
  - Files: `ingestion/embedder.py`, `tests/ingestion/test_embedder.py`.

- [x] **M2.5 — `ingestion/image_transcriber.py` (local-path path only)**
  - Acceptance: `transcribe_images(article, browser_context=None)` and
    `count_uncached(images)` per `SPEC_image_transcriber.md` — **exactly one
    `ImageTranscription` per `article.images` entry, in order, never shorter**; branch on
    `ImageRef.local_path` (set → read from disk, no httpx); `data/image_cache.db`
    `source_key` cache (`file:path:mtime:size`), cache hit skips the vision call when the
    recorded `VISION_MODEL` matches; type detection + `TRANSCRIBE_TYPES` gate;
    `media_type` from the Pillow-normalised file; `MAX_IMAGES_PER_ARTICLE` cap still emits
    an entry per over-cap image; missing local file → `skipped=True, "unavailable"`.
    **The Phase-2 download path is stubbed/deferred to M3** — a paywall image with
    `browser_context=None` returns `skipped=True, "no_browser_context"`.
  - Verify: `tests/ingestion/test_image_transcriber.py` with `MockVisionProvider` and
    fixture images (chart PNG, table PNG, photo JPG, tracking-pixel PNG) — `len(result)
    == len(article.images)` on every path; local read makes zero HTTP calls; cache key
    changes with file bytes; a cached transcription under a different `VISION_MODEL` is
    not reused; `count_uncached` touches no API.
  - Files: `ingestion/image_transcriber.py`, `tests/ingestion/test_image_transcriber.py`,
    `tests/fixtures/images/**`.

- [ ] **M2.6 — `scheduler/monthly_job.py` (corpus mode only)**
  - Acceptance: `run_corpus_sync(corpus_dir, dry_run, only, limit, force)`,
    `ingest_article(article, context=None, stats, run_id, force)`, `run_inspect`,
    `run_reset(assume_yes)`, `run_prune(dry_run, force)` per `SPEC_monthly_job.md` — the
    shared sub-pipeline (stub guard → `is_changed` → transcribe → chunk → embed →
    `delete_by_url` before `upsert` → `metadata_db.upsert_article`), `--only` abort on an
    unknown stem, `--force` skips `is_changed` but not the stub check, `--reset` order
    (drop collection → drop rows) and cache-preservation, `--prune` >50% guard,
    `finish_run` on every exit path incl. `except BaseException`. **No Playwright import,
    no `BrowserContext`, no email code path** — the email-mode functions are stubbed with
    a `NotImplementedError("Milestone 3")` and a CLI that rejects `--once` / bare invoke.
  - Verify: `tests/scheduler/test_monthly_job.py` — the **Corpus mode**, **Prune**,
    **Selection/force/reset**, and **Both modes** (concurrency guard, crash → finish_run)
    bullet groups from § Testing Notes, all with `md_loader` / `vector_store` /
    `metadata_db` / `image_transcriber` mocked. Assert `login` is never imported and
    `transcribe_images` is called with `context=None`.
  - Files: `scheduler/monthly_job.py`, `tests/scheduler/test_monthly_job.py`.

- [ ] **M2.7 — `query/retriever.py`**
  - Acceptance: `retrieve(query, top_k, filters)` per `SPEC_retriever.md` — `embed_query`
    → `vector_store.search(top_k*2, filters)` → discard `< MIN_SCORE_THRESHOLD` →
    `MAX_CHUNKS_PER_ARTICLE` cap → trim to `top_k`, sorted by score desc. `ValueError` on
    empty query, truncate over-long query with a WARNING, `ModelMismatchError` if
    `vector_store.recorded_model()` ≠ the configured embedding model. Query rewriting /
    hybrid stay behind their `ENABLE_*` flags (off).
  - Verify: `tests/query/test_retriever.py` — mock `embed_query` + an in-memory
    `VectorStore`; assert ordering, threshold exclusion, per-article cap, empty-query
    raise, and the model-mismatch guard.
  - Files: `query/retriever.py`, `tests/query/test_retriever.py`.

- [ ] **M2.8 — `query/answerer.py`**
  - Acceptance: `answer(query, results) -> Answer` per `SPEC_answerer.md` — empty results
    → graceful "not found" with **no** provider call; cap at `MAX_CONTEXT_CHUNKS` +
    truncate at `MAX_CHUNK_CHARS`; system + user message shape; `provider.complete`;
    `[N]` citation parse → `Source` list (explicit field mapping, out-of-range dropped
    with a WARNING); token counts passed through. No SDK import.
  - Verify: `tests/query/test_answerer.py` with `MockTextProvider` — `sources` only holds
    cited & in-range indices; empty results path never calls the provider; the 20→6 cap;
    `Answer.model` / token counts come from the `TextResponse`.
  - Files: `query/answerer.py`, `tests/query/test_answerer.py`.

- [ ] **M2.9 — `app.py`**
  - Acceptance: CLI (`query`, `--top-k`, `--tags`, `--date-after/before`, `--json`,
    `--sync-corpus`, `--check-email` → "Phase 2 not built", `--stats`, `--dry-run`) and
    the Streamlit UI per `SPEC_app.md` — `retrieve` → `answer` → render with `[N]`
    citations and source cards; `local:` ids rendered as plain text; sidebar stats by
    source; the "Check email" button disabled when `phase2_configured()` is False;
    background-thread ingestion guarded by `metadata_db.has_open_run()`.
  - Verify: `tests/test_app.py` — CLI `--stats` and `--json` (valid JSON → `Answer`),
    unknown `--tags` returns a graceful empty answer, empty-KB message names the corpus
    path; Streamlit via `streamlit.testing` or mocked `st.*`. One integration test:
    seeded in-memory Qdrant + `MockTextProvider`, a known question returns the right
    article as top source.
  - Files: `app.py`, `tests/test_app.py`.

- [ ] **M2.10 — Milestone 2 gate**
  - Acceptance: `ruff` clean; `pytest` green; coverage ≥ 85 % overall, ≥ 95 % for
    `md_loader.py` and `chunker.py`. **SPEC.md § Success Criteria "Phase 1" holds**: a
    `--corpus --dry-run` over a fixture corpus is clean; a second `--corpus` run is all
    skips with zero `vector_store.upsert`; `app.py "<question>"` cites the right article;
    the suite passes with no Playwright / mailbox / credentials. Decisions log updated.
  - Verify: `ruff check . && ruff format --check . && pytest --cov --cov-fail-under=85`.
  - Files: none (gate only).

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

- **M2.4** — `embed_query` truncates at `CHUNK_SIZE` (512) tokens — the local BGE/nomic
  models' window and already the configured budget; no separate query-limit constant.
- **M2.5** — `transcribe_images` is `async` (for the M3 download path) but the corpus
  branch does no awaiting. The image cache is its own sqlite db (`data/image_cache.db`),
  opened per call. The Phase-2 download raises `NotImplementedError` — unreachable in M2
  since corpus refs always have `local_path`; a web ref with `browser_context=None` is
  short-circuited to `skipped="no_browser_context"` before that. `media_type` is derived
  from the cached file's format (`image/png`, or `image/jpeg` when the source is JPEG).
- **M2.2** — the fixture corpus is built in `tmp_path` by a `write_article()` helper
  (real image bytes, no binaries in git) rather than checked in under
  `tests/fixtures/corpus/`. The spec's **markdown↔HTML parity test** needs
  `extractor.extract()` and is deferred to M3, where the extractor lands. `_looks_like_date`
  rejects a caption only when the *whole* string parses as a date and is ≤ 30 chars, so a
  long descriptive caption starting with a month name is still kept.
- **M2.1** — installed SDK majors (`anthropic` 1.2, `openai` 3.6) still expose the calls
  the spec's code uses (`messages.create(system=…)`, `chat.completions.create`,
  `.usage.*`), so the provider code is unchanged from the spec. Additions: each provider
  lazy-imports its SDK in `__init__` (keeps `import llm_provider` cheap and honours
  "import only what the backend needs"); rate-limit retry uses a predicate on
  `type(exc).__name__ == "RateLimitError"` rather than importing both SDKs' exception
  classes at module load; `TextResponse` lives in `llm_provider.py` (not `models.py`) as
  the spec has it; `_reset_providers_for_tests()` clears the singletons for tests.
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
- **M1.6** — `DISTANCE_METRIC` / `ON_DISK_PAYLOAD` / `HNSW_M` / `HNSW_EF_CONSTRUCT` are now
  module constants in `vector_store.py`, not `config.py` (`DISTANCE_METRIC` is a
  `qdrant_client` enum; `config` must stay import-light). `SPEC_vector_store.md` updated.
- **M1.6** — qdrant-client **1.19** dropped `client.search()`; the module uses
  `client.query_points(...).points`. The embedding model name isn't known at `__init__`
  (it rides on `EmbeddedChunk`), so it is recorded lazily on the first `upsert` onto a
  sentinel point and checked on every later `upsert`; `recorded_model()` exposes it for
  the retriever. `__init__` only guards the vector **dimension**. Sentinel is filtered out
  of `count()` and `search()`. `SPEC_vector_store.md` Core Logic + Public Interface updated.
- **M1.6** — date filters use `models.DatetimeRange` (payload stores `published_at` as an
  ISO-8601 string); `SearchResult.published_at` is parsed back with
  `datetime.fromisoformat`.

