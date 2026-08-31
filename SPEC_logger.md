# `logger.py` — Shared Logging Standard

---
```
module:     logger.py
spec:       SPEC_logger.md
layer:      Shared foundation
depends_on: config.py  (LOG_FILE, LOG_LEVEL)
used_by:    every module in the project
services:   filesystem  (rotating log file)
```
---

## Purpose
Define a single, consistent logging configuration used by every component in the system. All modules import from this file — no component configures its own logger independently.

---

## Design Principles
- **Structured output**: every log line carries a fixed set of fields so logs are grep-able and machine-parseable
- **One logger per component**: each module gets a child logger named after its module path (e.g. `knowledge_repo.scraper.crawler`)
- **Two outputs always**: `stderr` (human-readable) and a rotating file (machine-readable JSON)
- **No sensitive data**: credentials, session tokens, and API keys are never logged — use placeholder strings like `<redacted>`
- **Fail-safe**: logging must never raise an exception that disrupts the pipeline; errors in the logging subsystem are swallowed silently

---

## Log Levels — Usage Contract

| Level | When to use | Examples |
|---|---|---|
| `DEBUG` | Granular operational detail — only useful during development | "Chunk 004 produced 312 tokens", "Cache hit for session state" |
| `INFO` | Normal operational milestones — one or two per major step | "Scraped 24 articles", "Ingestion run started (run_id=abc123)" |
| `WARNING` | Something unexpected but recoverable; pipeline continues | "Article stub skipped (< 100 words)", "Score below threshold — 0 results returned" |
| `ERROR` | A single item failed; pipeline continues with next item | "Failed to extract article https://…: TimeoutError", "Embedding failed for chunk_id xyz" |
| `CRITICAL` | The entire run cannot continue | "Qdrant unreachable", "Authentication failed — aborting ingestion" |

---

## Shared Logger Module (`logger.py`)

```python
import logging
import logging.handlers
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from config import LOG_FILE, LOG_LEVEL

# ── JSON formatter ──────────────────────────────────────────────────────────

# Instance attributes every LogRecord carries. A key on a record that is NOT in here
# arrived via a `log.info(..., extra={...})` call. Build it from a real record — NOT
# from `logging.LogRecord.__dict__`, which is the *class* dict (methods only) and would
# let every built-in field (funcName, lineno, process, …) leak into every line.
_RESERVED_KEYS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line for structured log parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        # Attach any extra= kwargs passed at call site
        for key, value in record.__dict__.items():
            if key not in _RESERVED_KEYS:
                payload[key] = value
        # Attach exception info if present
        if record.exc_info:
            exc_type = record.exc_info[0]
            payload["exception"] = {
                "type":       exc_type.__name__ if exc_type else None,
                "message":    str(record.exc_info[1]),
                "traceback":  traceback.format_exception(*record.exc_info),
            }
        return json.dumps(payload, default=str)


# ── Human-readable formatter ─────────────────────────────────────────────────
HUMAN_FMT = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
HUMAN_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ── Root setup (called once at process start) ────────────────────────────────
def configure_logging() -> None:
    """Call once at application entry point (app.py, monthly_job.py, CLI)."""
    # Fail-safe: a broken handler (disk full, closed stream) must never surface as an
    # exception in the pipeline. The stdlib swallows handler errors when this is False.
    logging.raiseExceptions = False

    root = logging.getLogger("knowledge_repo")
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if root.handlers:
        return  # already configured; idempotent

    # Handler 1: stderr (human-readable)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(HUMAN_FMT, HUMAN_DATEFMT))
    root.addHandler(console)

    # Handler 2: rotating JSON file
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)


# ── Per-module factory ───────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a child logger. Call at module top-level:
        log = get_logger(__name__)
    """
    return logging.getLogger(f"knowledge_repo.{name}")
```

---

## Usage Pattern (every component)

```python
# At module top — one line per file
from logger import get_logger
log = get_logger(__name__)   # e.g. "knowledge_repo.scraper.crawler"

# In functions
log.info("Crawl started", extra={"index_url": INDEX_URL})
log.warning("Stub article skipped", extra={"url": article.url, "word_count": article.word_count})
log.error("Extraction failed", extra={"url": raw_page.url}, exc_info=True)
log.critical("Qdrant unreachable — aborting", extra={"host": QDRANT_HOST, "port": QDRANT_PORT})
```

The `exc_info=True` flag attaches the full traceback to the JSON log entry automatically.

---

## Standard Extra Fields by Component

Each component documents its own extra fields in its spec. The following fields are universally expected:

| Field | Type | Components that set it |
|---|---|---|
| `url` | str | crawler, extractor, md_loader, chunker, embedder, scheduler |
| `md_file` | str | md_loader, scheduler (corpus mode) — the `.md` path being processed |
| `corpus_dir` | str | md_loader, scheduler (corpus mode) |
| `source` | str | scheduler, metadata_db — `"corpus"` or `"web"` |
| `trigger` | str | scheduler — `"corpus"`, `"email"`, `"manual"`, `"prune"` |
| `run_id` | str | scheduler, all components called from scheduler |
| `article_count` | int | crawler, scheduler |
| `chunk_id` | str | chunker, embedder, vector_store |
| `chunk_count` | int | chunker, scheduler |
| `model_name` | str | embedder, vector_store, retriever |
| `score` | float | retriever |
| `query` | str | retriever, answerer |
| `input_tokens` | int | answerer |
| `output_tokens` | int | answerer |
| `sender` | str | email_reader |
| `email_uid` | str | email_reader |
| `error_type` | str | all components on ERROR/CRITICAL |

---

## Config additions (`config.py`)

```python
LOG_FILE   = "logs/knowledge_repo.log"   # rotating JSON log
LOG_LEVEL  = "INFO"                       # DEBUG | INFO | WARNING | ERROR | CRITICAL
```

---

## Log file layout

```
logs/
  knowledge_repo.log        ← current (JSON, one object per line)
  knowledge_repo.log.1      ← previous rotation
  knowledge_repo.log.2
  ...up to .5
```

Add `logs/` to `.gitignore`.

---

## Querying logs

```bash
# All ERRORs from today
grep '"level": "ERROR"' logs/knowledge_repo.log | jq .

# All failures for a specific URL
grep 'example.com/article-7' logs/knowledge_repo.log | jq .

# All ingestion run summaries
grep '"msg": "Ingestion run complete"' logs/knowledge_repo.log | jq '{ts,stats}'

# Corpus syncs only, with their stats
grep '"trigger": "corpus"' logs/knowledge_repo.log | jq '{ts,msg,corpus_dir,stats}'

# Email-triggered runs only
grep '"trigger": "email"' logs/knowledge_repo.log | jq '{ts,msg,email_uid,stats}'

# Corpus files whose frontmatter is missing a url (fix these in the corpus)
grep 'synthesised' logs/knowledge_repo.log | jq '{md_file,synthesised_url}'

# Corpus images referenced but not present on disk
grep '"msg": "Image file not found"' logs/knowledge_repo.log | jq '{md_file,image_path}'

# Last 5 CRITICAL events
grep '"level": "CRITICAL"' logs/knowledge_repo.log | tail -5 | jq .
```
