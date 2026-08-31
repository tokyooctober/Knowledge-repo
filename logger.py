"""The one logging configuration. Every module does `log = get_logger(__name__)` and
nothing else — no component configures a logger of its own.

Two handlers on the `knowledge_repo` logger: stderr (human-readable) and a rotating
`logs/knowledge_repo.log` (one JSON object per line). Logging never raises out to the
pipeline — `logging.raiseExceptions` is turned off in `configure_logging()`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import traceback
from datetime import UTC, datetime
from pathlib import Path

from config import LOG_FILE, LOG_LEVEL

# Instance attributes every LogRecord carries. Anything on a record that is NOT in here
# arrived via a `log.info(..., extra={...})` call and belongs in the JSON payload.
_RESERVED_KEYS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for `grep | jq` log queries."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_KEYS:
                payload[key] = value
        if record.exc_info:
            exc_type, exc_val, _ = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_val),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        return json.dumps(payload, default=str)


HUMAN_FMT = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
HUMAN_DATEFMT = "%Y-%m-%d %H:%M:%S"

_ROOT_NAME = "knowledge_repo"


def configure_logging() -> None:
    """Call once at an application entry point (app.py, monthly_job.py, a CLI).

    Idempotent: a second call is a no-op. Adds a stderr handler (human-readable) and a
    rotating JSON file handler (10 MB × 5).
    """
    logging.raiseExceptions = False  # a broken handler must never break the pipeline

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if root.handlers:
        return

    console = logging.StreamHandler()  # stderr
    console.setFormatter(logging.Formatter(HUMAN_FMT, HUMAN_DATEFMT))
    root.addHandler(console)

    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return the child logger `knowledge_repo.<name>`. Call at module top:

    log = get_logger(__name__)
    """
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
