"""Tests for logger.py — JSON shape, the extra= passthrough, idempotent setup, and the
fail-safe (a broken handler must not break the caller).
"""

from __future__ import annotations

import json
import logging

import pytest

import logger as logger_module
from logger import JsonFormatter, configure_logging, get_logger

_ROOT = "knowledge_repo"


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Each test starts and ends with the knowledge_repo logger bare."""
    root = logging.getLogger(_ROOT)
    saved = root.handlers[:]
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(saved)
    logging.raiseExceptions = True


def _record(msg: str = "hi", **extra) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="knowledge_repo.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(rec, key, value)
    return rec


# ── JsonFormatter ───────────────────────────────────────────────────────────


class TestJsonFormatter:
    def test_base_fields(self):
        out = json.loads(JsonFormatter().format(_record("scraped 24 articles")))
        assert out["level"] == "INFO"
        assert out["logger"] == "knowledge_repo.test"
        assert out["msg"] == "scraped 24 articles"
        assert "ts" in out

    def test_extra_kwargs_are_attached(self):
        out = json.loads(JsonFormatter().format(_record(url="https://x/a", word_count=42)))
        assert out["url"] == "https://x/a"
        assert out["word_count"] == 42

    def test_reserved_record_fields_are_not_leaked(self):
        """The regression guard: a naive 'anything not on the class' check dumps
        funcName / lineno / process / etc. into every line. The payload holds only the
        four base fields plus whatever came via extra=."""
        out = json.loads(JsonFormatter().format(_record("hi")))
        assert set(out) == {"ts", "level", "logger", "msg"}
        for noise in (
            "funcName",
            "lineno",
            "process",
            "thread",
            "threadName",
            "pathname",
            "filename",
            "module",
            "args",
            "created",
            "msecs",
            "relativeCreated",
            "name",
            "levelno",
        ):
            assert noise not in out

    def test_exception_block(self):
        try:
            raise ValueError("bad url")
        except ValueError:
            import sys

            rec = _record("extraction failed")
            rec.exc_info = sys.exc_info()
        out = json.loads(JsonFormatter().format(rec))
        assert out["exception"]["type"] == "ValueError"
        assert out["exception"]["message"] == "bad url"
        assert isinstance(out["exception"]["traceback"], list)

    def test_non_json_extra_is_coerced(self):
        out = json.loads(JsonFormatter().format(_record(path=object())))
        assert isinstance(out["path"], str)  # default=str, never a crash


# ── configure_logging ───────────────────────────────────────────────────────


class TestConfigureLogging:
    def test_adds_two_handlers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logger_module, "LOG_FILE", str(tmp_path / "logs" / "kr.log"))
        configure_logging()
        root = logging.getLogger(_ROOT)
        kinds = {type(h).__name__ for h in root.handlers}
        assert "StreamHandler" in kinds
        assert "RotatingFileHandler" in kinds

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logger_module, "LOG_FILE", str(tmp_path / "logs" / "kr.log"))
        configure_logging()
        configure_logging()
        configure_logging()
        assert len(logging.getLogger(_ROOT).handlers) == 2

    def test_creates_log_dir_and_writes_json_lines(self, tmp_path, monkeypatch):
        log_path = tmp_path / "nested" / "logs" / "kr.log"
        monkeypatch.setattr(logger_module, "LOG_FILE", str(log_path))
        configure_logging()

        get_logger("ingestion.chunker").info("chunking complete", extra={"total_chunks": 7})
        for h in logging.getLogger(_ROOT).handlers:
            h.flush()

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["logger"] == "knowledge_repo.ingestion.chunker"
        assert obj["total_chunks"] == 7

    def test_a_broken_handler_does_not_raise_to_the_caller(self, tmp_path, monkeypatch):
        """A handler whose underlying sink fails (disk full, closed stream) must be
        swallowed — configure_logging() sets logging.raiseExceptions = False, so the
        stdlib handleError path stays silent instead of re-raising."""
        monkeypatch.setattr(logger_module, "LOG_FILE", str(tmp_path / "logs" / "kr.log"))
        configure_logging()

        class BrokenStream:
            def write(self, *_a):
                raise RuntimeError("disk full")

            def flush(self):
                pass

        logging.getLogger(_ROOT).addHandler(logging.StreamHandler(BrokenStream()))
        get_logger("scraper.crawler").error("boom", extra={"url": "x"})  # must not raise

    def test_configure_logging_disables_raise_exceptions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logger_module, "LOG_FILE", str(tmp_path / "logs" / "kr.log"))
        logging.raiseExceptions = True
        configure_logging()
        assert logging.raiseExceptions is False


# ── get_logger ──────────────────────────────────────────────────────────────


def test_get_logger_namespacing():
    assert get_logger("scraper.crawler").name == "knowledge_repo.scraper.crawler"
    assert get_logger(__name__).name.startswith("knowledge_repo.")
