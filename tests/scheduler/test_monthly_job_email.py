"""Tests for scheduler/monthly_job.py email mode (Milestone 3). Everything is mocked —
email_reader, login, crawler, extractor, ingest_article — with a real file MetadataDB so
the run-row error codes are observable.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import scheduler.monthly_job as mj
from models import ArticleLink, EmailUpdate, RawPage
from storage.metadata_db import MetadataDB


class _Login:
    """Stand-in for scraper.login with the exception classes the scheduler catches."""

    class ManualLoginRequiredError(RuntimeError): ...

    class ManualLoginTimeoutError(RuntimeError): ...

    class LoginStateError(RuntimeError): ...

    def __init__(self):
        self.context_calls = 0
        self.close_calls = 0
        self.raise_on_context = None

    async def get_authenticated_context(self):
        self.context_calls += 1
        if self.raise_on_context:
            raise self.raise_on_context
        return _Ctx()

    async def close_browser(self):
        self.close_calls += 1


class _Ctx:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _Crawler:
    SessionExpiredError = type("SessionExpiredError", (RuntimeError,), {})

    def __init__(self, pages=None, raise_with=None):
        self.pages = pages or []
        self.raise_with = raise_with

    async def fetch_pages(self, context, urls, known_urls):
        if self.raise_with:
            raise self.raise_with
        return self.pages


def _email(url="https://www.example.com/premium-2026-5-10/") -> EmailUpdate:
    return EmailUpdate(
        email_uid="uid-1",
        sender="author@example.com",
        subject="May Premium Report",
        received_at=datetime(2026, 5, 10, tzinfo=UTC),
        article_links=[ArticleLink(title="May Premium Report", url=url)],
        raw_body="body",
    )


def _raw(url="https://www.example.com/premium-2026-5-10/") -> RawPage:
    return RawPage(
        url=url, html="<html>...</html>", fetched_at=datetime.now(UTC), status_code=200, is_new=True
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "meta.db"
    monkeypatch.setattr(mj, "MetadataDB", lambda *a, **k: MetadataDB(str(db_file)))

    reader = type("R", (), {})()
    reader.email = _email()
    reader.raise_with = None
    reader.marked = []

    def read_update_email():
        if reader.raise_with:
            raise reader.raise_with
        return reader.email

    reader.read_update_email = read_update_email
    reader.mark_processed = lambda uid: reader.marked.append(uid)

    login = _Login()
    crawler = _Crawler(pages=[_raw()])

    monkeypatch.setattr("inbox.email_reader.read_update_email", reader.read_update_email)
    monkeypatch.setattr("inbox.email_reader.mark_processed", reader.mark_processed)
    monkeypatch.setattr("scraper.login.get_authenticated_context", login.get_authenticated_context)
    monkeypatch.setattr("scraper.login.close_browser", login.close_browser)
    monkeypatch.setattr("scraper.login.ManualLoginRequiredError", _Login.ManualLoginRequiredError)
    monkeypatch.setattr("scraper.login.ManualLoginTimeoutError", _Login.ManualLoginTimeoutError)
    monkeypatch.setattr("scraper.login.LoginStateError", _Login.LoginStateError)
    monkeypatch.setattr("scraper.crawler.fetch_pages", crawler.fetch_pages)
    monkeypatch.setattr("scraper.crawler.SessionExpiredError", _Crawler.SessionExpiredError)
    monkeypatch.setattr("ingestion.extractor.extract", lambda rp: _Article(rp.url))
    monkeypatch.setattr(mj, "VectorStore", lambda: object())

    ingested = []

    async def fake_ingest(article, context, stats, run_id, force=False, *, db, store):
        ingested.append((article, context))
        stats["new"] += 1

    monkeypatch.setattr(mj, "ingest_article", fake_ingest)

    from types import SimpleNamespace

    return SimpleNamespace(
        db_file=db_file,
        reader=reader,
        login=login,
        crawler=crawler,
        ingested=ingested,
        last_run=lambda: MetadataDB(str(db_file)).get_run_history()[0],
    )


class _Article:
    def __init__(self, url):
        self.url = url
        self.is_stub = False
        self.source = "web"


# ── happy path ─────────────────────────────────────────────────────────────


async def test_fresh_email_scrapes_and_marks_processed(env):
    stats = await mj.run_email_triggered()
    assert stats["new"] == 1 and stats["failed"] == 0
    assert env.reader.marked == ["uid-1"]
    assert env.login.close_calls == 1  # browser closed on the happy path
    assert env.last_run()["error_code"] is None


async def test_no_email_writes_nothing(env):
    env.reader.email = None
    stats = await mj.run_email_triggered()
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    assert env.reader.marked == []
    assert env.login.context_calls == 0  # never authenticated


async def test_ingest_context_is_the_browser_context(env):
    await mj.run_email_triggered()
    _, ctx = env.ingested[0]
    assert isinstance(ctx, _Ctx)  # image_transcriber gets a real context in email mode


# ── failure paths ──────────────────────────────────────────────────────────


async def test_an_article_failure_leaves_the_email_unread(env, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("extract blew up")

    monkeypatch.setattr(mj, "ingest_article", boom)
    stats = await mj.run_email_triggered()
    assert stats["failed"] == 1
    assert env.reader.marked == []  # NOT marked — next poll retries
    assert env.login.close_calls == 1


async def test_mailbox_unreachable_records_the_code_and_never_authenticates(env):
    env.reader.raise_with = _err("MailboxConnectionError")
    stats = await mj.run_email_triggered()
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    assert env.last_run()["error_code"] == "mailbox_unreachable"
    assert env.login.context_calls == 0


async def test_unparseable_email_records_email_unparseable(env):
    env.reader.raise_with = _err("NoLinksFoundError")
    await mj.run_email_triggered()
    assert env.last_run()["error_code"] == "email_unparseable"


async def test_login_required_aborts_before_the_fetch_loop(env):
    env.login.raise_on_context = _Login.ManualLoginRequiredError("no tty")
    stats = await mj.run_email_triggered()
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    assert env.last_run()["error_code"] == "login_required"
    assert env.ingested == []
    assert env.login.close_calls == 0  # no context was created


async def test_login_timeout_records_login_timeout(env):
    env.login.raise_on_context = _Login.ManualLoginTimeoutError("5 min")
    await mj.run_email_triggered()
    assert env.last_run()["error_code"] == "login_timeout"


async def test_session_expired_records_session_error_and_closes_browser(env, monkeypatch):
    monkeypatch.setattr(env.crawler, "raise_with", _Crawler.SessionExpiredError("precheck"))
    await mj.run_email_triggered()
    assert env.last_run()["error_code"] == "session_error"
    assert env.login.close_calls == 1  # context WAS created — must be torn down


async def test_login_state_error_records_login_state_error(env, monkeypatch):
    monkeypatch.setattr(env.crawler, "raise_with", _Login.LoginStateError("weird"))
    await mj.run_email_triggered()
    assert env.last_run()["error_code"] == "login_state_error"
    assert env.login.close_calls == 1


# ── dry run / guard ────────────────────────────────────────────────────────


async def test_dry_run_does_not_mark_or_write(env):
    stats = await mj.run_email_triggered(dry_run=True)
    assert env.reader.marked == []
    assert env.ingested == []
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}


async def test_open_run_blocks_an_email_run(env):
    db = MetadataDB(str(env.db_file))
    db.start_run("email")
    db.close()
    stats = await mj.run_email_triggered()
    assert stats == {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    assert env.login.context_calls == 0


# ── corpus mode never touches the browser ──────────────────────────────────


async def test_corpus_mode_never_calls_close_browser(env, tmp_path, monkeypatch):
    import scraper.login as real_login

    calls = {"n": 0}

    async def counting_close():
        calls["n"] += 1

    monkeypatch.setattr(real_login, "close_browser", counting_close)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    await mj.run_corpus_sync(str(corpus))  # empty corpus, aborts early
    assert calls["n"] == 0


def _err(name: str) -> Exception:
    return type(name, (Exception,), {})(name)
