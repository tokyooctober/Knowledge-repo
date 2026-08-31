"""Tests for inbox/email_reader.py — anchor-phrase URL extraction and domain validation.

`.eml` bodies are inline strings parsed by `parse_eml`; the mailbox search is patched with
`_RawEmail` lists. No real Gmail / IMAP connection.
"""

from __future__ import annotations

import pytest

import inbox.email_reader as er
from inbox.email_reader import (
    InvalidDomainError,
    NoLinksFoundError,
    SenderMismatchError,
    read_update_email,
)

SENDER = "author@example.com"
DOMAIN = "example.com"


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.setattr(er, "TRUSTED_SENDER", SENDER)
    monkeypatch.setattr(er, "SITE_DOMAIN", DOMAIN)
    monkeypatch.setattr(er, "EMAIL_SUBJECT_PATTERN", "premium report")
    monkeypatch.setattr(er, "require_phase2_config", lambda: None)


def _eml(
    body: str, *, sender=SENDER, subject="May 2026 Premium Report", ctype="text/plain"
) -> bytes:
    headers = (
        f"From: {sender}\r\n"
        f"Subject: {subject}\r\n"
        "Date: Mon, 10 May 2026 09:00:00 +0000\r\n"
        f"Content-Type: {ctype}; charset=utf-8\r\n"
    )
    return (headers + "\r\n" + body).encode()


PLAIN = """\
Hello,

A new premium report is available in the members area:
https://www.example.com/members/

Here's a direct link to the report:
https://www.example.com/premium-2026-5-10/

Best regards, The Author
"""

INLINE = "Here's a direct link to the report: https://www.example.com/premium-2026-5-10/"

HTML = """\
<p>A new premium report is available in the members area:
<a href="https://www.example.com/members/">members</a></p>
<p>Here's a direct link to the report:</p>
<p><a href="https://www.example.com/premium-2026-5-10/">read it</a></p>
"""


def _install(monkeypatch, *emls):
    raws = [er.parse_eml(e, f"uid{i}") for i, e in enumerate(emls)]
    monkeypatch.setattr(er, "_search", lambda: raws)
    return raws


# ── happy paths ────────────────────────────────────────────────────────────


def test_plain_text_next_line_fallback(monkeypatch):
    _install(monkeypatch, _eml(PLAIN))
    update = read_update_email()
    assert update.article_links[0].url == "https://www.example.com/premium-2026-5-10/"
    assert update.article_links[0].title == "May 2026 Premium Report"
    assert update.sender == SENDER


def test_inline_same_line_capture(monkeypatch):
    _install(monkeypatch, _eml(INLINE))
    assert read_update_email().article_links[0].url == "https://www.example.com/premium-2026-5-10/"


def test_html_body_fallback(monkeypatch):
    _install(monkeypatch, _eml(HTML, ctype="text/html"))
    assert read_update_email().article_links[0].url == "https://www.example.com/premium-2026-5-10/"


def test_members_only_email_raises_no_links(monkeypatch):
    body = "A new report is in the members area:\nhttps://www.example.com/members/\n\nThanks."
    _install(monkeypatch, _eml(body))
    with pytest.raises(NoLinksFoundError):
        read_update_email()


def test_no_anchor_phrase_raises_no_links(monkeypatch):
    _install(monkeypatch, _eml("Just some text with https://www.example.com/premium-x/ in it."))
    with pytest.raises(NoLinksFoundError):
        read_update_email()


# ── validation ─────────────────────────────────────────────────────────────


def test_wrong_domain_raises(monkeypatch):
    body = "Here's a direct link to the report: https://attacker.com/premium-2026-5-10/"
    _install(monkeypatch, _eml(body))
    with pytest.raises(InvalidDomainError):
        read_update_email()


def test_lookalike_domain_fails_the_dot_boundary(monkeypatch):
    body = "Here's a direct link to the report: https://notexample.com/premium-x/"
    _install(monkeypatch, _eml(body))
    with pytest.raises(InvalidDomainError):
        read_update_email()


def test_http_not_https_is_rejected(monkeypatch):
    body = "Here's a direct link to the report: http://www.example.com/premium-x/"
    _install(monkeypatch, _eml(body))
    with pytest.raises(InvalidDomainError):
        read_update_email()


def test_sender_mismatch_raises(monkeypatch):
    _install(monkeypatch, _eml(INLINE, sender="spoofer@evil.com"))
    with pytest.raises(SenderMismatchError):
        read_update_email()


def test_sender_with_display_name_is_accepted(monkeypatch):
    _install(monkeypatch, _eml(INLINE, sender=f"The Author <{SENDER}>"))
    assert read_update_email() is not None


# ── selection ──────────────────────────────────────────────────────────────


def test_no_candidates_returns_none(monkeypatch):
    monkeypatch.setattr(er, "_search", lambda: [])
    assert read_update_email() is None


def test_subject_not_matching_the_regex_is_dropped(monkeypatch):
    _install(monkeypatch, _eml(INLINE, subject="Weekly Market Note"))
    assert read_update_email() is None


def test_most_recent_of_several_is_used(monkeypatch, caplog):
    import logging

    old = _eml("Here's a direct link to the report: https://www.example.com/premium-old/")
    new = _eml("Here's a direct link to the report: https://www.example.com/premium-new/")
    raws = [er.parse_eml(old, "old"), er.parse_eml(new, "new")]
    raws[0].received_at = raws[1].received_at.replace(year=2025)
    monkeypatch.setattr(er, "_search", lambda: raws)
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.inbox.email_reader"):
        update = read_update_email()
    assert update.article_links[0].url.endswith("premium-new/")
    assert any("Multiple emails" in r.message for r in caplog.records)


def test_trailing_punctuation_is_stripped_from_the_url(monkeypatch):
    body = "Here's a direct link to the report: https://www.example.com/premium-x/>."
    _install(monkeypatch, _eml(body))
    assert read_update_email().article_links[0].url == "https://www.example.com/premium-x/"


# ── mark_processed / logging hygiene ───────────────────────────────────────


def test_mark_processed_swallows_backend_errors(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(er, "EMAIL_BACKEND", "imap")
    monkeypatch.setattr(er, "_imap_mark", lambda uid: (_ for _ in ()).throw(OSError("down")))
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.inbox.email_reader"):
        er.mark_processed("uid0")  # must not raise
    assert any("Mark-as-read failed" in r.message for r in caplog.records)


def test_app_password_never_appears_in_logs(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(er, "IMAP_APP_PASSWORD", "hunter2-secret-app-pw")
    _install(monkeypatch, _eml(INLINE))
    with caplog.at_level(logging.DEBUG, logger="knowledge_repo.inbox.email_reader"):
        read_update_email()
    assert not any("hunter2" in r.getMessage() + str(r.__dict__) for r in caplog.records)


# ── more parsing paths ─────────────────────────────────────────────────────


def test_url_two_lines_below_the_anchor(monkeypatch):
    body = "Here's a direct link to the report:\n\nhttps://www.example.com/premium-x/\n"
    _install(monkeypatch, _eml(body))
    assert read_update_email().article_links[0].url == "https://www.example.com/premium-x/"


def test_multipart_prefers_plain_text(monkeypatch):
    raw = (
        b"From: author@example.com\r\nSubject: May Premium Report\r\n"
        b'Content-Type: multipart/alternative; boundary="B"\r\n\r\n'
        b"--B\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Here's a direct link to the report: https://www.example.com/premium-plain/\r\n"
        b"--B\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        b"<p>Here's a direct link to the report: "
        b'<a href="https://www.example.com/premium-html/">x</a></p>\r\n--B--\r\n'
    )
    monkeypatch.setattr(er, "_search", lambda: [er.parse_eml(raw, "m")])
    assert read_update_email().article_links[0].url == "https://www.example.com/premium-plain/"


# ── IMAP backend (imaplib mocked) ──────────────────────────────────────────


class _FakeIMAP:
    instances: list = []

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.stored: list = []
        _FakeIMAP.instances.append(self)

    def login(self, user, pw):
        if pw == "wrong":
            import imaplib

            raise imaplib.IMAP4.error("bad creds")

    def select(self, folder):
        return ("OK", [b"1"])

    def search(self, charset, *criteria):
        self.last_criteria = criteria
        return ("OK", [b"7"])

    def fetch(self, uid, spec):
        return ("OK", [(b"7 (RFC822 {n}", _eml(INLINE))])

    def store(self, uid, flags, value):
        self.stored.append((uid, flags, value))

    def logout(self):
        pass


@pytest.fixture
def fake_imap(monkeypatch):
    _FakeIMAP.instances.clear()
    monkeypatch.setattr(er, "EMAIL_BACKEND", "imap")
    monkeypatch.setattr(er, "IMAP_APP_PASSWORD", "app-pw")
    monkeypatch.setattr("imaplib.IMAP4_SSL", _FakeIMAP)
    return _FakeIMAP


def test_imap_search_and_extract(fake_imap, monkeypatch):
    update = read_update_email()
    assert update.article_links[0].url == "https://www.example.com/premium-2026-5-10/"
    crit = fake_imap.instances[0].last_criteria
    assert "UNSEEN" in crit and "FROM" in crit and "SUBJECT" in crit


def test_imap_connection_refused_raises(monkeypatch):
    monkeypatch.setattr(er, "EMAIL_BACKEND", "imap")

    def boom(host, port):
        raise OSError("connection refused")

    monkeypatch.setattr("imaplib.IMAP4_SSL", boom)
    with pytest.raises(er.MailboxConnectionError):
        read_update_email()


def test_imap_bad_credentials_raises(monkeypatch):
    monkeypatch.setattr(er, "EMAIL_BACKEND", "imap")
    monkeypatch.setattr(er, "IMAP_APP_PASSWORD", "wrong")
    monkeypatch.setattr("imaplib.IMAP4_SSL", _FakeIMAP)
    with pytest.raises(er.MailboxAuthError):
        read_update_email()


def test_imap_mark_processed_sets_seen(fake_imap):
    er.mark_processed("7")
    assert any("\\Seen" in str(s) for inst in fake_imap.instances for s in inst.stored)
