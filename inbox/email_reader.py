"""Phase 2 entry point: watch a mailbox for the trusted sender's new-report notification,
extract the single direct article URL, and hand it to the scheduler.

**URL extraction only.** The URL is a paywall-protected members link — fetching it without
`scraper/login.py`'s authenticated session returns the login page, not the article. This
module holds no website credentials and performs no website login.

`read_update_email()` does NOT mark the email processed. `mark_processed()` is a separate
call the scheduler makes only after the article is safely in the index — so a failed run
simply retries on the next poll instead of losing the month's report.
"""

from __future__ import annotations

import email
import imaplib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.header import decode_header
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from config import (
    EMAIL_BACKEND,
    EMAIL_SUBJECT_PATTERN,
    GMAIL_CREDENTIALS_FILE,
    GMAIL_TOKEN_FILE,
    IMAP_APP_PASSWORD,
    IMAP_HOST,
    IMAP_PORT,
    IMAP_USERNAME,
    MAILBOX_FOLDER,
    PROCESSED_LABEL,
    SINCE_DAYS,
    SITE_DOMAIN,
    TRUSTED_SENDER,
    require_phase2_config,
)
from logger import get_logger
from models import ArticleLink, EmailUpdate

log = get_logger(__name__)


class MailboxConnectionError(Exception):
    """The mailbox server could not be reached."""


class MailboxAuthError(Exception):
    """The mailbox rejected the credentials."""


class AuthRefreshError(Exception):
    """A Gmail OAuth token could not be refreshed."""


class SenderMismatchError(Exception):
    """The From header does not match TRUSTED_SENDER exactly."""


class NoLinksFoundError(Exception):
    """No article URL after the anchor phrase."""


class InvalidDomainError(Exception):
    """The extracted URL is not https, or not on SITE_DOMAIN."""


_ANCHOR_INLINE = re.compile(
    # anchor phrase then the URL ON THE SAME LINE (horizontal space only after the colon);
    # a URL on a following line falls through to the line-scan step.
    r"here[’'’]s\s+a\s+direct\s+link\s+to\s+the\s+report[ \t]*:?[ \t]*(https?://\S+)",
    re.IGNORECASE,
)
_ANCHOR_LINE = re.compile(r"direct link to the report", re.IGNORECASE)
_ANY_URL = re.compile(r"https?://\S+")


@dataclass
class _RawEmail:
    uid: str
    from_addr: str
    subject: str
    received_at: datetime
    text_body: str
    html_body: str


# ── public interface ───────────────────────────────────────────────────────


def read_update_email() -> EmailUpdate | None:
    """Extract the direct article link from the latest new-report notification email.

    Returns None if no unprocessed qualifying email is present. Does NOT mark anything —
    see mark_processed().

    Raises: MailboxConnectionError, MailboxAuthError, AuthRefreshError,
            SenderMismatchError, NoLinksFoundError, InvalidDomainError.
    """
    require_phase2_config()
    candidates = _search()
    chosen = _pick(candidates)
    if chosen is None:
        log.info(
            "No qualifying email",
            extra={"since_days": SINCE_DAYS, "trusted_sender": TRUSTED_SENDER},
        )
        return None

    log.info(
        "Email fetched",
        extra={"email_uid": chosen.uid, "sender": chosen.from_addr, "subject": chosen.subject},
    )
    if _addr(chosen.from_addr) != _addr(TRUSTED_SENDER):
        log.error(
            "Sender mismatch",
            extra={
                "email_uid": chosen.uid,
                "claimed_sender": chosen.from_addr,
                "trusted_sender": TRUSTED_SENDER,
            },
        )
        raise SenderMismatchError(f"{chosen.from_addr} != {TRUSTED_SENDER}")

    work_text = chosen.text_body or BeautifulSoup(chosen.html_body, "lxml").get_text("\n")
    url = _extract_url(work_text, chosen.html_body, chosen.uid)
    url = _validate_url(url, chosen.uid)

    log.info(
        "URL extracted and validated",
        extra={"email_uid": chosen.uid, "url": url, "domain": urlparse(url).netloc},
    )
    return EmailUpdate(
        email_uid=chosen.uid,
        sender=TRUSTED_SENDER,
        subject=chosen.subject,
        received_at=chosen.received_at,
        article_links=[ArticleLink(title=chosen.subject.strip(), url=url)],
        raw_body=work_text,
    )


def mark_processed(email_uid: str) -> None:
    """Mark one email handled — Gmail applies PROCESSED_LABEL and marks read; IMAP sets
    \\Seen. Called by the scheduler ONLY after the article is in the index."""
    require_phase2_config()
    try:
        if EMAIL_BACKEND == "gmail":
            _gmail_mark(email_uid)
        else:
            _imap_mark(email_uid)
        log.debug(
            "Email marked as processed",
            extra={"email_uid": email_uid, "backend": EMAIL_BACKEND, "label": PROCESSED_LABEL},
        )
    except Exception as exc:  # noqa: BLE001 - a failed mark must not abort; URL is extracted
        log.warning(
            "Mark-as-read failed", extra={"email_uid": email_uid, "error_type": type(exc).__name__}
        )


# ── pure parsing ───────────────────────────────────────────────────────────


def _addr(header_value: str) -> str:
    m = re.search(r"<([^>]+)>", header_value)
    return (m.group(1) if m else header_value).strip().lower()


def _pick(candidates: list[_RawEmail]) -> _RawEmail | None:
    matching = [c for c in candidates if re.search(EMAIL_SUBJECT_PATTERN, c.subject, re.IGNORECASE)]
    dropped = len(candidates) - len(matching)
    if dropped:
        log.debug("Candidates dropped by the client-side subject regex", extra={"count": dropped})
    if not matching:
        return None
    if len(matching) > 1:
        log.warning(
            "Multiple emails found",
            extra={"count": len(matching), "email_uids": [c.uid for c in matching]},
        )
    return max(matching, key=lambda c: c.received_at)


def _extract_url(text: str, html: str, uid: str) -> str:
    if m := _ANCHOR_INLINE.search(text):
        raw = m.group(1)
        log.debug(
            "Anchor phrase found (same-line capture)", extra={"email_uid": uid, "raw_url": raw}
        )
        return _strip_trailing(raw, uid)

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _ANCHOR_LINE.search(line):
            for nxt in lines[i + 1 : i + 6]:
                if u := _ANY_URL.search(nxt):
                    log.debug(
                        "Anchor phrase found (next-line fallback)",
                        extra={"email_uid": uid, "raw_url": u.group(0)},
                    )
                    return _strip_trailing(u.group(0), uid)

    if html:
        soup = BeautifulSoup(html, "lxml")
        anchor = soup.find(string=_ANCHOR_LINE)
        if anchor:
            for el in anchor.find_all_next(["a", "p", "div"], limit=5):
                href = el.get("href") if el.name == "a" else None
                href = href or (el.get_text(strip=True) if el.name != "a" else None)
                if href and href.startswith("https://"):
                    log.debug(
                        "Anchor phrase found (HTML fallback)",
                        extra={"email_uid": uid, "raw_url": href},
                    )
                    return _strip_trailing(href, uid)

    log.error("Anchor phrase not found", extra={"email_uid": uid, "body_preview": text[:500]})
    raise NoLinksFoundError("no URL after the anchor phrase")


def _strip_trailing(url: str, uid: str) -> str:
    cleaned = url.rstrip("_*>).,\"'")
    if cleaned != url:
        log.debug(
            "URL trailing chars stripped", extra={"email_uid": uid, "before": url, "after": cleaned}
        )
    return cleaned


def _validate_url(url: str, uid: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    on_site = host == SITE_DOMAIN or host.endswith("." + SITE_DOMAIN)
    if parsed.scheme != "https" or not on_site:
        log.error(
            "URL domain invalid",
            extra={"email_uid": uid, "extracted_url": url, "expected_domain": SITE_DOMAIN},
        )
        raise InvalidDomainError(f"{url} is not https on {SITE_DOMAIN}")
    return url


def _decode(value: str | None) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    return "".join(
        (p.decode(enc or "utf-8", "replace") if isinstance(p, bytes) else p) for p, enc in parts
    )


def parse_eml(raw: bytes, uid: str = "eml") -> _RawEmail:
    """Parse an RFC 2822 message into a _RawEmail. Shared by the IMAP path and tests."""
    msg = email.message_from_bytes(raw)
    text_body = html_body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and not text_body:
                text_body = part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", "replace"
                )
            elif ctype == "text/html" and not html_body:
                html_body = part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", "replace"
                )
    else:
        payload = msg.get_payload(decode=True).decode(
            msg.get_content_charset() or "utf-8", "replace"
        )
        if msg.get_content_type() == "text/html":
            html_body = payload
        else:
            text_body = payload
    try:
        received = parsedate_to_datetime(msg["Date"]) or datetime.now(UTC)
    except (TypeError, ValueError):
        received = datetime.now(UTC)
    return _RawEmail(
        uid=uid,
        from_addr=_decode(msg["From"]),
        subject=_decode(msg["Subject"]),
        received_at=received,
        text_body=text_body,
        html_body=html_body,
    )


# ── backends ───────────────────────────────────────────────────────────────


def _search() -> list[_RawEmail]:
    if EMAIL_BACKEND == "gmail":
        return _gmail_search()
    return _imap_search()


def _imap_search() -> list[_RawEmail]:
    try:
        conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    except OSError as exc:
        log.critical(
            "IMAP connection refused / timeout",
            extra={"host": IMAP_HOST, "port": IMAP_PORT},
            exc_info=True,
        )
        raise MailboxConnectionError(f"{IMAP_HOST}:{IMAP_PORT}") from exc
    try:
        try:
            conn.login(IMAP_USERNAME, IMAP_APP_PASSWORD)
        except imaplib.IMAP4.error as exc:
            log.critical("IMAP login rejected", extra={"backend": "imap"})
            raise MailboxAuthError("IMAP login rejected") from exc

        conn.select(MAILBOX_FOLDER)
        since = (datetime.now(UTC) - timedelta(days=SINCE_DAYS)).strftime("%d-%b-%Y")
        literal = max(re.split(r"[^A-Za-z0-9 ]+", EMAIL_SUBJECT_PATTERN), key=len).strip()
        criteria = ["UNSEEN", "FROM", TRUSTED_SENDER, "SINCE", since]
        if literal:
            criteria += ["SUBJECT", literal]
        typ, data = conn.search(None, *criteria)
        uids = data[0].split() if typ == "OK" and data and data[0] else []
        out: list[_RawEmail] = []
        for uid in uids:
            typ, msg_data = conn.fetch(uid, "(RFC822)")
            if typ == "OK" and msg_data and msg_data[0]:
                out.append(parse_eml(msg_data[0][1], uid.decode()))
        return out
    finally:
        try:
            conn.logout()
        except (imaplib.IMAP4.error, OSError):  # pragma: no cover
            pass


def _imap_mark(email_uid: str) -> None:
    conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try:
        conn.login(IMAP_USERNAME, IMAP_APP_PASSWORD)
        conn.select(MAILBOX_FOLDER)
        conn.store(email_uid, "+FLAGS", "\\Seen")
    finally:
        conn.logout()


def _gmail_service():  # pragma: no cover - needs Google libs + a real token
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    creds = None
    try:
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, scopes)
    except (FileNotFoundError, ValueError):
        pass
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:
            log.critical("Gmail token refresh failed", extra={"token_file": GMAIL_TOKEN_FILE})
            raise AuthRefreshError("token refresh failed") from exc
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, scopes)
        creds = flow.run_local_server(port=0)
        with open(GMAIL_TOKEN_FILE, "w") as fh:
            fh.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def _gmail_search() -> list[_RawEmail]:  # pragma: no cover - exercised via a mocked service
    service = _gmail_service()
    literal = max(re.split(r"[^A-Za-z0-9 ]+", EMAIL_SUBJECT_PATTERN), key=len).strip()
    query = f'from:{TRUSTED_SENDER} is:unread newer_than:{SINCE_DAYS}d "{literal}"'
    resp = service.users().messages().list(userId="me", q=query).execute()
    out: list[_RawEmail] = []
    for ref in resp.get("messages", []):
        raw = service.users().messages().get(userId="me", id=ref["id"], format="raw").execute()
        import base64

        out.append(parse_eml(base64.urlsafe_b64decode(raw["raw"]), ref["id"]))
    return out


def _gmail_mark(email_uid: str) -> None:  # pragma: no cover
    service = _gmail_service()
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    label_id = next((lbl["id"] for lbl in labels if lbl["name"] == PROCESSED_LABEL), None)
    body = {"removeLabelIds": ["UNREAD"], "addLabelIds": [label_id] if label_id else []}
    service.users().messages().modify(userId="me", id=email_uid, body=body).execute()
