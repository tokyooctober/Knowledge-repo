# `inbox/email_reader.py` — Email Reader

---
```
module:     inbox/email_reader.py
spec:       inbox/SPEC_email_reader.md
layer:      Email trigger
depends_on: config.py · logger.py
used_by:    scheduler/monthly_job.py
services:   Gmail API · IMAP (mailbox)
```
---

## Purpose
Monitor a designated mailbox for new-report notification emails from a specific trusted sender. Parse the email to locate the direct article link — which appears after a known anchor phrase — and return it as the URL for that month's ingestion run.

This is the **Phase 2** entry point. It is the only way a new report enters the system;
the Phase 1 markdown corpus at `MD_CORPUS_DIR` covers the back-catalogue and is loaded by
`ingestion/md_loader.py` without touching the mailbox.

> **Package renamed `email/` → `inbox/`.** A top-level package named `email` shadows the
> Python standard library's `email` module, which `google-api-python-client`, `imaplib`
> and `smtplib` all import — the failure surfaces deep inside a dependency and is
> miserable to diagnose. `mailbox` is *also* stdlib, so it is not a valid alternative;
> `inbox` is free. Imports are `from inbox.email_reader import read_update_email`, and
> this spec moved to `inbox/SPEC_email_reader.md` to keep the
> module-path-mirrors-spec-path convention intact.

**The email reader's sole job is URL extraction.** It does not fetch or read the article content. The extracted URL points to a page that sits behind the site's login/password wall; only the Playwright crawler (operating with an authenticated browser session from `login.py`) can retrieve the actual article content.

---

## Responsibilities
- Connect to the mailbox via Gmail API (primary) or IMAP (fallback)
- Identify unread emails from `TRUSTED_SENDER` with a subject matching `EMAIL_SUBJECT_PATTERN`
- Verify the sender identity as a spoofing guard
- Extract the single article URL from the anchor phrase position
- Validate the extracted URL is on the expected domain
- Mark processed emails as read and labelled to prevent reprocessing
- Return a structured `EmailUpdate` object for the scheduler — the URL in this object is a **paywall-protected link** that requires website authentication to access
- Log all connection attempts, parsing steps, failures, and extracted URLs

> **Separation of concerns**: The email reader holds no website credentials and performs no website login. It only reads email. Website authentication is the exclusive responsibility of `scraper/login.py`.

---

## Real Email Format

The incoming notification email from the trusted sender follows this pattern (based on production sample):

```
Subject: <report title, e.g. "May 2026 Premium Report">

Body (HTML):
  Hello,

  A new premium report is available in the members area:
  https://www.example.com/members/

  Here's a direct link to the report:
  https://www.example.com/premium-2026-5-10/

  - The macro section of this report focuses on...
  - The investment analysis section provides...

  Best regards, The Author
```

### Anchor phrase
The parser locates the URL by finding the anchor phrase:

```
"Here's a direct link to the report:"
```

The URL immediately follows this phrase — either on the same line (plain text) or in the next non-empty line / `<a href>` tag (HTML). Everything before the anchor phrase (including the members-area link) is ignored.

### Why anchor-phrase parsing instead of "first URL" or "all URLs"
The email contains at least two URLs:
1. `https://www.example.com/members/` — the general members area (always present, always first)
2. `https://www.example.com/premium-YYYY-M-DD/` — the specific new article (the one we want)

Anchoring on the phrase "direct link to the report" is robust: it targets exactly the right URL regardless of how many other links appear in the email.

---

## Parsing Strategy

### Step 1 — Prefer plain-text body
```
If email has text/plain part:
  work_text = plain_text_part
Else:
  work_text = BeautifulSoup(html_part).get_text(separator='\n')
```

### Step 2 — Locate anchor phrase (case-insensitive)
```python
ANCHOR_PATTERN = re.compile(
    r"here[''\u2019]s\s+a\s+direct\s+link\s+to\s+the\s+report\s*:?\s*(https?://\S+)",
    re.IGNORECASE
)
```

The regex captures the URL on the same line as the anchor if they appear together, or falls through to Step 3 if the URL is on the next line.

### Step 3 — Fallback: URL on the line following the anchor
```python
lines = work_text.splitlines()
for i, line in enumerate(lines):
    if re.search(r"direct link to the report", line, re.IGNORECASE):
        # scan the next few lines for the first https URL
        for j in range(i + 1, min(i + 5, len(lines))):
            match = re.search(r"https?://\S+", lines[j])
            if match:
                return match.group(0).rstrip("_*>)")  # strip trailing markdown
```

### Step 4 — HTML fallback (if text body unavailable)
```python
soup = BeautifulSoup(html_body, "lxml")
# Find the tag containing the anchor text
anchor_tag = soup.find(string=re.compile(r"direct link to the report", re.I))
if anchor_tag:
    # Walk forward through siblings/next elements to find first <a href>
    for el in anchor_tag.find_all_next(["a", "p", "div"], limit=5):
        href = el.get("href") or el.get_text()
        if href and href.startswith("https://"):
            return href
```

### Step 5 — Validation
```python
parsed = urlparse(extracted_url)
assert parsed.scheme == "https"
assert parsed.netloc == SITE_DOMAIN or parsed.netloc.endswith("." + SITE_DOMAIN)
# Dot boundary required. A bare endswith(SITE_DOMAIN) also accepts
# "notexample.com" and "evil-example.com" — an attacker-controlled host that
# passes validation and then gets fetched with the session cookies attached.
```

> **No content fetching here.** The validated URL is returned as-is. Because it is behind the site's login wall, any attempt to fetch it without a valid authenticated session will return the login page HTML, not the article. The scheduler passes it to `login.py` → `crawler.py` which hold the authenticated Playwright context.

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `TRUSTED_SENDER` | config / `.env` | Exact from-address (e.g. `author@example.com`) |
| `EMAIL_SUBJECT_PATTERN` | config / `.env` | Case-insensitive regex checked client-side against the subject; its longest literal run is what the server-side search filters on (e.g. `premium report`) |
| `MAILBOX_FOLDER` | config | IMAP folder / Gmail label (default: `INBOX`) |
| `SINCE_DAYS` | config | Search window in days (default: `40`) |
| `SITE_DOMAIN` | config | Domain the article URL must belong to (e.g. `example.com`) |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `EmailUpdate` | dataclass | Parsed update; `article_links` always contains exactly one item |
| `None` | — | No qualifying unprocessed email found |

### Dataclasses
```python
@dataclass
class ArticleLink:
    title: str | None   # email subject (used as a human-readable label)
    url:   str          # paywall-protected article URL; requires site login to access

@dataclass
class EmailUpdate:
    email_uid:     str
    sender:        str
    subject:       str
    received_at:   datetime
    article_links: list[ArticleLink]   # always length 1 for this email format
    raw_body:      str                 # full body preserved for debug logging
```

> `ArticleLink.url` is a members-only URL. Accessing it directly in a browser (without being logged in) redirects to the login page. The crawler receives this URL and fetches it through an authenticated Playwright session managed by `login.py`.

---

## Backend Options

### Option A: Gmail API (recommended)
- **Library**: `google-api-python-client`, `google-auth-oauthlib`
- **Auth**: OAuth2 with offline refresh token stored in `session/gmail_token.json`
- **Scopes**: `https://www.googleapis.com/auth/gmail.modify`
- **First run**: browser OAuth consent; subsequent runs use refresh token silently

### Option B: IMAP (universal fallback)
- **Library**: `imaplib` + `email` (both stdlib)
- **Auth**: `IMAP_USERNAME` + `IMAP_APP_PASSWORD` from `.env`
- **Server**: `IMAP_HOST:IMAP_PORT` (SSL, port 993)

Set `EMAIL_BACKEND = "gmail"` or `"imap"` in `config.py`.

---

## Full Core Logic

```
1. CONNECT
   Gmail: load/refresh token from gmail_token.json; build service object
   IMAP:  SSL connect to IMAP_HOST:IMAP_PORT; login

2. SEARCH  (server-side pre-filter, then a client-side check)
   Server side — both backends do a substring match, not a regex:
     FROM    = TRUSTED_SENDER
     SUBJECT contains the LONGEST literal run in EMAIL_SUBJECT_PATTERN
              (for the default "premium report", the whole string)
     SINCE   = today - SINCE_DAYS
     UNSEEN  (unread only)
   Client side — re.search(EMAIL_SUBJECT_PATTERN, subject, re.IGNORECASE) on each hit,
   so a pattern like r"premium\s+report" still works even though the server saw only
   "premium report". A hit that fails the client-side check is dropped with a DEBUG line.

   0 results → log INFO, return None
   >1 results → log WARNING, sort by date desc, use most recent

3. FETCH & DECODE
   Download full RFC 2822 message
   Decode subject (handle =?utf-8?... encoded-words)
   Verify FROM header == TRUSTED_SENDER (exact); mismatch → raise SenderMismatchError

4. EXTRACT BODY
   Prefer text/plain part; fall back to HTML→text via BeautifulSoup

5. PARSE URL  (Steps 1–4 from Parsing Strategy above)
   Regex anchor-phrase capture (same line)
   → Fallback: scan next 5 lines
   → Fallback: HTML <a href> after anchor element
   No URL found → raise NoLinksFoundError

6. VALIDATE URL
   Must start with https://
   Must end with SITE_DOMAIN; otherwise raise InvalidDomainError

7. BUILD ArticleLink
   title = email subject (stripped)
   url   = validated URL

8. DO NOT MARK AS PROCESSED HERE
   read_update_email() returns the EmailUpdate WITHOUT marking anything. Marking is
   a separate call the scheduler makes only after the article is safely in the index:

       mark_processed(email_uid)      Gmail: apply PROCESSED_LABEL; mark as read
                                      IMAP:  set \Seen flag on UID

   WHY. Marking here consumes the notification before the scrape has even started.
   If login fails, the crawl times out, or ingestion raises, the run aborts — but the
   email is already \Seen and labelled, the search filter is UNSEEN, and the next
   12-hourly --once run finds nothing. That month's report is then lost with no error
   anywhere, and the only recovery is to dig the URL out of the logs and re-run with
   --url. Deferring the mark makes a failed run simply retry on the next poll.

   The cost of deferring is a possible double-ingest if the process dies between the
   successful ingest and the mark. That is harmless: the URL is already in the
   database, so the retry hits the unchanged-hash path and skips.

9. RETURN EmailUpdate(
       email_uid     = message UID,
       sender        = TRUSTED_SENDER,
       subject       = decoded subject,
       received_at   = parsed Date header,
       article_links = [ArticleLink(title, url)],
       raw_body      = full body text,
   )
```

---

## Configuration Constants

```python
EMAIL_BACKEND          = "gmail"
TRUSTED_SENDER         = os.environ.get("TRUSTED_SENDER", "")   # ← set in .env
EMAIL_SUBJECT_PATTERN  = os.environ.get("EMAIL_SUBJECT_PATTERN", "premium report")
                               # case-insensitive regex, checked client-side; server-side
                               # search uses its longest literal run
MAILBOX_FOLDER         = "INBOX"
SINCE_DAYS             = 40
SITE_DOMAIN            = os.environ.get("SITE_DOMAIN", "")   # URL domain allowlist; required
PROCESSED_LABEL        = "knowledge-repo/processed"
GMAIL_TOKEN_FILE       = "session/gmail_token.json"
GMAIL_CREDENTIALS_FILE = "session/gmail_credentials.json"
IMAP_HOST              = "imap.gmail.com"
IMAP_PORT              = 993
IMAP_USERNAME          = os.environ.get("IMAP_USERNAME", "")
IMAP_APP_PASSWORD      = os.environ.get("IMAP_APP_PASSWORD", "")
```

---

## Error Handling

| Scenario | Behaviour | Log level |
|---|---|---|
| No qualifying unread email | Return `None`; scheduler skips run | INFO |
| Gmail token refresh fails | Raise `AuthRefreshError`; log token file path (not contents) | CRITICAL |
| IMAP connection refused / timeout | Raise `MailboxConnectionError`; log host and port | CRITICAL |
| IMAP login rejected | Raise `MailboxAuthError`; never log password | CRITICAL |
| FROM ≠ TRUSTED_SENDER | Raise `SenderMismatchError`; log both addresses | ERROR |
| Anchor phrase not found in body | Raise `NoLinksFoundError`; log first 500 chars of body | ERROR |
| URL found but wrong domain | Raise `InvalidDomainError`; log extracted URL and expected domain | ERROR |
| URL does not start with https:// | Raise `InvalidDomainError` | ERROR |
| Mark-as-read fails | Log WARNING; do not abort (URL already extracted) | WARNING |
| Multiple qualifying emails | Use most recent; log count and all UIDs | WARNING |
| Markdown/punctuation stripped from URL | Log DEBUG showing raw and cleaned URL | DEBUG |

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.inbox.email_reader"
```

| Event | Level | Extra fields |
|---|---|---|
| Connecting to mailbox | INFO | `backend`, `folder` |
| Connection established | DEBUG | `backend`, `folder` |
| Search issued | DEBUG | `trusted_sender`, `since_days`, `subject_pattern` |
| No qualifying email | INFO | `since_days`, `trusted_sender` |
| Multiple emails found | WARNING | `count`, `email_uids` |
| Email fetched | INFO | `email_uid`, `sender`, `subject`, `received_at` |
| Sender verified | DEBUG | `email_uid`, `sender` |
| Sender mismatch | ERROR | `email_uid`, `claimed_sender`, `trusted_sender` |
| Plain-text body used | DEBUG | `email_uid`, `body_length_chars` |
| Falling back to HTML body | DEBUG | `email_uid` |
| Anchor phrase found (same-line capture) | DEBUG | `email_uid`, `raw_url` |
| Anchor phrase found (next-line fallback) | DEBUG | `email_uid`, `raw_url`, `line_offset` |
| Anchor phrase found (HTML fallback) | DEBUG | `email_uid`, `raw_url` |
| URL trailing chars stripped | DEBUG | `email_uid`, `before`, `after` |
| Anchor phrase not found | ERROR | `email_uid`, `body_preview` (first 500 chars) |
| URL domain invalid | ERROR | `email_uid`, `extracted_url`, `expected_domain` |
| URL extracted and validated | INFO | `email_uid`, `url`, `domain` |
| Email marked as processed | DEBUG | `email_uid`, `backend`, `label` |
| Mark-as-read failed | WARNING | `email_uid`, `error_type` |
| Gmail token expired — refreshing | DEBUG | `token_file` |
| Gmail auth failure | CRITICAL | `error_type`, `token_file` |
| IMAP auth failure | CRITICAL | `backend`, `error_type` — never log password |

---

## Key Dependencies
- `google-api-python-client` — Gmail API
- `google-auth-oauthlib` — OAuth2 flow
- `google-auth-httplib2` — HTTP transport for Gmail
- `imaplib`, `email` — IMAP (stdlib)
- `beautifulsoup4` + `lxml` — HTML body stripping and anchor element search
- `re`, `urllib.parse` — URL extraction and validation (stdlib)

---

## `.env` additions
```
TRUSTED_SENDER=author@example.com
SITE_DOMAIN=example.com              # required — the URL domain allowlist
EMAIL_SUBJECT_PATTERN=premium report  # optional — default shown
IMAP_USERNAME=you@gmail.com           # imap backend only
IMAP_APP_PASSWORD=xxxx xxxx xxxx      # imap backend only
```

## `session/` additions (add to `.gitignore`)
```
session/gmail_token.json
session/gmail_credentials.json
```

---

## Public Interface
```python
def mark_processed(email_uid: str) -> None:
    """Mark one email as handled: Gmail applies PROCESSED_LABEL and marks read;
    IMAP sets \\Seen. Called by the scheduler ONLY after the article is in the
    index — see step 8 of Core Logic for why this is not done inside
    read_update_email().
    """

def read_update_email() -> EmailUpdate | None:
    """Connect to mailbox and extract the direct article link from the latest
    new-report notification email.

    Returns EmailUpdate with exactly one ArticleLink if a new qualifying email
    is found and successfully parsed.
    Returns None if no unprocessed notification email is present.
    Raises: MailboxConnectionError, SenderMismatchError,
            NoLinksFoundError, InvalidDomainError.
    Marks the email as read/processed before returning.

    IMPORTANT: The URL inside ArticleLink is a paywall-protected members-only
    link. The caller (scheduler) must pass it to the Playwright crawler with
    an authenticated browser session before any content can be retrieved.
    Fetching the URL without authentication returns the site login page, not
    the article.
    """
```

---

## First-time Gmail Setup
```
1. console.cloud.google.com → new project → enable Gmail API
2. OAuth2 credentials (Desktop app) → download → save as session/gmail_credentials.json
3. python -c "from inbox.email_reader import read_update_email; read_update_email()"
4. Browser opens for consent → approve → session/gmail_token.json written
5. All subsequent runs refresh silently
```

---

## Testing Notes

### Fixture emails (save as `.eml` files in `tests/fixtures/`)

**`valid_report_email.eml`** — matches production format:
```
From: author@example.com
Subject: May 2026 Premium Report
Content-Type: text/plain

Hello,

A new premium report is available in the members area:
https://www.example.com/members/

Here's a direct link to the report:
https://www.example.com/premium-2026-5-10/

Best regards, The Author
```

**`valid_report_html.eml`** — HTML body version with `<a href>`:
```html
<p>Here's a direct link to the report:</p>
<p><a href="https://www.example.com/premium-2026-5-10/">
  https://www.example.com/premium-2026-5-10/
</a></p>
```

**`no_anchor_email.eml`** — missing anchor phrase entirely

**`wrong_domain_email.eml`** — URL present but on `attacker.com`

**`members_only_email.eml`** — only the `/members/` URL, no direct link

### Assertions
- `valid_report_email.eml` → `EmailUpdate` with `article_links[0].url == "https://www.example.com/premium-2026-5-10/"`
- `valid_report_html.eml` → same URL extracted via HTML fallback path
- `no_anchor_email.eml` → raises `NoLinksFoundError`
- `wrong_domain_email.eml` → raises `InvalidDomainError`
- `members_only_email.eml` → raises `NoLinksFoundError` (members URL appears before anchor phrase, not after)
- Assert `SenderMismatchError` when FROM ≠ TRUSTED_SENDER
- Assert `read_update_email()` returns `None` when mailbox has no UNSEEN matching emails
- Assert a subject matching `EMAIL_SUBJECT_PATTERN=r"premium\s+report"` but not the literal
  "premium report" is accepted (server pre-filter is substring, final check is the regex)
- Assert a `SenderMismatch`-clean email whose subject fails the client-side regex is dropped
- Assert `IMAP_APP_PASSWORD` never appears in any log output
- Assert `gmail_token.json` contents never appear in any log output
- Mock all mailbox I/O — never hit real inbox in tests
