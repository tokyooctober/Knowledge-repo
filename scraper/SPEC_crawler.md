# `scraper/crawler.py` — Article Crawler

---
```
module:     scraper/crawler.py
spec:       scraper/SPEC_crawler.md
layer:      Scraping
depends_on: config.py · logger.py
            scraper/login.py  (BrowserContext, ensure_authenticated)
            storage/metadata_db.py  (known_urls for is_new flagging)
used_by:    scheduler/monthly_job.py
services:   target website  (Playwright, authenticated HTTP)
output:     list[RawPage]  →  passed to ingestion/extractor.py
```
---

## Purpose
Given an authenticated Playwright `BrowserContext` and an explicit list of article URLs (from the email reader — this is a **Phase 2 only** module), navigate to each URL and return the fully rendered HTML. Because every URL is behind the site's login wall, the crawler resolves login walls inline: after any navigation it hands the page to `login.ensure_authenticated()`, which — if a wall is there — opens the window for a **human to sign in**, waits, returns the page to the URL the crawler asked for, and hands control back. The crawler then extracts the article text from that page exactly as if the wall had never appeared.

---

## Responsibilities
- Validate the session is live with a lightweight pre-check before entering the fetch loop
- Deduplicate URLs before fetching
- Navigate to each article URL through the authenticated `BrowserContext`
- After every navigation, call `login.ensure_authenticated(page, url)` — passing the URL it asked for — to resolve any login wall and guarantee the page is back at `url` before extracting HTML
- Rate-limit requests between fetches
- Return a list of `RawPage` objects for the ingestion pipeline

---

## Inline login handling

When the crawler navigates to a direct article URL, the site may respond with a login wall
rather than the article — even when the session appeared valid at the pre-check stage. This
happens because:

- The article URL carries its own access-control check independent of session cookies
- The session cookie may have a shorter TTL on deep-linked premium content
- The site uses a login overlay or modal injected after the page loads (200 status, no redirect)

Rather than raising an error and delegating back to the scheduler, the crawler resolves
login walls **on the spot** by calling `login.ensure_authenticated(page, url)` after every
`page.goto(url)` — passing the URL it requested, because a `goto` that was redirected to a
login page cannot recover that URL from `page.url`. That function:

1. Checks whether the page already shows `SUCCESS_SELECTOR` — the common case, a cheap
   no-op that returns `True` and touches nothing
2. Checks whether the page is actually a login wall (a login form, or a login URL). A page
   that is merely missing `SUCCESS_SELECTOR` — a 404, a moved article — is **not** a login
   wall; `ensure_authenticated` returns `True` and the crawler's own guard (step c) skips it
3. If it is a wall: brings the browser window to the front and waits, polling, while a
   **human signs in** — password, 2FA, captcha, whatever the site asks for
4. **Navigates the page back to the URL the crawler requested**, because a completed login
   normally lands on a members dashboard, not on the article
5. Confirms `SUCCESS_SELECTOR` on that page, saves the refreshed session to `state.json`,
   and returns `True`

The crawler needs no knowledge of any of it. Its contract with `ensure_authenticated` is a
single line: **when it returns `True`, `page` is at the URL you asked for and is
authenticated — read it.** That is the handoff.

If `ensure_authenticated` returns `False` — nobody was available to log in, the wait timed
out, or the account cannot see this particular report — the crawler logs an error, skips
that URL, and continues with the next one. It does **not** abort the whole run.

### Do not re-read `url` after `ensure_authenticated`

Step (d) calls `page.content()`, not `page.goto(url)` again. `ensure_authenticated` has
already put the page where it belongs; a second navigation would throw away a freshly
authenticated page load, double the request count, and give the site another chance to
serve a wall. Equally, do not cache `page.url` from before the call and compare — the
function may legitimately have navigated several times to get back.

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `browser_context` | `login.py` | Authenticated Playwright `BrowserContext` — mandatory |
| `urls` | Scheduler | Paywall-protected article URLs (from the email reader) |
| `known_urls` | `metadata_db` | Already-indexed URLs — used to set `RawPage.is_new` |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `list[RawPage]` | Python list | One entry per successfully fetched article |

### `RawPage` dataclass

Defined once in `models.py` (see [SPEC.md](../SPEC.md)); shown here for reference.

```python
@dataclass
class RawPage:
    url:         str
    html:        str
    fetched_at:  datetime
    status_code: int   # 200 for every page that reaches the extractor — see step (e)
    is_new:      bool   # diagnostic only (url not in known_urls); nothing branches on it
```

---

## Core Logic

```
0. SESSION PRE-CHECK
   page = await browser_context.new_page()
   await page.goto(HEALTH_CHECK_URL, waitUntil="networkidle")
   authenticated = await login.ensure_authenticated(page, HEALTH_CHECK_URL)
   IF not authenticated:
     await page.close()
     raise SessionExpiredError("Pre-check failed — manual login was not completed")
   # The pre-check is deliberately the FIRST thing that can ask for a human. If a login
   # is needed, it is asked for once, here, before 40 URLs start marching past.
   IF SUCCESS_SELECTOR still absent after ensure_authenticated returned True:
     await page.close()
     raise LoginStateError("Unexpected page state at health check URL")
   await page.close()
   log DEBUG "Pre-check passed"

1. DEDUPLICATE
   urls = list(dict.fromkeys(urls))
   log DEBUG deduplicated count

2. FETCH LOOP
   results = []
   for url in urls:

     Every step below runs inside ONE try/finally per URL:

         page = await browser_context.new_page()
         try:
             ... steps a–e ...
         finally:
             await page.close()          ← the ONLY page.close() in the loop

     Closing the page in `finally` is what makes the retry, the login-failure skip,
     and the guard below safe. An earlier revision closed the page in step (c) and
     then queried it in step (d) — TargetClosedError on every URL. Do not reintroduce
     a close inside a branch.

     a. NAVIGATE  (bind the response — step (e) needs it)
        response = await page.goto(url, waitUntil="networkidle",
                                   timeout=PAGE_TIMEOUT_MS)
        # On PlaywrightTimeoutError: log WARNING, sleep 5 s, retry ONCE.
        # If the retry also times out, log ERROR and `continue` — the finally
        # block closes the page, so no Page leaks per timed-out URL.
        # page.goto() returns None only for a same-document navigation, which
        # cannot happen on a fresh page; step (e) still guards for it.

     b. LOGIN-WALL CHECK  (may block on a human; may re-navigate the page)
        authenticated = await login.ensure_authenticated(page, url)

        IF not authenticated:
          log ERROR "Login wall not cleared — skipping URL"
          continue                 ← skip this URL; do not abort the run

        # On True, `page` is authenticated AND back at `url`. Do not re-navigate.

     c. GUARD: verify SUCCESS_SELECTOR present in the DOM
        (catches silent login-page returns with HTTP 200)
        IF await page.query_selector(SUCCESS_SELECTOR) is None:
          log WARNING "SUCCESS_SELECTOR absent in fetched page — possible login page"
          continue                 ← skip; do not add to results

        This must be a DOM query on a LIVE page, not `SUCCESS_SELECTOR not in html`.
        SUCCESS_SELECTOR is a CSS selector (".member-content"); the literal string
        ".member-content" never appears in markup that reads class="member-content",
        so the substring form skips every URL.

     d. CONTENT EXTRACTION  (after the guard, before the finally closes the page)
        html       = await page.content()
        fetched_at = datetime.now(UTC)

     e. APPEND RESULT
        # status_code: the crawler got here only after ensure_authenticated returned
        # True AND the step (c) SUCCESS_SELECTOR guard passed, so `page` currently holds
        # a rendered, authenticated copy of `url`. `response` from step (a) may be a
        # pre-login 403 or a redirect to /login — recording that would make the
        # extractor's `status_code != 200` guard silently drop a page that was in fact
        # fetched fine. Record the status of the page we actually have:
        status = response.status if (response is not None
                                     and 200 <= response.status < 300) else 200
        results.append(RawPage(
          url        = url,
          html       = html,
          fetched_at = fetched_at,
          status_code = status,
          is_new     = (url not in known_urls),
        ))

     f. RATE LIMIT
        await asyncio.sleep(CRAWL_DELAY_MS / 1000)

3. RETURN results
```

### Why inline login instead of raising SessionExpiredError?

The previous design raised `SessionExpiredError` on any login prompt and delegated re-authentication back to the scheduler. This had two problems:

1. **Granularity**: a login prompt on URL #3 of 40 aborted the entire fetch call, losing URLs #1 and #2 already successfully fetched (or requiring the scheduler to track partial progress).
2. **Redundancy**: the scheduler's re-auth retry loop was doing work that the crawler is better placed to do — the `Page` object is already open at the right URL.

The inline approach handles the login at the point where it occurs, saves the refreshed session immediately so no later URL in the run needs another one, and continues the fetch loop. Since the login is now performed by a human, keeping it at the point of occurrence matters even more: the person is asked once, in context, and the run resumes by itself — no restart, no re-fetch of the URLs already done. The scheduler's re-auth retry loop is retained as a last-resort fallback but should rarely fire.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Empty `urls` list | Raise `ValueError("urls list must not be empty")` |
| Pre-check: `ensure_authenticated` returns `False` | Raise `SessionExpiredError` — no human available, or the login was not completed; abort run |
| Pre-check: login succeeded but `SUCCESS_SELECTOR` still absent | Raise `LoginStateError` — unexpected site state; abort run |
| Per-URL: `ensure_authenticated` returns `False` | Log ERROR; skip URL; continue with next |
| Per-URL: page has no `SUCCESS_SELECTOR` and is not a login wall (404, moved article) | `ensure_authenticated` returns `True` as a no-op; the step (c) guard skips the URL. **No browser window opens and nothing waits** |
| Per-URL: `SUCCESS_SELECTOR` absent in DOM after `ensure_authenticated` returned `True` | Log WARNING; skip URL (silent login-page return, or an unreachable article) |
| Page load timeout (first attempt) | Retry once after 5 s; if still timing out, log ERROR and skip URL |
| Pre-login `response` was 3xx/4xx but the page then authenticated and passed the step (c) guard | Record `status_code = 200` — the page in hand is a good copy of `url` (see step (e)) |
| Page reaches step (c) with no `SUCCESS_SELECTOR` and no login wall (404, moved article) | Skipped by the guard — never reaches step (e), so no `RawPage` is emitted |

---

## Configuration Constants

```python
CRAWL_DELAY_MS    = 2_000    # ms between article fetches
PAGE_TIMEOUT_MS   = 30_000   # max wait per page navigation
HEALTH_CHECK_URL  = os.environ.get("HEALTH_CHECK_URL", "")           # required; members-only page
SUCCESS_SELECTOR  = os.environ.get("SUCCESS_SELECTOR", ".member-content")  # present only when authed
```

> `LOGIN_FORM_SELECTOR`, `LOGIN_URL_FRAGMENT`, `INTERACTIVE_LOGIN`, `MANUAL_LOGIN_TIMEOUT_MS`
> and `MANUAL_LOGIN_POLL_MS` are defined in `config.py` and used exclusively by `login.py`.
> The crawler has no knowledge of login forms, and there are no credentials in the system
> to have knowledge of.

> `PAGE_TIMEOUT_MS` bounds a page load only. It does **not** bound the wait for a human —
> that is `MANUAL_LOGIN_TIMEOUT_MS`, and a single URL in the fetch loop can therefore take
> minutes. Do not wrap the fetch loop in an overall `asyncio.timeout` derived from
> `PAGE_TIMEOUT_MS × len(urls)`.

---

## Key Dependencies
- `playwright` — async browser API
- `asyncio` — fetch loop timing (stdlib)
- `scraper/login.py` — `ensure_authenticated()` called per page

---

## Public Interface

```python
async def fetch_pages(
    context: BrowserContext,
    urls: list[str],
    known_urls: set[str],
) -> list[RawPage]:
    """Fetch fully rendered HTML for each paywall-protected article URL.

    Performs a session pre-check before entering the fetch loop.
    Calls login.ensure_authenticated(page, url) after every navigation to
    handle inline login prompts without interrupting the fetch loop.
    Skips individual URLs whose login fails; does not abort the run.
    Deduplicates input URLs. Rate-limited by CRAWL_DELAY_MS.
    Raises SessionExpiredError if the pre-check login fails (credentials invalid).
    Raises LoginStateError if the pre-check page is in an unexpected state.
    """
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.scraper.crawler"
```

| Event | Level | Extra fields |
|---|---|---|
| Crawl started | INFO | `url_count`, `known_url_count` |
| Pre-check started | DEBUG | `health_check_url` |
| Pre-check passed | DEBUG | `health_check_url` |
| Pre-check login failed — aborting | CRITICAL | `health_check_url`, `error_type` |
| Pre-check unexpected state — aborting | CRITICAL | `health_check_url`, `page_title` |
| Duplicate URLs removed | DEBUG | `original_count`, `deduplicated_count` |
| Navigating to article | DEBUG | `url`, `article_number`, `total_articles`, `is_new` |
| Login wall detected by ensure_authenticated | INFO | `url` — the wait-for-human detail is logged by login.py |
| Login wall cleared — continuing | INFO | `url` |
| Login wall not cleared — skipping URL | ERROR | `url` |
| SUCCESS_SELECTOR absent after auth — skipping | WARNING | `url`, `success_selector` |
| Article fetched | DEBUG | `url`, `html_length_chars` |
| Pre-login response was non-2xx — recording 200 for authenticated page | DEBUG | `url`, `pre_login_status` |
| Page timeout — retrying | WARNING | `url`, `attempt`, `timeout_ms` |
| Page timeout after retry — skipping | ERROR | `url`, `attempts` |
| Rate-limit sleep | DEBUG | `delay_ms` |
| Crawl complete | INFO | `fetched`, `skipped_login_fail`, `skipped_no_selector`, `failed_timeout` |
| Manual logins performed during crawl | INFO | `manual_login_count` — >1 in a run means the session is not persisting; check `state.json` writes |

---

## Testing Notes

- Use `pytest-playwright` with a local HTTP test server serving fixture pages
- **No login wall**: serve article page with `SUCCESS_SELECTOR` → assert `ensure_authenticated` returns on the fast path; article returned
- **Login wall on article URL**: serve a login form at the article URL; flip the fixture's auth flag mid-poll to simulate the human, and have the login land on a **dashboard** page, not the article → assert the returned `RawPage.html` is the ARTICLE's HTML and `RawPage.url` is the article URL. This is the test that proves control comes back to the crawler
- **Login wall — nobody logs in**: fixture never flips → assert that URL is skipped and the run continues to the next URL
- **Dead URL, no login wall**: serve a 404 with neither `SUCCESS_SELECTOR` nor a login form → assert the URL is skipped by the step (c) guard and that no manual-login prompt was ever emitted
- **Pre-check passes**: serve `HEALTH_CHECK_URL` with `SUCCESS_SELECTOR` → assert fetch loop entered
- **Pre-check needs login**: serve `HEALTH_CHECK_URL` as a login form; flip the auth flag mid-poll → assert the pre-check passes and the fetch loop is entered
- **Pre-check login not completed**: fixture never flips → assert `SessionExpiredError` raised and the fetch loop is not entered
- **Non-interactive run** (`INTERACTIVE_LOGIN=never`) with an expired session → assert `SessionExpiredError` is raised from the pre-check promptly, with no five-minute wait
- **Silent 200 login page**: serve article URL with HTTP 200 but no `SUCCESS_SELECTOR` → assert URL skipped after `ensure_authenticated` returns `True` and the step (c) DOM guard fires
- **Pre-login 403 then authenticated**: serve the article URL as 403 with a login form; flip the fixture mid-poll and land on the article → assert the emitted `RawPage.status_code == 200`, so the extractor does not drop it
- Assert `ensure_authenticated` is always called with the URL from step (a), never with `page.url`
- Assert `is_new` set correctly from `known_urls`
- Assert deduplication before fetching
- Assert `CRAWL_DELAY_MS` sleep between fetches
- Assert empty `urls` raises `ValueError`
