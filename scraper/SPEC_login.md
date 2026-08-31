# `scraper/login.py` — Session Manager (human-in-the-loop login)

---
```
module:     scraper/login.py
spec:       scraper/SPEC_login.md
layer:      Scraping
depends_on: config.py · logger.py
used_by:    scheduler/monthly_job.py  (get_authenticated_context, close_browser)
            scraper/crawler.py        (ensure_authenticated, per-page login)
services:   target website  (Playwright, authenticated HTTP)
files:      session/state.json  (Playwright browser storage state)
```
---

## Purpose

Get the browser into an authenticated state and keep it there, then hand control back to
the crawler.

**The login itself is performed by a human, not by this module.** This module never types
a username or a password. It opens a real, visible browser window at the page that is
asking for a login, tells the operator to sign in, and waits — polling the live page until
the site shows an authenticated page. The moment it does, the session is saved and control
returns to the caller, which resumes exactly where it was.

Why: the site's login is not a form this code can reliably drive. It can present a captcha,
a 2FA code, an email magic link, a consent interstitial, or a rebuilt form with different
field names. Every one of those breaks a scripted fill-and-submit, and each break costs a
whole ingestion run. A human sitting at the machine handles all of them without this module
knowing anything about the login page at all.

Two responsibilities:

1. **Session bootstrap** (`get_authenticated_context`) — launch Playwright, restore
   `state.json` if it is still valid, otherwise run a manual login. Returns a live
   `BrowserContext` used for the entire ingestion run.

2. **Inline login** (`ensure_authenticated(page, expected_url)`) — called by the crawler
   *on every page after navigation*, with the URL the crawler asked for passed explicitly.
   Detects a login wall, runs a manual login on the spot if one is found, **returns the
   page to `expected_url`**, and reports success. The crawler then extracts the article
   text from that page as if the wall had never been there.

---

## Responsibilities

- Launch a Playwright browser — **headful** whenever a human could be asked to log in
- Load and validate a persisted session (`state.json`) on every run
- Detect a login wall on any live Playwright `Page`
- Open the login page for the human, wait for them to finish, and confirm success
- **After a manual login, navigate the page to the `expected_url` the caller passed in — a
  completed login rarely lands there on its own**
- Save updated session state after any successful login
- Refuse to wait for a human when no human can answer (unattended runs) — fail fast instead
- Hold no credentials of any kind

---

## What this module does *not* do

- It does not read `SITE_USERNAME` / `SITE_PASSWORD` — those settings no longer exist
- It does not fill or submit the login form
- It does not know the login form's field names or the submit button's selector
- It does not handle captchas, 2FA, or magic links — the human does, in the open window

The only thing it knows about the site's authentication is how to tell **authenticated**
from **not authenticated**: `SUCCESS_SELECTOR`.

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `LOGIN_URL` | `.env` | Full URL of the dedicated login page — where the human is sent when no login wall is on screen |
| `STATE_FILE` | config | Path to persisted Playwright session (default: `session/state.json`) |
| `SUCCESS_SELECTOR` | config | CSS selector present **only** on authenticated pages (e.g. `.member-content`) — the single source of truth for "logged in" |
| `LOGIN_FORM_SELECTOR` | config | CSS selector for the login form — used **only to recognise a login wall**, never to fill it |
| `LOGIN_URL_FRAGMENT` | config | Substring of the login page URL — secondary login-wall signal |
| `MANUAL_LOGIN_TIMEOUT_MS` | config | How long to wait for the human (default: 300 000 ms = 5 min) |
| `MANUAL_LOGIN_POLL_MS` | config | Poll interval while waiting (default: 1 000 ms) |
| `INTERACTIVE_LOGIN` | `.env` | `auto` \| `always` \| `never` — whether a human can be asked at all |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `state.json` | File (JSON) | Playwright browser storage state (cookies + localStorage) |
| `BrowserContext` | Object | Live authenticated context — passed to `crawler.py` |
| `bool` | Return value of `ensure_authenticated` | `True` if the page is (or became) authenticated **and is at `expected_url`**; `False` otherwise |

---

## Is a human available?

A manual login is worth waiting for only when somebody is at the machine. A `--once` run
under cron at 03:00 has nobody to open a browser window for; blocking it for five minutes
and then failing wastes the window and hides the real problem.

```python
def _human_available() -> bool:
    """True when this run may block waiting for a person to log in."""
    if config.INTERACTIVE_LOGIN == "always":  return True
    if config.INTERACTIVE_LOGIN == "never":   return False
    return sys.stdin.isatty()                 # "auto"
```

`INTERACTIVE_LOGIN=always` is the setting for a Streamlit run: `app.py` runs the ingestion
in a background thread with no tty, but the operator is sitting in front of the same
desktop and can use the window that opens. Set it in `.env` on a desktop install; leave it
at `auto` on a server.

This flag also decides the browser mode: **`headless = not _human_available()`**. A headless
browser cannot be handed to a person, and the decision has to be made at launch, before it
is known whether the restored session is still good. So an interactive run is always
headful — even when it turns out no login is needed and the window is never touched — and
an unattended run is always headless and never waits.

---

## Login-wall detection

Detection uses two signals, in this order, and the order matters:

```
1. SUCCESS_SELECTOR present  → authenticated. Nothing to do. (fast path)

2. SUCCESS_SELECTOR absent AND
   (LOGIN_FORM_SELECTOR present  OR  LOGIN_URL_FRAGMENT in page.url)
                              → login wall. Ask the human.

3. SUCCESS_SELECTOR absent, no login signal
                              → NOT a login wall. Return True; do nothing.
```

Case 3 is what stops a 404, a moved article, or a plain public page from opening a browser
window and stalling the run for five minutes. It is not this module's job to decide such a
page is broken — it says "there is no login problem here", and the crawler's own
`SUCCESS_SELECTOR` guard skips the URL with a WARNING. Never simplify detection down to
"`SUCCESS_SELECTOR` is absent, therefore log in": one dead URL in the email would then
block every run.

`SUCCESS_SELECTOR` presence, not the absence of a form, is what proves success. A login
form can vanish from the DOM for reasons that have nothing to do with a successful login.

| Scenario | Caught by |
|---|---|
| Redirect to a dedicated `/login` URL | `LOGIN_URL_FRAGMENT` in `page.url` |
| Login form embedded in the article page (200, no redirect) | `LOGIN_FORM_SELECTOR` in DOM |
| Login overlay / modal injected over the article | `LOGIN_FORM_SELECTOR` in DOM |

---

## Core Logic

### `get_authenticated_context()`

```
0. config.require_phase2_config()
   Launch Playwright; launch Browser with headless = not _human_available()

1. IF state.json exists AND is readable:
   a. Restore BrowserContext from state.json
   b. Open a new page
   c. Navigate to HEALTH_CHECK_URL, waitUntil="networkidle"
   d. IF SUCCESS_SELECTOR present → session valid
        log INFO "Session valid — no login needed"
        close page; return context
   e. IF login wall detected (rules above) → session expired
        log WARNING "Session expired — manual login required"
        _manual_login(page, reason="session expired")
        save context.storage_state() → state.json
        close page; return context
   f. ELSE → unexpected page state
        log ERROR; raise LoginStateError(page.url, page.title)

2. ELSE (no state.json):
   a. Create fresh BrowserContext
   b. Open a new page
   c. Navigate to LOGIN_URL
   d. _manual_login(page, reason="no saved session")
   e. Save context.storage_state() → state.json
   f. Close page; return context
```

### `ensure_authenticated(page, expected_url)`

Called by the crawler after every `page.goto()`. Operates on the live page in place.
`expected_url` is the URL the crawler passed to `page.goto()` — passed as an argument,
**not** read back from `page.url`, because a `goto` that hit a login wall may already have
redirected the page to `/login`, and "navigate back to `page.url`" would then loop on the
login page. The crawler holds the canonical URL for the request; this function is told it.

```
1. IF SUCCESS_SELECTOR present:
     log DEBUG "Already authenticated — passing"
     return True                                   ← fast path, the common case

2. IF no login signal (case 3 above):
     log DEBUG "No login wall — passing"
     return True                                   ← let the crawler's guard handle it

3. Login wall on this page.
     target_url = expected_url                     ← the argument, never page.url
     log INFO "Login wall detected — manual login required" {page_url: target_url}

4. TRY:
     _manual_login(page, reason="login wall on article URL")
   EXCEPT ManualLoginRequiredError, ManualLoginTimeoutError:
     save debug screenshot → debug/login_fail_{timestamp}.png
     log ERROR "Manual login not completed — skipping URL"
     return False                                  ← NEVER let it propagate

5. Save context.storage_state() → state.json
   (persist immediately, so the remaining URLs in this run need no further login)

6. HAND CONTROL BACK — return the page to where the crawler wanted it.
     IF page.url != target_url:
       await page.goto(target_url, waitUntil="networkidle", timeout=PAGE_TIMEOUT_MS)

     A successful login almost never leaves the browser on the article. The site sends
     the human to a members dashboard, an interstitial, or a "welcome back" page. If this
     step is skipped, the crawler extracts the dashboard's HTML and files it under the
     article's URL — a silent, plausible-looking wrong result that no error surfaces.
     This re-navigation is the handoff. It is not optional.

7. VERIFY the article page is now authenticated:
     IF SUCCESS_SELECTOR present on target_url:
       log INFO "Manual login successful — control returned to crawler"
       return True
     ELSE:
       log ERROR "Authenticated, but article still not accessible"
       return False        ← e.g. logged in on an account without access to this report

8. Never raises. See the contract note under Public Interface.
```

### `_manual_login(page, reason)` (private helper)

```
1. IF NOT _human_available():
     raise ManualLoginRequiredError(
       "A login is required but this run is not interactive. "
       "Run an ingestion from a terminal on the desktop (or set INTERACTIVE_LOGIN=always) "
       "to sign in once; the saved session is reused by later runs.")

2. IF the current page shows no login form (e.g. we were sent here by an expired session
   at HEALTH_CHECK_URL and the site answered with a bare 403 page):
     await page.goto(LOGIN_URL, waitUntil="networkidle")

3. Bring the window forward:  await page.bring_to_front()

4. Tell the human, on BOTH channels — the log file is not where somebody is looking:
     - log INFO "Waiting for manual login" {login_url, timeout_s, reason}
     - print to stderr, unmissable and free of jargon:

         ────────────────────────────────────────────────────────────
          LOGIN REQUIRED
          A browser window is open at:  <page.url>
          Sign in there — password, 2FA, captcha, whatever it asks.
          Crawling resumes by itself the moment you are signed in.
          Waiting up to 5 minutes.  Ctrl-C to abort the run.
         ────────────────────────────────────────────────────────────

5. POLL — this is the whole handoff mechanism:
     deadline = now + MANUAL_LOGIN_TIMEOUT_MS
     WHILE now < deadline:
       IF await page.query_selector(SUCCESS_SELECTOR) is not None:
         log INFO "Login detected", elapsed_s
         print "  ✓ Signed in — resuming." to stderr
         return
       await asyncio.sleep(MANUAL_LOGIN_POLL_MS / 1000)

   The human presses nothing and confirms nothing. Being signed in IS the signal.
   Do not replace this with input(): it deadlocks under Streamlit and in any thread
   without a tty, and it trusts the human's word over the page's state.

   Poll with query_selector in a loop, NOT page.wait_for_selector(timeout=…): the human
   will navigate between login, 2FA and dashboard pages, and a wait_for_selector handle
   bound to a document that gets torn down raises instead of waiting.
   Swallow per-poll navigation errors (the page is mid-load) and keep polling.

6. Timeout with SUCCESS_SELECTOR never seen:
     save debug screenshot → debug/login_timeout_{timestamp}.png
     raise ManualLoginTimeoutError(page.url, timeout_ms)
```

### `close_browser()`

Unchanged from the previous design.

```
Close the Browser and stop the Playwright driver.

get_authenticated_context() starts a Playwright driver process and a Browser, and returns
only the BrowserContext. `await context.close()` closes neither — so a `--once` run under
cron leaks a Chromium and a node process every 12 hours.

The module keeps the Browser and Playwright handles in module-level globals alongside the
context singleton; this closes them in order (context → browser → playwright.stop()) and
is idempotent. The scheduler calls it in the same `finally` block as context.close().
```

---

## Configuration Constants

```python
STATE_FILE              = "session/state.json"
LOGIN_URL               = os.environ.get("LOGIN_URL", "")           # required; validated at use
HEALTH_CHECK_URL        = os.environ.get("HEALTH_CHECK_URL", "")    # required; a members-only page
                                                                   # require_phase2_config() checks both
                                                                   # are set and on SITE_DOMAIN

# Login-wall detection only — NEVER filled or submitted by this module
LOGIN_URL_FRAGMENT      = "/login"
LOGIN_FORM_SELECTOR     = "input[name='username'], input[type='email'], input[name='log']"

# The single source of truth for "authenticated"
SUCCESS_SELECTOR        = os.environ.get("SUCCESS_SELECTOR", ".member-content")

# Human-in-the-loop login
INTERACTIVE_LOGIN       = os.environ.get("INTERACTIVE_LOGIN", "auto")  # auto|always|never
MANUAL_LOGIN_TIMEOUT_MS = 300_000   # 5 min for the human to finish signing in
MANUAL_LOGIN_POLL_MS    = 1_000     # how often to re-check for SUCCESS_SELECTOR

BROWSER_TYPE            = "chromium"
DEBUG_SCREENSHOT_DIR    = "debug/"
```

> There is no `HEADLESS` constant. Headlessness is derived — `headless = not
> _human_available()` — because a run that might need a person must be visible and a run
> that cannot have one should not paint a window on a server. A hardcoded `HEADLESS = True`
> is what makes a manual login impossible; do not reintroduce it.

> `PASSWORD_SELECTOR`, `SUBMIT_SELECTOR`, `SITE_USERNAME`, `SITE_PASSWORD` and
> `LOGIN_TIMEOUT_MS` are **removed**. Nothing in the system types credentials any more.

> `LOGIN_FORM_SELECTOR` may now be loose — it only has to be sensitive enough to recognise
> that a login is being asked for. It no longer has to identify one exact fillable input,
> which is what made it brittle.

---

## Error Handling

| Scenario | Behaviour | Log level |
|---|---|---|
| `state.json` corrupted / unreadable | Delete file; proceed to manual login | WARNING |
| `HEALTH_CHECK_URL` unreachable (network down) | Raise `ConnectionError` | CRITICAL |
| Login needed, `_human_available()` false | Raise `ManualLoginRequiredError` with the "run it from a terminal" hint — **immediately, no waiting** | CRITICAL |
| Human does not finish within `MANUAL_LOGIN_TIMEOUT_MS` | Screenshot; raise `ManualLoginTimeoutError` | ERROR |
| Human aborts with Ctrl-C | `KeyboardInterrupt` propagates; scheduler's `finally` closes the browser | — |
| Login succeeded but the article is still inaccessible | `ensure_authenticated` returns `False`; URL skipped (wrong account / not entitled) | ERROR |
| Unexpected page state at health check (no success marker, no login signal) | Raise `LoginStateError` with page URL and title | ERROR |
| Any of the above inside `ensure_authenticated` | Caught, logged, `False` returned — never propagated | ERROR |

### Exceptions defined by this module

```python
class ManualLoginRequiredError(RuntimeError):
    """A login is needed but this run cannot ask a human (INTERACTIVE_LOGIN=never,
    or no tty under 'auto'). Raised immediately, without waiting."""

class ManualLoginTimeoutError(RuntimeError):
    """A human was asked but SUCCESS_SELECTOR never appeared within
    MANUAL_LOGIN_TIMEOUT_MS."""

class LoginStateError(RuntimeError):
    """A page shows neither SUCCESS_SELECTOR nor any login signal where one of the
    two was required (health check). Carries page URL and title."""
```

`SessionExpiredError` belongs to `scraper/crawler.py`, which raises it when the pre-check
cannot be cleared; this module never raises it.

`AuthenticationError` is gone. It meant "the stored password is wrong", a state that can no
longer exist. Its two callers now see `ManualLoginRequiredError` (nobody there to log in)
or `ManualLoginTimeoutError` (somebody was, but didn't).

---

## Key Dependencies
- `playwright` — async API
- `python-dotenv` — `.env` loading
- `asyncio` — poll loop (stdlib)
- `sys`, `pathlib` (stdlib)

---

## Public Interface

```python
async def get_authenticated_context() -> BrowserContext:
    """Return a live, authenticated Playwright BrowserContext.

    Calls config.require_phase2_config() first. LOGIN_URL is read with
    os.environ.get(..., "") so that `import config` works on a corpus-only install,
    which means the check has to happen here, at point of use. A missing setting then
    raises ConfigError naming it, instead of a Playwright error about navigating to
    the empty string.

    Reuses session/state.json when it still passes the health check. Otherwise opens a
    visible browser window and waits for a human to sign in, then saves the new session.

    Raises: ConnectionError, ManualLoginRequiredError, ManualLoginTimeoutError,
            LoginStateError.
    """

async def ensure_authenticated(page: Page, expected_url: str) -> bool:
    """Resolve any login wall on the current page and leave the page at expected_url.

    Called by the crawler after every page.goto(url), with that same url as
    expected_url — and by the crawler's pre-check with HEALTH_CHECK_URL. The URL is
    an explicit argument, not page.url: a goto that hit a wall may have already
    redirected, and navigating back to page.url would loop on the login page.

    Returns True  — the page shows SUCCESS_SELECTOR and is at expected_url.
                    The crawler may extract the article text.
    Returns True  — (no-op) no login wall was found; the crawler's own guard decides
                    what to do with a page that has no SUCCESS_SELECTOR.
    Returns False — a login wall was found and could not be cleared, or the login
                    succeeded but the article is still inaccessible.

    On success, persists the refreshed session to STATE_FILE so the remaining URLs in
    this run — and later runs — need no further login.

    Never raises. _manual_login() raises ManualLoginRequiredError and
    ManualLoginTimeoutError; this function CATCHES both, logs them, and returns False.
    Without that catch the exception escapes fetch_pages, past the scheduler's
    SessionExpiredError/LoginStateError handlers, so the browser context is never closed
    and the ingestion_runs row is never finished — i.e. one un-cleared login wall would
    wedge the whole system instead of producing the SessionExpiredError the design
    depends on.
    """

async def close_browser() -> None:
    """Close the Browser and stop the Playwright driver. Idempotent.

    Closes context → browser → playwright.stop(). The scheduler calls it in the same
    `finally` block as context.close(), on every Phase 2 exit path.
    """
```

### Teardown contract

| Object | Created by | Closed by |
|---|---|---|
| `Playwright` driver | `get_authenticated_context()` | `close_browser()` |
| `Browser` | `get_authenticated_context()` | `close_browser()` |
| `BrowserContext` | `get_authenticated_context()` | scheduler `finally`, then `close_browser()` |
| `Page` | crawler, per URL | crawler, per URL |

Phase 1 creates none of these. A corpus sync must never import Playwright at all.

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.scraper.login"
```

| Event | Level | Extra fields |
|---|---|---|
| Loading session from state.json | DEBUG | `state_file` |
| Session valid — no login needed | INFO | `state_file`, `health_check_url` |
| Session expired — manual login required | WARNING | `state_file`, `reason` |
| No state.json — manual login required | INFO | `login_url` |
| Browser launched | DEBUG | `headless`, `interactive_login`, `browser_type` |
| Waiting for manual login | INFO | `login_url`, `timeout_s`, `reason` |
| Login detected | INFO | `page_url`, `elapsed_s` |
| Session state saved | DEBUG | `state_file` |
| Returning to pre-login URL | DEBUG | `target_url`, `url_after_login` |
| **ensure_authenticated**: already authenticated — passing | DEBUG | `page_url` |
| **ensure_authenticated**: no login wall — passing | DEBUG | `page_url` |
| **ensure_authenticated**: login wall detected | INFO | `page_url`, `signal` (`form` / `url_fragment`) |
| **ensure_authenticated**: manual login successful — control returned | INFO | `page_url` |
| **ensure_authenticated**: authenticated but article inaccessible | ERROR | `page_url`, `success_selector` |
| Manual login required but run is not interactive | CRITICAL | `page_url`, `interactive_login` |
| Manual login timed out | ERROR | `page_url`, `timeout_ms`, `screenshot_path` |
| Unexpected page state | ERROR | `page_url`, `page_title` |
| state.json corrupted — deleting | WARNING | `state_file`, `error_type` |

All `except` blocks use `exc_info=True` to attach full tracebacks to JSON log entries.

The stderr prompt in `_manual_login` step 4 is a `print`, deliberately not a log call. The
log goes to a JSON file nobody is watching; the person who has to act is looking at the
terminal.

---

## Security Notes
- **No credentials anywhere.** No env vars, no config constants, no arguments, nothing to
  leak into a log line or a traceback. The human types their password into the site's own
  page, in a normal browser window.
- `state.json` contains live session tokens — `chmod 600`, add to `.gitignore`
- Debug screenshots saved to `debug/` may contain a half-filled login page — `.gitignore`
- The headful window is a real browser the operator can see and close. If they close it
  mid-wait, Playwright raises on the next poll; `ensure_authenticated` returns `False` and
  `get_authenticated_context` propagates.

---

## Testing Notes

- Mock Playwright with `pytest-playwright` fixtures; serve login and article pages from a
  local HTTP server. Simulate the human by flipping the fixture server's auth flag from the
  test after N polls — no browser automation of the login form is needed, and none should
  be written, because the production path has none.
- **Valid state.json**: assert `get_authenticated_context` returns without opening `LOGIN_URL`
  and without printing the prompt
- **Expired state.json**: health check serves a login form → assert the prompt is printed,
  the poll loop runs, and `state.json` is rewritten once the fixture flips
- **No state.json**: assert `LOGIN_URL` is opened and the poll loop runs
- **Not interactive** (`INTERACTIVE_LOGIN=never`): assert `ManualLoginRequiredError` is
  raised **without any sleep** — assert total elapsed time is well under
  `MANUAL_LOGIN_POLL_MS`, so a regression that waits first cannot pass
- **Headless derivation**: assert `browser.launch(headless=True)` when
  `INTERACTIVE_LOGIN=never`, `headless=False` when `always`
- **Timeout**: fixture never flips → assert `ManualLoginTimeoutError` and a screenshot on
  disk, with the clock monkeypatched so the test does not actually take five minutes
- **`ensure_authenticated` — authenticated page**: assert `True` returned with no
  navigation and no prompt
- **`ensure_authenticated` — no success marker, no login signal** (a 404): assert `True`
  returned immediately, prompt NOT printed, no window opened. This is the regression test
  for "one dead URL stalls every run"
- **`ensure_authenticated` — login wall, human logs in, site lands on the dashboard**:
  assert the page is navigated to the `expected_url` argument and that the HTML the
  crawler then reads is the ARTICLE's, not the dashboard's. The load-bearing test of this
  spec
- **`ensure_authenticated` — goto already redirected to `/login` before the call**: enter
  with `page.url` == the login URL and `expected_url` == the article URL → assert the
  final page is the article, not the login page (the regression the argument prevents)
- **`ensure_authenticated` — login succeeds but article still walled**: assert `False`
- **`ensure_authenticated` never raises**: parametrise over `ManualLoginRequiredError` and
  `ManualLoginTimeoutError` from `_manual_login` → assert `False` returned, nothing propagates
- Assert no test fixture and no source line references `SITE_USERNAME` or `SITE_PASSWORD`
