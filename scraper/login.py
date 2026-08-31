"""Get the browser into an authenticated state and keep it there, then hand control back
to the crawler.

**The login is performed by a human, not by this module.** It never types a username or a
password. It opens a visible window at the page asking for a login, tells the operator to
sign in, and polls the live page until `SUCCESS_SELECTOR` appears. The only thing it knows
about the site's auth is: that selector is on the page, or it is not.

No credentials anywhere — no env vars, no config constants, no arguments.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import config
from config import (
    BROWSER_TYPE,
    DEBUG_SCREENSHOT_DIR,
    HEALTH_CHECK_URL,
    INTERACTIVE_LOGIN,
    LOGIN_FORM_SELECTOR,
    LOGIN_URL,
    LOGIN_URL_FRAGMENT,
    MANUAL_LOGIN_POLL_MS,
    MANUAL_LOGIN_TIMEOUT_MS,
    PAGE_TIMEOUT_MS,
    STATE_FILE,
    SUCCESS_SELECTOR,
)
from logger import get_logger

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page

log = get_logger(__name__)

_playwright = None
_browser = None
_context = None


class ManualLoginRequiredError(RuntimeError):
    """A login is needed but this run cannot ask a human. Raised immediately, no waiting."""


class ManualLoginTimeoutError(RuntimeError):
    """A human was asked but SUCCESS_SELECTOR never appeared within the timeout."""


class LoginStateError(RuntimeError):
    """A page shows neither SUCCESS_SELECTOR nor any login signal where one was required."""


def _human_available() -> bool:
    if INTERACTIVE_LOGIN == "always":
        return True
    if INTERACTIVE_LOGIN == "never":
        return False
    return sys.stdin.isatty()


async def _selector_present(page: Page, selector: str) -> bool:
    """query_selector, tolerant of a page that is mid-navigation (retry once)."""
    for _ in range(2):
        try:
            return await page.query_selector(selector) is not None
        except Exception:  # noqa: BLE001 - page torn down mid-poll; try again
            await asyncio.sleep(MANUAL_LOGIN_POLL_MS / 1000)
    return False


async def _detect(page: Page) -> str:
    """'authenticated' | 'wall' | 'no_signal'."""
    if await _selector_present(page, SUCCESS_SELECTOR):
        return "authenticated"
    if await _selector_present(page, LOGIN_FORM_SELECTOR) or LOGIN_URL_FRAGMENT in page.url:
        return "wall"
    return "no_signal"


async def get_authenticated_context() -> BrowserContext:
    """A live, authenticated Playwright BrowserContext. Reuses session/state.json when it
    still passes the health check; otherwise opens a visible window and waits for a human."""
    global _playwright, _browser, _context
    config.require_phase2_config()

    from playwright.async_api import async_playwright

    _playwright = await async_playwright().start()
    headless = not _human_available()
    _browser = await getattr(_playwright, BROWSER_TYPE).launch(headless=headless)
    log.debug(
        "Browser launched",
        extra={
            "headless": headless,
            "interactive_login": INTERACTIVE_LOGIN,
            "browser_type": BROWSER_TYPE,
        },
    )

    state = Path(STATE_FILE)
    if state.is_file():
        try:
            _context = await _browser.new_context(storage_state=str(state))
        except (ValueError, OSError):
            log.warning("state.json corrupted — deleting", extra={"state_file": STATE_FILE})
            state.unlink(missing_ok=True)
            _context = await _browser.new_context()
        else:
            page = await _context.new_page()
            await page.goto(
                HEALTH_CHECK_URL, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS
            )
            status = await _detect(page)
            if status == "authenticated":
                log.info(
                    "Session valid — no login needed",
                    extra={"state_file": STATE_FILE, "health_check_url": HEALTH_CHECK_URL},
                )
                await page.close()
                return _context
            if status == "wall":
                log.warning(
                    "Session expired — manual login required",
                    extra={"state_file": STATE_FILE, "reason": "health check wall"},
                )
                await _manual_login(page, reason="session expired")
                await _save_state()
                await page.close()
                return _context
            await page.close()
            raise LoginStateError(f"unexpected page state at {HEALTH_CHECK_URL}")
    else:
        _context = await _browser.new_context()

    log.info("No state.json — manual login required", extra={"login_url": LOGIN_URL})
    page = await _context.new_page()
    await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    await _manual_login(page, reason="no saved session")
    await _save_state()
    await page.close()
    return _context


async def ensure_authenticated(page: Page, expected_url: str) -> bool:
    """Resolve any login wall on `page` and leave it at `expected_url`. Called by the
    crawler after every page.goto(url), with that same url. Never raises."""
    status = await _detect(page)
    if status == "authenticated":
        log.debug(
            "ensure_authenticated: already authenticated — passing", extra={"page_url": page.url}
        )
        return True
    if status == "no_signal":
        log.debug("ensure_authenticated: no login wall — passing", extra={"page_url": page.url})
        return True

    log.info("ensure_authenticated: login wall detected", extra={"page_url": expected_url})
    try:
        await _manual_login(page, reason="login wall on article URL")
    except (ManualLoginRequiredError, ManualLoginTimeoutError):
        await _screenshot(page, "login_fail")
        log.error("Manual login not completed — skipping URL", exc_info=True)
        return False

    await _save_state()
    if page.url.rstrip("/") != expected_url.rstrip("/"):
        await page.goto(expected_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)

    if await page.query_selector(SUCCESS_SELECTOR) is not None:
        log.info(
            "ensure_authenticated: manual login successful — control returned",
            extra={"page_url": expected_url},
        )
        return True
    log.error(
        "ensure_authenticated: authenticated but article inaccessible",
        extra={"page_url": expected_url, "success_selector": SUCCESS_SELECTOR},
    )
    return False


async def _manual_login(page: Page, reason: str) -> None:
    if not _human_available():
        raise ManualLoginRequiredError(
            "A login is required but this run is not interactive. Run an ingestion from a "
            "terminal on the desktop (or set INTERACTIVE_LOGIN=always) to sign in once; "
            "the saved session is reused by later runs."
        )

    if (
        not await _selector_present(page, LOGIN_FORM_SELECTOR)
        and LOGIN_URL_FRAGMENT not in page.url
    ):
        await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    try:
        await page.bring_to_front()
    except Exception:  # noqa: BLE001 - headless / detached; not essential
        pass

    timeout_s = MANUAL_LOGIN_TIMEOUT_MS / 1000
    log.info(
        "Waiting for manual login",
        extra={"login_url": page.url, "timeout_s": timeout_s, "reason": reason},
    )
    print(
        "\n" + "─" * 60 + "\n LOGIN REQUIRED\n"
        f" A browser window is open at:  {page.url}\n"
        " Sign in there — password, 2FA, captcha, whatever it asks.\n"
        " Crawling resumes by itself the moment you are signed in.\n"
        f" Waiting up to {int(timeout_s / 60)} minutes.  Ctrl-C to abort the run.\n" + "─" * 60,
        file=sys.stderr,
    )

    deadline = time.monotonic() + timeout_s
    start = time.monotonic()
    while time.monotonic() < deadline:
        if await _selector_present(page, SUCCESS_SELECTOR):
            log.info(
                "Login detected",
                extra={"page_url": page.url, "elapsed_s": round(time.monotonic() - start, 1)},
            )
            print("  ✓ Signed in — resuming.", file=sys.stderr)
            return
        await asyncio.sleep(MANUAL_LOGIN_POLL_MS / 1000)

    await _screenshot(page, "login_timeout")
    raise ManualLoginTimeoutError(f"{page.url} — {MANUAL_LOGIN_TIMEOUT_MS} ms")


async def _save_state() -> None:
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    await _context.storage_state(path=STATE_FILE)
    log.debug("Session state saved", extra={"state_file": STATE_FILE})


async def _screenshot(page: Page, tag: str) -> None:
    Path(DEBUG_SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(DEBUG_SCREENSHOT_DIR) / f"{tag}_{int(time.time())}.png"
    for _ in range(3):  # a reloading page may not be screenshot-stable on the first try
        try:
            await page.screenshot(path=str(path), timeout=3000)
            return
        except Exception:  # noqa: BLE001 - a debug screenshot must never break the flow
            await asyncio.sleep(0.2)


async def close_browser() -> None:
    """Close the Browser and stop the Playwright driver. Idempotent."""
    global _playwright, _browser, _context
    for closer in (
        lambda: _context and _context.close(),
        lambda: _browser and _browser.close(),
        lambda: _playwright and _playwright.stop(),
    ):
        try:
            result = closer()
            if result is not None:
                await result
        except Exception:  # noqa: BLE001
            pass
    _playwright = _browser = _context = None
