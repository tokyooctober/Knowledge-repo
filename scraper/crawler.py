"""Phase 2 only. Given an authenticated BrowserContext and a list of paywalled article
URLs, navigate to each and return the fully rendered HTML.

Login walls are resolved **inline**: after every navigation the page goes to
`login.ensure_authenticated(page, url)`, which (if a wall is there) asks a human to sign
in, navigates back to `url`, and hands control back. When it returns True the page is at
`url` and authenticated — read it. When it returns False the URL is skipped and the run
continues.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from config import CRAWL_DELAY_MS, HEALTH_CHECK_URL, PAGE_TIMEOUT_MS, SUCCESS_SELECTOR
from logger import get_logger
from models import RawPage
from scraper import login
from scraper.login import LoginStateError

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext

log = get_logger(__name__)

_RETRY_SLEEP_S = 5


class SessionExpiredError(RuntimeError):
    """The pre-check login could not be cleared — nobody available, or the wait timed out."""


async def fetch_pages(
    context: BrowserContext,
    urls: list[str],
    known_urls: set[str],
) -> list[RawPage]:
    """Fully rendered HTML for each paywall-protected URL. Pre-checks the session, then
    fetches each URL, resolving login walls inline. Skips a URL whose login fails; never
    aborts the run for one URL. Raises SessionExpiredError / LoginStateError only if the
    pre-check fails."""
    if not urls:
        raise ValueError("urls list must not be empty")

    await _precheck(context)

    deduped = list(dict.fromkeys(urls))
    log.info("Crawl started", extra={"url_count": len(deduped), "known_url_count": len(known_urls)})
    if len(deduped) != len(urls):
        log.debug(
            "Duplicate URLs removed",
            extra={"original_count": len(urls), "deduplicated_count": len(deduped)},
        )

    results: list[RawPage] = []
    counters = {"skipped_login_fail": 0, "skipped_no_selector": 0, "failed_timeout": 0}
    for i, url in enumerate(deduped, start=1):
        page = await context.new_page()
        try:
            response = await _navigate(page, url)
            if response is _TIMED_OUT:
                counters["failed_timeout"] += 1
                continue

            if not await login.ensure_authenticated(page, url):
                log.error("Login wall not cleared — skipping URL", extra={"url": url})
                counters["skipped_login_fail"] += 1
                continue

            if await page.query_selector(SUCCESS_SELECTOR) is None:
                log.warning(
                    "SUCCESS_SELECTOR absent after auth — skipping",
                    extra={"url": url, "success_selector": SUCCESS_SELECTOR},
                )
                counters["skipped_no_selector"] += 1
                continue

            html = await page.content()
            pre_status = response.status if response is not None else 0
            status = pre_status if 200 <= pre_status < 300 else 200
            if status != pre_status:
                log.debug(
                    "Pre-login response was non-2xx — recording 200",
                    extra={"url": url, "pre_login_status": pre_status},
                )
            results.append(
                RawPage(
                    url=url,
                    html=html,
                    fetched_at=datetime.now(UTC),
                    status_code=status,
                    is_new=url not in known_urls,
                )
            )
            log.debug("Article fetched", extra={"url": url, "html_length_chars": len(html)})
        finally:
            await page.close()

        if i < len(deduped):
            await asyncio.sleep(CRAWL_DELAY_MS / 1000)

    log.info("Crawl complete", extra={"fetched": len(results), **counters})
    return results


_TIMED_OUT = object()


async def _navigate(page, url: str):
    for attempt in (1, 2):
        try:
            return await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            if attempt == 1:
                log.warning(
                    "Page timeout — retrying",
                    extra={"url": url, "attempt": attempt, "timeout_ms": PAGE_TIMEOUT_MS},
                )
                await asyncio.sleep(_RETRY_SLEEP_S)
            else:
                log.error(
                    "Page timeout after retry — skipping", extra={"url": url, "attempts": attempt}
                )
    return _TIMED_OUT


async def _precheck(context: BrowserContext) -> None:
    page = await context.new_page()
    try:
        await page.goto(HEALTH_CHECK_URL, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        if not await login.ensure_authenticated(page, HEALTH_CHECK_URL):
            log.critical(
                "Pre-check login failed — aborting", extra={"health_check_url": HEALTH_CHECK_URL}
            )
            raise SessionExpiredError("Pre-check failed — manual login was not completed")
        if await page.query_selector(SUCCESS_SELECTOR) is None:
            log.critical(
                "Pre-check unexpected state — aborting",
                extra={"health_check_url": HEALTH_CHECK_URL},
            )
            raise LoginStateError("Unexpected page state at health check URL")
        log.debug("Pre-check passed", extra={"health_check_url": HEALTH_CHECK_URL})
    finally:
        await page.close()
