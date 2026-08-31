"""Tests for scraper/crawler.py — real headless Chromium against the fixture site.

The load-bearing test is `test_wall_then_dashboard_returns_the_article_html`: a login wall
on the article URL, the "human" signs in, and the crawler must come back with the
ARTICLE's HTML at the article URL — not the welcome-back page.
"""

from __future__ import annotations

import pytest

import config
import scraper.crawler as cr
import scraper.login as lg
from scraper.crawler import SessionExpiredError, fetch_pages
from scraper.login import close_browser
from tests.scraper._server import fixture_site


@pytest.fixture
def site():
    with fixture_site() as s:
        yield s


@pytest.fixture(autouse=True)
def wire(monkeypatch, site, tmp_path):
    monkeypatch.setattr(config, "require_phase2_config", lambda: None)
    for mod in (lg, cr):
        monkeypatch.setattr(mod, "HEALTH_CHECK_URL", site.base + "/members/", raising=False)
        monkeypatch.setattr(mod, "SUCCESS_SELECTOR", ".member-content", raising=False)
    monkeypatch.setattr(lg, "LOGIN_URL", site.base + "/login")
    monkeypatch.setattr(lg, "LOGIN_FORM_SELECTOR", "input[name='username']")
    monkeypatch.setattr(lg, "LOGIN_URL_FRAGMENT", "/login")
    monkeypatch.setattr(lg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(lg, "DEBUG_SCREENSHOT_DIR", str(tmp_path / "debug"))
    monkeypatch.setattr(lg, "MANUAL_LOGIN_POLL_MS", 50)
    monkeypatch.setattr(lg, "MANUAL_LOGIN_TIMEOUT_MS", 8000)
    monkeypatch.setattr(lg, "PAGE_TIMEOUT_MS", 8000)
    monkeypatch.setattr(cr, "PAGE_TIMEOUT_MS", 8000)
    monkeypatch.setattr(cr, "CRAWL_DELAY_MS", 0)
    yield


@pytest.fixture(autouse=True)
async def _teardown():
    yield
    await close_browser()


async def _context(site):
    from playwright.async_api import async_playwright

    lg._playwright = await async_playwright().start()
    lg._browser = await lg._playwright.chromium.launch(headless=True)
    lg._context = await lg._browser.new_context()
    return lg._context


# ── happy path ─────────────────────────────────────────────────────────────


async def test_no_wall_fetches_all_urls(site):
    site.state.authed = True
    ctx = await _context(site)
    pages = await fetch_pages(ctx, [site.base + "/premium-x", site.base + "/premium-y"], set())
    assert {p.url for p in pages} == {site.base + "/premium-x", site.base + "/premium-y"}
    assert all(p.status_code == 200 for p in pages)
    assert all("member-content" in p.html for p in pages)


async def test_is_new_flag_from_known_urls(site):
    site.state.authed = True
    ctx = await _context(site)
    known = {site.base + "/premium-x"}
    pages = await fetch_pages(ctx, [site.base + "/premium-x", site.base + "/premium-y"], known)
    by_url = {p.url: p for p in pages}
    assert by_url[site.base + "/premium-x"].is_new is False
    assert by_url[site.base + "/premium-y"].is_new is True


async def test_duplicate_urls_are_fetched_once(site):
    site.state.authed = True
    ctx = await _context(site)
    pages = await fetch_pages(ctx, [site.base + "/premium-x"] * 3, set())
    assert len(pages) == 1


# ── inline login ───────────────────────────────────────────────────────────


async def test_control_returns_to_the_crawler_after_an_inline_login(site, monkeypatch):
    """ensure_authenticated is called with the crawler's URL, lands the page on a
    welcome-back page, then (its job) navigates back — the crawler must read the ARTICLE."""
    site.state.authed = True
    ctx = await _context(site)
    seen = []

    async def ea(page, expected_url):
        seen.append(expected_url)
        if expected_url.endswith("/premium-x"):
            await page.goto(site.base + "/dashboard")  # login "lands elsewhere"
            await page.goto(expected_url)  # ...then ensure_authenticated navigates back
        return True

    monkeypatch.setattr(lg, "ensure_authenticated", ea)
    pages = await fetch_pages(ctx, [site.base + "/premium-x"], set())
    assert seen[-1] == site.base + "/premium-x"  # called with the crawler's URL
    assert pages[0].url == site.base + "/premium-x"
    assert "Report X" in pages[0].html  # the article, not the dashboard


async def test_nobody_logs_in_skips_that_url_and_continues(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "never")
    site.state.authed = True
    ctx = await _context(site)

    async def wall_then_fail(page, expected_url):
        # premium-y is walled and nobody logs in; premium-x is fine
        if expected_url.endswith("/premium-y"):
            return False
        return True

    monkeypatch.setattr(lg, "ensure_authenticated", wall_then_fail)
    pages = await fetch_pages(ctx, [site.base + "/premium-y", site.base + "/premium-x"], set())
    assert {p.url for p in pages} == {site.base + "/premium-x"}


async def test_dead_url_is_skipped_by_the_guard_no_prompt(site, capsys):
    site.state.authed = True
    ctx = await _context(site)
    pages = await fetch_pages(ctx, [site.base + "/404-page", site.base + "/premium-x"], set())
    assert {p.url for p in pages} == {site.base + "/premium-x"}
    assert "LOGIN REQUIRED" not in capsys.readouterr().err


async def test_pre_login_403_still_records_status_200(site, monkeypatch):
    site.state.authed = True
    ctx = await _context(site)

    async def fake_ea(page, url):
        return True

    async def nav_403(page, url):
        await page.goto(url, wait_until="domcontentloaded")
        return type("R", (), {"status": 403})()

    monkeypatch.setattr(lg, "ensure_authenticated", fake_ea)
    monkeypatch.setattr(cr, "_navigate", nav_403)

    pages = await fetch_pages(ctx, [site.base + "/premium-x"], set())
    assert pages and pages[0].status_code == 200


# ── pre-check ──────────────────────────────────────────────────────────────


async def test_precheck_failure_raises_session_expired_before_the_loop(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "never")
    site.state.authed = False
    ctx = await _context(site)
    with pytest.raises(SessionExpiredError):
        await fetch_pages(ctx, [site.base + "/premium-x"], set())


async def test_empty_urls_raises_value_error(site):
    site.state.authed = True
    ctx = await _context(site)
    with pytest.raises(ValueError, match="empty"):
        await fetch_pages(ctx, [], set())


async def test_precheck_authed_but_no_selector_raises_login_state_error(site, monkeypatch):
    site.state.authed = True
    monkeypatch.setattr(lg, "HEALTH_CHECK_URL", site.base + "/public")  # authed, no member content
    monkeypatch.setattr(cr, "HEALTH_CHECK_URL", site.base + "/public")

    async def ea_true(page, url):
        return True

    monkeypatch.setattr(lg, "ensure_authenticated", ea_true)
    ctx = await _context(site)
    from scraper.login import LoginStateError

    with pytest.raises(LoginStateError):
        await fetch_pages(ctx, [site.base + "/premium-x"], set())


# ── _navigate retry ───────────────────────────────────────────────────────


async def test_navigate_retries_once_on_timeout_then_succeeds(monkeypatch):
    from playwright.async_api import TimeoutError as PWTimeout

    monkeypatch.setattr(cr.asyncio, "sleep", lambda _s: _noop())
    calls = {"n": 0}

    class _Page:
        async def goto(self, url, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PWTimeout("slow")
            return type("R", (), {"status": 200})()

    resp = await cr._navigate(_Page(), "http://x/y")
    assert calls["n"] == 2 and resp.status == 200


async def test_navigate_gives_up_after_the_retry(monkeypatch):
    from playwright.async_api import TimeoutError as PWTimeout

    monkeypatch.setattr(cr.asyncio, "sleep", lambda _s: _noop())

    class _Page:
        async def goto(self, url, **k):
            raise PWTimeout("always slow")

    assert await cr._navigate(_Page(), "http://x/y") is cr._TIMED_OUT


async def _noop():
    return None
