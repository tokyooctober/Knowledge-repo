"""Tests for scraper/login.py — real headless Chromium against the threaded fixture site.

The "human signs in" by the fixture flipping its auth flag after N login hits; the login
form auto-reloads so the poll sees `.member-content` without a concurrent task.
"""

from __future__ import annotations

import json
import time

import pytest

import config
import scraper.login as lg
from scraper.login import (
    ManualLoginRequiredError,
    ManualLoginTimeoutError,
    close_browser,
    ensure_authenticated,
    get_authenticated_context,
)
from tests.scraper._server import fixture_site


@pytest.fixture
def site():
    with fixture_site() as s:
        yield s


@pytest.fixture(autouse=True)
def wire(monkeypatch, site, tmp_path):
    monkeypatch.setattr(config, "require_phase2_config", lambda: None)
    monkeypatch.setattr(lg, "HEALTH_CHECK_URL", site.base + "/members/")
    monkeypatch.setattr(lg, "LOGIN_URL", site.base + "/login")
    monkeypatch.setattr(lg, "SUCCESS_SELECTOR", ".member-content")
    monkeypatch.setattr(lg, "LOGIN_FORM_SELECTOR", "input[name='username']")
    monkeypatch.setattr(lg, "LOGIN_URL_FRAGMENT", "/login")
    monkeypatch.setattr(lg, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(lg, "DEBUG_SCREENSHOT_DIR", str(tmp_path / "debug"))
    monkeypatch.setattr(lg, "MANUAL_LOGIN_POLL_MS", 50)
    monkeypatch.setattr(lg, "MANUAL_LOGIN_TIMEOUT_MS", 8000)
    monkeypatch.setattr(lg, "PAGE_TIMEOUT_MS", 8000)
    yield


@pytest.fixture(autouse=True)
async def _teardown():
    yield
    await close_browser()


def _valid_state(path: str):
    from pathlib import Path

    Path(path).write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")


# ── get_authenticated_context ──────────────────────────────────────────────


async def test_valid_state_needs_no_login(site, capsys, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    _valid_state(lg.STATE_FILE)
    site.state.authed = True  # session still good

    ctx = await get_authenticated_context()
    assert ctx is not None
    assert "LOGIN REQUIRED" not in capsys.readouterr().err
    assert "/login" not in "".join(site.state.requests)


async def test_expired_state_triggers_manual_login_and_rewrites_state(site, capsys, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    _valid_state(lg.STATE_FILE)
    site.state.authed = False
    site.state.flip_after = 2  # "human" signs in after a couple of polls

    ctx = await get_authenticated_context()
    assert ctx is not None
    assert "LOGIN REQUIRED" in capsys.readouterr().err
    saved = json.loads(open(lg.STATE_FILE).read())
    assert "cookies" in saved  # state.json was (re)written by playwright


async def test_no_state_opens_login_url_and_waits(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    site.state.flip_after = 2

    ctx = await get_authenticated_context()
    assert ctx is not None
    assert any(r == "/login" for r in site.state.requests)


async def test_non_interactive_raises_immediately_without_waiting(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "never")
    monkeypatch.setattr(lg, "MANUAL_LOGIN_POLL_MS", 5000)  # a wait would be very visible

    started = time.monotonic()
    with pytest.raises(ManualLoginRequiredError):
        await get_authenticated_context()
    assert time.monotonic() - started < 3.0  # launched a browser, but never polled


async def test_headless_is_derived_from_interactivity(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "never")
    with pytest.raises(ManualLoginRequiredError):
        await get_authenticated_context()
    assert lg._browser is not None  # a browser was launched (headless) before raising


async def test_timeout_raises_and_screenshots(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    monkeypatch.setattr(lg, "MANUAL_LOGIN_TIMEOUT_MS", 300)  # never flips -> times out fast
    with pytest.raises(ManualLoginTimeoutError):
        await get_authenticated_context()
    from pathlib import Path

    assert any(Path(lg.DEBUG_SCREENSHOT_DIR).glob("login_timeout_*.png"))


# ── ensure_authenticated ───────────────────────────────────────────────────


async def _fresh_page(site):
    """A context + page, bypassing get_authenticated_context (which we test separately)."""
    from playwright.async_api import async_playwright

    lg._playwright = await async_playwright().start()
    lg._browser = await lg._playwright.chromium.launch(headless=True)
    lg._context = await lg._browser.new_context()
    return await lg._context.new_page()


async def test_ea_authenticated_page_passes_fast(site):
    site.state.authed = True
    page = await _fresh_page(site)
    await page.goto(site.base + "/premium-x")
    assert await ensure_authenticated(page, site.base + "/premium-x") is True


async def test_ea_dead_url_is_not_a_login_wall(site, capsys):
    site.state.authed = True
    page = await _fresh_page(site)
    await page.goto(site.base + "/404-page")
    assert await ensure_authenticated(page, site.base + "/404-page") is True  # no-op
    assert "LOGIN REQUIRED" not in capsys.readouterr().err


async def test_ea_wall_then_dashboard_returns_to_the_article(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    site.state.authed = False
    site.state.flip_after = 2

    page = await _fresh_page(site)
    await page.goto(site.base + "/premium-y")  # lands on the login form
    ok = await ensure_authenticated(page, site.base + "/premium-y")

    assert ok is True
    assert page.url.rstrip("/") == (site.base + "/premium-y").rstrip("/")
    assert "Report Y" in await page.content()  # the ARTICLE, not the welcome-back page


async def test_ea_never_raises_when_non_interactive(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "never")
    site.state.authed = False
    page = await _fresh_page(site)
    await page.goto(site.base + "/premium-x")
    assert await ensure_authenticated(page, site.base + "/premium-x") is False  # not an exception


async def test_ea_navigates_back_when_login_lands_elsewhere(site, monkeypatch):
    """The load-bearing handoff: the poll finishes on /login (the welcome-back page), and
    ensure_authenticated must navigate the page to expected_url before returning."""
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    site.state.flip_after = 2

    page = await _fresh_page(site)
    await page.goto(site.base + "/login")  # the form; poll will land on the welcome-back page here
    ok = await ensure_authenticated(page, site.base + "/premium-x")

    assert ok is True
    assert page.url.rstrip("/") == (site.base + "/premium-x").rstrip("/")
    assert "Report X" in await page.content()


async def test_ea_login_succeeds_but_article_still_walled(site, monkeypatch):
    """Signed in on an account without access to this report → False (not an exception)."""
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    site.state.flip_after = 2

    page = await _fresh_page(site)
    await page.goto(site.base + "/login")
    # expected_url is a path the fixture never serves member content for, even when authed
    ok = await ensure_authenticated(page, site.base + "/public")
    assert ok is False


async def test_corrupted_state_json_is_deleted_and_login_proceeds(site, monkeypatch):
    monkeypatch.setattr(lg, "INTERACTIVE_LOGIN", "always")
    from pathlib import Path

    Path(lg.STATE_FILE).write_text("not json at all", encoding="utf-8")
    site.state.flip_after = 2

    ctx = await get_authenticated_context()
    assert ctx is not None
