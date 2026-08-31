"""Tests for config.py — import must never fail on a Phase-1-only box, and the Phase-2
validators must name exactly what is wrong.
"""

from __future__ import annotations

import importlib

import pytest

import config as config_module
from models import ConfigError

# The env vars config reads that a stray developer .env might set — cleared for every test
# so results do not depend on the machine.
_ENV_KEYS = (
    "MD_CORPUS_DIR",
    "AUTHOR_NAME",
    "TRUSTED_SENDER",
    "EMAIL_SUBJECT_PATTERN",
    "SITE_DOMAIN",
    "IMAP_USERNAME",
    "IMAP_APP_PASSWORD",
    "LOGIN_URL",
    "SUCCESS_SELECTOR",
    "HEALTH_CHECK_URL",
    "INTERACTIVE_LOGIN",
)


@pytest.fixture
def load_config(monkeypatch):
    """Reload config.py under a controlled environment.

    Patches load_dotenv to a no-op so a real .env on the dev's machine cannot leak in,
    clears every var config reads, applies the test's overrides, and reloads. monkeypatch
    restores the environment afterwards; a final reload restores the module.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    def _load(**overrides: str):
        for key in _ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config_module)

    yield _load
    importlib.reload(config_module)


# ── import safety ────────────────────────────────────────────────────────────


def test_imports_with_empty_environment(load_config):
    cfg = load_config()
    assert cfg.LOGIN_URL == ""
    assert cfg.TRUSTED_SENDER == ""
    assert cfg.SITE_DOMAIN == ""
    assert cfg.HEALTH_CHECK_URL == ""


def test_phase1_and_neutral_defaults(load_config):
    cfg = load_config()
    assert cfg.MD_CORPUS_DIR == "corpus"
    assert cfg.AUTHOR_NAME == "the author"  # no real name baked in
    assert cfg.PIPELINE_VERSION == 1
    assert cfg.EMBEDDING_DIM == 1024
    assert cfg.EMAIL_SUBJECT_PATTERN == "premium report"


def test_env_overrides_apply(load_config):
    cfg = load_config(MD_CORPUS_DIR="/data/corpus", AUTHOR_NAME="A. Writer")
    assert cfg.MD_CORPUS_DIR == "/data/corpus"
    assert cfg.AUTHOR_NAME == "A. Writer"


# ── require_phase2_config ────────────────────────────────────────────────────


def test_missing_everything_names_all_four(load_config):
    cfg = load_config()
    with pytest.raises(ConfigError) as exc:
        cfg.require_phase2_config()
    msg = str(exc.value)
    for name in ("LOGIN_URL", "TRUSTED_SENDER", "SITE_DOMAIN", "HEALTH_CHECK_URL"):
        assert name in msg


def test_missing_one_names_only_that_one(load_config):
    cfg = load_config(
        TRUSTED_SENDER="author@example.com",
        SITE_DOMAIN="example.com",
        HEALTH_CHECK_URL="https://www.example.com/members/",
        # LOGIN_URL omitted
    )
    with pytest.raises(ConfigError) as exc:
        cfg.require_phase2_config()
    msg = str(exc.value)
    assert "LOGIN_URL" in msg
    assert "TRUSTED_SENDER" not in msg


def test_cross_check_rejects_url_on_a_different_host(load_config):
    cfg = load_config(
        LOGIN_URL="https://www.example.com/login",
        TRUSTED_SENDER="author@example.com",
        SITE_DOMAIN="example.com",
        HEALTH_CHECK_URL="https://members.OTHERSITE.com/",  # wrong host
    )
    with pytest.raises(ConfigError, match="HEALTH_CHECK_URL"):
        cfg.require_phase2_config()


def test_consistent_config_passes(load_config):
    cfg = load_config(
        LOGIN_URL="https://www.example.com/login",
        TRUSTED_SENDER="author@example.com",
        SITE_DOMAIN="example.com",
        HEALTH_CHECK_URL="https://www.example.com/members/",
    )
    assert cfg.require_phase2_config() is None


# ── phase2_configured ────────────────────────────────────────────────────────


def test_phase2_configured_false_then_true(load_config):
    cfg = load_config()
    assert cfg.phase2_configured() is False

    cfg = load_config(
        LOGIN_URL="https://www.example.com/login",
        TRUSTED_SENDER="author@example.com",
        SITE_DOMAIN="example.com",
        HEALTH_CHECK_URL="https://www.example.com/members/",
    )
    assert cfg.phase2_configured() is True


def test_phase2_configured_never_raises(load_config):
    cfg = load_config(LOGIN_URL="https://www.example.com/login")  # partial
    assert cfg.phase2_configured() is False  # no exception escapes
