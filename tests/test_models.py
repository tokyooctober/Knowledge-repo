"""Tests for models.py — the identity helpers and the shared exception hierarchy.

The dataclasses are plain data holders; the load-bearing logic is content_hash() and
canonical_url(), which every article passes through before it reaches storage.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from models import (
    ConfigError,
    ImageRef,
    ModelMismatchError,
    VisionNotSupportedError,
    canonical_url,
    content_hash,
)


def _img(src: str, local_path: str | None) -> ImageRef:
    return ImageRef(
        src=src, local_path=local_path, alt="", caption="", position=0, is_paywall=False
    )


# ── content_hash ─────────────────────────────────────────────────────────────


class TestContentHash:
    def test_deterministic(self):
        args = ("body text", ["| a | b |\n|---|---|\n| 1 | 2 |"], [_img("http://x/c.png", None)])
        assert content_hash(*args) == content_hash(*args)

    def test_changes_with_body_text(self):
        a = content_hash("body one", [], [])
        b = content_hash("body two", [], [])
        assert a != b

    def test_changes_with_tables(self):
        a = content_hash("body", ["| a |\n|---|\n| 1 |"], [])
        b = content_hash("body", ["| a |\n|---|\n| 2 |"], [])
        assert a != b

    def test_web_image_contributes_only_its_url(self, tmp_path):
        """A web ImageRef never touches the filesystem, so a local_path=None image
        hashes purely off src — re-scraping unchanged HTML is stable."""
        first = content_hash("body", [], [_img("https://site/chart.png", None)])
        second = content_hash("body", [], [_img("https://site/chart.png", None)])
        assert first == second

    def test_web_image_url_matters(self):
        a = content_hash("body", [], [_img("https://site/a.png", None)])
        b = content_hash("body", [], [_img("https://site/b.png", None)])
        assert a != b

    def test_path_independent(self, tmp_path):
        """The image's *contents* are hashed, not the article path — content_hash takes
        no path argument, so a renamed .md with identical body/tables/images is stable."""
        f = tmp_path / "chart.png"
        f.write_bytes(b"PNGDATA")
        ref = _img("images/x/00-chart.png", str(f))
        assert content_hash("body", [], [ref]) == content_hash("body", [], [ref])

    def test_corpus_image_bytes_change_changes_hash(self, tmp_path):
        f = tmp_path / "chart.png"
        f.write_bytes(b"small")
        ref = _img("images/x/00-chart.png", str(f))
        before = content_hash("body", [], [ref])

        f.write_bytes(b"a much larger chart payload than before")
        after = content_hash("body", [], [ref])
        assert before != after

    def test_corpus_image_mtime_change_changes_hash(self, tmp_path):
        """Same size, new mtime (an in-place chart swap of identical byte length) still
        forces a re-ingest — mtime_ns is in the formula."""
        f = tmp_path / "chart.png"
        f.write_bytes(b"12345678")
        ref = _img("images/x/00-chart.png", str(f))
        before = content_hash("body", [], [ref])

        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        after = content_hash("body", [], [ref])
        assert before != after

    def test_missing_corpus_image_raises(self, tmp_path):
        """content_hash stats the file; a caller must resolve/drop missing images first
        (md_loader does). A vanished file is a hard error here, not a silent skip."""
        ref = _img("images/x/gone.png", str(tmp_path / "gone.png"))
        with pytest.raises(FileNotFoundError):
            content_hash("body", [], [ref])

    def test_returns_hex_sha256(self):
        h = content_hash("body", [], [])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ── canonical_url ────────────────────────────────────────────────────────────


class TestCanonicalUrl:
    def test_empty_in_empty_out(self):
        assert canonical_url("") == ""

    def test_http_becomes_https(self):
        assert canonical_url("http://example.com/premium-x") == "https://example.com/premium-x"

    def test_strips_www(self):
        assert canonical_url("https://www.example.com/premium-x") == "https://example.com/premium-x"

    def test_strips_trailing_slash(self):
        assert canonical_url("https://example.com/premium-x/") == "https://example.com/premium-x"

    def test_drops_query_and_fragment(self):
        got = canonical_url("https://example.com/premium-x?utm_source=news#section")
        assert got == "https://example.com/premium-x"

    def test_bare_root_keeps_slash(self):
        assert canonical_url("https://example.com") == "https://example.com/"

    def test_strips_surrounding_whitespace(self):
        assert canonical_url("  https://example.com/x  ") == "https://example.com/x"

    def test_corpus_and_email_spellings_collapse(self):
        """The regression the whole normal form exists to prevent: the corpus writes a
        bare URL, the email link carries a trailing slash and a utm_ query, and both must
        resolve to one row."""
        corpus = canonical_url("https://www.example.com/premium-2021-6-27")
        email = canonical_url("https://example.com/premium-2021-6-27/?utm_source=newsletter")
        assert corpus == email == "https://example.com/premium-2021-6-27"

    def test_lowercases_host_only(self):
        got = canonical_url("https://Example.COM/Premium-X")
        assert got == "https://example.com/Premium-X"


# ── exception hierarchy ──────────────────────────────────────────────────────


def test_vision_not_supported_is_a_config_error():
    """get_vision_provider() callers that catch ConfigError must also catch this."""
    assert issubclass(VisionNotSupportedError, ConfigError)


def test_model_mismatch_is_standalone():
    assert not issubclass(ModelMismatchError, ConfigError)


def test_exceptions_are_exceptions():
    for exc in (ConfigError, ModelMismatchError, VisionNotSupportedError):
        assert issubclass(exc, Exception)


# ── dataclasses smoke ────────────────────────────────────────────────────────


def test_dataclasses_construct_by_keyword():
    from models import Answer, Article, Source

    art = Article(
        url="https://example.com/x",
        title="T",
        author="A",
        published_at=None,
        fetched_at=datetime.now(UTC),
        tags=[],
        body_text="",
        tables_md=[],
        images=[],
        word_count=0,
        content_hash="0" * 64,
        is_stub=True,
        source="corpus",
        source_path="/tmp/x.md",
    )
    assert art.source == "corpus"

    ans = Answer(
        query="q",
        response="r",
        sources=[Source(1, "T", "u", None, 0.9)],
        model="m",
        input_tokens=1,
        output_tokens=2,
    )
    assert ans.sources[0].index == 1
