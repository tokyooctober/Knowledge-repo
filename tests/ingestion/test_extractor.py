"""Tests for ingestion/extractor.py — HTML -> Article, the Phase-2 counterpart of
md_loader. Fixtures are inline HTML strings; SITE_DOMAIN is patched to example.com.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import ingestion.extractor as ex
from models import RawPage

_SENTENCE = (
    "The money supply expanded by roughly forty percent during the pandemic era, an "
    "unprecedented pace of central-bank balance-sheet growth that reshaped the outlook "
    "for inflation and real yields across the developed world. "
)
# comfortably above MIN_WORD_COUNT after trafilatura trims
PROSE = "<p>" + (_SENTENCE * 5) + "</p><p>" + (_SENTENCE * 5) + "</p>"

DATA_TABLE = (
    "<table><caption>M2 by year</caption>"
    "<tr><th>Year</th><th>M2 (trillions)</th></tr>"
    "<tr><td>2020</td><td>19.1</td></tr>"
    "<tr><td>2021</td><td>21.7</td></tr>"
    "<tr><td>2022</td><td>21.4</td></tr></table>"
)

LAYOUT_TABLE = "<table><tr><td>logo</td></tr></table>"


def _html(*, body_extra="", head_extra="") -> str:
    return (
        f"<html><head><title>M2 Report</title>{head_extra}</head>"
        f"<body><article>{PROSE}{body_extra}</article></body></html>"
    )


def _page(html: str, *, status=200, url="https://www.example.com/premium-2022-3-1") -> RawPage:
    return RawPage(
        url=url,
        html=html,
        fetched_at=datetime(2022, 3, 5, tzinfo=UTC),
        status_code=status,
        is_new=True,
    )


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setattr(ex, "SITE_DOMAIN", "example.com")


# ── basics ─────────────────────────────────────────────────────────────────


def test_non_200_returns_none():
    assert ex.extract(_page(_html(), status=404)) is None


def test_text_only_article():
    art = ex.extract(_page(_html()))
    assert art.source == "web" and art.source_path is None
    assert art.url == "https://example.com/premium-2022-3-1"  # canonicalised
    assert "forty percent" in art.body_text
    assert art.tables_md == [] and art.images == []
    assert not art.is_stub


def test_stub_when_body_is_short():
    art = ex.extract(_page("<html><body><article><p>Too short.</p></article></body></html>"))
    assert art.is_stub is True


# ── tables ─────────────────────────────────────────────────────────────────


def test_data_table_is_markdown_prefixed_and_out_of_body():
    art = ex.extract(_page(_html(body_extra=DATA_TABLE)))
    assert len(art.tables_md) == 1
    assert art.tables_md[0].startswith('[Table 1: "M2 by year"]')
    assert "| 2020" in art.tables_md[0].replace(" ", "") or "2020" in art.tables_md[0]
    assert "19.1" not in art.body_text  # table removed from prose


def test_layout_table_is_skipped():
    art = ex.extract(_page(_html(body_extra=LAYOUT_TABLE)))
    assert art.tables_md == []


def test_multiple_tables_get_sequential_indices():
    art = ex.extract(_page(_html(body_extra=DATA_TABLE + DATA_TABLE)))
    assert art.tables_md[0].startswith("[Table 1")
    assert art.tables_md[1].startswith("[Table 2")


# ── images ─────────────────────────────────────────────────────────────────


_FIGURE = (
    '<figure><img src="/wp-content/chart.png" alt="M2"><figcaption>M2 supply</figcaption></figure>'
)


def test_relative_image_is_resolved_and_flagged_paywall():
    art = ex.extract(_page(_html(body_extra=_FIGURE)))
    assert len(art.images) == 1
    im = art.images[0]
    assert im.src == "https://www.example.com/wp-content/chart.png"
    assert im.local_path is None
    assert im.is_paywall is True
    assert im.caption == "M2 supply"


def test_external_cdn_image_is_not_paywall():
    art = ex.extract(_page(_html(body_extra='<img src="https://cdn.other.com/c.png" alt="x">')))
    assert art.images[0].is_paywall is False


def test_data_uri_image_is_skipped():
    art = ex.extract(_page(_html(body_extra='<img src="data:image/png;base64,AAAA" alt="x">')))
    assert art.images == []


# ── metadata ───────────────────────────────────────────────────────────────


def test_author_and_date_from_meta_tags():
    head = (
        '<meta name="author" content="A. Writer">'
        '<meta property="article:published_time" content="2022-03-01">'
    )
    art = ex.extract(_page(_html(head_extra=head)))
    assert art.author and "Writer" in art.author
    assert art.published_at.year == 2022 and art.published_at.month == 3


def test_title_falls_back_to_the_title_tag():
    art = ex.extract(_page(_html()))
    assert art.title == "M2 Report"


# ── hashing / fallback ─────────────────────────────────────────────────────


def test_content_hash_changes_with_tables_and_images():
    base = ex.extract(_page(_html())).content_hash
    with_table = ex.extract(_page(_html(body_extra=DATA_TABLE))).content_hash
    with_img = ex.extract(
        _page(_html(body_extra='<img src="https://cdn.x.com/a.png" alt="">'))
    ).content_hash
    assert base != with_table != with_img and base != with_img


def test_canonical_url_and_deterministic_hash():
    a = ex.extract(_page(_html(), url="http://www.example.com/premium-2022-3-1/?utm=1"))
    b = ex.extract(_page(_html(), url="https://example.com/premium-2022-3-1"))
    assert a.url == b.url
    assert a.content_hash == b.content_hash


def test_beautifulsoup_fallback_when_trafilatura_is_empty(monkeypatch):
    monkeypatch.setattr(ex.trafilatura, "extract", lambda *a, **k: None)
    art = ex.extract(_page(_html()))
    assert "forty percent" in art.body_text  # recovered via soup.get_text


# ── quieter paths ─────────────────────────────────────────────────────────


def test_table_pandas_failure_falls_back_to_hand_rolled(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(
        "pandas.read_html", lambda *a, **k: (_ for _ in ()).throw(ValueError("merged cells"))
    )
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.extractor"):
        art = ex.extract(_page(_html(body_extra=DATA_TABLE)))
    assert len(art.tables_md) == 1
    assert "| Year | M2 (trillions) |" in art.tables_md[0]  # hand-rolled pipe row
    assert any("fallback used" in r.message for r in caplog.records)


def test_non_http_image_scheme_is_skipped():
    art = ex.extract(_page(_html(body_extra='<img src="ftp://host/x.png" alt="x">')))
    assert art.images == []


def test_www_host_matches_site_domain():
    art = ex.extract(_page(_html(body_extra='<img src="https://www.example.com/a.png" alt="">')))
    assert art.images[0].is_paywall is True


def test_title_from_h1_when_no_title_tag():
    html = f"<html><body><article><h1>Bare H1 Title</h1>{PROSE}</article></body></html>"
    assert ex.extract(_page(html)).title == "Bare H1 Title"


def test_metadata_extraction_failure_is_non_fatal(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("trafilatura choked")

    monkeypatch.setattr(ex.trafilatura, "bare_extraction", boom)
    art = ex.extract(_page(_html()))
    assert art.title == "M2 Report"  # recovered from <title>
    assert art.author is None


def test_tags_come_through_from_metadata(monkeypatch):
    class _Doc:
        title = "M2 Report"
        author = "A. Writer"
        date = "2022-03-01"
        tags = ["macro", "liquidity"]

    monkeypatch.setattr(ex.trafilatura, "bare_extraction", lambda *a, **k: _Doc())
    art = ex.extract(_page(_html()))
    assert art.tags == ["macro", "liquidity"]


# ── parity with md_loader (the test deferred from M2.2) ────────────────────


def test_markdown_and_html_produce_the_same_article_shape(tmp_path, monkeypatch):
    import yaml

    from ingestion.md_loader import load_article

    corpus = tmp_path / "corpus"
    (corpus / "images" / "premium-2022-3-1").mkdir(parents=True)
    (corpus / "images" / "premium-2022-3-1" / "00-c.png").write_bytes(b"\x89PNG" + b"x" * 6000)
    fm = {
        "title": "M2 Report",
        "url": "https://www.example.com/premium-2022-3-1",
        "author": "A. Writer",
        "published": "2022-03-01",
    }
    md_body = (
        "The money supply expanded by roughly forty percent during the pandemic era. " * 8
        + "\n\n![c](images/premium-2022-3-1/00-c.png)"
    )
    (corpus / "premium-2022-3-1.md").write_text(
        f"---\n{yaml.safe_dump(fm).strip()}\n---\n{md_body}", encoding="utf-8"
    )

    md_article = load_article(corpus / "premium-2022-3-1.md")
    html_article = ex.extract(
        _page(
            _html(
                head_extra='<meta name="author" content="A. Writer">'
                '<meta property="article:published_time" content="2022-03-01">',
                body_extra='<img src="/images/premium-2022-3-1/00-c.png" alt="c">',
            )
        )
    )

    import dataclasses

    assert {f.name for f in dataclasses.fields(md_article)} == {
        f.name for f in dataclasses.fields(html_article)
    }
    assert md_article.url == html_article.url
    assert md_article.source == "corpus" and html_article.source == "web"
    assert type(md_article.images[0]) is type(html_article.images[0])
    # downstream sees the same field types
    for field in ("body_text", "tables_md", "images", "word_count", "content_hash"):
        assert type(getattr(md_article, field)) is type(getattr(html_article, field))
