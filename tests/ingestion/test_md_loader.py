"""Tests for ingestion/md_loader.py — corpus built in tmp_path so image bytes are real.

The load-bearing test is `test_colliding_basenames_resolve_to_their_own_folder`: every
article has an `images/<slug>/00-*.jpg`, and a basename lookup would cross-link them.
"""

from __future__ import annotations

import logging

import pytest
import yaml

import ingestion.md_loader as md
from ingestion.md_loader import (
    CorpusEmptyError,
    CorpusNotFoundError,
    iter_article_paths,
    load_article,
    load_corpus,
)

PNG_A = b"\x89PNG\r\n\x1a\n" + b"AAAA" * 400
PNG_B = b"\x89PNG\r\n\x1a\n" + b"BBBB" * 500


def write_article(corpus, stem, *, fm=None, body="", images=None):
    """Create corpus/<stem>.md and any corpus/images/... files it references."""
    fm = {"title": "Default Title", "url": f"https://www.example.com/{stem}", **(fm or {})}
    fm_text = yaml.safe_dump(fm, sort_keys=False).strip()
    (corpus / f"{stem}.md").write_text(f"---\n{fm_text}\n---\n{body}", encoding="utf-8")
    for rel, data in (images or {}).items():
        f = corpus / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    return d


BODY = "This article discusses monetary policy and the money supply at length. " * 20


# ── iter_article_paths ─────────────────────────────────────────────────────


def test_missing_corpus_raises(tmp_path):
    with pytest.raises(CorpusNotFoundError):
        iter_article_paths(str(tmp_path / "nope"))


def test_empty_corpus_raises(corpus):
    with pytest.raises(CorpusEmptyError):
        iter_article_paths(str(corpus))


def test_only_top_level_md_is_an_article(corpus):
    write_article(corpus, "real", body=BODY)
    (corpus / "images" / "2021-real").mkdir(parents=True)
    (corpus / "images" / "2021-real" / "notes.md").write_text("not an article", encoding="utf-8")
    paths = iter_article_paths(str(corpus))
    assert [p.name for p in paths] == ["real.md"]


# ── metadata resolution ────────────────────────────────────────────────────


def test_title_falls_back_to_h1_then_stem(corpus):
    write_article(corpus, "a", fm={"title": ""}, body=f"# The Real Heading\n\n{BODY}")
    write_article(corpus, "b", fm={"title": ""}, body=BODY)
    assert load_article(corpus / "a.md").title == "The Real Heading"
    assert load_article(corpus / "b.md").title == "b"


def test_missing_url_is_synthesised_with_one_warning(corpus, caplog):
    write_article(corpus, "no-url-2021", fm={"url": ""}, body=BODY)
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.md_loader"):
        art = load_article(corpus / "no-url-2021.md")
    assert art.url == "local:no-url-2021"
    assert sum("synthesised" in r.message for r in caplog.records) == 1


def test_empty_author_falls_back_to_author_name(corpus, monkeypatch):
    monkeypatch.setattr(md, "AUTHOR_NAME", "The Author")
    write_article(corpus, "a", fm={"author": ""}, body=BODY)
    assert load_article(corpus / "a.md").author == "The Author"


def test_trailing_slash_url_hashes_the_same(corpus):
    write_article(corpus, "a", fm={"url": "https://www.example.com/x"}, body=BODY)
    write_article(corpus, "b", fm={"url": "https://example.com/x/?utm=1"}, body=BODY)
    a, b = load_article(corpus / "a.md"), load_article(corpus / "b.md")
    assert a.url == b.url == "https://example.com/x"


def test_unparseable_published_is_none_not_a_crash(corpus):
    write_article(corpus, "a", fm={"published": "sometime in Q3"}, body=BODY)
    assert load_article(corpus / "a.md").published_at is None


# ── images ─────────────────────────────────────────────────────────────────


def test_image_markup_removed_and_local_path_resolves(corpus):
    write_article(
        corpus,
        "a",
        body=f"{BODY}\n\n![chart](images/2021-a/00-c.jpg)\n\n{BODY}",
        images={"images/2021-a/00-c.jpg": PNG_A},
    )
    art = load_article(corpus / "a.md")
    assert len(art.images) == 1
    ref = art.images[0]
    assert ref.local_path.endswith("00-c.jpg")
    assert (corpus / "images/2021-a/00-c.jpg").samefile(ref.local_path)
    assert ref.is_paywall is False and ref.local_path is not None
    assert "images/2021-a/00-c.jpg" not in art.body_text
    assert "![chart]" not in art.body_text


def test_colliding_basenames_resolve_to_their_own_folder(corpus):
    write_article(
        corpus,
        "2021-a",
        body=f"{BODY}\n\n![c](images/2021-a/00-cover.jpg)",
        images={"images/2021-a/00-cover.jpg": PNG_A},
    )
    write_article(
        corpus,
        "2021-b",
        body=f"{BODY}\n\n![c](images/2021-b/00-cover.jpg)",
        images={"images/2021-b/00-cover.jpg": PNG_B},
    )
    a = load_article(corpus / "2021-a.md")
    b = load_article(corpus / "2021-b.md")
    assert "2021-a" in a.images[0].local_path and "2021-b" not in a.images[0].local_path
    assert "2021-b" in b.images[0].local_path and "2021-a" not in b.images[0].local_path
    assert a.content_hash != b.content_hash


def test_missing_image_is_dropped_article_survives(corpus, caplog):
    write_article(corpus, "a", body=f"{BODY}\n\n![x](images/2021-a/gone.jpg)")
    art = load_article(corpus / "a.md")
    assert art is not None
    assert art.images == []


@pytest.mark.parametrize(
    "bad_src",
    [
        "../secrets/x.jpg",
        "/etc/passwd",
        "https://cdn.example.com/x.png",
        "data:image/png;base64,AAAA",
    ],
)
def test_escaping_image_paths_are_skipped(corpus, bad_src):
    write_article(corpus, "a", body=f"{BODY}\n\n![x]({bad_src})")
    art = load_article(corpus / "a.md")
    assert art is not None and art.images == []


def test_cover_referenced_three_times_yields_one_ref(corpus):
    body = (
        f"# Title\n"
        f"![cover](images/2021-a/00-cover.jpg)\n"
        f"*June 27, 2021*\n"
        f'![Premium Feature Image](images/2021-a/00-cover.jpg "https://www.example.com/x.jpg")\n\n'
        f"{BODY}"
    )
    write_article(
        corpus,
        "2021-a",
        fm={"cover": "images/2021-a/00-cover.jpg"},
        body=body,
        images={"images/2021-a/00-cover.jpg": PNG_A},
    )
    art = load_article(corpus / "2021-a.md")
    assert len(art.images) == 1
    assert art.images[0].position == 0


def test_url_title_attribute_is_not_the_caption(corpus):
    body = f'{BODY}\n\n![x](images/2021-a/00-c.jpg "https://www.example.com/original.jpg")'
    write_article(corpus, "2021-a", body=body, images={"images/2021-a/00-c.jpg": PNG_A})
    assert load_article(corpus / "2021-a.md").images[0].caption == ""


def test_bare_date_italic_is_not_the_caption(corpus):
    body = f"{BODY}\n\n![x](images/2021-a/00-c.jpg)\n*June 27, 2021*\n\n{BODY}"
    write_article(corpus, "2021-a", body=body, images={"images/2021-a/00-c.jpg": PNG_A})
    assert load_article(corpus / "2021-a.md").images[0].caption == ""


def test_real_italic_caption_is_kept(corpus):
    body = f"{BODY}\n\n![x](images/2021-a/00-c.jpg)\n*M2 money supply, 2010 to 2024*\n\n{BODY}"
    write_article(corpus, "2021-a", body=body, images={"images/2021-a/00-c.jpg": PNG_A})
    assert load_article(corpus / "2021-a.md").images[0].caption == "M2 money supply, 2010 to 2024"


def test_images_saved_mismatch_warns_once_and_still_loads(corpus, caplog):
    write_article(
        corpus,
        "2021-a",
        fm={"images_saved": 5},
        body=f"{BODY}\n\n![x](images/2021-a/00-c.jpg)",
        images={"images/2021-a/00-c.jpg": PNG_A},
    )
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.md_loader"):
        art = load_article(corpus / "2021-a.md")
    assert art is not None and not art.is_stub
    assert sum("images_saved" in r.message for r in caplog.records) == 1


# ── tables ─────────────────────────────────────────────────────────────────


def test_gfm_table_extracted_verbatim_and_removed_from_body(corpus):
    table = "| Year | Return |\n|------|--------|\n| 2020 | +302% |\n| 2021 | +60% |"
    write_article(corpus, "a", body=f"{BODY}\n\n{table}\n\n{BODY}")
    art = load_article(corpus / "a.md")
    assert len(art.tables_md) == 1
    assert "| Year | Return |" in art.tables_md[0]
    assert "+302%" not in art.body_text


# ── hashing / stub / parse failures ────────────────────────────────────────


def test_content_hash_stable_across_rename_changes_with_body(corpus):
    write_article(corpus, "a", body=BODY)
    h1 = load_article(corpus / "a.md").content_hash
    (corpus / "a.md").rename(corpus / "a-renamed.md")
    assert load_article(corpus / "a-renamed.md").content_hash == h1

    write_article(corpus, "b", body=BODY + " Extra sentence here about liquidity.")
    assert load_article(corpus / "b.md").content_hash != h1


def test_content_hash_changes_when_image_bytes_change(corpus):
    write_article(
        corpus,
        "a",
        body=f"{BODY}\n\n![x](images/2021-a/00-c.jpg)",
        images={"images/2021-a/00-c.jpg": PNG_A},
    )
    before = load_article(corpus / "a.md").content_hash
    (corpus / "images/2021-a/00-c.jpg").write_bytes(PNG_B)
    assert load_article(corpus / "a.md").content_hash != before


def test_stub_returns_article_not_none(corpus):
    write_article(
        corpus,
        "a",
        body="Twenty words or so, definitely well under the hundred word "
        "minimum this loader enforces for a real article body here.",
    )
    art = load_article(corpus / "a.md")
    assert art is not None and art.is_stub is True
    assert len(art.content_hash) == 64


def test_malformed_yaml_returns_none(corpus):
    (corpus / "bad.md").write_text(
        '---\ntitle: "unterminated\n bad: : :\n---\nbody', encoding="utf-8"
    )
    assert load_article(corpus / "bad.md") is None


def test_no_frontmatter_still_loads(corpus, caplog):
    (corpus / "plain.md").write_text(f"# Just A Heading\n\n{BODY}", encoding="utf-8")
    art = load_article(corpus / "plain.md")
    assert art is not None
    assert art.url == "local:plain"
    assert art.title == "Just A Heading"


def test_images_resolve_against_the_articles_own_directory(tmp_path):
    """The same fixture loaded from a copied corpus at a different path — every local_path
    points inside that copy, proving nothing reads the MD_CORPUS_DIR constant."""
    import shutil

    src = tmp_path / "corpus"
    src.mkdir()
    write_article(
        src,
        "2021-a",
        body=f"{BODY}\n\n![x](images/2021-a/00-c.jpg)",
        images={"images/2021-a/00-c.jpg": PNG_A},
    )
    dst = tmp_path / "copy"
    shutil.copytree(src, dst)

    art = load_article(dst / "2021-a.md")
    assert str(dst) in art.images[0].local_path
    assert str(src) not in art.images[0].local_path


def test_load_corpus_drops_none_results(corpus):
    write_article(corpus, "good", body=BODY)
    (corpus / "bad.md").write_text("---\n: : :\n---\n", encoding="utf-8")
    arts = load_corpus(str(corpus))
    assert [a.url for a in arts] == ["https://example.com/good"]


# ── coverage of the quieter paths ─────────────────────────────────────────


def test_non_image_subfolder_is_logged_at_debug(corpus, caplog):
    write_article(corpus, "a", body=BODY)
    (corpus / "drafts").mkdir()
    with caplog.at_level(logging.DEBUG, logger="knowledge_repo.ingestion.md_loader"):
        iter_article_paths(str(corpus))
    assert any("Non-markdown subfolder" in r.message for r in caplog.records)


def test_invalid_utf8_returns_none(corpus):
    (corpus / "bad.md").write_bytes(b"---\ntitle: x\n---\n\xff\xfe not utf-8 \x80")
    assert load_article(corpus / "bad.md") is None


def test_html_img_tag_is_extracted(corpus):
    body = f'{BODY}\n\n<img src="images/2021-a/00-c.jpg" alt="a chart">\n\n{BODY}'
    write_article(corpus, "2021-a", body=body, images={"images/2021-a/00-c.jpg": PNG_A})
    art = load_article(corpus / "2021-a.md")
    assert len(art.images) == 1
    assert art.images[0].alt == "a chart"
    assert "<img" not in art.body_text


def test_very_long_italic_line_is_not_treated_as_a_date(corpus):
    caption = "A very long descriptive caption that runs well past thirty characters here"
    body = f"{BODY}\n\n![x](images/2021-a/00-c.jpg)\n*{caption}*\n\n{BODY}"
    write_article(corpus, "2021-a", body=body, images={"images/2021-a/00-c.jpg": PNG_A})
    assert load_article(corpus / "2021-a.md").images[0].caption == caption


def test_truncated_flag_warns(corpus, caplog):
    write_article(corpus, "a", fm={"truncated": True}, body=BODY)
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.md_loader"):
        load_article(corpus / "a.md")
    assert any("truncated" in r.message.lower() for r in caplog.records)


def test_word_count_divergence_warns(corpus, caplog):
    write_article(corpus, "a", fm={"word_count": 5000}, body=BODY)  # BODY ~180 words
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.md_loader"):
        load_article(corpus / "a.md")
    assert any("word_count diverge" in r.message for r in caplog.records)


def test_fewer_files_written_than_resolved_is_only_debug(corpus, caplog):
    """images_saved counts files the export wrote; the loader counts references, and the
    cover is referenced more than once — declared < resolved is normal, not a WARNING."""
    body = f"{BODY}\n\n![a](images/2021-a/00-a.jpg)\n\n![b](images/2021-a/01-b.jpg)"
    write_article(
        corpus,
        "2021-a",
        fm={"images_saved": 1},
        body=body,
        images={"images/2021-a/00-a.jpg": PNG_A, "images/2021-a/01-b.jpg": PNG_B},
    )
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.md_loader"):
        art = load_article(corpus / "2021-a.md")
    assert len(art.images) == 2
    assert not any("images_saved" in r.message for r in caplog.records)
