"""Tests for ingestion/chunker.py — body/table/image chunking, the never-split rule for
tables and transcriptions, contiguous indices, and the stub short-circuit.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

import ingestion.chunker as ck
from ingestion.chunker import chunk_article
from models import Article, ImageRef, ImageTranscription

SENTENCE = "The money supply expanded sharply during the stimulus period and then contracted. "
LONG_BODY = SENTENCE * 60  # comfortably many chunks at CHUNK_SIZE=512


def _article(*, body=LONG_BODY, tables=None, is_stub=False, url="https://example.com/a") -> Article:
    return Article(
        url=url,
        title="Article A",
        author="W",
        published_at=datetime(2021, 6, 27, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        tags=["macro"],
        body_text=body,
        tables_md=tables or [],
        images=[],
        word_count=len(body.split()),
        content_hash="h" * 64,
        is_stub=is_stub,
        source="corpus",
        source_path="/c/a.md",
    )


def _transcription(text: str, *, skipped=False, image_type="chart") -> ImageTranscription:
    ref = ImageRef(src="x", local_path="/x", alt="", caption="", position=0, is_paywall=False)
    return ImageTranscription(
        image_ref=ref,
        file_path="/cache/x.png",
        file_hash="fh",
        image_type=image_type,
        transcription=text,
        model="mock-vision",
        input_tokens=1,
        output_tokens=1,
        skipped=skipped,
        skip_reason="photo" if skipped else None,
    )


# ── body ────────────────────────────────────────────────────────────────────


def test_stub_returns_empty(caplog):
    with caplog.at_level(logging.DEBUG, logger="knowledge_repo.ingestion.chunker"):
        assert chunk_article(_article(is_stub=True)) == []


def test_body_chunks_are_typed_and_overlap(caplog):
    chunks = chunk_article(_article())
    body = [c for c in chunks if c.content_type == "body"]
    assert len(body) >= 2
    assert all(c.chunk_id.split("_")[1] == "b" for c in body)
    # consecutive body chunks share a tail/head (RecursiveCharacterTextSplitter overlap)
    tail = body[0].text[-40:]
    assert any(word in body[1].text for word in tail.split()[:3])


def test_body_indices_are_contiguous_from_zero():
    body = [c for c in chunk_article(_article()) if c.content_type == "body"]
    assert [c.chunk_index for c in body] == list(range(len(body)))


def test_short_body_chunk_is_discarded(caplog):
    art = _article(body="Only a handful of words here, well under thirty.")
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.chunker"):
        chunks = chunk_article(art)
    assert [c for c in chunks if c.content_type == "body"] == []
    assert any("too short" in r.message for r in caplog.records)


# ── tables ──────────────────────────────────────────────────────────────────

_TABLE = (
    "| Year | M2 (trillions) | Change |\n"
    "|------|----------------|--------|\n"
    "| 2019 | 15.3 | +4% |\n"
    "| 2020 | 19.1 | +25% |\n"
    "| 2021 | 21.7 | +14% |\n"
    "| 2022 | 21.4 | -1% |\n"
)


def test_each_table_is_one_unsplit_chunk():
    chunks = chunk_article(_article(tables=[_TABLE, _TABLE]))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) == 2
    assert tables[0].text == _TABLE.strip() or tables[0].text == _TABLE
    assert all(c.chunk_id.split("_")[1] == "t" for c in tables)


def test_oversized_table_warns_but_is_kept(caplog):
    huge = _TABLE + ("| 20XX | 99.9 | +0% |\n" * 400)
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.chunker"):
        chunks = chunk_article(_article(tables=[huge]))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) == 1  # not split
    assert any("oversized" in r.message for r in caplog.records)


def test_tiny_table_is_skipped():
    chunks = chunk_article(_article(tables=["| a | b |\n|---|---|\n| 1 | 2 |"]))
    assert [c for c in chunks if c.content_type == "table"] == []


# ── image transcriptions ────────────────────────────────────────────────────

_TRANSCRIPT = (
    "Line chart. X axis years 2010 to 2024. Y axis trillions of USD. The series grows "
    "steadily then surges after 2020 before a small decline through 2023. Key values noted."
)


def test_non_skipped_transcription_becomes_one_chunk():
    chunks = chunk_article(_article(), [_transcription(_TRANSCRIPT)])
    images = [c for c in chunks if c.content_type == "image_transcription"]
    assert len(images) == 1
    assert images[0].chunk_id.split("_")[1] == "i"
    assert images[0].text == _TRANSCRIPT


def test_skipped_transcription_produces_no_chunk():
    chunks = chunk_article(
        _article(),
        [
            _transcription(_TRANSCRIPT),
            _transcription("", skipped=True),
            _transcription(_TRANSCRIPT),
        ],
    )
    images = [c for c in chunks if c.content_type == "image_transcription"]
    assert len(images) == 2
    assert [c.chunk_index for c in images] == [0, 1]  # contiguous despite the skip


def test_none_transcriptions_yields_only_body_and_table():
    chunks = chunk_article(_article(tables=[_TABLE]), image_transcriptions=None)
    assert {c.content_type for c in chunks} == {"body", "table"}


# ── totals ──────────────────────────────────────────────────────────────────


def test_total_chunks_is_the_combined_count():
    chunks = chunk_article(_article(tables=[_TABLE]), [_transcription(_TRANSCRIPT)])
    assert len({c.total_chunks for c in chunks}) == 1
    assert chunks[0].total_chunks == len(chunks)
    body = sum(c.content_type == "body" for c in chunks)
    assert chunks[0].total_chunks == body + 1 + 1


def test_chunk_carries_article_metadata():
    c = chunk_article(_article())[0]
    assert c.article_url == "https://example.com/a"
    assert c.article_title == "Article A"
    assert c.tags == ["macro"]
    assert c.published_at == datetime(2021, 6, 27, tzinfo=UTC)


# ── header splitting (optional, off by default) ─────────────────────────────


def test_header_splitting_path(monkeypatch):
    monkeypatch.setattr(ck, "USE_HEADER_SPLITTING", True)
    body = f"# Macro\n\n{SENTENCE * 20}\n\n## Rates\n\n{SENTENCE * 20}"
    chunks = chunk_article(_article(body=body))
    assert [c for c in chunks if c.content_type == "body"]
    assert all(c.total_chunks == len(chunks) for c in chunks)


# ── edges ──────────────────────────────────────────────────────────────────


def test_empty_body_and_no_tables_produces_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.chunker"):
        assert chunk_article(_article(body="")) == []
    assert any("produced nothing" in r.message for r in caplog.records)


def test_short_transcription_is_dropped():
    chunks = chunk_article(_article(), [_transcription("too short")])
    assert [c for c in chunks if c.content_type == "image_transcription"] == []
