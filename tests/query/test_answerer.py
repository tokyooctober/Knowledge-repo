"""Tests for query/answerer.py — the empty-results short-circuit, the context cap and
truncation, and citation parsing into the Source list.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

import query.answerer as ans
from models import SearchResult


def _result(i: int, *, text="an excerpt about the money supply", url=None) -> SearchResult:
    return SearchResult(
        score=0.9 - i * 0.01,
        text=text,
        chunk_id=f"c{i}",
        article_url=url or f"https://example.com/{i}",
        article_title=f"Article {i}",
        published_at=datetime(2021, 6, 27, tzinfo=UTC),
        tags=[],
        content_type="body",
        chunk_index=i,
    )


@pytest.fixture
def provider(mock_text_provider, monkeypatch):
    monkeypatch.setattr(ans, "get_text_provider", lambda: mock_text_provider)
    return mock_text_provider


def test_empty_results_short_circuits_without_a_provider_call(provider):
    result = ans.answer("what about M2?", [])
    assert result.sources == []
    assert "couldn't find" in result.response
    assert provider.calls == []  # no LLM call


def test_answer_passes_through_model_and_token_counts(provider):
    provider.response_text = "The money supply grew [1]."
    result = ans.answer("q", [_result(1)])
    assert result.model == "mock-text"
    assert result.input_tokens > 0 and result.output_tokens > 0


def test_only_cited_and_in_range_indices_become_sources(provider):
    provider.response_text = "Point one [1]. Point three [3]. A stray [9] cite."
    results = [_result(i) for i in range(1, 5)]  # 4 excerpts
    result = ans.answer("q", results)
    assert [s.index for s in result.sources] == [1, 3]  # 9 dropped, 2/4 uncited


def test_out_of_range_citation_is_logged(provider, caplog):
    provider.response_text = "Answer [5]."
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.query.answerer"):
        ans.answer("q", [_result(1)])
    assert any("Out-of-range citation" in r.message for r in caplog.records)


def test_no_citations_logs_a_warning_and_returns_empty_sources(provider, caplog):
    provider.response_text = "A confident answer with no citations at all."
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.query.answerer"):
        result = ans.answer("q", [_result(1)])
    assert result.sources == []
    assert any("no citation markers" in r.message for r in caplog.records)


def test_context_is_capped_at_max_context_chunks(provider, monkeypatch):
    monkeypatch.setattr(ans, "MAX_CONTEXT_CHUNKS", 6)
    provider.response_text = "cites [1][20]"
    results = [_result(i) for i in range(1, 21)]  # 20 in
    result = ans.answer("q", results)
    # the prompt only saw 6 excerpts, so [20] is out of range and dropped
    assert [s.index for s in result.sources] == [1]
    context = provider.calls[0][1]["content"]
    assert context.count("URL: https://example.com/") == 6


def test_long_excerpt_is_truncated_in_the_context(provider, monkeypatch):
    monkeypatch.setattr(ans, "MAX_CHUNK_CHARS", 50)
    provider.response_text = "ok [1]"
    long_text = "word " * 200
    ans.answer("q", [_result(1, text=long_text)])
    context = provider.calls[0][1]["content"]
    assert " …" in context
    assert len(context) < len(long_text)


def test_system_prompt_carries_the_author_name(provider):
    provider.response_text = "x [1]"
    ans.answer("q", [_result(1)])
    system = provider.calls[0][0]["content"]
    assert ans.AUTHOR_NAME in system
