"""Tests for ingestion/embedder.py — batching, model_name passthrough, the query prefix,
and over-long-query truncation. The provider is the deterministic MockEmbeddingProvider.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

import ingestion.embedder as emb
from ingestion.embedder import embed_chunks, embed_query
from models import Chunk, ModelMismatchError


@pytest.fixture
def provider(mock_embedding_provider, monkeypatch):
    monkeypatch.setattr(emb, "get_embedding_provider", lambda: mock_embedding_provider)
    return mock_embedding_provider


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"h_b_{i:04d}",
        article_url="https://example.com/a",
        article_title="A",
        published_at=datetime(2021, 1, 1, tzinfo=UTC),
        tags=[],
        text=f"chunk number {i} about liquidity and rates",
        content_type="body",
        chunk_index=i,
        total_chunks=10,
        word_count=6,
    )


def test_empty_input_returns_empty(provider):
    assert embed_chunks([]) == []
    assert provider.calls == []


def test_output_length_matches_input_and_carries_model_name(provider):
    chunks = [_chunk(i) for i in range(5)]
    result = embed_chunks(chunks)
    assert len(result) == 5
    assert all(ec.model_name == "mock-embed" for ec in result)
    assert all(len(ec.embedding) == 8 for ec in result)
    assert result[0].chunk is chunks[0]


def test_provider_is_called_once_per_batch_not_per_chunk(provider, monkeypatch):
    monkeypatch.setattr(emb, "BATCH_SIZE", 2)
    embed_chunks([_chunk(i) for i in range(5)])
    assert [len(c) for c in provider.calls] == [2, 2, 1]  # 3 batches, not 5


def test_dimension_mismatch_raises(provider, monkeypatch):
    monkeypatch.setattr(provider, "embedding_dim", 1024)  # provider claims 1024, returns 8
    with pytest.raises(ModelMismatchError, match="1024"):
        embed_chunks([_chunk(0)])


def test_embed_query_applies_the_prefix(provider):
    embed_query("what happened to M2")
    assert provider.calls[-1] == ["query: what happened to M2"]


def test_embed_query_returns_a_single_vector(provider):
    vec = embed_query("rates")
    assert isinstance(vec, list) and len(vec) == 8


def test_over_long_query_is_truncated_with_a_warning(provider, monkeypatch, caplog):
    monkeypatch.setattr(emb, "_QUERY_TOKEN_LIMIT", 5)
    with caplog.at_level(logging.WARNING, logger="knowledge_repo.ingestion.embedder"):
        embed_query("one two three four five six seven eight nine ten")
    assert any("truncated" in r.message for r in caplog.records)
    sent = provider.calls[-1][0]
    assert sent.startswith("query: ")
    assert len(sent.split()) < 10  # fewer words than the original after truncation
