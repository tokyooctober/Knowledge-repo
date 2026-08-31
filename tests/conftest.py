"""Shared pytest fixtures and deterministic mock providers.

The mocks satisfy the `llm_provider` Protocols without importing an SDK or making a
network call. Patch the `get_*_provider` factories with these in any test that would
otherwise reach a real model.
"""

from __future__ import annotations

import hashlib

import pytest

from llm_provider import TextResponse


class MockEmbeddingProvider:
    model_name = "mock-embed"
    embedding_dim = 8
    query_prefix = "query: "

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i] / 255.0 for i in range(8)]


class MockTextProvider:
    model_name = "mock-text"

    def __init__(self, response_text: str = "A grounded answer [1].") -> None:
        self.response_text = response_text
        self.calls: list[list[dict]] = []

    def complete(
        self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0
    ) -> TextResponse:
        self.calls.append(messages)
        return TextResponse(
            content=self.response_text,
            model=self.model_name,
            input_tokens=len(str(messages)) // 4,
            output_tokens=len(self.response_text) // 4,
        )


class MockVisionProvider:
    model_name = "mock-vision"
    supports_vision = True

    def __init__(self, response_text: str = "Line chart. X: years. Y: USD trillions.") -> None:
        self.response_text = response_text
        self.calls: list[bytes] = []

    def complete_with_image(
        self, image_bytes: bytes, media_type: str, text_prompt: str, max_tokens: int = 400
    ) -> TextResponse:
        self.calls.append(image_bytes)
        return TextResponse(
            content=self.response_text,
            model=self.model_name,
            input_tokens=50,
            output_tokens=len(self.response_text) // 4,
        )


@pytest.fixture
def mock_text_provider() -> MockTextProvider:
    return MockTextProvider()


@pytest.fixture
def mock_vision_provider() -> MockVisionProvider:
    return MockVisionProvider()


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()
