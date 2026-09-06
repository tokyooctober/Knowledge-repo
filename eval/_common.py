"""Shared helpers: JSONL IO, timestamps, and (for `.venv-eval` only) the
Langchain LLM / embeddings wrappers RAGAS needs, all pointed at local Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.eval_config import (
    EVAL_API_KEY,
    EVAL_BASE_URL,
    EVAL_EMBEDDING_MODEL,
    EVAL_LLM_TIMEOUT,
)


def ts() -> str:
    """UTC timestamp safe for filenames: 20260904T142530Z."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"not serialisable: {type(obj)}")


# ── RAGAS model wrappers (import only inside .venv-eval) ─────────────────────


class _NoSamplingChatAnthropic:
    """Mixin: strip temperature/top_k/top_p from every outgoing request.

    RAGAS's LangchainLLMWrapper.generate() unconditionally does
    `self.langchain_llm.temperature = 1e-8` (any small positive value) before every
    call, on any langchain_llm that has a `temperature` attribute — it has no idea
    Opus 5 / Sonnet 5 reject sampling params outright now that thinking replaces them
    (400 "temperature is deprecated for this model"). langchain_anthropic doesn't drop
    these for Opus 5 / Sonnet 5 — with `anthropic>=1` installed, its own
    `_route_unsupported_sampling_params` *relocates* them into `payload["extra_body"]`
    instead (the SDK's escape hatch, kept for older models that still accept them via
    that path), so popping the top-level keys alone is a no-op. Strip both places.
    """

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)  # type: ignore[misc]
        for key in ("temperature", "top_k", "top_p"):
            payload.pop(key, None)
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            for key in ("temperature", "top_k", "top_p"):
                extra_body.pop(key, None)
            if not extra_body:
                payload.pop("extra_body", None)
        return payload


def build_chat_llm(model: str):
    """A langchain chat model for `model`.

    Routes to Anthropic (reads `ANTHROPIC_API_KEY`) when `model` is a Claude model id
    (starts with "claude-"); otherwise a ChatOpenAI bound to the local Ollama endpoint.
    Same routing for EVAL_GEN_MODEL and EVAL_JUDGE_MODEL — either can be a Claude model
    independently of the other. See eval/README.md "Using the Anthropic API".
    """
    if model.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic

        class _ChatAnthropicNoSampling(_NoSamplingChatAnthropic, ChatAnthropic):
            pass

        # Thinking explicitly disabled — RAGAS expects `.content` as a plain string;
        # with thinking on, a model may return a content-block list (thinking + text)
        # instead, and whether that happens is prompt-dependent (Opus 5 did on a
        # trivial prompt, Sonnet 5 didn't) — disabling it removes that variability.
        return _ChatAnthropicNoSampling(
            model=model,
            thinking={"type": "disabled"},
            timeout=EVAL_LLM_TIMEOUT,
            max_retries=2,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=EVAL_BASE_URL,
        api_key=EVAL_API_KEY,
        temperature=0.0,
        timeout=EVAL_LLM_TIMEOUT,
        max_retries=2,
    )


def build_embeddings():
    """Local HF embeddings — the same model the app indexes with."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EVAL_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_ragas_llm(model: str):
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(build_chat_llm(model))


def build_ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(build_embeddings())
