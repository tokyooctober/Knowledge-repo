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


def build_chat_llm(model: str):
    """A langchain ChatOpenAI bound to the local Ollama endpoint."""
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
