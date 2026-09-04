"""Evaluation config — env-driven, all pointing at local Ollama.

Kept separate from the app's `config.py` so the eval harness never widens the
app's config surface. Imports cleanly in both virtualenvs (stdlib + dotenv only).
`EVAL_TOP_K` falls back to the app's `DEFAULT_TOP_K` when that import is available
(Phase 1, `.venv`); in `.venv-eval` it just uses the literal default.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover - .venv-eval always has dotenv via ragas
    pass

_DEFAULT_TOP_K = 6
try:
    from config import DEFAULT_TOP_K as _DEFAULT_TOP_K  # type: ignore[no-redef]
except Exception:  # noqa: BLE001 - app config not importable from .venv-eval
    pass

# ── paths ───────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "dataset"
RESULTS_DIR = EVAL_DIR / "results"
CANDIDATES_PATH = DATASET_DIR / "candidates.jsonl"
TESTSET_PATH = DATASET_DIR / "testset.jsonl"
HUMAN_LABELS_PATH = DATASET_DIR / "human_labels.jsonl"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── local LLM endpoint (Ollama, OpenAI-compatible) ──────────────────────────
EVAL_BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:11434/v1")
EVAL_API_KEY = os.environ.get("EVAL_API_KEY", "ollama")  # Ollama ignores it; must be non-empty

# Model that writes the synthetic test set (one-time).
EVAL_GEN_MODEL = os.environ.get("EVAL_GEN_MODEL", "qwen2.5:14b-instruct")

# Model that scores every eval run (RAGAS judge). Pick with eval/calibrate_judge.py.
EVAL_JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "qwen2.5:14b-instruct")

# Embeddings for answer_relevancy etc. — the same model the app indexes with.
EVAL_EMBEDDING_MODEL = os.environ.get("EVAL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

# ── knobs ───────────────────────────────────────────────────────────────────
TESTSET_SIZE = int(os.environ.get("EVAL_TESTSET_SIZE", "60"))
EVAL_TOP_K = int(os.environ.get("EVAL_TOP_K", str(_DEFAULT_TOP_K)))

# LLM request timeout (seconds) — CPU inference of a 14-24B judge is slow.
EVAL_LLM_TIMEOUT = float(os.environ.get("EVAL_LLM_TIMEOUT", "600"))
