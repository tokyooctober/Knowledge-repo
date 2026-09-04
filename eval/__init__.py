"""RAGAS evaluation harness for the knowledge-repo RAG pipeline.

Two phases, deliberately split across two virtualenvs so RAGAS's dependency tree
never touches the app's `.venv` (the app pins `langchain-text-splitters`):

    eval/run_system.py    Phase 1 — .venv       run retrieve() + answer(), dump JSONL
    eval/score_ragas.py   Phase 2 — .venv-eval  load JSONL, run RAGAS metrics

Everything runs against the local Ollama endpoint — no API keys. See eval/README.md.
"""
