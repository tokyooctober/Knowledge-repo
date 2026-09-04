"""Judge-independent retrieval metrics — no LLM, exact, fast.

Compares the URLs the retriever returned against the reference article URLs the
test set attributes each question to (`source_urls`). Use as a sanity check on the
RAGAS `context_precision` / `context_recall` numbers, which depend on a local judge.

    python eval/metrics_simple.py eval/results/run_<ts>.jsonl
    python eval/results/run_<ts>.jsonl  ->  {hit_rate, mrr, recall_at_k, precision_at_k, n}

Importable: `retrieval_metrics(rows) -> dict`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _first_rank(retrieved: list[str], gold: set[str]) -> int | None:
    for i, url in enumerate(retrieved, 1):
        if url in gold:
            return i
    return None


def retrieval_metrics(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("source_urls")]
    if not scored:
        return {"n": 0, "note": "no rows have source_urls — cannot score retrieval by URL"}

    hits = ranks = rec = prec = 0.0
    for r in scored:
        gold = set(r["source_urls"])
        retrieved = r.get("retrieved_urls", [])
        retrieved_set = set(retrieved)
        rank = _first_rank(retrieved, gold)
        hits += 1.0 if rank else 0.0
        ranks += (1.0 / rank) if rank else 0.0
        rec += len(gold & retrieved_set) / len(gold)
        prec += (len(gold & retrieved_set) / len(retrieved_set)) if retrieved_set else 0.0

    n = len(scored)
    return {
        "n": n,
        "hit_rate": round(hits / n, 4),
        "mrr": round(ranks / n, 4),
        "recall_at_k": round(rec / n, 4),
        "precision_at_k": round(prec / n, 4),
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python eval/metrics_simple.py <run_*.jsonl>")
    rows = [json.loads(line) for line in Path(argv[0]).read_text().splitlines() if line.strip()]
    print(json.dumps(retrieval_metrics(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
