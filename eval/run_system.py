"""Phase 1 — run the real RAG pipeline over the frozen test set.

Runs in the app's `.venv` (needs `config`, `query.*`, Qdrant, the local embedder).
For every row in eval/dataset/testset.jsonl it calls the same `retrieve()` and
`answer()` the app uses, and records what came back. No grading here.

Output: eval/results/run_<ts>.jsonl — one row per question:
    {id, user_input, response, retrieved_contexts, retrieved_urls, retrieved_scores,
     reference, reference_contexts, source_urls, answer_model, num_sources}

    .venv/bin/python eval/run_system.py
    .venv/bin/python eval/run_system.py --top-k 8 --limit 5 --tag smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._common import append_jsonl, read_jsonl, ts
from eval.eval_config import EVAL_TOP_K, RESULTS_DIR, TESTSET_PATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_system")
    ap.add_argument("--top-k", type=int, default=EVAL_TOP_K)
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--tag", default="", help="label baked into the output filename")
    ap.add_argument("--testset", default=str(TESTSET_PATH))
    args = ap.parse_args(argv)

    from logger import configure_logging, get_logger

    configure_logging()
    log = get_logger("eval.run_system")

    rows = read_jsonl(args.testset)
    if not rows:
        raise SystemExit(
            f"No frozen test set at {args.testset} — "
            "run eval/build_testset.py then eval/review_testset.py"
        )
    if args.limit:
        rows = rows[: args.limit]

    from query.answerer import answer
    from query.retriever import retrieve

    suffix = f"_{args.tag}" if args.tag else ""
    out = Path(RESULTS_DIR) / f"run_{ts()}{suffix}.jsonl"
    print(f"Running {len(rows)} questions (top_k={args.top_k}) -> {out}")

    t0 = time.time()
    for n, row in enumerate(rows, 1):
        q = row["user_input"]
        try:
            results = retrieve(q, top_k=args.top_k)
            ans = answer(q, results)
        except Exception:  # noqa: BLE001 - one bad question must not abort the run
            log.error("Question failed", extra={"id": row.get("id"), "q": q}, exc_info=True)
            continue

        append_jsonl(
            out,
            {
                "id": row.get("id", f"q{n:03d}"),
                "user_input": q,
                "response": ans.response,
                "retrieved_contexts": [r.text for r in results],
                "retrieved_urls": [r.article_url for r in results],
                "retrieved_scores": [round(r.score, 4) for r in results],
                "reference": row.get("reference", ""),
                "reference_contexts": row.get("reference_contexts", []),
                "source_urls": row.get("source_urls", []),
                "answer_model": ans.model,
                "num_sources": len(ans.sources),
            },
        )
        print(f"  [{n}/{len(rows)}] {len(results)} ctx, {len(ans.sources)} cited  · {q[:70]}")

    print(f"Done in {time.time() - t0:.0f}s -> {out}")
    print(f"Next (in .venv-eval):  .venv-eval/bin/python eval/score_ragas.py --results {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
