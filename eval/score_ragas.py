"""Phase 2 — score a Phase-1 run with RAGAS. Runs in `.venv-eval`.

Loads eval/results/run_<ts>.jsonl, evaluates retrieval and answer quality with a
local Ollama judge + local BGE embeddings, and writes:
    eval/results/scores_<ts>.json   full aggregate + per-row scores + config
    eval/results/scores_<ts>.md     human-readable summary table

Metrics
    retrieval : context_precision (LLM, w/ reference), context_recall (LLM)
                + judge-independent hit_rate / mrr / recall@k from metrics_simple
    answer    : faithfulness, answer_relevancy, answer_correctness (needs reference)

    .venv-eval/bin/python eval/score_ragas.py --results eval/results/run_<ts>.jsonl
    .venv-eval/bin/python eval/score_ragas.py --results <f> --judge-model phi4 --passes 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._common import read_jsonl, ts
from eval.eval_config import EVAL_JUDGE_MODEL, EVAL_LLM_TIMEOUT, RESULTS_DIR
from eval.metrics_simple import retrieval_metrics


def _load_metrics():
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    return {
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "answer_correctness": AnswerCorrectness(),
    }


def _to_dataset(rows: list[dict]):
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for r in rows:
        if not r.get("retrieved_contexts") and not r.get("response"):
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["user_input"],
                response=r.get("response", ""),
                retrieved_contexts=r.get("retrieved_contexts", []),
                reference=r.get("reference", ""),
                reference_contexts=r.get("reference_contexts", []),
            )
        )
    return EvaluationDataset(samples=samples)


def _evaluate_once(dataset, metrics, judge_model: str, max_workers: int = 4) -> dict:
    from ragas import evaluate
    from ragas.run_config import RunConfig

    from eval._common import build_ragas_embeddings, build_ragas_llm

    # Without an explicit RunConfig, ragas.evaluate() falls back to its own default
    # (up to 16 concurrent jobs, a short per-job timeout unrelated to EVAL_LLM_TIMEOUT).
    # A local Ollama server processes generations one at a time (--np 1) — 16 concurrent
    # RAGAS jobs queue behind that single slot and expire waiting their turn, which
    # looks like the judge model being unreliable when it's really a concurrency
    # mismatch. Match max_workers to what the backend can actually run in parallel
    # (low for local Ollama, higher is fine for a real API judge like gpt-4o), and tie
    # the per-job timeout to EVAL_LLM_TIMEOUT so a legitimately slow local model isn't
    # killed mid-response either.
    run_config = RunConfig(max_workers=max_workers, timeout=EVAL_LLM_TIMEOUT)

    result = evaluate(
        dataset=dataset,
        metrics=list(metrics.values()),
        llm=build_ragas_llm(judge_model),
        embeddings=build_ragas_embeddings(),
        run_config=run_config,
    )
    df = result.to_pandas()
    out = {}
    for name, metric in metrics.items():
        # RAGAS's actual output column is metric.name, which doesn't always match the
        # label we use here — e.g. LLMContextPrecisionWithReference().name is
        # "llm_context_precision_with_reference", not "context_precision". Looking up
        # by our own dict key silently found nothing and dropped the metric with no
        # error (df.columns just never had "context_precision" in it), even though
        # RAGAS had computed it correctly for every row.
        col = getattr(metric, "name", name)
        if col in df.columns:
            vals = [v for v in df[col].tolist() if isinstance(v, (int, float)) and v == v]
            out[name] = round(statistics.fmean(vals), 4) if vals else None
    return {"aggregate": out, "per_row": df.to_dict(orient="records")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="score_ragas")
    ap.add_argument("--results", required=True, help="eval/results/run_<ts>.jsonl from Phase 1")
    ap.add_argument("--judge-model", default=EVAL_JUDGE_MODEL)
    ap.add_argument("--passes", type=int, default=2, help="repeat LLM metrics, report the mean")
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="concurrent RAGAS judge calls (ragas defaults to 16 — too many for a local "
        "Ollama server, which serves one generation at a time: excess jobs queue behind "
        "it and expire against ragas's own per-job timeout, which looks like judge "
        "failure but is really a concurrency mismatch. Raise this for a real API judge "
        "like gpt-4o, which can actually handle concurrent requests)",
    )
    args = ap.parse_args(argv)

    rows = read_jsonl(args.results)
    if not rows:
        raise SystemExit(f"empty results file: {args.results}")

    simple = retrieval_metrics(rows)
    print(f"judge-independent retrieval: {simple}")

    metrics = _load_metrics()
    dataset = _to_dataset(rows)
    is_local = not (
        args.judge_model.startswith("claude-")
        or args.judge_model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4"))
        or "/" in args.judge_model
    )
    provider = "local" if is_local else "hosted"
    print(
        f"scoring {len(dataset)} rows x {len(metrics)} metrics x {args.passes} pass(es) "
        f"with judge={args.judge_model} ({provider}, max_workers={args.max_workers}) …"
    )

    passes = [
        _evaluate_once(dataset, metrics, args.judge_model, max_workers=args.max_workers)
        for _ in range(args.passes)
    ]
    agg = {}
    for name in metrics:
        vals = [p["aggregate"][name] for p in passes if p["aggregate"].get(name) is not None]
        agg[name] = round(statistics.fmean(vals), 4) if vals else None

    stamp = ts()
    payload = {
        "results_file": args.results,
        "scored_at": stamp,
        "judge_model": args.judge_model,
        "passes": args.passes,
        "n_rows": len(dataset),
        "ragas_aggregate": agg,
        "ragas_per_pass": [p["aggregate"] for p in passes],
        "retrieval_simple": simple,
        "per_row_last_pass": passes[-1]["per_row"],
    }
    json_path = Path(RESULTS_DIR) / f"scores_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md = _summary_md(payload)
    md_path = Path(RESULTS_DIR) / f"scores_{stamp}.md"
    md_path.write_text(md)

    print("\n" + md)
    print(f"\nwrote {json_path}\n      {md_path}")
    return 0


def _summary_md(p: dict) -> str:
    a = p["ragas_aggregate"]
    s = p["retrieval_simple"]

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    lines = [
        f"# Eval scores — {p['scored_at']}",
        "",
        f"- results: `{p['results_file']}`",
        f"- judge: `{p['judge_model']}` (local) · passes: {p['passes']} · rows: {p['n_rows']}",
        "- **Local judge → treat as relative. Compare runs, do not report absolutes.**",
        "",
        "## Retrieval",
        "",
        "| metric | score | kind |",
        "|---|---|---|",
        f"| context_precision | {fmt(a.get('context_precision'))} | RAGAS (LLM) |",
        f"| context_recall | {fmt(a.get('context_recall'))} | RAGAS (LLM) |",
        f"| hit_rate@k | {fmt(s.get('hit_rate'))} | exact (URL overlap) |",
        f"| mrr | {fmt(s.get('mrr'))} | exact (URL overlap) |",
        f"| recall@k | {fmt(s.get('recall_at_k'))} | exact (URL overlap) |",
        f"| precision@k | {fmt(s.get('precision_at_k'))} | exact (URL overlap) |",
        "",
        "## Answering",
        "",
        "| metric | score |",
        "|---|---|",
        f"| faithfulness | {fmt(a.get('faithfulness'))} |",
        f"| answer_relevancy | {fmt(a.get('answer_relevancy'))} |",
        f"| answer_correctness | {fmt(a.get('answer_correctness'))} |",
        "",
        f"per-pass RAGAS aggregates: `{p['ragas_per_pass']}`",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
