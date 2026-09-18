"""Pick the local judge model — the one whose RAGAS scores agree best with you.

One-time. Prep eval/dataset/human_labels.jsonl (≈15 rows) by copying rows from a
Phase-1 run_<ts>.jsonl and adding two binary human columns:

    {..., "human_faithful": 1, "human_answer_ok": 0}

    human_faithful   1 if the answer is fully supported by retrieved_contexts
    human_answer_ok  1 if the answer matches `reference` (correct + complete enough)

Then:

    .venv-eval/bin/python eval/calibrate_judge.py --judges qwen2.5:14b-instruct,phi4

For each judge it runs RAGAS faithfulness + answer_correctness over the labelled
rows, prints agreement@0.5 and Pearson r vs the human columns, and writes:
    eval/results/calibration_<ts>.json   full per-judge scores
    eval/results/calibration_<ts>.md     human-readable summary table
Set EVAL_JUDGE_MODEL in .env to the winner.
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
from eval.eval_config import HUMAN_LABELS_PATH, RESULTS_DIR


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return round(num / den, 3) if den else None


def _agreement(pred: list[float], human: list[int], thresh: float = 0.5) -> float:
    hits = sum(1 for p, h in zip(pred, human, strict=True) if int(p >= thresh) == h)
    return round(hits / len(human), 3) if human else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calibrate_judge")
    ap.add_argument("--judges", required=True, help="comma-separated model ids to compare")
    ap.add_argument("--labels", default=str(HUMAN_LABELS_PATH))
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="concurrent RAGAS judge calls — see score_ragas.py --max-workers help; "
        "a local Ollama judge serves one generation at a time, so keep this low for "
        "those and raise it only for a real API judge (gpt-4o, openai/*, claude-*)",
    )
    args = ap.parse_args(argv)

    rows = read_jsonl(args.labels)
    if not rows:
        raise SystemExit(
            f"no labels at {args.labels} — copy ~15 rows from a run_*.jsonl and add "
            '"human_faithful" / "human_answer_ok" columns'
        )
    missing = [r.get("id") for r in rows if "human_faithful" not in r or "human_answer_ok" not in r]
    if missing:
        raise SystemExit(f"rows missing human columns: {missing}")

    from ragas.metrics import AnswerCorrectness, Faithfulness

    from eval.score_ragas import _evaluate_once, _to_dataset

    dataset = _to_dataset(rows)
    metrics = {"faithfulness": Faithfulness(), "answer_correctness": AnswerCorrectness()}
    h_faith = [int(r["human_faithful"]) for r in rows]
    h_ok = [int(r["human_answer_ok"]) for r in rows]

    print(f"{len(rows)} labelled rows · judges: {args.judges}\n")
    best = None
    per_judge = []
    for judge in [j.strip() for j in args.judges.split(",") if j.strip()]:
        print(f"── {judge} ──")
        out = _evaluate_once(dataset, metrics, judge, max_workers=args.max_workers)
        per = out["per_row"]
        faith = [float(x.get("faithfulness", 0) or 0) for x in per]
        corr = [float(x.get("answer_correctness", 0) or 0) for x in per]

        a_faith = _agreement(faith, h_faith)
        a_ok = _agreement(corr, h_ok)
        score = statistics.fmean([a_faith, a_ok])
        r_faith = _pearson(faith, [float(x) for x in h_faith])
        r_ok = _pearson(corr, [float(x) for x in h_ok])
        print(f"  faithfulness   agree@0.5={a_faith}  pearson_r={r_faith}")
        print(f"  answer_correct agree@0.5={a_ok}  pearson_r={r_ok}")
        print(f"  combined agreement = {score:.3f}\n")
        per_judge.append(
            {
                "judge": judge,
                "faithfulness": {"agree_at_0.5": a_faith, "pearson_r": r_faith},
                "answer_correctness": {"agree_at_0.5": a_ok, "pearson_r": r_ok},
                "combined_agreement": round(score, 3),
            }
        )
        if best is None or score > best[1]:
            best = (judge, score)

    print(f"→ best judge: {best[0]}  (combined agreement {best[1]:.3f})")
    print(f"  set it:  echo 'EVAL_JUDGE_MODEL={best[0]}' >> .env")

    stamp = ts()
    payload = {
        "labels_file": args.labels,
        "scored_at": stamp,
        "n_rows": len(rows),
        "judges": per_judge,
        "best_judge": best[0],
        "best_combined_agreement": round(best[1], 3),
    }
    json_path = Path(RESULTS_DIR) / f"calibration_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md_path = Path(RESULTS_DIR) / f"calibration_{stamp}.md"
    md_path.write_text(_summary_md(payload))

    print(f"\nwrote {json_path}\n      {md_path}")
    return 0


def _summary_md(p: dict) -> str:
    lines = [
        f"# Judge calibration — {p['scored_at']}",
        "",
        f"- labels: `{p['labels_file']}` ({p['n_rows']} rows)",
        f"- **best judge: `{p['best_judge']}`** "
        f"(combined agreement {p['best_combined_agreement']})",
        "",
        "| judge | faithfulness agree@0.5 | faithfulness r "
        "| answer_correctness agree@0.5 | answer_correctness r | combined |",
        "|---|---|---|---|---|---|",
    ]
    for j in p["judges"]:
        f_, c_ = j["faithfulness"], j["answer_correctness"]
        lines.append(
            f"| `{j['judge']}` | {f_['agree_at_0.5']} | {f_['pearson_r']} | "
            f"{c_['agree_at_0.5']} | {c_['pearson_r']} | {j['combined_agreement']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
