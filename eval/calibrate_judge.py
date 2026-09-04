"""Pick the local judge model — the one whose RAGAS scores agree best with you.

One-time. Prep eval/dataset/human_labels.jsonl (≈15 rows) by copying rows from a
Phase-1 run_<ts>.jsonl and adding two binary human columns:

    {..., "human_faithful": 1, "human_answer_ok": 0}

    human_faithful   1 if the answer is fully supported by retrieved_contexts
    human_answer_ok  1 if the answer matches `reference` (correct + complete enough)

Then:

    .venv-eval/bin/python eval/calibrate_judge.py --judges qwen2.5:14b-instruct,phi4

For each judge it runs RAGAS faithfulness + answer_correctness over the labelled
rows and prints agreement@0.5 and Pearson r vs the human columns. Set
EVAL_JUDGE_MODEL in .env to the winner.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._common import read_jsonl
from eval.eval_config import HUMAN_LABELS_PATH


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
    ap.add_argument("--judges", required=True, help="comma-separated Ollama model names")
    ap.add_argument("--labels", default=str(HUMAN_LABELS_PATH))
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
    for judge in [j.strip() for j in args.judges.split(",") if j.strip()]:
        print(f"── {judge} ──")
        out = _evaluate_once(dataset, metrics, judge)
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
        if best is None or score > best[1]:
            best = (judge, score)

    print(f"→ best judge: {best[0]}  (combined agreement {best[1]:.3f})")
    print(f"  set it:  echo 'EVAL_JUDGE_MODEL={best[0]}' >> .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
