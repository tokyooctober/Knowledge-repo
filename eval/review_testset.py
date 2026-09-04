"""Human review gate — turn generated candidates into the frozen eval set.

Walks eval/dataset/candidates.jsonl, showing each still-`pending` row. You accept,
reject, edit, or skip. Every decision is written straight back to candidates.jsonl
(so the review is resumable) and eval/dataset/testset.jsonl is regenerated from the
accepted rows after each step.

    .venv-eval/bin/python eval/review_testset.py            # review pending rows
    .venv-eval/bin/python eval/review_testset.py --all      # revisit every row
    .venv-eval/bin/python eval/review_testset.py --rebuild  # just regenerate testset.jsonl

Keys:  a accept   r reject   e edit (in $EDITOR)   s skip   b back   q quit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._common import read_jsonl, write_jsonl
from eval.eval_config import CANDIDATES_PATH, TESTSET_PATH

_ACCEPTED = "accepted"


def _rebuild_testset(rows: list[dict]) -> int:
    kept = [
        {
            "id": r["id"],
            "user_input": r["user_input"],
            "reference": r["reference"],
            "reference_contexts": r.get("reference_contexts", []),
            "source_urls": r.get("source_urls", []),
            "synthesizer": r.get("synthesizer", ""),
        }
        for r in rows
        if r.get("review") == _ACCEPTED
    ]
    write_jsonl(TESTSET_PATH, kept)
    return len(kept)


def _show(row: dict, idx: int, total: int) -> None:
    w = 100
    print("\n" + "=" * w)
    print(
        f"[{idx + 1}/{total}]  id={row['id']}  synth={row.get('synthesizer', '?')}  "
        f"status={row.get('review', 'pending')}"
    )
    print("-" * w)
    print("Q:  " + "\n    ".join(textwrap.wrap(row["user_input"], w - 4)))
    print("\nA:  " + "\n    ".join(textwrap.wrap(row["reference"], w - 4)))
    ctxs = row.get("reference_contexts", [])
    print(f"\nreference_contexts ({len(ctxs)}):")
    for j, c in enumerate(ctxs):
        snippet = c if len(c) < 400 else c[:400] + " …"
        print(f"  [{j}] " + "\n      ".join(textwrap.wrap(snippet, w - 6)))
    if row.get("source_urls"):
        print("\nsource_urls: " + ", ".join(row["source_urls"]))
    print("=" * w)


def _edit(row: dict) -> dict:
    keys = ("user_input", "reference", "reference_contexts", "source_urls")
    editable = {k: row.get(k, [] if k.endswith(("s", "contexts")) else "") for k in keys}
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        json.dump(editable, fh, indent=2, ensure_ascii=False)
        path = fh.name
    editor = os.environ.get("EDITOR", "nano")
    subprocess.call([editor, path])
    try:
        with open(path, encoding="utf-8") as fh:
            patch = json.load(fh)
        row.update({k: patch[k] for k in editable if k in patch})
        print("  edit applied")
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  edit discarded — invalid JSON: {exc}")
    finally:
        os.unlink(path)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="review_testset")
    ap.add_argument("--all", action="store_true", help="revisit every row, not just pending")
    ap.add_argument("--rebuild", action="store_true", help="regenerate testset.jsonl and exit")
    args = ap.parse_args(argv)

    rows = read_jsonl(CANDIDATES_PATH)
    if not rows:
        raise SystemExit(f"No candidates at {CANDIDATES_PATH} — run eval/build_testset.py first")

    if args.rebuild:
        n = _rebuild_testset(rows)
        print(f"Rebuilt {TESTSET_PATH} with {n} accepted rows")
        return 0

    queue = (
        list(range(len(rows)))
        if args.all
        else [i for i, r in enumerate(rows) if r.get("review", "pending") == "pending"]
    )
    if not queue:
        print("Nothing pending. Use --all to revisit, or --rebuild.")
        return 0

    pos = 0
    while 0 <= pos < len(queue):
        i = queue[pos]
        _show(rows[i], pos, len(queue))
        choice = input("[a]ccept [r]eject [e]dit [s]kip [b]ack [q]uit > ").strip().lower()
        if choice == "a":
            rows[i]["review"] = _ACCEPTED
            pos += 1
        elif choice == "r":
            rows[i]["review"] = "rejected"
            pos += 1
        elif choice == "e":
            rows[i] = _edit(rows[i])
        elif choice == "s":
            pos += 1
        elif choice == "b":
            pos = max(0, pos - 1)
        elif choice == "q":
            break
        else:
            print("  ?")
        write_jsonl(CANDIDATES_PATH, rows)
        _rebuild_testset(rows)

    accepted = _rebuild_testset(rows)
    pending = sum(1 for r in rows if r.get("review", "pending") == "pending")
    print(f"\nDone. accepted={accepted}  pending={pending}  -> {TESTSET_PATH}")
    print("Commit it:  git add eval/dataset/testset.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
