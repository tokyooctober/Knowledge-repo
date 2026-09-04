# Evaluation (RAGAS)

Measures **retrieval** (context precision/recall) and **answering** (faithfulness,
answer relevancy, answer correctness) of the RAG pipeline.

Everything runs against local Ollama — no API keys. A local ~14B judge is noisier
than a GPT-4-class one, so **treat every score as relative**: compare runs and
catch regressions, don't publish an absolute "faithfulness = 0.82".

## Why two virtualenvs

RAGAS pulls a `langchain-core` that can clash with the app's `langchain-text-splitters`.
So the pipeline is split and the two halves hand off a JSONL file:

| Phase | venv | script |
|---|---|---|
| 1 — run the system (`retrieve()` + `answer()`) | `.venv` | `eval/run_system.py` |
| 2 — score with RAGAS | `.venv-eval` | `eval/score_ragas.py` |

Generation and review also live in `.venv-eval`.

## One-time setup

```bash
python3 -m venv .venv-eval
.venv-eval/bin/pip install -r eval/requirements-eval.txt
# optional stronger judge:
ollama pull phi4
```

Config knobs (all optional, read from `.env` / environment — see `eval/eval_config.py`):
`EVAL_GEN_MODEL`, `EVAL_JUDGE_MODEL`, `EVAL_BASE_URL`, `EVAL_TESTSET_SIZE`, `EVAL_TOP_K`,
`EVAL_LLM_TIMEOUT`.

## Build the test set (once, then freeze)

```bash
# 0. generate synthetic candidates from a spread of corpus articles
.venv-eval/bin/python eval/build_testset.py --size 60 --articles 40 --seed 7
#    -> eval/dataset/candidates.jsonl

# 1. HUMAN REVIEW — accept / edit / reject each row
.venv-eval/bin/python eval/review_testset.py
#    -> eval/dataset/testset.jsonl   (only accepted rows; this is the frozen set)
git add eval/dataset/testset.jsonl && git commit -m "eval: freeze test set"
```

`review_testset.py` writes decisions back to `candidates.jsonl` after every keystroke,
so it is resumable. `--all` revisits everything; `--rebuild` just regenerates
`testset.jsonl` from the accepted rows.

## Pick the judge (once)

```bash
# copy ~15 rows from any run_*.jsonl into eval/dataset/human_labels.jsonl and add
#   "human_faithful": 0|1 , "human_answer_ok": 0|1
.venv-eval/bin/python eval/calibrate_judge.py --judges qwen2.5:14b-instruct,phi4
# -> set EVAL_JUDGE_MODEL in .env to the winner
```

## Run an evaluation

```bash
# Phase 1 — app venv
.venv/bin/python eval/run_system.py                 # -> eval/results/run_<ts>.jsonl
.venv/bin/python eval/metrics_simple.py eval/results/run_<ts>.jsonl   # quick, no LLM

# Phase 2 — eval venv (slow on CPU; runs the LLM metrics twice and averages)
.venv-eval/bin/python eval/score_ragas.py --results eval/results/run_<ts>.jsonl
# -> eval/results/scores_<ts>.json  +  scores_<ts>.md
```

Smoke test first: `eval/run_system.py --limit 3` then `score_ragas.py --passes 1`.

## Regression check

A deliberately degraded run must score visibly lower:

```bash
.venv/bin/python eval/run_system.py --top-k 1 --tag degraded
.venv-eval/bin/python eval/score_ragas.py --results eval/results/run_<ts>_degraded.jsonl
```

## Files

| file | role |
|---|---|
| `eval_config.py` | env-driven config (models, paths, knobs) |
| `_common.py` | JSONL IO, timestamps, RAGAS model wrappers |
| `build_testset.py` | RAGAS synthetic generation → `candidates.jsonl` |
| `review_testset.py` | human accept/reject → frozen `testset.jsonl` |
| `calibrate_judge.py` | choose `EVAL_JUDGE_MODEL` against ~15 hand labels |
| `run_system.py` | Phase 1: real `retrieve()` + `answer()` → `run_<ts>.jsonl` |
| `metrics_simple.py` | judge-independent hit_rate / MRR / recall@k |
| `score_ragas.py` | Phase 2: RAGAS metrics → `scores_<ts>.{json,md}` |
| `dataset/testset.jsonl` | the frozen eval set (committed) |
| `results/` | run + score outputs (gitignored) |
