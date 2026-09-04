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
```

Config knobs (all optional, read from `.env` / environment — see `eval/eval_config.py`):

| env var | default | controls |
|---|---|---|
| `EVAL_GEN_MODEL` | `qwen2.5:14b-instruct` | model that writes the synthetic test set (`build_testset.py`) — one-time |
| `EVAL_JUDGE_MODEL` | `qwen2.5:14b-instruct` | model that scores every eval run (`score_ragas.py`, `calibrate_judge.py`) |
| `EVAL_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | embeddings for `answer_relevancy` etc. — keep matched to the app's index |
| `EVAL_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `EVAL_API_KEY` | `ollama` | ignored by Ollama, must be non-empty |
| `EVAL_TESTSET_SIZE` | `60` | default `--size` for `build_testset.py` |
| `EVAL_TOP_K` | app's `DEFAULT_TOP_K` (6) | retrieval depth in `run_system.py` |
| `EVAL_LLM_TIMEOUT` | `600` | per-request timeout (s) — raise for bigger models on CPU |

## Using a stronger judge / generator model

The defaults (`qwen2.5:14b-instruct`, ~9 GB) fit a 16 GB machine with room to spare.
A larger model gives steadier RAGAS scores; the trade-off is RAM and CPU time.

Candidates, all via Ollama:

| model | `ollama pull` | size (Q4) | fits 16 GB? |
|---|---|---|---|
| `phi4` (14B, strong reasoning) | `ollama pull phi4` | ~9 GB | yes, alongside another ~7 GB model |
| `mistral-small` (24B) | `ollama pull mistral-small` | ~14 GB | tight — only with nothing else loaded |
| `gemma2:27b` | `ollama pull gemma2:27b` | ~16 GB | no (swaps; too slow on CPU) |

### To switch — exact steps

1. **Pull it** and confirm the exact tag (the env var must match `ollama list` byte-for-byte):

   ```bash
   ollama pull phi4
   ollama list          # -> use the NAME column verbatim, e.g. "phi4:latest" or "phi4"
   ```

2. **Free VRAM/RAM if you're near the limit** — the ingest vision model may still be resident:

   ```bash
   ollama ps                       # what's loaded now
   ollama stop qwen2.5vl:7b        # unload the vision model; ingest reloads it on demand
   ```

3. **Set the env var.** Persistent — add to `.env` in the repo root:

   ```bash
   # .env
   EVAL_JUDGE_MODEL=phi4     # scores every run
   EVAL_GEN_MODEL=phi4       # only if you also want it to write the test set
   EVAL_LLM_TIMEOUT=1200     # a 24B model on CPU can exceed the 600 s default
   ```

   Or for a single run without editing `.env`:

   ```bash
   EVAL_JUDGE_MODEL=phi4 EVAL_LLM_TIMEOUT=1200 \
     .venv-eval/bin/python eval/score_ragas.py --results eval/results/run_<ts>.jsonl
   ```

   `score_ragas.py --judge-model phi4` overrides it for that invocation only.

4. **Verify the endpoint answers with that model:**

   ```bash
   .venv-eval/bin/python -c "from eval._common import build_chat_llm; \
     print(build_chat_llm('phi4').invoke('reply OK').content)"
   ```

### What each role actually affects

- **`EVAL_JUDGE_MODEL`** takes effect on the *next* `score_ragas.py` run. Scores from
  different judges are **not comparable** — if you change it, re-score any past run you
  still want to compare against (`score_ragas.py --results <old run.jsonl>`), and note the
  judge in `scores_<ts>.md` (it already records `judge_model`).
- **`EVAL_GEN_MODEL`** only matters while generating candidates. Once
  `eval/dataset/testset.jsonl` is frozen and committed, changing it does nothing until you
  re-run `build_testset.py` **and** `review_testset.py` to rebuild the frozen set.
- Don't change **`EVAL_EMBEDDING_MODEL`** unless the app's index (`LOCAL_EMBEDDING_MODEL`
  in `config.py`) changed too — they must match.

### Let calibration make the call

Rather than guessing, run each candidate against ~15 hand-labelled rows and keep the one
that agrees best with you:

```bash
.venv-eval/bin/python eval/calibrate_judge.py --judges qwen2.5:14b-instruct,phi4,mistral-small
# prints combined agreement per model; set EVAL_JUDGE_MODEL in .env to the winner
```

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

Prepare the label file, then run the calibration described in
[Let calibration make the call](#let-calibration-make-the-call):

```bash
# copy ~15 rows from any run_*.jsonl into eval/dataset/human_labels.jsonl and add
# two columns per row:
#   "human_faithful": 0|1   -> 1 if the answer is fully supported by retrieved_contexts
#   "human_answer_ok": 0|1   -> 1 if the answer matches `reference` (correct + complete)
.venv-eval/bin/python eval/calibrate_judge.py --judges qwen2.5:14b-instruct,phi4
# -> set EVAL_JUDGE_MODEL in .env to the model with the highest combined agreement
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
