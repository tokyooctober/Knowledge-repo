# What the eval scores mean

Nine numbers, what each one is actually measuring, and where to expect them to sit.

Written against run `20260907T030102Z` (judge `qwen2.5:14b-instruct`, 50 rows,
2 passes, `top_k=6`) — see `eval/results/scores_20260907T114514Z.md`. The
interpretation is general; the specific values are that run's.

---

## The short version

**The system is honest but under-retrieving.** It doesn't invent things
(faithfulness 0.875, context precision 0.864) and when it finds the right article
it ranks it near the top. Everything downstream is capped by retrieval coverage,
and specifically by questions that need more than one source article — recall
drops from 0.786 to 0.452 there. The one number that looks alarming,
`precision@k` at 0.181, is structurally capped and is not a problem.

---

## The scorecard

### Retrieval

| metric | value | kind | verdict |
|---|---|---|---|
| `context_precision` | 0.864 \* | LLM-judged | solid |
| `context_recall` | 0.704 | LLM-judged | watch |
| `hit_rate@k` | 0.760 | exact | watch |
| `mrr` | 0.591 | exact | solid |
| `recall@k` | 0.637 | exact | watch |
| `precision@k` | 0.181 | exact | capped — see Finding 1 |

### Answering

| metric | value | kind | verdict |
|---|---|---|---|
| `faithfulness` | 0.875 | LLM-judged | solid |
| `answer_relevancy` | 0.648 | LLM-judged | watch |
| `answer_correctness` | 0.625 | LLM-judged | solid (harsh metric) |

\* `context_precision` is a pass-2-only figure, recovered after a mis-keyed
column in `score_ragas.py` silently dropped it from the aggregate (fixed in
`65667b5`). Every other metric varied under 1% between its two passes, so treat
it as a solid estimate rather than a guaranteed match to a clean re-run.

---

## Two families, not one

The report mixes two kinds of measurement that are easy to conflate. They can
disagree sharply while both being right.

**Exact metrics** — `hit_rate`, `mrr`, `recall@k`, `precision@k` — come from
`eval/metrics_simple.py` with no model involved. They compare the URLs of the
retrieved articles against the `source_urls` the test set attributes each
question to. Deterministic and cheap; they only know about article identity, not
meaning.

**LLM-judged metrics** — the five RAGAS ones — have a judge model read the actual
text. They capture semantics the URL comparison can't, but they inherit the
judge's own limits, and ours is a local 14B.

This is why `context_precision` (0.864) and `precision@k` (0.181) coexist without
contradiction: a chunk from an article the test set didn't cite can still be
genuinely useful context. The judge says yes; the URL comparison says no.

---

## Retrieval metrics

### `hit_rate@k` — 0.760 · exact

*Did at least one correct article appear anywhere in the six retrieved chunks?*

Binary per question, then averaged. The crudest "did retrieval work at all"
signal: for 76% of questions, something from the right article made it in. It
sets the ceiling on every answering metric — a question that misses here cannot
be answered from context at all.

### `mrr` — 0.591 · exact

*How high up the list did the first correct article appear?*

Mean Reciprocal Rank. A correct article at position 1 scores 1.0, at position 2
scores 0.5, at position 3 scores 0.33; a complete miss scores 0. So it blends
ranking quality with hit rate.

> Dividing the misses back out — 0.591 ÷ 0.760 — gives a **conditional MRR of
> 0.78**. When the retriever finds the right article at all, it is typically
> placing it at rank 1 or 2. Ranking is not the problem; coverage is.

### `recall@k` — 0.637 · exact

*Of the articles a question actually needs, what fraction did we retrieve?*

For a single-source question this is 0 or 1 and identical to hit rate. For a
two-source question, finding one of the two scores 0.5. The whole gap between
hit rate (0.760) and recall (0.637) is therefore multi-source questions — see
Finding 2.

### `precision@k` — 0.181 · exact

*Of the articles we retrieved, what fraction were the right ones?*

Note the denominator in our implementation: `metrics_simple.py:41` divides by the
number of **unique retrieved URLs**, not by `k`. Six chunks dedupe to between two
and six distinct articles.

Don't optimize this number. Lowering `top_k` would raise it while giving the
answering model less to work with. See Finding 1 for what it actually has room
to be.

### `context_precision` — 0.864 · LLM-judged

*Of the chunks we retrieved, how many were genuinely useful — weighted toward the
top?*

The judge reads each chunk against the reference answer and asks whether it
helped, then weights by rank so useful chunks appearing early count for more.
Because it judges usefulness rather than URL identity, it is **not** a semantic
version of `precision@k` and should never be compared against it. At 0.864 the
context handed to the answering model is overwhelmingly on-topic.

### `context_recall` — 0.704 · LLM-judged

*How much of the reference answer could actually be supported by what we
retrieved?*

RAGAS breaks the reference answer into sentences and checks each one against the
retrieved context. At 0.704, roughly 30% of the information needed for a complete
answer simply wasn't in front of the model — which lines up with the multi-source
coverage gap the exact metrics show. **This is the retrieval number most worth
moving.**

---

## Answering metrics

### `faithfulness` — 0.875 · LLM-judged

*Of the claims in the answer, how many are supported by the retrieved context?*

The hallucination detector, and the metric that matters most for whether a RAG
system can be trusted. RAGAS decomposes the answer into atomic statements and
checks each against the context. At 0.875 the app is overwhelmingly sticking to
its sources rather than filling gaps from pretraining — the harder half of the
problem, and we have it.

### `answer_relevancy` — 0.648 · LLM-judged

*Does the answer address the question that was actually asked?*

Computed backwards: the judge generates questions that the answer would be a good
answer to, then measures embedding similarity against the real question. It
penalizes hedging, padding, and partial answers — the shapes an answer takes when
the context was thin. Expect this to rise with `context_recall`.

### `answer_correctness` — 0.625 · LLM-judged

*Does the answer match the reference answer?*

A blend of factual overlap (F1 over claims against the reference) and plain
semantic similarity.

> Read this one generously. It is harsh by construction: it penalizes correct
> answers phrased differently, and it penalizes answers that add *true*
> information the reference happened to omit. 0.625 does not mean 37.5% wrong —
> it is the least literal of the nine.

---

## Finding 1 — `precision@k` is near its ceiling

**With six chunks retrieved and only one to three correct articles in existence,
perfect retrieval would still score 0.303.**

Across the 50 questions, 28 have a single source article, 21 have two, and one has
three. All 50 retrieve exactly six chunks, deduping to between two and six
distinct articles. A question with one gold article that retrieves six distinct
ones tops out at 1/6 = 0.167 *even with flawless retrieval*.

| | precision@k |
|---|---|
| achieved | **0.181** |
| best possible at k=6 | 0.303 |
| **share of maximum achievable** | **59.9%** |

Best possible is computed per question as
`min(gold articles, unique retrieved articles) / unique retrieved articles`, then
averaged over all 50 rows.

Supporting distributions:

| gold articles per question | 1 | 2 | 3 |
|---|---|---|---|
| questions | 28 | 21 | 1 |

| unique retrieved articles | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| questions | 1 | 4 | 11 | 13 | 21 |

We are at 59.9% of the maximum achievable, not at 18% of a meaningful 100%. This
metric is mostly reporting that `k` is larger than the number of right answers —
which is the intended design.

---

## Finding 2 — multi-source questions are the real gap

**Questions needing two articles find one of them 71% of the time, and both of
them almost never.**

Splitting the exact metrics by how many source articles each question requires
separates a healthy system from a struggling one:

| articles needed | n | `recall@k` | `hit_rate` | `mrr` |
|---|---|---|---|---|
| 1 | 28 | **0.786** | 0.786 | 0.607 |
| 2 | 21 | **0.452** | 0.714 | 0.552 |
| 3 | 1 | 0.333 | 1.000 | 1.000 |

For single-article questions recall and hit rate are the same measure by
definition, so those two columns match exactly. The gap only opens once a
question needs a second article. The 3-article row is a single observation and
carries no signal — reported for completeness only.

This is expected behaviour rather than a bug, and it names the fix. A single
dense-vector query pulls semantically similar chunks, and similar chunks cluster
inside one article — so the retriever confidently returns six passages from the
one newsletter that best matches the question, and never reaches the second one
the question also needs. Since 40 of the 60 generated candidates came from the
multi-hop synthesizers, this shape dominates the test set.

**This is the one actionable retrieval finding.** Query decomposition, a second
retrieval pass, or per-article diversity in the ranking would all attack it.
`context_recall`, `answer_relevancy` and `answer_correctness` are all bottlenecked
on the same missing context and should move together if it lands.

Note that MRR barely degrades between the two groups (0.607 → 0.552), which
confirms this is a coverage problem and not a ranking problem — a better reranker
would not help much.

---

## Reference bands

Working heuristics in common use — soft guidance, not authoritative benchmarks.
Published RAGAS figures vary widely by corpus and question style.

| metric | weak | acceptable | strong | this run |
|---|---|---|---|---|
| `faithfulness` | < 0.70 | 0.80 – 0.90 | > 0.90 | 0.875 |
| `context_precision` | < 0.60 | 0.70 – 0.85 | > 0.85 | 0.864 |
| `context_recall` | < 0.60 | 0.70 – 0.85 | > 0.85 | 0.704 |
| `answer_relevancy` | < 0.60 | 0.65 – 0.80 | > 0.80 | 0.648 |
| `answer_correctness` | < 0.50 | 0.60 – 0.75 | > 0.75 | 0.625 |
| `hit_rate@k` | < 0.70 | 0.75 – 0.90 | > 0.90 | 0.760 |
| `mrr` | < 0.50 | 0.60 – 0.80 | > 0.80 | 0.591 |
| `recall@k` | < 0.60 | 0.70 – 0.85 | > 0.85 | 0.637 |
| `precision@k` | *no useful band — capped by k, see Finding 1* | | | 0.181 |

---

## How to use these numbers

**Compare our own runs, not anyone else's.** That caveat is already printed at the
top of every scores file, and it is the right one: the local 14B judge agreed
with our 16 hand labels 87.5% of the time in `calibrate_judge.py` — better than
`openai/gpt-4o` and `phi4` at 0.812 — but agreement is not ground truth, and 16
labels is a small sample where a single row moves the number.

What the harness is genuinely good for is direction: change one thing in
retrieval, re-run `score_ragas.py`, and read the delta. The absolute values will
drift if the judge changes; the deltas will not.

```bash
.venv-eval/bin/python eval/score_ragas.py \
    --results eval/results/run_<ts>.jsonl \
    --passes 2 --max-workers 3
```

## Reproducing the two findings

Neither analysis is in the harness — both were computed ad hoc from the Phase-1
results file. The inputs are `source_urls` and `retrieved_urls` on each row of
`eval/results/run_<ts>.jsonl`; Finding 1 needs the per-question ceiling
`min(len(gold), len(unique_retrieved)) / len(unique_retrieved)`, and Finding 2
needs the exact metrics grouped by `len(set(source_urls))`. Worth folding into
`metrics_simple.py` if we start tracking them run over run.

## See also

- `eval/README.md` — running the harness, judge/provider configuration
- `eval/metrics_simple.py` — the exact metrics, ~40 lines, worth reading directly
- `eval/score_ragas.py` — RAGAS wiring, `RunConfig` concurrency, aggregation
- `eval/calibrate_judge.py` — how the judge was selected against hand labels
