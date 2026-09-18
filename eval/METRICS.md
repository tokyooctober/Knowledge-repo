# What the eval scores mean

Nine numbers, what each one is actually measuring, and where to expect them to sit.

Written against run `20260918T090849Z` (judge `qwen2.5:14b-instruct`, 50 rows,
2 passes, `top_k=6`) — see `eval/results/scores_20260918T121413Z.md`. The
interpretation is general; the specific values are that run's.

---

## The short version

**Retrieval is no longer the bottleneck; generation is.** Cross-encoder reranking
moved every retrieval metric substantially — `mrr` by 44% — and the judge now
rates the retrieved context highly on both relevance (`context_precision` 0.945)
and coverage (`context_recall` 0.802). The right material reaches the answerer
reliably. Yet the three answering metrics did not move: the answerer is not
turning better context into better answers. The one number that looks alarming,
`precision@k` at 0.238, is structurally capped and is not a problem.

---

## The scorecard

### Retrieval

| metric | value | kind | verdict |
|---|---|---|---|
| `context_precision` | 0.945 | LLM-judged | strong |
| `context_recall` | 0.802 | LLM-judged | solid |
| `hit_rate@k` | 0.880 | exact | solid |
| `mrr` | 0.852 | exact | strong |
| `recall@k` | 0.773 | exact | solid |
| `precision@k` | 0.238 | exact | capped — see Finding 1 |

### Answering

| metric | value | kind | verdict |
|---|---|---|---|
| `faithfulness` | 0.906 | LLM-judged | strong |
| `answer_relevancy` | 0.658 | LLM-judged | watch |
| `answer_correctness` | 0.622 | LLM-judged | watch (harsh metric) |

---

## What changed since 2026-09-07, and what didn't

Same 50 questions, same judge, same 2 passes. `query/answerer.py` was untouched
throughout, so **retrieval was the only variable** — this is an A/B, not a loose
before/after.

| metric | 2026-09-07 | 2026-09-18 | change |
|---|---|---|---|
| `context_precision` | 0.864 \* | 0.945 | +0.081 |
| `context_recall` | 0.704 | 0.802 | +0.098 |
| `hit_rate@k` | 0.760 | 0.880 | +0.120 |
| `mrr` | 0.591 | 0.852 | **+0.261** |
| `recall@k` | 0.637 | 0.773 | +0.136 |
| `precision@k` | 0.181 | 0.238 | +0.057 |
| `faithfulness` | 0.875 | 0.906 | +0.031 |
| `answer_relevancy` | 0.648 | 0.658 | +0.010 |
| `answer_correctness` | 0.625 | 0.622 | −0.003 |

\* the old `context_precision` was a pass-2-only figure recovered after a mis-keyed
column in `score_ragas.py` silently dropped it from the aggregate (fixed in
`65667b5`), so it is the least comparable of the nine.

What produced the retrieval gain: a cross-encoder (`bge-reranker-v2-m3`) that
scores (query, chunk) jointly over the top 48 candidates, instead of ranking by
comparing two independently computed embeddings. Query decomposition was measured
over the same 50 questions and **changed the result on none of them**, so it is
disabled — see `query/SPEC_retriever.md`.

**The answering metrics are flat, and that is a real finding rather than noise.**
The two scoring passes agreed to four decimal places on `context_precision`,
`context_recall` and `answer_relevancy`, and to within 0.008 on the other two — far
tighter than the ±0.107 pass-to-pass spread seen on a 22-row sample. A move under
~0.05 on the answering side should not be read as real, and none of these reach it.

So the constraint has shifted. Further retrieval work — the remaining oracle gap, a
recency prior, chunking — now has clearly diminishing returns. The limiting factor
is the answerer (`qwen2.5:14b-instruct`) or the answer prompt. Note also that
`answer_correctness` grades against RAGAS's synthetic reference answers, which is a
harsh and somewhat artificial target.

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

This is why `context_precision` (0.945) and `precision@k` (0.238) coexist without
contradiction: a chunk from an article the test set didn't cite can still be
genuinely useful context. The judge says yes; the URL comparison says no.

---

## Retrieval metrics

### `hit_rate@k` — 0.880 · exact

*Did at least one correct article appear anywhere in the six retrieved chunks?*

Binary per question, then averaged. The crudest "did retrieval work at all"
signal: for 88% of questions, something from the right article made it in. It
sets the ceiling on every answering metric — a question that misses here cannot
be answered from context at all.

### `mrr` — 0.852 · exact

*How high up the list did the first correct article appear?*

Mean Reciprocal Rank. A correct article at position 1 scores 1.0, at position 2
scores 0.5, at position 3 scores 0.33; a complete miss scores 0. So it blends
ranking quality with hit rate.

> Dividing the misses back out — 0.852 ÷ 0.880 — gives a **conditional MRR of
> 0.97**. When the retriever finds the right article at all, it now places it at
> rank 1 almost every time. This is the metric cross-encoder reranking moved most
> (+0.261), which is what a reranker is for: it cannot add articles the vector
> search never returned, but it is very good at ordering the ones it did.

### `recall@k` — 0.773 · exact

*Of the articles a question actually needs, what fraction did we retrieve?*

For a single-source question this is 0 or 1 and identical to hit rate. For a
two-source question, finding one of the two scores 0.5. The whole gap between
hit rate (0.880) and recall (0.773) is therefore multi-source questions — see
Finding 2.

### `precision@k` — 0.238 · exact

*Of the articles we retrieved, what fraction were the right ones?*

Note the denominator in our implementation: `metrics_simple.py:41` divides by the
number of **unique retrieved URLs**, not by `k`. Six chunks dedupe to between two
and six distinct articles.

Don't optimize this number. Lowering `top_k` would raise it while giving the
answering model less to work with. See Finding 1 for what it actually has room
to be.

### `context_precision` — 0.945 · LLM-judged

*Of the chunks we retrieved, how many were genuinely useful — weighted toward the
top?*

The judge reads each chunk against the reference answer and asks whether it
helped, then weights by rank so useful chunks appearing early count for more.
Because it judges usefulness rather than URL identity, it is **not** a semantic
version of `precision@k` and should never be compared against it. At 0.945 the
context handed to the answering model is almost entirely on-topic.

### `context_recall` — 0.802 · LLM-judged

*How much of the reference answer could actually be supported by what we
retrieved?*

RAGAS breaks the reference answer into sentences and checks each one against the
retrieved context. At 0.802, about 20% of the information needed for a complete
answer still isn't in front of the model — down from 30%, and what remains lines
up with the multi-source coverage gap the exact metrics show.

---

## Answering metrics

### `faithfulness` — 0.906 · LLM-judged

*Of the claims in the answer, how many are supported by the retrieved context?*

The hallucination detector, and the metric that matters most for whether a RAG
system can be trusted. RAGAS decomposes the answer into atomic statements and
checks each against the context. At 0.906 the app is overwhelmingly sticking to
its sources rather than filling gaps from pretraining — the harder half of the
problem, and we have it.

### `answer_relevancy` — 0.658 · LLM-judged

*Does the answer address the question that was actually asked?*

Computed backwards: the judge generates questions that the answer would be a good
answer to, then measures embedding similarity against the real question. It
penalizes hedging, padding, and partial answers — the shapes an answer takes when
the context was thin.

> The 2026-09-07 note here expected this to rise with `context_recall`. It didn't:
> `context_recall` went 0.704 → 0.802 while this moved 0.648 → 0.658. Better
> context did not by itself produce more relevant answers.

### `answer_correctness` — 0.622 · LLM-judged

*Does the answer match the reference answer?*

A blend of factual overlap (F1 over claims against the reference) and plain
semantic similarity.

> Read this one generously. It is harsh by construction: it penalizes correct
> answers phrased differently, and it penalizes answers that add *true*
> information the reference happened to omit. 0.622 does not mean 37.8% wrong —
> it is the least literal of the nine.

---

## Finding 1 — `precision@k` is near its ceiling

**With six chunks retrieved and only one to three correct articles in existence,
perfect retrieval would still score 0.311.**

Across the 50 questions, 28 have a single source article, 21 have two, and one has
three. All 50 retrieve exactly six chunks, deduping to between two and six
distinct articles. A question with one gold article that retrieves six distinct
ones tops out at 1/6 = 0.167 *even with flawless retrieval*.

| | precision@k |
|---|---|
| achieved | **0.238** |
| best possible given the same retrieved-set sizes | 0.311 |
| **share of maximum achievable** | **76.5%** (was 59.9% on 2026-09-07) |

Best possible is computed per question as
`min(gold articles, unique retrieved articles) / unique retrieved articles`, then
averaged over all 50 rows.

Supporting distributions:

| gold articles per question | 1 | 2 | 3 |
|---|---|---|---|
| questions | 28 | 21 | 1 |

| unique retrieved articles | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| questions | 1 | 4 | 12 | 13 | 20 |

We are at 76.5% of the maximum achievable, not at 24% of a meaningful 100%. This
metric is mostly reporting that `k` is larger than the number of right answers —
which is the intended design.

---

## Finding 2 — multi-source questions are the real gap

**Questions needing two articles now find one of them 86% of the time, but still
find both only about a third of the time.**

Splitting the exact metrics by how many source articles each question requires
separates a healthy system from a struggling one:

| articles needed | n | `recall@k` | `hit_rate` | `mrr` |
|---|---|---|---|---|
| 1 | 28 | **0.893** | 0.893 | 0.869 |
| 2 | 21 | **0.619** | 0.857 | 0.821 |
| 3 | 1 | 0.667 | 1.000 | 1.000 |

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

The gap narrowed but did not close: two-article questions went from 0.452 to
0.619 recall, and remain the largest single deficit.

> **A prediction recorded here was wrong, and it is worth keeping visible.** The
> 2026-09-07 version of this section concluded, from MRR degrading only mildly
> between groups (0.607 → 0.552), that "this is a coverage problem and not a
> ranking problem — a better reranker would not help much." A cross-encoder
> reranker was then the single largest improvement measured: `mrr` +0.261,
> `recall@k` +0.136.
>
> The reasoning conflated two different things. MRR only describes where the
> *first* correct article lands, and that was already reasonable. It says nothing
> about whether the *second* one reaches the top six — which is exactly what
> reranking fixed. Ranking quality was being judged on a statistic that could not
> see the failure.

Query decomposition was subsequently measured on all 50 questions and changed the
result on none of them; it is disabled. What remains untried against this gap is
per-article diversity in the ranking, and retrieval depth beyond the 48-candidate
pool the reranker currently sees.

---

## Reference bands

Working heuristics in common use — soft guidance, not authoritative benchmarks.
Published RAGAS figures vary widely by corpus and question style.

| metric | weak | acceptable | strong | this run |
|---|---|---|---|---|
| `faithfulness` | < 0.70 | 0.80 – 0.90 | > 0.90 | **0.906** |
| `context_precision` | < 0.60 | 0.70 – 0.85 | > 0.85 | **0.945** |
| `context_recall` | < 0.60 | 0.70 – 0.85 | > 0.85 | 0.802 |
| `answer_relevancy` | < 0.60 | 0.65 – 0.80 | > 0.80 | 0.658 |
| `answer_correctness` | < 0.50 | 0.60 – 0.75 | > 0.75 | 0.622 |
| `hit_rate@k` | < 0.70 | 0.75 – 0.90 | > 0.90 | 0.880 |
| `mrr` | < 0.50 | 0.60 – 0.80 | > 0.80 | **0.852** |
| `recall@k` | < 0.60 | 0.70 – 0.85 | > 0.85 | 0.773 |
| `precision@k` | *no useful band — capped by k, see Finding 1* | | | 0.238 |

Three metrics now sit in the "strong" band that previously did not. The two that
still don't — `answer_relevancy` and `answer_correctness` — are both answering
metrics, and neither moved when retrieval improved sharply.

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
