# Retrieval investigation timeline — 2026-09-04 → 2026-09-18

What was hypothesised, what was measured, what got thrown away, and the one change that
survived every guard — between the `20260907T030102Z` baseline and the `20260918` re-baseline.

A visual version of this document is published as an artifact:
<https://claude.ai/artifact/RX4wBShbmT1mcGaF5YP3yH> (private — access is per-person).
`TIMELINE.html` in this directory is a byte-for-byte copy of that page.

Reconstructed from the session transcripts, git history and the eval logs. Retrieval metrics
(`hit_rate@k`, `mrr`, `recall@k`, `precision@k`) are exact URL-overlap measures with no judge
involved and reproduce to four decimals across runs; the five RAGAS metrics were scored by
`qwen2.5:14b-instruct` over two passes.

**Outcome:** recall@k +21%, mrr +44%, all three answering metrics flat.
The bottleneck moved from retrieval to generation.

---

## Phase 1 — Getting a test set worth grading against (09-04 → 09-07)

Two questions had to be settled empirically rather than by reputation: which model writes the
questions, and which model grades the answers.

| when | what happened |
|---|---|
| 09-04 22:02 | `build_testset.py --size 10` with `qwen2.5:14b-instruct`. Three import failures in a row — `eval`, `langchain_community…vertexai`, `frontmatter` — then a RAGAS parse failure and a timeout at `--size 2`. Nothing generated. A separate sampling bug surfaced: the script could reference articles that were never ingested → `--ingested-only` added. |
| 09-05 20:14 | Intended phi4 run. The `.env` edit had only ever been pasted into chat, never written to disk, so the run silently used the default model. Not a phi4 run at all. Verified against `ollama list` before retrying. |
| 09-05 21:46 | The real phi4 run, 60 samples over 40 articles. ~45 min, all 60 rows written — but **10 of 40 articles (25%) failed RAGAS's structured-output extraction**. Verdict at the time: *phi4 isn't reliably producing the strict JSON RAGAS's knowledge-graph extractors require.* Kept as `phi4-candidates.jsonl`, never used. |
| 09-05 22:36–22:41 | Switched to `claude-opus-5`. Three fast failures: a routing-label bug, then two API key fragments pasted together in `.env` (confirmed by a 401). |
| 09-05 22:49–23:05 | Fourth attempt succeeded: **~16 min, 0% extraction failures**. Only 2m12s of that was sample generation; the rest was building the knowledge graph. → `candidates.jsonl` |
| 09-06 | Human review: **50 accepted / 10 rejected**. `testset.jsonl` frozen. |

### Judge calibration (09-07)

The first attempt died on `TimeoutError` almost everywhere — RAGAS defaults to ~16 concurrent
calls against an Ollama server that serves one generation at a time. Fixed with `--max-workers`
and an explicit timeout. Each judge then scored 32 jobs against 16 hand-labelled rows.

| judge | faithfulness | correctness | combined | wall time |
|---|---|---|---|---|
| **qwen2.5:14b-instruct** | 1.000 | 0.750 | **0.875** | 16.5 min |
| openai/gpt-4o | 1.000 | 0.625 | 0.812 | 3 min |
| phi4 | 1.000 | 0.625 | 0.812 | 19.5 min |
| mistral-small | 1.000 | 0.562 | 0.781 | 84.5 min |

The winner is free, local, and already the app's production answerer. Note the frontier model did
*worse* on this material. mistral-small's 84 minutes was the first sighting of VRAM spillover on
the 16 GB card — a constraint that returns in Phase 5.

**09-07 baseline:** Phase 1 (6.5 min) + Phase 2 (3 min) over all 50 frozen questions. Every number
below is measured against this run.

---

## Phase 2 — Re-testing a decision already made (09-14)

Re-ran generation on `qwen2.5:14b-instruct` at identical parameters (`--size 60 --articles 40
--seed 7`), written to its own file so the frozen set could not be clobbered, and logged this time
— the lesson from phi4, whose run left no log at all.

| generator | duration | extraction failures | outcome |
|---|---|---|---|
| qwen2.5:14b-instruct | 50m 55s | 10/40 (25%) | ran, flawed |
| phi4 | ~45 min | 10/40 (25%) | ran, flawed |
| **claude-opus-5** | **~16 min** | **0/40 (0%)** | clean · frozen |

Two 14B-class models, one identical failure mode. Original decision upheld; the new candidates
were not adopted.

---

## Phase 3 — Query decomposition, falsified three times (09-14 → 09-17)

**Hypothesis:** splitting a compound question into sub-questions raises recall on questions that
need two or three source articles.

### Test 1 — full n=50, decomposition ON

Every retrieval metric fell.

| metric | OFF (baseline) | ON | change |
|---|---|---|---|
| context_precision | 0.864 | 0.837 | −0.027 |
| context_recall | 0.704 | 0.533 | −0.171 |
| hit_rate@k | 0.760 | 0.620 | −0.140 |
| mrr | 0.591 | 0.511 | −0.080 |
| recall@k | 0.637 | 0.543 | −0.094 |
| precision@k | 0.181 | 0.164 | −0.017 |
| answer_relevancy | 0.648 | 0.546 | −0.102 |

Even `precision@k` — the metric decomposition was built to raise — went down. Only faithfulness
ticked up (+0.021).

### Root cause — revised three times

**Read 1.** Fragments *replace* the holistic query's search instead of adding to it, so chunks the
full question found well are dropped by the score threshold.

**Read 2.** Score data killed that. `MIN_SCORE_THRESHOLD = 0.35` **never fires in production** —
all 120 observed scores sit between 0.6398 and 0.8304, and every question fills all six slots.
Nothing is ever filtered; the contest is purely ordinal, and the margins are hairline:

| question | spread across the 6 kept results |
|---|---|
| q004 | 0.0207 |
| q007 | 0.0243 |
| q012 | 0.0288 |
| q009 | 0.0335 |
| q008 | 0.0363 |

**Read 3.** Comparing the ON and OFF score sets for q004 overturned read 2 as well:

| arm | score range of the 6 kept chunks | gold article |
|---|---|---|
| decomp OFF | 0.6698 – 0.6905 | present · 0.6714, rank 5 |
| decomp ON | 0.6398 – 0.6583 | **absent entirely** |

The sets do not overlap. `_fuse()` kept the max score and sorted descending, so had the gold chunk
surfaced in either fragment's results at anything ≥ 0.6583 it would have ranked *first*. It was
never out-ranked — **the fragment searches never retrieved it at all.** The holistic query is never
searched, so anything only the whole question would have found is unreachable, and with all six
slots always full that loss is total and silent.

**The proof that the scale was meaningless.** On q007 and q012 the fragment scores came back
*higher* than the holistic ones — q007 ON at 0.7574–0.7867 against OFF at 0.7500–0.7743 — while
retrieving **worse** context. Higher number, worse answer. These are cosine scores against
different query vectors; the scale never carried across them. That is exactly the assumption
`_fuse()` was built on.

**Consequences for the fix.** Saturation is the design, not the defect, so no fix may try to
relieve it:

- widening `top_k` when decomposition fires changes the answerer's contract and games the @k metric;
- budgeting slots per subquery makes q004 worse by construction, since false splits dominate;
- reserving the holistic top-*m* fails because q004's gold ranked 5th of 6;
- plain RRF fails too — gold at holistic rank 5 with no fragment support scores 1/(60+5) = 0.0154
  and loses to anything ranking 3rd in two fragment lists at 2/(60+3) = 0.0317.

**The rule that came out of it:** *fragments decide what gets considered; the original question
decides what wins.* Every contender measured against one vector — the user's actual question.

### Why the regression tests didn't catch it

The three xfail tests failed, so they detected *something* — but they encoded the wrong mechanism,
and shared three structural flaws:

1. They made the threshold the decisive lever — a lever production never pulls. Fixtures used
   scores of 0.2–0.3 against a 0.35 threshold; real scores are 0.64–0.83.
2. They never saturated `top_k` (1–2 results per query against `top_k=6`), so the crowding-out
   code path is literally unreachable. `MAX_CHUNKS_PER_ARTICLE` never fires either.
3. They asserted **presence**, not **survival**. Production asks whether a chunk beats 23
   competitors for one of 6 slots.

A naive "append the holistic results" fix would have turned all three green while still regressing
in production. **They would have green-lit a fix that doesn't work** — worse than having no tests.

### Test 2 — the additive fix (n=10)

Flagged before starting: all ten smoke questions are single-source, and decomp=OFF already scores
1.000 on `hit_rate` and `recall` there — *a ceiling that can be tied but not beaten.* Run went
ahead with success redefined as "same or better on every metric".

**Change:** always search the original query, widened to `top_k × HOLISTIC_OVERFETCH` = 48; rank by
score against the real question via `_merge_by_holistic_rank()`; delete `_fuse()`, now dead, which
encoded the cross-query comparability assumption. Goal met on iteration 1 of 3.

Reported honestly: retrieval came back **byte-identical to OFF on 10/10**, so the faithfulness and
correctness gains could only be answerer nondeterminism, not the fix.

### Test 3 — the multi-source slice (n=22)

A bigger prefix was rejected: the test set is **sorted by gold count**, so multi-source questions
all sit in rows 31–50 and `--limit 30` would mostly re-confirm known ties. A targeted 22-row slice
of every 2+-gold question replaced it.

| arm | hit_rate@k | mrr | recall@k | precision@k |
|---|---|---|---|---|
| decomp OFF | 0.7273 | 0.5720 | 0.4470 | 0.1833 |
| decomp ON — original | 0.5909 | 0.5242 | 0.4167 | 0.1742 |
| decomp ON — fixed | 0.7273 | 0.5720 | 0.4470 | 0.1833 |

On the exact questions it was built for, the original implementation made things *worse*. The fixed
version is byte-identical to OFF on 22/22 — despite firing on all 22 with 2–4 subqueries each.
One LLM call plus several vector searches per query, for zero change in output.

The OFF numbers reproduce exactly from the independent 2026-09-07 run — a clean cross-check that
the OFF path is deterministic and untouched.

### Two more levers, both falsified

**Slot competition?** No. 12 of 22 questions already return 6 distinct articles, 6 more return 5,
and the per-article cap binds on only 3 of 22 — yet just 20 of 45 gold articles are found at all.

**Per-article chunk cap?** Asked in session: *could we use the metadata as a filter into the
top_k?* Two corrections came back first — the dedup key must be `article_url`, not `article_title`
(URLs are the stable identity and what recall is scored by), and it isn't new machinery, since
`MAX_CHUNKS_PER_ARTICLE = 3` already does it. The assumption holds but is small: 10 of 22 questions
repeat an article, consuming 15 of 132 slots (11%).

| `MAX_CHUNKS_PER_ARTICLE` | hit_rate@k | mrr | recall@k | precision@k | avg distinct articles |
|---|---|---|---|---|---|
| 1 | 0.7273 | 0.5720 | 0.4697 | 0.1591 | 6.00 |
| 2 | 0.7273 | 0.5720 | 0.4470 | 0.1773 | 5.41 |
| 3 (current) | 0.7273 | 0.5720 | 0.4470 | 0.1833 | 5.32 |

Fifteen slots freed, **one** extra gold article surfaced, precision@k down 13% relative.

**The failure was more useful than a success would have been.** Fourteen of the fifteen freed slots
filled with non-gold articles — so the missed gold isn't sitting just below the cutoff waiting for
room. Slot-allocation policy only reshuffles who gets in from a candidate pool that mostly doesn't
contain the answer. Which raises the question that actually discriminates: if they're not just
below the line, *where are they?*

### The reframe — where the missing gold actually ranks

Each possible outcome was bound to a fix **before** the run, so the result would dictate the answer
rather than be read to fit a preferred one:

| if gold sits at… | then the fix is |
|---|---|
| ranks 7–48 | a reranker over the pool we already fetch |
| ranks 49–200 | a much wider pool *and* a reranker |
| beyond 200 | embedding or chunking — no retrieval policy helps |

Deep top-200 search, 45 gold articles pooled across the 22 multi-source questions:

| rank band | count | share | reachable by |
|---|---|---|---|
| 1–6 · already retrieved | 20 | 44.4% | — |
| 7–12 | 3 | 6.7% | a slightly wider window |
| **13–48** | **11** | **24.4%** | **reranking the pool we already fetch** |
| 49–200 | 8 | 17.8% | wider pool + reranker |
| beyond 200 | 3 | 6.7% | nothing retrieval-side |

**14 of the 25 misses were already inside the 48-candidate pool** and being thrown away by the
score-ordered trim to six. That makes this a ranking-quality problem — not an embedding problem,
not a chunking problem, and not one that query splitting or slot reallocation can touch.

*Read it as a hypothesis-generator, not a verdict.* It samples the multi-source slice only;
single-source questions, where baseline recall is already 1.000, are excluded, so this is the
distribution for hard questions rather than for all queries. At 45 observations, one article moving
band shifts a share by ~2 points. What carries the conclusion is that its prediction then held
out-of-sample — confirmed at n=32 *with* single-source included, and again at n=50.

### Why a cross-encoder, specifically

A cross-encoder scores the (query, chunk) pair **jointly**, in one forward pass, instead of
comparing two embeddings computed independently of one another. That independence is the same
defect diagnosed upstream — it is why fragment scores of 0.7867 beat holistic scores of 0.7743
while retrieving worse context, and why a compound question's second topic gets buried. If
independently-computed similarity is the weakness, joint scoring is the direct answer.

Practically: `sentence-transformers` was already a dependency and ships cross-encoders — no new
stack, no re-ingest — and the 48-candidate pool was already being fetched. The first experiment
cost nothing but a model download and a read-only simulation.

---

## Phase 4 — Cross-encoder reranking, and one false positive on the way (09-17)

### The retraction

MiniLM-L6 over the 48-pool lifted recall 0.447 → 0.530 on the multi-source slice, every metric up —
reported as *"the first intervention that actually works."*

Retracted one turn later. That was measured on the multi-source slice **alone** — the exact
methodological error criticised one turn earlier. Adding four single-source questions turned the win
into a wash: **every reranker configuration degraded single-source recall**, where the baseline is a
perfect 1.000.

### The harness, not the method

Chunks are 512 tokens by design (`CHUNK_SIZE=512`), and all three candidate rerankers cap at 512
**architecturally** (`max_position_embeddings`), not as a harness setting:

| model | pairs exceeding 512 tokens |
|---|---|
| MiniLM-L6 | 17/48 (35%) |
| bge-reranker-base | 33/48 (69%) |
| bge-reranker-large | 33/48 (69%) |

Median pair length for the bge models was 550 tokens. The regression was truncation. Re-run on
`bge-reranker-v2-m3` (8192-token context, zero truncation): single-source fully preserved at 1.000,
multi-source recall 0.417 → 0.583, ALL recall 0.650 → 0.750.

### The ablation (n=32 — 22 multi-source + 10 single-source, 47 gold articles)

**RRF (reciprocal rank fusion)** merges ranked lists using only positions: `score = 1 / (k + rank)`,
summed across lists. It is free — pure arithmetic — and structurally immune to the cross-query
score-scale bug, because no scores are involved. Small `k` (10) emphasises top ranks; `k = 60`
flattens, especially with lists only 48 long.

The design insight: RRF and the cross-encoder operate at different stages and different costs, so
RRF's best job isn't to *compete* with the reranker but to improve **which candidates reach it**, at
zero cost, while the reranker's budget stays pinned at 48 pairs.

| arm | what it does | isolates |
|---|---|---|
| **A1** baseline cosine @48 | vector search → post-filter → 6 | control |
| **A2** cross-encoder @48 | same 48 → cross-encoder rescores all → post-filter | reranking alone |
| **A3** RRF(cosine, CE) | rank the same 48 twice, fuse the two rankings | does blending beat a full reorder? |
| **A4** RRF(holistic, subqueries), no CE | fuse holistic + subquery lists by rank | does decomposition help when fused correctly? |
| **A5** RRF(holistic, subqueries) → CE @48 | RRF picks *which* 48 reach the reranker, then CE ranks them | the full pipeline |
| oracle @48 / @200 | hypothetical perfect reranker | ceilings |

| arm | ALL recall | ALL hit | MULTI recall | SINGLE recall |
|---|---|---|---|---|
| A1 baseline cosine @48 | 0.620 | 0.812 | 0.447 | 1.000 |
| **A2 cross-encoder @48** | **0.740** | **0.906** | **0.621** | **1.000** |
| A3 RRF(cosine, CE) k=10 | 0.708 | 0.875 | 0.576 | 1.000 |
| A3 RRF(cosine, CE) k=60 | 0.677 | 0.844 | 0.576 | 0.900 |
| A4 RRF(holistic, subq) unweighted | 0.620 | 0.781 | 0.492 | 0.900 |
| A4 RRF(holistic, subq) holistic×2 | 0.604 | 0.781 | 0.470 | 0.900 |
| A5 RRF → CE @48 | 0.755 | 0.906 | 0.644 | 1.000 |
| oracle @48 | 0.833 | 0.906 | 0.758 | 1.000 |
| oracle @200 | 0.953 | 1.000 | 0.932 | 1.000 |

**What held up.** The cross-encoder replicates at 3× the sample, single-source untouched at 1.000.
Baseline MULTI of 0.447 matches the independent n=22 measurement exactly.

**A3 falsified.** RRF blending was predicted to protect single-source. It doesn't need protecting —
pure cross-encoding already holds 1.000 — and blending actively dilutes correct promotions. k=60
even breaks a single-source question.

**A4 falsified.** This is the fair test decomposition never got: fused by rank, so the score-scale
bug cannot apply, purely as a recall expander. Result identical to baseline, with single-source
regressing. Weighting the holistic list made it worse. **Third independent null.**

**A5 not banked.** +0.015 ALL recall over A2 is roughly half a gold article across 32 questions,
bought with an LLM call plus up to four extra vector searches per query. Flagged as
indistinguishable from A2 rather than declared the winner — a call vindicated at n=50.

**Controls that made the ablation trustworthy:** single-source recall reported separately on every
arm; the identical production post-filter on all arms; subqueries generated once and cached, so the
LLM's nondeterminism could not become a hidden variable between A4 and A5.

### Depth and cost

| candidates reranked | latency | ALL recall | SINGLE recall |
|---|---|---|---|
| none | ~0 ms | 0.650 | 1.000 |
| 6 | ~230 ms | 0.650 | 1.000 |
| 12 | 473 ms | 0.550 | 0.750 |
| 24 | 968 ms | 0.700 | 1.000 |
| **48** | **1844 ms** | **0.750** | **1.000** |
| 200 | 7398 ms | 0.700 | 1.000 |

Returns are near-linear at ~38 ms per pair, so halving latency halves the benefit. Twelve
candidates scores *below doing nothing* — too shallow a pool means the reranker demotes correct
chunks without better alternatives to promote. (n=10 here, so the single-source row is 4 questions;
treat that specific anomaly as suggestive, not established. The 12-vs-48 direction is not in doubt.)
Going deeper hurts too: more distractors to misrank. **48 or nothing.**

**Where 48 came from — honestly:** arbitrary. It was `top_k × 8` from the decomposition fix
("start at 8"), and every later experiment inherited it because it was already being fetched. The
rank-band diagnostic justified it *afterwards* (48 captures 76% of gold, 12 captures 51%), and
latency caps the usable range anyway (96 ≈ 3.7 s, 128 ≈ 4.9 s, on top of a 7.7 s pipeline).
Best tested value, not a proven optimum — 32/64/96 were never probed.

**Decision:** cross-encoder alone at depth 48 — no RRF, no decomposition.

---

## Phase 5 — Making it deployable on one 16 GB card (09-17 → 09-18)

**Part A shipped.** 401 tests green, and the shipped `retrieve()` reproduced the A2 prototype
numbers to four decimals. It also surfaced two real CUDA OOM warnings — a 365 MB allocation failing
with 171 MB free on a 16 GB card — and showed that of 4.20 s of retrieval, **decomposition was
2.2 s, the larger half, buying nothing.**

**VRAM.** The default batch of 32 was the worst cell on both axes: with variable-length inputs every
pair pads to the longest chunk in the batch.

| batch size | peak VRAM | activations | latency/query |
|---|---|---|---|
| 32 (default) | 4,387 MiB | 909 MiB | 1,898 ms |
| 16 | 3,931 MiB | 454 MiB | 1,840 ms |
| **8** | **3,705 MiB** | **228 MiB** | **1,761 ms** |
| 4 | 3,591 MiB | 114 MiB | 1,700 ms |
| 2 | 3,534 MiB | 57 MiB | 1,716 ms |

`batch_size=8` saves 681 MiB and ~137 ms — better on both dimensions, no trade-off. And
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` targets the real OOM cause: fragmentation, not
exhaustion. An earlier warning about "~14.7 GB permanently co-resident" was retracted — nothing is
permanently resident; Ollama unloads after a five-minute keep-alive, and all three models coexist
only during an active query. → `78c8664`

**The pool-depth bug — found by reading code, because no measurement could find it.** Pool size was
decided by whether decomposition happened to fire: `top_k * (HOLISTIC_OVERFETCH if subqueries else
2)` — 48 candidates if the question split, 12 if it didn't. The reranker's needs never entered
into it.

No question in the test set is under six words — shortest 11, median 29, longest 84 — because RAGAS
writes verbose synthetic questions. Real users type *"why is gold up?"* (4 words,
`MIN_DECOMPOSITION_WORDS = 6`). **Every eval question takes the 48 path, so no measurement on this
test set can detect the problem.** The trap: disabling a feature that measurably does nothing would
have silently dropped every query to 12 candidates — worse than no reranking — and it would have
looked like the reranker's fault. → `c39d3a2`

**Part B — decomposition's last test (n=50, 5 arms).**

| arm | ALL recall | ALL hit | MULTI recall | SINGLE recall |
|---|---|---|---|---|
| A2 rerank only, no decomposition | 0.7733 | 0.8800 | 0.6212 | 0.8929 |
| Shipped `_merge_by_holistic_rank` | 0.7733 | 0.8800 | 0.6212 | 0.8929 |
| A5 RRF(k=60) | 0.7833 | 0.8800 | 0.6439 | 0.8929 |
| Reserve 36 holistic + 12 subquery | 0.7833 | 0.8800 | 0.6439 | 0.8929 |
| oracle @48 | 0.8733 | 0.9200 | 0.7576 | 0.9643 |

Decomposition fired on 47 of 50 questions (avg 2.9 subqueries). The shipped merge differs from not
decomposing at all on **zero** of 50 questions. Repaired variants gain a net one question in fifty
— two better (q038, q057), one worse (q035) — a coin flip. RRF and simple slot reservation produce
identical metrics, so the fusion machinery and its `k` parameter buy nothing over holding 12 slots
open.

Technically this passed pre-registered criterion 2 — which was then stated plainly to have been
written without a minimum effect size, and this effect falls below what it implicitly assumed was
meaningful. **Disabled:** ~2.2 s on 94% of queries for a coin-flip-level gain. → `cf093cb`

**Cold start.** Measured rather than assumed:

| component | cold | warm | recurs? |
|---|---|---|---|
| BGE embedder first load | 11.56 s | 0.02 s | once per process |
| Reranker load | 6.13 s | — | once per process |
| Qdrant first search | 1.19 s | — | once per process |
| Ollama completion | 3.92 s | 0.26 s | every 5-min idle |

The embedder costs more than the reranker — the expectation had been the reverse. ~18.9 s is once
per process; only ~3.7 s recurs. Chose a synchronous in-process pre-warm behind a visible spinner:
an honest "Loading models…" beats a fast-looking app that then hangs for 20 s. Raising
`OLLAMA_KEEP_ALIVE` was rejected — ~9 GB held permanently to save 3.7 s re-creates exactly the
headroom pressure just resolved. First query **35.4 s → 15.4 s**. → `f742484`

---

## Phase 6 — The full 50-question re-baseline (09-18)

Phase 1 reproduced the Part B A2 arm to four decimals — confirmation that the harness and the
simulation agree, which is what made the three-hour scoring run worth starting. Phase 2: **2 h 48 m,
zero errors, zero timeouts.**

| metric | 2026-09-07 | 2026-09-18 | change |
|---|---|---|---|
| context_precision | 0.864 | 0.945 | +0.081 |
| context_recall | 0.704 | 0.802 | +0.098 |
| hit_rate@k | 0.760 | 0.880 | +0.120 |
| mrr | 0.591 | 0.852 | **+0.261 (+44%)** |
| recall@k | 0.637 | 0.773 | +0.136 |
| precision@k | 0.181 | 0.238 | +0.057 |
| faithfulness | 0.875 | 0.906 | +0.031 |
| answer_relevancy | 0.648 | 0.658 | +0.010 |
| answer_correctness | 0.625 | 0.622 | −0.003 |

All three answering metrics sit inside the ±0.05 threshold set in advance. The two scoring passes
were unusually stable — context_precision, context_recall and answer_relevancy identical to four
decimals, against ±0.107 of wobble at n=22 — so **the flatness is a finding, not noise swamping a
small gain.**

One number nobody predicted: `mrr` moved further than recall did. The reranker isn't only finding
more gold articles, it's putting them near the top.

> **The bottleneck has moved from retrieval to generation.** The right material is now reliably in
> front of the answerer; it just isn't producing better answers from it. Further retrieval work —
> the remaining 0.10 oracle gap, a recency prior, chunking — has clearly diminishing returns. The
> constraint is `qwen2.5:14b-instruct` as the answerer, or the answer prompt itself. Note also that
> `answer_correctness` is graded against RAGAS's synthetic reference answers, a harsh and somewhat
> artificial target.

→ `43981af` re-baselined `METRICS.md`.

---

## The methodological lesson

**Measuring only where an intervention should help is how you get a false positive.** It happened
twice, one turn apart — first with decomposition on the multi-source slice, then with the reranker
on the same slice, by the same author who had just criticised it.

Three devices eventually stopped it:

1. **A single-source guard** reported separately on every measurement. Aggregate-only reporting
   hides regressions on the majority population.
2. **An effect-size threshold written down before the run**, not after seeing the numbers.
3. **An oracle ceiling**, to say how much of the gap was ever reachable.

A fourth, from Phase 5: some defects are invisible to every measurement your test set can make.
The pool-depth bug had to be found by reading code.

## Commits

| sha | what |
|---|---|
| `68ff615` | snapshot before the reranking work |
| `d468ee8` | Part A — cross-encoder reranking wired into `retrieve()`, three-phase UI |
| `78c8664` | reranker batch size set explicitly; CUDA allocator flag documented |
| `c39d3a2` | stop query decomposition from shrinking the reranker's candidate pool |
| `cf093cb` | disable decomposition — it changed nothing on all 50 eval questions |
| `f742484` | load models at startup so the first question isn't 20 s slower than the rest |
| `43981af` | re-baseline `METRICS.md` on the post-reranking run |
