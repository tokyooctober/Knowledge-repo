"""Phase 0 — generate a synthetic evaluation test set with RAGAS, fully local.

Runs in `.venv-eval`. Reads the markdown corpus via the app's own loader
(`ingestion.md_loader.load_corpus`), samples a spread of articles across the
publication timeline, and asks RAGAS `TestsetGenerator` (Ollama judge +
local BGE embeddings) to write questions with reference answers and reference
contexts.

Output: eval/dataset/candidates.jsonl — one row per generated sample:
    {id, user_input, reference, reference_contexts, source_urls, synthesizer}

Nothing here is trusted yet. Run eval/review_testset.py next to hand-accept rows
into the frozen eval/dataset/testset.jsonl.

    .venv-eval/bin/python eval/build_testset.py --size 60 --articles 40 --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._common import ts, write_jsonl
from eval.eval_config import CANDIDATES_PATH, EVAL_GEN_MODEL, TESTSET_SIZE


def _sample_articles(n: int, seed: int):
    from ingestion.md_loader import load_corpus

    corpus = [a for a in load_corpus() if not a.is_stub and a.body_text.strip()]
    if not corpus:
        sys.exit("Corpus is empty or all stubs — check MD_CORPUS_DIR in .env")

    corpus.sort(key=lambda a: (a.published_at is None, a.published_at))
    if n >= len(corpus):
        picked = corpus
    else:
        # even stride across the timeline, then jitter within each bucket
        rng = random.Random(seed)
        step = len(corpus) / n
        picked = [
            corpus[min(len(corpus) - 1, int(i * step) + rng.randrange(max(1, int(step))))]
            for i in range(n)
        ]
        picked = list({a.url: a for a in picked}.values())  # dedupe
    print(f"Sampled {len(picked)} / {len(corpus)} articles for generation")
    return picked


def _to_documents(articles):
    from langchain_core.documents import Document

    return [
        Document(
            page_content=a.body_text,
            metadata={
                "url": a.url,
                "title": a.title,
                "published_at": a.published_at.isoformat() if a.published_at else None,
            },
        )
        for a in articles
    ]


def _tokens(text: str) -> set[str]:
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {w for w in words if len(w) > 3}


def _attribute(contexts: list[str], articles) -> list[str]:
    """Best-effort: map each reference context back to a source article by token overlap.

    RAGAS 0.2 does not expose per-context source metadata on the sample, so we recover it
    here. Used only by the judge-independent retrieval metrics (metrics_simple)."""
    corpus_tokens = [(a.url, _tokens(a.body_text)) for a in articles]
    hits: set[str] = set()
    for ctx in contexts:
        ct = _tokens(ctx)
        if len(ct) < 5:
            continue
        best_url, best_frac = None, 0.0
        for url, at in corpus_tokens:
            frac = len(ct & at) / len(ct)
            if frac > best_frac:
                best_url, best_frac = url, frac
        if best_url and best_frac >= 0.6:
            hits.add(best_url)
    return sorted(hits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_testset")
    ap.add_argument("--size", type=int, default=TESTSET_SIZE, help="number of samples to generate")
    ap.add_argument("--articles", type=int, default=40, help="articles to sample as source")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(CANDIDATES_PATH))
    args = ap.parse_args(argv)

    from ragas.testset import TestsetGenerator

    from eval._common import build_ragas_embeddings, build_ragas_llm

    articles = _sample_articles(args.articles, args.seed)
    docs = _to_documents(articles)

    print(f"Generating {args.size} samples with judge={EVAL_GEN_MODEL} (local) …")
    generator = TestsetGenerator(
        llm=build_ragas_llm(EVAL_GEN_MODEL),
        embedding_model=build_ragas_embeddings(),
    )
    dataset = generator.generate_with_langchain_docs(docs, testset_size=args.size)

    rows = []
    for i, sample in enumerate(dataset.to_list()):
        contexts = sample.get("reference_contexts", [])
        rows.append(
            {
                "id": f"q{i:03d}",
                "user_input": sample.get("user_input", ""),
                "reference": sample.get("reference", ""),
                "reference_contexts": contexts,
                "source_urls": _attribute(contexts, articles),
                "synthesizer": sample.get("synthesizer_name", ""),
                "review": "pending",  # pending | accepted | rejected
                "generated_at": ts(),
            }
        )

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} candidates -> {args.out}")
    print("Next: .venv-eval/bin/python eval/review_testset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
