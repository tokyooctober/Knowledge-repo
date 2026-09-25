"""User-facing query interface — CLI and Streamlit.

    streamlit run app.py                    # web UI
    python app.py "a question"              # one-shot CLI
    python app.py --sync-corpus [--dry-run] # Phase 1 ingestion
    python app.py --stats

Phase 2 (`--check-email`) is not built until Milestone 3.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Callable
from datetime import date, datetime

from config import AUTHOR_NAME, DEFAULT_TOP_K
from logger import configure_logging, get_logger
from models import Answer

log = get_logger(__name__)


# ── query path ─────────────────────────────────────────────────────────────


def run_query(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> Answer:
    """`progress` reports the current phase, for UIs that show a wait indicator. Retrieval is
    no longer instant once reranking is on, so the phases are worth surfacing."""
    from config import ENABLE_RERANK, RERANK_POOL
    from query.answerer import answer
    from query.retriever import retrieve

    if progress:
        progress(
            f"Searching the corpus and reranking up to {RERANK_POOL} candidates…"
            if ENABLE_RERANK
            else "Searching the corpus…"
        )
    results = retrieve(query, top_k=top_k, filters=filters)
    log.info(
        "Retrieval complete",
        extra={
            "query": query,
            "result_count": len(results),
            "top_score": results[0].score if results else None,
        },
    )
    if progress:
        progress(f"Generating an answer from {len(results)} sources…")
    return answer(query, results)


def warm_models() -> dict[str, float]:
    """Load the embedder, vector store, reranker and (if hybrid) BM25 encoder before the first
    question arrives.

    Measured cold: embedder 11.6 s, reranker 6.1 s, first Qdrant search 1.2 s. All three are
    objects that live for the life of the process, so without this the first user question
    pays ~19 s of loading and every later one pays none — which reads as the app being
    mysteriously slow rather than as a one-time cost.

    Ollama's answerer is deliberately *not* warmed here. It unloads on its own keep-alive
    (5 min by default), so warming it would buy exactly one question while holding ~9 GB of
    VRAM. Warming is an optimisation, so a failure is logged and swallowed — the lazy path
    still works.
    """
    from config import ENABLE_HYBRID_SEARCH, ENABLE_RERANK
    from ingestion.embedder import embed_query
    from ingestion.sparse_encoder import encode_query
    from query.retriever import _get_reranker, _get_store

    def _warm_store():
        _get_store().search(query_vector=embed_query("warm up"), top_k=1, filters=None)

    timings: dict[str, float] = {}
    for name, fn in (
        ("embedder", lambda: embed_query("warm up")),
        ("vector_store", _warm_store),
        ("reranker", _get_reranker if ENABLE_RERANK else None),
        ("bm25_encoder", (lambda: encode_query("warm up")) if ENABLE_HYBRID_SEARCH else None),
    ):
        if fn is None:
            continue
        start = time.perf_counter()
        try:
            fn()
        except Exception:  # noqa: BLE001 - warming is best-effort; the lazy path still runs
            log.warning("Warm-up step failed", extra={"step": name}, exc_info=True)
            continue
        timings[name] = round(time.perf_counter() - start, 2)
    log.info("Model warm-up complete", extra=timings)
    return timings


def _parse_filters(args: argparse.Namespace) -> dict | None:
    filters: dict = {}
    if args.tags:
        filters["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.date_after:
        filters["date_after"] = datetime.fromisoformat(args.date_after)
    if args.date_before:
        filters["date_before"] = datetime.fromisoformat(args.date_before)
    return filters or None


def is_web_url(url: str) -> bool:
    """A `local:` id has no page behind it — render it as plain text, never a link."""
    return url.startswith(("http://", "https://"))


# ── CLI rendering ──────────────────────────────────────────────────────────


def _json_default(obj):
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    raise TypeError(f"not serialisable: {type(obj)}")


def _render_cli(ans: Answer, *, show_citations: bool, as_json: bool) -> str:
    if as_json:
        return json.dumps(dataclasses.asdict(ans), default=_json_default, indent=2)

    lines = [ans.response, ""]
    if show_citations and ans.sources:
        lines.append("Sources")
        lines.append("-------")
        for s in ans.sources:
            when = s.published_at.date().isoformat() if s.published_at else "n.d."
            link = s.url if is_web_url(s.url) else f"{s.url} (no web page — fix frontmatter)"
            lines.append(f"[{s.index}] {s.title} — {when} — score {s.score:.2f}\n    {link}")
    return "\n".join(lines)


# ── stats / ingestion ──────────────────────────────────────────────────────


def _print_stats() -> None:
    from storage.metadata_db import MetadataDB

    db = MetadataDB()
    stats = db.get_stats()
    db.close()
    arts = stats["articles"]
    by_src = stats["by_source"]
    print(f"Articles: {arts['total']} total ({arts['active']} active, {arts['archived']} archived)")
    print(f"  by source: corpus {by_src['corpus']}, web {by_src['web']}")
    versions = stats["pipeline_versions"]
    if len(versions) > 1:
        print(f"  WARNING: {len(versions)} pipeline versions in the index — re-ingest unfinished")
    corpus_run = stats["last_run"].get("corpus")
    if corpus_run:
        print(
            f"  last corpus sync: {corpus_run['started_at']} ({corpus_run['error_code'] or 'ok'})"
        )


def _sync_corpus(dry_run: bool) -> None:
    import asyncio

    from scheduler.monthly_job import run_corpus_sync

    stats = asyncio.run(run_corpus_sync(dry_run=dry_run))
    print(f"corpus sync: {stats}")


def _check_email(dry_run: bool) -> None:
    import asyncio

    from config import phase2_configured
    from scheduler.monthly_job import run_email_triggered

    if not phase2_configured():
        print(
            "Phase 2 is not configured — set LOGIN_URL / TRUSTED_SENDER / SITE_DOMAIN / "
            "HEALTH_CHECK_URL in .env.",
            file=sys.stderr,
        )
        return
    stats = asyncio.run(run_email_triggered(dry_run=dry_run))
    print(f"email check: {stats}")


# ── argparse ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app", description=f"Query {AUTHOR_NAME}'s articles")
    p.add_argument("query", nargs="?", help="natural-language question")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--tags", help="comma-separated tag filter")
    p.add_argument("--date-after", metavar="YYYY-MM-DD")
    p.add_argument("--date-before", metavar="YYYY-MM-DD")
    p.add_argument("--no-citations", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--sync-corpus", action="store_true")
    p.add_argument("--check-email", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats", action="store_true")
    return p


_ERRORS = {
    "empty_query": "Please enter a question.",
    "no_results": "I couldn't find relevant content for this question. Try rephrasing or "
    "broadening your search.",
    "vector_store": "The knowledge base is unavailable. Check that Qdrant is running.",
    "llm": "Could not generate an answer right now. Please try again in a moment.",
}


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)

    if args.stats:
        _print_stats()
        return 0
    if args.check_email:
        _check_email(args.dry_run)
        return 0
    if args.sync_corpus:
        _sync_corpus(args.dry_run)
        return 0

    if not args.query or not args.query.strip():
        print(_ERRORS["empty_query"], file=sys.stderr)
        return 2

    from models import ModelMismatchError
    from storage.vector_store import VectorStoreConnectionError

    try:
        ans = run_query(args.query, args.top_k, _parse_filters(args))
    except VectorStoreConnectionError:
        print(_ERRORS["vector_store"], file=sys.stderr)
        return 1
    except ModelMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(_render_cli(ans, show_citations=not args.no_citations, as_json=args.json))
    return 0


# ── Streamlit ──────────────────────────────────────────────────────────────


def _streamlit_app() -> None:  # pragma: no cover - exercised via `streamlit run`
    import streamlit as st

    from config import phase2_configured

    configure_logging()
    st.title(f"🔍 Knowledge Repository — {AUTHOR_NAME}")

    @st.cache_resource(show_spinner=False)
    def _warm_once():
        """Once per process, not per session — these models are shared by every viewer."""
        return warm_models()

    with st.spinner("Loading models (first run only)…"):
        _warm_once()

    query = st.text_input("Ask a question:")
    col1, col2 = st.columns(2)
    tags = col1.text_input("Tags (comma-separated)")
    date_after = col2.text_input("Date after (YYYY-MM-DD)")

    if st.button("Ask") and query.strip():
        filters: dict = {}
        if tags.strip():
            filters["tags"] = [t.strip() for t in tags.split(",")]
        if date_after.strip():
            filters["date_after"] = datetime.fromisoformat(date_after.strip())
        with st.status("Working…", expanded=True) as status:
            ans = run_query(
                query, DEFAULT_TOP_K, filters or None, progress=lambda msg: status.update(label=msg)
            )
            status.update(label="Done", state="complete", expanded=False)
        st.markdown(ans.response)
        st.subheader("Sources")
        for s in ans.sources:
            when = s.published_at.date().isoformat() if s.published_at else "n.d."
            if is_web_url(s.url):
                st.markdown(f"[{s.index}] [{s.title}]({s.url}) — {when} — score {s.score:.2f}")
            else:
                st.markdown(f"[{s.index}] {s.title} — {when} — score {s.score:.2f} _(no link)_")

    with st.sidebar:
        st.header("Stats")
        from storage.metadata_db import MetadataDB

        db = MetadataDB()
        stats = db.get_stats()
        db.close()
        st.write(stats["articles"])
        st.write("by source:", stats["by_source"])
        st.button("Sync corpus", on_click=lambda: _sync_corpus(False))
        st.button(
            "Check email",
            disabled=not phase2_configured(),
            help="Set LOGIN_URL / TRUSTED_SENDER in .env" if not phase2_configured() else None,
        )


_UNDER_STREAMLIT = False
try:  # pragma: no cover
    import streamlit.runtime as _st_runtime

    # True only inside a real `streamlit run` session — importing the module is not enough,
    # so `python app.py …` still reaches main() with streamlit installed.
    _UNDER_STREAMLIT = _st_runtime.exists()
except ImportError:  # pragma: no cover
    pass


if _UNDER_STREAMLIT:  # pragma: no cover
    _streamlit_app()
elif __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
