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
from datetime import date, datetime

from config import AUTHOR_NAME, DEFAULT_TOP_K
from logger import configure_logging, get_logger
from models import Answer

log = get_logger(__name__)


# ── query path ─────────────────────────────────────────────────────────────


def run_query(query: str, top_k: int = DEFAULT_TOP_K, filters: dict | None = None) -> Answer:
    from query.answerer import answer
    from query.retriever import retrieve

    results = retrieve(query, top_k=top_k, filters=filters)
    log.info(
        "Retrieval complete",
        extra={
            "query": query,
            "result_count": len(results),
            "top_score": results[0].score if results else None,
        },
    )
    return answer(query, results)


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
        print("Email-triggered ingestion is not built yet (Milestone 3).", file=sys.stderr)
        return 2
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
        with st.spinner("Retrieving and generating…"):
            ans = run_query(query, DEFAULT_TOP_K, filters or None)
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


try:  # pragma: no cover
    import streamlit.runtime as _st_runtime

    if _st_runtime.exists():
        _streamlit_app()
except ImportError:  # pragma: no cover
    pass


if __name__ == "__main__" and "streamlit" not in sys.modules:  # pragma: no cover
    raise SystemExit(main())
