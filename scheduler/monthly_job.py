"""Ingestion orchestrator across two content sources.

Mode 1 — corpus sync (Phase 1): load markdown from `MD_CORPUS_DIR`, ingest new and changed
files, never remove anything (see `run_prune`). No browser, no mailbox, no credentials.

Mode 2 — email-triggered (Phase 2): `run_email_triggered` checks the mailbox for a new
report, authenticates against the site (a human signs in when the session expires),
scrapes the linked article, and ingests it. `start_scheduler` runs it on a cron + interval
schedule.

The shared sub-pipeline `ingest_article` is the only place vectors are written:
    stub guard -> is_changed -> transcribe -> chunk -> embed
                -> delete_by_url (if replacing) -> vector_store.upsert
                -> metadata_db.upsert_article
`delete_by_url` before `upsert` is what stops a re-run from accumulating duplicate vectors.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Allow `python scheduler/monthly_job.py` (not just `-m scheduler.monthly_job`).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MD_CORPUS_DIR
from ingestion import md_loader
from ingestion.chunker import chunk_article
from ingestion.embedder import embed_chunks
from ingestion.image_transcriber import count_uncached, transcribe_images
from llm_provider import get_embedding_provider
from logger import configure_logging, get_logger
from storage.metadata_db import MetadataDB
from storage.vector_store import VectorStore

log = get_logger(__name__)

_EMPTY_STATS = {"new": 0, "updated": 0, "skipped": 0, "failed": 0}


def _stats() -> dict:
    return dict(_EMPTY_STATS)


# ── shared sub-pipeline ─────────────────────────────────────────────────────


async def ingest_article(
    article,
    context,
    stats: dict,
    run_id: str,
    force: bool = False,
    *,
    db: MetadataDB,
    store: VectorStore,
) -> None:
    """change check -> transcribe -> chunk -> embed -> store. Mutates `stats` in place.
    `context` is None for corpus articles. Raises only on programmer error."""
    if article is None or article.is_stub:
        stats["skipped"] += 1
        log.warning("Article stub or load failed", extra={"run_id": run_id})
        return

    if not force and not db.is_changed(article.url, article.content_hash):
        db.update_last_scraped(article.url)
        if article.source_path is not None:
            db.touch_source_path(article.url, article.source_path)
        stats["skipped"] += 1
        log.debug("Article unchanged — skipped", extra={"run_id": run_id, "url": article.url})
        return

    transcriptions = await transcribe_images(article, context)
    chunks = chunk_article(article, transcriptions)
    embedded = embed_chunks(chunks)
    model_name = embedded[0].model_name if embedded else get_embedding_provider().model_name

    existed = db.get_article(article.url) is not None
    if existed:
        store.delete_by_url(article.url)
        stats["updated"] += 1
        log.info(
            "Article updated — re-ingested",
            extra={
                "run_id": run_id,
                "url": article.url,
                "chunk_count": len(chunks),
                "source": article.source,
            },
        )
    else:
        stats["new"] += 1
        log.info(
            "Article new — ingested",
            extra={
                "run_id": run_id,
                "url": article.url,
                "chunk_count": len(chunks),
                "source": article.source,
            },
        )

    if embedded:
        store.upsert(embedded)
    db.upsert_article(article, len(chunks), model_name)


# ── corpus sync (Mode 1) ───────────────────────────────────────────────────


async def run_corpus_sync(
    corpus_dir: str | None = None,
    dry_run: bool = False,
    only: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    corpus_dir = corpus_dir or MD_CORPUS_DIR
    db = MetadataDB()
    if (open_id := db.has_open_run()) is not None:
        log.warning("Concurrent run detected — aborting", extra={"existing_run_id": open_id})
        db.close()
        return _stats()

    run_id = db.start_run("corpus", corpus_dir=corpus_dir)
    stats = _stats()
    projected = {"files": 0, "images": 0, "images_uncached": 0}
    store: VectorStore | None = None
    started = time.monotonic()
    try:
        try:
            paths = md_loader.iter_article_paths(corpus_dir)
        except md_loader.CorpusNotFoundError:
            log.critical(
                "Corpus directory not found",
                extra={"run_id": run_id, "corpus_dir": corpus_dir},
                exc_info=True,
            )
            db.finish_run(run_id, stats, error="corpus_not_found")
            return stats
        except md_loader.CorpusEmptyError:
            log.critical(
                "Corpus directory contains no .md files",
                extra={"run_id": run_id, "corpus_dir": corpus_dir},
            )
            db.finish_run(run_id, stats, error="corpus_empty")
            return stats

        if only:
            wanted = set(only)
            paths = [p for p in paths if p.stem in wanted]
            if missing := sorted(wanted - {p.stem for p in paths}):
                log.critical(
                    "--only named files that are not in the corpus",
                    extra={"run_id": run_id, "missing": missing},
                )
                db.finish_run(run_id, stats, error="selection_not_found")
                return stats
        if limit is not None:
            paths = paths[:limit]
        if only or limit is not None:
            log.info(
                "Partial corpus run",
                extra={"run_id": run_id, "selected": len(paths), "only": only, "limit": limit},
            )

        known_urls = db.get_known_urls()
        log.info(
            "Corpus scanned",
            extra={
                "run_id": run_id,
                "corpus_dir": corpus_dir,
                "file_count": len(paths),
                "known_url_count": len(known_urls),
            },
        )

        if not dry_run:
            store = VectorStore()

        for path in paths:
            try:
                article = md_loader.load_article(path)
                if article is None:
                    stats["failed"] += 1
                    log.error(
                        "Article could not be loaded",
                        extra={"run_id": run_id, "md_file": str(path)},
                    )
                    continue

                if dry_run:
                    status = (
                        "new"
                        if article.url not in known_urls
                        else "changed"
                        if db.is_changed(article.url, article.content_hash)
                        else "unchanged"
                    )
                    projected["files"] += 1
                    projected["images"] += len(article.images)
                    projected["images_uncached"] += count_uncached(article.images)
                    log.info(
                        "DRY RUN — would ingest",
                        extra={
                            "run_id": run_id,
                            "md_file": str(path),
                            "url": article.url,
                            "status": status,
                            "word_count": article.word_count,
                            "image_count": len(article.images),
                        },
                    )
                    continue

                await ingest_article(article, None, stats, run_id, force, db=db, store=store)
            except Exception:
                stats["failed"] += 1
                log.error(
                    "Article ingestion failed",
                    extra={"run_id": run_id, "md_file": str(path)},
                    exc_info=True,
                )
                continue

        if dry_run:
            log.info("DRY RUN — projected cost", extra={"run_id": run_id, **projected})
        db.finish_run(run_id, stats)
        log.info(
            "Corpus sync complete",
            extra={
                "run_id": run_id,
                "corpus_dir": corpus_dir,
                **stats,
                "elapsed_s": round(time.monotonic() - started, 1),
            },
        )
        return stats
    except BaseException:
        log.critical("Run crashed", extra={"run_id": run_id}, exc_info=True)
        db.finish_run(run_id, stats, error="crashed")
        raise
    finally:
        db.close()


# ── reset ──────────────────────────────────────────────────────────────────


async def run_reset(assume_yes: bool = False) -> dict:
    """Empty the index: drop the Qdrant collection, THEN delete every article row.
    Keeps the image transcription cache and the ingestion_runs history."""
    db = MetadataDB()
    store = VectorStore()
    articles = db.get_stats()["articles"]["total"]
    points = store.count()
    log.warning("Reset requested", extra={"article_count": articles, "point_count": points})

    if not assume_yes:
        if not sys.stdin.isatty():
            log.error("Reset refused — not a tty and --yes not passed")
            db.close()
            return {"articles_removed": 0, "points_removed": 0, "aborted": True}
        print(f"About to delete {articles} articles and {points} vectors. Type 'yes': ")
        if input().strip().lower() != "yes":
            log.info("Reset declined at prompt")
            db.finish_run(db.start_run("reset"), _stats(), error="declined")
            db.close()
            return {"articles_removed": 0, "points_removed": 0, "aborted": True}

    run_id = db.start_run("reset")
    store.drop_collection()  # vectors first
    removed = db.drop_all_articles()  # then rows
    db.finish_run(run_id, _stats())
    log.warning(
        "Reset complete",
        extra={
            "run_id": run_id,
            "articles_removed": removed,
            "points_removed": points,
            "cache_preserved": True,
        },
    )
    db.close()
    return {"articles_removed": removed, "points_removed": points, "aborted": False}


# ── prune ──────────────────────────────────────────────────────────────────


async def run_prune(dry_run: bool = False, force: bool = False) -> dict:
    db = MetadataDB()
    run_id = db.start_run("prune", corpus_dir=MD_CORPUS_DIR)
    stats = {**_stats(), "archived": 0, "chunks_removed": 0, "aborted": False}
    try:
        corpus_rows = db.get_corpus_articles()
        gone = [
            r for r in corpus_rows if not r["source_path"] or not Path(r["source_path"]).exists()
        ]
        for row in corpus_rows:
            if row not in gone:
                stats["skipped"] += 1

        if corpus_rows and len(gone) > len(corpus_rows) / 2 and not force:
            log.critical(
                "Prune would archive >50% of the corpus — aborting",
                extra={
                    "run_id": run_id,
                    "candidate_count": len(gone),
                    "total_count": len(corpus_rows),
                },
            )
            stats["aborted"] = True
            db.finish_run(run_id, _stats(), error="prune_guard")
            return stats

        store = VectorStore() if (gone and not dry_run) else None
        for row in gone:
            log.info(
                "Prune candidate found",
                extra={"run_id": run_id, "url": row["url"], "source_path": row["source_path"]},
            )
            if dry_run:
                continue
            removed = store.delete_by_url(row["url"])
            db.archive_article(row["url"])
            stats["archived"] += 1
            stats["chunks_removed"] += removed

        db.finish_run(run_id, stats)
        log.info("Prune complete", extra={"run_id": run_id, "archived_count": stats["archived"]})
        return stats
    except BaseException:
        log.critical("Prune crashed", extra={"run_id": run_id}, exc_info=True)
        db.finish_run(run_id, stats, error="crashed")
        raise
    finally:
        db.close()


# ── inspect ────────────────────────────────────────────────────────────────


def run_inspect(target: str) -> dict:
    """Print one article's full ingestion detail. Loads and stores nothing."""
    db = MetadataDB()
    try:
        path = Path(target)
        if path.exists() and path.suffix == ".md":
            article = md_loader.load_article(path)
            stored = db.get_article(article.url) if article else None
        else:
            stored = db.get_article(target)
            src = stored["source_path"] if stored else None
            article = md_loader.load_article(Path(src)) if src else None

        report = {
            "target": target,
            "loaded": article is not None,
            "url": article.url if article else (stored["url"] if stored else None),
            "stored": stored is not None,
        }
        if article:
            report["images"] = [
                {"src": i.src, "local_path": i.local_path, "caption": i.caption}
                for i in article.images
            ]
            chunks = chunk_article(article, None)
            report["chunk_count"] = len(chunks)
            report["chunk_types"] = sorted({c.content_type for c in chunks})
            report["content_hash"] = article.content_hash
        if stored:
            report["stored_hash"] = stored["content_hash"]
            report["stored_chunk_count"] = stored["chunk_count"]
            report["drift"] = bool(article) and stored["content_hash"] != article.content_hash
        log.info("Inspect", extra={"report": report})
        print(report)
        return report
    finally:
        db.close()


# ── email-triggered run (Mode 2) ───────────────────────────────────────────

_MAILBOX_ERRORS = ("MailboxConnectionError", "MailboxAuthError", "AuthRefreshError")
_EMAIL_DATA_ERRORS = ("SenderMismatchError", "NoLinksFoundError", "InvalidDomainError")


async def run_email_triggered(dry_run: bool = False) -> dict:
    """Check the mailbox for a new-report email; scrape and ingest the article if found.

    Returns all-zero stats and writes nothing if there is no new email. Marks the email
    processed only after a clean ingest — a failed run leaves it unread for the next poll.
    """
    from inbox import email_reader
    from ingestion.extractor import extract
    from scraper import crawler, login

    db = MetadataDB()
    if (open_id := db.has_open_run()) is not None:
        log.warning("Concurrent run detected — aborting", extra={"existing_run_id": open_id})
        db.close()
        return _stats()

    run_id = db.start_run("email")
    stats = _stats()
    context = None
    store: VectorStore | None = None
    started = time.monotonic()
    try:
        try:
            email_update = email_reader.read_update_email()
        except Exception as exc:
            name = type(exc).__name__
            code = (
                "mailbox_unreachable"
                if name in _MAILBOX_ERRORS
                else "email_unparseable"
                if name in _EMAIL_DATA_ERRORS
                else None
            )
            if code is None:
                raise
            log.critical(
                "Email check failed", extra={"run_id": run_id, "error_type": name}, exc_info=True
            )
            db.finish_run(run_id, stats, error=code)
            return stats

        if email_update is None:
            log.info("No new report email — run skipped", extra={"run_id": run_id})
            db.finish_run(run_id, stats)
            return stats

        log.info(
            "Email trigger received",
            extra={
                "run_id": run_id,
                "email_uid": email_update.email_uid,
                "subject": email_update.subject,
                "link_count": len(email_update.article_links),
            },
        )
        urls = [link.url for link in email_update.article_links]
        known_urls = db.get_known_urls()

        try:
            context = await login.get_authenticated_context()
        except login.ManualLoginRequiredError:
            log.critical(
                "Login required but run is not interactive — run an ingestion from a "
                "terminal once to refresh the session",
                extra={"run_id": run_id},
                exc_info=True,
            )
            db.finish_run(run_id, stats, error="login_required")
            return stats
        except login.ManualLoginTimeoutError:
            log.error(
                "Manual login was not completed in time", extra={"run_id": run_id}, exc_info=True
            )
            db.finish_run(run_id, stats, error="login_timeout")
            return stats
        log.info("Website authentication confirmed", extra={"run_id": run_id})

        try:
            raw_pages = await crawler.fetch_pages(context, urls, known_urls)
        except crawler.SessionExpiredError:
            log.critical(
                "Pre-check login was not completed", extra={"run_id": run_id}, exc_info=True
            )
            db.finish_run(run_id, stats, error="session_error")
            return stats
        except login.LoginStateError:
            log.critical(
                "Unexpected site state during pre-check", extra={"run_id": run_id}, exc_info=True
            )
            db.finish_run(run_id, stats, error="login_state_error")
            return stats

        if not dry_run:
            store = VectorStore()
        for raw_page in raw_pages:
            try:
                article = extract(raw_page)
                if dry_run:
                    log.info(
                        "DRY RUN — would ingest", extra={"run_id": run_id, "url": raw_page.url}
                    )
                    continue
                await ingest_article(article, context, stats, run_id, db=db, store=store)
            except Exception:
                stats["failed"] += 1
                log.error(
                    "Article ingestion failed",
                    extra={"run_id": run_id, "url": raw_page.url},
                    exc_info=True,
                )
                continue

        if not dry_run and stats["failed"] == 0:
            email_reader.mark_processed(email_update.email_uid)
            log.info(
                "Email marked processed",
                extra={"run_id": run_id, "email_uid": email_update.email_uid},
            )
        elif not dry_run:
            log.warning(
                "Email left unread — run did not fully succeed; the next poll will retry",
                extra={
                    "run_id": run_id,
                    "email_uid": email_update.email_uid,
                    "failed": stats["failed"],
                },
            )

        db.finish_run(run_id, stats)
        log.info(
            "Email-triggered run complete",
            extra={"run_id": run_id, **stats, "elapsed_s": round(time.monotonic() - started, 1)},
        )
        return stats
    except BaseException:
        log.critical("Run crashed", extra={"run_id": run_id}, exc_info=True)
        db.finish_run(run_id, stats, error="crashed")
        raise
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:  # noqa: BLE001
                pass
            await login.close_browser()
        db.close()


def start_scheduler() -> None:
    """Start the APScheduler blocking scheduler (email mode). Does not return."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    from config import EMAIL_POLL_INTERVAL, SCHEDULE_DAY, SCHEDULE_HOUR

    def _job():
        asyncio.run(run_email_triggered())

    scheduler = BlockingScheduler()
    scheduler.add_job(_job, "cron", day=SCHEDULE_DAY, hour=SCHEDULE_HOUR, minute=0)
    scheduler.add_job(_job, "interval", hours=EMAIL_POLL_INTERVAL)
    log.info(
        "Email scheduler started",
        extra={
            "schedule_day": SCHEDULE_DAY,
            "schedule_hour": SCHEDULE_HOUR,
            "poll_interval_h": EMAIL_POLL_INTERVAL,
        },
    )
    scheduler.start()


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="monthly_job")
    p.add_argument("--corpus", action="store_true", help="sync MD_CORPUS_DIR")
    p.add_argument("--dir", help="override MD_CORPUS_DIR for one run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", help="comma-separated .md stems")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip the --reset prompt")
    p.add_argument("--inspect", metavar="URL|PATH")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--clear-lock", action="store_true")
    p.add_argument("--once", action="store_true", help="check email once, ingest, exit")
    return p


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)

    if args.stats:
        db = MetadataDB()
        print(db.get_stats())
        db.close()
        return 0
    if args.clear_lock:
        db = MetadataDB()
        print(f"cleared {db.clear_open_runs()} stale run(s)")
        db.close()
        return 0
    if args.inspect:
        run_inspect(args.inspect)
        return 0
    if args.reset:
        asyncio.run(run_reset(assume_yes=args.yes))
        return 0
    if args.prune:
        asyncio.run(run_prune(dry_run=args.dry_run, force=args.force))
        return 0
    if args.corpus:
        only = args.only.split(",") if args.only else None
        asyncio.run(run_corpus_sync(args.dir, args.dry_run, only, args.limit, args.force))
        return 0
    if args.once:
        asyncio.run(run_email_triggered(dry_run=args.dry_run))
        return 0

    start_scheduler()  # blocking; email mode
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
