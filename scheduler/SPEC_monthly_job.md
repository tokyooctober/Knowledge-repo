# `scheduler/monthly_job.py` — Ingestion Orchestrator

---
```
module:     scheduler/monthly_job.py
spec:       scheduler/SPEC_monthly_job.md
layer:      Orchestration
depends_on: config.py · logger.py
            ingestion/md_loader.py          (iter_article_paths, load_article)   ← Phase 1
            inbox/email_reader.py           (read_update_email, mark_processed)  ← Phase 2
            scraper/login.py                (get_authenticated_context, close_browser)  ← Phase 2
            scraper/crawler.py              (fetch_pages)                        ← Phase 2
            ingestion/extractor.py          (extract)                            ← Phase 2
            ingestion/image_transcriber.py  (transcribe_images)
            ingestion/chunker.py            (chunk_article)
            ingestion/embedder.py           (embed_chunks)
            storage/vector_store.py         (upsert, delete_by_url)
            storage/metadata_db.py          (is_changed, upsert_article, start_run, finish_run)
used_by:    app.py  (manual trigger from sidebar)
            OS cron / APScheduler  (automated schedule)
files:      MD_CORPUS_DIR/*.md  (Phase 1 source, read-only)
```
---

## Purpose
Orchestrate the diff-aware ingestion pipeline across two content sources: a **corpus sync**
that loads markdown files from disk (Phase 1), and **email-triggered runs** that scrape a
newly published report from a link in a notification email (Phase 2).

Only the acquisition step differs. Both modes converge on an `Article` and share one
ingestion sub-pipeline: transcribe images → chunk → embed → store.

---

## Two Operating Modes

### Mode 1: Corpus sync (Phase 1 — initial load and ongoing re-sync)
Reads the markdown corpus at `MD_CORPUS_DIR`, ingesting files that are new or whose
content has changed. Safe and cheap to re-run: drop new markdown into the folder, run it
again, and only the delta is processed. No email, no browser, no credentials, no network
except the embedding and vision providers.

```bash
python scheduler/monthly_job.py --corpus
python scheduler/monthly_job.py --corpus --dry-run
python scheduler/monthly_job.py --corpus --dir /other/corpus
python scheduler/monthly_job.py --corpus --only 2021-06-27-premium-2021-6-27
```

The first load is worth staging rather than running blind — see
*The controlled first load* below for the dry-run → pilot → inspect → reset sequence and
what each gate is actually testing.

### Mode 2: Email-triggered run (Phase 2 — ongoing monthly updates)
The default operational mode. Checks the mailbox for an unread notification email from the
trusted sender, extracts the single new article URL, authenticates against the site, and
scrapes and ingests it.

```bash
python scheduler/monthly_job.py           # start blocking scheduler
python scheduler/monthly_job.py --once    # check email once and exit
```

| | Mode 1 | Mode 2 |
|---|---|---|
| Source | `MD_CORPUS_DIR/*.md` | Email link → live site |
| Producer of `Article` | `md_loader.load_article()` | `extractor.extract()` |
| Needs Playwright | no | yes |
| Needs credentials | no | yes |
| Change detection | `content_hash` per file | `content_hash` per scrape |
| Typical cadence | on demand, whenever files change | monthly / every 12h |

---

## Corpus Format

See [ingestion/SPEC_md_loader.md](../ingestion/SPEC_md_loader.md) for the full frontmatter
contract and folder layout. The scheduler's only concerns are:

- `md_loader.iter_article_paths()` returns the `.md` files to consider
- `md_loader.load_article(path)` returns an `Article`, or `None` for an unparseable file
- `article.url` (from frontmatter, or a synthesised `local:<stem>`) is the identity key
- `article.content_hash` drives the new / changed / unchanged decision

---

## Scheduling (email-triggered mode)

### Option A: APScheduler (in-process)
```python
from apscheduler.schedulers.blocking import BlockingScheduler
scheduler = BlockingScheduler()
scheduler.add_job(run_email_triggered, "cron", day=1, hour=3, minute=0)
scheduler.start()
```
Polls on the 1st of each month at 03:00. Because the email may arrive any day, an interval
trigger (e.g. every 12 hours) is also supported so a new report is picked up promptly.

### Option B: OS cron (recommended for production)
```
0 */12 * * * /path/to/venv/bin/python /path/to/scheduler/monthly_job.py --once
```
Checks every 12 hours and exits. The `UNSEEN` email filter prevents reprocessing.

The corpus sync is **not** scheduled. It runs when you change the corpus.

---

## Core Pipeline Logic

### Shared ingestion sub-pipeline

Both modes call this with an `Article` and an optional `BrowserContext`. It is the only
place vectors are written.

```
ingest_article(article, context=None, stats, run_id) -> None

  1. STUB GUARD
     if article is None or article.is_stub:
       stats["skipped"] += 1
       log.warning("Article stub or load failed", extra={run_id, url})
       return

  2. CHANGE DETECTION
     if not metadata_db.is_changed(article.url, article.content_hash):
       metadata_db.update_last_scraped(article.url)
       if article.source_path is not None:
         # A rename leaves content_hash identical, so this is the only chance to
         # record the new path. Skip it and the stored source_path goes stale, and
         # the next --prune archives an article that is still sitting in the corpus.
         metadata_db.touch_source_path(article.url, article.source_path)
       stats["skipped"] += 1
       log.debug("Article unchanged — skipped", extra={run_id, url})
       return

  3. VISION TRANSCRIPTION
     # image_transcriber branches per image on ImageRef.local_path:
     #   local_path set  → read from disk        (corpus)
     #   local_path None → download via context  (web)
     # context may be None in corpus mode; transcribe_images must accept that.
     transcriptions = await image_transcriber.transcribe_images(article, context)

  4. CHUNK + EMBED
     chunks   = chunker.chunk_article(article, transcriptions)
     embedded = embedder.embed_chunks(chunks)

  5. REPLACE OR INSERT
     if metadata_db.get_article(article.url):
       vector_store.delete_by_url(article.url)      # remove stale vectors FIRST
       stats["updated"] += 1
       log.info("Article updated", extra={run_id, url, chunk_count, source})
     else:
       stats["new"] += 1
       log.info("Article new — ingested", extra={run_id, url, chunk_count, source})

     vector_store.upsert(embedded)
     metadata_db.upsert_article(article, len(chunks), embedded[0].model_name)
```

`delete_by_url` before `upsert` is what keeps re-runs from accumulating duplicate vectors
for an edited article. The two are not atomic — see *Failure modes* below.

### Corpus sync pipeline (Mode 1)

```
run_id = metadata_db.start_run(trigger="corpus", corpus_dir=MD_CORPUS_DIR)
stats  = {new: 0, updated: 0, skipped: 0, failed: 0}

1. SCAN CORPUS
   TRY:
     paths = md_loader.iter_article_paths(corpus_dir)
   EXCEPT CorpusNotFoundError:
     log.critical("Corpus directory not found", extra={run_id, corpus_dir}, exc_info=True)
     metadata_db.finish_run(run_id, stats, error="corpus_not_found")
     return stats
   EXCEPT CorpusEmptyError:
     log.critical("Corpus directory contains no .md files",
                  extra={run_id, corpus_dir})
     metadata_db.finish_run(run_id, stats, error="corpus_empty")
     return stats

   if only:                              # --only STEM,STEM
     wanted  = set(only)
     paths   = [p for p in paths if p.stem in wanted]
     missing = wanted - {p.stem for p in paths}
     if missing:
       log.critical("--only named files that are not in the corpus",
                    extra={run_id, corpus_dir, missing: sorted(missing)})
       metadata_db.finish_run(run_id, stats, error="selection_not_found")
       return stats
   if limit:                             # --limit N, applied after --only
     paths = paths[:limit]
   if only or limit:
     log.info("Partial corpus run", extra={run_id, selected: len(paths),
                                           only: only, limit: limit})

   known_urls = metadata_db.get_known_urls()
   log.info("Corpus scanned", extra={run_id, corpus_dir, file_count=len(paths),
                                     known_url_count=len(known_urls)})

2. INGEST EACH FILE
   for path in paths:
     TRY:
       article = md_loader.load_article(path)
       if article is None:
         stats["failed"] += 1
         log.error("Article could not be loaded", extra={run_id, md_file: str(path)})
         continue

       if dry_run:
         status = "new" if article.url not in known_urls \
                  else ("changed" if metadata_db.is_changed(article.url,
                                                            article.content_hash)
                        else "unchanged")
         projected["files"]    += 1
         projected["images"]   += len(article.images)
         projected["uncached"] += count_uncached(article.images)   # image_transcriber
         log.info("DRY RUN — would ingest", extra={run_id, md_file, url, status,
                                                   word_count, image_count})
         continue

       await ingest_article(article, context=None, stats=stats, run_id=run_id,
                            force=force)

     EXCEPT Exception:
       stats["failed"] += 1
       log.error("Article ingestion failed",
                 extra={run_id, md_file: str(path), error_type}, exc_info=True)
       continue   # one bad file never aborts the run

3. FINISH
   metadata_db.finish_run(run_id, stats)
   log.info("Corpus sync complete", extra={run_id, corpus_dir, ...stats, elapsed_s})
   return stats
```

No `BrowserContext` is created and no credentials are read. A corpus sync must succeed on
a machine that has never had Playwright installed.

### Email-triggered pipeline (Mode 2)

```
run_id = metadata_db.start_run(trigger="email")
stats  = {new: 0, updated: 0, skipped: 0, failed: 0}

1. CHECK EMAIL
   TRY:
     email_update = email_reader.read_update_email()
   EXCEPT (MailboxConnectionError, MailboxAuthError, AuthRefreshError) as e:
     log.critical("Mailbox unreachable or auth failed",
                  extra={run_id, error_type}, exc_info=True)
     metadata_db.finish_run(run_id, stats, error="mailbox_unreachable")
     return stats
   EXCEPT (SenderMismatchError, NoLinksFoundError, InvalidDomainError) as e:
     # A malformed or unexpected email is a data problem, not an outage.
     log.error("Email could not be parsed — no URL extracted",
               extra={run_id, error_type}, exc_info=True)
     metadata_db.finish_run(run_id, stats, error="email_unparseable")
     return stats

   If email_update is None:
     log.info("No new report email — run skipped", extra={run_id})
     metadata_db.finish_run(run_id, stats)
     return stats

   log.info("Email trigger received",
            extra={run_id, email_uid, sender, subject, received_at, link_count})

   urls = [link.url for link in email_update.article_links]

2. FILTER ALREADY-INDEXED
   known_urls = metadata_db.get_known_urls()
   # URLs already present are still fetched: the email announces a new report, so a
   # match here means either a resend or an edit. The content_hash check in
   # ingest_article() decides whether anything is re-embedded.
   log.debug("URLs from email", extra={run_id, url_count: len(urls),
                                       already_known: len(set(urls) & known_urls)})

3. AUTHENTICATE
   TRY:
     context = await login.get_authenticated_context()
   EXCEPT ManualLoginRequiredError:
     # Session expired and this run has no human to ask (cron, INTERACTIVE_LOGIN=never).
     # login.py raises this immediately — it does NOT wait out MANUAL_LOGIN_TIMEOUT_MS.
     log.critical("Login required but run is not interactive — "
                  "run an ingestion from a terminal once to refresh the session",
                  extra={run_id}, exc_info=True)
     metadata_db.finish_run(run_id, stats, error="login_required")
     return stats          # the finally block closes context AND browser
   EXCEPT ManualLoginTimeoutError:
     log.error("Manual login was not completed in time", extra={run_id}, exc_info=True)
     metadata_db.finish_run(run_id, stats, error="login_timeout")
     return stats
   log.info("Website authentication confirmed", extra={run_id})
   # Login walls on individual article URLs are resolved by the crawler through
   # login.ensure_authenticated(), which asks the human and then returns the page to
   # the requested URL. The scheduler is not involved in per-URL logins.

   # NOTE: this step can block for minutes while a person signs in. That is deliberate
   # and must not be wrapped in a timeout — the alternative is a failed run.

4. FETCH PAGES
   TRY:
     raw_pages = await crawler.fetch_pages(context, urls, known_urls)
   EXCEPT SessionExpiredError:
     log.critical("Pre-check login was not completed — no human available, "
                  "or the sign-in timed out",
                  extra={run_id}, exc_info=True)
     metadata_db.finish_run(run_id, stats, error="session_error")
     return stats          # the finally block closes context AND browser
   EXCEPT LoginStateError:
     log.critical("Unexpected site state during pre-check — aborting",
                  extra={run_id}, exc_info=True)
     metadata_db.finish_run(run_id, stats, error="login_state_error")
     return stats          # the finally block closes context AND browser

5. INGEST EACH PAGE
   for raw_page in raw_pages:
     TRY:
       article = extractor.extract(raw_page)
       if dry_run:
         log.info("DRY RUN — would ingest", extra={run_id, url: raw_page.url})
         continue
       await ingest_article(article, context=context, stats=stats, run_id=run_id)
     EXCEPT Exception:
       stats["failed"] += 1
       log.error("Article ingestion failed",
                 extra={run_id, url, error_type}, exc_info=True)
       continue

6. MARK THE EMAIL PROCESSED  (only now, after a successful ingest)
   if not dry_run and stats["failed"] == 0:
     email_reader.mark_processed(email_update.email_uid)
     log.info("Email marked processed", extra={run_id, email_uid})
   else:
     log.warning("Email left unread — run did not fully succeed; the next poll "
                 "will retry it", extra={run_id, email_uid, failed: stats["failed"]})

7. FINISH   (the finally block closes the context and the browser)
   metadata_db.finish_run(run_id, stats)
   log.info("Email-triggered run complete", extra={run_id, ...stats, elapsed_s})
   return stats
```

Teardown lives in the `finally` block described under *`finish_run` is mandatory on every
exit path* below — never inline at each return, which is how the abort paths in step 4 got
missed in an earlier revision.

---

## Idempotency Guarantees

| Scenario | Behaviour |
|---|---|
| Corpus file already indexed, content unchanged | Skipped (hash match); `stats["skipped"]` |
| Corpus file already indexed, content edited | Old vectors deleted, re-embedded; `stats["updated"]` |
| Corpus file renamed, content unchanged | Skipped (hash ignores the `.md` path); `source_path` is updated in place so `--prune` still sees the file |
| Corpus file's frontmatter `url` edited | Treated as a **new** article; the old row is orphaned (see below) |
| Same corpus synced twice with no changes | Second run is all skips — no duplicate vectors |
| New `.md` added to the corpus | Ingested on the next `--corpus` run |
| `.md` deleted from the corpus | **Not removed from the index** — see below |
| Email URL already in database, content unchanged | Skipped; email marked processed (nothing to retry) |
| Email run fails before the ingest completes | Email left UNREAD — the next poll retries it instead of losing the month's report |
| Process dies between a successful ingest and `mark_processed` | Next poll re-ingests; the unchanged hash makes it a skip |
| Email URL already in database, content changed | Re-embedded; `stats["updated"]` |
| Scheduler fires while a previous run is active | Detected via open `ingestion_runs` row; WARNING, new run aborted |
| Login prompt on article URL | Handled inline by `crawler.py`; scheduler unaware |
| Inline login fails on one URL | That URL skipped; run continues |
| Pre-check login fails (credentials wrong) | Run aborted; recorded as `session_error` |

### Deletions are not synced

A corpus sync only ever adds and updates. Removing a `.md` file leaves its article and
vectors in place, and it will keep appearing as a citation. Same for an article whose
frontmatter `url` was edited — the old identity is orphaned.

This is deliberate: a sync that deletes from the index on the basis of a file being absent
will wipe the database the first time someone runs it with `MD_CORPUS_DIR` pointing at the
wrong folder, or at a folder that is still syncing from cloud storage. Reconciliation is an
explicit, separate operation:

```bash
python scheduler/monthly_job.py --prune --dry-run   # list indexed corpus articles with no file
python scheduler/monthly_job.py --prune             # archive them: drop vectors, keep the row
```

`--prune` **archives, it does not delete**. For each corpus article whose `.md` file is
gone it calls `vector_store.delete_by_url(url)` — so the article stops being retrieved and
can no longer appear as a citation — and `metadata_db.archive_article(url)`, which sets
`status = 'archived'` and keeps the row. Nothing in this system ever hard-deletes an
article.

That combination is what makes `--prune` safe to run on a hunch. Pointing
`MD_CORPUS_DIR` at the wrong folder, or at one still syncing from cloud storage, archives
everything — recoverable by restoring the corpus and re-running `--corpus`, which sees the
files as changed (no vectors, archived row) and re-ingests them. The same mistake against a
hard delete would be unrecoverable without a re-embed of the entire back-catalogue anyway,
but would also lose `first_scraped` and the run history.

`--prune` considers only articles with `source = "corpus"`; web-sourced articles have no
file to be missing and are never touched.

An archived article whose `.md` file reappears is picked up by the next `--corpus` run:
`upsert_article` sets `status = 'active'` again.

### Failure modes worth knowing

- `delete_by_url` then `upsert` is not a transaction. A crash between them leaves an
  article with a metadata row and no vectors. It self-heals on the next sync only if the
  content hash also changed — otherwise the article is silently unsearchable. Recover with
  `--url <url>` (forces re-ingest regardless of hash).
- `metadata_db.upsert_article` runs after `vector_store.upsert`. A crash between them
  leaves vectors with no metadata row, so the next run treats the article as new and
  writes a second copy. `delete_by_url` at the start of the next ingest cleans this up.
- A run that crashes hard leaves an open `ingestion_runs` row, which blocks the next run
  via the concurrency guard. `--stats` reports it; `--clear-lock` closes it.

### `finish_run` is mandatory on every exit path

`has_open_run()` blocks **all** modes, not just the one that crashed — an unparseable
notification email would otherwise wedge the corpus sync too, silently, on a cron
schedule. So every public entry point wraps its whole body:

```python
run_id = metadata_db.start_run(trigger=…, **context)
try:
    ...                                   # the pipeline
except BaseException:
    log.critical("Run crashed", extra={run_id}, exc_info=True)
    metadata_db.finish_run(run_id, stats, error="crashed")
    raise
finally:
    if context is not None:               # Phase 2 only
        await context.close()             # the BrowserContext
        await login.close_browser()       # the Browser AND the Playwright driver
```

`context.close()` alone is not teardown. `get_authenticated_context()` also starts a
Playwright driver process and a Browser, and returns neither — so without
`login.close_browser()` a `--once` run under cron leaks a Chromium and a node process
every 12 hours. `close_browser()` is idempotent; call it on every Phase 2 exit path,
including the two step-4 aborts.

The bare `except BaseException` exists to close the run row, not to swallow the error — it
re-raises. Catching only `Exception` would leave the row open on `KeyboardInterrupt`, which
is exactly how a cron job gets killed mid-run.

The guard is advisory, not a lock: two processes started in the same second can both pass
it. That is acceptable for a single-user system, and `delete_by_url`-before-`upsert` limits
the damage to wasted work rather than duplicate vectors.

---

## CLI Interface

```
Usage:
  python scheduler/monthly_job.py                        Start blocking scheduler (email mode)
  python scheduler/monthly_job.py --once                 Check email once, ingest if found, exit
  python scheduler/monthly_job.py --corpus               Sync MD_CORPUS_DIR (new + changed)
  python scheduler/monthly_job.py --corpus --dir PATH    Sync a different corpus directory
  python scheduler/monthly_job.py --corpus --dry-run     Report what would change, write nothing
  python scheduler/monthly_job.py --corpus --only A,B    Sync only these files (by .md stem)
  python scheduler/monthly_job.py --corpus --limit N     Sync at most the first N eligible files
  python scheduler/monthly_job.py --corpus --force       Re-ingest regardless of content_hash
  python scheduler/monthly_job.py --prune [--dry-run]    Archive corpus articles whose .md file is gone
  python scheduler/monthly_job.py --prune --force        Override the >50%-of-corpus safety abort
  python scheduler/monthly_job.py --dry-run              Email dry-run: parse email, log URL, no writes
  python scheduler/monthly_job.py --url URL              Re-process a single URL (force re-ingest)
  python scheduler/monthly_job.py --file PATH            Re-process a single .md file (force re-ingest)
  python scheduler/monthly_job.py --inspect URL|PATH     Dump one article's full ingestion detail
  python scheduler/monthly_job.py --reset                Empty the index and start over (prompts)
  python scheduler/monthly_job.py --stats                Print database statistics and exit
  python scheduler/monthly_job.py --clear-lock           Close a stale open run row and exit
```

`--dry-run` with `--corpus` logs every file with its status (`new` / `changed` /
`unchanged`) and writes nothing to Qdrant or SQLite. Run it after any change to
`MD_CORPUS_DIR` before committing to a real sync.

`--dir` overrides `MD_CORPUS_DIR` for one run without editing config — useful for testing
against a fixture corpus.

### `--only` and `--limit`: ingesting part of the corpus

Both narrow which files a `--corpus` run processes, and **neither changes anything else**:
the scan, the `known_urls` lookup, the new/changed/unchanged decision, and the whole
ingestion sub-pipeline are the code the full run uses. That is the entire point. A staged
first load is only evidence about the full load if the two run the same path.

- `--only STEM[,STEM…]` selects by `.md` filename stem, so a deliberate choice can be
  named and repeated: `--only 2021-06-27-premium-2021-6-27`. A stem that matches no file
  is a CRITICAL and an abort, not a silent no-op — a typo that quietly ingests nothing
  looks exactly like a run that found nothing to do.
- `--limit N` takes the first N files of the same sorted order `iter_article_paths()`
  returns. Useful for "prove it survives fifty", useless for choosing representative
  articles, because sorted order means oldest-first and the oldest exports are the least
  typical.

Prefer these over the two things they replace:

`--file PATH` looks like the way to ingest one article, and is not, for a staged load: it
force-re-ingests a single path and skips the scan and the change-detection branch
entirely, so it exercises a path the real run never takes.

Copying a few `.md` files to a scratch folder and pointing `--dir` at it is worse, and is
now a specific trap: images live in per-article subfolders under `images/`, so a copy that
takes the markdown without `images/<article-slug>/` produces articles that ingest cleanly
with zero images and no error. The pilot passes, the images half of the pipeline was never
run, and nothing says so. Run against the real corpus directory and narrow with `--only`.

### `--force`: re-ingest without a wipe

`--corpus --force` re-ingests every selected article regardless of `content_hash`,
skipping the `is_changed()` call rather than adding a case to it. It exists for the
situation `content_hash` cannot see: **the corpus is unchanged and our handling of it is
not.**

Fix a bug in image resolution, change chunk boundaries, rewrite the vision prompt — every
hash is identical, every article is "unchanged", and the index keeps its wrong vectors
forever without a single log line. `--force` re-ingests them.

Bumping `config.PIPELINE_VERSION` in the same commit as the fix does the same thing
automatically and permanently (see [SPEC_metadata_db.md](../storage/SPEC_metadata_db.md)),
and is the right answer for a change you are shipping. `--force` is for the change you are
still iterating on, where you want to re-run the same article twenty times without
bumping a version number twenty times.

`--force` combines with `--only`, which is the loop you actually want while debugging:

```bash
python scheduler/monthly_job.py --corpus --only 2021-06-27-premium-2021-6-27 --force
python scheduler/monthly_job.py --inspect https://www.example.com/premium-2021-6-27
```

Note that `--force` still pays for embeddings every time. Vision transcription is cached
on image content, so re-running the same article is cheap in the expensive half — which is
what makes this loop affordable at all.

### `--inspect`: what "validated" actually means

Takes a url or a `.md` path, loads and ingests nothing, and prints one article's full
ingestion detail: every resolved image with the subfolder it came from, each caption, the
references that were deduped, the chunk boundaries with the first and last line of each,
and one retrieval round-trip (embed a query, search, print the top hit's chunk and its
citation url).

It reads the stored row and the live corpus file and reports both, so a drift between the
index and the corpus is visible in one command rather than inferred from two.

Reading the run log is not validation. The failures this pipeline actually produces are
the ones that log nothing: an image resolved from the wrong article's folder, a caption
that is a URL, a cover transcribed three times, a url stored in two spellings. Every one
of those is a clean INFO line in the log and a wrong answer six weeks later.

### `--reset`: what it empties, and what it must not

Returns the index to a first-run state. Three persistent stores exist; `--reset` clears
two of them:

| Store | On `--reset` | Why |
|---|---|---|
| `data/metadata.db` — `articles` | **Emptied** (`drop_all_articles()`) | The index being rebuilt |
| Qdrant collection | **Dropped and recreated empty** (`drop_collection()`) | Same |
| `data/metadata.db` — `ingestion_runs` | **Kept** | The audit trail of what was ingested and when, including the runs whose articles were just deleted. A reset that turns out to be a mistake still needs to be explicable |
| `data/image_cache.db` + normalised files under `images/` | **Kept** | See below |

**The image transcription cache is not part of the index and must survive a reset.** It is
keyed on image content (`file:path:mtime:size`), not on run or article identity, so every
entry stays valid across a wipe — and vision calls are the dominant cost of this pipeline.
Clearing it makes the second attempt at a first load re-pay the entire vision bill to
produce, by construction, identical transcriptions.

This has to be stated because it is counter-intuitive in exactly the wrong direction:
"start clean" reads like it should include the cache, and deleting `images/` alongside
`data/` feels tidier than deleting one and not the other. It is the single most expensive
mistake available in this repo, and it is silent — the second run just costs money and
takes hours.

Order matters. Drop the vectors **before** emptying the metadata rows:

```
1. confirm (see below)
2. vector_store.drop_collection()
3. metadata_db.drop_all_articles()
```

Interrupted after step 2 you have empty vectors and stale rows, and the next `--corpus`
sees every article as "unchanged" and re-ingests nothing — a silently empty index.
Interrupted after step 3, or in the reverse order, you have orphaned vectors that no url
lookup will ever clean up, because `delete_by_url` needs a row to know what to delete.
Both are bad; the first is recoverable with `--force`, the second needs a second reset.

`--reset` prompts for confirmation on stdin, printing the current article and point counts
first, and aborts on anything but an explicit yes. It refuses to run non-interactively
unless `--yes` is also passed, so it can never be reached from a cron line or a scheduled
run by accident. It writes an `ingestion_runs` row with `trigger='reset'`.

`--reset` does not ingest. It leaves an empty index and exits, so the run that repopulates
it is a separate, ordinary `--corpus` whose stats are its own.

---

## The controlled first load

The corpus is a few hundred articles and one full pass costs real money in vision and
embedding calls, so the first load is worth staging. Four gates, cheapest first — each one
answers a question the next one is too expensive to be asking.

### Gate 1 — dry-run the whole corpus

```bash
python scheduler/monthly_job.py --corpus --dry-run
```

Free: no embedding calls, no vision calls, no writes. It runs `load_article()` over every
file, so it surfaces every loader-level problem in the corpus at once — malformed
frontmatter, a missing image subfolder, `truncated: true`, an `images_saved` mismatch, a
synthesised `local:` url.

**This gate exists because the pilot cannot replace it.** Two articles validate the
pipeline; they say nothing about file 217. This is the only step that is both complete and
free, so it runs first and it runs over everything.

Pass condition: zero ERROR lines, and every WARNING read rather than counted. A corpus
this size will produce some; the point is to have decided about each one.

The dry run also prints the projected cost of the full load — total files, total resolved
images, and of those, how many are already in the transcription cache. That number is the
input to the decision at gate 4, and it is available before anything has been spent.

### Gate 2 — ingest two articles end to end

```bash
python scheduler/monthly_job.py --corpus --only <image-heavy>,<structurally-odd>
```

Choose deliberately, not alphabetically. One article with many images exercises
transcription, dedup and the cache — the expensive, most breakable half. One structurally
odd article (many tables, or a `truncated: true`, or the oldest export format) exercises
the parsing edges. Sorted order gives you neither, which is why `--limit` is the weaker
tool here.

### Gate 3 — inspect, against a written checklist

```bash
python scheduler/monthly_job.py --inspect <url>
```

Check, for each pilot article:

- [ ] every image's `local_path` is inside **that article's own** `images/<slug>/` folder —
      the basename-resolution regression is silent and this is the only place it surfaces
- [ ] the cover produced exactly one `ImageRef`, not two or three
- [ ] no caption is a URL or a bare date
- [ ] the stored url is the canonical form, and matches what a Phase 2 email link for the
      same article would normalise to
- [ ] `images_saved` matches the number of images resolved
- [ ] chunk count is plausible for the word count, and no chunk splits a table mid-row
- [ ] the retrieval round-trip returns a relevant chunk and cites the right url

Then re-run gate 2's command with `--force` and confirm the article ingests to the *same*
chunk count. A pipeline that is not deterministic across two runs of one unchanged file
will not be debuggable at three hundred.

### Gate 4 — reset, then load

Estimate first, from the pilot's actual numbers and gate 1's projection: images per
article and chunks per article, times the corpus size, minus what the cache already holds.
Decide whether to spend it before spending it.

```bash
python scheduler/monthly_job.py --reset
python scheduler/monthly_job.py --corpus
```

**Consider skipping the reset.** Its only job is to remove vectors produced by code you no
longer trust. If gates 2 and 3 passed, the pilot articles are good, and a plain `--corpus`
run treats them as unchanged and skips them — correct, and two articles cheaper. Reset
when you changed the pipeline between the pilot and the full run and did not bump
`PIPELINE_VERSION`, or when you want `first_scraped` to mean something across the whole
corpus. Not as ceremony.

A `--corpus` run over a few hundred articles is long and each article is committed as it
finishes, so an interruption is not a disaster: re-running resumes, because everything
already stored is unchanged and skipped. Interrupting is safe; the concurrency guard, not
the pipeline, is what makes two simultaneous runs a problem.

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.scheduler.monthly_job"
```

| Event | Level | Extra fields |
|---|---|---|
| Run started | INFO | `run_id`, `trigger` (`"corpus"` / `"email"` / `"manual"`) |
| Corpus scanned | INFO | `run_id`, `corpus_dir`, `file_count` |
| Corpus directory not found | CRITICAL | `run_id`, `corpus_dir`, `exc_info=True` |
| Corpus directory empty | CRITICAL | `run_id`, `corpus_dir` |
| Article could not be loaded | ERROR | `run_id`, `md_file` |
| Email trigger received | INFO | `run_id`, `email_uid`, `subject`, `link_count` |
| No new report email | INFO | `run_id` |
| Mailbox unreachable | CRITICAL | `run_id`, `error_type`, `exc_info=True` |
| Authentication started | DEBUG | `run_id` |
| Authentication succeeded | INFO | `run_id` |
| Pre-check login failed — aborting | CRITICAL | `run_id`, `error_type`, `exc_info=True` |
| Inline login on URL handled by crawler | — | logged by `scraper/login.py` and `scraper/crawler.py` |
| Fetching pages | INFO | `run_id`, `url_count` |
| Article stub — skipped | WARNING | `run_id`, `url`, `word_count` |
| Article unchanged — skipped | DEBUG | `run_id`, `url` |
| Article new — ingested | INFO | `run_id`, `url`, `chunk_count`, `source` |
| Article updated — re-ingested | INFO | `run_id`, `url`, `chunk_count`, `source` |
| Article failed | ERROR | `run_id`, `url` or `md_file`, `error_type`, `exc_info=True` |
| DRY RUN — would ingest | INFO | `run_id`, `url`, `status`, `md_file` (corpus mode) |
| DRY RUN — projected cost | INFO | `run_id`, `files`, `images`, `images_uncached` |
| Partial corpus run | INFO | `run_id`, `selected`, `only`, `limit` |
| `--only` named files not in the corpus — aborting | CRITICAL | `run_id`, `corpus_dir`, `missing` |
| Article re-ingested by `--force` | INFO | `run_id`, `url`, `chunk_count` |
| Article re-ingested — pipeline version changed | INFO | `run_id`, `url`, `stored_version`, `current_version` |
| Reset requested | WARNING | `run_id`, `article_count`, `point_count` |
| Reset declined at prompt | INFO | `run_id` |
| Reset complete | WARNING | `run_id`, `articles_removed`, `points_removed`, `cache_preserved=True` |
| Prune candidate found | INFO | `run_id`, `url`, `source_path` |
| Article archived (prune) | INFO | `run_id`, `url`, `source_path`, `chunks_removed` |
| Prune complete | INFO | `run_id`, `archived_count` |
| Prune would archive >50% of the corpus — aborting | CRITICAL | `run_id`, `corpus_dir`, `candidate_count`, `total_count` |
| Dry-run mode — no writes | WARNING | `run_id`, `trigger` |
| Concurrent run detected | WARNING | `run_id`, `existing_run_id` |
| Run complete | INFO | `run_id`, `new`, `updated`, `skipped`, `failed`, `elapsed_s` |
| Run crashed | CRITICAL | `run_id`, `error_type`, `exc_info=True` |

Every corpus-mode log line carries `md_file` alongside `url`, so a warning about a
synthesised `local:` id can be traced back to the file that needs its frontmatter fixed.

---

## Configuration Constants

```python
MD_CORPUS_DIR        = os.environ.get("MD_CORPUS_DIR", "corpus")   # Phase 1 source
PIPELINE_VERSION     = 1      # bump when loader/chunker/transcriber semantics change
SCHEDULE_DAY         = 1      # day of month (APScheduler mode)
SCHEDULE_HOUR        = 3      # hour of day
EMAIL_POLL_INTERVAL  = 12     # hours between checks (interval mode)
LOG_FILE             = "logs/knowledge_repo.log"
LOG_LEVEL            = "INFO"
```

---

## Public Interface

```python
async def run_corpus_sync(
    corpus_dir: str | None = None,
    dry_run: bool = False,
    only: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Ingest new and changed markdown articles from the corpus directory.

    Defaults to MD_CORPUS_DIR. Skips unchanged files (content_hash match and
    matching pipeline_version), re-embeds changed ones, ingests new ones.
    Never removes anything — see run_prune().

    only:  restrict to these .md filename stems. A stem matching no file aborts
           the run with error="selection_not_found" — never a silent no-op.
    limit: process at most this many files, applied after `only`, in the sorted
           order iter_article_paths() returns.
    force: ingest every selected article regardless of is_changed(). Does not
           bypass the stub check, and does not make ingestion destructive —
           delete_by_url still runs before upsert, as on any re-ingest.

    Neither `only` nor `limit` changes any other behaviour: same scan, same
    change detection, same sub-pipeline. A staged load is only evidence about
    the full load if it runs the same path.

    Returns stats dict: {new, updated, skipped, failed}.
    Writes nothing if dry_run=True; a dry run additionally logs the projected
    cost of the selection (files, images, images not already transcription-cached).
    """

async def run_reset(assume_yes: bool = False) -> dict:
    """Empty the index: drop the Qdrant collection, then delete every article row.

    In that order — vectors first. Interrupted the other way round leaves
    orphaned vectors that no url lookup can reach, because delete_by_url needs
    the metadata row to know what to delete.

    KEEPS the image transcription cache (data/image_cache.db and the normalised
    files under images/) and the ingestion_runs history. The cache is keyed on
    image content, not on run identity, so every entry survives a reset — and
    vision calls are the dominant cost of a full load. Clearing it re-pays the
    entire vision bill to regenerate identical transcriptions.

    Prompts on stdin with the current article and point counts. Refuses to run
    without a tty unless assume_yes=True, so no scheduled run can reach it.
    Does not ingest: leaves an empty index and exits.
    Returns {"articles_removed": int, "points_removed": int}.
    """

def run_inspect(target: str) -> dict:
    """Print one article's full ingestion detail. Loads and stores nothing.

    target is a url or a .md path. Reports, from the live corpus file and the
    stored row side by side: resolved images with their source subfolder,
    captions, deduped references, chunk boundaries, and one retrieval
    round-trip (embed a query, search, print the top hit and its citation url).

    Exists because the run log cannot validate this pipeline: an image resolved
    from the wrong article's folder, a caption that is a URL, a cover transcribed
    three times and a url stored in two spellings all log as clean INFO lines.
    """

async def run_email_triggered(dry_run: bool = False) -> dict:
    """Check mailbox for a new report email; scrape and ingest the article if found.

    Returns stats dict: {new, updated, skipped, failed}.
    Returns immediately with all-zero stats if no new email.
    Writes nothing if dry_run=True.
    """

async def run_prune(dry_run: bool = False, force: bool = False) -> dict:
    """Archive corpus articles whose source_path no longer exists on disk.

    For each: vector_store.delete_by_url(url) then metadata_db.archive_article(url).
    The metadata row survives with status='archived'; nothing is hard-deleted.
    Never touches source="web" articles.

    Refuses to run and returns aborted=True if more than half the indexed corpus
    articles are candidates, unless force=True. That is the signature of a wrong
    MD_CORPUS_DIR or a corpus that has not finished syncing from cloud storage,
    not of a real cleanup. --force exposes the override on the CLI.

    Returns the SAME stats shape as the other two modes, so finish_run() can
    write it without special-casing:
        {"new": 0, "updated": 0, "skipped": int, "failed": 0,
         "archived": int, "chunks_removed": int, "aborted": bool}
    "skipped" counts corpus articles whose file was found and left alone.
    The extra keys are ignored by finish_run's column mapping and appear only in
    the run-complete log line. On abort every counter is 0 and aborted=True.
    """

async def ingest_article(
    article: Article | None,
    context: "BrowserContext | None",
    stats: dict,
    run_id: str,
    force: bool = False,
) -> None:
    """Shared sub-pipeline: change check → transcribe → chunk → embed → store.

    force=True skips the is_changed() call. It does not skip the stub check and
    does not change what is written.

    context is None for corpus articles; transcribe_images must handle that.
    Mutates stats in place. Raises only on programmer error — I/O failures are
    caught by the caller's per-article try block.
    """

def start_scheduler() -> None:
    """Start APScheduler blocking scheduler (email mode). Does not return."""
```

---

## Testing Notes

Mock all I/O (`md_loader`, `email_reader`, `login`, `crawler`, `vector_store`,
`metadata_db`, `image_transcriber`).

Selection, force and reset:

- Assert `--only` with two stems processes exactly those two files and that the
  new/changed/unchanged decision for each is the one the unfiltered run would have made
- Assert `--only` with an unknown stem aborts with `error="selection_not_found"` and
  ingests nothing — the regression is a typo that silently ingests zero files and looks
  like success
- Assert `--limit N` applies after `--only`, and that both together never process a file
  outside the `--only` set
- Assert `--force` re-ingests an article whose `content_hash` and `pipeline_version` both
  match, and that `delete_by_url` still runs before the upsert
- Assert `--force` does **not** ingest a stub
- Assert a bumped `PIPELINE_VERSION` makes an otherwise unchanged article re-ingest with
  no flags passed
- Assert `--reset` calls `drop_collection()` before `drop_all_articles()`
- Assert `--reset` does not touch the image cache: patch the cache module and assert no
  clear/delete call reaches it (this is the expensive silent regression — a reset that
  clears the cache costs a full vision re-run and raises no error)
- Assert `--reset` leaves `ingestion_runs` rows intact
- Assert `--reset` aborts when stdin is not a tty and `--yes` was not passed
- Assert a `--corpus` run interrupted after k articles, then re-run, ingests only the
  remaining files

**Corpus mode**
- Assert an unchanged corpus produces all skips and zero `vector_store.upsert` calls
- Assert an edited file calls `delete_by_url` **before** `upsert`, exactly once each
- Assert a renamed file with identical content is skipped (hash is path-independent)
- Assert a new file increments `new` and calls `upsert` once
- Assert `load_article` returning `None` increments `failed` and the run continues
- Assert a stub article increments `skipped` and is never embedded
- Assert `--dry-run` calls no write method on any storage component
- Assert `CorpusNotFoundError` and `CorpusEmptyError` each abort with the right error code
- Assert no `BrowserContext` is created and `login` is never imported at call time
- Assert `transcribe_images` is called with `context=None`
- Assert a deleted file leaves the index untouched (no implicit prune)

**Prune**
- Assert only `source="corpus"` articles with a missing `source_path` are selected
- Assert `source="web"` articles are never pruned even if `source_path` is None
- Assert `--prune --dry-run` writes nothing
- Assert prune calls `delete_by_url` **and** `archive_article`, and never `delete_article`
- Assert the row survives with `status='archived'` and its `first_scraped` intact
- Assert an archived article is absent from `get_known_urls()` and from search results
- Assert restoring the `.md` file and running `--corpus` sets `status='active'` and
  re-embeds it (the archived row must not cause an unchanged-hash skip with no vectors)
- Assert the >50% guard aborts without writing, and that `force=True` overrides it

**Email mode**
- Assert all-zero stats and no writes when `read_update_email()` returns `None`
- Assert `stats["new"] == 1` for a freshly parsed email URL
- Assert `stats["skipped"] == 1` if the URL is indexed and content is unchanged
- Assert a single article failure increments `failed` and the run continues
- Assert `MailboxConnectionError` records the error and returns without ingesting
- Assert `mark_processed` is NOT called when the crawl or ingest fails, and IS called
  after a clean run
- Assert `mark_processed` is not called under `--dry-run`
- Assert `context.close()` **and** `login.close_browser()` are both called on the
  `SessionExpiredError`, `LoginStateError`, `ManualLoginRequiredError` and
  `ManualLoginTimeoutError` abort paths, not just the happy path
- Mock `get_authenticated_context` raising `ManualLoginRequiredError` → assert the run
  records `login_required` and returns without entering the fetch loop
- Assert `close_browser()` is never called in corpus mode (no browser was ever started)
- Mock crawler raising `SessionExpiredError` → assert run aborts and records `session_error`
- Mock crawler returning a partial `raw_pages` list → assert the returned pages are
  processed and the skipped count is right
- Inline login is tested in `scraper/crawler.py` tests — the scheduler is unaware of it

**Both modes**
- Assert the concurrent-run guard fires when an open `ingestion_runs` row exists
- Assert a top-level crash still calls `metadata_db.finish_run`
- Assert both modes reach `ingest_article` with an `Article` the sub-pipeline cannot
  distinguish by source, except for the `context` argument
