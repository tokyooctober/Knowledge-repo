# `app.py` — Query Interface

---
```
module:     app.py
spec:       SPEC_app.md
layer:      Interface  (entry point)
depends_on: config.py · logger.py
            query/retriever.py   (retrieve)
            query/answerer.py    (answer)
            scheduler/monthly_job.py  (run_corpus_sync, run_email_triggered — sidebar trigger)
            storage/metadata_db.py   (get_stats, get_run_history — sidebar display)
used_by:    user  (Streamlit browser UI or CLI)
services:   all downstream services via the modules above
```
---

## Purpose
Provide a user-facing interface to query the knowledge repository. Accepts natural language questions, orchestrates retrieval and answering, and displays grounded, cited responses.

---

## Responsibilities
- Accept user questions via CLI or web UI (Streamlit)
- Call `retriever.retrieve()` and `answerer.answer()` in sequence
- Render the answer with formatted citations and source links
- Display retrieval metadata (scores, article dates) for transparency
- Support optional metadata filters (by tag or date) at query time
- Provide a simple command to trigger a manual ingestion run

---

## Interface Options

### Option A: Streamlit web UI (recommended for daily use)
```bash
streamlit run app.py
```
Renders at `http://localhost:8501`. Provides a text input, answer display, and collapsible source cards.

### Option B: CLI (for scripting and testing)
```bash
python app.py "What does she say about compounding knowledge?"
python app.py "What are her views on writing?" --tags essay,craft
python app.py "Her recent thoughts on AI" --date-after 2025-01-01
python app.py --sync-corpus     # load new/changed markdown from MD_CORPUS_DIR
python app.py --stats           # show DB stats (article count, last run, etc.)
```

---

## Query Flow

```
User question
     │
     ▼
retriever.retrieve(query, top_k, filters)
     │  returns List[SearchResult]
     ▼
answerer.answer(query, results)
     │  returns Answer
     ▼
Display:
  - Answer.response (with [N] citations highlighted)
  - Source cards: [N] Title | Date | URL | Score
```

---

## Streamlit UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  🔍  Knowledge Repository — [AUTHOR_NAME]               │
├─────────────────────────────────────────────────────────┤
│  Ask a question:                                        │
│  [_______________________________________________] [Ask] │
│                                                         │
│  Filters (optional):  Tags: [_______]  Date after: [__] │
├─────────────────────────────────────────────────────────┤
│  Answer                                                 │
│  ──────                                                 │
│  [Response text with inline [1] [2] citations...]       │
│                                                         │
│  Sources                                                │
│  ───────                                                │
│  [1] Article Title — 12 Jan 2025 — score: 0.82  [↗]   │
│  [2] Another Article — 3 Mar 2024 — score: 0.71  [↗]   │
│                                                         │
│  ▸ Show retrieved excerpts (debug)                      │
├─────────────────────────────────────────────────────────┤
│  Sidebar: Stats │ Run history │ Trigger ingestion       │
└─────────────────────────────────────────────────────────┘
```

### Key UI behaviours
- Spinner shown while retrieving and generating
- Citation `[N]` markers in response text are rendered as superscripts
- Source cards link directly to the original article (opens in new tab)
- A source whose `url` starts with `local:` has no web page behind it — the article came
  from the markdown corpus without a `url` in its frontmatter. Render it as plain text with
  the article title and no link, never as a dead hyperlink. Its presence is a prompt to fix
  that file's frontmatter.
- "Show retrieved excerpts" expander reveals raw chunk text + `chunk_id` + `content_type` +
  scores (debug mode); `content_type` also labels each source card ("from a chart", "from a
  table") using the value on `SearchResult`
- Sidebar shows: total articles indexed, last ingestion run date and stats
- Sidebar has two ingestion buttons, not one:
  - **Sync corpus** → `run_corpus_sync()` — safe to press any time; reports new / updated /
    skipped counts. Needs no credentials.
  - **Check email** → `run_email_triggered()` — needs a mailbox and a live site session.
    Disable it with an explanatory tooltip when `LOGIN_URL` or `TRUSTED_SENDER` is unset,
    so a corpus-only installation does not surface a button that can only fail.
    This run may need **you** to sign in: if the saved session has expired, a browser
    window opens and the run waits for you. Set `INTERACTIVE_LOGIN=always` in `.env` for
    a desktop install — the ingestion runs in a background thread with no tty, so the
    `auto` default would decide no human is present and fail the run instead of asking.
    While a run is waiting for a sign-in, show that state in the sidebar ("Waiting for
    you to sign in — see the browser window") rather than a bare spinner.
- Both run in a background thread and are disabled while a run is in flight
  (`metadata_db.has_open_run()`)
- Sidebar stats break the article count down by `source` (corpus / web)

---

## CLI Argument Reference
```
positional:
  query                   Natural language question (quoted)

optional:
  --top-k N               Number of chunks to retrieve (default: 6)
  --tags TAG1,TAG2        Filter results to these tags
  --date-after YYYY-MM-DD Filter results published after this date
  --date-before YYYY-MM-DD Filter results published before this date
  --no-citations          Print answer without source list
  --json                  Output Answer as JSON (for piping); datetimes serialised as
                          ISO 8601 strings via a custom default= encoder
  --sync-corpus           Run the markdown corpus sync and exit (Phase 1)
  --check-email           Run one email-triggered ingestion and exit (Phase 2)
  --dry-run               Modifier for either ingestion flag: report, write nothing
  --stats                 Print repository statistics and exit

App exposes the everyday ingestion flags only. Staged-load and recovery operations
(--only, --limit, --force, --inspect, --reset) live on scheduler/monthly_job.py and are
deliberately not mirrored here: --reset in particular should not be one typo away from a
query the user runs daily.
```

---

## Configuration Constants
```python
AUTHOR_NAME        = "..."          # displayed in UI header and system prompt
DEFAULT_TOP_K      = 6
STREAMLIT_PORT     = 8501
SHOW_SCORES        = True           # show retrieval scores in source cards
```

---

## Error Handling

| Scenario | User-facing message |
|---|---|
| Empty query submitted | "Please enter a question." |
| No results found | "I couldn't find relevant content for this question. Try rephrasing or broadening your search." |
| Vector store unavailable | "The knowledge base is unavailable. Check that Qdrant is running." |
| Claude API error | "Could not generate an answer right now. Please try again in a moment." |
| Empty knowledge base (0 articles) | "The knowledge base is empty. Load your markdown corpus first: `python app.py --sync-corpus`" — name the corpus path so a wrong `MD_CORPUS_DIR` is obvious |

---

## Key Dependencies
- `streamlit` — web UI
- `argparse` — CLI parsing (stdlib)
- `threading` — background ingestion trigger in Streamlit
- `query/retriever.py` and `query/answerer.py`
- `scheduler/monthly_job.py` — `run_corpus_sync` / `run_email_triggered` for the CLI flags and sidebar buttons

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.app"
```

| Event | Level | Extra fields |
|---|---|---|
| App started | INFO | `interface` (`"streamlit"` or `"cli"`) |
| Query received | INFO | `query_length_chars`, `top_k`, `filters` |
| Retrieval complete | INFO | `query`, `result_count`, `top_score` |
| Answer generated | INFO | `query`, `source_count`, `input_tokens`, `output_tokens` |
| Empty query submitted | WARNING | `interface` |
| Knowledge base empty (0 articles) | WARNING | `interface` |
| Vector store unavailable | CRITICAL | `error_type` |
| Claude API unavailable | ERROR | `error_type` |
| Manual ingestion triggered from UI | INFO | `trigger_source` (`"streamlit_sidebar"`) |
| Ingestion thread started | INFO | — |
| Stats page rendered | DEBUG | `article_count`, `last_run_id`, `last_run_ts` |

---

## Testing Notes
- CLI: assert `--stats` prints article count without crashing
- CLI: assert `--json` output is valid JSON parseable to `Answer`
- CLI: assert unknown `--tags` returns empty answer gracefully (not an error)
- Streamlit: use `streamlit.testing` or mock `st.*` calls for unit tests
- Integration: end-to-end test with a seeded in-memory Qdrant — assert a known question returns the correct article as the top source
