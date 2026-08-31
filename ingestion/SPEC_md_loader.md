# `ingestion/md_loader.py` — Markdown Corpus Loader

---
```
module:     ingestion/md_loader.py
spec:       ingestion/SPEC_md_loader.md
layer:      Ingestion (Phase 1 source)
depends_on: config.py · logger.py · models.py
used_by:    scheduler/monthly_job.py  (run_corpus_sync)
services:   none — local filesystem only
files:      MD_CORPUS_DIR/*.md                (article markdown, read-only)
            MD_CORPUS_DIR/images/<article>/*  (per-article images, read-only)
```
---

## Purpose
Turn a local folder of markdown articles plus their images into `Article` objects — the
same dataclass `ingestion/extractor.py` produces from scraped HTML. This is the **Phase 1**
content source: the back-catalogue arrives as files on disk rather than as URLs to scrape.

`md_loader.py` is the direct counterpart of `extractor.py`. They are the only two modules
that produce an `Article`; everything downstream (`image_transcriber` → `chunker` →
`embedder` → storage) is identical for both and knows nothing about where the article
came from.

```
Phase 1  MD_CORPUS_DIR/*.md  →  md_loader.load_article()   ─┐
                                                            ├─→  Article  →  image_transcriber → …
Phase 2  scraped HTML        →  extractor.extract()        ─┘
```

---

## Responsibilities

- Enumerate `*.md` files in `MD_CORPUS_DIR`
- Parse YAML frontmatter into article metadata
- Separate body prose, GFM tables, and image references
- Resolve every image reference, **by the whole relative path written in the markdown**,
  to an absolute path under `MD_CORPUS_DIR/images/<article>/`
- Compute a deterministic `content_hash` for change detection
- Reject stubs (below `MIN_WORD_COUNT`)
- **Not** responsible for: vision transcription (`image_transcriber.py`), chunking
  (`chunker.py`), or deciding what to re-ingest (`scheduler/monthly_job.py`)

---

## Corpus Layout

One markdown file per article at the top level, and **one image subfolder per article**
under `images/`:

```
MD_CORPUS_DIR/                                   # e.g. …/corpus/example-author — one author
├── 2021-05-16-premium-2021-5-16.md
├── 2021-06-27-premium-2021-6-27.md
└── images/
    ├── 2021-05-16-premium-2021-5-16/
    │   ├── 00-4c91be07a2.jpg
    │   └── 01-a1b2c3d4e5.png
    └── 2021-06-27-premium-2021-6-27/
        ├── 00-558cf4ffe8.jpg
        └── 01-9f8e7d6c5b.png
```

The subfolder is named `<published-date>-<url-slug>` and by convention matches the
markdown stem — but **nothing in the loader may depend on that**. Images are resolved
from the path the markdown actually wrote
(`images/2021-06-27-premium-2021-6-27/00-558cf4ffe8.jpg`), never by reconstructing a
folder name from the article filename. The convention is the export tool's, and the
export tool is not part of this repo.

Image filenames are unique **only within their own subfolder**. The ordinal prefix
repeats in every article — every article has a `00-*`, most have a `01-*` — so a
bare-basename lookup across `images/` is ambiguous by construction and will silently
attach one article's chart to another. This is the single rule to preserve here: resolve
the whole relative path, never just the basename.

Only `*.md` files at the top level of `MD_CORPUS_DIR` are treated as articles. Neither
`images/` nor anything beneath it is scanned for articles — the per-article subfolders
carry article-shaped names, which makes a naive recursive walk look plausible and wrong.
Any other subfolder is ignored with a DEBUG log line.

---

## Frontmatter Contract

An unedited article as the export tool writes it:

```markdown
---
title: "Multi-Stock and Sector Update"
subtitle: "June 27, 2021 This report focuses on macro updates, the recent Fed update, …"
author: ""
published: "2021-06-27"
url: "https://www.example.com/premium-2021-6-27"
scraped_at: "2026-08-30T12:18:03+00:00"
cover: "images/2021-06-27-premium-2021-6-27/00-558cf4ffe8.jpg"
truncated: false
word_count: 6951
images_saved: 37
---
# Multi-Stock and Sector Update
![cover](images/2021-06-27-premium-2021-6-27/00-558cf4ffe8.jpg)
*June 27, 2021*
![Premium Feature Image](images/2021-06-27-premium-2021-6-27/00-558cf4ffe8.jpg "https://www.example.com/wp-content/uploads/premium-feature.jpg")

The body of the article starts here…
```

### Keys the loader consumes

| Key | Required | Type | Fallback if absent **or empty** |
|---|---|---|---|
| `title` | recommended | str | First `# H1` in the body; else the filename stem |
| `url` | **yes** | str | Synthesised as `local:<filename-stem>`, logged WARNING |
| `published` | recommended | ISO date str | `None` |
| `author` | no | str | `AUTHOR_NAME` from config |
| `tags` | no | list[str] | `[]` |

Empty string counts as absent. The export writes `author: ""` on every file, so the
fallback must be `metadata.get("author") or AUTHOR_NAME` — an `is None` test leaves every
article in the corpus authored by the empty string, and every citation unattributed.

`url` is the **primary key** for the whole system — `metadata_db`, `vector_store`
payloads, and citations in `answerer.py` all key on it. Pass it through
`models.canonical_url()` before use: the export writes bare URLs with no trailing slash
(`https://www.example.com/premium-2021-6-27`) while a Phase 2 email link for the same
article usually carries a trailing slash and tracking query. Two spellings of one URL are
two articles in the database, and the corpus copy is the one that silently goes stale. A
corpus file without a `url` still loads (so a missing field never blocks a run), but its
citations will carry a `local:` identifier rather than a clickable link. Treat a WARNING
here as something to fix in the corpus, not in the code.

### Keys the export writes that the loader does not trust

Recorded here so a future reader knows each one is a deliberate no-op, not an oversight.

| Key | Treatment |
|---|---|
| `subtitle` | Teaser text; its opening sentences are repeated verbatim at the top of the body. **Not** indexed separately — doing so puts a near-duplicate chunk in the vector store, and duplicate chunks crowd out real ones in a top-k retrieval. |
| `scraped_at` | When the export ran. **Not** `fetched_at`, which is when *this* pipeline ingested the file. Ignored. |
| `cover` | Relative path to the hero image, repeated in the body as `![cover](…)` and again as `![Premium Feature Image](…)` — the same file, up to three references. Do not read it as a separate image; body extraction already finds it and step 4 dedups the repeats. |
| `truncated` | `true` means the export captured an incomplete body. The article still loads (a partial article beats no article), with a WARNING carrying the url. |
| `word_count` | The export's own count, taken over raw markdown including markup. `md_loader` computes its own over cleaned prose, so the two never agree exactly — only the order of magnitude is meaningful. See *Export cross-checks*. |
| `images_saved` | How many files the export wrote to disk for this article. The most useful of the four: compared against the number of images actually resolved, it catches a missing or half-synced image subfolder. See *Export cross-checks*. |

Unknown frontmatter keys are ignored, not an error.

---

## Inputs

| Input | Type | Source |
|---|---|---|
| `md_path` | `Path` | A single `.md` file under `MD_CORPUS_DIR` |
| `MD_CORPUS_DIR` | `str` | `config.py` |
| `MD_IMAGES_SUBDIR` | `str` | `config.py` (default `"images"`) |
| `MIN_WORD_COUNT` | `int` | `config.py` — shared with `extractor.py` |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `Article` | dataclass | Same shape as `extractor.extract()` output, with `source="corpus"` |
| `None` | — | Unparseable file only (bad UTF-8, malformed YAML) |

A file below `MIN_WORD_COUNT` is **not** `None`: it returns an `Article` with
`is_stub=True` and a valid `content_hash`. The distinction is load-bearing — the scheduler
counts `None` as `stats["failed"]` with an ERROR and a stub as `stats["skipped"]` with a
WARNING, and only the stub carries the hash that makes a later edit past `MIN_WORD_COUNT`
register as a change rather than as a brand-new article.

The `Article` and `ImageRef` dataclasses are defined once in `models.py` and documented in
[SPEC.md](../SPEC.md). Two fields exist specifically to let one pipeline serve both
sources:

```python
# models.py — the fields that distinguish the two sources
@dataclass
class Article:
    ...
    source:      str          # "corpus" (Phase 1) | "web" (Phase 2)
    source_path: str | None   # absolute .md path for corpus articles; None for web

@dataclass
class ImageRef:
    src:        str           # absolute URL (web) or original markdown path (corpus)
    local_path: str | None    # absolute path on disk; None for web images
    alt:        str
    caption:    str
    position:   int
    is_paywall: bool          # always False for corpus images
```

`image_transcriber.py` branches on `local_path`: when it is set, the image is read from
disk and no download is attempted.

---

## Core Logic

```
1. READ FILE
   text = md_path.read_text(encoding="utf-8")
   If decode fails → log ERROR, return None

2. SPLIT FRONTMATTER
   Use python-frontmatter (or a manual --- ... --- split + yaml.safe_load)
   post.metadata → dict; post.content → markdown body
   If the YAML is malformed → log ERROR with the filename, return None
   If there is no frontmatter block → metadata = {}, body = whole file, log WARNING

3. RESOLVE METADATA
   title        = metadata.get("title") or first_h1(body) or md_path.stem
   url          = models.canonical_url(metadata.get("url") or "") \
                  or f"local:{md_path.stem}"                       (WARNING if synthesised)
   published_at = parse_date(metadata.get("published"))            (None if absent/invalid)
   author       = metadata.get("author") or AUTHOR_NAME
   tags         = metadata.get("tags") or []

4. EXTRACT IMAGES  (before tables — image syntax can appear inside a table cell)
   article_dir = md_path.parent.resolve() # NOT the MD_CORPUS_DIR constant — see below
                                         # .resolve() on both sides, or the containment
                                         # check below compares a symlink to its target
   seen = {}                             # resolved path → position of its first reference
   For each markdown image ![alt](path "title") and each <img src=...> in body:
     If path is an http(s) URL or a data: URI:
       log WARNING {md_file, image_path} and SKIP — the corpus is local by definition
     resolved = (article_dir / path).resolve()
     If article_dir is not among resolved.parents:     # "../" escape, or an absolute path
       log WARNING {md_file, image_path} and SKIP
     If not resolved.exists():
       log WARNING {md_file, image_path} and SKIP this image (do not fail the article)
     If resolved in seen:                              # the cover, referenced 2–3 times
       log DEBUG {md_file, image_path, first_position: seen[resolved]} and SKIP
     seen[resolved] = i
     images.append(ImageRef(
       src=path, local_path=str(resolved), alt=alt,
       caption=caption_for(title, body, i), position=i,
       is_paywall=False))
   Remove the image markup from the body so alt text is not chunked as prose.

   Resolve the WHOLE relative path, never the basename. Image filenames are unique only
   within their own article subfolder — every article has a `00-*` — so a basename lookup
   across images/ resolves to whichever article the filesystem happens to return first.
   That failure is silent: the article loads, the chart is real, and it belongs to a
   different report.

   Dedup by resolved path, not by the written path. The cover image is referenced from
   the frontmatter `cover:` key, from `![cover](…)`, and again from
   `![Premium Feature Image](…)`, sometimes with different spellings of the same
   relative path. Each surviving duplicate costs one vision API call in
   image_transcriber.py and adds one redundant chunk to the vector store.

   caption_for() must reject two things the export reliably produces:
     - A title attribute holding the ORIGINAL REMOTE URL rather than a caption:
       `![Premium Feature Image](images/…/00-….jpg "https://www.example.com/…jpg")`.
       A title that parses as an http(s) URL is not a caption — discard it.
     - The nearest-following-italic-line fallback picking up a bare date. Every export
       puts `*June 27, 2021*` directly under the cover image.
   Fall through to an empty caption in both cases. An embedded URL or a bare date is
   worse than no caption: it becomes prose in the transcription context and in the chunk.

   Images resolve relative to the article's own directory, never to the MD_CORPUS_DIR
   config constant. That is what makes `--dir /other/corpus` and `--file PATH` work: with
   a constant, an override directory would silently resolve every image against the
   default corpus — finding nothing, or worse, finding a same-named image from the wrong
   corpus.

5. EXTRACT TABLES
   Find GFM pipe tables (a header row, a |---|---| separator row, then body rows).
   tables_md.append(the table block verbatim — it is already Markdown)
   Remove each table block from the body.
   Tables are kept verbatim; unlike extractor.py there is no HTML-to-Markdown conversion
   and no layout-table heuristic, because markdown tables are data tables by construction.

6. CLEAN BODY
   body_text = remaining markdown with heading markers, emphasis, and link syntax
               normalised to plain prose ([text](url) → text; keep the text, drop the URL)
   Collapse runs of 3+ blank lines to 2.
   word_count = len(body_text.split())

7. CONTENT HASH
   content_hash = models.content_hash(body_text, tables_md, images)

   Call the shared helper in models.py — do NOT reimplement the formula here. It is
   defined once, in SPEC.md, and extractor.py calls the same function; two copies of it
   drift and then the same article hashes differently depending on which producer loaded
   it. Corpus images contribute size + mtime, so replacing chart-01.png in place forces a
   re-ingest; the image cache in image_transcriber.py is keyed on the same size+mtime
   pair, so touching the bytes both re-ingests the article and misses its transcription
   cache entry.

8. STUB CHECK
   If word_count < MIN_WORD_COUNT:
     log WARNING {md_file, word_count}
     return Article(..., content_hash=content_hash, is_stub=True)
   The hash is computed first so a stub still carries one — the scheduler records it as
   skipped, and a later edit that grows the file past MIN_WORD_COUNT is then detected as
   a change rather than as a new article.

9. RETURN
   Article(url=url, title=title, author=author, published_at=published_at,
           fetched_at=datetime.now(UTC), tags=tags, body_text=body_text,
           tables_md=tables_md, images=images, word_count=word_count,
           content_hash=content_hash, is_stub=False,
           source="corpus", source_path=str(md_path))
```

---

## Export cross-checks

Three frontmatter keys are the export tool's own account of what it wrote. They are not
authoritative and never overwrite a computed value, but comparing them costs nothing and
catches the corpus problems that are otherwise invisible — a half-synced image folder
looks exactly like an article that happened to have no charts.

Run these after step 4, before the stub check:

| Check | Level | Meaning |
|---|---|---|
| `images_saved` > number of images resolved | WARNING `{md_file, images_saved, images_resolved}` | Files are missing from `images/<article>/`, or the folder has not finished syncing from cloud storage. The article still loads, minus those charts. |
| `images_saved` < number of images resolved | DEBUG | Normal: the export counts files written, the loader counts references, and the cover is referenced more than once. |
| `truncated: true` | WARNING `{md_file, url}` | The export captured a partial body. Loads anyway. |
| frontmatter `word_count` and computed `word_count` differ by more than 50% | WARNING `{md_file, declared, computed}` | Body cleaning ate more than markup — usually a body that is mostly tables, or a partly-empty export. |

None of these is fatal, and none changes what is stored. They exist so that a corpus that
degrades between runs says so in the log rather than in a bad answer six weeks later.

---

## Configuration Constants

```python
MD_CORPUS_DIR    = os.environ.get("MD_CORPUS_DIR", "corpus")  # *.md + images/ per author
MD_IMAGES_SUBDIR = "images"                                   # holds one subfolder per article
MIN_WORD_COUNT   = 100                                        # shared with extractor.py
AUTHOR_NAME      = os.environ.get("AUTHOR_NAME", "the author")  # frontmatter author fallback
```

---

## Error Handling

| Failure | Level | Behaviour |
|---|---|---|
| `MD_CORPUS_DIR` does not exist | CRITICAL | `iter_article_paths()` raises `CorpusNotFoundError`; run aborts |
| `MD_CORPUS_DIR` contains no `*.md` | CRITICAL | Raise `CorpusEmptyError`; run aborts |
| File is not valid UTF-8 | ERROR | Skip file, continue corpus |
| Malformed YAML frontmatter | ERROR | Skip file, continue corpus |
| No frontmatter block at all | WARNING | Load anyway using body-derived title and synthesised url |
| `url` missing from frontmatter | WARNING | Synthesise `local:<stem>`, continue |
| `published` unparseable | WARNING | `published_at = None`, continue |
| Referenced image not found on disk | WARNING | Skip that image, keep the article |
| Image path escapes the article folder (`../`, absolute path, http(s), `data:`) | WARNING | Skip that image, keep the article |
| Same file referenced more than once (the cover) | DEBUG | Later references dropped; one `ImageRef` per file |
| `images/` or one article's subfolder missing entirely | WARNING | All its images skipped; article still loads as text |
| `images_saved` exceeds the number of images resolved | WARNING | Article loads; corpus is incomplete on disk |
| `truncated: true` in frontmatter | WARNING | Article loads; body is known to be partial |
| Body below `MIN_WORD_COUNT` | WARNING | Return `is_stub=True` |

A single bad file never aborts a corpus run — the scheduler counts it in `stats["failed"]`
and continues. Only a missing or empty corpus directory is fatal.

---

## Key Dependencies

```
python-frontmatter>=1.1   # YAML frontmatter parsing
PyYAML>=6.0               # transitively required by python-frontmatter
python-dateutil>=2.9      # tolerant date parsing (shared with extractor.py)
```

No network, no browser, no LLM. This module is pure filesystem and string work, which
makes it the cheapest part of the pipeline to test.

---

## Public Interface

```python
from pathlib import Path
from models import Article

class CorpusNotFoundError(Exception): ...
class CorpusEmptyError(Exception): ...

def iter_article_paths(corpus_dir: str | None = None) -> list[Path]:
    """Return sorted *.md paths at the top level of the corpus directory.

    Raises CorpusNotFoundError if the directory is missing,
    CorpusEmptyError if it contains no .md files.
    """

def load_article(md_path: Path) -> Article | None:
    """Parse one markdown file into an Article.

    Images resolve against md_path.parent joined with the relative path written in
    the markdown, so this works for any corpus directory without reading the
    MD_CORPUS_DIR constant, and cannot cross-link two articles' images.
    Returns None if the file cannot be parsed. Returns an Article with
    is_stub=True (and a valid content_hash) if the body is below MIN_WORD_COUNT.
    Never raises for per-file problems — logs and returns None.
    """

def load_corpus(corpus_dir: str | None = None) -> list[Article]:
    """Convenience wrapper: iter_article_paths + load_article, dropping Nones."""
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.ingestion.md_loader"
```

| Event | Level | Extra fields |
|---|---|---|
| Corpus scan started | INFO | `corpus_dir` |
| Corpus scan complete | INFO | `corpus_dir`, `file_count` |
| Corpus directory missing | CRITICAL | `corpus_dir`, `exc_info=True` |
| Corpus directory empty | CRITICAL | `corpus_dir` |
| Non-markdown subfolder ignored | DEBUG | `path` |
| Article loaded | DEBUG | `md_file`, `url`, `word_count`, `table_count`, `image_count` |
| No frontmatter block | WARNING | `md_file` |
| Malformed frontmatter | ERROR | `md_file`, `error_type`, `exc_info=True` |
| `url` missing — synthesised | WARNING | `md_file`, `synthesised_url` |
| `published` unparseable | WARNING | `md_file`, `raw_value` |
| Image file not found | WARNING | `md_file`, `image_path` |
| Image path outside the article folder | WARNING | `md_file`, `image_path` |
| Duplicate image reference dropped | DEBUG | `md_file`, `image_path`, `first_position` |
| Fewer images resolved than `images_saved` | WARNING | `md_file`, `images_saved`, `images_resolved` |
| Article marked `truncated` by the export | WARNING | `md_file`, `url` |
| Declared and computed `word_count` diverge | WARNING | `md_file`, `declared`, `computed` |
| Article is a stub | WARNING | `md_file`, `word_count`, `min_required` |
| File unreadable | ERROR | `md_file`, `error_type`, `exc_info=True` |

---

## Testing Notes

Fixture corpus under `tests/fixtures/corpus/` covering: full frontmatter as the export
writes it (empty `author`, `subtitle`, `cover`, `truncated`, `word_count`,
`images_saved`), missing `url`, a `url` with a trailing slash, missing frontmatter
entirely, malformed YAML, a GFM table, an image whose file exists, an image whose file is
missing, a cover referenced three times, and a 20-word stub.

Crucially, the fixture needs **two** articles whose image subfolders both contain a file
named `00-*.jpg`, with different bytes. That pair is what makes the basename-resolution
regression fail a test instead of shipping.

- Assert `title` falls back to H1, then to filename stem
- Assert a missing `url` produces `local:<stem>` and logs exactly one WARNING
- Assert `tables_md` contains the table verbatim and `body_text` does not
- Assert image markup is removed from `body_text` and each resolved `ImageRef.local_path`
  points at an existing file
- Assert an image reference with no matching file is dropped and the article still loads
- Assert each image resolves inside **its own** article subfolder: with two fixture
  articles that both contain `00-cover.jpg`, assert neither article's `local_path` points
  into the other's folder (the basename-resolution regression)
- Assert a path with a `../` prefix, an absolute path, an `http(s)` URL and a `data:` URI
  are each skipped with a WARNING and do not abort the article
- Assert the cover referenced from `cover:`, `![cover](…)` and `![Premium Feature
  Image](…)` yields exactly one `ImageRef`, at the position of its first body reference
- Assert a title attribute holding an `https://…` URL does not become the caption
- Assert an italic line containing only a date does not become the caption
- Assert `author: ""` falls back to `AUTHOR_NAME`, not to the empty string
- Assert `url` with and without a trailing slash produce the same `content_hash` key —
  i.e. that both go through `models.canonical_url()`
- Assert `images_saved: 5` with three resolvable images logs exactly one WARNING and
  still returns the article
- Assert `is_paywall is False` and `local_path is not None` for every corpus image
- Assert `content_hash` is stable across a `.md` file rename and changes when body text changes
- Assert `content_hash` changes when a referenced image file's bytes change under the same
  filename (this is the regression that silently stales charts)
- Assert a stub `Article` still carries a non-empty `content_hash`
- Assert images resolve against the article's own directory: load the same fixture from a
  copied corpus at a different path and assert every `local_path` points inside that copy
- Assert a 20-word file returns `is_stub=True`, not `None`
- Assert malformed YAML returns `None` and does not raise
- Assert `CorpusNotFoundError` / `CorpusEmptyError` on a missing and an empty directory
- **Parity test**: load one fixture as markdown and the equivalent as HTML through
  `extractor.extract()`; assert both produce an `Article` with the same field set, so the
  downstream pipeline cannot tell them apart
