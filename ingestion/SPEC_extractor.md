# `ingestion/extractor.py` — Content Extractor

---
```
module:     ingestion/extractor.py
spec:       ingestion/SPEC_extractor.md
layer:      Ingestion — extraction
depends_on: config.py · logger.py
            models.py  (RawPage → Article, ImageRef)
used_by:    scheduler/monthly_job.py
input:      RawPage  (from scraper/crawler.py)
output:     Article  →  passed to ingestion/chunker.py
                        and ingestion/image_transcriber.py
services:   none  (pure HTML parsing, no network calls)
```
---

## Purpose
**Phase 2 only.** Transform raw HTML from the crawler into a clean, structured `Article`
object. The Phase 1 counterpart, which builds the same `Article` from markdown files on
disk, is [ingestion/SPEC_md_loader.md](SPEC_md_loader.md).

 Extracts body text, HTML tables as Markdown, image metadata (URLs, alt text, captions) for downstream vision transcription, and article metadata. Does not perform vision transcription itself — that is the responsibility of `ingestion/image_transcriber.py`.

---

## Responsibilities
- Extract main article body text (not nav/sidebar/footer)
- Extract HTML tables and convert them to Markdown for direct indexing
- Collect all `<img>` elements in the article body: URL, alt text, caption, position
- Extract title, author, publication date, and tags
- Detect and skip non-article pages (404s, stubs)
- Normalise whitespace and encoding
- Compute a content hash for change detection

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `raw_page` | `RawPage` from crawler | `.url`, `.html`, `.fetched_at`, `.status_code` |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `Article` | dataclass | Structured article with text, tables, and image references |
| `None` | — | If page is non-article (404, redirect, stub) |

### `Article` dataclass

`Article` and `ImageRef` are defined once in `models.py` and documented in
[SPEC.md](../SPEC.md) — do not re-declare them here, and read the field list there. What
this module fixes on every object it produces:

| Field | Value from `extractor.py` |
|---|---|
| `source` | `"web"` |
| `source_path` | `None` |
| `ImageRef.local_path` | `None` (the image lives on the site, not on disk) |
| `ImageRef.is_paywall` | `True` iff `src`'s host is on `SITE_DOMAIN` |
| `content_hash` | `models.content_hash(body_text, tables_md, images)` — the shared helper; every web `ImageRef` contributes `f"{img.src}:"` and nothing touches the filesystem |
| `url` | `models.canonical_url(raw_page.url)` |

`extractor.py` sets `source="web"`, `source_path=None`, and
`local_path=None` on every object it produces; `ingestion/md_loader.py` is the counterpart
that sets `source="corpus"` and populates `local_path`. Everything downstream consumes both
without branching on the source — see *The one-Article invariant* in
[SPEC.md](../SPEC.md).

---

## Core Logic

```
1. Skip if raw_page.status_code != 200 → return None

2. PARSE HTML
   soup = BeautifulSoup(html, "lxml")
   article_el = soup.select_one(ARTICLE_BODY_SELECTOR)
   if not article_el: article_el = soup  # fallback to whole doc

3. EXTRACT HTML TABLES (before text extraction strips them)
   tables_md = []
   for table in article_el.find_all("table"):
     TRY:
       df = pd.read_html(str(table))[0]
       tables_md.append(df.to_markdown(index=False))
     EXCEPT:
       # Hand-roll as pipe-table if pandas fails (e.g. merged cells)
       tables_md.append(table_to_markdown_fallback(table))
     table.decompose()    # remove from soup so it doesn't appear in body_text

4. COLLECT IMAGE REFS (before text extraction)
   images = []
   for i, img in enumerate(article_el.find_all("img")):
     src = urljoin(article_url, img.get("src", ""))
     if not src or not src.startswith("http"): continue
     alt     = img.get("alt", "").strip()
     caption = ""
     parent  = img.find_parent(["figure", "div"])
     if parent:
       cap_el = parent.find(["figcaption", "caption"])
       if cap_el: caption = cap_el.get_text(strip=True)
     is_paywall = urlparse(src).netloc.endswith(SITE_DOMAIN)
     images.append(ImageRef(src=src, local_path=None, alt=alt, caption=caption,
                            position=i, is_paywall=is_paywall))
     # Construct by keyword, never positionally. local_path MUST be None here:
     # image_transcriber.py treats a non-None local_path as "read this off local disk",
     # so a positional slip would send it looking for a web image on the filesystem.

5. EXTRACT BODY TEXT
   PRIMARY: trafilatura.extract(html, include_images=False, include_tables=False)
     (tables and images already handled in steps 3–4)
   FALLBACK: article_el.get_text(separator="\n", strip=True)

6. CLEAN TEXT
   Collapse multiple blank lines → single blank line
   Strip leading/trailing whitespace per paragraph
   Remove zero-width and non-printable characters

7. COMPUTE METADATA
   word_count   = len(body_text.split())
   is_stub      = word_count < MIN_WORD_COUNT
   content_hash = models.content_hash(body_text, tables_md, images)
   # Shared helper — defined once in models.py (see SPEC.md), called by md_loader.py too.
   # Do not reimplement it here. Every ImageRef from this module has local_path=None, so
   # each contributes f"{img.src}:" and nothing touches the filesystem.

8. RETURN Article(url=models.canonical_url(raw_page.url), ...)
   # Same normal form md_loader.py applies to the frontmatter url. An email link with a
   # trailing slash or a utm_ query must resolve to the row the corpus already created,
   # or the article is ingested twice and only the web copy stays current.
```

---

## Table Extraction Detail

HTML tables in financial articles typically fall into two categories:

**Data tables** (structured rows/columns with headers): `pandas.read_html()` handles these correctly. Output is GitHub-Flavored Markdown table format, which chunkers and embedding models handle well.

**Layout tables** (used for non-tabular formatting): detected by checking if any `<th>` elements are present or if the table has > 1 column and > 2 rows. If neither is true, the table is skipped (likely layout-only).

Each extracted table is prefixed with its position index and caption (if any):
```
[Table 1: "Historical Bitcoin Returns by Year"]
| Year | Return |
|------|--------|
| 2020 | +302%  |
| 2021 | +60%   |
```

This prefix ensures the chunker and retriever have context for what the table contains.

---

## Image URL Resolution

Image `src` attributes may be:
- Absolute: `https://www.example.com/wp-content/uploads/chart.png` → used as-is
- Relative: `/wp-content/uploads/chart.png` → resolved with `urljoin(article.url, src)`
- Data URIs: `data:image/png;base64,...` → skipped (already embedded; no download needed, but large; log at DEBUG)
- External CDN: `https://cdn.example.com/chart.png` → `is_paywall = False`; can be fetched without auth

`is_paywall` is set to `True` if the image hostname ends with `SITE_DOMAIN`. Paywall images require the authenticated session cookies to download.

---

## Configuration Constants
```python
MIN_WORD_COUNT        = 100
ARTICLE_BODY_SELECTOR = "article, .post-content, .entry-content, main"
SITE_DOMAIN           = os.environ.get("SITE_DOMAIN", "")   # for is_paywall detection; required for Phase 2
```

---

## Error Handling

| Scenario | Behaviour | Log level |
|---|---|---|
| `status_code != 200` | Return `None` | DEBUG |
| trafilatura returns empty body | Fall back to BeautifulSoup `.get_text()` | WARNING |
| Both text extractors return empty | Return `Article` with `is_stub=True` | WARNING |
| `pandas.read_html()` fails on a table | Fall back to hand-rolled Markdown table; log table index | WARNING |
| `<img>` has no `src` attribute | Skip image; log at DEBUG | DEBUG |
| Image `src` is a data URI | Skip; log at DEBUG | DEBUG |
| `urljoin` fails on malformed src | Skip image; log at WARNING | WARNING |
| Date parsing fails | Set `published_at=None`; log at DEBUG | DEBUG |

---

## Key Dependencies
- `trafilatura` — primary body text extraction
- `beautifulsoup4` + `lxml` — HTML parsing, table/image extraction
- `pandas` — HTML table to Markdown conversion
- `tabulate` — Markdown table rendering (pandas dependency)
- `python-dateutil` — robust date parsing
- `hashlib`, `urllib.parse` — stdlib

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.ingestion.extractor"
```

| Event | Level | Extra fields |
|---|---|---|
| Skipping non-200 page | DEBUG | `url`, `status_code` |
| trafilatura extraction succeeded | DEBUG | `url`, `word_count` |
| Trafilatura returned empty — falling back | WARNING | `url` |
| BeautifulSoup fallback succeeded | INFO | `url`, `word_count` |
| Both extractors returned empty body | WARNING | `url`, `is_stub=True` |
| HTML tables found and converted | DEBUG | `url`, `table_count` |
| Table conversion failed — fallback used | WARNING | `url`, `table_index`, `error_type` |
| Image refs collected | DEBUG | `url`, `image_count`, `paywall_count` |
| Image src skipped (data URI) | DEBUG | `url`, `position` |
| Image src resolution failed | WARNING | `url`, `raw_src`, `error_type` |
| Article below MIN_WORD_COUNT (stub) | WARNING | `url`, `word_count` |
| Extraction complete | DEBUG | `url`, `title`, `word_count`, `table_count`, `image_count`, `content_hash` |

---

## Public Interface
```python
def extract(raw_page: RawPage) -> Article | None:
    """Extract structured content from a raw HTML page.

    Returns None if the page is not a valid article (404, login redirect, etc.).
    Returns Article with is_stub=True if body text is below MIN_WORD_COUNT.
    Tables are extracted to Markdown and removed from body_text.
    Images are collected as ImageRef objects for downstream vision transcription.
    Does NOT download images or transcribe visual content.
    """
```

---

## Testing Notes
- Fixture HTML files for 5+ representative articles covering: text-only, text + tables, text + charts, text + tables + charts
- Assert tables in HTML are present in `article.tables_md` and absent from `article.body_text`
- Assert layout-only tables (no `<th>`, single column) are skipped
- Assert each table is prefixed with its index and caption
- Assert `ImageRef.is_paywall` is `True` for images on SITE_DOMAIN, `False` for external CDN
- Assert relative image URLs are resolved to absolute
- Assert data URI images are skipped
- Assert `content_hash` is deterministic and changes when tables or images change
- Assert stub detection for articles below MIN_WORD_COUNT
- Assert date parsing handles ISO, US, and UK formats
