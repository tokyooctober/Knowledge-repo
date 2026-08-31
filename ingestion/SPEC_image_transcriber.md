# `ingestion/image_transcriber.py` — Image Transcriber

---
```
module:     ingestion/image_transcriber.py
spec:       ingestion/SPEC_image_transcriber.md
layer:      Ingestion — vision
depends_on: config.py · logger.py · llm_provider.py (VisionProvider)
            scraper/login.py (BrowserContext — Phase 2 only, optional)
            models.py  (Article, ImageRef → ImageTranscription)
used_by:    scheduler/monthly_job.py
input:      Article.images  (list[ImageRef] from ingestion/md_loader.py  ← Phase 1
                                             or ingestion/extractor.py  ← Phase 2)
output:     list[ImageTranscription]  →  passed to ingestion/chunker.py
services:   vision LLM  (via llm_provider.py)
            target website  (Phase 2 only — authenticated image download via cookies)
files:      MD_CORPUS_DIR/images/<article>/*  (Phase 1 source images, read-only)
            images/{hash}.png        (normalised disk cache)
            data/image_cache.db      (source key → file_path SQLite index)
```
---

## Purpose
Download article images through an authenticated session and transcribe their content using Claude's vision API. Produces structured text descriptions of charts, graphs, and image-based tables that can be embedded and searched alongside article body text. This component bridges the gap between visual financial data and the text-based RAG pipeline.

---

## Responsibilities
- Download each paywall-protected image using authenticated session cookies from Playwright
- Download non-paywall images (external CDN) directly via `httpx`
- Detect image type: chart, data table, diagram, photo/illustration, or unknown
- Call the Claude vision API with a targeted transcription prompt
- Return a list of `ImageTranscription` objects attached to the article
- Cache transcriptions by image content hash to avoid redundant API calls
- Log all download attempts, API calls, costs (token counts), and failures

---

## Why vision transcription matters for this use case
The author's premium reports are investment research. Key data lives in charts:
- Time-series charts (M2 money supply, Bitcoin price, interest rates)
- Ratio charts (gold/silver, equity valuations)
- Annotated bar/scatter charts with labelled data points
- Data tables rendered as images (yield curves, performance tables)

Without transcription, a question like *"What did the chart show about M2 growth in 2022?"* returns nothing — the data exists only as pixels. With transcription, the chart's axis labels, values, trend description, and key annotations become searchable text.

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `article` | `Article` from extractor | Contains `article.images` list of `ImageRef` objects |
| `browser_context` | `login.py` | Playwright context, used for paywall image download. **Optional — `None` in Phase 1.** Passing `None` when an image has no `local_path` is an error for that image, not for the run. |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `List[ImageTranscription]` | Python list | **One entry per image in `article.images`, in the same order.** Never shorter. |

Skipped images are represented, not omitted: they appear with `skipped=True` and a
`skip_reason`, and `chunker.py` filters them (`if t.skipped: continue`). Returning a
filtered list instead would make the caller unable to tell "image 3 was a photo" from
"image 3 failed", and would make it impossible to report per-article image coverage.

The one exception is an image that could not be resolved at all (local file gone, download
failed): those also return an entry, with `skipped=True` and `skip_reason="unavailable"`.

Because `chunker.py` assigns `chunk_index` by enumerating only the *non-skipped*
transcriptions, image chunk indices are contiguous regardless of how many were skipped —
and a `chunk_id` therefore changes if an earlier image's skip status changes. That is
acceptable only because `delete_by_url` clears all of an article's vectors before re-upsert;
do not add an incremental-update path that assumes chunk_ids are stable across re-ingests.

### `ImageTranscription` dataclass
```python
@dataclass
class ImageTranscription:
    image_ref:        ImageRef      # the source ImageRef from the article
    file_path:        str           # path to cached image on disk: images/{hash}.{ext}
    file_hash:        str           # SHA-256 of image bytes (used for cache key)
    image_type:       str           # "chart" | "table" | "diagram" | "photo" | "unknown"
    transcription:    str           # full text output from Claude vision
    model:            str           # e.g. "claude-sonnet-4-20250514"
    input_tokens:     int
    output_tokens:    int
    skipped:          bool          # True if image was skipped (too small, unsupported type, etc.)
    skip_reason:      str | None    # reason string if skipped
```

---

## Image Source Resolution

Every `ImageRef` carries a `local_path`. It is the only thing this module branches on —
never `article.source`.

```
for ref in article.images:
    if ref.local_path is not None:        # Phase 1 — markdown corpus
        image_bytes = Path(ref.local_path).read_bytes()
        # no network, no cookies, no browser_context needed
    else:                                 # Phase 2 — scraped web page
        if browser_context is None and ref.is_paywall:
            log.error("Paywall image with no browser context — skipping",
                      extra={url: ref.src})
            emit(ref, skipped=True, skip_reason="no_browser_context")
            continue    # continue the LOOP, never the output list — one entry per image
        image_bytes = await download(ref, browser_context)
```

Local files skip the entire download path: no `httpx`, no cookies, no timeout, no retry.
They still go through the same validation, cache, classification, and transcription steps
below, so a chart is transcribed identically whether it arrived from disk or from the
site.

If `Path(ref.local_path)` does not exist at transcription time, log an ERROR with
`md_file` and `local_path` and emit an entry with `skipped=True`,
`skip_reason="unavailable"` — the article is still ingested with its remaining content. (`md_loader.py` already drops references to missing files at load time,
so this only fires if a file disappeared between load and transcription.)

---

## Image Download Strategy *(Phase 2 only)*

### Paywall images (`ImageRef.is_paywall == True`)
These are served on `SITE_DOMAIN` (e.g. `example.com`) and require authentication. Two options:

**Option A: Cookie injection with `httpx` (preferred)**
```python
cookies = {c["name"]: c["value"]
           for c in await browser_context.cookies()}
async with httpx.AsyncClient(cookies=cookies, follow_redirects=True) as client:
    response = await client.get(image_url)
    assert response.status_code == 200
    image_bytes = response.content
```
This is fast, lightweight, and avoids spinning up a browser tab per image.

**Option B: Playwright page download (fallback)**
```python
page = await browser_context.new_page()
response = await page.goto(image_url)
image_bytes = await response.body()
await page.close()
```
Used if cookie injection fails (some sites use JS-gated image delivery).

### Non-paywall images (`ImageRef.is_paywall == False`)
Plain `httpx.AsyncClient().get(url)` — no auth needed.

Corpus images (`local_path` set) never reach this section. `md_loader` has already
resolved each one to a file inside that article's own `images/<article>/` subfolder and
deduped repeated references, so this module never resolves a path or guards a duplicate. `is_paywall` is always `False`
for them, but they are resolved from disk before the paywall branch is evaluated.

### Download validation
After downloading, check:
- `Content-Type` header starts with `image/`
- File size > `MIN_IMAGE_BYTES` (default: 5000 bytes — skips tracking pixels and icons)
- Image dimensions > `MIN_IMAGE_PIXELS` × `MIN_IMAGE_PIXELS` (default: 100×100 px, checked with `Pillow`)

### `media_type` for the vision call
`media_type` is derived from the **normalised** cached file, not the source: `Pillow`
opens the bytes, and the disk cache is written as `image/png` unless the source is already
JPEG (kept as `image/jpeg`, smaller). The value passed to
`provider.complete_with_image(..., media_type=...)` is therefore always `"image/png"` or
`"image/jpeg"` — never `"image/webp"` or a guessed type from a `Content-Type` header that
may be wrong. A format Pillow cannot open is `skip_reason="invalid_image"`.

---

## Disk Cache

```
images/
  {sha256_of_image_bytes}.png    ← normalised to PNG by Pillow
  {sha256_of_image_bytes}.jpg    ← kept as JPEG if source is JPEG (smaller)
```

The cache is keyed by **source key**, because the image bytes hash is not known until the
image has been fetched:

| Source | `source_key` |
|---|---|
| Web (`local_path is None`) | the absolute image URL |
| Corpus (`local_path` set) | `f"file:{local_path}:{mtime_ns}:{size}"` |

```sql
CREATE TABLE IF NOT EXISTS image_cache (
    source_key   TEXT PRIMARY KEY,   -- image URL, or file:path:mtime:size
    file_path    TEXT NOT NULL,      -- normalised copy under images/
    file_hash    TEXT NOT NULL,      -- sha256 of image bytes
    image_type   TEXT,               -- cached classification
    transcription TEXT,              -- cached transcription text
    model        TEXT,
    cached_at    TEXT NOT NULL
);
```

A hit on `source_key` skips the fetch. A hit that also has a non-null `transcription`
recorded against the **current** `VISION_MODEL` skips the vision call entirely — this is
what makes a corpus re-sync cheap: editing one article's prose does not re-pay for
transcribing its twelve unchanged charts.

Including `mtime_ns` and `size` in the corpus key means replacing an image file in the
corpus invalidates its cache entry automatically, while a re-sync that touches nothing
costs zero vision tokens.

> **Migration note.** The previous schema keyed on `url TEXT PRIMARY KEY` and cached only
> the file, not the transcription. There is no upgrade path written for it; delete
> `data/image_cache.db` and let it rebuild. The normalised files under `images/` can be
> kept — they are re-registered on first use.

### The cache is not part of the index

`data/image_cache.db` and the normalised files under `images/` survive
`monthly_job.py --reset`, deliberately. The cache is keyed on image content, not on run or
article identity, so no entry is invalidated by emptying the index — and vision calls are
the dominant cost of a full corpus load. Clearing the cache as part of "starting clean"
re-pays that entire bill to produce, by construction, byte-identical transcriptions.

This is worth stating twice (it is also in
[SPEC_monthly_job.md](../scheduler/SPEC_monthly_job.md)) because the mistake is silent:
nothing errors, the run just costs money and takes hours. The only correct reason to
delete this database is the schema migration noted above.

---

## Vision Transcription

### Model selection
All vision API calls are made through `llm_provider.get_vision_provider()`. The transcriber contains no SDK imports. Set `VISION_BACKEND` and `VISION_MODEL` in `config.py` to select any supported provider. Refer to `SPEC_llm_provider.md` for the full model reference table.

**Recommended vision models by capability:**

| Model | Best for | Notes |
|---|---|---|
| `claude-sonnet-4-20250514` (Anthropic) | Highest accuracy on dense charts with small text | Cloud; per-token cost |
| `llama3.2-vision:90b` (Ollama/vLLM) | Strong chart reading; free locally | GPU required (90B); ≥48 GB VRAM |
| `qwen2.5vl:72b` (Ollama/vLLM) | Excellent table extraction; strong on data | GPU required; ≥40 GB VRAM |
| `llama3.2-vision:11b` (Ollama) | Moderate quality; runs on 16 GB VRAM | Good for iteration; lower cost |
| `minicpm-v` (Ollama) | Lightweight; acceptable for simple charts | Runs on 8 GB VRAM |

> **Hardware note**: running a 70–90B vision model locally requires a GPU with substantial VRAM. For a pure CPU/low-resource setup, use the Anthropic or Together.ai cloud backend for vision while keeping embeddings local.

### Type detection call
```python
provider = llm_provider.get_vision_provider()
type_response = provider.complete_with_image(
    image_bytes=image_bytes,
    media_type=media_type,
    text_prompt=TYPE_DETECTION_PROMPT,
    max_tokens=10,
)
image_type = type_response.content.strip().lower()
```

### Transcription call
```python
transcription_response = provider.complete_with_image(
    image_bytes=image_bytes,
    media_type=media_type,
    text_prompt=build_transcription_prompt(article, image_ref, image_type),
    max_tokens=VISION_MAX_TOKENS,
)
```

### Transcription prompt (for chart / table / diagram)
```
You are transcribing visual content from a premium financial research report for a searchable text database.

This image is a {image_type}. The article title is: "{article.title}".
{If alt text is non-empty: 'The image alt text is: "{image_ref.alt}".' }
{If caption is non-empty: 'The image caption is: "{image_ref.caption}".' }

Your task:
1. State the chart/table type (e.g. "Line chart", "Bar chart", "Data table").
2. State the title or heading visible in the image (or infer from context).
3. State the X-axis label and range (for charts), or column headers (for tables).
4. State the Y-axis label, units, and range (for charts), or row labels (for tables).
5. Describe the key data: the overall trend, notable peaks/troughs, and specific values
   at key points (start, end, major inflections). For tables, read out all cell values.
6. State the main insight or takeaway a reader would draw from this visual.

Be precise about numbers. If a value is hard to read exactly, give a reasonable estimate
and note the uncertainty (e.g. "approximately 4.2 trillion, ±0.1T").
Do not describe colours or visual style — focus entirely on the data and insight.
Limit your response to 300 words.
```

### Output format in the corpus
Each transcription is stored with a clear prefix so it is distinguishable during chunking and retrieval:

```
[Chart 1: "US M2 Money Supply 2010–2024"]
Line chart. X-axis: years 2010–2024. Y-axis: trillions of USD (0–22T).
Trend: money supply grew steadily at ~5% annually 2010–2019, then surged 40% in 2020–2021
(peak ~21.7T in March 2022), followed by the first YoY decline in modern history (-4.4%)
through mid-2023, before stabilising near 20.8T by end 2023.
Key values: 2010: ~8.6T, 2019: ~15.3T, 2022 peak: 21.7T, 2023 trough: ~20.4T.
Insight: The 2020–21 stimulus-driven expansion and subsequent 2022–23 contraction is the
central thesis supporting the report's inflation and liquidity analysis.
```

---

## Token Cost Estimate

Per image, approximate Claude Sonnet usage:
- Type detection: ~50 input tokens (image) + ~5 output tokens
- Transcription: ~1500 input tokens (image + prompt) + ~200 output tokens

For a report with 8 charts/tables:
- ~12,400 input tokens + ~1,640 output tokens ≈ $0.04–0.08 per report

Cache hit rate is high: the same chart (e.g. a recurring Bitcoin price chart) will be downloaded and transcribed only on first appearance; subsequent articles referencing the same image URL hit the cache at zero cost.

---

## Configuration Constants

```python
# Vision model and backend are configured in config.py via VISION_BACKEND / VISION_MODEL.
# image_transcriber.py calls llm_provider.get_vision_provider() — no model names here.
IMAGE_CACHE_DIR        = "images/"
IMAGE_CACHE_DB         = "data/image_cache.db"
VISION_MAX_TOKENS      = 400
MIN_IMAGE_BYTES        = 5_000
MIN_IMAGE_PIXELS       = 100
TRANSCRIBE_TYPES       = {"chart", "table", "diagram"}
MAX_IMAGE_MB           = 5
DOWNLOAD_TIMEOUT_S     = 30
MAX_IMAGES_PER_ARTICLE = 20
```

---

## Error Handling

| Scenario | Behaviour | Log level |
|---|---|---|
| Image URL returns non-200 | Skip image; `skipped=True`, `skip_reason="download_failed"` | WARNING |
| Image too small (bytes or pixels) | Skip; `skip_reason="too_small"` | DEBUG |
| Image too large (> MAX_IMAGE_MB) | Skip; `skip_reason="too_large"` | WARNING |
| Content-Type not image/* | Skip; `skip_reason="not_image"` | WARNING |
| Cookie injection → 401 on paywall image | Retry with Playwright page download | WARNING |
| Playwright download also fails | Skip; `skip_reason="auth_failed"` | ERROR |
| Vision API: type detection returns unexpected value | Default to `"unknown"`, skip transcription | WARNING |
| Vision API: transcription rate limit | Retry with exponential backoff (max 3) | WARNING |
| Vision API: transcription fails after retries | Skip; `skipped=True`, `skip_reason="api_error"` | ERROR |
| Pillow cannot open image bytes | Skip; `skip_reason="invalid_image"` | WARNING |
| `MAX_IMAGES_PER_ARTICLE` reached | Log remaining count; stop *transcribing*, but still emit an entry per remaining image with `skipped=True`, `skip_reason="over_cap"` | WARNING |

---

## Key Dependencies
- `llm_provider.py` — provides `VisionProvider` (no direct SDK imports in image_transcriber.py)
- `httpx` — async HTTP download
- `Pillow` — image validation and format normalisation
- `sqlite3` — image URL → file path cache (stdlib)
- `hashlib` — image content hash (stdlib)

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.ingestion.image_transcriber"
```

| Event | Level | Extra fields |
|---|---|---|
| Starting image transcription for article | INFO | `url`, `image_count`, `vision_backend`, `vision_model` |
| Cache hit — skipping download | DEBUG | `image_url`, `file_path` |
| Downloading paywall image via cookies | DEBUG | `image_url`, `position` |
| Downloading image (no auth) | DEBUG | `image_url`, `position` |
| Download failed | WARNING | `image_url`, `status_code`, `error_type` |
| Cookie injection failed — retrying via Playwright | WARNING | `image_url`, `error_type` |
| Playwright download also failed | ERROR | `image_url`, `error_type` |
| Image too small — skipped | DEBUG | `image_url`, `size_bytes`, `width`, `height` |
| Image too large — skipped | WARNING | `image_url`, `size_mb` |
| Image saved to disk | DEBUG | `file_path`, `file_hash`, `size_bytes` |
| Type detection call started | DEBUG | `image_url`, `vision_backend`, `vision_model` |
| Image type detected | DEBUG | `image_url`, `image_type` |
| Image skipped (photo/unknown type) | DEBUG | `image_url`, `image_type` |
| Transcription call started | DEBUG | `image_url`, `image_type`, `vision_backend`, `vision_model` |
| Transcription complete | INFO | `image_url`, `image_type`, `input_tokens`, `output_tokens` |
| Vision API rate limit — retrying | WARNING | `image_url`, `attempt`, `retry_after_s` |
| Vision API failed after retries | ERROR | `image_url`, `attempts`, `error_type` |
| MAX_IMAGES_PER_ARTICLE reached | WARNING | `url`, `processed`, `remaining` |
| Article transcription complete | INFO | `url`, `transcribed`, `skipped`, `total_input_tokens`, `total_output_tokens` |

---

## Public Interface
```python
def count_uncached(images: list[ImageRef]) -> int:
    """How many of these images would need a vision call right now.

    Pure cache lookup: builds each source_key and counts the misses, plus the
    hits whose transcription was recorded against a different VISION_MODEL.
    Touches no network and no vision API.

    Used by monthly_job.py --corpus --dry-run to project the cost of a load
    before committing to it, which is the only point in the process where that
    number is both accurate and free.
    """

async def transcribe_images(
    article: Article,
    browser_context: "BrowserContext | None" = None,
) -> list[ImageTranscription]:
    """Resolve and transcribe all images in an article.

    Reads images from disk when ImageRef.local_path is set (Phase 1, corpus).
    Downloads paywall images via authenticated session cookies (Phase 2).
    Downloads non-paywall images directly (Phase 2).
    browser_context may be None; required only for images with local_path=None.
    Checks disk cache before downloading; caches new downloads.
    Classifies each image and transcribes charts, tables, and diagrams.
    Skips photos, unknown types, and images that fail validation.
    Returns exactly one ImageTranscription per entry in article.images, in the
    same order — NEVER a shorter list. Images that are skipped, capped, or
    unavailable come back with skipped=True and a skip_reason. chunker.py
    filters them; nothing else may assume the list was pre-filtered.
    """
```

---

## Testing Notes
- Fixture images: one line chart PNG, one data table PNG, one photo JPG, one tiny tracking pixel PNG
- Assert line chart PNG → `image_type="chart"` and `transcription` contains axis label keywords
- Assert data table PNG → `image_type="table"` and `transcription` contains numeric values
- Assert photo JPG → `skipped=True`, `skip_reason` contains "photo"
- Assert tracking pixel → `skipped=True`, `skip_reason="too_small"`
- **Local path**: an `ImageRef` with `local_path` set is transcribed with `browser_context=None`
  and makes zero HTTP calls (assert the `httpx` client is never constructed)
- **Local path missing on disk**: logs ERROR, is skipped, and the other images in the
  article are still transcribed
- **Paywall image with `browser_context=None`**: logged as ERROR and skipped, run continues
- **Cache key**: the same corpus file transcribed twice makes one vision call; touching the
  file's bytes changes `source_key` and forces a second call
- **Cache invalidation**: a cached transcription recorded against a different `VISION_MODEL`
  is not reused
- **Parity**: the same chart as a local file and as a downloaded URL produces the same
  `image_type` and an equivalent `transcription`
- Assert cache hit: calling twice with same image URL → API called only once
- Assert paywall image download uses cookies from `browser_context.cookies()`
- Assert non-paywall image download uses plain `httpx` (no cookies)
- Assert cookie injection failure triggers Playwright fallback
- Assert `MAX_IMAGES_PER_ARTICLE` stops *transcribing* after N images while still
  returning `len(article.images)` entries
- Assert `len(result) == len(article.images)` on every path: photos, tiny images, missing
  local files, paywall-without-context, and over-cap
- Mock Claude API; assert correct vision message format (base64 image block + text prompt)
- Assert `input_tokens` and `output_tokens` are populated on successful transcription
