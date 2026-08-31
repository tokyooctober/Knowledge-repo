"""Resolve an article's images and transcribe the charts / tables / diagrams to searchable
text via the vision provider.

Branches on `ImageRef.local_path`, never on `article.source`:
  - set  -> read from disk (Phase 1, corpus). No network, no cookies.
  - None -> download via the Playwright browser context (Phase 2).

**Milestone 2 implements the local path only.** The Phase-2 download path is stubbed:
a web image with no `browser_context` comes back `skipped=True, "no_browser_context"`.

Returns exactly one `ImageTranscription` per entry in `article.images`, in order — never a
shorter list. `chunker.py` filters the skipped ones.
"""

from __future__ import annotations

import io
import sqlite3
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, UnidentifiedImageError

from config import (
    IMAGE_CACHE_DB,
    IMAGE_CACHE_DIR,
    MAX_IMAGE_MB,
    MAX_IMAGES_PER_ARTICLE,
    MIN_IMAGE_BYTES,
    MIN_IMAGE_PIXELS,
    TRANSCRIBE_TYPES,
    VISION_MAX_TOKENS,
    VISION_MODEL,
)
from llm_provider import get_vision_provider
from logger import get_logger
from models import ImageTranscription

if TYPE_CHECKING:
    from models import Article, ImageRef

log = get_logger(__name__)

_VALID_TYPES = {"chart", "table", "diagram", "photo", "unknown"}

TYPE_DETECTION_PROMPT = (
    "What kind of image is this? Answer with exactly one word: chart, table, diagram, "
    "photo, or unknown."
)


def _transcription_prompt(article: Article, ref: ImageRef, image_type: str) -> str:
    lines = [
        "You are transcribing visual content from a premium financial research report for "
        "a searchable text database.",
        "",
        f'This image is a {image_type}. The article title is: "{article.title}".',
    ]
    if ref.alt:
        lines.append(f'The image alt text is: "{ref.alt}".')
    if ref.caption:
        lines.append(f'The image caption is: "{ref.caption}".')
    lines += [
        "",
        "State the chart/table type, its title, the axis labels/units and ranges (or "
        "column and row headers), the key data (trend, peaks, troughs, values at start, "
        "end and major inflections; for a table, read out every cell), and the main "
        "takeaway. Be precise about numbers; note any estimate's uncertainty. Do not "
        "describe colours or visual style. Limit your response to 300 words.",
    ]
    return "\n".join(lines)


# ── cache ───────────────────────────────────────────────────────────────────

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS image_cache (
    source_key    TEXT PRIMARY KEY,
    file_path     TEXT NOT NULL,
    file_hash     TEXT NOT NULL,
    image_type    TEXT,
    transcription TEXT,
    model         TEXT,
    cached_at     TEXT NOT NULL
);
"""


def _cache() -> sqlite3.Connection:
    Path(IMAGE_CACHE_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(IMAGE_CACHE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(_CACHE_DDL)
    conn.commit()
    return conn


def _source_key(ref: ImageRef) -> str:
    if ref.local_path is None:
        return ref.src
    st = Path(ref.local_path).stat()
    return f"file:{ref.local_path}:{st.st_mtime_ns}:{st.st_size}"


def _cache_lookup(conn: sqlite3.Connection, source_key: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM image_cache WHERE source_key = ?", (source_key,)).fetchone()


def count_uncached(images: list[ImageRef]) -> int:
    """How many of these images would need a vision call right now. Pure cache lookup —
    no network, no vision API. Used by `--corpus --dry-run` to project cost."""
    conn = _cache()
    try:
        n = 0
        for ref in images:
            try:
                key = _source_key(ref)
            except OSError:
                n += 1
                continue
            row = _cache_lookup(conn, key)
            if row is None or not row["transcription"] or row["model"] != VISION_MODEL:
                n += 1
        return n
    finally:
        conn.close()


# ── main entry ──────────────────────────────────────────────────────────────


async def transcribe_images(
    article: Article,
    browser_context: object | None = None,
) -> list[ImageTranscription]:
    """One `ImageTranscription` per `article.images` entry, in order. Skipped, capped and
    unavailable images come back with `skipped=True` and a `skip_reason`."""
    provider = get_vision_provider()
    conn = _cache()
    log.info(
        "Starting image transcription for article",
        extra={
            "url": article.url,
            "image_count": len(article.images),
            "vision_model": provider.model_name,
        },
    )
    Path(IMAGE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    results: list[ImageTranscription] = []
    transcribed = 0
    try:
        for ref in article.images:
            if transcribed >= MAX_IMAGES_PER_ARTICLE:
                log.warning(
                    "MAX_IMAGES_PER_ARTICLE reached",
                    extra={"url": article.url, "processed": transcribed},
                )
                results.append(_skip(ref, "over_cap"))
                continue

            image_bytes = _resolve_bytes(ref, browser_context)
            if image_bytes is None:
                results.append(_skip(ref, _unavailable_reason(ref)))
                continue

            entry = _handle_one(conn, provider, article, ref, image_bytes)
            results.append(entry)
            if not entry.skipped:
                transcribed += 1

        log.info(
            "Article transcription complete",
            extra={
                "url": article.url,
                "transcribed": transcribed,
                "skipped": sum(r.skipped for r in results),
            },
        )
        return results
    finally:
        conn.close()


def _unavailable_reason(ref: ImageRef) -> str:
    return "unavailable" if ref.local_path is not None else "no_browser_context"


def _resolve_bytes(ref: ImageRef, browser_context: object | None) -> bytes | None:
    if ref.local_path is not None:
        try:
            return Path(ref.local_path).read_bytes()
        except OSError:
            log.error(
                "Local image gone at transcription time",
                extra={"local_path": ref.local_path},
                exc_info=True,
            )
            return None
    # Phase 2 web image
    if browser_context is None:
        log.error("Paywall image with no browser context — skipping", extra={"url": ref.src})
        return None
    raise NotImplementedError("Phase 2 image download lands in Milestone 3")  # pragma: no cover


def _handle_one(
    conn, provider, article: Article, ref: ImageRef, image_bytes: bytes
) -> ImageTranscription:
    source_key = _source_key(ref)

    valid, reason, image = _validate(image_bytes)
    if not valid:
        log.debug("Image failed validation", extra={"url": ref.src, "skip_reason": reason})
        return _skip(ref, reason)

    file_hash = sha256(image_bytes).hexdigest()
    ext = "jpg" if image.format == "JPEG" else "png"
    file_path = str(Path(IMAGE_CACHE_DIR) / f"{file_hash}.{ext}")
    if not Path(file_path).exists():
        Path(file_path).write_bytes(image_bytes)

    cached = _cache_lookup(conn, source_key)
    if cached and cached["transcription"] and cached["model"] == provider.model_name:
        log.debug("Cache hit — skipping vision call", extra={"image_url": ref.src})
        return ImageTranscription(
            image_ref=ref,
            file_path=cached["file_path"],
            file_hash=cached["file_hash"],
            image_type=cached["image_type"],
            transcription=cached["transcription"],
            model=cached["model"],
            input_tokens=0,
            output_tokens=0,
            skipped=False,
            skip_reason=None,
        )

    media_type = "image/jpeg" if ext == "jpg" else "image/png"
    image_type = _classify(provider, image_bytes, media_type)

    if image_type not in TRANSCRIBE_TYPES:  # "photo" or "unknown"
        log.debug(
            "Image skipped (photo/unknown type)",
            extra={"image_url": ref.src, "image_type": image_type},
        )
        _cache_write(conn, source_key, file_path, file_hash, image_type, "", provider.model_name)
        return _skip(
            ref, image_type, image_type=image_type, file_path=file_path, file_hash=file_hash
        )

    resp = provider.complete_with_image(
        image_bytes=image_bytes,
        media_type=media_type,
        text_prompt=_transcription_prompt(article, ref, image_type),
        max_tokens=VISION_MAX_TOKENS,
    )
    _cache_write(conn, source_key, file_path, file_hash, image_type, resp.content, resp.model)
    log.info(
        "Transcription complete",
        extra={
            "image_url": ref.src,
            "image_type": image_type,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        },
    )
    return ImageTranscription(
        image_ref=ref,
        file_path=file_path,
        file_hash=file_hash,
        image_type=image_type,
        transcription=resp.content,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        skipped=False,
        skip_reason=None,
    )


def _classify(provider, image_bytes: bytes, media_type: str) -> str:
    resp = provider.complete_with_image(
        image_bytes=image_bytes,
        media_type=media_type,
        text_prompt=TYPE_DETECTION_PROMPT,
        max_tokens=10,
    )
    word = resp.content.strip().lower().split()[0] if resp.content.strip() else "unknown"
    return word if word in _VALID_TYPES else "unknown"


def _validate(image_bytes: bytes) -> tuple[bool, str | None, Image.Image | None]:
    if len(image_bytes) < MIN_IMAGE_BYTES:
        return False, "too_small", None
    if len(image_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        return False, "too_large", None
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "invalid_image", None
    if min(image.size) < MIN_IMAGE_PIXELS:
        return False, "too_small", None
    return True, None, image


def _skip(
    ref: ImageRef,
    reason: str,
    *,
    image_type: str = "unknown",
    file_path: str = "",
    file_hash: str = "",
) -> ImageTranscription:
    return ImageTranscription(
        image_ref=ref,
        file_path=file_path,
        file_hash=file_hash,
        image_type=image_type,
        transcription="",
        model="",
        input_tokens=0,
        output_tokens=0,
        skipped=True,
        skip_reason=reason,
    )


def _cache_write(conn, source_key, file_path, file_hash, image_type, transcription, model) -> None:
    from datetime import UTC, datetime

    conn.execute(
        "INSERT INTO image_cache (source_key, file_path, file_hash, image_type, "
        "transcription, model, cached_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(source_key) DO UPDATE SET file_path=excluded.file_path, "
        "file_hash=excluded.file_hash, image_type=excluded.image_type, "
        "transcription=excluded.transcription, model=excluded.model, cached_at=excluded.cached_at",
        (
            source_key,
            file_path,
            file_hash,
            image_type,
            transcription,
            model,
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
