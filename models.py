"""Shared dataclasses, the two identity helpers, and the cross-module exceptions.

`models.py` is the root of the dependency graph: it imports nothing first-party, so
`config.py` can import `ConfigError` from here without a cycle. Keep it that way — never
import `config` or `logger` into this module (see SPEC.md § Shared exceptions).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# ── Shared exceptions ─────────────────────────────────────────────────────────


class ConfigError(Exception):
    """A required setting is missing or has an unusable value.

    Raised at point of use (`config.require_phase2_config`), never at import. Also
    raised by `llm_provider` for an unknown ``*_BACKEND`` value.
    """


class ModelMismatchError(Exception):
    """The configured embedding model does not match the one recorded on the Qdrant
    collection (or the metadata row).

    Raised by `vector_store`, `metadata_db`, `embedder`, and `retriever`. The message
    carries both model names and the re-index instruction.
    """


class VisionNotSupportedError(ConfigError):
    """``VISION_MODEL`` cannot accept image input.

    Raised by `llm_provider.get_vision_provider()` at startup.
    """


# ── Email trigger (Phase 2) ──────────────────────────────────────────────────


@dataclass
class ArticleLink:
    title: str | None  # email subject, or None
    url: str


@dataclass
class EmailUpdate:
    email_uid: str
    sender: str
    subject: str
    received_at: datetime
    article_links: list[ArticleLink]  # length 1 for this email format
    raw_body: str


# ── Scraping (Phase 2) ───────────────────────────────────────────────────────


@dataclass
class RawPage:
    url: str
    html: str
    fetched_at: datetime
    status_code: int
    """The crawler's own request status AFTER any inline login is resolved and the
    ``SUCCESS_SELECTOR`` guard passes — 200 for every page that reaches the extractor.
    See SPEC_crawler.md step (e)."""
    is_new: bool
    """Diagnostic only: ``url not in known_urls`` at fetch time. Change detection is
    ``metadata_db.is_changed()``; nothing downstream branches on this field."""


# ── Articles and images (both phases) ────────────────────────────────────────


@dataclass
class ImageRef:
    src: str  # absolute URL (web) or original markdown path (corpus)
    local_path: str | None  # absolute path on disk (corpus); None for web
    alt: str
    caption: str  # nearest figcaption / italic caption line; "" if none
    position: int  # 0-based index within the article
    is_paywall: bool  # True if src is on SITE_DOMAIN; always False for corpus


@dataclass
class Article:
    url: str  # PRIMARY KEY across the whole system
    title: str
    author: str | None
    published_at: datetime | None
    fetched_at: datetime
    tags: list[str]
    body_text: str  # clean prose, newline-separated paragraphs
    tables_md: list[str]  # one Markdown table per entry
    images: list[ImageRef]
    word_count: int  # body_text only
    content_hash: str  # see content_hash() below — one formula, both sources
    is_stub: bool
    source: str  # "corpus" | "web"
    source_path: str | None  # absolute .md path (corpus); None for web


@dataclass
class ImageTranscription:
    image_ref: ImageRef  # the source ImageRef from the article
    file_path: str  # normalised copy on disk: images/{hash}.{ext}
    file_hash: str  # sha256 of image bytes
    image_type: str  # "chart" | "table" | "diagram" | "photo" | "unknown"
    transcription: str  # vision output; "" if skipped
    model: str
    input_tokens: int
    output_tokens: int
    skipped: bool  # True for photos, tiny images, validation failures
    skip_reason: str | None


# ── Chunking and embedding ───────────────────────────────────────────────────


@dataclass
class Chunk:
    chunk_id: str  # f"{url_hash}_{content_type[0]}_{chunk_index}"
    article_url: str
    article_title: str
    published_at: datetime | None
    tags: list[str]
    text: str
    content_type: str  # "body" | "table" | "image_transcription"
    chunk_index: int  # 0-based within this content_type group
    total_chunks: int  # total for this article, all types
    word_count: int


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]
    model_name: str


# ── Retrieval and answers ────────────────────────────────────────────────────


@dataclass
class SearchResult:
    score: float
    text: str
    chunk_id: str  # stable id from the Qdrant payload (chunker.py assigned it)
    article_url: str
    article_title: str
    published_at: datetime | None
    tags: list[str]
    content_type: str
    chunk_index: int


@dataclass
class Source:
    index: int
    title: str
    url: str
    published_at: datetime | None
    score: float


@dataclass
class Answer:
    query: str
    response: str
    sources: list[Source]
    model: str
    input_tokens: int
    output_tokens: int


# ── The two identity helpers ─────────────────────────────────────────────────


def content_hash(body_text: str, tables_md: list[str], images: list[ImageRef]) -> str:
    """The ONE hash formula. Both md_loader.py and extractor.py call this.

    A web image (``local_path is None``) contributes only ``f"{img.src}:"`` — a re-scrape
    of unchanged HTML hashes the same. A corpus image also contributes its size and mtime,
    so replacing a chart file in place forces a re-ingest. The ``.md`` file's own path and
    mtime are never hashed, so renaming an article does not force a re-embed.
    """
    parts = [body_text, *tables_md]
    for img in images:  # already in document order
        if img.local_path is None:  # web image: URL is the only identity
            parts.append(f"{img.src}:")
        else:  # corpus image: bytes may change in place
            st = Path(img.local_path).stat()
            parts.append(f"{img.src}:{st.st_size}:{st.st_mtime_ns}")
    return sha256("".join(parts).encode()).hexdigest()


def canonical_url(raw: str) -> str:
    """The ONE url normal form. Both md_loader.py and extractor.py call this.

    ``http`` and ``https`` are one article; ``www.`` is stripped; a trailing slash is not
    identity; query and fragment are dropped. Safe for this site because article identity
    is entirely in the path — revisit before pointing the pipeline at a site that
    paginates or versions through query strings.
    """
    if not raw:
        return ""
    u = urlsplit(raw.strip())
    scheme = "https"  # http and https are one article
    netloc = u.netloc.lower().removeprefix("www.")
    path = u.path.rstrip("/") or "/"  # trailing slash is not identity
    return urlunsplit((scheme, netloc, path, "", ""))  # query and fragment dropped
