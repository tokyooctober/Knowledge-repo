"""Split a clean `Article` — body prose, extracted tables, vision transcriptions — into
overlapping, typed `Chunk`s for embedding and retrieval.

Body text is split with overlap. Tables and transcriptions are never split: half a table
is malformed Markdown, and a reader would never answer a table question from half a table.
Every chunk carries `content_type` so retrieval can filter to charts or tables only.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_WORDS,
    TOKENIZER,
    USE_HEADER_SPLITTING,
)
from logger import get_logger
from models import Chunk

if TYPE_CHECKING:
    from models import Article, ImageTranscription

log = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_TYPE_CODE = {"body": "b", "table": "t", "image_transcription": "i"}


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )


def _wc(text: str) -> int:
    return len(text.split())


def _token_count(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding(TOKENIZER).encode(text))


def chunk_text(article: Article) -> list[Chunk]:
    """Body + table chunks only (no image transcriptions). [] for a stub."""
    if article.is_stub:
        log.debug(
            "Skipping stub article",
            extra={"url": article.url, "word_count": article.word_count},
        )
        return []

    url_hash = sha256(article.url.encode()).hexdigest()[:8]
    log.debug(
        "Body text chunking started",
        extra={"url": article.url, "body_length_chars": len(article.body_text)},
    )
    return _body_chunks(article, url_hash) + _table_chunks(article, url_hash)


def chunk_images(
    article: Article,
    image_transcriptions: list[ImageTranscription] | None = None,
) -> list[Chunk]:
    """Image-transcription chunks only. [] for a stub or when there are none."""
    if article.is_stub:
        return []
    url_hash = sha256(article.url.encode()).hexdigest()[:8]
    return _image_chunks(article, url_hash, image_transcriptions or [])


def finalize_chunks(article: Article, chunks: list[Chunk]) -> list[Chunk]:
    """Set the shared `total_chunks` across a full (text + image) chunk list and log the
    summary. Call once after `chunk_text` and `chunk_images` results are merged."""
    if not chunks:
        log.warning("All chunks filtered — article produced nothing", extra={"url": article.url})

    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    body_n = sum(c.content_type == "body" for c in chunks)
    table_n = sum(c.content_type == "table" for c in chunks)
    image_n = sum(c.content_type == "image_transcription" for c in chunks)
    log.info(
        "Chunking complete",
        extra={
            "url": article.url,
            "total_chunks": total,
            "body_chunks": body_n,
            "table_chunks": table_n,
            "image_chunks": image_n,
        },
    )
    return chunks


def chunk_article(
    article: Article,
    image_transcriptions: list[ImageTranscription] | None = None,
) -> list[Chunk]:
    """Body + table + image-transcription chunks for one article.

    Returns [] for a stub. `image_transcriptions=None` yields only body + table chunks.
    All chunks share `total_chunks`; `chunk_index` is 0-based within each content-type
    group and contiguous (skipped items leave no gap).
    """
    if article.is_stub:
        log.debug(
            "Skipping stub article",
            extra={"url": article.url, "word_count": article.word_count},
        )
        return []
    chunks = chunk_text(article) + chunk_images(article, image_transcriptions)
    return finalize_chunks(article, chunks)


def _make(article: Article, url_hash: str, content_type: str, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{url_hash}_{_TYPE_CODE[content_type]}_{index:04d}",
        article_url=article.url,
        article_title=article.title,
        published_at=article.published_at,
        tags=list(article.tags),
        text=text,
        content_type=content_type,
        chunk_index=index,
        total_chunks=0,  # set by the caller once the full list is known
        word_count=_wc(text),
    )


def _body_chunks(article: Article, url_hash: str) -> list[Chunk]:
    if not article.body_text.strip():
        return []
    if USE_HEADER_SPLITTING and article.body_text.lstrip().startswith("#"):
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        ).split_text(article.body_text)
        raw = [s.page_content for s in sections]
    else:
        raw = [article.body_text]

    splitter = _splitter()
    texts: list[str] = []
    for section in raw:
        texts.extend(splitter.split_text(section))

    out: list[Chunk] = []
    for text in texts:
        if _wc(text) < MIN_CHUNK_WORDS:
            log.warning(
                "Body chunk too short — discarded",
                extra={
                    "url": article.url,
                    "word_count": _wc(text),
                    "min_chunk_words": MIN_CHUNK_WORDS,
                },
            )
            continue
        index = len(out)
        if _token_count(text) > CHUNK_SIZE:  # pragma: no cover - the char splitter prevents this
            log.warning(
                "Body chunk exceeds CHUNK_SIZE",
                extra={"url": article.url, "chunk_index": index, "token_count": _token_count(text)},
            )
        out.append(_make(article, url_hash, "body", index, text))
    return out


def _table_chunks(article: Article, url_hash: str) -> list[Chunk]:
    out: list[Chunk] = []
    for table_md in article.tables_md:
        if _wc(table_md) < MIN_CHUNK_WORDS:
            log.warning(
                "Table chunk too short — skipped",
                extra={"url": article.url, "table_index": len(out), "word_count": _wc(table_md)},
            )
            continue
        index = len(out)
        if _token_count(table_md) > CHUNK_SIZE:
            log.warning(
                "Table chunk oversized",
                extra={
                    "url": article.url,
                    "table_index": index,
                    "token_count": _token_count(table_md),
                },
            )
        out.append(_make(article, url_hash, "table", index, table_md))
    return out


def _image_chunks(
    article: Article, url_hash: str, transcriptions: list[ImageTranscription]
) -> list[Chunk]:
    out: list[Chunk] = []
    for transcription in transcriptions:
        if transcription.skipped:
            log.debug(
                "Image transcription skipped",
                extra={"url": article.url, "skip_reason": transcription.skip_reason},
            )
            continue
        text = transcription.transcription
        if _wc(text) < MIN_CHUNK_WORDS:
            continue
        index = len(out)
        out.append(_make(article, url_hash, "image_transcription", index, text))
        log.debug(
            "Image transcription chunk added",
            extra={
                "url": article.url,
                "image_index": index,
                "image_type": transcription.image_type,
                "word_count": _wc(text),
            },
        )
    return out
