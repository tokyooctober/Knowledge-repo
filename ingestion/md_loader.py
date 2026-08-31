"""Phase 1 source: turn a local folder of markdown articles + their images into `Article`
objects — the same dataclass `ingestion/extractor.py` builds from scraped HTML.

The single rule that must never regress: images resolve by the **whole relative path**
written in the markdown, against the article's own directory (`md_path.parent`), never by
basename and never against the `MD_CORPUS_DIR` constant. Every article has an `images/…/00-*`
file; a basename lookup would silently attach one report's chart to another.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter
import yaml
from dateutil import parser as date_parser

from config import AUTHOR_NAME, MIN_WORD_COUNT
from logger import get_logger
from models import Article, ImageRef, canonical_url, content_hash

log = get_logger(__name__)


class CorpusNotFoundError(Exception):
    """MD_CORPUS_DIR does not exist."""


class CorpusEmptyError(Exception):
    """MD_CORPUS_DIR exists but contains no top-level *.md files."""


# ── corpus scan ─────────────────────────────────────────────────────────────


def iter_article_paths(corpus_dir: str | None = None) -> list[Path]:
    """Sorted *.md paths at the **top level** of the corpus directory.

    `images/` and everything under it is never scanned — the per-article subfolders carry
    article-shaped names, which makes a recursive walk look plausible and wrong.
    """
    from config import MD_CORPUS_DIR

    root = Path(corpus_dir or MD_CORPUS_DIR)
    log.info("Corpus scan started", extra={"corpus_dir": str(root)})
    if not root.is_dir():
        log.critical("Corpus directory missing", extra={"corpus_dir": str(root)}, exc_info=True)
        raise CorpusNotFoundError(str(root))

    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name != "images":
            log.debug("Non-markdown subfolder ignored", extra={"path": str(child)})

    paths = sorted(p for p in root.glob("*.md") if p.is_file())
    if not paths:
        log.critical("Corpus directory empty", extra={"corpus_dir": str(root)})
        raise CorpusEmptyError(str(root))
    log.info("Corpus scan complete", extra={"corpus_dir": str(root), "file_count": len(paths)})
    return paths


def load_corpus(corpus_dir: str | None = None) -> list[Article]:
    """`iter_article_paths` + `load_article`, dropping the `None`s."""
    return [a for p in iter_article_paths(corpus_dir) if (a := load_article(p)) is not None]


# ── one article ────────────────────────────────────────────────────────────

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_MD_IMAGE = re.compile(
    r"""!\[(?P<alt>[^\]]*)\]\(\s*(?P<src><[^>]+>|[^)\s]+)(?:\s+"(?P<title>[^"]*)")?\s*\)"""
)
_HTML_IMAGE = re.compile(r"<img\b[^>]*?>", re.IGNORECASE)
_HTML_ATTR = re.compile(r"""(\w+)\s*=\s*["']([^"']*)["']""")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*\s*$")
_URL = re.compile(r"^https?://", re.IGNORECASE)


def load_article(md_path: Path) -> Article | None:
    """Parse one markdown file into an `Article` (`source="corpus"`).

    Returns `None` only for a file that cannot be parsed at all (bad UTF-8, malformed
    YAML). A body below `MIN_WORD_COUNT` returns an `Article` with `is_stub=True` and a
    valid `content_hash` — the scheduler counts that as skipped, not failed, and a later
    edit past the threshold then registers as a change rather than a new article.
    """
    md_path = Path(md_path)

    # 1. READ
    try:
        text = md_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        log.error(
            "File unreadable",
            extra={"md_file": str(md_path), "error_type": type(exc).__name__},
            exc_info=True,
        )
        return None

    # 2. FRONTMATTER
    try:
        post = frontmatter.loads(text)
    except (yaml.YAMLError, ValueError) as exc:
        log.error(
            "Malformed frontmatter",
            extra={"md_file": str(md_path), "error_type": type(exc).__name__},
            exc_info=True,
        )
        return None
    metadata: dict = post.metadata or {}
    body: str = post.content
    if not metadata:
        log.warning("No frontmatter block", extra={"md_file": str(md_path)})

    # 3. METADATA
    title = _first(metadata.get("title")) or _first_h1(body) or md_path.stem
    raw_url = _first(metadata.get("url"))
    url = canonical_url(raw_url or "")
    if not url:
        url = f"local:{md_path.stem}"
        log.warning(
            "url missing — synthesised", extra={"md_file": str(md_path), "synthesised_url": url}
        )
    published_at = _parse_date(metadata.get("published"), md_path)
    author = _first(metadata.get("author")) or AUTHOR_NAME
    tags = list(metadata.get("tags") or [])

    # 4. IMAGES (before tables — image syntax can sit inside a table cell)
    body, images = _extract_images(body, md_path)

    # 4b. EXPORT CROSS-CHECKS (cheap, never authoritative)
    _cross_check(metadata, images, body, md_path)

    # 5. TABLES
    body, tables_md = _extract_tables(body)

    # 6. CLEAN BODY
    body_text = _clean_body(body)
    word_count = len(body_text.split())

    # 7. HASH (shared helper — never reimplemented here)
    chash = content_hash(body_text, tables_md, images)

    # 8. STUB
    is_stub = word_count < MIN_WORD_COUNT
    if is_stub:
        log.warning(
            "Article is a stub",
            extra={
                "md_file": str(md_path),
                "word_count": word_count,
                "min_required": MIN_WORD_COUNT,
            },
        )

    log.debug(
        "Article loaded",
        extra={
            "md_file": str(md_path),
            "url": url,
            "word_count": word_count,
            "table_count": len(tables_md),
            "image_count": len(images),
        },
    )
    return Article(
        url=url,
        title=title,
        author=author,
        published_at=published_at,
        fetched_at=_utcnow(),
        tags=tags,
        body_text=body_text,
        tables_md=tables_md,
        images=images,
        word_count=word_count,
        content_hash=chash,
        is_stub=is_stub,
        source="corpus",
        source_path=str(md_path.resolve()),
    )


# ── helpers ────────────────────────────────────────────────────────────────


def _utcnow():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def _first(value: object) -> str:
    """Frontmatter value as a stripped string; '' for None/empty (empty == absent)."""
    if value is None:
        return ""
    return str(value).strip()


def _first_h1(body: str) -> str:
    m = _H1.search(body)
    return m.group(1).strip() if m else ""


def _parse_date(value: object, md_path: Path):
    raw = _first(value)
    if not raw:
        return None
    try:
        return date_parser.parse(raw)
    except (ValueError, OverflowError):
        log.warning("published unparseable", extra={"md_file": str(md_path), "raw_value": raw})
        return None


def _looks_like_date(s: str) -> bool:
    if not s or len(s) > 30:
        return False
    try:
        date_parser.parse(s, fuzzy=False)
        return True
    except (ValueError, OverflowError):
        return False


def _caption(title: str, following: str) -> str:
    """Pick a caption, rejecting the two things the export reliably produces: a title
    attribute holding the original remote URL, and a nearest-italic line that is a bare
    date (`*June 27, 2021*` under every cover)."""
    for candidate in (title.strip(), following.strip()):
        if not candidate or _URL.match(candidate) or _looks_like_date(candidate):
            continue
        return candidate
    return ""


def _extract_images(body: str, md_path: Path) -> tuple[str, list[ImageRef]]:
    article_dir = md_path.parent.resolve()
    matches: list[tuple[int, int, str, str, str]] = []  # start, end, alt, src, title
    for m in _MD_IMAGE.finditer(body):
        src = m.group("src").strip().strip("<>")
        matches.append(
            (m.start(), m.end(), (m.group("alt") or "").strip(), src, m.group("title") or "")
        )
    for m in _HTML_IMAGE.finditer(body):
        attrs = dict(_HTML_ATTR.findall(m.group(0)))
        if "src" in attrs:
            matches.append(
                (m.start(), m.end(), attrs.get("alt", "").strip(), attrs["src"].strip(), "")
            )
    matches.sort(key=lambda t: t[0])

    images: list[ImageRef] = []
    seen: dict[Path, int] = {}
    spans: list[tuple[int, int]] = []
    for position, (start, end, alt, src, title) in enumerate(matches):
        spans.append((start, end))
        if _URL.match(src) or src.startswith("data:"):
            log.warning(
                "Image path is remote or a data URI — skipped",
                extra={"md_file": str(md_path), "image_path": src},
            )
            continue
        resolved = (article_dir / src).resolve()
        if article_dir not in resolved.parents:
            log.warning(
                "Image path outside the article folder",
                extra={"md_file": str(md_path), "image_path": src},
            )
            continue
        if not resolved.exists():
            log.warning("Image file not found", extra={"md_file": str(md_path), "image_path": src})
            continue
        if resolved in seen:
            log.debug(
                "Duplicate image reference dropped",
                extra={
                    "md_file": str(md_path),
                    "image_path": src,
                    "first_position": seen[resolved],
                },
            )
            continue
        seen[resolved] = position
        following = _following_italic(body, end)
        images.append(
            ImageRef(
                src=src,
                local_path=str(resolved),
                alt=alt,
                caption=_caption(title, following),
                position=position,
                is_paywall=False,
            )
        )

    # strip every image span from the body so alt text is not chunked as prose
    for start, end in sorted(spans, reverse=True):
        body = body[:start] + body[end:]
    return body, images


_ITALIC_LINE = re.compile(r"\s*\n\s*[*_]([^*_\n]+)[*_]\s*(?:\n|$)")


def _following_italic(body: str, after: int) -> str:
    m = _ITALIC_LINE.match(body, after)
    return m.group(1).strip() if m else ""


def _extract_tables(body: str) -> tuple[str, list[str]]:
    lines = body.split("\n")
    kept: list[str] = []
    tables: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if "|" in line and "-" in nxt and _TABLE_SEP.match(nxt):
            block = [line, nxt]
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                block.append(lines[j])
                j += 1
            tables.append("\n".join(block).strip())
            i = j
        else:
            kept.append(line)
            i += 1
    return "\n".join(kept), tables


_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC = re.compile(r"(\*|_)(.+?)\1")
_CODE = re.compile(r"`([^`]+)`")
_BQ = re.compile(r"^>\s?", re.MULTILINE)
_LIST = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_HR = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")


def _clean_body(body: str) -> str:
    body = _MD_IMAGE.sub("", body)
    body = _HTML_IMAGE.sub("", body)
    body = _LINK.sub(r"\1", body)
    body = _HEADING.sub("", body)
    body = _HR.sub("", body)
    body = _BOLD.sub(r"\2", body)
    body = _ITALIC.sub(r"\2", body)
    body = _CODE.sub(r"\1", body)
    body = _BQ.sub("", body)
    body = _LIST.sub("", body)
    body = _BLANKS.sub("\n\n", body)
    return "\n".join(line.rstrip() for line in body.split("\n")).strip()


def _cross_check(metadata: dict, images: list[ImageRef], body: str, md_path: Path) -> None:
    declared_images = metadata.get("images_saved")
    if isinstance(declared_images, int):
        if declared_images > len(images):
            log.warning(
                "Fewer images resolved than images_saved",
                extra={
                    "md_file": str(md_path),
                    "images_saved": declared_images,
                    "images_resolved": len(images),
                },
            )
        elif declared_images < len(images):
            log.debug(
                "More image references than files written (cover repeats)",
                extra={"md_file": str(md_path)},
            )
    if metadata.get("truncated") is True:
        log.warning(
            "Article marked truncated by the export",
            extra={"md_file": str(md_path), "url": _first(metadata.get("url"))},
        )
    declared_wc = metadata.get("word_count")
    computed_wc = len(body.split())
    if isinstance(declared_wc, int) and declared_wc > 0 and computed_wc > 0:
        if abs(declared_wc - computed_wc) / max(declared_wc, computed_wc) > 0.5:
            log.warning(
                "Declared and computed word_count diverge",
                extra={"md_file": str(md_path), "declared": declared_wc, "computed": computed_wc},
            )
