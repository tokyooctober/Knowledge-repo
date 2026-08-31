"""Phase 2 counterpart of `md_loader`: turn scraped HTML (`RawPage`) into an `Article`.

Produces the same dataclass `md_loader.load_article` does — `source="web"`,
`source_path=None`, every `ImageRef.local_path=None`. Everything downstream consumes both
without branching on the source. Does not download or transcribe images.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from config import ARTICLE_BODY_SELECTOR, MIN_WORD_COUNT, SITE_DOMAIN
from logger import get_logger
from models import Article, ImageRef, canonical_url, content_hash

if TYPE_CHECKING:
    from models import RawPage

log = get_logger(__name__)

_ZERO_WIDTH = re.compile(r"[​-‏‪-‮﻿]")
_BLANKS = re.compile(r"\n{3,}")


def extract(raw_page: RawPage) -> Article | None:
    """Structured `Article` from a raw HTML page, or `None` if the page is not an article
    (non-200). An article below `MIN_WORD_COUNT` returns with `is_stub=True`."""
    if raw_page.status_code != 200:
        log.debug(
            "Skipping non-200 page",
            extra={"url": raw_page.url, "status_code": raw_page.status_code},
        )
        return None

    url = canonical_url(raw_page.url)
    soup = BeautifulSoup(raw_page.html, "lxml")
    article_el = soup.select_one(ARTICLE_BODY_SELECTOR) or soup

    tables_md = _extract_tables(article_el, raw_page.url)
    images = _extract_images(article_el, raw_page.url)

    body_text = _extract_body(raw_page.html, article_el, raw_page.url)
    meta = _metadata(raw_page.html, soup)

    word_count = len(body_text.split())
    is_stub = word_count < MIN_WORD_COUNT
    if is_stub:
        log.warning(
            "Article below MIN_WORD_COUNT (stub)", extra={"url": url, "word_count": word_count}
        )

    chash = content_hash(body_text, tables_md, images)
    log.debug(
        "Extraction complete",
        extra={
            "url": url,
            "title": meta["title"],
            "word_count": word_count,
            "table_count": len(tables_md),
            "image_count": len(images),
            "content_hash": chash,
        },
    )
    return Article(
        url=url,
        title=meta["title"],
        author=meta["author"],
        published_at=meta["published_at"],
        fetched_at=raw_page.fetched_at,
        tags=meta["tags"],
        body_text=body_text,
        tables_md=tables_md,
        images=images,
        word_count=word_count,
        content_hash=chash,
        is_stub=is_stub,
        source="web",
        source_path=None,
    )


# ── tables ─────────────────────────────────────────────────────────────────


def _is_layout_table(table) -> bool:
    """Layout tables have no <th> and are not clearly tabular (>1 col AND >2 rows)."""
    if table.find("th"):
        return False
    rows = table.find_all("tr")
    cols = max((len(r.find_all(["td", "th"])) for r in rows), default=0)
    return not (cols > 1 and len(rows) > 2)


def _extract_tables(article_el, url: str) -> list[str]:
    import pandas as pd

    out: list[str] = []
    for table in list(article_el.find_all("table")):
        if _is_layout_table(table):
            log.debug("Layout table skipped", extra={"url": url})
            table.decompose()
            continue
        caption_el = table.find("caption")
        caption = caption_el.get_text(strip=True) if caption_el else ""
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
            md = df.to_markdown(index=False)
        except (ValueError, ImportError):
            log.warning(
                "Table conversion failed — fallback used",
                extra={"url": url, "table_index": len(out)},
            )
            md = _table_fallback(table)
        prefix = (
            f'[Table {len(out) + 1}: "{caption}"]\n' if caption else f"[Table {len(out) + 1}]\n"
        )
        out.append(prefix + md)
        table.decompose()
    if out:
        log.debug("HTML tables found and converted", extra={"url": url, "table_count": len(out)})
    return out


def _table_fallback(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    if len(rows) >= 1:
        rows.insert(1, "|" + "|".join(["---"] * (rows[0].count("|") - 1)) + "|")
    return "\n".join(rows)


# ── images ─────────────────────────────────────────────────────────────────


def _extract_images(article_el, article_url: str) -> list[ImageRef]:
    images: list[ImageRef] = []
    for i, img in enumerate(article_el.find_all("img")):
        raw = img.get("src", "").strip()
        if not raw or raw.startswith("data:"):
            log.debug(
                "Image src skipped (empty or data URI)", extra={"url": article_url, "position": i}
            )
            continue
        src = urljoin(article_url, raw)
        if not src.startswith("http"):
            log.warning("Image src resolution failed", extra={"url": article_url, "raw_src": raw})
            continue
        caption = ""
        parent = img.find_parent(["figure", "div"])
        if parent and (cap := parent.find(["figcaption", "caption"])):
            caption = cap.get_text(strip=True)
        images.append(
            ImageRef(
                src=src,
                local_path=None,
                alt=img.get("alt", "").strip(),
                caption=caption,
                position=i,
                is_paywall=_on_site(src),
            )
        )
    log.debug(
        "Image refs collected",
        extra={
            "url": article_url,
            "image_count": len(images),
            "paywall_count": sum(im.is_paywall for im in images),
        },
    )
    return images


def _on_site(src: str) -> bool:
    host = urlparse(src).netloc.lower().removeprefix("www.")
    return host == SITE_DOMAIN or host.endswith("." + SITE_DOMAIN)


# ── body + metadata ────────────────────────────────────────────────────────


def _extract_body(html: str, article_el, url: str) -> str:
    text = trafilatura.extract(
        html, include_images=False, include_tables=False, favor_precision=True
    )
    if text and len(text.split()) >= MIN_WORD_COUNT:
        log.debug(
            "trafilatura extraction succeeded", extra={"url": url, "word_count": len(text.split())}
        )
    else:
        log.warning(
            "Trafilatura returned little — falling back to BeautifulSoup", extra={"url": url}
        )
        text = article_el.get_text(separator="\n", strip=True)
    return _clean(text or "")


def _clean(text: str) -> str:
    text = _ZERO_WIDTH.sub("", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def _metadata(html: str, soup: BeautifulSoup) -> dict:
    title = author = None
    published_at: datetime | None = None
    tags: list[str] = []
    try:
        doc = trafilatura.bare_extraction(
            html, with_metadata=True, include_tables=False, include_images=False
        )
        if doc is not None:
            title = doc.title or None
            author = doc.author or None
            tags = list(doc.tags or [])
            if doc.date:
                published_at = _parse_date(doc.date)
    except Exception:  # noqa: BLE001 - metadata is best-effort
        log.debug("trafilatura metadata extraction failed", exc_info=True)

    if not title:
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif h1 := soup.find("h1"):
            title = h1.get_text(strip=True)
    return {
        "title": title or "Untitled",
        "author": author,
        "published_at": published_at,
        "tags": tags,
    }


def _parse_date(value: str) -> datetime | None:
    try:
        return date_parser.parse(value)
    except (ValueError, OverflowError):
        log.debug("Date parsing failed", extra={"raw": value})
        return None
