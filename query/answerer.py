"""Assemble a grounded prompt from retrieved chunks and call the text provider for a
cited answer. Final stage of the query pipeline.

Empty results short-circuit to a graceful "not found" with no provider call. Otherwise the
excerpts are capped at `MAX_CONTEXT_CHUNKS` and truncated at `MAX_CHUNK_CHARS`, the model
answers with `[N]` citations, and only the cited, in-range excerpts become `Source`s.
"""

from __future__ import annotations

import re

from config import AUTHOR_NAME, MAX_CHUNK_CHARS, MAX_CONTEXT_CHUNKS, MAX_OUTPUT_TOKENS
from llm_provider import get_text_provider
from logger import get_logger
from models import Answer, SearchResult, Source

log = get_logger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = f"""You are an expert research assistant with deep knowledge of {AUTHOR_NAME}'s writing and ideas.

You will be given a question and a set of numbered excerpts from {AUTHOR_NAME}'s articles.

Rules:
1. Answer using ONLY information present in the provided excerpts.
2. Cite sources inline using [N] notation, where N is the excerpt number.
3. If the excerpts do not contain enough information to answer, say so clearly — do not speculate.
4. If multiple excerpts support the same point, cite all of them: [1][3].
5. Preserve {AUTHOR_NAME}'s voice and terminology when summarising.
6. Keep your answer concise: 2–4 paragraphs unless detail is explicitly requested."""

_NOT_FOUND = "I couldn't find relevant content in the knowledge base for this question."


def answer(query: str, results: list[SearchResult]) -> Answer:
    """Grounded, cited answer for `query` from `results`. A graceful not-found Answer if
    `results` is empty — no provider call is made."""
    if not results:
        log.info("No results — returning graceful answer", extra={"query": query})
        return Answer(
            query=query, response=_NOT_FOUND, sources=[], model="", input_tokens=0, output_tokens=0
        )

    excerpts = results[:MAX_CONTEXT_CHUNKS]
    context = _context_block(excerpts, query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    log.debug(
        "Context block assembled",
        extra={"query": query, "excerpt_count": len(excerpts), "total_context_chars": len(context)},
    )

    provider = get_text_provider()
    resp = provider.complete(messages, max_tokens=MAX_OUTPUT_TOKENS)
    log.info(
        "LLM call complete",
        extra={
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        },
    )

    sources = _sources(resp.content, excerpts, query)
    return Answer(
        query=query,
        response=resp.content,
        sources=sources,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


def _truncate(text: str, chunk_id: str) -> str:
    if len(text) <= MAX_CHUNK_CHARS:
        return text
    cut = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0]
    log.debug(
        "Chunk truncated to MAX_CHUNK_CHARS",
        extra={"chunk_id": chunk_id, "original_chars": len(text)},
    )
    return cut + " …"


def _context_block(excerpts: list[SearchResult], query: str) -> str:
    parts = ["EXCERPTS:", ""]
    for i, r in enumerate(excerpts, start=1):
        when = r.published_at.date().isoformat() if r.published_at else "n.d."
        parts.append(f'[{i}] "{_truncate(r.text, r.chunk_id)}"')
        parts.append(f"Source: {r.article_title} ({when})")
        parts.append(f"URL: {r.article_url}")
        parts.append("")
    parts.append(f"QUESTION: {query}")
    return "\n".join(parts)


def _sources(response_text: str, excerpts: list[SearchResult], query: str) -> list[Source]:
    cited = {int(n) for n in _CITATION.findall(response_text)}
    if not cited:
        log.warning("Response contains no citation markers", extra={"query": query})

    out: list[Source] = []
    for i in sorted(cited):
        if not 1 <= i <= len(excerpts):
            log.warning(
                "Out-of-range citation index skipped",
                extra={"citation_index": i, "max_valid_index": len(excerpts)},
            )
            continue
        r = excerpts[i - 1]
        out.append(
            Source(
                index=i,
                title=r.article_title,
                url=r.article_url,
                published_at=r.published_at,
                score=r.score,
            )
        )
    return out
