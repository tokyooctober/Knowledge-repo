"""Turn `Chunk`s into `EmbeddedChunk`s (indexing) and a query string into one vector.

Contains no SDK imports and no model names — everything goes through
`llm_provider.get_embedding_provider()`, so the same provider embeds the corpus and the
query. The instruction `query_prefix` (BGE / nomic) is applied at **query time only**.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import BATCH_SIZE, CHUNK_SIZE, TOKENIZER
from llm_provider import get_embedding_provider
from logger import get_logger
from models import EmbeddedChunk, ModelMismatchError

if TYPE_CHECKING:
    from models import Chunk

log = get_logger(__name__)

# The local embedding models (BGE, nomic) have a 512-token window; a longer query is
# silently truncated by the model, so truncate it ourselves and say so.
_QUERY_TOKEN_LIMIT = CHUNK_SIZE


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Embed chunk texts in batches. Returns `EmbeddedChunk`s with `.embedding` and
    `.model_name` populated. `[]` in → `[]` out."""
    if not chunks:
        return []

    provider = get_embedding_provider()
    log.debug(
        "Provider acquired",
        extra={"model_name": provider.model_name, "embedding_dim": provider.embedding_dim},
    )

    out: list[EmbeddedChunk] = []
    for batch_index, batch in enumerate(_batched(chunks, BATCH_SIZE)):
        vectors = provider.embed([c.text for c in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != provider.embedding_dim:
                log.critical(
                    "Embedding dimension mismatch",
                    extra={
                        "expected_dim": provider.embedding_dim,
                        "actual_dim": len(vector),
                        "model_name": provider.model_name,
                    },
                )
                raise ModelMismatchError(
                    f"{provider.model_name} returned a {len(vector)}-d vector, expected "
                    f"{provider.embedding_dim}. Re-index from scratch after any model change."
                )
            out.append(
                EmbeddedChunk(chunk=chunk, embedding=list(vector), model_name=provider.model_name)
            )
        log.debug(
            "Embedding batch complete", extra={"batch_index": batch_index, "batch_size": len(batch)}
        )

    log.info(
        "All chunks embedded",
        extra={"total_chunks": len(out), "model_name": provider.model_name},
    )
    return out


def embed_query(query_text: str) -> list[float]:
    """Embed one query string for retrieval. Prepends the provider's `query_prefix` and
    truncates to the model's token window with a WARNING."""
    provider = get_embedding_provider()
    text = _truncate(query_text, provider.model_name)
    prefixed = provider.query_prefix + text
    log.debug(
        "Query embedded",
        extra={"query_length_chars": len(query_text), "backend_model": provider.model_name},
    )
    return list(provider.embed([prefixed])[0])


def _truncate(text: str, model_name: str) -> str:
    import tiktoken

    enc = tiktoken.get_encoding(TOKENIZER)
    tokens = enc.encode(text)
    if len(tokens) <= _QUERY_TOKEN_LIMIT:
        return text
    log.warning(
        "Query truncated (over token limit)",
        extra={
            "original_tokens": len(tokens),
            "max_tokens": _QUERY_TOKEN_LIMIT,
            "model_name": model_name,
        },
    )
    return enc.decode(tokens[:_QUERY_TOKEN_LIMIT])
