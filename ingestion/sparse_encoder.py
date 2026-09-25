"""BM25 sparse vectors for chunk texts (indexing) and for a query string.

Wraps fastembed's `Qdrant/bm25`: tokenise, drop stopwords, stem, hash each term to an id.
Documents carry the BM25 term-frequency part (k, b and `BM25_AVG_LEN` baked in); queries
carry weight 1.0 per term. The IDF part is applied by Qdrant at search time — the collection's
`bm25` sparse vector is created with `Modifier.IDF` — so it always reflects the current corpus
and never needs recomputing when chunks are added or removed.

The model is a stemmer and a stopword list, not a neural network, but loading it still reads
files (downloaded from Hugging Face on first use), so it is cached per process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client import models

from config import BM25_AVG_LEN
from logger import get_logger

if TYPE_CHECKING:
    from fastembed.sparse.bm25 import Bm25

log = get_logger(__name__)

BM25_MODEL = "Qdrant/bm25"

_model: Bm25 | None = None


def _get_model() -> Bm25:
    global _model
    if _model is None:
        from fastembed.sparse.bm25 import Bm25

        _model = Bm25(BM25_MODEL, avg_len=BM25_AVG_LEN)
        log.debug("BM25 encoder loaded", extra={"model": BM25_MODEL, "avg_len": BM25_AVG_LEN})
    return _model


def _to_sparse(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )


def encode_documents(texts: list[str]) -> list[models.SparseVector]:
    """One BM25 document vector per text. `[]` in → `[]` out."""
    if not texts:
        return []
    return [_to_sparse(e) for e in _get_model().embed(texts)]


def encode_query(text: str) -> models.SparseVector:
    """The query's terms, weight 1.0 each — Qdrant supplies the IDF."""
    return _to_sparse(next(iter(_get_model().query_embed(text))))


def token_length(text: str) -> int:
    """BM25's view of a document's length: terms left after stopword removal and stemming.
    The mean over the corpus is what `BM25_AVG_LEN` should be."""
    from fastembed.common.utils import remove_non_alphanumeric

    model = _get_model()
    return len(model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text))))


def _reset_for_tests() -> None:
    global _model
    _model = None
