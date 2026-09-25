"""The sole gateway between the application and Qdrant.

Collection creation, upsert of embedded chunks, top-k vector search, and delete-by-URL.
`search` returns Qdrant's raw top-k by score — `MIN_SCORE_THRESHOLD` and
`MAX_CHUNKS_PER_ARTICLE` are `query/retriever.py`'s job, not this module's.

The embedding model that built the collection is recorded on a sentinel point (Qdrant has
no first-class collection metadata). A first `upsert` writes it; every later `upsert`
checks it and raises `ModelMismatchError` on a mismatch. `drop_collection()` clears it —
that is the sanctioned way to change embedding models.

Each point also carries a BM25 sparse vector named `bm25` (IDF applied by Qdrant), which
`search_sparse` queries for hybrid retrieval. The dense vector stays the unnamed default, so
dense search is unchanged. Collections created before hybrid search have no `bm25` config;
`add_sparse_vectors()` migrates them in place (`monthly_job.py --add-sparse`).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_HOST,
    QDRANT_IN_MEMORY,
    QDRANT_PATH,
    QDRANT_PORT,
    UPSERT_BATCH_SIZE,
)
from ingestion.sparse_encoder import encode_documents, encode_query, token_length
from logger import get_logger
from models import ModelMismatchError, SearchResult

if TYPE_CHECKING:
    from models import EmbeddedChunk

log = get_logger(__name__)

# Index tuning. Not in config.py: DISTANCE_METRIC would drag qdrant_client into every
# module's import of config, and these four are implementation detail, not user knobs.
DISTANCE_METRIC = models.Distance.COSINE
ON_DISK_PAYLOAD = True
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
SPARSE_VECTOR_NAME = "bm25"
_DENSE = ""  # Qdrant's name for the unnamed default vector
_SCROLL_PAGE = 500

_POINT_NAMESPACE = uuid.UUID("6f6b1b9e-0000-4000-8000-000000000001")
_SENTINEL_ID = str(uuid.uuid5(_POINT_NAMESPACE, f"{COLLECTION_NAME}::sentinel"))
_NOT_SENTINEL = models.FieldCondition(key="_sentinel", match=models.MatchValue(value=True))


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class VectorStoreConnectionError(Exception):
    """Qdrant is unreachable at the configured host:port."""


class VectorStore:
    def __init__(self) -> None:
        try:
            if QDRANT_IN_MEMORY:
                self.client = QdrantClient(location=":memory:")
            elif QDRANT_PATH:  # pragma: no cover - embedded on-disk store
                self.client = QdrantClient(path=QDRANT_PATH)
            else:  # pragma: no cover - needs a running Qdrant
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self._init_collection()
        except (ConnectionError, OSError, ApiException) as exc:  # pragma: no cover - needs a server
            log.critical(
                "Qdrant unreachable",
                extra={"host": QDRANT_HOST, "port": QDRANT_PORT, "error_type": type(exc).__name__},
                exc_info=True,
            )
            raise VectorStoreConnectionError(f"{QDRANT_HOST}:{QDRANT_PORT}") from exc

    # ── collection lifecycle ───────────────────────────────────────────────

    def _init_collection(self) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            info = self.client.get_collection(COLLECTION_NAME)
            actual_dim = info.config.params.vectors.size
            if actual_dim != EMBEDDING_DIM:
                log.critical(
                    "Model mismatch detected",
                    extra={
                        "collection_name": COLLECTION_NAME,
                        "stored_dim": actual_dim,
                        "configured_dim": EMBEDDING_DIM,
                    },
                )
                raise ModelMismatchError(
                    f"Collection {COLLECTION_NAME!r} has {actual_dim}-d vectors, config "
                    f"EMBEDDING_DIM is {EMBEDDING_DIM}. Change the model, then re-index "
                    f"from scratch (monthly_job.py --reset)."
                )
            log.debug(
                "Collection already exists",
                extra={"collection_name": COLLECTION_NAME, "point_count": self.count()},
            )
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM, distance=DISTANCE_METRIC, on_disk=ON_DISK_PAYLOAD
            ),
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        )
        log.info(
            "Creating new collection",
            extra={
                "collection_name": COLLECTION_NAME,
                "dim": EMBEDDING_DIM,
                "distance_metric": DISTANCE_METRIC.value,
            },
        )

    def drop_collection(self) -> None:
        """`--reset` only. Drops every point and the collection, then recreates it empty
        with the configured dim recorded again — a half-reset that leaves no collection
        turns the next `upsert` into an obscure 404.
        """
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        self._init_collection()
        log.info("Collection dropped and recreated", extra={"collection_name": COLLECTION_NAME})

    def has_sparse(self) -> bool:
        """Whether the collection has the `bm25` sparse vector that hybrid search needs."""
        sparse = self.client.get_collection(COLLECTION_NAME).config.params.sparse_vectors
        return bool(sparse) and SPARSE_VECTOR_NAME in sparse

    def add_sparse_vectors(self) -> dict:
        """Give every point a BM25 sparse vector computed from its stored `text`.

        A collection created before hybrid search has no `bm25` config, and embedded Qdrant
        cannot add one to an existing collection — so this reads every point (dense vector,
        payload, id) into memory, recreates the collection with the sparse config, and writes
        them all back with a sparse vector added. Dense vectors, ids and payloads are copied
        unchanged, so dense search returns exactly what it did before. Take a copy of the
        storage first (`monthly_job.py --add-sparse` does): a crash between the delete and the
        re-upsert loses the index.

        Already migrated → only the sparse vectors are recomputed (e.g. after changing
        `BM25_AVG_LEN`). Idempotent either way.
        """
        points = self._scroll_all(with_vectors=True)
        real = [p for p in points if not (p.payload or {}).get("_sentinel")]
        texts = [(p.payload or {}).get("text", "") for p in real]
        lengths = [token_length(t) for t in texts]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        sparse = encode_documents(texts)
        migrated = not self.has_sparse()

        if migrated:
            log.warning(
                "Recreating collection with a BM25 sparse vector",
                extra={"collection_name": COLLECTION_NAME, "point_count": len(points)},
            )
            by_id = {p.id: sv for p, sv in zip(real, sparse, strict=True)}
            new_points = []
            for p in points:
                vector: dict = {_DENSE: _dense_of(p)}
                if p.id in by_id:
                    vector[SPARSE_VECTOR_NAME] = by_id[p.id]
                new_points.append(models.PointStruct(id=p.id, vector=vector, payload=p.payload))
            self.client.delete_collection(COLLECTION_NAME)
            self._init_collection()
            for start in range(0, len(new_points), UPSERT_BATCH_SIZE):
                self.client.upsert(
                    COLLECTION_NAME, points=new_points[start : start + UPSERT_BATCH_SIZE], wait=True
                )
        else:
            updates = [
                models.PointVectors(id=p.id, vector={SPARSE_VECTOR_NAME: sv})
                for p, sv in zip(real, sparse, strict=True)
            ]
            for start in range(0, len(updates), UPSERT_BATCH_SIZE):
                self.client.update_vectors(
                    COLLECTION_NAME, points=updates[start : start + UPSERT_BATCH_SIZE], wait=True
                )

        after = self.client.count(COLLECTION_NAME, exact=True).count
        if after != len(points):
            log.critical(
                "Sparse migration lost points",
                extra={"before": len(points), "after": after},
            )
            raise RuntimeError(
                f"Collection had {len(points)} points before the BM25 migration and {after} "
                "after. Restore the storage copy taken before the run."
            )
        result = {
            "migrated": migrated,
            "points": after,
            "sparse_vectors_written": len(sparse),
            "measured_avg_len": round(avg_len, 1),
        }
        log.info("BM25 sparse vectors written", extra=result)
        return result

    def _scroll_all(self, with_vectors: bool) -> list[models.Record]:
        records: list[models.Record] = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                COLLECTION_NAME,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            records.extend(page)
            if offset is None:
                return records

    # ── model sentinel ────────────────────────────────────────────────────

    def recorded_model(self) -> str | None:
        """The embedding model name that built this collection, or None if nothing has
        been upserted yet. `query/retriever.py` reads this to guard query-time embedding.
        """
        points = self.client.retrieve(COLLECTION_NAME, ids=[_SENTINEL_ID], with_payload=True)
        if not points:
            return None
        return points[0].payload.get("model_name")

    def _record_model(self, model_name: str) -> None:
        self.client.upsert(
            COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=_SENTINEL_ID,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"_sentinel": True, "model_name": model_name},
                )
            ],
            wait=True,
        )

    # ── writes ────────────────────────────────────────────────────────────

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        if not embedded_chunks:
            return 0

        model_name = embedded_chunks[0].model_name
        recorded = self.recorded_model()
        if recorded is None:
            self._record_model(model_name)
        elif recorded != model_name:
            log.critical(
                "Model mismatch detected",
                extra={
                    "collection_name": COLLECTION_NAME,
                    "stored_model": recorded,
                    "configured_model": model_name,
                },
            )
            raise ModelMismatchError(
                f"Collection {COLLECTION_NAME!r} was built with {recorded!r}; this upsert "
                f"used {model_name!r}. Re-index from scratch (monthly_job.py --reset)."
            )

        if self.has_sparse():
            sparse = encode_documents([ec.chunk.text for ec in embedded_chunks])
            vectors = [
                {_DENSE: list(ec.embedding), SPARSE_VECTOR_NAME: sv}
                for ec, sv in zip(embedded_chunks, sparse, strict=True)
            ]
        else:  # a pre-hybrid collection: dense only until `add_sparse_vectors()` migrates it
            vectors = [list(ec.embedding) for ec in embedded_chunks]

        points = [
            models.PointStruct(
                id=_point_id(ec.chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": ec.chunk.chunk_id,
                    "article_url": ec.chunk.article_url,
                    "article_title": ec.chunk.article_title,
                    "published_at": (
                        ec.chunk.published_at.isoformat()
                        if ec.chunk.published_at is not None
                        else None
                    ),
                    "tags": ec.chunk.tags,
                    "text": ec.chunk.text,
                    "content_type": ec.chunk.content_type,
                    "chunk_index": ec.chunk.chunk_index,
                    "total_chunks": ec.chunk.total_chunks,
                    "model_name": ec.model_name,
                },
            )
            for ec, vector in zip(embedded_chunks, vectors, strict=True)
        ]

        upserted = 0
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            self.client.upsert(COLLECTION_NAME, points=batch, wait=True)
            upserted += len(batch)
            log.debug(
                "Upsert batch complete",
                extra={"batch_index": start // UPSERT_BATCH_SIZE, "upserted_count": len(batch)},
            )
        return upserted

    def delete_by_url(self, article_url: str) -> int:
        flt = models.Filter(
            must=[
                models.FieldCondition(key="article_url", match=models.MatchValue(value=article_url))
            ]
        )
        before = self.client.count(COLLECTION_NAME, count_filter=flt, exact=True).count
        self.client.delete(
            COLLECTION_NAME, points_selector=models.FilterSelector(filter=flt), wait=True
        )
        if before == 0:
            log.warning("Delete matched 0 points", extra={"url": article_url})
        else:
            log.info("Delete by URL complete", extra={"url": article_url, "deleted_count": before})
        return before

    # ── reads ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 6,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        query_filter = self._build_filter(filters)
        response = self.client.query_points(
            COLLECTION_NAME,
            query=list(query_vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = [self._to_result(p) for p in response.points]
        if not results:
            log.debug("Search returned 0 results", extra={"top_k": top_k, "filters": filters})
        return results

    def search_sparse(
        self,
        query_text: str,
        top_k: int = 6,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """BM25 top-k for `query_text`. `.score` is a BM25 score — a different scale from
        cosine, so callers must fuse by rank, never by score. Requires `has_sparse()`."""
        response = self.client.query_points(
            COLLECTION_NAME,
            query=encode_query(query_text),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        results = [self._to_result(p) for p in response.points]
        if not results:
            log.debug(
                "Sparse search returned 0 results", extra={"top_k": top_k, "filters": filters}
            )
        return results

    def score_by_ids(self, query_vector: list[float], chunk_ids: list[str]) -> dict[str, float]:
        """Cosine similarity of `query_vector` to each of `chunk_ids`, by chunk id. Lets a
        chunk that only BM25 found carry the same kind of score as everything else."""
        if not chunk_ids:
            return {}
        response = self.client.query_points(
            COLLECTION_NAME,
            query=list(query_vector),
            limit=len(chunk_ids),
            query_filter=models.Filter(
                must=[models.HasIdCondition(has_id=[_point_id(c) for c in chunk_ids])],
                must_not=[_NOT_SENTINEL],
            ),
            with_payload=["chunk_id"],
        )
        return {(p.payload or {}).get("chunk_id", ""): p.score for p in response.points}

    @staticmethod
    def _build_filter(filters: dict | None) -> models.Filter:
        must: list[models.Condition] = []
        if filters:
            if tags := filters.get("tags"):
                must.append(
                    models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags)))
                )
            if (after := filters.get("date_after")) is not None:
                must.append(
                    models.FieldCondition(
                        key="published_at", range=models.DatetimeRange(gte=_as_iso(after))
                    )
                )
            if (before := filters.get("date_before")) is not None:
                must.append(
                    models.FieldCondition(
                        key="published_at", range=models.DatetimeRange(lte=_as_iso(before))
                    )
                )
            if ctype := filters.get("content_type"):
                must.append(
                    models.FieldCondition(key="content_type", match=models.MatchValue(value=ctype))
                )
        return models.Filter(must=must or None, must_not=[_NOT_SENTINEL])

    @staticmethod
    def _to_result(point: models.ScoredPoint) -> SearchResult:
        payload = point.payload or {}
        published_raw = payload.get("published_at")
        return SearchResult(
            score=point.score,
            text=payload.get("text", ""),
            chunk_id=payload.get("chunk_id", ""),
            article_url=payload.get("article_url", ""),
            article_title=payload.get("article_title", ""),
            published_at=(datetime.fromisoformat(published_raw) if published_raw else None),
            tags=payload.get("tags", []),
            content_type=payload.get("content_type", ""),
            chunk_index=payload.get("chunk_index", 0),
        )

    def count(self) -> int:
        """Real points only — the model sentinel is excluded."""
        return self.client.count(
            COLLECTION_NAME,
            count_filter=models.Filter(must_not=[_NOT_SENTINEL]),
            exact=True,
        ).count

    def stats(self) -> dict:
        info = self.client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "points": self.count(),
            "model_name": self.recorded_model(),
            "dim": info.config.params.vectors.size,
            "status": str(info.status),
        }


def _as_iso(value: datetime | str) -> str:
    return value.isoformat() if isinstance(value, datetime) else value


def _dense_of(record: models.Record) -> list[float]:
    """A record's dense vector, whether it came back bare (a dense-only collection) or keyed
    by name (a collection that also has sparse vectors)."""
    vector = record.vector
    return list(vector[_DENSE] if isinstance(vector, dict) else vector)
