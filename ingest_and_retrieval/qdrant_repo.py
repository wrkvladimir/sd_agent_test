import logging
import os
import time
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from uuid import uuid4

from .schemas import ParagraphChunk


logger = logging.getLogger("ingest_and_retrieval.qdrant")

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client(request_id: str) -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        logger.info("request_id=%s creating Qdrant client url=%s", request_id, url)
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    recreate_collection: bool,
    request_id: str,
) -> None:
    if recreate_collection:
        logger.info(
            "request_id=%s recreating collection name=%s vector_size=%d",
            request_id,
            collection,
            vector_size,
        )
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    try:
        info = client.get_collection(collection_name=collection)
        existing_size = info.config.params.vectors.size  # type: ignore[attr-defined]
        logger.info(
            "request_id=%s collection %s already exists with vector_size=%d",
            request_id,
            collection,
            existing_size,
        )
        if existing_size != vector_size:
            msg = (
                f"Размер вектора в коллекции ({existing_size}) "
                f"не совпадает с размером эмбеддинга ({vector_size})"
            )
            logger.error("request_id=%s %s", request_id, msg)
            raise ValueError(msg)
    except Exception:  # noqa: BLE001
        logger.info(
            "request_id=%s collection %s does not exist, creating with vector_size=%d",
            request_id,
            collection,
            vector_size,
        )
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: List[ParagraphChunk],
    vectors: List[List[float]],
    request_id: str,
) -> int:
    if not chunks:
        return 0

    if len(chunks) != len(vectors):
        msg = f"Количество чанков ({len(chunks)}) не совпадает с количеством векторов ({len(vectors)})"
        logger.error("request_id=%s %s", request_id, msg)
        raise ValueError(msg)

    points: List[PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid4())
        payload = {
            "title": chunk.title,
            "text_paragraph": chunk.text_paragraph,
            "source_id": chunk.source_id,
            "source_date": chunk.source_date,
            "source_label": chunk.source_label,
            "ingested_at": chunk.ingested_at,
            "source_document_id": chunk.source_document_id,
        }
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    logger.info(
        "request_id=%s upserting %d points into collection=%s",
        request_id,
        len(points),
        collection,
    )
    t0 = time.monotonic()
    client.upsert(
        collection_name=collection,
        points=points,
        wait=True,
    )
    dt = time.monotonic() - t0
    logger.info(
        "request_id=%s upsert completed for %d points in %.3f seconds",
        request_id,
        len(points),
        dt,
    )
    return len(points)


def search_points(
    client: QdrantClient,
    collection: str,
    query_vector: List[float],
    limit: int,
    request_id: str,
) -> List[dict]:
    logger.info(
        "request_id=%s searching in collection=%s limit=%d",
        request_id,
        collection,
        limit,
    )
    t0 = time.monotonic()
    # Используем рекомендованный Query API: query_points
    resp = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    results = resp.points

    dt = time.monotonic() - t0
    logger.info(
        "request_id=%s search completed in %.3f seconds, got %d candidates",
        request_id,
        dt,
        len(results),
    )
    points: List[dict] = []
    for scored in results:
        points.append(
            {
                "id": scored.id,
                "payload": scored.payload or {},
                "score": float(scored.score) if scored.score is not None else None,
            }
        )
    return points


def fetch_all_points(
    client: QdrantClient,
    collection: str,
    request_id: str,
    batch_size: int = 256,
) -> list[dict]:
    logger.info(
        "request_id=%s fetching all points from collection=%s batch_size=%d",
        request_id,
        collection,
        batch_size,
    )
    all_points: list[dict] = []
    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_offset,
            with_vectors=False,
        )
        if not records:
            break
        for rec in records:
            all_points.append(
                {
                    "id": rec.id,
                    "payload": rec.payload,
                }
            )
        if next_offset is None:
            break

    logger.info(
        "request_id=%s fetched %d points from collection=%s",
        request_id,
        len(all_points),
        collection,
    )
    return all_points


def clear_collection(
    client: QdrantClient,
    collection: str,
    request_id: str,
) -> int:
    try:
        count_result = client.count(collection_name=collection, exact=True)
        existing_points = int(count_result.count)
    except Exception:  # noqa: BLE001
        logger.info(
            "request_id=%s collection %s does not exist or count failed, nothing to clear",
            request_id,
            collection,
        )
        return 0

    try:
        info = client.get_collection(collection_name=collection)
        vector_size = info.config.params.vectors.size  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "request_id=%s failed to read collection config for %s: %s",
            request_id,
            collection,
            exc,
        )
        raise

    logger.info(
        "request_id=%s clearing collection=%s, existing_points=%d, vector_size=%d",
        request_id,
        collection,
        existing_points,
        vector_size,
    )
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    return existing_points


def get_collection_summary(
    client: QdrantClient,
    collection: str,
    request_id: str,
) -> dict:
    """
    Возвращает краткую информацию о коллекции:
    количество точек, размер вектора и метрику расстояния.
    """
    try:
        count_result = client.count(collection_name=collection, exact=True)
        total_points = int(count_result.count)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "request_id=%s failed to count points in %s: %s",
            request_id,
            collection,
            exc,
        )
        total_points = 0

    vector_size: int | None = None
    distance: str | None = None
    try:
        info = client.get_collection(collection_name=collection)
        params = info.config.params.vectors  # type: ignore[attr-defined]
        # В зависимости от версии клиента vectors может быть объектом или dict.
        if isinstance(params, dict):
            vector_size = int(params.get("size", 0)) or None
            distance = str(params.get("distance")) if params.get("distance") else None
        else:
            vector_size = getattr(params, "size", None)
            distance = getattr(params, "distance", None)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "request_id=%s failed to get collection config for %s: %s",
            request_id,
            collection,
            exc,
        )

    logger.info(
        "request_id=%s collection_summary name=%s total_points=%d vector_size=%s distance=%s",
        request_id,
        collection,
        total_points,
        vector_size,
        distance,
    )

    return {
        "total_points": total_points,
        "vector_size": vector_size,
        "distance": distance,
    }
