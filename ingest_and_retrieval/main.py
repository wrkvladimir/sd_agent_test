import logging
import os
from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI

from .embeddings import embed_chunks, embed_query
from .dedup import (
    build_existing_text_index,
    deduplicate_against_existing,
    deduplicate_in_memory,
)
from .html_ingest import parse_html_to_chunks, read_source_html
from .qdrant_repo import (
    clear_collection,
    ensure_collection,
    fetch_all_points,
    get_collection_summary,
    get_qdrant_client,
    search_points,
    upsert_chunks,
)
from .schemas import (
    AllDataResponse,
    Chunk,
    ClearDataResponse,
    IngestRequest,
    IngestResponse,
    ParagraphChunk,
    SearchDebugInfo,
    SearchRequest,
    SearchResponse,
)
from .reranker import get_reranker_model, rerank


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = FastAPI(title="Ingest and Retrieval Service")

logger = logging.getLogger("ingest_and_retrieval.api")


@app.on_event("startup")
async def startup_warmup() -> None:
    """
    Загружаем модели эмбеддингов и reranker при старте сервиса,
    чтобы первый запрос не блокировался на их инициализации.
    """
    request_id = str(uuid4())
    logger.info("request_id=%s startup warmup started", request_id)

    # Тёплый старт для embedding-модели.
    try:
        embed_query("warmup", request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "request_id=%s failed to warm up embedding model: %s",
            request_id,
            exc,
        )

    # Тёплый старт для reranker'а, если он включён.
    try:
        reranker_enabled = os.getenv("RERANKER_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if reranker_enabled:
            get_reranker_model(request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "request_id=%s failed to warm up reranker model: %s",
            request_id,
            exc,
        )

    logger.info("request_id=%s startup warmup completed", request_id)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    request_id = str(uuid4())
    logger.info(
        "request_id=%s ingest started source_type=%s source_id=%s collection=%s recreate_collection=%s",
        request_id,
        request.source_type,
        request.source_id,
        request.collection,
        request.recreate_collection,
    )

    try:
        html = read_source_html(request, request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("request_id=%s failed to read source: %s", request_id, exc)
        return IngestResponse(
            status="error",
            collection=request.collection,
            source_id=request.source_id,
            articles_total=0,
            paragraphs_total=0,
            indexed_points=0,
            warnings=[f"Ошибка чтения источника: {exc}"],
        )

    parsed = parse_html_to_chunks(
        html=html,
        source_document_id=request.source_id,
        request_id=request_id,
    )
    chunks: List[ParagraphChunk] = parsed["chunks"]
    warnings = parsed["warnings"]
    articles_total = parsed["articles_total"]
    paragraphs_total = parsed["paragraphs_total"]

    # Дедупликация внутри одного инжеста (одного документа).
    chunks, dedup_warnings = deduplicate_in_memory(chunks=chunks, request_id=request_id)
    warnings.extend(dedup_warnings)

    # Дедупликация относительно уже существующих точек в коллекции.
    client = get_qdrant_client(request_id)
    try:
        existing_points = fetch_all_points(
            client=client,
            collection=request.collection,
            request_id=request_id,
        )
        existing_index = build_existing_text_index(existing_points, request_id)
        chunks, dedup_existing_warnings = deduplicate_against_existing(
            chunks=chunks,
            existing_text_index=existing_index,
            request_id=request_id,
        )
        warnings.extend(dedup_existing_warnings)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "request_id=%s failed to load existing points for deduplication: %s",
            request_id,
            exc,
        )

    if not chunks:
        logger.warning(
            "request_id=%s no chunks parsed, skipping embedding and upsert",
            request_id,
        )
        return IngestResponse(
            status="ok",
            collection=request.collection,
            source_id=request.source_id,
            articles_total=articles_total,
            paragraphs_total=paragraphs_total,
            indexed_points=0,
            warnings=warnings,
        )

    try:
        vectors = embed_chunks(chunks, request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("request_id=%s failed to encode embeddings: %s", request_id, exc)
        warnings.append(f"Ошибка при вычислении эмбеддингов: {exc}")
        return IngestResponse(
            status="error",
            collection=request.collection,
            source_id=request.source_id,
            articles_total=articles_total,
            paragraphs_total=paragraphs_total,
            indexed_points=0,
            warnings=warnings,
        )

    vector_size = len(vectors[0]) if vectors else 0
    client = get_qdrant_client(request_id)

    try:
        ensure_collection(
            client=client,
            collection=request.collection,
            vector_size=vector_size,
            recreate_collection=request.recreate_collection,
            request_id=request_id,
        )
        indexed_points = upsert_chunks(
            client=client,
            collection=request.collection,
            chunks=chunks,
            vectors=vectors,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("request_id=%s failed to upsert into Qdrant: %s", request_id, exc)
        warnings.append(f"Ошибка записи в Qdrant: {exc}")
        return IngestResponse(
            status="error",
            collection=request.collection,
            source_id=request.source_id,
            articles_total=articles_total,
            paragraphs_total=paragraphs_total,
            indexed_points=0,
            warnings=warnings,
        )

    logger.info(
        "request_id=%s ingest completed collection=%s source_id=%s articles_total=%d paragraphs_total=%d indexed_points=%d",
        request_id,
        request.collection,
        request.source_id,
        articles_total,
        paragraphs_total,
        indexed_points,
    )

    return IngestResponse(
        status="ok",
        collection=request.collection,
        source_id=request.source_id,
        articles_total=articles_total,
        paragraphs_total=paragraphs_total,
        indexed_points=indexed_points,
        warnings=warnings,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    request_id = str(uuid4())

    default_collection = os.getenv("SEARCH_COLLECTION", "support_knowledge")
    collection = default_collection
    top_k = int(os.getenv("SEARCH_TOP_K", "5"))
    search_limit = int(os.getenv("SEARCH_LIMIT", str(top_k * 3)))
    score_threshold = float(os.getenv("SEARCH_SCORE_THRESHOLD", "0.0"))
    reranker_enabled = os.getenv("RERANKER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

    logger.info(
        "request_id=%s search started collection=%s top_k=%d limit=%d threshold=%s reranker_enabled=%s",
        request_id,
        collection,
        top_k,
        search_limit,
        score_threshold,
        reranker_enabled,
    )

    client = get_qdrant_client(request_id)

    try:
        query_vector = embed_query(request.query, request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("request_id=%s failed to encode query: %s", request_id, exc)
        debug = SearchDebugInfo(
            used_threshold=score_threshold,
            initial_candidates_count=0,
            after_threshold_count=0,
            vector_scores=[],
            rerank_scores=[],
        )
        return SearchResponse(chunks=[], debug=debug if request.with_debug else None)

    try:
        candidates = search_points(
            client=client,
            collection=collection,
            query_vector=query_vector,
            limit=search_limit,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("request_id=%s failed to search in Qdrant: %s", request_id, exc)
        debug = SearchDebugInfo(
            used_threshold=score_threshold,
            initial_candidates_count=0,
            after_threshold_count=0,
            vector_scores=[],
            rerank_scores=[],
        )
        return SearchResponse(chunks=[], debug=debug if request.with_debug else None)

    initial_count = len(candidates)

    # Фильтрация по порогу векторного скора.
    filtered = []
    vector_scores: List[float] = []
    for item in candidates:
        score = item.get("score")
        if score is None:
            continue
        if score < score_threshold:
            continue
        filtered.append(item)
        vector_scores.append(score)

    after_threshold_count = len(filtered)

    if not filtered:
        logger.info(
            "request_id=%s search: no candidates passed threshold (initial=%d threshold=%s)",
            request_id,
            initial_count,
            score_threshold,
        )
        debug = SearchDebugInfo(
            used_threshold=score_threshold,
            initial_candidates_count=initial_count,
            after_threshold_count=after_threshold_count,
            vector_scores=vector_scores,
            rerank_scores=[],
        )
        return SearchResponse(chunks=[], debug=debug if request.with_debug else None)

    # Готовим тексты для reranker'а — только text_paragraph.
    texts_for_rerank: List[str] = []
    for item in filtered:
        payload = item.get("payload") or {}
        text_paragraph = payload.get("text_paragraph") or ""
        texts_for_rerank.append(str(text_paragraph))

    rerank_scores: List[float] = []
    if reranker_enabled:
        try:
            rerank_scores = rerank(
                query=request.query,
                texts=texts_for_rerank,
                request_id=request_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("request_id=%s failed to rerank candidates: %s", request_id, exc)
            # Если reranker упал, вернём результаты только по векторному скору.
            rerank_scores = []

    # Объединяем кандидатов с rerank-скорами.
    combined = []
    for idx, item in enumerate(filtered):
        score = rerank_scores[idx] if rerank_scores and idx < len(rerank_scores) else item.get("score")
        combined.append((item, float(score) if score is not None else 0.0))

    # Сортируем по финальному скору и берём top_k.
    combined.sort(key=lambda x: x[1], reverse=True)
    combined = combined[:top_k]

    chunks: List[Chunk] = []
    final_rerank_scores: List[float] = []
    for item, final_score in combined:
        payload = item.get("payload") or {}
        text_paragraph = payload.get("text_paragraph") or ""
        metadata = {
            "title": payload.get("title"),
            "source_id": payload.get("source_id"),
            "source_date": payload.get("source_date"),
            "source_label": payload.get("source_label"),
            "source_document_id": payload.get("source_document_id"),
            "vector_score": item.get("score"),
        }
        chunks.append(
            Chunk(
                id=item.get("id"),
                text=str(text_paragraph),
                metadata=metadata,
                score=final_score,
            )
        )
        final_rerank_scores.append(final_score)

    # Логируем пару примеров кандидатов для отладки.
    for idx, (item, final_score) in enumerate(combined[:3]):
        payload = item.get("payload") or {}
        logger.debug(
            "request_id=%s result_sample idx=%d id=%s title=%r vector_score=%s final_score=%s",
            request_id,
            idx,
            item.get("id"),
            payload.get("title"),
            item.get("score"),
            final_score,
        )

    logger.info(
        "request_id=%s search completed: initial=%d after_threshold=%d final=%d",
        request_id,
        initial_count,
        after_threshold_count,
        len(chunks),
    )

    debug = SearchDebugInfo(
        used_threshold=score_threshold,
        initial_candidates_count=initial_count,
        after_threshold_count=after_threshold_count,
        vector_scores=vector_scores,
        rerank_scores=final_rerank_scores,
    )

    return SearchResponse(chunks=chunks, debug=debug if request.with_debug else None)


@app.get("/data", response_model=AllDataResponse)
async def get_all_data(collection: str = "support_knowledge") -> AllDataResponse:
    request_id = str(uuid4())
    logger.info(
        "request_id=%s get_all_data started collection=%s",
        request_id,
        collection,
    )
    client = get_qdrant_client(request_id)
    try:
        summary = get_collection_summary(client=client, collection=collection, request_id=request_id)
        points = fetch_all_points(client=client, collection=collection, request_id=request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("request_id=%s failed to fetch data from Qdrant: %s", request_id, exc)
        # В случае ошибки вернём пустой список, детали смотрим в логах.
        return AllDataResponse(
            collection=collection,
            total_points=0,
            vector_size=None,
            distance=None,
            points=[],
        )

    return AllDataResponse(
        collection=collection,
        total_points=summary["total_points"],
        vector_size=summary["vector_size"],
        distance=summary["distance"],
        points=points,
    )


@app.post("/clear_data", response_model=ClearDataResponse)
async def clear_data(collection: str = "support_knowledge") -> ClearDataResponse:
    request_id = str(uuid4())
    logger.info(
        "request_id=%s clear_data started collection=%s",
        request_id,
        collection,
    )
    client = get_qdrant_client(request_id)
    try:
        deleted = clear_collection(client=client, collection=collection, request_id=request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("request_id=%s failed to clear collection %s: %s", request_id, collection, exc)
        # Если не удалось почистить, считаем, что ничего не удалили; детали в логах.
        return ClearDataResponse(collection=collection, deleted_points=0)

    logger.info(
        "request_id=%s clear_data completed collection=%s deleted_points=%d",
        request_id,
        collection,
        deleted,
    )
    return ClearDataResponse(collection=collection, deleted_points=deleted)
