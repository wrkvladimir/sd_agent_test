import logging
import os
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .schemas import ParagraphChunk


logger = logging.getLogger("ingest_and_retrieval.embeddings")

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model(request_id: str) -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        logger.info("request_id=%s loading embedding model %s", request_id, model_name)
        t0 = time.monotonic()
        _embedding_model = SentenceTransformer(model_name)
        dt = time.monotonic() - t0
        logger.info("request_id=%s embedding model loaded in %.3f seconds", request_id, dt)
    return _embedding_model


def embed_chunks(chunks: List[ParagraphChunk], request_id: str) -> List[List[float]]:
    texts = [chunk.text_for_embedding for chunk in chunks]
    if not texts:
        return []
    model = get_embedding_model(request_id)
    logger.info("request_id=%s start encoding %d texts with BGE-M3", request_id, len(texts))
    t0 = time.monotonic()
    vectors = model.encode(texts, batch_size=32, convert_to_numpy=True)
    dt = time.monotonic() - t0
    vector_dim = int(vectors.shape[1]) if hasattr(vectors, "shape") and vectors.size > 0 else 0
    logger.info(
        "request_id=%s finished encoding %d texts in %.3f seconds (vector_dim=%d)",
        request_id,
        len(texts),
        dt,
        vector_dim,
    )
    # Преобразуем в обычный список списков для совместимости с qdrant-client
    return [vec.tolist() for vec in vectors]


def embed_query(query: str, request_id: str) -> List[float]:
    """
    Считает эмбеддинг для текстового запроса.
    """
    model = get_embedding_model(request_id)
    logger.info("request_id=%s start encoding query with BGE-M3", request_id)
    t0 = time.monotonic()
    vector = model.encode([query], batch_size=1, convert_to_numpy=True)[0]
    dt = time.monotonic() - t0
    dim = int(vector.shape[0]) if hasattr(vector, "shape") else len(vector)
    logger.info(
        "request_id=%s finished encoding query in %.3f seconds (vector_dim=%d)",
        request_id,
        dt,
        dim,
    )
    return vector.tolist()
