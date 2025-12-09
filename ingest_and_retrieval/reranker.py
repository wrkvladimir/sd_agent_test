import logging
import os
import time
from typing import List, Optional

from sentence_transformers import CrossEncoder


logger = logging.getLogger("ingest_and_retrieval.reranker")

_reranker_model: Optional[CrossEncoder] = None


def get_reranker_model(request_id: str) -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
        logger.info("request_id=%s loading reranker model %s", request_id, model_name)
        t0 = time.monotonic()
        _reranker_model = CrossEncoder(model_name, max_length=512)
        dt = time.monotonic() - t0
        logger.info("request_id=%s reranker model loaded in %.3f seconds", request_id, dt)
    return _reranker_model


def rerank(
    query: str,
    texts: List[str],
    request_id: str,
) -> List[float]:
    """
    Считает rerank scores для пар (query, text).
    """
    if not texts:
        return []

    model = get_reranker_model(request_id)
    pairs = [(query, t) for t in texts]
    logger.info(
        "request_id=%s start reranking %d candidates with BGE-reranker",
        request_id,
        len(texts),
    )
    t0 = time.monotonic()
    scores = model.predict(pairs)
    dt = time.monotonic() - t0
    logger.info(
        "request_id=%s finished reranking %d candidates in %.3f seconds",
        request_id,
        len(texts),
        dt,
    )
    return [float(s) for s in scores]

