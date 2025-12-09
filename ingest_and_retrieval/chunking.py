import logging
import os
from typing import List


logger = logging.getLogger("ingest_and_retrieval.chunking")


def _get_chunk_params() -> tuple[int, int]:
    """
    Возвращает параметры чанкинга:
    максимальную длину чанка и перекрытие.

    Значения берутся из переменных окружения:
    - CHUNK_MAX_LENGTH (по умолчанию 512)
    - CHUNK_OVERLAP (по умолчанию 100)
    """
    try:
        max_length = int(os.getenv("CHUNK_MAX_LENGTH", "512"))
    except ValueError:
        max_length = 512

    try:
        overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    except ValueError:
        overlap = 100

    if max_length <= 0:
        max_length = 512
    if overlap < 0:
        overlap = 0
    if overlap >= max_length:
        overlap = max_length // 4

    return max_length, overlap


def split_text_to_chunks(text: str, request_id: str) -> List[str]:
    """
    Делит текст на чанки фиксированной длины с перекрытием.

    Мы работаем по символам, а не по токенам, чтобы
    не завязываться жёстко на конкретный токенайзер.
    """
    max_length, overlap = _get_chunk_params()

    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap

    logger.info(
        "request_id=%s split text into %d chunks (max_length=%d, overlap=%d, original_len=%d)",
        request_id,
        len(chunks),
        max_length,
        overlap,
        text_len,
    )

    return chunks

