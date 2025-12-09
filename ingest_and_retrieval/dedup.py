import logging
from typing import Dict, List, Set, Tuple

from .schemas import ParagraphChunk


logger = logging.getLogger("ingest_and_retrieval.dedup")


def _normalize_text(value: str) -> str:
    # Простая нормализация для поиска точных дублей.
    return " ".join(value.lower().split())


def deduplicate_in_memory(
    chunks: List[ParagraphChunk],
    request_id: str,
) -> Tuple[List[ParagraphChunk], List[str]]:
    """
    Ищет дубли среди чанков одного инжеста по text_paragraph.
    Оставляем первый, остальные выкидываем, пишем в warnings и логи.
    """
    seen: Set[str] = set()
    unique_chunks: List[ParagraphChunk] = []
    warnings: List[str] = []

    for idx, chunk in enumerate(chunks):
        key = _normalize_text(chunk.text_paragraph)
        if key in seen:
            msg = (
                f"Дубликат абзаца внутри источника: idx={idx}, "
                f"source_id={chunk.source_id!r}, title={chunk.title!r}"
            )
            warnings.append(msg)
            logger.info("request_id=%s %s", request_id, msg)
            continue
        seen.add(key)
        unique_chunks.append(chunk)

    if len(unique_chunks) != len(chunks):
        logger.info(
            "request_id=%s deduplicate_in_memory: removed %d duplicates (from %d to %d)",
            request_id,
            len(chunks) - len(unique_chunks),
            len(chunks),
            len(unique_chunks),
        )

    return unique_chunks, warnings


def build_existing_text_index(
    existing_points: List[Dict],
    request_id: str,
) -> Set[str]:
    """
    Строит индекс уже залитых абзацев по text_paragraph из payload.
    """
    index: Set[str] = set()
    for point in existing_points:
        payload = point.get("payload") or {}
        text = payload.get("text_paragraph")
        if not text:
            continue
        key = _normalize_text(str(text))
        index.add(key)

    logger.info(
        "request_id=%s existing_text_index size=%d",
        request_id,
        len(index),
    )
    return index


def deduplicate_against_existing(
    chunks: List[ParagraphChunk],
    existing_text_index: Set[str],
    request_id: str,
) -> Tuple[List[ParagraphChunk], List[str]]:
    """
    Ищет дубли по text_paragraph относительно уже залитых в коллекцию точек.
    Оставляем только те чанки, которых ещё нет в индексе.
    """
    if not existing_text_index:
        return chunks, []

    unique_chunks: List[ParagraphChunk] = []
    warnings: List[str] = []

    for idx, chunk in enumerate(chunks):
        key = _normalize_text(chunk.text_paragraph)
        if key in existing_text_index:
            msg = (
                f"Дубликат абзаца относительно коллекции: idx={idx}, "
                f"source_id={chunk.source_id!r}, title={chunk.title!r}"
            )
            warnings.append(msg)
            logger.info("request_id=%s %s", request_id, msg)
            continue
        unique_chunks.append(chunk)

    if len(unique_chunks) != len(chunks):
        logger.info(
            "request_id=%s deduplicate_against_existing: removed %d duplicates (from %d to %d)",
            request_id,
            len(chunks) - len(unique_chunks),
            len(chunks),
            len(unique_chunks),
        )

    return unique_chunks, warnings

