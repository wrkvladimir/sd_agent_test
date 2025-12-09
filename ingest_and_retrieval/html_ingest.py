import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from bs4 import BeautifulSoup

from .chunking import split_text_to_chunks
from .schemas import IngestRequest, ParagraphChunk


logger = logging.getLogger("ingest_and_retrieval.html")


def read_source_html(request: IngestRequest, request_id: str) -> str:
    if request.source_type == "file":
        if not request.path:
            msg = "path is required for source_type='file'"
            logger.error("request_id=%s %s", request_id, msg)
            raise ValueError(msg)
        path = request.path
        abs_path = os.path.join(os.getcwd(), path)
        logger.info(
            "request_id=%s reading HTML from file path=%s abs_path=%s",
            request_id,
            path,
            abs_path,
        )
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()

    if request.source_type == "inline_html":
        if not request.html:
            msg = "html is required for source_type='inline_html'"
            logger.error("request_id=%s %s", request_id, msg)
            raise ValueError(msg)
        logger.info("request_id=%s using inline HTML source_id=%s", request_id, request.source_id)
        return request.html

    msg = f"unsupported source_type='{request.source_type}'"
    logger.error("request_id=%s %s", request_id, msg)
    raise ValueError(msg)


def parse_html_to_chunks(
    html: str,
    source_document_id: str,
    request_id: str,
) -> Dict[str, Any]:
    warnings: List[str] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    soup = BeautifulSoup(html, "lxml")

    container = soup.find(attrs={"aria-label": "База знаний (чанки)"})
    if container is None:
        warning = "Не найден контейнер с aria-label='База знаний (чанки)'"
        logger.warning("request_id=%s %s", request_id, warning)
        return {
            "chunks": [],
            "warnings": [warning],
            "articles_total": 0,
            "paragraphs_total": 0,
        }

    source_label = container.get("aria-label", "База знаний (чанки)")

    articles = container.find_all("article", class_="kb-item")
    articles_total = len(articles)
    paragraphs_total = 0
    chunks: List[ParagraphChunk] = []

    logger.info(
        "request_id=%s found %d article.kb-item elements under source_label='%s'",
        request_id,
        articles_total,
        source_label,
    )

    for article in articles:
        article_source_id = article.get("data-id")
        article_source_date = article.get("data-date")

        title_el = article.find("h2")
        title = title_el.get_text(strip=True) if title_el else None

        paragraphs = article.find_all("p")
        for p in paragraphs:
            raw_text = p.get_text(" ", strip=True)
            if not raw_text:
                # Пустые абзацы нам не нужны, но это не ошибка.
                warning = f"Пустой <p> в article data-id={article_source_id!r}"
                warnings.append(warning)
                logger.debug("request_id=%s %s", request_id, warning)
                continue

            paragraphs_total += 1

            # Делим абзац на чанки по длине с перекрытием.
            paragraph_chunks = split_text_to_chunks(raw_text, request_id)
            for text_paragraph in paragraph_chunks:
                if title:
                    text_for_embedding = f"{title}. {text_paragraph}"
                else:
                    text_for_embedding = text_paragraph

                chunk = ParagraphChunk(
                    title=title,
                    text_paragraph=text_paragraph,
                    text_for_embedding=text_for_embedding,
                    source_id=article_source_id,
                    source_date=article_source_date,
                    source_label=source_label,
                    ingested_at=now_iso,
                    source_document_id=source_document_id,
                )
                chunks.append(chunk)

    logger.info(
        "request_id=%s parsed HTML into %d chunks from %d articles and %d paragraphs",
        request_id,
        len(chunks),
        articles_total,
        paragraphs_total,
    )

    for idx, chunk in enumerate(chunks[:3]):
        logger.debug(
            "request_id=%s sample_chunk idx=%d source_id=%s date=%s title=%r text_paragraph_preview=%r",
            request_id,
            idx,
            chunk.source_id,
            chunk.source_date,
            chunk.title,
            (chunk.text_paragraph[:150] + "...") if len(chunk.text_paragraph) > 150 else chunk.text_paragraph,
        )

    return {
        "chunks": chunks,
        "warnings": warnings,
        "articles_total": articles_total,
        "paragraphs_total": paragraphs_total,
    }
