from __future__ import annotations

import asyncio
import logging
from typing import List

import httpx

from .config import settings
from .schemas import Chunk


logger = logging.getLogger("chat_app.retriever")


class KBRetriever:
    """
    Thin HTTP client that proxies search requests to the
    ingest-and-retrieval service via its public HTTP API.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.retrieval_url).rstrip("/")

    async def search(self, query: str) -> List[Chunk]:
        """
        Perform semantic search in the knowledge base using the underlying
        ingest-and-retrieval service.

        Мы не импортируем внутренние Pydantic-модели того сервиса и
        работаем только с его JSON-ответом.
        """
        payload = {"query": query, "with_debug": False}

        data = {}
        last_error: Exception | None = None
        # Ingest service does model warmup at startup; allow a longer retry window.
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(base_url=self._base_url, timeout=30.0) as client:
                    response = await client.post("/search", json=payload)
                    response.raise_for_status()
                    data = response.json()
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "kb_search_failed attempt=%d base_url=%s error=%r",
                    attempt,
                    self._base_url,
                    exc,
                )
                # Exponential backoff with cap.
                delay = min(8.0, 0.5 * (2 ** (attempt - 1)))
                await asyncio.sleep(delay)

        if last_error is not None:
            # Production fallback: treat as empty context to avoid failing /chat.
            return []

        chunks_raw = data.get("chunks", []) or []
        chunks: List[Chunk] = []
        for item in chunks_raw:
            chunks.append(
                Chunk(
                    id=item.get("id"),
                    text=item.get("text") or "",
                    metadata=item.get("metadata") or {},
                    score=item.get("score"),
                )
            )
        return chunks
