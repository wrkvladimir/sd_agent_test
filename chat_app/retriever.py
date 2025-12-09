from __future__ import annotations

from typing import List

import httpx

from .config import settings
from .schemas import Chunk


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

        async with httpx.AsyncClient(base_url=self._base_url) as client:
            response = await client.post("/search", json=payload)
            response.raise_for_status()
            data = response.json()

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
