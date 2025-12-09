from __future__ import annotations

import logging
from typing import Any, Dict, List

import anyio
from openai import OpenAI

from .config import settings


logger = logging.getLogger("chat_app.llm")


class LLMClient:
    """
    Thin wrapper around OpenAI-compatible chat API (e.g. OpenRouter).

    Использует настройки:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL
    - LLM_MODEL
    и делает один вызов chat-completions для получения ответа.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        api_key = api_key or settings.llm_api_key
        if not api_key:
            raise RuntimeError("LLM API key (OPENAI_API_KEY) is not set")

        base_url = base_url or settings.llm_base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model or settings.llm_model

    async def complete_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Execute a chat completion request and return assistant text.

        Вызов выполняется в отдельном потоке, чтобы не блокировать event loop.
        """

        def _call() -> Any:
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
            )

        try:
            response = await anyio.to_thread.run_sync(_call)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM request failed: %s", exc)
            raise

        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        return content or ""
