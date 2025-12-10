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

        logger.info(
            "llm_client_init model=%s base_url=%s",
            self._model,
            base_url,
        )

    async def complete_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Execute a chat completion request and return assistant text.

        Вызов выполняется в отдельном потоке, чтобы не блокировать event loop.
        """

        def _call() -> Any:
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            # Логируем полный payload без чувствительных данных (api_key скрыт внутри клиента),
            # чтобы можно было воспроизвести запрос вручную.
            try:
                import json

                logger.info(
                    "llm_chat_request model=%s payload=%s",
                    self._model,
                    json.dumps(kwargs, ensure_ascii=False),
                )
            except Exception:  # noqa: BLE001
                logger.info("llm_chat_request model=%s payload=<unserializable>", self._model)

            return self._client.chat.completions.create(**kwargs)

        try:
            response = await anyio.to_thread.run_sync(_call)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM request failed: %s", exc)
            raise

        choice = response.choices[0]
        content = getattr(choice.message, "content", None)

        if not content:
            # Пишем в логи "сырое" представление ответа, чтобы отладить пустой контент.
            logger.warning(
                "llm_empty_content model=%s raw_choice=%r",
                self._model,
                choice,
            )

        return content or ""
