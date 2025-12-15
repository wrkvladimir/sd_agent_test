from __future__ import annotations

import logging
from typing import Any, Dict, List

import anyio
from openai import APIStatusError, OpenAI, RateLimitError

from chat_app.config import settings
from chat_app.runtime_config import (
    get_effective_openai_api_key,
    get_effective_openai_api_keys,
    mark_openai_api_key_rate_limited,
)


logger = logging.getLogger("chat_app.llm_v0_1")


class LLMClient:
    """
    Thin wrapper around OpenAI-compatible chat API (e.g. OpenRouter).

    Используется только в пайплайне версии 0.1.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        base_url = base_url or settings.llm_base_url
        self._base_url = base_url
        self._model = model or settings.llm_model

        logger.info(
            "llm_client_init_v0_1 model=%s base_url=%s",
            self._model,
            self._base_url,
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

        keys = get_effective_openai_api_keys()
        if not keys:
            raise RuntimeError("LLM API key (OPENAI_API_KEY) is not set")

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Логируем payload, чтобы можно было воспроизвести запрос вручную.
        try:
            import json

            logger.info(
                "llm_chat_request_v0_1 model=%s payload=%s",
                self._model,
                json.dumps(kwargs, ensure_ascii=False),
            )
        except Exception:  # noqa: BLE001
            logger.info("llm_chat_request_v0_1 model=%s payload=<unserializable>", self._model)

        last_exc: Exception | None = None
        for _attempt in range(max(1, len(keys))):
            def _call() -> Any:
                api_key = get_effective_openai_api_key()
                if not api_key:
                    raise RuntimeError("LLM API key (OPENAI_API_KEY) is not set")
                client = OpenAI(api_key=api_key, base_url=self._base_url)
                return client.chat.completions.create(**kwargs)

            try:
                response = await anyio.to_thread.run_sync(_call)
                break
            except RateLimitError as exc:
                last_exc = exc
                mark_openai_api_key_rate_limited()
                continue
            except APIStatusError as exc:
                last_exc = exc
                if getattr(exc, "status_code", None) == 429:
                    mark_openai_api_key_rate_limited()
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("LLM v0.1 request failed: %s", exc)
                raise
        else:
            assert last_exc is not None
            raise last_exc

        choice = response.choices[0]
        content = getattr(choice.message, "content", None)

        if not content:
            logger.warning(
                "llm_empty_content_v0_1 model=%s raw_choice=%r",
                self._model,
                choice,
            )

        return content or ""
