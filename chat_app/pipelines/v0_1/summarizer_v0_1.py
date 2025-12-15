from __future__ import annotations

from typing import List

import logging

from chat_app.memory import BaseConversationMemory
from chat_app.schemas import HistoryItem, MessageRole

from .llm_client_v0_1 import LLMClient


logger = logging.getLogger("chat_app.summarizer_v0_1")


class Summarizer:
    """
    Версия 0.1: суммаризатор истории диалога через тот же LLM.
    """

    async def update_summary(self, memory: BaseConversationMemory, conversation_id: str) -> None:
        history_response = memory.get_history(conversation_id)
        items: List[HistoryItem] = history_response.history

        logger.info(
            "update_summary_start_v0_1 conversation_id=%s history_len=%d",
            conversation_id,
            len(items),
        )

        if not items:
            logger.info(
                "update_summary_skip_v0_1 conversation_id=%s reason=no_history",
                conversation_id,
            )
            return

        tail = items[-16:]
        dialog_lines: List[str] = []
        for item in tail:
            role = "user" if item.role == MessageRole.USER else "assistant"
            dialog_lines.append(f"{role}: {item.content}")
        dialog_text = "\n".join(dialog_lines)

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты помогаешь составлять краткое резюме диалога поддержки. "
                    "На основе истории сообщений между пользователем (user) и агентом (assistant) "
                    "сделай сжатое повествовательное резюме на русском в 1–3 предложениях. "
                    "Пиши в форме: «Вы спрашивали ..., я объяснил ...». "
                    "Не используй формат «Пользователь: ...», «Агент: ...» и не перечисляй все сообщения. "
                    "Не добавляй никаких пояснений про то, что это резюме — просто сам текст резюме."
                ),
            },
            {
                "role": "user",
                "content": f"История диалога:\n{dialog_text}",
            },
        ]

        logger.info(
            "update_summary_llm_request_v0_1 conversation_id=%s system=%r user=%r",
            conversation_id,
            messages[0]["content"],
            messages[1]["content"],
        )

        llm = LLMClient()
        try:
            summary_text = await llm.complete_chat(messages, temperature=0.1)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "update_summary_failed_v0_1 conversation_id=%s error=%r",
                conversation_id,
                exc,
            )
            return

        summary_text = (summary_text or "").strip()
        logger.info(
            "update_summary_llm_response_v0_1 conversation_id=%s full_summary=%r",
            conversation_id,
            summary_text,
        )

        state = memory.get_state(conversation_id)
        state.summary = summary_text
        memory.save_state(state)

