from __future__ import annotations

from typing import List

from .llm_client import LLMClient
from .memory import BaseConversationMemory
from .schemas import HistoryItem, MessageRole


class Summarizer:
    """
    Summarizer for conversation history.

    В данном прототипе summary обновляется с помощью LLM:
    - берём последние несколько сообщений (user/assistant),
    - просим модель сделать краткое повествовательное резюме.
    """

    async def update_summary(self, memory: BaseConversationMemory, conversation_id: str) -> None:
        history_response = memory.get_history(conversation_id)
        items: List[HistoryItem] = history_response.history

        if not items:
            return

        # Берём последние N сообщений для контекста суммаризации.
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

        llm = LLMClient()
        try:
            summary_text = await llm.complete_chat(messages, max_tokens=160, temperature=0.1)
        except Exception:
            # В случае ошибки LLM оставляем существующее summary без изменений.
            return

        state = memory.get_state(conversation_id)
        state.summary = summary_text.strip()
        memory.save_state(state)
