from __future__ import annotations

from typing import List

from .memory import BaseConversationMemory
from .schemas import HistoryItem, MessageRole


class Summarizer:
    """
    Simple, cheap summarizer for conversation history.

    В данном прототипе summary обновляется эвристикой:
    - берём последние несколько сообщений (user/assistant),
    - формируем короткое текстовое описание.
    Важно: здесь не используется LLM, чтобы не блокировать ответы.
    """

    async def update_summary(self, memory: BaseConversationMemory, conversation_id: str) -> None:
        history_response = memory.get_history(conversation_id)
        items: List[HistoryItem] = history_response.history

        if not items:
            return

        # Берём последние 6 сообщений для краткого описания.
        tail = items[-6:]
        parts: List[str] = []

        for item in tail:
            if item.role == MessageRole.USER:
                parts.append(f"Пользователь: {item.content}")
            elif item.role == MessageRole.ASSISTANT:
                parts.append(f"Агент: {item.content}")

        summary_text = "Сделай краткое резюме последних сообщений: " + " | ".join(parts)

        state = memory.get_state(conversation_id)
        state.summary = summary_text
        memory.save_state(state)

