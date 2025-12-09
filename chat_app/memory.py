from __future__ import annotations

import json
from typing import Dict, List

import redis

from .config import settings
from .schemas import ConversationState, HistoryItem, HistoryResponse, SummaryResponse


class BaseConversationMemory:
    """
    Abstract interface for conversation memory.

    Concrete implementations may use Redis, a database or in-memory storage.
    """

    def get_state(self, conversation_id: str) -> ConversationState:
        raise NotImplementedError

    def save_state(self, state: ConversationState) -> None:
        raise NotImplementedError

    def append_history(self, conversation_id: str, item: HistoryItem) -> None:
        raise NotImplementedError

    def get_history(self, conversation_id: str) -> HistoryResponse:
        raise NotImplementedError

    def get_summary(self, conversation_id: str) -> SummaryResponse:
        raise NotImplementedError


class InMemoryConversationMemory(BaseConversationMemory):
    """
    Simple in-memory implementation of conversation memory.

    Suitable for local development and tests. Can be replaced by a Redis-based
    implementation without changing the public interface.
    """

    def __init__(self) -> None:
        self._states: Dict[str, ConversationState] = {}
        self._history: Dict[str, List[HistoryItem]] = {}

    def get_state(self, conversation_id: str) -> ConversationState:
        state = self._states.get(conversation_id)
        if state is None:
            state = ConversationState(conversation_id=conversation_id)
            self._states[conversation_id] = state
        return state

    def save_state(self, state: ConversationState) -> None:
        self._states[state.conversation_id] = state

    def append_history(self, conversation_id: str, item: HistoryItem) -> None:
        self._history.setdefault(conversation_id, []).append(item)

    def get_history(self, conversation_id: str) -> HistoryResponse:
        items = self._history.get(conversation_id, [])
        return HistoryResponse(conversation_id=conversation_id, history=items)

    def get_summary(self, conversation_id: str) -> SummaryResponse:
        state = self.get_state(conversation_id)
        # Stub: summary will be updated by a dedicated summarizer component.
        return SummaryResponse(conversation_id=conversation_id, summary=state.summary)


class RedisConversationMemory(BaseConversationMemory):
    """
    Redis-backed implementation of conversation memory.

    Хранит:
    - состояние диалога в ключе conv:{conversation_id}:state (JSON ConversationState)
    - историю сообщений в списке conv:{conversation_id}:history (JSON HistoryItem)
    """

    def __init__(self, url: str | None = None) -> None:
        redis_url = url or settings.redis_url
        # decode_responses=True — чтобы получать/записывать строки (а не bytes).
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _state_key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}:state"

    def _history_key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}:history"

    def get_state(self, conversation_id: str) -> ConversationState:
        raw = self._client.get(self._state_key(conversation_id))
        if not raw:
            return ConversationState(conversation_id=conversation_id)
        try:
            data = json.loads(raw)
            return ConversationState.model_validate(data)
        except Exception:  # noqa: BLE001
            # В случае проблем с форматом — начинаем с чистого состояния.
            return ConversationState(conversation_id=conversation_id)

    def save_state(self, state: ConversationState) -> None:
        # mode="json" гарантирует сериализацию datetime и других типов
        # в JSON-дружественный формат.
        data = state.model_dump(mode="json")
        self._client.set(self._state_key(state.conversation_id), json.dumps(data, ensure_ascii=False))

    def append_history(self, conversation_id: str, item: HistoryItem) -> None:
        # mode="json" конвертирует timestamp (datetime) в строку.
        data = item.model_dump(mode="json")
        self._client.rpush(self._history_key(conversation_id), json.dumps(data, ensure_ascii=False))

    def get_history(self, conversation_id: str) -> HistoryResponse:
        raw_items = self._client.lrange(self._history_key(conversation_id), 0, -1)
        items: List[HistoryItem] = []
        for raw in raw_items:
            try:
                data = json.loads(raw)
                items.append(HistoryItem.model_validate(data))
            except Exception:  # noqa: BLE001
                continue
        return HistoryResponse(conversation_id=conversation_id, history=items)

    def get_summary(self, conversation_id: str) -> SummaryResponse:
        state = self.get_state(conversation_id)
        return SummaryResponse(conversation_id=conversation_id, summary=state.summary)
