from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schemas import Chunk, ConversationState, ScenarioDefinition


@dataclass
class ScenarioRunResult:
    """
    Result of executing a scenario for a single user message.

    context_text: accumulated text instructions that must be added
                  to the LLM prompt.
    last_step_id: identifier of the last executed node (for debugging).
    state:        updated ConversationState.
    """

    context_text: str
    last_step_id: Optional[str]
    state: ConversationState


class ScenarioToolRunner:
    """
    Engine responsible for executing JSON-defined scenarios.

    The concrete logic (text nodes, tool calls, condition handling and
    template substitutions) will be implemented here.
    """

    def __init__(self, tools: Optional[Dict[str, Any]] = None) -> None:
        # tools: mapping from tool name to callable
        self._tools: Dict[str, Any] = tools or {}

    async def run(
        self,
        scenario: ScenarioDefinition,
        state: ConversationState,
        user_message: str,
        kb_chunks: List[Chunk],
    ) -> ScenarioRunResult:
        """
        Execute a single scenario for the given user message.

        Для текущего прототипа:
        - сценарий "ДР" считается релевантным только при первом сообщении
          (message_index == 1);
        - конкретная логика разбора JSON-ноды будет добавлена позже.
        """
        _ = (user_message, kb_chunks)  # placeholders для будущей логики

        # Простейший precondition: сценарий применяется только к первому
        # сообщению диалога. Для других сценариев здесь можно будет задать
        # свои условия.
        if state.message_index != 1:
            return ScenarioRunResult(context_text="", last_step_id=None, state=state)

        # Пока сценарий ничего не добавляет к контексту; позже здесь будет:
        # - обход нод (text/tool/if/end)
        # - вызовы tools
        # - формирование текста для special_instructions.
        return ScenarioRunResult(context_text="", last_step_id=None, state=state)
