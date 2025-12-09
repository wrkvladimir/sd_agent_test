from __future__ import annotations

import logging

from .memory import BaseConversationMemory
from .llm_client import LLMClient
from .prompting import PromptBuilder
from .retriever import KBRetriever
from .scenario_registry import registry as scenario_registry
from .scenario_runner import ScenarioRunResult, ScenarioToolRunner
from .schemas import ChatRequest, ChatResponse, HistoryItem, MessageRole
from .tools.user_data import get_user_data


logger = logging.getLogger("chat_app.orchestrator")


class ChatOrchestrator:
    """
    High-level orchestrator for the chat agent.

    Coordinates:
    - conversation state / memory
    - scenario execution
    - retrieval from the knowledge base
    - prompt construction and LLM invocation
    """

    def __init__(
        self,
        memory: BaseConversationMemory,
        retriever: KBRetriever,
        scenario_runner: ScenarioToolRunner,
    ) -> None:
        self._memory = memory
        self._retriever = retriever
        self._scenario_runner = scenario_runner
        self._prompt_builder = PromptBuilder()
        self._llm_client = LLMClient()

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Handle a single chat turn.

        This method wires together:
        - state loading / incrementing message index
        - scenario execution (only for the first message in a dialog)
        - retrieval from KB
        - LLM response generation (currently stubbed)
        """
        state = self._memory.get_state(request.conversation_id)
        state.message_index += 1

        # При первом сообщении используем тул get_user_data, чтобы
        # заполнить профиль пользователя случайным именем и возрастом.
        if state.message_index == 1 and (not state.user_profile.name or state.user_profile.age is None):
            user_data = get_user_data()
            state.user_profile.name = user_data.name
            state.user_profile.age = user_data.age
            state.user_profile.birthday_date = user_data.birthday_date

        # 1) Сохраняем сообщение пользователя в историю.
        self._memory.append_history(
            request.conversation_id,
            HistoryItem(role=MessageRole.USER, content=request.message),
        )

        # 2) Retrieval (R) — ищем релевантные чанки в базе знаний.
        kb_chunks = await self._retriever.search(query=request.message)

        # 3) Tools (T) — прогоняем все доступные сценарии/инструменты поверх
        # результатов поиска и состояния диалога. Для каждого сценария его
        # precondition и логика выполнения инкапсулированы внутри
        # ScenarioToolRunner.
        scenario_context_parts: list[str] = []
        applied_scenarios: list[str] = []
        for scenario in scenario_registry.all().values():
            run_result: ScenarioRunResult = await self._scenario_runner.run(
                scenario=scenario,
                state=state,
                user_message=request.message,
                kb_chunks=kb_chunks,
            )
            state = run_result.state
            if run_result.context_text:
                scenario_context_parts.append(run_result.context_text)
                applied_scenarios.append(scenario.name)

        scenario_context = "\n\n".join(scenario_context_parts)
        last_step_id = ", ".join(applied_scenarios) if applied_scenarios else None

        # 4) Generation (G) — строим промпт и получаем ответ от LLM.
        # Пока берём всю историю; в будущем можно ограничить хвостом.
        history = self._memory.get_history(request.conversation_id).history
        prompt = self._prompt_builder.build_prompt(
            state=state,
            history_tail=history,
            scenario_context=scenario_context,
            kb_chunks=kb_chunks,
            user_message=request.message,
        )
        logger.info(
            "conversation_id=%s built_prompt_messages=%s",
            request.conversation_id,
            prompt["messages"],
        )
        try:
            answer_text = await self._llm_client.complete_chat(prompt["messages"])
            logger.info(
                "conversation_id=%s llm_answer_preview=%r",
                request.conversation_id,
                answer_text[:200],
            )
        except Exception:
            answer_text = (
                "Сейчас у меня не получается получить ответ от модели. "
                "Попробуйте, пожалуйста, повторить запрос позже."
            )

        # Append assistant reply to history and persist state.
        self._memory.append_history(
            request.conversation_id,
            HistoryItem(role=MessageRole.ASSISTANT, content=answer_text),
        )
        self._memory.save_state(state)

        return ChatResponse(
            conversation_id=request.conversation_id,
            answer=answer_text,
            chunks=kb_chunks,
            last_step_scenario=last_step_id,
        )
