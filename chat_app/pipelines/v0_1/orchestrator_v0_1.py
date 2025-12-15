from __future__ import annotations

import logging

from chat_app.memory import BaseConversationMemory
from chat_app.retriever import KBRetriever
from chat_app.scenario_registry import registry as scenario_registry
from chat_app.schemas import ChatRequest, ChatResponse, HistoryItem, MessageRole
from chat_app.tools.user_data import get_user_data

from .llm_client_v0_1 import LLMClient
from .prompting_v0_1 import PromptBuilder
from .scenario_runner_v0_1 import ScenarioRunResult, ScenarioToolRunner


logger = logging.getLogger("chat_app.orchestrator_v0_1")


class ChatOrchestrator:
    """
    Версия 0.1: линейный оркестратор чат-агента.
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
        state = self._memory.get_state(request.conversation_id)
        state.message_index += 1

        if state.message_index == 1 and (not state.user_profile.name or state.user_profile.age is None):
            user_data = get_user_data()
            state.user_profile.name = user_data.name
            state.user_profile.age = user_data.age

        self._memory.append_history(
            request.conversation_id,
            HistoryItem(role=MessageRole.USER, content=request.message),
        )

        kb_chunks = await self._retriever.search(query=request.message)

        scenario_context_parts: list[str] = []
        applied_scenarios: list[str] = []
        for scenario in scenario_registry.all().values():
            if not getattr(scenario, "enabled", True):
                continue
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

        history = self._memory.get_history(request.conversation_id).history
        prompt = self._prompt_builder.build_prompt(
            state=state,
            history_tail=history,
            scenario_context=scenario_context,
            kb_chunks=kb_chunks,
            user_message=request.message,
        )
        logger.info(
            "conversation_id=%s built_prompt_messages_v0_1=%s",
            request.conversation_id,
            prompt["messages"],
        )
        try:
            answer_text = await self._llm_client.complete_chat(prompt["messages"])
            logger.info(
                "conversation_id=%s llm_answer_preview_v0_1=%r",
                request.conversation_id,
                answer_text[:200],
            )
        except Exception as exc:
            logger.error(
                "conversation_id=%s llm_request_failed_v0_1 error=%r",
                request.conversation_id,
                exc,
            )
            msg = str(exc).lower()
            if any(k in msg for k in ("401", "unauthorized", "invalid api key", "authentication")):
                reason = " Причина: проблема с токеном доступа или авторизацией."
            elif any(k in msg for k in ("429", "rate limit", "too many requests", "quota")):
                reason = " Причина: временное превышение лимитов запросов к LLM-сервису."
            elif any(k in msg for k in ("timeout", "timed out", "connection", "connecterror", "network")):
                reason = " Причина: проблемы с сетевым доступом или таймаут соединения с LLM-сервисом."
            else:
                reason = " Причина: внутренняя ошибка на стороне LLM-сервиса."

            answer_text = (
                "Сейчас у меня не получается получить ответ от модели."
                f"{reason} Попробуйте, пожалуйста, повторить запрос позже."
            )

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
