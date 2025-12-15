from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from chat_app.config import settings
from chat_app.memory import BaseConversationMemory
from chat_app.retriever import KBRetriever
from chat_app.schemas import ChatRequest, ChatResponse, HistoryItem, MessageRole
from chat_app.scenario_registry import ScenarioRegistry
from .graph_state import AgentState, InstructionBlock, JudgeDecision, ToolsContext
from .prompt_builder_v1 import PromptBuilderV1
from .subgraphs.tools_subgraph import build_tools_subgraph
from .summarizer_v1_0 import Summarizer
from .tool_registry import ToolRegistry
from chat_app.tools.registry import build_tool_function_map

import anyio
from openai import OpenAI
from chat_app.runtime_config import get_effective_openai_api_key


logger = logging.getLogger("chat_app.graph_pipeline_v1_0")


class _OpenRouterClient:
    def __init__(self) -> None:
        self._model = settings.llm_model

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        model: str | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> str:
        def _call() -> Any:
            api_key = get_effective_openai_api_key()
            if not api_key:
                raise RuntimeError("LLM API key (OPENAI_API_KEY) is not set")
            client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
            payload: Dict[str, Any] = {
                "model": model or self._model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            try:
                logger.info("llm_request_v1_0 payload=%s", json.dumps(payload, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                logger.info("llm_request_v1_0 payload=<unserializable>")
            return client.chat.completions.create(**payload)

        response = await anyio.to_thread.run_sync(_call)
        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or ""
        logger.info(
            "llm_response_v1_0 model=%s chars=%d content=%s",
            model or self._model,
            len(content),
            content,
        )
        return content

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        schema: Dict[str, Any],
        name: str,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> Dict[str, Any]:
        rf_schema = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        try:
            raw = await self.chat(
                messages,
                temperature=temperature,
                model=model,
                response_format=rf_schema,
            )
            return json.loads((raw or "").strip())
        except Exception:  # noqa: BLE001
            pass

        try:
            raw = await self.chat(
                messages,
                temperature=temperature,
                model=model,
                response_format={"type": "json_object"},
            )
            return json.loads((raw or "").strip())
        except Exception:  # noqa: BLE001
            raw = await self.chat(messages, temperature=temperature, model=model)

        # Best-effort fallback: extract first JSON object.
        text = (raw or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return {}


class GraphChatPipelineV10:
    def __init__(
        self,
        *,
        memory: BaseConversationMemory,
        retriever: KBRetriever,
        scenario_registry: ScenarioRegistry,
    ) -> None:
        self._memory = memory
        self._retriever = retriever
        self._scenario_registry = scenario_registry

        self._client = _OpenRouterClient()
        self._summarizer = Summarizer()

        tools = ToolRegistry()
        for name, func in build_tool_function_map().items():
            tools.register(name, func)
        self._tools = tools

        self._prompt_builder = PromptBuilderV1()
        self._tools_subgraph = build_tools_subgraph(
            scenario_registry=self._scenario_registry,
            tools=self._tools,
            llm_chat=lambda messages: self._client.chat(
                messages,
                temperature=0.0,
                model=settings.condition_model,
            ),
            llm_chat_json=lambda messages, schema, name: self._client.chat_json(
                messages,
                schema=schema,
                name=name,
                temperature=0.0,
                model=settings.condition_model,
            ),
        )
        self._graph = self._build_graph()

    def export_graph_mermaid(self, *, xray: int = 1) -> str:
        return self._graph.get_graph(xray=xray).draw_mermaid()

    def export_graph_png(self, *, xray: int = 1) -> bytes:
        return self._graph.get_graph(xray=xray).draw_mermaid_png()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        async def load_state(state: AgentState) -> Dict:
            conv_state = self._memory.get_state(state["conversation_id"])
            history = self._memory.get_history(state["conversation_id"]).history
            return {
                "conv_state": conv_state,
                "history": history,
                "judge_attempts": 0,
                "tools_context": {
                    "facts": {},
                    "instruction_blocks": [],
                    "applied": [],
                },
            }

        async def append_user(state: AgentState) -> Dict:
            conv_state = state["conv_state"]
            conv_state.message_index += 1
            self._memory.append_history(
                state["conversation_id"],
                HistoryItem(role=MessageRole.USER, content=state["user_message"]),
            )
            history = self._memory.get_history(state["conversation_id"]).history
            return {"conv_state": conv_state, "history": history}

        async def retrieval(state: AgentState) -> Dict:
            kb_chunks = await self._retriever.search(query=state["user_message"])
            return {"kb_chunks": kb_chunks}

        async def tools_subgraph_node(state: AgentState) -> Dict:
            out = await self._tools_subgraph.ainvoke(state)
            return {"tools_context": out.get("tools_context")}

        async def build_messages(state: AgentState) -> Dict:
            conv_state = state["conv_state"]
            history = state.get("history") or []
            kb_chunks = state.get("kb_chunks") or []
            tools_context: ToolsContext = state.get("tools_context") or {}

            messages = self._prompt_builder.build_messages(
                conv_state=conv_state,
                history_tail=history,
                kb_chunks=kb_chunks,
                tools_context=tools_context,
                user_message=state["user_message"],
            )
            return {"prompt_messages": messages}

        async def llm_generate(state: AgentState) -> Dict:
            answer_draft = await self._client.chat(
                state["prompt_messages"],
                temperature=0.1,
                model=settings.llm_model,
            )
            return {"answer_draft": (answer_draft or "").strip()}

        async def judge_evaluate(state: AgentState) -> Dict:
            tools_context: ToolsContext = state.get("tools_context") or {}
            blocks: List[InstructionBlock] = tools_context.get("instruction_blocks") or []
            answer_draft = state.get("answer") or state.get("answer_draft") or ""
            kb_chunks = state.get("kb_chunks") or []
            conv_state = state["conv_state"]

            judge_rules = [
                b for b in blocks if b.get("target") == "judge" and b.get("kind") == "rule" and b.get("text")
            ]
            rules_text = "\n".join(f"- {b.get('text','')}" for b in judge_rules).strip()

            facts_lines: List[str] = []
            if conv_state.user_profile.name:
                facts_lines.append(f"- user_profile.name: {conv_state.user_profile.name}")
            if conv_state.user_profile.age is not None:
                facts_lines.append(f"- user_profile.age: {conv_state.user_profile.age}")

            tc_facts = tools_context.get("facts") or {}
            if "tool:get_user_data" in tc_facts:
                data = tc_facts.get("tool:get_user_data") or {}
                safe = {k: data.get(k) for k in ("name", "age") if k in data}
                facts_lines.append(f"- tool:get_user_data: {safe}")

            required_agent_blocks = [
                b
                for b in blocks
                if b.get("target") == "agent" and b.get("kind") == "required" and isinstance(b.get("text"), str)
            ]
            required_preview = [str(b.get("text")) for b in required_agent_blocks][:50]
            required_text = "\n".join(f"- {t}" for t in required_preview).strip()

            if kb_chunks:
                ctx_lines = ["База знаний (релевантные фрагменты):"]
                for idx, ch in enumerate(kb_chunks, start=1):
                    ctx_lines.append(f"[{idx}] {ch.text}")
                context_text = "\n".join(ctx_lines)
            else:
                context_text = "Релевантных фрагментов базы знаний не найдено."
            judge_system = (
                "Ты — строгий, но не занудный редактор ответа службы поддержки.\n"
                "Твоя задача: решить, нужно ли править ответ ассистента. Не переписывай без необходимости.\n"
                "Проверяй два типа проблем:\n"
                "1) фактологические: ответ не должен утверждать то, чего нет в context/tools_context;\n"
                "2) сценарные: ответ должен соблюдать rules и не применять ветки без оснований.\n"
                "Также проверь стиль:\n"
                "- Запрещены эмодзи/смайлики.\n"
                "- Запрещены обещания будущих действий/обновлений/уведомлений (например: «мы обязательно сообщим», «передали разработчикам», «в следующем обновлении»), если этого нет в context.\n"
                "Если правка нужна — верни JSON строго формата:\n"
                '{"action":"revise","reasons":["..."],"patch_instructions":"..."}\n'
                "Если правка не нужна — верни JSON:\n"
                '{"action":"pass","reasons":["ok"],"patch_instructions":""}\n'
                "Ограничение: не предлагай более 1-2 точечных правок. Без фанатизма.\n"
                "Учитывай правила:\n"
                f"{rules_text}\n\n"
                "facts_summary:\n"
                + ("\n".join(facts_lines).strip() if facts_lines else "- (нет)\n")
                + "\n\n"
                "required_instructions_summary:\n"
                + (required_text if required_text else "- (нет)\n")
                + "\n\n"
                f"context:\n{context_text}\n"
            )
            judge_user = (
                f"Последнее сообщение пользователя:\n{state['user_message']}\n\n"
                f"Черновик ответа ассистента:\n{answer_draft}\n"
            )
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "reasons", "patch_instructions"],
                "properties": {
                    "action": {"type": "string", "enum": ["pass", "revise"]},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "patch_instructions": {"type": "string"},
                },
            }
            data = await self._client.chat_json(
                [{"role": "system", "content": judge_system}, {"role": "user", "content": judge_user}],
                schema=schema,
                name="judge_decision",
                temperature=0.0,
                model=settings.judge_model,
            )
            decision: JudgeDecision = {"action": "pass", "reasons": ["ok"], "patch_instructions": ""}
            if isinstance(data, dict) and data.get("action") in ("pass", "revise"):
                decision = {
                    "action": data.get("action"),
                    "reasons": data.get("reasons") or [],
                    "patch_instructions": data.get("patch_instructions") or "",
                }
            logger.info(
                "judge_decision_v1_0 conversation_id=%s attempts=%s action=%s reasons=%s patch=%r",
                state.get("conversation_id"),
                state.get("judge_attempts"),
                decision.get("action"),
                decision.get("reasons"),
                (decision.get("patch_instructions") or "")[:2000],
            )
            return {"judge_decision": decision}

        def judge_router(state: AgentState) -> str:
            decision: JudgeDecision = state.get("judge_decision") or {}
            attempts = int(state.get("judge_attempts") or 0)
            if decision.get("action") == "revise" and attempts < 2:
                return "revise"
            return "persist"

        async def judge_revise(state: AgentState) -> Dict:
            decision: JudgeDecision = state.get("judge_decision") or {}
            patch = decision.get("patch_instructions") or ""
            original = state.get("answer") or state.get("answer_draft") or ""
            attempts = int(state.get("judge_attempts") or 0) + 1
            tools_context: ToolsContext = state.get("tools_context") or {}
            blocks: List[InstructionBlock] = tools_context.get("instruction_blocks") or []
            kb_chunks = state.get("kb_chunks") or []
            conv_state = state["conv_state"]

            facts_lines: List[str] = []
            if conv_state.user_profile.name:
                facts_lines.append(f"- user_profile.name: {conv_state.user_profile.name}")
            if conv_state.user_profile.age is not None:
                facts_lines.append(f"- user_profile.age: {conv_state.user_profile.age}")
            tc_facts = tools_context.get("facts") or {}
            if "tool:get_user_data" in tc_facts:
                data = tc_facts.get("tool:get_user_data") or {}
                safe = {k: data.get(k) for k in ("name", "age") if k in data}
                facts_lines.append(f"- tool:get_user_data: {safe}")

            required_agent_blocks = [
                b
                for b in blocks
                if b.get("target") == "agent" and b.get("kind") == "required" and isinstance(b.get("text"), str)
            ]
            required_preview = [str(b.get("text")) for b in required_agent_blocks][:12]
            must_keep_text = "\n".join(f"- {t}" for t in required_preview).strip()

            if kb_chunks:
                ctx_lines = ["База знаний (релевантные фрагменты):"]
                for idx, ch in enumerate(kb_chunks, start=1):
                    ctx_lines.append(f"[{idx}] {ch.text}")
                context_text = "\n".join(ctx_lines)
            else:
                context_text = "Релевантных фрагментов базы знаний не найдено."

            revise_system = (
                "Ты правишь ответ службы поддержки строго по инструкциям редактора.\n"
                "Не добавляй новых фактов, которых нет в контексте.\n"
                "Сделай минимальные правки.\n"
                "Верни только финальный текст ответа, без списков изменений и без пояснений.\n"
                "Нельзя удалять обязательные требования из must_keep, если они не противоречат context.\n"
                "Не используй эмодзи/смайлики; если они есть в исходном тексте — убери.\n"
                "Не добавляй обещания будущих действий/обновлений/уведомлений, если этого нет в context.\n"
            )
            revise_user = (
                f"Инструкции для правки:\n{patch}\n\n"
                f"Исходный ответ:\n{original}\n"
                "\n\nfacts_summary:\n"
                + ("\n".join(facts_lines).strip() if facts_lines else "- (нет)\n")
                + "\n\nmust_keep:\n"
                + (must_keep_text if must_keep_text else "- (нет)\n")
                + "\n\ncontext:\n"
                + context_text
                + "\n"
            )
            revised = await self._client.chat(
                [{"role": "system", "content": revise_system}, {"role": "user", "content": revise_user}],
                temperature=0.1,
                model=settings.revise_model,
            )
            logger.info(
                "judge_revise_v1_0 conversation_id=%s attempt=%d patch=%r before=%r after=%r",
                state.get("conversation_id"),
                attempts,
                patch[:2000],
                original[:4000],
                (revised or "").strip()[:4000],
            )
            return {"answer": (revised or "").strip(), "judge_attempts": attempts}

        async def persist_answer(state: AgentState) -> Dict:
            answer = (state.get("answer") or state.get("answer_draft") or "").strip()
            self._memory.append_history(
                state["conversation_id"],
                HistoryItem(role=MessageRole.ASSISTANT, content=answer),
            )
            self._memory.save_state(state["conv_state"])
            history = self._memory.get_history(state["conversation_id"]).history
            return {"answer": answer, "history": history}

        async def launch_summary(state: AgentState) -> Dict:
            conversation_id = state["conversation_id"]

            async def _run() -> None:
                await self._summarizer.update_summary(self._memory, conversation_id)

            try:
                import asyncio

                asyncio.create_task(_run())
                logger.info("summary_launch_v1_0 conversation_id=%s", conversation_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("summary_launch_failed_v1_0 conversation_id=%s error=%r", conversation_id, exc)
            return {}

        graph.add_node("load_state", load_state)
        graph.add_node("append_user", append_user)
        graph.add_node("retrieval", retrieval)
        graph.add_node("tools_subgraph", tools_subgraph_node)
        graph.add_node("build_messages", build_messages)
        graph.add_node("llm_generate", llm_generate)
        graph.add_node("judge_evaluate", judge_evaluate)
        graph.add_node("judge_revise", judge_revise)
        graph.add_node("persist_answer", persist_answer)
        graph.add_node("launch_summary", launch_summary)

        graph.set_entry_point("load_state")
        graph.add_edge("load_state", "append_user")
        graph.add_edge("append_user", "retrieval")
        graph.add_edge("retrieval", "tools_subgraph")
        graph.add_edge("tools_subgraph", "build_messages")
        graph.add_edge("build_messages", "llm_generate")
        graph.add_edge("llm_generate", "judge_evaluate")
        graph.add_conditional_edges(
            "judge_evaluate",
            judge_router,
            {
                "revise": "judge_revise",
                "persist": "persist_answer",
            },
        )
        graph.add_edge("judge_revise", "judge_evaluate")
        graph.add_edge("persist_answer", "launch_summary")
        graph.add_edge("launch_summary", END)

        return graph.compile()

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        initial: AgentState = {"conversation_id": request.conversation_id, "user_message": request.message}
        out: AgentState = await self._graph.ainvoke(initial)
        tools_context: ToolsContext = out.get("tools_context") or {}
        applied = tools_context.get("applied") or []
        last_step_id = ", ".join(a.get("name") for a in applied if a.get("name")) if applied else None
        return ChatResponse(
            conversation_id=request.conversation_id,
            answer=out.get("answer") or out.get("answer_draft") or "",
            chunks=out.get("kb_chunks") or [],
            last_step_scenario=last_step_id,
        )
