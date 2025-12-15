from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, TypedDict

import anyio
from langgraph.graph import END, StateGraph
from openai import OpenAI

from chat_app.config import settings
from chat_app.memory import BaseConversationMemory
from chat_app.schemas import HistoryItem, MessageRole
from chat_app.runtime_config import get_effective_openai_api_key


logger = logging.getLogger("chat_app.summarizer_v1_0")


class _OpenRouterClient:
    def __init__(self) -> None:
        self._model = settings.summary_model or settings.llm_model

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: Dict[str, Any] | None = None,
    ) -> str:
        def _call() -> Any:
            api_key = get_effective_openai_api_key()
            if not api_key:
                raise RuntimeError("LLM API key (OPENAI_API_KEY) is not set")
            client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
            payload: Dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            try:
                logger.info("llm_request_summary_v1_0 payload=%s", json.dumps(payload, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                logger.info("llm_request_summary_v1_0 payload=<unserializable>")
            return client.chat.completions.create(**payload)

        response = await anyio.to_thread.run_sync(_call)
        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or ""
        logger.info(
            "llm_response_summary_v1_0 model=%s chars=%d content=%s",
            self._model,
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
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        rf_schema = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        try:
            raw = await self.chat(messages, temperature=temperature, response_format=rf_schema)  # type: ignore[arg-type]
            return json.loads((raw or "").strip())
        except Exception:  # noqa: BLE001
            pass

        try:
            raw = await self.chat(messages, temperature=temperature, response_format={"type": "json_object"})  # type: ignore[arg-type]
            return json.loads((raw or "").strip())
        except Exception:  # noqa: BLE001
            raw = await self.chat(messages, temperature=temperature)

        text = (raw or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return {}


class Summarizer:
    """
    v1.0: summary обновляется отдельным LangGraph-графом, но вызывается как BackgroundTask из FastAPI.
    """

    def __init__(self) -> None:
        self._llm = _OpenRouterClient()

    def _build_graph(self, *, memory: BaseConversationMemory):
        class SummaryState(TypedDict, total=False):
            conversation_id: str
            history: List[HistoryItem]
            messages: List[Dict[str, str]]
            skip: bool
            summary: str

        graph = StateGraph(SummaryState)

        async def load_history(state: Dict[str, Any]) -> Dict:
            conversation_id = state["conversation_id"]
            history = memory.get_history(conversation_id).history
            return {"history": history}

        async def build_messages(state: Dict[str, Any]) -> Dict:
            items: List[HistoryItem] = state.get("history") or []
            if not items:
                return {"messages": [], "skip": True}

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
                        "сделай сжатое повествовательное резюме на русском в 1–5 предложениях. "
                        "Пиши в форме: «Вы спрашивали ..., я объяснил ...». "
                        "Не используй формат «Пользователь: ...», «Агент: ...» и не перечисляй все сообщения. "
                        "Верни только текст резюме без заголовков (например, «Резюме:») и без списков. "
                        "Не добавляй никаких пояснений про то, что это резюме — просто сам текст резюме. "
                        "Не цитируй дословно токсичные/неприличные/оскорбительные фразы пользователя; "
                        "обсценную лексику не повторяй вообще — если важно, напиши нейтрально «пользователь выражался грубо» или опусти. "
                        "Не включай персональные данные и уникальные идентификаторы (имена, телефоны, почты, ID) — если встречаются, опусти. "
                        "Верни СТРОГО JSON без лишнего текста формата: {\"summary\": \"...\"}."
                    ),
                },
                {"role": "user", "content": f"История диалога:\n{dialog_text}"},
            ]
            return {"messages": messages, "skip": False}

        async def llm_summary(state: Dict[str, Any]) -> Dict:
            if state.get("skip"):
                return {"summary": ""}
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
            }
            data = await self._llm.chat_json(state["messages"], schema=schema, name="dialog_summary", temperature=0.1)
            summary = data.get("summary") if isinstance(data, dict) else ""
            return {"summary": str(summary or "").strip()}

        async def save_summary(state: Dict[str, Any]) -> Dict:
            if state.get("skip"):
                return {}
            conversation_id = state["conversation_id"]
            conv_state = memory.get_state(conversation_id)
            conv_state.summary = state.get("summary") or ""
            memory.save_state(conv_state)
            return {}

        graph.add_node("load_history", load_history)
        graph.add_node("build_messages", build_messages)
        graph.add_node("llm_summary", llm_summary)
        graph.add_node("save_summary", save_summary)

        graph.set_entry_point("load_history")
        graph.add_edge("load_history", "build_messages")
        graph.add_edge("build_messages", "llm_summary")
        graph.add_edge("llm_summary", "save_summary")
        graph.add_edge("save_summary", END)

        return graph.compile()

    async def update_summary(self, memory: BaseConversationMemory, conversation_id: str) -> None:
        try:
            graph = self._build_graph(memory=memory)
            await graph.ainvoke({"conversation_id": conversation_id})
        except Exception as exc:  # noqa: BLE001
            logger.error("update_summary_failed_v1_0 conversation_id=%s error=%r", conversation_id, exc)
