from __future__ import annotations

import logging

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException

from .memory import BaseConversationMemory, InMemoryConversationMemory, RedisConversationMemory
from .orchestrator import ChatOrchestrator
from .retriever import KBRetriever
from .scenario_registry import registry as scenario_registry
from .scenario_runner import ScenarioToolRunner
from .schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    ScenarioDefinition,
    ScenarioUpsertResponse,
    SummaryResponse,
)
from .summarizer import Summarizer
from .tools.user_data import get_user_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = FastAPI(title="Chat Agent Service")


# Пытаемся использовать Redis как основное хранилище.
try:
    _memory: BaseConversationMemory = RedisConversationMemory()
except Exception:
    # Фоллбек на in-memory, если Redis недоступен.
    _memory = InMemoryConversationMemory()
_retriever = KBRetriever()
_scenario_runner = ScenarioToolRunner(tools={"get_user_data": get_user_data})
_summarizer = Summarizer()
_orchestrator = ChatOrchestrator(
    memory=_memory,
    retriever=_retriever,
    scenario_runner=_scenario_runner,
)


@app.on_event("startup")
async def startup_event() -> None:
    # Load default scenario definition from disk (if present).
    scenario_registry.load_default_from_disk()


def get_memory() -> BaseConversationMemory:
    return _memory


def get_orchestrator() -> ChatOrchestrator:
    return _orchestrator


def get_summarizer() -> Summarizer:
    return _summarizer


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
    memory: BaseConversationMemory = Depends(get_memory),
    summarizer: Summarizer = Depends(get_summarizer),
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
) -> ChatResponse:
    response = await orchestrator.handle_chat(request)
    if background_tasks is not None:
        background_tasks.add_task(summarizer.update_summary, memory, request.conversation_id)
    else:
        # Fallback: обновляем summary синхронно, если фоновые задачи недоступны.
        await summarizer.update_summary(memory, request.conversation_id)
    return response


@app.get("/history", response_model=HistoryResponse)
async def get_history(
    conversation_id: str,
    memory: BaseConversationMemory = Depends(get_memory),
) -> HistoryResponse:
    return memory.get_history(conversation_id)


@app.get("/summary", response_model=SummaryResponse)
async def get_summary(
    conversation_id: str,
    memory: BaseConversationMemory = Depends(get_memory),
) -> SummaryResponse:
    return memory.get_summary(conversation_id)


@app.post("/scenarios", response_model=ScenarioUpsertResponse)
async def add_scenario(definition: ScenarioDefinition) -> ScenarioUpsertResponse:
    """Add or update a scenario definition available to the agent."""
    scenario_registry.add(definition)
    return ScenarioUpsertResponse(name=definition.name)


@app.get("/scenarios", response_model=list[ScenarioDefinition])
async def list_scenarios() -> list[ScenarioDefinition]:
    """Return all currently registered scenarios."""
    return list(scenario_registry.all().values())


@app.delete("/scenarios/{name}", response_model=ScenarioUpsertResponse)
async def delete_scenario(name: str) -> ScenarioUpsertResponse:
    """Delete scenario by name."""
    if scenario_registry.get(name) is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    scenario_registry.remove(name)
    return ScenarioUpsertResponse(name=name, status="deleted")
