from __future__ import annotations

import logging

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from .config import settings
from .memory import BaseConversationMemory, InMemoryConversationMemory, RedisConversationMemory
from .retriever import KBRetriever
from .scenario_registry import registry as scenario_registry
from .schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    ScenarioPatchRequest,
    ScenarioDefinition,
    ScenarioUpsertResponse,
    SummaryResponse,
    ToolSpec,
)
from .tools.user_data import get_user_data
from .tools.registry import list_tool_specs
from .sgr.schemas import SgrConvertRequest, SgrConvertResponse
from .sgr.converter import sgr_convert_text
from .sgr.langchain_chain.pipeline import SgrConvertError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = FastAPI(title="Chat Agent Service")

logger = logging.getLogger("chat_app.api")

from .pipelines.v0_1.orchestrator_v0_1 import ChatOrchestrator as ChatOrchestratorV01
from .pipelines.v0_1.scenario_runner_v0_1 import ScenarioToolRunner
from .pipelines.v0_1.summarizer_v0_1 import Summarizer as SummarizerV01
from .pipelines.v1_0.graph_pipeline import GraphChatPipelineV10 as ChatOrchestratorV10
from .pipelines.v1_0.summarizer_v1_0 import Summarizer as SummarizerV10
from .runtime_config import get_effective_agent_pipeline_version


# Пытаемся использовать Redis как основное хранилище.
try:
    _memory: BaseConversationMemory = RedisConversationMemory()
except Exception:
    # Фоллбек на in-memory, если Redis недоступен.
    _memory = InMemoryConversationMemory()
_retriever = KBRetriever()
_scenario_runner = ScenarioToolRunner(tools={"get_user_data": get_user_data})

_orchestrator_v01 = ChatOrchestratorV01(
    memory=_memory,
    retriever=_retriever,
    scenario_runner=_scenario_runner,
)
_orchestrator_v10 = ChatOrchestratorV10(
    memory=_memory,
    retriever=_retriever,
    scenario_registry=scenario_registry,
)
_summarizer_v01 = SummarizerV01()
_summarizer_v10 = SummarizerV10()

_orchestrators = {
    "0.1": _orchestrator_v01,
    "1.0": _orchestrator_v10,
}
_summarizers = {
    "0.1": _summarizer_v01,
    "1.0": _summarizer_v10,
}


@app.on_event("startup")
async def startup_event() -> None:
    # Load default scenario definition from disk (if present).
    scenario_registry.load_default_from_disk()


def get_memory() -> BaseConversationMemory:
    return _memory


def _normalize_pipeline_version(version: str | None) -> str:
    v = (version or "").strip()
    if v in _orchestrators:
        return v
    effective_default = get_effective_agent_pipeline_version()
    return effective_default if effective_default in _orchestrators else "0.1"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
async def get_config() -> dict[str, object]:
    """
    Runtime config for UI.
    Returns the default pipeline version used when request header is absent.
    """
    default_version = _normalize_pipeline_version(None)
    return {
        "default_pipeline_version": default_version,
        "supported_pipeline_versions": sorted(list(_orchestrators.keys())),
    }


@app.get("/tools", response_model=list[ToolSpec])
async def list_tools() -> list[ToolSpec]:
    return list_tool_specs()


@app.post(
    "/sgr/convert",
    response_model=SgrConvertResponse,
    response_model_exclude_none=True,
    response_model_exclude_defaults=True,
)
async def sgr_convert(request: SgrConvertRequest) -> Response:
    """
    Convert plain-language SGR text into a ScenarioDefinition + diagnostics.
    Does NOT upsert into ScenarioRegistry (caller can POST /scenarios).
    """
    tools = list_tool_specs()
    logger.info("sgr_convert_request chars=%d name_hint=%r strict=%s", len(request.text or ""), request.name_hint, request.strict)
    try:
        scenario, diagnostics, questions = await sgr_convert_text(
            text=request.text,
            available_tools=tools,
            name_hint=request.name_hint,
            strict=request.strict,
            return_diagnostics=request.return_diagnostics,
        )
        logger.info(
            "sgr_convert_result name=%r code_len=%d questions=%d trace_id=%r",
            scenario.name,
            len(scenario.code or []),
            len(questions or []),
            (diagnostics or {}).get("trace_id"),
        )
        return SgrConvertResponse(scenario=scenario, diagnostics=diagnostics, questions=questions)
    except SgrConvertError as exc:
        # Contract: return HTTP 422 with a flat JSON body (no "detail") and no scenario.
        return JSONResponse(status_code=422, content=exc.to_422())
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"LLM rate limited: {exc}") from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=f"LLM authentication error: {exc}") from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise HTTPException(status_code=504, detail=f"LLM timeout/connection error: {exc}") from exc
    except APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"SGR convert failed: {exc}") from exc


@app.get("/graph")
async def get_graph(
    format: str = "mermaid",
    xray: int = 1,
    pipeline_version: str | None = Header(default=None, alias="X-Agent-Pipeline-Version"),
) -> Response:
    """
    Debug endpoint: визуализация LangGraph-графа (v1.0).
    format=mermaid|png, xray=0|1 (раскрывает subgraph-узлы).
    """
    version = _normalize_pipeline_version(pipeline_version)
    orchestrator = _orchestrators[version]
    if not hasattr(orchestrator, "export_graph_mermaid"):
        raise HTTPException(status_code=404, detail="Graph is not available for this pipeline version")

    if format == "png":
        png = orchestrator.export_graph_png(xray=xray)  # type: ignore[attr-defined]
        return Response(content=png, media_type="image/png")

    mermaid = orchestrator.export_graph_mermaid(xray=xray)  # type: ignore[attr-defined]
    return PlainTextResponse(content=mermaid, media_type="text/plain; charset=utf-8")


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    memory: BaseConversationMemory = Depends(get_memory),
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
    pipeline_version: str | None = Header(default=None, alias="X-Agent-Pipeline-Version"),
) -> ChatResponse:
    version = _normalize_pipeline_version(pipeline_version)
    orchestrator = _orchestrators[version]
    summarizer = _summarizers[version]

    response = await orchestrator.handle_chat(request)
    # v1.0: суммаризация запускается внутри графа как fire-and-forget (launch_summary).
    # v0.1: запускаем как FastAPI BackgroundTasks (как раньше).
    if version != "1.0":
        if background_tasks is not None:
            background_tasks.add_task(summarizer.update_summary, memory, request.conversation_id)
        else:
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
    # Для отладки проблем с summary логируем наличие истории и самого summary.
    history = memory.get_history(conversation_id)
    summary = memory.get_summary(conversation_id)
    logger.info(
        "get_summary conversation_id=%s history_len=%d summary_present=%s",
        conversation_id,
        len(history.history),
        bool(summary.summary),
    )
    return summary


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


@app.patch("/scenarios/{name}", response_model=ScenarioUpsertResponse)
async def patch_scenario(name: str, patch: ScenarioPatchRequest) -> ScenarioUpsertResponse:
    scenario = scenario_registry.get(name)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    updated = scenario.model_copy()
    if patch.enabled is not None:
        updated.enabled = bool(patch.enabled)
    scenario_registry.add(updated)
    return ScenarioUpsertResponse(name=name, status="updated")
