from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import httpx
import redis
from fastapi import BackgroundTasks, FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger("ui_backend")

CHAT_APP_URL = os.getenv("CHAT_APP_URL", "http://chat-app:8000")
INGEST_URL = os.getenv("INGEST_URL", "http://ingest-and-retrieval:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


@dataclass
class ChatJob:
    job_id: str
    status: str  # pending, done, error
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


app = FastAPI(title="UI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, ChatJob] = {}

redis_client: Optional[redis.Redis] = None
RUNTIME_CONFIG_KEY = "runtime_config:v1"


class RuntimeConfigPatch(BaseModel):
    OPENAI_API_KEY: str | None = None
    RERANKER_ENABLED: bool | None = None
    SEARCH_TOP_K: int | None = None
    SEARCH_LIMIT: int | None = None
    SEARCH_SCORE_THRESHOLD: float | None = None
    CHUNK_MAX_LENGTH: int | None = None
    CHUNK_OVERLAP: int | None = None
    AGENT_PIPELINE_VERSION: str | None = None


def _parse_env_defaults() -> Dict[str, Any]:
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(str(raw).strip())
        except Exception:
            return default

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(str(raw).strip())
        except Exception:
            return default

    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "RERANKER_ENABLED": _env_bool("RERANKER_ENABLED", False),
        "SEARCH_TOP_K": _env_int("SEARCH_TOP_K", 5),
        "SEARCH_LIMIT": _env_int("SEARCH_LIMIT", 8),
        "SEARCH_SCORE_THRESHOLD": _env_float("SEARCH_SCORE_THRESHOLD", 0.5),
        "CHUNK_MAX_LENGTH": _env_int("CHUNK_MAX_LENGTH", 512),
        "CHUNK_OVERLAP": _env_int("CHUNK_OVERLAP", 100),
        "AGENT_PIPELINE_VERSION": os.getenv("AGENT_PIPELINE_VERSION", "0.1"),
    }


def _get_runtime_overrides() -> Dict[str, Any]:
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Runtime config is not available (Redis unavailable)")
    raw = redis_client.get(RUNTIME_CONFIG_KEY)
    if not raw:
        return {}
    try:
        import json

        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_runtime_overrides(overrides: Dict[str, Any]) -> None:
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Runtime config is not available (Redis unavailable)")
    import json

    redis_client.set(RUNTIME_CONFIG_KEY, json.dumps(overrides, ensure_ascii=False))


@app.on_event("startup")
async def startup_event() -> None:
    # Монтируем статику (простой фронт) из ui_frontend.
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    global redis_client  # noqa: PLW0603
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # Быстрая проверка соединения.
        client.ping()
        redis_client = client
        logger.info("connected to redis at %s", REDIS_URL)
    except Exception as exc:  # noqa: BLE001
        redis_client = None
        logger.warning("redis is not available for UI backend: %r", exc)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<html><body><h1>UI backend is running</h1></body></html>")


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def api_config() -> JSONResponse:
    """
    Proxy runtime config from chat-app (e.g., default pipeline version).
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(f"{CHAT_APP_URL}/config")
        except httpx.HTTPError as exc:
            logger.error("config request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call chat-app config") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.post("/api/ingest")
async def api_ingest(payload: Dict[str, Any] | None = Body(default=None)) -> JSONResponse:
    """
    Прокси для запуска инжеста.
    Если payload не передан, используем значения по умолчанию (data/test.html).
    Если payload передан (например, inline_html), пробрасываем его дальше,
    добавляя при необходимости значения по умолчанию для collection и др.
    """
    if not payload:
        body: Dict[str, Any] = {
            "source_type": "file",
            "source_id": "test_html",
            "path": "data/test.html",
            "html": None,
            "collection": "support_knowledge",
            "recreate_collection": False,
        }
    else:
        body = dict(payload)
        body.setdefault("collection", "support_knowledge")
        body.setdefault("recreate_collection", False)

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(f"{INGEST_URL}/ingest", json=body)
        except httpx.HTTPError as exc:
            logger.error("ingest request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call ingest service") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.post("/api/clear")
async def api_clear() -> JSONResponse:
    """
    Очистка коллекции в Qdrant через ingest-сервис.
    """
    params = {"collection": "support_knowledge"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{INGEST_URL}/clear_data", params=params)
        except httpx.HTTPError as exc:
            logger.error("clear_data request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call clear_data") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/api/data")
async def api_data() -> JSONResponse:
    """
    Получение содержимого коллекции из Qdrant через ingest-сервис.
    """
    params = {"collection": "support_knowledge"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"{INGEST_URL}/data", params=params)
        except httpx.HTTPError as exc:
            logger.error("data request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call data") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


async def _run_chat_job(job_id: str, conversation_id: str, message: str, pipeline_version: str | None) -> None:
    """
    Фоновая задача: делает запрос к chat-app и сохраняет результат в in-memory хранилище jobs.
    """
    logger.info("chat_job_start job_id=%s conversation_id=%s", job_id, conversation_id)
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            headers: Dict[str, str] = {}
            if pipeline_version in {"0.1", "1.0"}:
                headers["X-Agent-Pipeline-Version"] = pipeline_version
            resp = await client.post(
                f"{CHAT_APP_URL}/chat",
                json={"conversation_id": conversation_id, "message": message},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            jobs[job_id].status = "done"
            jobs[job_id].result = data
            logger.info("chat_job_done job_id=%s", job_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("chat_job_failed job_id=%s error=%r", job_id, exc)
            jobs[job_id].status = "error"
            jobs[job_id].error = str(exc)


@app.post("/api/chat/send")
async def api_chat_send(background_tasks: BackgroundTasks, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Создать чат-запрос как long-running job.
    payload: {conversation_id: str, message: str}
    """
    conversation_id = payload.get("conversation_id")
    message = payload.get("message")
    pipeline_version = payload.get("pipeline_version")
    if not conversation_id or not isinstance(conversation_id, str):
        raise HTTPException(status_code=400, detail="conversation_id is required")
    if not message or not isinstance(message, str):
        raise HTTPException(status_code=400, detail="message is required")
    if pipeline_version is not None and pipeline_version not in {"0.1", "1.0"}:
        raise HTTPException(status_code=400, detail="pipeline_version must be 0.1 or 1.0")

    job_id = str(uuid.uuid4())
    jobs[job_id] = ChatJob(job_id=job_id, status="pending")
    background_tasks.add_task(_run_chat_job, job_id, conversation_id, message, pipeline_version)
    return {"job_id": job_id}


@app.get("/api/chat/poll")
async def api_chat_poll(job_id: str) -> Dict[str, Any]:
    """
    Получить статус long-running chat-задачи.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return asdict(job)


@app.get("/api/chat/history")
async def api_chat_history(conversation_id: str) -> JSONResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"{CHAT_APP_URL}/history", params={"conversation_id": conversation_id})
        except httpx.HTTPError as exc:
            logger.error("history request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call history") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/api/chat/summary")
async def api_chat_summary(conversation_id: str) -> JSONResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"{CHAT_APP_URL}/summary", params={"conversation_id": conversation_id})
        except httpx.HTTPError as exc:
            logger.error("summary request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call summary") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/api/chat/conversations")
async def api_chat_conversations() -> Dict[str, Any]:
    """
    Возвращает список известных conversation_id из Redis (conv:{id}:state).
    Работает только если Redis доступен и chat-app использует тот же инстанс.
    """
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Conversations listing is not available")
    try:
        keys = redis_client.keys("conv:*:state")
        conv_ids = []
        for k in keys:
            parts = k.split(":")
            if len(parts) >= 3:
                conv_ids.append(parts[1])
        conv_ids = sorted(set(conv_ids))
        return {"conversations": conv_ids}
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to list conversations from redis: %r", exc)
        raise HTTPException(status_code=500, detail="Failed to list conversations") from exc


@app.post("/api/sgr/convert")
async def api_sgr_convert(payload: Dict[str, Any]) -> JSONResponse:
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(f"{CHAT_APP_URL}/sgr/convert", json=payload)
        except httpx.HTTPError as exc:
            logger.error("sgr convert request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call sgr convert") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/api/scenarios")
async def api_list_scenarios() -> JSONResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"{CHAT_APP_URL}/scenarios")
        except httpx.HTTPError as exc:
            logger.error("scenarios list request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call scenarios list") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.post("/api/scenarios")
async def api_upsert_scenario(payload: Dict[str, Any]) -> JSONResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{CHAT_APP_URL}/scenarios", json=payload)
        except httpx.HTTPError as exc:
            logger.error("scenario upsert request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call scenario upsert") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.patch("/api/scenarios/{name}")
async def api_patch_scenario(name: str, payload: Dict[str, Any]) -> JSONResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.patch(f"{CHAT_APP_URL}/scenarios/{name}", json=payload)
        except httpx.HTTPError as exc:
            logger.error("scenario patch request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call scenario patch") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.delete("/api/scenarios/{name}")
async def api_delete_scenario(name: str) -> JSONResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.delete(f"{CHAT_APP_URL}/scenarios/{name}")
        except httpx.HTTPError as exc:
            logger.error("scenario delete request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call scenario delete") from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.get("/api/graph")
async def api_graph(format: str = "mermaid", xray: int = 1) -> Response:
    """
    Proxy /graph from chat-app with an appropriate pipeline header.
    Browser links can't send custom headers, so we inject X-Agent-Pipeline-Version here.
    """
    defaults = _parse_env_defaults()
    overrides = _get_runtime_overrides()
    effective = dict(defaults)
    for k, v in overrides.items():
        if k in effective:
            effective[k] = v

    version = str(effective.get("AGENT_PIPELINE_VERSION") or "").strip()
    # /graph is only available for v1.0; ensure it works even if default pipeline is 0.1.
    if version != "1.0":
        version = "1.0"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(
                f"{CHAT_APP_URL}/graph",
                params={"format": format, "xray": xray},
                headers={"X-Agent-Pipeline-Version": version},
            )
        except httpx.HTTPError as exc:
            logger.error("graph request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call graph") from exc

    media_type = resp.headers.get("content-type") or "text/plain; charset=utf-8"
    return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)


@app.get("/api/graph/mermaid-live")
async def api_graph_mermaid_live(xray: int = 1) -> Response:
    """
    Open graph visualization directly in mermaid.live instead of rendering PNG via mermaid.ink.
    """
    defaults = _parse_env_defaults()
    overrides = _get_runtime_overrides()
    effective = dict(defaults)
    for k, v in overrides.items():
        if k in effective:
            effective[k] = v

    version = str(effective.get("AGENT_PIPELINE_VERSION") or "").strip()
    if version != "1.0":
        version = "1.0"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(
                f"{CHAT_APP_URL}/graph",
                params={"format": "mermaid", "xray": xray},
                headers={"X-Agent-Pipeline-Version": version},
            )
        except httpx.HTTPError as exc:
            logger.error("graph mermaid request failed: %r", exc)
            raise HTTPException(status_code=502, detail="Failed to call graph") from exc

    mermaid_text = resp.text
    try:
        import base64
        import zlib

        compressed = zlib.compress(mermaid_text.encode("utf-8"))
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        url = f"https://mermaid.live/edit#pako:{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to build mermaid.live url: %r", exc)
        raise HTTPException(status_code=500, detail="Failed to build mermaid.live URL") from exc

    return RedirectResponse(url=url)


@app.get("/api/runtime-config")
async def api_runtime_config() -> Dict[str, Any]:
    defaults = _parse_env_defaults()
    overrides = _get_runtime_overrides()
    effective = dict(defaults)
    for k, v in overrides.items():
        if k in effective:
            effective[k] = v
    return {"values": effective, "overrides": overrides, "defaults": defaults}


@app.patch("/api/runtime-config")
async def api_runtime_config_patch(patch: RuntimeConfigPatch) -> Dict[str, Any]:
    defaults = _parse_env_defaults()
    overrides = _get_runtime_overrides()
    patch_dict = patch.model_dump(exclude_unset=True)

    for k, v in patch_dict.items():
        if k == "OPENAI_API_KEY" and isinstance(v, str) and not v.strip():
            v = None
        if v is None:
            overrides.pop(k, None)
        else:
            overrides[k] = v

    _set_runtime_overrides(overrides)
    effective = dict(defaults)
    for k, v in overrides.items():
        if k in effective:
            effective[k] = v
    return {"values": effective, "overrides": overrides, "defaults": defaults}
