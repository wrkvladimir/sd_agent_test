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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


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


async def _run_chat_job(job_id: str, conversation_id: str, message: str) -> None:
    """
    Фоновая задача: делает запрос к chat-app и сохраняет результат в in-memory хранилище jobs.
    """
    logger.info("chat_job_start job_id=%s conversation_id=%s", job_id, conversation_id)
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(
                f"{CHAT_APP_URL}/chat",
                json={"conversation_id": conversation_id, "message": message},
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
    if not conversation_id or not isinstance(conversation_id, str):
        raise HTTPException(status_code=400, detail="conversation_id is required")
    if not message or not isinstance(message, str):
        raise HTTPException(status_code=400, detail="message is required")

    job_id = str(uuid.uuid4())
    jobs[job_id] = ChatJob(job_id=job_id, status="pending")
    background_tasks.add_task(_run_chat_job, job_id, conversation_id, message)
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
