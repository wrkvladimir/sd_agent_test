from __future__ import annotations

import json
import logging
from typing import Any, Dict

import redis

from .config import settings


logger = logging.getLogger("chat_app.runtime_config")

RUNTIME_CONFIG_KEY = "runtime_config:v1"

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis | None:
    global _redis_client  # noqa: PLW0603
    if _redis_client is not None:
        return _redis_client
    try:
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime config redis unavailable: %r", exc)
        _redis_client = None
        return None


def get_runtime_overrides() -> Dict[str, Any]:
    client = _get_redis()
    if client is None:
        return {}
    try:
        raw = client.get(RUNTIME_CONFIG_KEY)
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read runtime config: %r", exc)
        return {}


def get_effective_openai_api_key() -> str | None:
    overrides = get_runtime_overrides()
    key = overrides.get("OPENAI_API_KEY")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return settings.llm_api_key


def get_effective_agent_pipeline_version() -> str:
    overrides = get_runtime_overrides()
    v = overrides.get("AGENT_PIPELINE_VERSION")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return settings.agent_pipeline_version

