from __future__ import annotations

import json
import logging
from typing import Any, Dict

import redis

from .config import settings


logger = logging.getLogger("chat_app.runtime_config")

RUNTIME_CONFIG_KEY = "runtime_config:v1"
OPENAI_KEY_ROTATION_COUNTER_KEY = "runtime_config:openai_api_key_rotation_counter:v1"

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
    keys = get_effective_openai_api_keys()
    if not keys:
        return None
    idx = get_openai_api_key_rotation_index()
    return keys[idx % len(keys)]


def get_effective_openai_api_keys() -> list[str]:
    """
    Returns OpenAI/OpenRouter API keys list.

    Supports a single key or a comma-separated list of keys in OPENAI_API_KEY.
    """
    overrides = get_runtime_overrides()
    raw = overrides.get("OPENAI_API_KEY")
    if not (isinstance(raw, str) and raw.strip()):
        raw = settings.llm_api_key or ""
    raw = str(raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def get_openai_api_key_rotation_index() -> int:
    """
    Current rotation counter (not modulo). Modulo is applied by consumers.
    Stored in Redis when available, falls back to 0.
    """
    client = _get_redis()
    if client is None:
        return 0
    try:
        raw = client.get(OPENAI_KEY_ROTATION_COUNTER_KEY)
        return int(raw) if raw is not None else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read openai key rotation index: %r", exc)
        return 0


def mark_openai_api_key_rate_limited() -> None:
    """
    Advances to the next key (if multiple are configured).
    No-op when Redis is unavailable or only one key is configured.
    """
    keys = get_effective_openai_api_keys()
    if len(keys) <= 1:
        return
    client = _get_redis()
    if client is None:
        return
    try:
        client.incr(OPENAI_KEY_ROTATION_COUNTER_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to advance openai key rotation index: %r", exc)


def get_effective_agent_pipeline_version() -> str:
    overrides = get_runtime_overrides()
    v = overrides.get("AGENT_PIPELINE_VERSION")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return settings.agent_pipeline_version
