from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import redis


logger = logging.getLogger("ingest_and_retrieval.runtime_config")

RUNTIME_CONFIG_KEY = "runtime_config:v1"

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis | None:
    global _redis_client  # noqa: PLW0603
    if _redis_client is not None:
        return _redis_client
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        client = redis.Redis.from_url(url, decode_responses=True)
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


def get_override_bool(name: str) -> bool | None:
    v = get_runtime_overrides().get(name)
    return v if isinstance(v, bool) else None


def get_override_int(name: str) -> int | None:
    v = get_runtime_overrides().get(name)
    return v if isinstance(v, int) else None


def get_override_float(name: str) -> float | None:
    v = get_runtime_overrides().get(name)
    if isinstance(v, (int, float)):
        return float(v)
    return None

