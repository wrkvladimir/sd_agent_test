from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Dict, Optional

from chat_app.pipelines.v1_0.graph_state import AgentState


ToolFunc = Callable[..., Any]


class ToolRegistry:
    def __init__(self, tools: Optional[Dict[str, ToolFunc]] = None) -> None:
        self._tools: Dict[str, ToolFunc] = dict(tools or {})

    def register(self, name: str, func: ToolFunc) -> None:
        self._tools[name] = func

    def get(self, name: str) -> ToolFunc | None:
        return self._tools.get(name)

    async def call(self, name: str, *, state: AgentState) -> Dict[str, Any]:
        func = self.get(name)
        if func is None:
            return {}

        try:
            sig = inspect.signature(func)
            kwargs = {"state": state} if "state" in sig.parameters else {}
            result = func(**kwargs)
            if isinstance(result, Awaitable):
                result = await result
        except Exception:  # noqa: BLE001
            return {}

        if isinstance(result, dict):
            return result
        return getattr(result, "__dict__", {"value": result})

