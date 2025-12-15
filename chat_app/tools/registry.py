from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from chat_app.schemas import ToolSpec

from .user_data import get_user_data


ToolFunc = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    spec: ToolSpec
    func: ToolFunc


def list_tool_definitions() -> List[ToolDefinition]:
    """
    Single source of truth for tools: both metadata (spec) and implementation (func).
    """
    return [
        ToolDefinition(
            spec=ToolSpec(
                name="get_user_data",
                description="Возвращает данные профиля пользователя (имя, возраст).",
                input_schema={"type": "object", "additionalProperties": False, "properties": {}},
                output_schema={
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            ),
            func=get_user_data,
        )
    ]


def list_tool_specs() -> List[ToolSpec]:
    return [d.spec for d in list_tool_definitions()]


def build_tool_function_map() -> Dict[str, ToolFunc]:
    return {d.spec.name: d.func for d in list_tool_definitions()}
