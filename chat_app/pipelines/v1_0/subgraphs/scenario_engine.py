from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from chat_app.pipelines.v1_0.graph_state import (
    AgentState,
    InstructionBlock,
    ScenarioMapResult,
)
from chat_app.pipelines.v1_0.tool_registry import ToolRegistry
from chat_app.schemas import ScenarioDefinition, ScenarioNode


_TEMPLATE_PATTERN = re.compile(r"\{=([^=]+)=\}")


def _sort_key(node: ScenarioNode) -> List[int]:
    parts: List[int] = []
    for part in node.id.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return parts


def _render_template(text: str, *, state: AgentState, tool_results: Dict[str, Dict[str, Any]]) -> str:
    conv_state = state["conv_state"]

    def replace(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if expr.startswith("@"):
            inner = expr[1:]
            parts = inner.split(".")
            if not parts:
                return "finderror"
            tool_name = parts[0]
            field = parts[1] if len(parts) > 1 else None
            tool_data = tool_results.get(tool_name) or {}
            if field is None:
                value = tool_data or None
            else:
                value = tool_data.get(field)
            return str(value) if value is not None else "finderror"

        if expr.startswith("dialog."):
            key = expr[len("dialog.") :]
            if key == "name":
                value = conv_state.user_profile.name
            elif key == "age":
                value = conv_state.user_profile.age
            elif key == "message_index":
                value = conv_state.message_index
            else:
                value = None
            return str(value) if value is not None else "finderror"

        return "finderror"

    return _TEMPLATE_PATTERN.sub(replace, text)


def _try_eval_message_index_condition(condition: str, *, message_index: int) -> Optional[bool]:
    """
    Best-effort evaluation for conditions that explicitly reference dialog.message_index.

    This enables "program-like" branching where children may include tool/if nodes,
    e.g. wrapping the whole scenario into `if (dialog.message_index == 1) { ... }`.
    """
    text = (condition or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if "не перв" in lowered and "сообщ" in lowered:
        return message_index != 1
    if "перв" in lowered and "сообщ" in lowered:
        return message_index == 1

    match = re.search(
        r"(?i)\b(?:dialog\.)?message_index\s*(==|!=|<=|>=|<|>)\s*(\d+)\b",
        text,
    )
    if not match:
        return None
    op = match.group(1)
    try:
        rhs = int(match.group(2))
    except Exception:  # noqa: BLE001
        return None

    if op == "==":
        return message_index == rhs
    if op == "!=":
        return message_index != rhs
    if op == "<":
        return message_index < rhs
    if op == "<=":
        return message_index <= rhs
    if op == ">":
        return message_index > rhs
    if op == ">=":
        return message_index >= rhs
    return None


async def run_scenario_map(
    *,
    state: AgentState,
    scenario: ScenarioDefinition,
    tools: ToolRegistry,
) -> ScenarioMapResult | None:
    conv_state = state["conv_state"]
    apply_only_index = (scenario.meta or {}).get("apply_only_message_index")
    if apply_only_index is not None:
        try:
            required_idx = int(apply_only_index)
        except Exception:  # noqa: BLE001
            required_idx = None
        if required_idx is not None and conv_state.message_index != required_idx:
            return None

    facts: Dict[str, Dict[str, Any]] = {}
    instruction_blocks: List[InstructionBlock] = []

    async def ensure_tool_data(tool_name: str) -> None:
        key = f"tool:{tool_name}"
        if key in facts:
            return

        if tool_name == "get_user_data":
            # v1.0: вызываем tool только при необходимости, но результат всегда доступен для шаблонов.
            profile = conv_state.user_profile
            if profile.name and profile.age is not None:
                facts[key] = {
                    "name": profile.name,
                    "age": profile.age,
                }
                return

        facts[key] = await tools.call(tool_name, state=state)
        if tool_name == "get_user_data":
            profile = conv_state.user_profile
            data = facts.get(key) or {}
            if not profile.name:
                profile.name = data.get("name")
            if profile.age is None and data.get("age") is not None:
                try:
                    profile.age = int(data.get("age"))
                except Exception:  # noqa: BLE001
                    profile.age = data.get("age")

    def add_text_block(node_id: str, text: str) -> None:
        rendered = _render_template(
            text,
            state=state,
            tool_results={k.removeprefix("tool:"): v for k, v in facts.items()},
        )
        instruction_blocks.append(
            {
                "id": f"scenario:{scenario.name}:text:{node_id}",
                "source": scenario.name,
                "target": "agent",
                "kind": "raw",
                "priority": 10,
                "text": rendered,
                "payload": {"node_id": node_id, "node_type": "text"},
            }
        )

    def add_conditional_program(node: ScenarioNode) -> None:
        condition = node.condition or ""
        true_texts: List[str] = []
        false_texts: List[str] = []

        for child in node.children or []:
            if child.type == "text" and child.text:
                true_texts.append(
                    _render_template(child.text, state=state, tool_results={k.removeprefix("tool:"): v for k, v in facts.items()})
                )
        for child in node.else_children or []:
            if child.type == "text" and child.text:
                false_texts.append(
                    _render_template(child.text, state=state, tool_results={k.removeprefix("tool:"): v for k, v in facts.items()})
                )

        instruction_blocks.append(
            {
                "id": f"scenario:{scenario.name}:if:{node.id}",
                "source": scenario.name,
                "target": "agent",
                "kind": "conditional",
                "priority": 10,
                "payload": {
                    "condition_id": node.id,
                    "condition": condition,
                    "when_true": true_texts,
                    "when_false": false_texts,
                    "apply_policy": {
                        "relevance_gate": "Если сообщение не относится к теме условия — игнорируй блок полностью.",
                        "true_gate": "Считай условие TRUE только если из сообщения явно следует, что условие выполняется.",
                        "false_gate": "Считай условие FALSE только если из сообщения явно следует, что условие НЕ выполняется, но тема та же.",
                        "unknown_gate": "Если упомянута тема, но непонятно TRUE/FALSE — не применяй when_false по умолчанию и лучше игнорируй блок.",
                    },
                    "condition_text": condition,
                },
            }
        )

        instruction_blocks.append(
            {
                "id": f"scenario:{scenario.name}:judge_rule:if:{node.id}",
                "source": scenario.name,
                "target": "judge",
                "kind": "rule",
                "priority": 10,
                "text": (
                    "Проверь, что условные сценарные инструкции применены только при явном подтверждении в сообщении пользователя. "
                    "Не допускай применения when_false по умолчанию при неоднозначности."
                ),
            }
        )

    async def process_nodes(nodes: List[ScenarioNode]) -> bool:
        """
        Returns True if `end` encountered and scenario execution should stop.
        """
        for node in sorted(nodes, key=_sort_key):
            if node.type == "end":
                return True
            if node.type == "tool" and node.tool:
                await ensure_tool_data(node.tool)
                continue
            if node.type == "text" and node.text:
                add_text_block(node.id, node.text)
                continue
            if node.type == "if":
                decided = _try_eval_message_index_condition(
                    str(node.condition or ""),
                    message_index=conv_state.message_index,
                )
                if decided is True:
                    if await process_nodes(list(node.children or [])):
                        return True
                    continue
                if decided is False:
                    if await process_nodes(list(node.else_children or [])):
                        return True
                    continue

                add_conditional_program(node)
                continue
        return False

    _ = await process_nodes(list(scenario.code or []))

    if not instruction_blocks and not facts:
        return None

    return ScenarioMapResult(
        scenario_name=scenario.name,
        facts=facts,
        instruction_blocks=instruction_blocks,
    )
