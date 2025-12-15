from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from chat_app.schemas import ScenarioDefinition, ToolSpec
from chat_app.sgr.langchain_chain.pipeline import SgrConvertError, sgr_convert_via_langchain


logger = logging.getLogger("chat_app.sgr")

_TEMPLATE_PATTERN = re.compile(r"\{=([^=]+)=\}")


def _extract_template_refs(text: str) -> List[str]:
    return [m.group(1).strip() for m in _TEMPLATE_PATTERN.finditer(text or "")]


def _validate_templates(
    *,
    scenario: ScenarioDefinition,
    available_tools: List[ToolSpec],
) -> Dict[str, Any]:
    tool_by_name: Dict[str, ToolSpec] = {t.name: t for t in (available_tools or [])}
    available = set(tool_by_name.keys())

    referenced_tools: set[str] = set()
    unknown_tool_refs: List[str] = []
    invalid_expressions: List[str] = []
    unknown_fields: List[str] = []

    def visit_nodes(nodes) -> None:
        for node in nodes or []:
            if node.type == "text" and node.text:
                for expr in _extract_template_refs(node.text):
                    if expr.startswith("@"):
                        inner = expr[1:]
                        tool_name = inner.split(".", 1)[0].strip()
                        if tool_name:
                            referenced_tools.add(tool_name)
                            if tool_name not in available:
                                unknown_tool_refs.append(tool_name)
                                continue
                            parts = inner.split(".", 1)
                            if len(parts) == 2:
                                field = parts[1].strip()
                                if field:
                                    spec = tool_by_name.get(tool_name)
                                    props = (
                                        (spec.output_schema or {}).get("properties")
                                        if spec and isinstance(spec.output_schema, dict)
                                        else None
                                    )
                                    if isinstance(props, dict) and field not in props:
                                        unknown_fields.append(f"{tool_name}.{field}")
                        continue

                    if expr.startswith("dialog."):
                        continue

                    invalid_expressions.append(expr)

            if node.type == "if":
                visit_nodes(node.children or [])
                visit_nodes(node.else_children or [])

    visit_nodes(scenario.code)

    return {
        "template_refs": {
            "referenced_tools": sorted(referenced_tools),
            "unknown_tools": sorted(set(unknown_tool_refs)),
            "unknown_fields": sorted(set(unknown_fields)),
            "invalid_expressions": sorted(set(invalid_expressions)),
        }
    }


def _has_actionable_nodes(nodes) -> bool:
    for node in nodes or []:
        if getattr(node, "type", None) in ("text", "tool", "if"):
            return True
        if getattr(node, "type", None) == "end":
            continue
        if getattr(node, "children", None) or getattr(node, "else_children", None):
            if _has_actionable_nodes(getattr(node, "children", None) or []):
                return True
            if _has_actionable_nodes(getattr(node, "else_children", None) or []):
                return True
    return False


def _contains_if(nodes) -> bool:
    for node in nodes or []:
        if getattr(node, "type", None) == "if":
            return True
        if _contains_if(getattr(node, "children", None) or []):
            return True
        if _contains_if(getattr(node, "else_children", None) or []):
            return True
    return False


def _validate_scenario_or_raise(*, scenario: ScenarioDefinition, input_text: str) -> None:
    if not _has_actionable_nodes(scenario.code):
        raise ValueError("scenario has no actionable nodes (only end or empty actions)")
    if "если" in (input_text or "").lower() and not _contains_if(scenario.code):
        raise ValueError("input contains 'если' but scenario has no if-nodes")


async def sgr_convert_text(
    *,
    text: str,
    available_tools: List[ToolSpec],
    name_hint: Optional[str] = None,
    strict: bool = True,
    return_diagnostics: bool = True,
) -> Tuple[ScenarioDefinition, Dict[str, Any], List[str]]:
    scenario, diagnostics, questions = await sgr_convert_via_langchain(
        text=text,
        available_tools=available_tools,
        name_hint=name_hint,
        strict=strict,
        return_diagnostics=return_diagnostics,
    )

    try:
        _validate_scenario_or_raise(scenario=scenario, input_text=text)
        if return_diagnostics:
            diagnostics.update(_validate_templates(scenario=scenario, available_tools=available_tools))
    except Exception as exc:  # noqa: BLE001
        trace_id = str((diagnostics or {}).get("trace_id") or "").strip()
        raise SgrConvertError(
            trace_id=trace_id or uuid.uuid4().hex[:10],
            failed_step="10_static_validation",
            diagnostics={**(diagnostics or {}), "error": repr(exc)},
            last_llm_raw="",
        ) from exc

    return scenario, diagnostics, questions

