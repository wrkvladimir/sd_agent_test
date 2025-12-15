from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import re

from chat_app.schemas import Chunk, ConversationState, ScenarioDefinition, ScenarioNode


@dataclass
class ScenarioRunResult:
    context_text: str
    last_step_id: Optional[str]
    state: ConversationState


class ScenarioToolRunner:
    """
    Версия 0.1: существующий раннер сценариев с YAML-подобным special_instructions.
    """

    def __init__(self, tools: Optional[Dict[str, Any]] = None) -> None:
        self._tools: Dict[str, Any] = tools or {}

    def _render_template(
        self,
        text: str,
        state: ConversationState,
        tools_results: Dict[str, Dict[str, Any]],
    ) -> str:
        pattern = re.compile(r"\{=([^=]+)=\}")

        def replace(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            if expr.startswith("@"):
                inner = expr[1:]
                parts = inner.split(".")
                if not parts:
                    return "finderror"
                tool_name = parts[0]
                field = parts[1] if len(parts) > 1 else None
                tool_data = tools_results.get(tool_name) or {}
                if field is None:
                    value = tool_data or None
                else:
                    value = tool_data.get(field)
                return str(value) if value is not None else "finderror"

            if expr.startswith("dialog."):
                key = expr[len("dialog.") :]
                try:
                    if key == "name":
                        value = state.user_profile.name
                    elif key == "age":
                        value = state.user_profile.age
                    elif key == "message_index":
                        value = state.message_index
                    else:
                        value = None
                    return str(value) if value is not None else "finderror"
                except Exception:  # noqa: BLE001
                    return "finderror"

            return "finderror"

        return pattern.sub(replace, text)

    def _sort_key(self, node: ScenarioNode) -> List[int]:
        parts: List[int] = []
        for part in node.id.split("."):
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(0)
        return parts

    def _indent_block(self, text: str, prefix: str) -> str:
        return "\n".join(prefix + line if line else prefix for line in text.splitlines())

    async def run(
        self,
        scenario: ScenarioDefinition,
        state: ConversationState,
        user_message: str,
        kb_chunks: List[Chunk],
    ) -> ScenarioRunResult:
        _ = kb_chunks

        if state.message_index != 1:
            return ScenarioRunResult(context_text="", last_step_id=None, state=state)

        scenario_name = (scenario.name or "").lower()
        if "дню рожд" in scenario_name or "день рожд" in scenario_name:
            text = user_message.lower()
            birthday_triggers = [
                "день рождения",
                "днём рождения",
                "с днем рождения",
                "с днём рождения",
                "днюха",
                "днюху",
                "у меня др",
                "мой др",
                "сегодня др",
                "сегодня день рождения",
                " др ",
                " др.",
                " др,",
                "др ",
                "др.",
                "др,",
                "др?",
                "годиков",
                "исполнилось",
                "исполнится",
            ]
            if not any(trigger in text for trigger in birthday_triggers):
                return ScenarioRunResult(context_text="", last_step_id=None, state=state)

        tools_results: Dict[str, Dict[str, Any]] = {}
        text_blocks: List[str] = []
        conditional_blocks: List[Dict[str, Any]] = []

        nodes = sorted(scenario.code, key=self._sort_key)

        def ensure_tool_data(tool_name: str) -> None:
            if tool_name in tools_results:
                return

            if tool_name == "get_user_data":
                # В версии 0.1 сценарии используют уже заполненный профиль пользователя.
                tools_results[tool_name] = {
                    "name": state.user_profile.name,
                    "age": state.user_profile.age,
                }
                return

            func = self._tools.get(tool_name)
            if not func:
                tools_results[tool_name] = {}
                return
            try:
                result = func()
            except Exception:  # noqa: BLE001
                tools_results[tool_name] = {}
                return
            if isinstance(result, dict):
                tools_results[tool_name] = result
            else:
                tools_results[tool_name] = getattr(result, "__dict__", {"value": result})

        def add_text_block(text: str) -> None:
            rendered = self._render_template(text, state, tools_results)
            text_blocks.append(rendered)

        def add_conditional_block(node: ScenarioNode) -> None:
            condition = node.condition or ""
            true_texts: List[str] = []
            false_texts: List[str] = []

            for child in node.children or []:
                if child.type == "text" and child.text:
                    true_texts.append(self._render_template(child.text, state, tools_results))
            for child in node.else_children or []:
                if child.type == "text" and child.text:
                    false_texts.append(self._render_template(child.text, state, tools_results))

            conditional_blocks.append(
                {
                    "condition": {
                        "description": condition,
                        "user_message": user_message,
                    },
                    "when_true": {"texts": true_texts},
                    "when_false": {"texts": false_texts},
                }
            )

        for node in nodes:
            if node.type == "end":
                break
            if node.type == "tool" and node.tool:
                ensure_tool_data(node.tool)
                continue
            if node.type == "text" and node.text:
                add_text_block(node.text)
                continue
            if node.type == "if":
                add_conditional_block(node)
                continue

        if not text_blocks and not conditional_blocks:
            return ScenarioRunResult(context_text="", last_step_id=None, state=state)

        instructions = (
            "special_instructions описывает дополнительные сценарные указания.\n"
            "- blocks: список обязательных текстов-инструкций, которые нужно учитывать при формировании ответа.\n"
            "- blocks_with_conditions: список условных блоков. Для КАЖДОГО такого блока действуй так:\n"
            "  1. Смотри на pair condition.description и condition.user_message. user_message — это последнее сообщение пользователя.\n"
            "  2. Сначала реши, относится ли user_message по смыслу к теме из condition.description.\n"
            "     - Если НЕ относится, полностью игнорируй этот блок и НЕ используй ни when_true, ни when_false.\n"
            "  3. Если сообщение относится к той же теме:\n"
            "     - Считай условие ИСТИННЫМ, только если из user_message явно следует, что описанная ситуация выполняется\n"
            '       (например: "сегодня у меня день рождения").\n'
            "     - Считай условие ЛОЖНЫМ, только если из user_message явно следует, что описанная ситуация НЕ выполняется,\n"
            '       но тема та же (например: "день рождения был на прошлой неделе" или "мой день рождения в августе").\n'
            "  4. Если из user_message нельзя однозначно понять, выполняется условие или нет,\n"
            "     лучше полностью игнорировать этот блок и НЕ использовать when_false.\n"
            "  5. Если условие ИСТИННО — учитывай в ответе только тексты из when_true.texts.\n"
            "     Если условие ЛОЖНО — учитывай в ответе только тексты из when_false.texts.\n"
            "  6. Не делай логических выводов сверх явно заданных текстов; просто выбирай между when_true,\n"
            "     when_false или полным игнорированием блока.\n"
        )

        lines: List[str] = []
        lines.append("instructions: |")
        lines.append(self._indent_block(instructions, "  "))

        if text_blocks:
            lines.append("blocks:")
            for txt in text_blocks:
                lines.append("  - text: |")
                lines.append(self._indent_block(txt, "      "))

        if conditional_blocks:
            lines.append("blocks_with_conditions:")
            for cond in conditional_blocks:
                lines.append("  - condition:")
                lines.append(f'      description: "{cond["condition"]["description"]}"')
                lines.append(f'      user_message: "{cond["condition"]["user_message"]}"')
                lines.append("    when_true:")
                lines.append("      texts:")
                if cond["when_true"]["texts"]:
                    for txt in cond["when_true"]["texts"]:
                        lines.append(f'        - "{txt}"')
                else:
                    lines.append("        # нет текстов для ветки when_true")
                lines.append("    when_false:")
                lines.append("      texts:")
                if cond["when_false"]["texts"]:
                    for txt in cond["when_false"]["texts"]:
                        lines.append(f'        - "{txt}"')
                else:
                    lines.append("        # нет текстов для ветки when_false")

        context_text = "\n".join(lines)

        try:
            state.scenario_runs.append(
                {
                    "name": scenario.name,
                    "at_message_index": state.message_index,
                    "executed": True,
                    "ts": datetime.utcnow().isoformat(),
                }
            )
        except Exception:  # noqa: BLE001
            pass

        return ScenarioRunResult(context_text=context_text, last_step_id=None, state=state)
