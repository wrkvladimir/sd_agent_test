from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import re

from .schemas import Chunk, ConversationState, ScenarioDefinition, ScenarioNode


@dataclass
class ScenarioRunResult:
    """
    Result of executing a scenario for a single user message.

    context_text: accumulated text instructions that must be added
                  to the LLM prompt.
    last_step_id: identifier of the last executed node (for debugging).
    state:        updated ConversationState.
    """

    context_text: str
    last_step_id: Optional[str]
    state: ConversationState


class ScenarioToolRunner:
    """
    Engine responsible for executing JSON-defined scenarios.

    The concrete logic (text nodes, tool calls, condition handling and
    template substitutions) will be implemented here.
    """

    def __init__(self, tools: Optional[Dict[str, Any]] = None) -> None:
        # tools: mapping from tool name to callable
        self._tools: Dict[str, Any] = tools or {}

    def _render_template(
        self,
        text: str,
        state: ConversationState,
        tools_results: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Подстановка плейсхолдеров формата {=...=} в текст.

        Поддерживаем:
        - {=@tool_name.field=} -> tools_results[tool_name][field]
        - {=dialog.name=}      -> state.user_profile.name
        - {=dialog.age=}       -> state.user_profile.age
        - {=dialog.message_index=} -> state.message_index

        Если значение не найдено или произошла ошибка, подставляем литерал
        'finderror', чтобы LLM мог проигнорировать соответствующее предложение.
        """

        pattern = re.compile(r"\{=([^=]+)=\}")

        def replace(match: re.Match[str]) -> str:
            expr = match.group(1).strip()

            # Ссылка на результат инструмента: {=@tool.field=}
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

            # Ссылка на параметры диалога: {=dialog.*=}
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

            # Неизвестный формат плейсхолдера.
            return "finderror"

        return pattern.sub(replace, text)

    def _sort_key(self, node: ScenarioNode) -> List[int]:
        parts = []
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
        """
        Execute a single scenario for the given user message.

        Простейший precondition: сценарий применяется только к первому
        сообщению диалога (message_index == 1).\n
        Логика разбора нод:
        - text: добавляется в special_instructions как текстовый блок;
        - tool: подготавливает данные в tools_results (например, get_user_data);
        - if: превращается в условный блок, выбор ветки оставляем на усмотрение LLM;
        - end: завершает обработку сценария.
        """
        _ = kb_chunks  # пока не используется, но может быть полезен в будущих сценариях

        if state.message_index != 1:
            return ScenarioRunResult(context_text="", last_step_id=None, state=state)

        tools_results: Dict[str, Dict[str, Any]] = {}
        text_blocks: List[str] = []
        conditional_blocks: List[Dict[str, Any]] = []

        nodes = sorted(scenario.code, key=self._sort_key)

        def ensure_tool_data(tool_name: str) -> None:
            if tool_name in tools_results:
                return
            if tool_name == "get_user_data":
                # Используем уже заполненный профиль пользователя.
                tools_results[tool_name] = {
                    "name": state.user_profile.name,
                    "age": state.user_profile.age,
                    "birthday_date": state.user_profile.birthday_date,
                }
            else:
                # Для прочих тулов попытаемся вызвать зарегистрированный callable.
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
                    # Попробуем взять __dict__ или обернуть в словарь.
                    tools_results[tool_name] = getattr(result, "__dict__", {"value": result})

        def add_text_block(text: str) -> None:
            rendered = self._render_template(text, state, tools_results)
            text_blocks.append(rendered)

        def add_conditional_block(node: ScenarioNode) -> None:
            condition = node.condition or ""
            # Собираем тексты для веток.
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

        # Обходим ноды сценария.
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

        # Собираем YAML-подобный объект instructions + blocks + blocks_with_conditions.
        instructions = (
            "special_instructions описывает дополнительные сценарные указания.\n"
            "- blocks: список обязательных текстов-инструкций, которые нужно учитывать при формировании ответа.\n"
            "- blocks_with_conditions: список условных блоков; для каждого:\n"
            "  * condition.description — условие в текстовой форме.\n"
            "  * condition.user_message — исходное сообщение пользователя.\n"
            "  * when_true.texts — тексты, которые нужно учитывать, если условие выполняется.\n"
            "  * when_false.texts — тексты, которые нужно учитывать, если условие не выполняется.\n"
            "Модель должна, используя description и user_message, решить, выполняется ли условие.\n"
            "При сравнении description и user_message обращай внимание на ключевые детали (кто, что, когда).\n"
            "Если описание похоже, но отличается по важным параметрам (например, говорится о том же событии,\n"
            "но явно указано, что оно произошло в другой день), считай, что условие НЕ выполняется и используй when_false.\n"
            "Если user_message вообще не касается темы, описанной в condition.description (по смыслу нет совпадения),\n"
            "игнорируй этот условный блок целиком и не используй ни when_true, ни when_false.\n"
            "Если да — учитывать тексты из when_true; если нет — из when_false.\n"
            "Если по контексту нельзя решить, игнорировать этот условный блок.\n"
            "Не делай логических выводов сверх явно заданных текстов; просто выбирай между when_true и when_false.\n"
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
                lines.append(
                    f'      description: "{cond["condition"]["description"]}"'
                )
                lines.append(
                    f'      user_message: "{cond["condition"]["user_message"]}"'
                )
                # when_true
                lines.append("    when_true:")
                lines.append("      texts:")
                if cond["when_true"]["texts"]:
                    for txt in cond["when_true"]["texts"]:
                        lines.append(f'        - "{txt}"')
                else:
                    lines.append("        # нет текстов для ветки when_true")
                # when_false
                lines.append("    when_false:")
                lines.append("      texts:")
                if cond["when_false"]["texts"]:
                    for txt in cond["when_false"]["texts"]:
                        lines.append(f'        - "{txt}"')
                else:
                    lines.append("        # нет текстов для ветки when_false")

        context_text = "\n".join(lines)

        # Логируем факт выполнения сценария в состоянии диалога.
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
            # Ошибка логирования сценария не должна ломать основной поток.
            pass

        return ScenarioRunResult(context_text=context_text, last_step_id=None, state=state)
