from __future__ import annotations

from typing import Any, Dict, List

import logging

from .schemas import Chunk, ConversationState, HistoryItem


logger = logging.getLogger("chat_app.prompting")


class PromptBuilder:
    """
    Responsible for constructing LLM prompts from:
    - conversation state and history
    - scenario-generated context
    - retrieved KB chunks
    - current user message
    """

    def build_prompt(
        self,
        state: ConversationState,
        history_tail: List[HistoryItem],
        scenario_context: str,
        kb_chunks: List[Chunk],
        user_message: str,
    ) -> Dict[str, Any]:
        """
        Build a prompt payload suitable for the downstream LLM client.

        YAML-подобный промпт с полями:
        - system, assistant_meta
        - dialog_params, dialog_summary, dialog_tail
        - context (чанки базы знаний)
        - special_instructions (из tools / сценариев)
        - new_user_message
        """
        # dialog_params
        name = state.user_profile.name or ""
        age = state.user_profile.age if state.user_profile.age is not None else ""
        user_birthday = state.user_profile.birthday_date or ""
        # Вычисление birthday_today пока не реализовано, используем False.
        birthday_today = False

        # dialog_summary
        dialog_summary = state.summary or ""

        # dialog_tail: берём последние 3 сообщения из истории.
        tail_items = history_tail[-3:] if history_tail else []
        dialog_tail_lines: List[str] = []
        for item in tail_items:
            dialog_tail_lines.append(f"  - role: {item.role.value}\n    content: {item.content!r}")
        dialog_tail_block = "\n".join(dialog_tail_lines) if dialog_tail_lines else "  # нет предыдущих сообщений"

        # context: чанки базы знаний.
        if kb_chunks:
            kb_lines: List[str] = ["База знаний (релевантные фрагменты):"]
            for idx, chunk in enumerate(kb_chunks, start=1):
                kb_lines.append(f"[{idx}] {chunk.text}")
            context_block = "\n".join(kb_lines)
        else:
            context_block = "Релевантных фрагментов базы знаний не найдено."

        # special_instructions: из сценариев / tools (YAML-объект с полями
        # instructions, blocks и blocks_with_conditions).
        special_instructions = scenario_context.strip() if scenario_context else ""

        # Сборка YAML-подобного промпта как одного system-сообщения.
        yaml_parts: List[str] = []

        yaml_parts.append(
            "system: |\n"
            "  Ты — агент технической поддержки.\n"
            "  Отвечай только на основе поля context и, при наличии, special_instructions.\n"
            "  Если разные источники противоречат друг другу, используй приоритет (от более важного к менее важному):\n"
            "    1) context,\n"
            "    2) special_instructions,\n"
            "    3) dialog_summary и dialog_tail. - наименее важный\n"
            "  Не считай свои прошлые ответы из dialog_tail более достоверными, чем context или special_instructions.\n"
            "  Не используй внешний мир или общие знания вне того, что явно дано в этом промпте.\n"
            "  Если context пустой, недостаточный или нерелевантный — честно напиши, что не нашёл точного ответа\n"
            "  и предложи эскалацию специалисту или переформулировку вопроса.\n"
            "  Всегда отвечай на русском языке и в формате диалога, дружелюбно и профессионально.\n"
            "  Ты понимаешь, что ты чат-агент поддержки.\n"
            "  Не раскрывай ход рассуждений, не описывай, как ты думаешь; отвечай кратко и по делу.\n"
            "  Если special_instructions не пустой, используй его как дополнительные инструкции,\n"
            "  но не нарушай правила из system.\n"
            "  Ответ нужно дать на последнее сообщение пользователя new_user_message,\n"
            "  учитывая dialog_summary и dialog_tail.\n"
            "  Не выдумывай факты, которых нет в context или special_instructions.\n"
            "  Не перечисляй dialog_params в ответе, если пользователь явно об этом не спрашивает.\n"
            "  Отвечай обычным текстом на русском, не в формате YAML и не повторяя структуру промпта.\n"
            "  Если dialog_params/message_index равен 1, то это первое сообщение в диалоге\n"
            "   - поздоровайся, если нет - не здоровайся, продолжай диалог словно он длится какое то время.\n"
            "  Не придумывай названия разделов, кнопок, экранов, статусов и других элементов интерфейса и системы,\n"
            "  о которой может идти речь. Ты знаешь только то, что попало в данный промпт, остальное не выдумывай,\n"
            "  если эти элементы прямо не указаны в context или special_instructions.\n"
            "  Старайся укладываться в 3–4 коротких предложения; списки используй только если вопрос явно требует шагов.\n"
        )

        yaml_parts.append(
            "assistant_meta: |\n"
            '  Если пользователь спросит "Кто ты?" — объясни, что ты виртуальный агент поддержки компании,\n'
            "  работающий с базой знаний и сценариями.\n"
            '  Если спросит "О чем мы общаемся?" — используй dialog_summary и dialog_tail,\n'
            "  чтобы кратко пересказать контекст диалога.\n"
        )

        yaml_parts.append(
            "dialog_params:\n"
            f"  message_index: {state.message_index}\n"
            f"  name: {name!r}\n"
            f"  age: {age!r}\n"
            f"  user_birthday: {user_birthday!r}\n"
            f"  birthday_today: {str(birthday_today).lower()}\n"
        )

        yaml_parts.append("dialog_summary: |\n" + f"  {dialog_summary}\n")

        yaml_parts.append("dialog_tail:\n" + dialog_tail_block + "\n")

        yaml_parts.append("context: |\n" + "  " + context_block.replace("\n", "\n  ") + "\n")

        yaml_parts.append(
            "special_instructions: |\n"
            + ("  " + special_instructions.replace("\n", "\n  ") + "\n" if special_instructions else "  \n")
        )

        yaml_parts.append("new_user_message: |\n" + "  " + user_message.replace("\n", "\n  ") + "\n")

        yaml_prompt = "\n".join(yaml_parts)

        logger.info("built_yaml_prompt:\n%s", yaml_prompt)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": yaml_prompt},
            {"role": "user", "content": user_message},
        ]

        return {"messages": messages}
