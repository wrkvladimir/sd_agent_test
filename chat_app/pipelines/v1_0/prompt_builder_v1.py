from __future__ import annotations

from typing import Dict, List

from chat_app.schemas import Chunk, ConversationState, HistoryItem, MessageRole

from .graph_state import InstructionBlock, ToolsContext


class PromptBuilderV1:
    """
    v1.0 PromptBuilder: строит OpenAI-like messages без YAML, но со смысловыми секциями v0.1.
    """

    def build_messages(
        self,
        *,
        conv_state: ConversationState,
        history_tail: List[HistoryItem],
        kb_chunks: List[Chunk],
        tools_context: ToolsContext,
        user_message: str,
    ) -> List[Dict[str, str]]:
        tail_items = history_tail or []
        # История на этом шаге уже содержит текущее сообщение пользователя (append_user).
        # Чтобы не дублировать user_message в dialog_tail, отбрасываем последний item,
        # если он совпадает с текущим user_message.
        if tail_items and tail_items[-1].role == MessageRole.USER and tail_items[-1].content == user_message:
            tail_items = tail_items[:-1]
        tail_items = tail_items[-4:] if tail_items else []
        dialog_tail = "\n".join(f"{i.role.value}: {i.content}" for i in tail_items) if tail_items else ""
        dialog_summary = conv_state.summary or ""

        if kb_chunks:
            kb_lines: List[str] = ["База знаний (релевантные фрагменты):"]
            for idx, chunk in enumerate(kb_chunks, start=1):
                kb_lines.append(f"[{idx}] {chunk.text}")
            context_text = "\n".join(kb_lines)
        else:
            context_text = "Релевантных фрагментов базы знаний не найдено."

        blocks: List[InstructionBlock] = tools_context.get("instruction_blocks") or []

        required_blocks = [
            b for b in blocks if b.get("target") == "agent" and b.get("kind") == "required" and b.get("text")
        ]
        conditional_blocks = [
            b for b in blocks if b.get("target") == "agent" and b.get("kind") == "conditional" and b.get("payload")
        ]

        required_lines: List[str] = []
        for b in sorted(required_blocks, key=lambda x: int(x.get("priority") or 10)):
            required_lines.append(f"- {b.get('text')}")

        conditional_lines: List[str] = []
        for b in sorted(conditional_blocks, key=lambda x: int(x.get("priority") or 10)):
            payload = b.get("payload") or {}
            condition = str(payload.get("condition") or payload.get("condition_text") or "")
            when_true = payload.get("when_true") or []
            when_false = payload.get("when_false") or []
            policy = payload.get("apply_policy") or {}
            conditional_lines.append(f"- condition: {condition}")
            if policy:
                conditional_lines.append("  apply_policy:")
                for k, v in policy.items():
                    conditional_lines.append(f"    - {k}: {v}")
            if when_true:
                conditional_lines.append("  when_true:")
                for t in when_true:
                    conditional_lines.append(f"    - {t}")
            if when_false:
                conditional_lines.append("  when_false:")
                for t in when_false:
                    conditional_lines.append(f"    - {t}")

        system_parts: List[str] = []
        system_parts.append(
            "Ты — агент поддержки компании.\n"
            "Отвечай только на основе:\n"
            "1) context (база знаний),\n"
            "2) tools_context (сценарные инструкции и факты),\n"
            "3) dialog_summary и dialog_tail (наименее важный источник).\n"
            "Если источники противоречат — следуй приоритету выше.\n"
            "Не используй внешний мир или общие знания вне того, что дано.\n"
            "Если context пустой/недостаточный — честно скажи, что не нашёл точной информации, и предложи уточнение/эскалацию.\n"
            "Всегда отвечай на русском, дружелюбно и профессионально, 3–4 коротких предложения.\n"
            "Не используй эмодзи/смайлики.\n"
            "Не обещай того, чего нет в context (например: «мы обязательно сообщим», «передали разработчикам», «выйдет в следующем обновлении»).\n"
            "Не раскрывай ход рассуждений.\n"
            "dialog_params/message_index — порядковый номер текущего сообщения пользователя в диалоге (считаются только сообщения пользователя, а не ответы ассистента). "
            "Не пересчитывай этот номер по истории вручную, используй только значение из dialog_params.\n"
        )

        system_parts.append(
            "assistant_meta:\n"
            '- Если пользователь спросит "Кто ты?" — объясни, что ты виртуальный агент поддержки компании, работающий с базой знаний и сценариями.\n'
            '- Если спросит "О чем мы общаемся?" — используй dialog_summary и dialog_tail, чтобы кратко пересказать контекст.\n'
        )

        system_parts.append(
            "dialog_params:\n"
            f"- message_index: {conv_state.message_index}\n"
        )

        system_parts.append(f"dialog_summary:\n{dialog_summary}\n")
        system_parts.append(f"dialog_tail:\n{dialog_tail}\n")
        system_parts.append(f"context:\n{context_text}\n")

        if required_lines or conditional_lines:
            tc_parts: List[str] = []
            if required_lines:
                tc_parts.append("required_blocks:\n" + "\n".join(required_lines))
            if conditional_lines:
                tc_parts.append(
                    "conditional_blocks:\n"
                    "Правила применения условных блоков:\n"
                    "- Если сообщение пользователя НЕ относится к теме condition — игнорируй блок полностью.\n"
                    "- Если относится и явно TRUE — используй только when_true.\n"
                    "- Если относится и явно FALSE — используй только when_false.\n"
                    "- Если неясно — игнорируй блок и НЕ выбирай when_false по умолчанию.\n"
                    + "\n".join(conditional_lines)
                )
            system_parts.append("tools_context:\n" + "\n\n".join(tc_parts) + "\n")

        system_text = "\n".join(system_parts).strip()

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_message},
        ]
