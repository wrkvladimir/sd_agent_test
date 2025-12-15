from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

from langgraph.graph import END, StateGraph

from chat_app.pipelines.v1_0.graph_state import AgentState, InstructionBlock, ScenarioMapResult, ToolsContext
from chat_app.pipelines.v1_0.subgraphs.scenario_engine import run_scenario_map
from chat_app.pipelines.v1_0.tool_registry import ToolRegistry
from chat_app.scenario_registry import ScenarioRegistry


LlmChat = Callable[[List[Dict[str, str]]], Awaitable[str]]
LlmChatJson = Callable[[List[Dict[str, str]], Dict[str, Any], str], Awaitable[Dict[str, Any]]]

logger = logging.getLogger("chat_app.tools_subgraph_v1_0")


def build_tools_subgraph(
    *,
    scenario_registry: ScenarioRegistry,
    tools: ToolRegistry,
    llm_chat: LlmChat,
    llm_chat_json: LlmChatJson,
):
    class ToolsSubgraphState(AgentState, total=False):
        scenario_map_results: List[ScenarioMapResult]
        scenario_decisions: Dict[str, List[str]]
        scenario_sources_with_condition: List[str]

    graph = StateGraph(ToolsSubgraphState)

    async def init_tools_state(state: ToolsSubgraphState) -> Dict:
        tools_context: ToolsContext = {
            "facts": {},
            "instruction_blocks": [],
            "applied": [],
        }
        return {"tools_context": tools_context}

    async def scenario_map_node(state: ToolsSubgraphState) -> Dict:
        scenarios = [s for s in scenario_registry.all().values() if getattr(s, "enabled", True)]
        if not scenarios:
            return {"scenario_map_results": []}

        async def _run(scn):
            return await run_scenario_map(state=state, scenario=scn, tools=tools)

        results = await asyncio.gather(*(_run(scn) for scn in scenarios))
        mapped: List[ScenarioMapResult] = [r for r in results if r is not None]  # type: ignore[comparison-overlap]
        return {"scenario_map_results": mapped}

    async def scenario_reduce_node(state: ToolsSubgraphState) -> Dict:
        tools_context: ToolsContext = state.get("tools_context") or {}
        results: List[ScenarioMapResult] = state.get("scenario_map_results") or []

        facts = dict(tools_context.get("facts") or {})
        instruction_blocks: List[InstructionBlock] = list(tools_context.get("instruction_blocks") or [])

        for result in results:
            for k, v in (result.facts or {}).items():
                facts.setdefault(k, v)
            instruction_blocks.extend(result.instruction_blocks or [])

        tools_context = {
            "facts": facts,
            "instruction_blocks": instruction_blocks,
            # applied будет собран позже (после decisions + summarize), чтобы не отмечать
            # сценарий как применённый при decision=ignore/unknown.
            "applied": list(tools_context.get("applied") or []),
        }

        return {"tools_context": tools_context}

    async def _decide_condition_via_llm(
        *,
        condition: str,
        user_message: str,
        message_index: Optional[int],
        when_true: List[str],
        when_false: List[str],
        facts: Dict[str, Dict[str, Any]],
    ) -> Tuple[Literal["ignore", "true", "false", "unknown"], Optional[str]]:
        """
        Production: LLM-driven control flow decision for a single conditional block.

        Returns:
        - decision: ignore|true|false|unknown
        - followup_question: optional clarification question for unknown
        """
        facts_preview = {}
        for k, v in facts.items():
            if k == "tool:get_user_data":
                facts_preview[k] = {kk: v.get(kk) for kk in ("name", "age") if kk in v}

        system = (
            "Ты — модуль принятия решения по условному ветвлению сценария поддержки.\n"
            "Нужно определить, относится ли последнее сообщение пользователя к условию, и если относится — истинно оно, ложно или неоднозначно.\n"
            "Важно: ты решаешь применимость сценарной ветки по последнему сообщению пользователя и dialog_params, а не «истинность во внешнем мире».\n"
            "Верни СТРОГО JSON без лишнего текста формата:\n"
            '{\n'
            '  "decision": "ignore|true|false|unknown",\n'
            '  "followup_question": "..." \n'
            "}\n"
            "Правила:\n"
            "- ignore: если сообщение не относится к теме условия.\n"
            "- true: только если из сообщения ЯВНО следует, что условие выполняется.\n"
            "- false: только если из сообщения ЯВНО следует, что условие НЕ выполняется, но тема та же.\n"
            "- unknown: если тема та же, но нельзя уверенно выбрать true/false.\n"
            "- Если условие про параметры диалога (например, dialog.message_index или «первое сообщение») —\n"
            "  используй dialog.message_index для принятия решения и НЕ выбирай ignore по причине «не по теме».\n"
            "- Если условие про «второе/третье/четвертое сообщение» — это строгое сравнение dialog.message_index с 2/3/4.\n"
            "- Если условие сформулировано как «Пользователь написал/сказал/сообщил ... что ...» — трактуй это как проверку факта высказывания в последнем сообщении.\n"
            "  TRUE: если пользователь в последнем сообщении утверждает это.\n"
            "  FALSE: если пользователь в последнем сообщении явно утверждает обратное.\n"
            "  UNKNOWN: только если по последнему сообщению реально непонятно, утверждает ли он это.\n"
            "  Не требуй внешнюю верификацию: слова пользователя достаточно для true/false.\n"
            "- Если в сообщении пользователя явно есть указание на время (например, слово 'сегодня'), и это соответствует смыслу condition,\n"
            "  не выбирай unknown: выбери true или false.\n"
            "Для unknown задавай только вопрос про уточнение формулировки последнего сообщения (без запроса персональных данных).\n"
            "Запрещено просить персональные данные или «верификацию» (например: дату рождения, паспорт, телефон, адрес, email, номер карты).\n"
            'Для unknown сформулируй короткий уточняющий вопрос (followup_question), иначе пустую строку.\n'
        )
        user = (
            f"Условие:\n{condition}\n\n"
            f"dialog_params:\n{json.dumps({'message_index': message_index}, ensure_ascii=False)}\n\n"
            f"Сообщение пользователя:\n{user_message}\n\n"
            f"Факты:\n{json.dumps(facts_preview, ensure_ascii=False)}\n\n"
            f"Ветка when_true (для понимания смысла):\n{json.dumps(when_true[:5], ensure_ascii=False)}\n\n"
            f"Ветка when_false (для понимания смысла):\n{json.dumps(when_false[:5], ensure_ascii=False)}\n"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "followup_question"],
            "properties": {
                "decision": {"type": "string", "enum": ["ignore", "true", "false", "unknown"]},
                "followup_question": {"type": "string"},
            },
        }
        data = await llm_chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema,
            "condition_decision",
        )
        decision = data.get("decision") if isinstance(data, dict) else None
        if decision not in ("ignore", "true", "false", "unknown"):
            decision = "unknown"
        followup = (data.get("followup_question") if isinstance(data, dict) else "") or ""
        followup_question = str(followup).strip()
        logger.info(
            "condition_decision_v1_0 decision=%s followup_present=%s condition=%r user_message=%r",
            decision,
            bool(followup_question),
            condition[:160],
            user_message[:160],
        )
        return decision, (followup_question or None)

    async def conditions_decide_node(state: ToolsSubgraphState) -> Dict:
        tools_context: ToolsContext = state.get("tools_context") or {}
        blocks: List[InstructionBlock] = list(tools_context.get("instruction_blocks") or [])
        user_message = state.get("user_message") or ""
        facts = tools_context.get("facts") or {}
        conv_state = state.get("conv_state")
        message_index = conv_state.message_index if conv_state is not None else None

        conditional_blocks = [b for b in blocks if b.get("kind") == "conditional" and b.get("target") == "agent"]
        scenario_sources_with_condition = sorted({(b.get("source") or "").strip() for b in conditional_blocks if (b.get("source") or "").strip()})

        async def decide(block: InstructionBlock):
            payload = block.get("payload") or {}
            condition = str(payload.get("condition") or payload.get("condition_text") or block.get("text") or "")
            if not condition:
                return ("ignore", block, [])
            when_true = [str(x) for x in (payload.get("when_true") or [])]
            when_false = [str(x) for x in (payload.get("when_false") or [])]

            decision, followup = await _decide_condition_via_llm(
                condition=condition,
                user_message=user_message,
                message_index=message_index,
                when_true=when_true,
                when_false=when_false,
                facts=facts,
            )

            applied_blocks: List[InstructionBlock] = []
            if decision == "true":
                for idx, txt in enumerate(when_true, start=1):
                    applied_blocks.append(
                        {
                            "id": f"{block.get('id')}:applied:true:{idx}",
                            "source": block.get("source") or "",
                            "target": "agent",
                            "kind": "raw",
                            "priority": int(block.get("priority") or 10),
                            "text": txt,
                        }
                    )
            elif decision == "false":
                for idx, txt in enumerate(when_false, start=1):
                    applied_blocks.append(
                        {
                            "id": f"{block.get('id')}:applied:false:{idx}",
                            "source": block.get("source") or "",
                            "target": "agent",
                            "kind": "raw",
                            "priority": int(block.get("priority") or 10),
                            "text": txt,
                        }
                    )
            elif decision == "unknown" and followup:
                # Для unknown сохраняем только уточняющий вопрос (без "безусловных" текстов сценария).
                applied_blocks.append(
                    {
                        "id": f"{block.get('id')}:applied:unknown:followup",
                        "source": block.get("source") or "",
                        "target": "agent",
                        "kind": "required",
                        "priority": int(block.get("priority") or 10),
                        "text": (
                            "В конце ответа задай уточняющий вопрос (сначала ответь на основной вопрос пользователя):\n"
                            f"{followup}"
                        ),
                    }
                )

            return (decision, block, applied_blocks)

        results = await asyncio.gather(*(decide(b) for b in conditional_blocks))

        new_blocks: List[InstructionBlock] = []
        applied_from_conditions: List[InstructionBlock] = []
        scenario_decisions: Dict[str, List[str]] = {}
        for decision, block, applied_blocks in results:
            src = (block.get("source") or "").strip()
            if src:
                scenario_decisions.setdefault(src, [])
                scenario_decisions[src].append(decision)
            # Для продакшн-логики conditional блоки не нужны в промпте:
            # мы либо применяем ветку, либо игнорируем, либо просим уточнение.
            if decision != "ignore":
                new_blocks.append(
                    {
                        "id": f"{block.get('id')}:decision",
                        "source": block.get("source") or "",
                        "target": "judge",
                        "kind": "rule",
                        "priority": int(block.get("priority") or 10),
                        "text": (
                            f"Условный блок {block.get('id')} был оценён как decision={decision}. "
                            "Проверь, что ответ не противоречит этому решению и не содержит утверждений из другой ветки."
                        ),
                    }
                )
            applied_from_conditions.extend(applied_blocks)

        # Пересобираем blocks: все не-conditional + отфильтрованные conditional + применённые required.
        keep = [b for b in blocks if not (b.get("kind") == "conditional" and b.get("target") == "agent")]
        blocks_out = keep + new_blocks + applied_from_conditions

        tools_context["instruction_blocks"] = blocks_out
        return {
            "tools_context": tools_context,
            "scenario_decisions": scenario_decisions,
            "scenario_sources_with_condition": scenario_sources_with_condition,
        }

    async def scenario_summarize_to_imperatives(state: ToolsSubgraphState) -> Dict:
        """
        Сжимает «сырые» блоки сценариев (kind=raw,target=agent) в короткие imperative-инструкции (kind=required).
        Делается отдельно для каждого сценария (source=scenario.name).
        """
        tools_context: ToolsContext = state.get("tools_context") or {}
        blocks: List[InstructionBlock] = list(tools_context.get("instruction_blocks") or [])
        scenario_decisions = state.get("scenario_decisions") or {}
        sources_with_condition = set(state.get("scenario_sources_with_condition") or [])
        scenario_names = [r.scenario_name for r in (state.get("scenario_map_results") or [])]

        def _decisions_for(source: str) -> set[str]:
            return set(str(x) for x in (scenario_decisions.get(source) or []) if x)

        enabled_for_summarize: set[str] = set()
        enabled_overall: set[str] = set()
        for name in scenario_names:
            dec = _decisions_for(name)
            if "true" in dec or "false" in dec:
                enabled_for_summarize.add(name)
                enabled_overall.add(name)
            elif "unknown" in dec:
                # unknown: оставляем только followup (и связанные judge rules), но не тянем "безусловные" тексты.
                enabled_overall.add(name)
            elif name not in sources_with_condition:
                # сценарий без condition-узлов — разрешаем как есть.
                enabled_for_summarize.add(name)
                enabled_overall.add(name)
            # ignore-only: не добавляем вообще ничего из сценария

        # Сначала отфильтруем блоки по enabled_overall:
        if scenario_names:
            filtered: List[InstructionBlock] = []
            for b in blocks:
                src = (b.get("source") or "").strip()
                if src in scenario_names and src not in enabled_overall:
                    continue
                # Для unknown: запрещаем "безусловные" raw-тексты сценария (они не должны попадать даже в summarize).
                if src in scenario_names and src in enabled_overall and src not in enabled_for_summarize:
                    if b.get("target") == "agent" and b.get("kind") == "raw":
                        continue
                filtered.append(b)
            blocks = filtered

        raw_blocks = [
            b for b in blocks if b.get("target") == "agent" and b.get("kind") == "raw" and (b.get("text") or "").strip()
        ]
        # Если ни один scenario не должен суммаризироваться — выходим, но пересобираем applied.
        if not raw_blocks:
            applied_sources: List[str] = []
            for b in blocks:
                src = (b.get("source") or "").strip()
                if src and src in scenario_names and b.get("target") == "agent" and b.get("kind") == "required":
                    applied_sources.append(src)
            tools_context["instruction_blocks"] = blocks
            tools_context["applied"] = [{"kind": "scenario", "name": s} for s in sorted(set(applied_sources))]
            return {"tools_context": tools_context}

        by_source: Dict[str, List[str]] = {}
        for b in raw_blocks:
            src = (b.get("source") or "").strip() or "unknown_scenario"
            if scenario_names and src in scenario_names and src not in enabled_for_summarize:
                continue
            by_source.setdefault(src, []).append(str(b.get("text") or "").strip())

        name = state.get("conv_state").user_profile.name if state.get("conv_state") else ""
        age = state.get("conv_state").user_profile.age if state.get("conv_state") else None

        async def summarize_one(source: str, texts: List[str]) -> Dict[str, Any]:
            system = (
                "Ты — модуль сжатия сценария поддержки в короткие imperative-инструкции для основного агента.\n"
                "На входе: куски текста сценария (после подстановок) и контекст о пользователе.\n"
                "На выходе: короткие обязательные инструкции, без пояснений и лишнего текста.\n"
                "Инструкции должны сохранять смысл сценария и быть применимыми при ответе на текущее сообщение пользователя.\n"
                "Если сценарий не добавляет ничего полезного для ответа — верни пустой список.\n"
                "Верни СТРОГО JSON:\n"
                '{\n'
                '  "agent_imperatives": ["..."],\n'
                '  "judge_rules": ["..."]\n'
                "}\n"
                "Правила:\n"
                "- agent_imperatives: 0..8 строк, каждая — короткая команда (imperative), без воды.\n"
                "- judge_rules: 0..8 строк, правила для LLM-судьи (как проверять ответ), без воды.\n"
                "- Не повторяй исходный текст сценария дословно, если он болтливый — сжимай.\n"
                "- Не добавляй новых фактов.\n"
                "- Если известно имя пользователя — требуй обращения по имени и укажи само имя.\n"
                "- Не используй эмодзи.\n"
            )
            user = (
                f"Сценарий: {source}\n\n"
                f"Последнее сообщение пользователя:\n{state.get('user_message') or ''}\n\n"
                "Известные факты о пользователе:\n"
                f"- name: {name or ''}\n"
                f"- age: {'' if age is None else age}\n"
                "Куски сценария (после подстановок):\n"
                + "\n".join(f"{i}. {t}" for i, t in enumerate(texts[:50], start=1))
            )
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent_imperatives", "judge_rules"],
                "properties": {
                    "agent_imperatives": {"type": "array", "items": {"type": "string"}},
                    "judge_rules": {"type": "array", "items": {"type": "string"}},
                },
            }
            data = await llm_chat_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                schema,
                "scenario_imperatives",
            )
            if not isinstance(data, dict):
                return {"agent_imperatives": [], "judge_rules": []}
            return data

        results = await asyncio.gather(*(summarize_one(src, texts) for src, texts in by_source.items()))

        # Удаляем raw-блоки из итоговых instruction_blocks и добавляем сжатые required/rule.
        keep_blocks = [b for b in blocks if not (b.get("target") == "agent" and b.get("kind") == "raw")]
        out_blocks: List[InstructionBlock] = list(keep_blocks)

        for (source, _texts), data in zip(by_source.items(), results, strict=False):
            imperatives = data.get("agent_imperatives") if isinstance(data, dict) else []
            rules = data.get("judge_rules") if isinstance(data, dict) else []

            # Fallback: если модель вернула пусто/мусор — оставим один короткий блок из первых строк.
            cleaned_imperatives = [str(x).strip() for x in (imperatives or []) if str(x).strip()]
            if not cleaned_imperatives:
                fallback = [t.strip() for t in by_source.get(source, []) if t.strip()][:3]
                cleaned_imperatives = fallback

            for idx, text in enumerate(cleaned_imperatives[:8], start=1):
                out_blocks.append(
                    {
                        "id": f"scenario:{source}:imperative:{idx}",
                        "source": source,
                        "target": "agent",
                        "kind": "required",
                        "priority": 10,
                        "text": text,
                    }
                )

            cleaned_rules = [str(x).strip() for x in (rules or []) if str(x).strip()]
            for idx, text in enumerate(cleaned_rules[:8], start=1):
                out_blocks.append(
                    {
                        "id": f"scenario:{source}:judge_rule:summarized:{idx}",
                        "source": source,
                        "target": "judge",
                        "kind": "rule",
                        "priority": 10,
                        "text": text,
                    }
                )

        tools_context["instruction_blocks"] = out_blocks
        applied_sources: List[str] = []
        if scenario_names:
            for b in out_blocks:
                src = (b.get("source") or "").strip()
                if src and src in scenario_names and b.get("target") == "agent" and b.get("kind") == "required":
                    applied_sources.append(src)
        tools_context["applied"] = [{"kind": "scenario", "name": s} for s in sorted(set(applied_sources))]
        logger.info(
            "scenario_imperatives_v1_0 sources=%s raw_blocks=%d out_blocks=%d",
            list(by_source.keys()),
            len(raw_blocks),
            len(out_blocks),
        )
        return {"tools_context": tools_context}

    graph.add_node("init_tools_state", init_tools_state)
    graph.add_node("scenario_map", scenario_map_node)
    graph.add_node("scenario_reduce", scenario_reduce_node)
    graph.add_node("conditions_decide", conditions_decide_node)
    graph.add_node("scenario_summarize", scenario_summarize_to_imperatives)

    graph.set_entry_point("init_tools_state")
    graph.add_edge("init_tools_state", "scenario_map")
    graph.add_edge("scenario_map", "scenario_reduce")
    graph.add_edge("scenario_reduce", "conditions_decide")
    graph.add_edge("conditions_decide", "scenario_summarize")
    graph.add_edge("scenario_summarize", END)

    return graph.compile()
