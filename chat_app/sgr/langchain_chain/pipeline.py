from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIStatusError, RateLimitError
from pydantic import BaseModel

from chat_app.config import settings
from chat_app.schemas import ScenarioDefinition, ToolSpec
from chat_app.runtime_config import get_effective_openai_api_key
from chat_app.runtime_config import get_effective_openai_api_keys, mark_openai_api_key_rate_limited

from .models import (
    Step1ExtractIntents,
    Step2GateAndCritique,
    Step3ToolsAndTemplates,
)


logger = logging.getLogger("chat_app.sgr")

T = TypeVar("T", bound=BaseModel)


class SgrConvertError(Exception):
    def __init__(
        self,
        *,
        trace_id: str,
        failed_step: str,
        diagnostics: Dict[str, Any],
        last_llm_raw: str,
    ) -> None:
        super().__init__(f"SGR convert failed at {failed_step} trace={trace_id}")
        self.trace_id = trace_id
        self.failed_step = failed_step
        self.diagnostics = diagnostics
        self.last_llm_raw = last_llm_raw

    def to_422(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "failed_step": self.failed_step,
            "diagnostics": self.diagnostics,
            "last_llm_raw": self.last_llm_raw,
        }


def _sgr_timeout_s() -> float:
    raw = os.getenv("SGR_TIMEOUT_S", "").strip()
    try:
        return float(raw) if raw else 35.0
    except Exception:  # noqa: BLE001
        return 35.0


def _sgr_trace_dir(trace_id: str) -> Path:
    base = Path(os.getenv("SGR_TRACE_DIR", "/tmp/sgr_traces"))
    path = base / trace_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty llm output")

    # Try strict first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001
        pass

    # Try to recover JSON from code fences / surrounding text.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object found in llm output")

    candidate = raw[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("parsed json is not an object")
    return obj


@dataclass
class _StepTrace:
    step: str
    request_path: str
    response_path: str
    duration_s: float
    model: str
    headers: Dict[str, Any]


async def _call_llm_step(
    *,
    llm: ChatOpenAI,
    trace_id: str,
    step: str,
    system: str,
    user: str,
    out_model: Type[T],
    timeout_s: float,
    trace_dir: Path,
) -> Tuple[T, str, _StepTrace, Dict[str, Any], Dict[str, Any]]:
    started = time.monotonic()

    req_path = trace_dir / f"{step}.request.json"
    resp_path = trace_dir / f"{step}.response.json"

    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    req_payload: Dict[str, Any] = {
        "model": settings.sgr_model,
        "timeout_s": timeout_s,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req_path.write_text(json.dumps(req_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if settings.sgr_log_prompts:
        logger.info("sgr_step_start trace=%s step=%s request_path=%s", trace_id, step, str(req_path))

    last_raw = ""
    headers: Dict[str, Any] = {}
    resp_payload: Dict[str, Any] = {}

    keys = get_effective_openai_api_keys()
    if not keys:
        raise RuntimeError("OPENAI_API_KEY is not set")

    last_exc: Exception | None = None
    for attempt in range(max(1, len(keys))):
        llm_attempt = llm if attempt == 0 else _build_llm()
        try:
            msg = await asyncio.wait_for(llm_attempt.ainvoke(messages), timeout=timeout_s)
            last_raw = str(getattr(msg, "content", "") or "")
            meta = getattr(msg, "response_metadata", None) or {}
            headers = dict(meta.get("headers") or {}) if isinstance(meta, dict) else {}
            parsed_obj = _extract_json_object(last_raw)
            parsed = out_model.model_validate(parsed_obj)
            resp_payload = {
                "raw": last_raw,
                "parsed_json": parsed_obj,
                "validated_output": cast(dict, parsed.model_dump()),
                "response_metadata": meta,
            }
            resp_path.write_text(json.dumps(resp_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            break
        except RateLimitError as exc:
            last_exc = exc
            mark_openai_api_key_rate_limited()
            continue
        except APIStatusError as exc:
            last_exc = exc
            if getattr(exc, "status_code", None) == 429:
                mark_openai_api_key_rate_limited()
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            duration_s = time.monotonic() - started
            if settings.sgr_log_prompts:
                logger.info(
                    "sgr_step_error trace=%s step=%s duration_s=%.3f error=%r response_path=%s",
                    trace_id,
                    step,
                    duration_s,
                    exc,
                    str(resp_path),
                )
            raise
    else:
        assert last_exc is not None
        duration_s = time.monotonic() - started
        if settings.sgr_log_prompts:
            logger.info(
                "sgr_step_error trace=%s step=%s duration_s=%.3f error=%r response_path=%s",
                trace_id,
                step,
                duration_s,
                last_exc,
                str(resp_path),
            )
        raise last_exc

    duration_s = time.monotonic() - started
    if settings.sgr_log_prompts:
        logger.info(
            "sgr_step_end trace=%s step=%s duration_s=%.3f response_path=%s",
            trace_id,
            step,
            duration_s,
            str(resp_path),
        )

    trace = _StepTrace(
        step=step,
        request_path=str(req_path),
        response_path=str(resp_path),
        duration_s=duration_s,
        model=settings.sgr_model,
        headers=headers,
    )
    return parsed, last_raw, trace, req_payload, resp_payload


def _build_llm() -> ChatOpenAI:
    api_key = get_effective_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return ChatOpenAI(
        model_name=settings.sgr_model,
        openai_api_key=api_key,
        openai_api_base=settings.llm_base_url,
        temperature=0.0,
        request_timeout=_sgr_timeout_s(),
        max_retries=0,
        include_response_headers=True,
    )


_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # emoji + symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)

_TEMPLATE_REF_RE = re.compile(r"\{=([^=]+)=\}")


def _strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text or "")


def _clean_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").strip()
    t = t.replace("```", "").strip()
    t = _strip_emojis(t).strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _extract_tool_refs(text: str) -> List[str]:
    refs: List[str] = []
    for m in _TEMPLATE_REF_RE.finditer(text or ""):
        expr = (m.group(1) or "").strip()
        if not expr.startswith("@"):
            continue
        inner = expr[1:]
        tool_name = inner.split(".", 1)[0].strip()
        if tool_name:
            refs.append(tool_name)
    return refs


def _looks_like_condition_check_intent(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    if not re.search(r"(?i)\b(определ|провер|выясн|убед|понят)\b", t):
        return False
    # Typical non-actionable meta-check phrasing.
    if "является ли" in t:
        return True
    if "сегодня" in t and ("день рождения" in t or "др" in t):
        return True
    if "дата" in t and ("сегодня" in t or "текущ" in t):
        return True
    return False


def _filter_questions(questions: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for q in questions or []:
        qq = _clean_text(str(q))
        if not qq:
            continue
        low = qq.lower()
        # We treat conditions as semantic checks on user message; do not ask "how to determine".
        if re.search(r"(?i)\bкак\s+(определ|провер|понят)\b", low):
            continue
        # Avoid redundant yes/no questions for conditions like "user wrote ...".
        if re.search(r"(?i)\b(сегодня|ваш|у вас)\b.*\bдень рождения\b", low):
            continue
        # Tool questions are replaced by missing_tools diagnostics.
        if re.search(r"(?i)\bкакой\s+инструмент\b|\bкакой\s+метод\b|\bкак\s+получить\b", low):
            continue
        if qq not in seen:
            out.append(qq)
            seen.add(qq)
    return out


def _scenario_name(name_hint: Optional[str], *, text: str, trace_id: str) -> str:
    if (name_hint or "").strip():
        return str(name_hint or "").strip()
    base = (text or "").strip()
    if base:
        base = re.sub(r"\s+", " ", base)
        return base[:72]
    return f"sgr:{trace_id}"


def _atomic_text_nodes(parent_id: str, branch_index: int, texts: List[str]) -> List[Dict[str, Any]]:
    cleaned: List[str] = []
    for t in texts or []:
        for line in str(t or "").splitlines():
            line = line.strip()
            if line:
                cleaned.append(line)
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return [{"id": f"{parent_id}.{branch_index}", "type": "text", "text": cleaned[0]}]
    return [
        {"id": f"{parent_id}.{branch_index}.{i}", "type": "text", "text": txt}
        for i, txt in enumerate(cleaned, start=1)
    ]


def _assemble_scenario(
    *,
    trace_id: str,
    input_text: str,
    name_hint: Optional[str],
    explicit_else_noop: bool,
    strict: bool,
    step2: Step2GateAndCritique,
    step3: Step3ToolsAndTemplates,
) -> ScenarioDefinition:
    intent_by_id = {i.id: i for i in (step2.intents or [])}

    tools_to_call = list(dict.fromkeys(step3.tools_to_call or []))
    templates_global = [t for t in (step3.templates or []) if t.target == "global"]
    templates_then = [t for t in (step3.templates or []) if t.target == "condition_then"]
    templates_else = [t for t in (step3.templates or []) if t.target == "condition_else"]

    code: List[Dict[str, Any]] = []
    next_id = 1

    for tool_name in tools_to_call:
        code.append({"id": str(next_id), "type": "tool", "tool": tool_name})
        next_id += 1

    def append_text_nodes(text: str) -> None:
        nonlocal next_id
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            code.append({"id": str(next_id), "type": "text", "text": line})
            next_id += 1

    # Unconditional texts
    for iid in step2.unconditional_intents or []:
        intent = intent_by_id.get(iid)
        if intent and (intent.text or "").strip():
            append_text_nodes(intent.text.strip())

    # Global templates as separate atomic text nodes.
    for t in templates_global:
        if (t.text or "").strip():
            append_text_nodes(t.text.strip())

    # Conditions
    for cond in step2.conditions or []:
        parent_id = str(next_id)
        next_id += 1

        if not (cond.condition_text or "").strip():
            raise ValueError(f"condition {cond.id} has empty condition_text")

        then_texts: List[str] = []
        for iid in cond.then_intents or []:
            intent = intent_by_id.get(iid)
            if intent and (intent.text or "").strip():
                then_texts.append(intent.text.strip())
        for t in templates_then:
            if t.condition_id == cond.id and (t.text or "").strip():
                then_texts.append(t.text.strip())

        else_texts: List[str] = []
        for iid in cond.else_intents or []:
            intent = intent_by_id.get(iid)
            if intent and (intent.text or "").strip():
                else_texts.append(intent.text.strip())
        for t in templates_else:
            if t.condition_id == cond.id and (t.text or "").strip():
                else_texts.append(t.text.strip())

        children = _atomic_text_nodes(parent_id, 1, then_texts)
        if not children and strict:
            raise ValueError(f"condition {cond.id} has no then-actions (then_intents/templates empty)")

        node: Dict[str, Any] = {
            "id": parent_id,
            "type": "if",
            "condition": (cond.condition_text or "").strip(),
            "children": children or [],
        }

        if else_texts:
            node["else_children"] = _atomic_text_nodes(parent_id, 2, else_texts)
        elif explicit_else_noop:
            node["else_children"] = []

        code.append(node)

    code.append({"id": str(next_id), "type": "end"})

    scenario = ScenarioDefinition.model_validate(
        {
            "name": _scenario_name(name_hint, text=input_text, trace_id=trace_id),
            "code": code,
        }
    )
    return scenario


def _text_has_explicit_noop_else(text: str) -> bool:
    t = (text or "").lower()
    if ("иначе" in t or "а если" in t or "если нет" in t or "если не" in t) and "ничего" in t:
        return True
    if "ничего не" in t and ("говор" in t or "дел" in t or "добав" in t):
        return True
    return False


async def sgr_convert_via_langchain(
    *,
    text: str,
    available_tools: List[ToolSpec],
    name_hint: Optional[str],
    strict: bool,
    return_diagnostics: bool,
) -> Tuple[ScenarioDefinition, Dict[str, Any], List[str]]:
    trace_id = uuid.uuid4().hex[:10]
    trace_dir = _sgr_trace_dir(trace_id)
    timeout_s = _sgr_timeout_s()

    llm = _build_llm()

    tool_specs = [
        {
            "name": t.name,
            "description": t.description,
            "output_fields": sorted(list((t.output_schema or {}).get("properties", {}).keys())),
        }
        for t in (available_tools or [])
    ]
    tool_names = [t["name"] for t in tool_specs if t.get("name")]

    trace_input = {
        "trace_id": trace_id,
        "text": text,
        "name_hint": name_hint,
        "strict": bool(strict),
        "return_diagnostics": bool(return_diagnostics),
        "model": settings.sgr_model,
        "base_url": settings.llm_base_url,
        "timeout_s": timeout_s,
        "available_tools": tool_specs,
    }
    (trace_dir / "00_convert_request.json").write_text(json.dumps(trace_input, ensure_ascii=False, indent=2), encoding="utf-8")

    diagnostics: Dict[str, Any] = {"trace_id": trace_id}
    last_llm_raw = ""
    step_traces: List[Dict[str, Any]] = []
    bundle_steps: List[Dict[str, Any]] = []

    def _push_trace(st: _StepTrace) -> None:
        step_traces.append(
            {
                "step": st.step,
                "duration_s": st.duration_s,
                "model": st.model,
                "request_path": st.request_path,
                "response_path": st.response_path,
                "headers": {k: v for k, v in (st.headers or {}).items() if str(k).lower() in ("x-request-id", "x-openrouter-request-id")},
            }
        )

    # Step 1: normalize + extract intents
    try:
        s1_system = (
            "Ты — конвертер SGR (plain text -> атомарные намерения).\n"
            "Задача: 1) нормализовать вход (без потери смысла), 2) выделить атомарные намерения изменить поведение агента.\n"
            "Требования:\n"
            "- 1 намерение = 1 intent.text (не склеивай через \\n).\n"
            "- Пиши как инструкции агенту в повелительном наклонении (например: \"Скажи привет\").\n"
            "- Не придумывай факты/инструменты.\n"
            "- Не добавляй намерения вида \"Определить/Проверить ...\" если это просто проверка условия по словам пользователя.\n"
            "- Не добавляй эмодзи.\n"
            "- questions добавляй только если без уточнения НЕЛЬЗЯ построить сценарий; не задавай вопросы типа \"как определить\".\n"
            "Верни СТРОГО JSON-объект формата:\n"
            '{\n'
            '  "normalized_text": "...",\n'
            '  "intents": [{"id":"i1","text":"..."}],\n'
            '  "questions": []\n'
            "}\n"
        )
        s1_user = f"strict={bool(strict)}\ntext:\n{text}\n"
        step1, last_llm_raw, tr, step_req, step_resp = await _call_llm_step(
            llm=llm,
            trace_id=trace_id,
            step="01_extract_intents",
            system=s1_system,
            user=s1_user,
            out_model=Step1ExtractIntents,
            timeout_s=timeout_s,
            trace_dir=trace_dir,
        )
        _push_trace(tr)
        bundle_steps.append({"step": tr.step, "duration_s": tr.duration_s, "request": step_req, "response": step_resp})
        step1.normalized_text = _clean_text(step1.normalized_text)
        for it in step1.intents or []:
            it.text = _clean_text(it.text)
        step1.questions = _filter_questions(step1.questions or [])
    except Exception as exc:  # noqa: BLE001
        diagnostics["llm_steps"] = step_traces
        (trace_dir / "98_error.json").write_text(
            json.dumps({"trace_id": trace_id, "failed_step": "01_extract_intents", "error": repr(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SgrConvertError(
            trace_id=trace_id,
            failed_step="01_extract_intents",
            diagnostics={**diagnostics, "error": repr(exc)},
            last_llm_raw=last_llm_raw,
        ) from exc

    # Step 2: critique + gating
    try:
        s2_system = (
            "Ты — модуль self-critique + gating для SGR.\n"
            "Задача:\n"
            "1) Проверь intents на полноту и непересечения относительно original_text.\n"
            "2) Если нужно — исправь/переформулируй intents (но не добавляй факты).\n"
            "3) Найди условия применения (если/иначе) и разложи на conditions.\n"
            "Правила:\n"
            "- condition_text пиши как понятную фразу для движка (предпочитай: 'Пользователь написал в чат что ...').\n"
            "- Не добавляй отдельные intents вида \"Определить/Проверить ...\" если это просто проверка condition по словам пользователя.\n"
            "- questions добавляй только если без уточнения НЕЛЬЗЯ построить сценарий; не задавай вопросы типа \"как определить\".\n"
            "Верни СТРОГО JSON-объект формата:\n"
            '{\n'
            '  "intents": [{"id":"i1","text":"..."}],\n'
            '  "unconditional_intents": ["i1"],\n'
            '  "conditions": [{"id":"c1","condition_text":"...","then_intents":["i1"],"else_intents":["i2"]}],\n'
            '  "questions": []\n'
            "}\n"
        )
        s2_user = (
            f"strict={bool(strict)}\n"
            f"original_text:\n{text}\n\n"
            f"normalized_text:\n{step1.normalized_text}\n\n"
            f"intents:\n{json.dumps([i.model_dump() for i in step1.intents], ensure_ascii=False, indent=2)}\n"
        )
        step2, last_llm_raw, tr, step_req, step_resp = await _call_llm_step(
            llm=llm,
            trace_id=trace_id,
            step="02_gate_and_critique",
            system=s2_system,
            user=s2_user,
            out_model=Step2GateAndCritique,
            timeout_s=timeout_s,
            trace_dir=trace_dir,
        )
        _push_trace(tr)
        bundle_steps.append({"step": tr.step, "duration_s": tr.duration_s, "request": step_req, "response": step_resp})
        for it in step2.intents or []:
            it.text = _clean_text(it.text)
        for c in step2.conditions or []:
            c.condition_text = _clean_text(c.condition_text)
        step2.questions = _filter_questions(step2.questions or [])

        # Drop non-actionable "check" intents from unconditional application when conditions exist.
        if step2.conditions:
            intent_by_id = {i.id: i for i in (step2.intents or [])}
            step2.unconditional_intents = [
                iid
                for iid in (step2.unconditional_intents or [])
                if iid in intent_by_id and not _looks_like_condition_check_intent(intent_by_id[iid].text)
            ]
    except Exception as exc:  # noqa: BLE001
        diagnostics["llm_steps"] = step_traces
        (trace_dir / "98_error.json").write_text(
            json.dumps({"trace_id": trace_id, "failed_step": "02_gate_and_critique", "error": repr(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SgrConvertError(
            trace_id=trace_id,
            failed_step="02_gate_and_critique",
            diagnostics={**diagnostics, "error": repr(exc)},
            last_llm_raw=last_llm_raw,
        ) from exc

    # Step 3: knowledge gaps + tools + templates
    try:
        s3_system = (
            "Ты — модуль knowledge-gap analysis + tool matching + templates для SGR.\n"
            "Задача:\n"
            "1) Сматчи на доступные tools (используй ТОЛЬКО имена из available_tools).\n"
            "2) Если нужного tool нет — добавь в missing_tools (НЕ выдумывай вызов в сценарии).\n"
            "3) Если текстовые инструкции должны подставлять результаты tool через шаблон {=@tool.field=} — добавь templates.\n"
            "Правила:\n"
            "- tools_to_call: только из available_tools.\n"
            "- templates: каждый шаблон — отдельная инструкция агенту (без эмодзи), не оформляй как markdown.\n"
            "- target=condition_then/condition_else требует condition_id.\n"
            "- Не задавай вопросы типа \"как определить\" для условий, которые проверяются по словам пользователя.\n"
            "Верни СТРОГО JSON-объект формата:\n"
            '{\n'
            '  "tools_to_call": ["get_user_data"],\n'
            '  "missing_tools": [{"name":"award_bonus_points","reason":"...","input_schema":{},"output_schema":{}}],\n'
            '  "templates": [{"id":"t1","target":"global|condition_then|condition_else","condition_id":"c1","text":"...","depends_on_tool":"get_user_data"}],\n'
            '  "questions": []\n'
            "}\n"
        )
        s3_user = (
            f"strict={bool(strict)}\n"
            f"available_tools:\n{json.dumps(tool_specs, ensure_ascii=False, indent=2)}\n\n"
            f"intents:\n{json.dumps([i.model_dump() for i in step2.intents], ensure_ascii=False, indent=2)}\n\n"
            f"conditions:\n{json.dumps([c.model_dump() for c in step2.conditions], ensure_ascii=False, indent=2)}\n"
        )
        step3, last_llm_raw, tr, step_req, step_resp = await _call_llm_step(
            llm=llm,
            trace_id=trace_id,
            step="03_tools_and_templates",
            system=s3_system,
            user=s3_user,
            out_model=Step3ToolsAndTemplates,
            timeout_s=timeout_s,
            trace_dir=trace_dir,
        )
        _push_trace(tr)
        bundle_steps.append({"step": tr.step, "duration_s": tr.duration_s, "request": step_req, "response": step_resp})
        step3.questions = _filter_questions(step3.questions or [])
        for t in step3.templates or []:
            t.text = _clean_text(t.text)
        for m in step3.missing_tools or []:
            m.reason = _clean_text(m.reason)
    except Exception as exc:  # noqa: BLE001
        diagnostics["llm_steps"] = step_traces
        (trace_dir / "98_error.json").write_text(
            json.dumps({"trace_id": trace_id, "failed_step": "03_tools_and_templates", "error": repr(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SgrConvertError(
            trace_id=trace_id,
            failed_step="03_tools_and_templates",
            diagnostics={**diagnostics, "error": repr(exc)},
            last_llm_raw=last_llm_raw,
        ) from exc

    # Hard policy: do not let LLM invent tool names.
    tools_to_call_before = list(step3.tools_to_call or [])
    step3.tools_to_call = [t for t in (step3.tools_to_call or []) if t in tool_names]
    for tmpl in step3.templates or []:
        if tmpl.depends_on_tool and tmpl.depends_on_tool not in tool_names:
            tmpl.depends_on_tool = None
    step3.missing_tools = [m for m in (step3.missing_tools or []) if (m.name or "").strip()]
    step3.templates = [t for t in (step3.templates or []) if (t.text or "").strip()]

    # Ensure we call tools required by any {=@tool.field=} references (in intents or templates),
    # even if the LLM forgot to list them in tools_to_call.
    needed_tools: List[str] = []
    seen_tools: set[str] = set()

    def add_tool(name: Optional[str]) -> None:
        n = (name or "").strip()
        if not n or n in seen_tools:
            return
        seen_tools.add(n)
        needed_tools.append(n)

    for tname in step3.tools_to_call or []:
        add_tool(tname)
    for tmpl in step3.templates or []:
        add_tool(tmpl.depends_on_tool)
        for tname in _extract_tool_refs(tmpl.text or ""):
            add_tool(tname)
    for intent in step2.intents or []:
        for tname in _extract_tool_refs(intent.text or ""):
            add_tool(tname)

    step3.tools_to_call = [t for t in needed_tools if t in tool_names]

    # Step 4: assemble scenario deterministically (no LLM)
    try:
        explicit_else_noop = _text_has_explicit_noop_else(text)
        scenario = _assemble_scenario(
            trace_id=trace_id,
            input_text=text,
            name_hint=name_hint,
            explicit_else_noop=explicit_else_noop,
            strict=bool(strict),
            step2=step2,
            step3=step3,
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["llm_steps"] = step_traces
        (trace_dir / "98_error.json").write_text(
            json.dumps({"trace_id": trace_id, "failed_step": "04_assemble_scenario", "error": repr(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SgrConvertError(
            trace_id=trace_id,
            failed_step="04_assemble_scenario",
            diagnostics={**diagnostics, "error": repr(exc)},
            last_llm_raw=last_llm_raw,
        ) from exc

    questions = _filter_questions([*(step1.questions or []), *(step2.questions or []), *(step3.questions or [])])

    if return_diagnostics:
        diagnostics.update(
            {
                "llm_steps": step_traces,
                "strict": bool(strict),
                "available_tools": [{"name": t.get("name"), "description": t.get("description", "")} for t in tool_specs],
                "intermediate": {
                    "step1": step1.model_dump(),
                    "step2": step2.model_dump(),
                    "step3": step3.model_dump(),
                },
                "missing_tools": [m.model_dump() for m in (step3.missing_tools or [])],
                "transforms": {
                    "filter_tools_to_call": {
                        "allowed_tool_names": tool_names,
                        "before": tools_to_call_before,
                        "after": list(step3.tools_to_call or []),
                    }
                },
            }
        )
    else:
        diagnostics = {"trace_id": trace_id}

    final_payload = {
        "scenario": scenario.model_dump(exclude_none=True, exclude_defaults=True),
        "diagnostics": diagnostics,
        "questions": questions,
    }
    (trace_dir / "99_convert_result.json").write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    trace_bundle = {
        "trace_id": trace_id,
        "input": trace_input,
        "steps": bundle_steps,
        "final": final_payload,
    }
    (trace_dir / "trace_bundle.json").write_text(json.dumps(trace_bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    return scenario, diagnostics, questions
