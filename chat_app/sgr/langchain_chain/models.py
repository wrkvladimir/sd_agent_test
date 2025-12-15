from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Intent(BaseModel):
    id: str = Field(..., description="Детерминированный id: i1, i2, ...")
    text: str = Field(..., description="Атомарная инструкция (1 намерение = 1 пункт).")


class Step1ExtractIntents(BaseModel):
    normalized_text: str
    intents: List[Intent] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)


class ConditionGate(BaseModel):
    id: str = Field(..., description="Детерминированный id: c1, c2, ...")
    condition_text: str = Field(..., description="Текст условия, как его должен понимать движок.")
    then_intents: List[str] = Field(default_factory=list, description="Intent ids для ветки then.")
    else_intents: List[str] = Field(default_factory=list, description="Intent ids для ветки else.")


class Step2GateAndCritique(BaseModel):
    intents: List[Intent] = Field(default_factory=list)
    unconditional_intents: List[str] = Field(default_factory=list, description="Intent ids без условий.")
    conditions: List[ConditionGate] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)


class MissingTool(BaseModel):
    name: str
    reason: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


class TemplatePlan(BaseModel):
    id: str = Field(..., description="Детерминированный id: t1, t2, ...")
    target: Literal["global", "condition_then", "condition_else"]
    condition_id: Optional[str] = None
    text: str = Field(..., description="Шаблонный текст с {=@tool.field=} или {=dialog.*=}.")
    depends_on_tool: Optional[str] = Field(default=None, description="Имя tool, результат которого нужен.")


class Step3ToolsAndTemplates(BaseModel):
    tools_to_call: List[str] = Field(default_factory=list)
    missing_tools: List[MissingTool] = Field(default_factory=list)
    templates: List[TemplatePlan] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
