from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from chat_app.schemas import ScenarioDefinition


class SgrConvertRequest(BaseModel):
    """
    Minimal request: only `text` is required.
    Everything else is optional and can be extended later.
    """

    text: str = Field(..., description="Описание сценария простым языком.")
    name_hint: Optional[str] = Field(default=None, description="Желаемое имя сценария (уникальность не гарантируется).")
    strict: bool = Field(default=True, description="Если true — лучше вернуть вопросы/пропуски, чем додумывать.")
    return_diagnostics: bool = Field(default=False, description="Если false — diagnostics будет минимальным (trace_id).")


class SgrConvertResponse(BaseModel):
    scenario: ScenarioDefinition
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    questions: List[str] = Field(default_factory=list)
