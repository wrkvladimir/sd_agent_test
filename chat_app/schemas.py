from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatRequest(BaseModel):
    """Incoming chat request from the client."""

    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response returned to the client."""

    conversation_id: str
    answer: str
    chunks: List["Chunk"]
    last_step_scenario: Optional[str] = None


class HistoryItem(BaseModel):
    """Single message in the conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HistoryResponse(BaseModel):
    """History payload for a given conversation."""

    conversation_id: str
    history: List[HistoryItem]


class SummaryResponse(BaseModel):
    """Summary payload for a given conversation."""

    conversation_id: str
    summary: str


class Chunk(BaseModel):
    """
    KB chunk returned to the client as part of /chat response.

    Структура совместима по полям с Chunk из ingest_and_retrieval,
    но определяется локально, чтобы не зависеть от кода другого сервиса.
    """

    id: Any
    text: str = ""
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None


class UserProfile(BaseModel):
    """Profile of the end user the agent is talking to."""

    name: Optional[str] = None
    age: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    """High-level conversation state stored in memory (e.g. Redis)."""

    conversation_id: str
    message_index: int = 0
    user_profile: UserProfile = Field(default_factory=UserProfile)
    summary: str = ""
    scenario_runs: List[Dict[str, Any]] = Field(default_factory=list)


class ScenarioNode(BaseModel):
    """
    Single node of a JSON-defined scenario.

    Supports:
    - text: instructions / additional context for LLM
    - tool: tool invocation (result stored in memory)
    - if: conditional branching with children / else_children
    - end: explicit scenario termination
    """

    id: str
    type: Literal["text", "tool", "if", "end"]

    text: Optional[str] = None
    tool: Optional[str] = None
    condition: Optional[str] = None

    children: Optional[List["ScenarioNode"]] = None
    else_children: Optional[List["ScenarioNode"]] = None


ScenarioNode.model_rebuild()


class ScenarioDefinition(BaseModel):
    """Top-level scenario definition loaded from JSON or API."""

    name: str
    code: List[ScenarioNode]
    # Optional scenario metadata/policy. Used by v1.0 scenario engine.
    meta: Dict[str, Any] = Field(default_factory=dict)
    # Scenario can be toggled without deletion.
    enabled: bool = True
    # Optional UI metadata (tooltip/preview).
    summary: Optional[str] = None
    admin_message: Optional[str] = None


class ScenarioPatchRequest(BaseModel):
    enabled: Optional[bool] = None


class ScenarioUpsertResponse(BaseModel):
    """Response returned when a scenario is added or updated."""

    name: str
    status: str = "ok"


class ToolSpec(BaseModel):
    """Tool metadata for UI and future orchestration."""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
