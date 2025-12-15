from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict

from chat_app.schemas import Chunk, ConversationState, HistoryItem


class JudgeDecision(TypedDict, total=False):
    action: Literal["pass", "revise"]
    reasons: List[str]
    patch_instructions: str


class InstructionBlock(TypedDict, total=False):
    id: str
    source: str
    target: Literal["agent", "judge"]
    # raw: внутренние «сырые» куски сценария, которые затем сжимаются в короткие imperative-инструкции.
    kind: Literal["required", "conditional", "rule", "raw"]
    priority: int
    text: str
    payload: Dict[str, Any]


class ToolsContext(TypedDict, total=False):
    facts: Dict[str, Dict[str, Any]]
    instruction_blocks: List[InstructionBlock]
    applied: List[Dict[str, str]]


class AgentState(TypedDict, total=False):
    conversation_id: str
    user_message: str

    conv_state: ConversationState
    history: List[HistoryItem]

    kb_chunks: List[Chunk]

    tools_context: ToolsContext

    prompt_messages: List[Dict[str, str]]

    answer_draft: str
    answer: str

    judge_attempts: int
    judge_decision: JudgeDecision


@dataclass(frozen=True)
class ScenarioMapResult:
    scenario_name: str
    facts: Dict[str, Dict[str, Any]]
    instruction_blocks: List[InstructionBlock]
