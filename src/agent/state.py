#state.py
#AgentState is shared data packet that travels through every LangGraph Node
"""
State schema for the AI Customer Support Agent graph.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, TypedDict, Any, Annotated
from pydantic import BaseModel, Field

MAX_MESSAGE_WINDOW = 50  # Keep up to 50 messages (25 full conversational turns)


def append_messages(existing: Optional[list[dict]], new: Optional[list[dict]]) -> list[dict]:
    """
    Appends new messages to conversation history idempotently.
    Safely handles single-message deltas, cumulative history passes,
    and deduplicates identical prefixes.
    """
    existing_list = list(existing) if existing else []
    new_list = list(new) if new else []

    if not existing_list:
        return new_list[-MAX_MESSAGE_WINDOW:]
    if not new_list:
        return existing_list[-MAX_MESSAGE_WINDOW:]

    # Case 1: `new` is already cumulative and starts with `existing`
    if len(new_list) >= len(existing_list) and new_list[:len(existing_list)] == existing_list:
        combined = new_list
    # Case 2: `existing` already ends with `new` (duplicate update)
    elif len(existing_list) >= len(new_list) and existing_list[-len(new_list):] == new_list:
        combined = existing_list
    # Case 3: Partial overlap (end of existing matches start of new)
    else:
        overlap = 0
        max_overlap = min(len(existing_list), len(new_list))
        for i in range(max_overlap, 0, -1):
            if existing_list[-i:] == new_list[:i]:
                overlap = i
                break
        if overlap > 0:
            combined = existing_list + new_list[overlap:]
        else:
            # Case 4: `new` is a fresh delta of new messages
            combined = existing_list + new_list

    return combined[-MAX_MESSAGE_WINDOW:]


def preserve_if_none(existing: Any, new: Any) -> Any:
    """Retains previous non-None entity value when incoming update is None."""
    return new if new is not None else existing


def append_trace(existing: Optional[list[StepLog]], new: Optional[list[StepLog]]) -> list[StepLog]:
    """
    Appends StepLog trace entries idempotently.
    Deduplicates any StepLogs that already exist in `existing` so traces never repeat.
    """
    existing_list = list(existing) if existing else []
    new_list = list(new) if new else []

    if not existing_list:
        return new_list
    if not new_list:
        return existing_list

    # Case 1: `new` is already cumulative and starts with `existing`
    if len(new_list) >= len(existing_list) and new_list[:len(existing_list)] == existing_list:
        return new_list

    # Case 2: Deduplicate any StepLogs in `new` that already exist in `existing`
    seen = {
        (log.node_name, getattr(log, "started_at", None), getattr(log, "output_summary", None))
        for log in existing_list
    }
    combined = list(existing_list)
    for log in new_list:
        key = (log.node_name, getattr(log, "started_at", None), getattr(log, "output_summary", None))
        if key not in seen:
            combined.append(log)
            seen.add(key)

    return combined


class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    RETURNED = "returned"
    REFUNDED = "refunded"


class OrderRecord(BaseModel):
    order_id: str
    customer_id: str
    status: OrderStatus
    total_amount: float
    placed_at: datetime
    delivered_at: Optional[datetime] = None
    items: list[str] = Field(default_factory=list)


class PolicyChunk(BaseModel):
    policy_id: str
    text: str
    source_doc: str
    similarity_score: float


class ActionType(str, Enum):
    ANSWER = "answer"
    TOOL_CALL = "tool_call"
    ESCALATE = "escalate"


class ReasonerDecision(BaseModel):
    action_type: ActionType
    rationale: str
    tool_name: Optional[str] = None
    tool_args: dict = Field(default_factory=dict)
    draft_response: Optional[str] = None


class ActionResult(BaseModel):
    tool_name: str
    success: bool
    output: str
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None


class EscalationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EscalationRecord(BaseModel):
    reason: str
    priority: EscalationPriority
    source_node: Literal["reasoner", "action"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StepLog(BaseModel):
    node_name: str
    started_at: datetime
    latency_ms: float
    input_summary: str
    output_summary: str
    error: Optional[str] = None


class ApprovalDecision(BaseModel):
    approved: bool
    reason: Optional[str] = None
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    PENDING_APPROVAL = "pending_approval"


class AgentState(TypedDict, total=False):
    # Persistence identifiers
    conversation_id: Annotated[str, preserve_if_none]
    customer_id: Annotated[str, preserve_if_none]

    # Pinned entity state
    order_context: Annotated[Optional[Any], preserve_if_none]
    customer_profile: Annotated[Optional[dict], preserve_if_none]
    ticket_count: Annotated[Optional[int], preserve_if_none]
    image_source: Annotated[Optional[str], preserve_if_none]

    # Conversation history
    messages: Annotated[list[dict], append_messages]

    # Per-turn working state
    intent: Optional[str]
    policy_context: list[PolicyChunk]
    decision: Optional[ReasonerDecision]
    action_result: Optional[ActionResult]
    escalation: Optional[EscalationRecord]
    approval_decision: Optional[ApprovalDecision]
    final_response: Optional[str]
    status: ConversationStatus

    # Observability
    trace: Annotated[list[StepLog], append_trace]
