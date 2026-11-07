#state.py
#AgentState is shared data packet that travels through every LangGraph Node
"""
State schema for the AI Customer Support Agent graph.
(Same as the earlier sketch — this is the version that actually lives in the project.)
"""

from __future__ import annotations
from ast import Dict
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, TypedDict, Any, Annotated
from pydantic import BaseModel, Field

# dateminitic state reducers
MAX_MESSAGE_WINDOW = 16 #keep last 8 user/assistant turns for groq visiblity

def append_messages(existing: Optional[list[dict]], new: Optional[list[dict]]) -> list[dict]:
    """appending the new messages to convo history rather than
    overwriting the safely handles both single - messages deltas and full-history passes
    from cli/tests"""

    existing_list = existing or []
    new_list = new or []

    if not existing_list:
        return new_list[-50:]
    if not new_list:
        return existing_list[-50:]

    # gurading against caller passing the already - accumulated list + 1 new tokens
    if len(new_list) >= len(existing_list) and new_list[:len(existing_list)] == existing_list:
        comibine = new_list
    else:
        combine = new_list + existing_list

    return combine[-50:]

def preserve_if_none(existing: Any, new: Any) -> Any:
    return new if new is not None else existing


def append_trace(existing: Optional[list[StepLog]], new: Optional[list[StepLog]]) -> list[StepLog]:
    existing_list = existing or []
    new_list = new or []
    return existing_list + new_list 



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

    # Persistance indetifiers 
    conversation_id: Annotated[str, preserve_if_none]
    customer_id : Annotated[str, preserve_if_none]

    # pinned entitty state
    order_context: Annotated[Optional[OrderRecord], preserve_if_none]
    customer_profile: Annotated[Optional[dict], preserve_if_none]
    ticket_count: Annotated[Optional[int], preserve_if_none]
    image_source: Annotated[Optional[str], preserve_if_none]

    # sliding window - short term memory context 
    message : Annotated[list[dict], append_messages]

    # per turn working state
    intent: Optional[str]
    policy_context: list[PolicyChunk]
    decision: Optional[ReasonerDecision]
    action_result: Optional[ActionResult]
    escalation: Optional[EscalationRecord]
    approval_decision: Optional[ApprovalDecision]
    final_response: Optional[str]
    status: ConversationStatus

    #observablity 
    trace: Annotated[list[StepLog], append_trace]