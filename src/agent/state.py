#state.py
#AgentState is shared data packet that travels through every LangGraph Node
"""
State schema for the AI Customer Support Agent graph.
(Same as the earlier sketch — this is the version that actually lives in the project.)
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

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
    action_typr: ActionType
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

class EscalationPriority:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EscalationRecord(BaseModel):
    reason: str
    priority: EscalationPriority
    source_node: Literal["reasoner", "action"]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class StepLog(BaseModel):
    node_name: str
    started_at: datetime
    latency_ms: float
    input_summary: str
    output_summary: str
    error: Optional[str] = None

class ConversationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalted"

class AgentState(TypedDict, total=False):
    conversation_id: str
    customer_id : str
    messages: list[dict]
    intent: Optional[str]
    order_context: Optional[OrderRecord]
    policy_context: list[PolicyChunk]
    decision: Optional[ReasonerDecision]
    action_result: Optional[ActionResult]
    escalation: Optional[EscalationRecord]
    final_response: Optional[str]
    status: ConversationStatus
    trace: list[StepLog]

    

