from __future__ import annotations
import os
import psycopg2
import psycopg2.extras
from typing import Optional, Any
from langgraph.types import interrupt, Command

from src.agent.observability.logger import with_logging
from src.agent.state import (
    ApprovalDecision,
    ConversationStatus,
    EscalationRecord,
    EscalationPriority,
)


DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/support_agent"

def record_escalation_in_db(
    reason: str,
    priority: str = "medium",
    source_node: str = "action",
    customer_id: Optional[str] = None,
    order_id: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Optional[str]:
    """Records an escalation into the PostgreSQL database if available."""
    db_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL)
    if not customer_id:
        return None

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # 1. Ensure support ticket exists
        cur.execute(
            """
            INSERT INTO support_tickets (customer_id, order_id, subject, status)
            VALUES (%s, %s, %s, 'escalated')
            RETURNING ticket_id
            """,
            (customer_id, order_id, f"Escalation: {reason[:100]}"),
        )
        ticket_id = cur.fetchone()["ticket_id"]

        # 2. Insert escalation record
        cur.execute(
            """
            INSERT INTO escalations (ticket_id, reason, priority, source_node)
            VALUES (%s, %s, %s::escalation_priority, %s::escalation_source)
            RETURNING escalation_id
            """,
            (ticket_id, reason, priority.lower(), source_node.lower()),
        )
        escalation_id = str(cur.fetchone()["escalation_id"])
        conn.commit()
        conn.close()
        return escalation_id
    except Exception:
        return None


def update_refund_status_in_db(order_id: str, new_status: str, database_url: Optional[str] = None) -> bool:
    """Updates refund record status in the database upon human approval or rejection."""
    db_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL)
    if not order_id:
        return False

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE refunds
            SET status = %s::refund_status,
                processed_at = CASE WHEN %s IN ('approved', 'processed') THEN now() ELSE processed_at END
            WHERE order_id = %s
            """,
            (new_status, new_status, order_id),
        )
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()
        return updated
    except Exception:
        return False


@with_logging("human_approval")
def human_approval_node(state: dict) -> dict:
    """
    Human-in-the-loop node.
    Pauses execution via LangGraph interrupt() when high-risk actions or
    escalations occur, persisting state to PostgreSQL.
    Resumes when human supervisor submits review decision.
    """
    escalation = state.get("escalation")
    action_result = state.get("action_result")
    order_context = state.get("order_context")
    decision = state.get("decision")

    order_id = None
    if order_context:
        order_id = getattr(order_context, "order_id", None) or (
            order_context.get("order_id") if isinstance(order_context, dict) else None
        )
    if not order_id and decision:
        tool_args = getattr(decision, "tool_args", {}) or (
            decision.get("tool_args", {}) if isinstance(decision, dict) else {}
        )
        order_id = tool_args.get("order_id")

    reason = ""
    if escalation:
        reason = getattr(escalation, "reason", "") or (
            escalation.get("reason", "") if isinstance(escalation, dict) else ""
        )
    elif action_result and getattr(action_result, "escalation_reason", None):
        reason = action_result.escalation_reason

    # Record in PostgreSQL escalations table
    customer_id = state.get("customer_id")
    record_escalation_in_db(
        reason=reason or "Human approval required",
        customer_id=customer_id,
        order_id=order_id,
        source_node="human_approval",
    )

    state["status"] = ConversationStatus.PENDING_APPROVAL

    # Construct approval request payload
    interrupt_payload = {
        "type": "human_approval_required",
        "conversation_id": state.get("conversation_id"),
        "customer_id": state.get("customer_id"),
        "order_id": order_id,
        "reason": reason,
        "action_name": getattr(action_result, "tool_name", None) if action_result else "review",
        "action_output": getattr(action_result, "output", "") if action_result else "",
    }

    # Pause execution until supervisor resumes
    review_response: dict[str, Any] = interrupt(interrupt_payload)

    # Execution resumes here once Command(resume=...) is provided
    approved = review_response.get("approved", False)
    reviewer = review_response.get("reviewer", "supervisor")
    notes = review_response.get("notes", "")

    state["approval_decision"] = ApprovalDecision(
        approved=approved,
        reason=notes,
        reviewer=reviewer,
    )

    # Reflect approval decision in database refunds table
    if order_id:
        new_status = "approved" if approved else "rejected"
        update_refund_status_in_db(order_id, new_status)

    state["status"] = ConversationStatus.RESOLVED
    return state


def resume_approval(
    app: Any,
    thread_id: str,
    approved: bool,
    reviewer: str = "supervisor",
    notes: str = "",
) -> dict:
    """
    Helper function to resume a paused conversation thread with supervisor decision.
    """
    resume_payload = {
        "approved": approved,
        "reviewer": reviewer,
        "notes": notes,
    }
    config = {"configurable": {"thread_id": str(thread_id)}}
    return app.invoke(Command(resume=resume_payload), config=config)
