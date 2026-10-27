from __future__ import annotations
import os
import uuid
import pytest
import psycopg2
import psycopg2.extras
from langgraph.checkpoint.memory import MemorySaver

from src.agent.llm_client import LLMClient
from src.agent.state import ActionType, ReasonerDecision, ConversationStatus
from src.agent.graph import build_graph
from src.agent.persistence import get_checkpointer, get_thread_config
from src.agent.human_approval import resume_approval


class ToolMockLLM(LLMClient):
    """Mock LLM that triggers initiate_refund tool calls for testing."""

    def __init__(self, refund_amount: float, order_id: str):
        self.refund_amount = refund_amount
        self.order_id = order_id

    def classify_intent(self, message: str, labels: list[str]) -> str:
        return "refund_request"

    def decide(self, message: str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision:
        return ReasonerDecision(
            action_type=ActionType.TOOL_CALL,
            rationale="Customer requested a refund",
            tool_name="initiate_refund",
            tool_args={"order_id": self.order_id, "amount": self.refund_amount, "reason": "Customer return"},
        )


@pytest.fixture
def db_conn():
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_agent")
    conn = psycopg2.connect(db_url)
    yield conn
    conn.close()


def create_test_order(conn, total_amount: float) -> tuple[str, str]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "INSERT INTO customers (full_name, email) VALUES (%s, %s) RETURNING customer_id",
        (f"Approval Test {uuid.uuid4()}", f"approval-{uuid.uuid4()}@example.com"),
    )
    customer_id = str(cur.fetchone()["customer_id"])
    cur.execute(
        "INSERT INTO orders (customer_id, status, total_amount) VALUES (%s, 'delivered', %s) RETURNING order_id",
        (customer_id, total_amount),
    )
    order_id = str(cur.fetchone()["order_id"])
    conn.commit()
    return customer_id, order_id


def test_auto_approved_refund_flow_bypasses_human_approval(db_conn):
    """Refunds under $75 and 80% should auto-approve without pausing."""
    customer_id, order_id = create_test_order(db_conn, total_amount=500.0)
    llm = ToolMockLLM(refund_amount=25.0, order_id=order_id)
    memory = MemorySaver()
    app = build_graph(llm, checkpointer=memory)

    thread_id = f"test-auto-{uuid.uuid4()}"
    config = get_thread_config(thread_id)

    initial_state = {
        "conversation_id": thread_id,
        "customer_id": customer_id,
        "messages": [{"role": "user", "content": "I want a refund for $25"}],
    }

    result = app.invoke(initial_state, config=config)

    assert result["status"] == ConversationStatus.RESOLVED
    assert "__interrupt__" not in result
    assert "approved and processed" in result["final_response"]

    # Verify executed nodes
    trace_nodes = [log.node_name for log in result["trace"]]
    assert "action" in trace_nodes
    assert "human_approval" not in trace_nodes


def test_escalated_refund_flow_pauses_and_resumes_on_approval(db_conn):
    """Refunds over $75 pause at human_approval and resume cleanly upon supervisor approval."""
    customer_id, order_id = create_test_order(db_conn, total_amount=500.0)
    llm = ToolMockLLM(refund_amount=150.0, order_id=order_id)
    memory = MemorySaver()
    app = build_graph(llm, checkpointer=memory)

    thread_id = f"test-escalate-{uuid.uuid4()}"
    config = get_thread_config(thread_id)

    initial_state = {
        "conversation_id": thread_id,
        "customer_id": customer_id,
        "messages": [{"role": "user", "content": "I want a refund for $150"}],
    }

    # First invoke - graph pauses at human_approval
    res1 = app.invoke(initial_state, config=config)

    assert "__interrupt__" in res1
    assert len(res1["__interrupt__"]) > 0
    interrupt_info = res1["__interrupt__"][0].value
    assert interrupt_info["type"] == "human_approval_required"
    assert interrupt_info["order_id"] == order_id

    # Verify refund status in DB is 'requested'
    cur = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT status FROM refunds WHERE order_id = %s", (order_id,))
    refund_row = cur.fetchone()
    assert refund_row["status"] == "requested"

    # Supervisor approves the refund
    res2 = resume_approval(app, thread_id=thread_id, approved=True, reviewer="supervisor_alice", notes="VIP customer exception")

    assert res2["status"] == ConversationStatus.RESOLVED
    assert res2["approval_decision"].approved is True
    assert "approved and processed" in res2["final_response"]

    # Verify refund in DB is updated to 'approved'
    cur.execute("SELECT status FROM refunds WHERE order_id = %s", (order_id,))
    updated_row = cur.fetchone()
    assert updated_row["status"] == "approved"


def test_escalated_refund_flow_rejects_and_informs_customer(db_conn):
    """When supervisor rejects an escalated refund, status reflects rejection."""
    customer_id, order_id = create_test_order(db_conn, total_amount=100.0)
    # $90 on a $100 order = 90% ratio, over 80% limit
    llm = ToolMockLLM(refund_amount=90.0, order_id=order_id)
    memory = MemorySaver()
    app = build_graph(llm, checkpointer=memory)

    thread_id = f"test-reject-{uuid.uuid4()}"
    config = get_thread_config(thread_id)

    initial_state = {
        "conversation_id": thread_id,
        "customer_id": customer_id,
        "messages": [{"role": "user", "content": "I want a refund for $90"}],
    }

    # Pause at interrupt
    app.invoke(initial_state, config=config)

    # Supervisor rejects
    res = resume_approval(
        app,
        thread_id=thread_id,
        approved=False,
        reviewer="supervisor_bob",
        notes="Order exceeded allowable return window",
    )

    assert res["status"] == ConversationStatus.RESOLVED
    assert res["approval_decision"].approved is False
    assert "could not be approved" in res["final_response"]
    assert "Order exceeded allowable return window" in res["final_response"]

    # Verify refund status in DB is 'rejected'
    cur = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT status FROM refunds WHERE order_id = %s", (order_id,))
    assert cur.fetchone()["status"] == "rejected"


def test_postgres_persistence_end_to_end(db_conn):
    """Validates that checkpoints persist in PostgreSQL and survive distinct app instances."""
    customer_id, order_id = create_test_order(db_conn, total_amount=400.0)
    llm = ToolMockLLM(refund_amount=120.0, order_id=order_id)

    # Use actual PostgreSQL checkpointer
    pg_checkpointer = get_checkpointer(use_postgres=True)
    app1 = build_graph(llm, checkpointer=pg_checkpointer)

    thread_id = f"test-pg-persist-{uuid.uuid4()}"
    config = get_thread_config(thread_id)

    initial_state = {
        "conversation_id": thread_id,
        "customer_id": customer_id,
        "messages": [{"role": "user", "content": "Need refund of $120"}],
    }

    # Invoke on app1, triggering pause and persisting to PostgreSQL
    res1 = app1.invoke(initial_state, config=config)
    assert "__interrupt__" in res1

    # Instantiate a completely NEW graph instance pointing to the same PostgreSQL checkpointer
    app2 = build_graph(llm, checkpointer=pg_checkpointer)

    # Resume from app2
    res2 = resume_approval(app2, thread_id=thread_id, approved=True, reviewer="manager_dan")

    assert res2["status"] == ConversationStatus.RESOLVED
    assert res2["approval_decision"].approved is True
    assert "approved and processed" in res2["final_response"]
