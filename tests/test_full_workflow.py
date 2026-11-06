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
from src.agent.tools.customer_tools import create_support_ticket


class ScriptedLLMClient(LLMClient):
    """
    Test LLM that responds according to a programmed sequence of decisions.
    """
    def __init__(self, intent: str, decision: ReasonerDecision):
        self._intent = intent
        self._decision = decision

    def classify_intent(self, message: str, labels: list[str]) -> str:
        return self._intent

    def decide(self, message: str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision:
        return self._decision


@pytest.fixture
def db_conn():
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_agent"
    )
    conn = psycopg2.connect(db_url)
    yield conn
    conn.close()


def setup_customer_and_order(conn, status="delivered", amount=100.0) -> tuple[str, str]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "INSERT INTO customers (full_name, email) VALUES (%s, %s) RETURNING customer_id",
        (f"Journey User {uuid.uuid4()}", f"journey-{uuid.uuid4()}@example.com"),
    )
    cust_id = str(cur.fetchone()["customer_id"])
    cur.execute(
        """
        INSERT INTO orders (customer_id, status, total_amount)
        VALUES (%s, %s::order_status, %s)
        RETURNING order_id
        """,
        (cust_id, status, amount),
    )
    order_id = str(cur.fetchone()["order_id"])
    conn.commit()
    return cust_id, order_id


def test_e2e_journey_order_status_success(db_conn):
    cust_id, order_id = setup_customer_and_order(db_conn, status="shipped")
    decision = ReasonerDecision(
        action_type=ActionType.ANSWER,
        rationale="Order found",
        draft_response=f"Your order {order_id} is currently shipped.",
    )
    llm = ScriptedLLMClient(intent="order_status", decision=decision)
    app = build_graph(llm, checkpointer=MemorySaver())

    thread_id = str(uuid.uuid4())
    config = get_thread_config(thread_id)

    state = {
        "conversation_id": thread_id,
        "customer_id": cust_id,
        "messages": [{"role": "user", "content": f"Where is my order {order_id}?"}],
    }

    result = app.invoke(state, config=config)

    assert result["intent"] == "order_status"
    assert result["status"] == ConversationStatus.RESOLVED
    assert f"Your order {order_id} is currently shipped." in result["final_response"]
    assert result["order_context"]["status"] == "shipped"


def test_e2e_journey_address_change_on_shipped_order_rejected_by_policy(db_conn):
    cust_id, order_id = setup_customer_and_order(db_conn, status="shipped")
    decision = ReasonerDecision(
        action_type=ActionType.TOOL_CALL,
        rationale="Customer requested address change",
        tool_name="change_shipping_address",
        tool_args={"order_id": order_id, "new_address": "123 Ocean Drive, Miami, FL"},
    )
    llm = ScriptedLLMClient(intent="shipping_question", decision=decision)
    app = build_graph(llm, checkpointer=MemorySaver())

    thread_id = str(uuid.uuid4())
    config = get_thread_config(thread_id)

    state = {
        "conversation_id": thread_id,
        "customer_id": cust_id,
        "messages": [
            {
                "role": "user",
                "content": f"Can I change the delivery address for order {order_id} to 123 Ocean Drive?",
            }
        ],
    }

    result = app.invoke(state, config=config)

    assert result["status"] == ConversationStatus.RESOLVED
    assert result["action_result"].success is False
    assert "refuse delivery" in result["final_response"]


def test_e2e_journey_repeat_contacts_detected_in_retrieval(db_conn):
    cust_id, order_id = setup_customer_and_order(db_conn, status="delivered")
    # Log 2 prior tickets
    create_support_ticket(cust_id, subject="Where is it?", order_id=order_id)
    create_support_ticket(cust_id, subject="Still not delivered", order_id=order_id)

    decision = ReasonerDecision(
        action_type=ActionType.ANSWER,
        rationale="Checking status",
        draft_response="Looking into your account.",
    )
    llm = ScriptedLLMClient(intent="order_status", decision=decision)
    app = build_graph(llm, checkpointer=MemorySaver())

    thread_id = str(uuid.uuid4())
    config = get_thread_config(thread_id)

    state = {
        "conversation_id": thread_id,
        "customer_id": cust_id,
        "messages": [{"role": "user", "content": f"Check order {order_id}"}],
    }

    result = app.invoke(state, config=config)

    # Retrieval should have enriched ticket_count = 2
    assert result["ticket_count"] == 2


def test_e2e_journey_damaged_item_multi_turn_with_photo_submission(db_conn, tmp_path, monkeypatch):
    from src.agent.tools import vision_tools
    monkeypatch.setattr(
        vision_tools,
        "inspect_damage_photo",
        lambda img, product_name=None, api_key=None: {
            "verified": True,
            "has_damage": True,
            "confidence": 0.98,
            "damage_description": "Severe crack across screen",
            "summary": "Physical damage confirmed",
        },
    )

    cust_id, order_id = setup_customer_and_order(db_conn, status="delivered")
    img_file = tmp_path / "broken_phone.jpg"
    img_file.write_bytes(b"\xff\xd8\xffdummyjpg")

    class MultiTurnDamagedLLM(LLMClient):
        def classify_intent(self, message: str, labels: list[str]) -> str:
            return "refund_request"

        def decide(self, message: str, order_context: dict | None, policy_context: list[str], image_source: str | None = None) -> ReasonerDecision:
            if image_source or "image attached" in message.lower():
                resolved_id = order_context.get("order_id") if order_context else order_id
                return ReasonerDecision(
                    action_type=ActionType.TOOL_CALL,
                    tool_name="report_damaged_item",
                    tool_args={
                        "order_id": str(resolved_id),
                        "photo_provided": True,
                        "image_source": image_source or str(img_file),
                    },
                    rationale="Customer submitted photo of damaged goods.",
                )
            return ReasonerDecision(
                action_type=ActionType.ANSWER,
                rationale="Asking for photo evidence",
                draft_response="Please provide photographic evidence of the damaged or defective item.",
            )

    llm = MultiTurnDamagedLLM()
    app = build_graph(llm, checkpointer=MemorySaver())

    thread_id = str(uuid.uuid4())
    config = get_thread_config(thread_id)

    # Turn 1: Customer reports damage without photo
    turn1_state = {
        "conversation_id": thread_id,
        "customer_id": cust_id,
        "messages": [{"role": "user", "content": f"My item for order {order_id} arrived completely smashed."}],
    }
    turn1_result = app.invoke(turn1_state, config=config)
    assert "photographic evidence" in turn1_result["final_response"].lower()

    # Turn 2: Customer submits image via /image input
    turn2_messages = list(turn1_result["messages"])
    turn2_messages.append({
        "role": "user",
        "content": f"[Image attached: {img_file}] Here is the photo of the broken item.",
        "image_source": str(img_file),
    })
    turn2_state = {
        "conversation_id": thread_id,
        "customer_id": cust_id,
        "messages": turn2_messages,
        "image_source": str(img_file),
    }

    turn2_result = app.invoke(turn2_state, config=config)

    # Retrieval should have retained the order_context across turns
    assert turn2_result.get("order_context") is not None
    assert str(turn2_result["order_context"]["order_id"]) == order_id

    # Damaged claim authorized!
    assert turn2_result["status"] == ConversationStatus.RESOLVED
    assert "replacement or refund authorized" in turn2_result["final_response"].lower()

