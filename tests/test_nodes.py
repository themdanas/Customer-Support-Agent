from __future__ import annotations
import pytest

from src.agent.llm_client import MockLLMClient
from src.agent.router import make_router_node
from src.agent.retrieval import retrieval_node
from src.agent.reasoner import make_reasoner_node
from src.agent.response_builder import response_builder_node
from src.agent.graph import build_graph
from src.agent.state import ActionType, ReasonerDecision, ConversationStatus

def test_router_node_classifies_intent():
    llm = MockLLMClient()
    router = make_router_node(llm)

    state = {
        "messages": [{"role": "user", "content": "Where is my order?"}],
    }
    updated_state = router(state)

    assert updated_state["intent"] == "order_status"
    assert len(updated_state["trace"]) == 1
    assert updated_state["trace"][0].node_name == "router"
    assert "intent=order_status" in updated_state["trace"][0].output_summary

def test_retrieval_node_handles_no_order():
    state = {
        "messages": [{"role": "user", "content": "Can you track order 12345?"}],
        "intent": "order_status",
    }
    updated_state = retrieval_node(state)

    # Since 12345 is not a valid UUID and no DB is mocked, order_context is None
    assert updated_state["order_context"] is None
    assert any(log.node_name == "retrieval" for log in updated_state["trace"])

def test_retrieval_node_queries_policy_for_policy_intent():
    state = {
        "messages": [{"role": "user", "content": "What is your refund policy?"}],
        "intent": "refund_request",
    }
    updated_state = retrieval_node(state)

    assert updated_state["order_context"] is None
    # Policy context should be populated from policy_index.pkl if present
    assert isinstance(updated_state.get("policy_context"), list)
    assert any(log.node_name == "retrieval" for log in updated_state["trace"])

def test_reasoner_node_with_order_context():
    llm = MockLLMClient()
    reasoner = make_reasoner_node(llm)

    state = {
        "messages": [{"role": "user", "content": "Where is my package?"}],
        "order_context": {"order_id": "fake-id", "status": "shipped"},
        "policy_context": [],
    }
    updated_state = reasoner(state)

    assert updated_state["decision"] is not None
    assert updated_state["decision"].action_type == ActionType.ANSWER
    assert "shipped" in updated_state["decision"].draft_response
    assert any(log.node_name == "reasoner" for log in updated_state["trace"])

def test_reasoner_node_without_order_context():
    llm = MockLLMClient()
    reasoner = make_reasoner_node(llm)

    state = {
        "messages": [{"role": "user", "content": "Where is my package?"}],
        "order_context": None,
        "policy_context": [],
    }
    updated_state = reasoner(state)

    assert updated_state["decision"] is not None
    assert updated_state["decision"].action_type == ActionType.ESCALATE

def test_response_builder_node_answer():
    decision = ReasonerDecision(
        action_type=ActionType.ANSWER,
        rationale="Order found",
        draft_response="Your order is currently shipped.",
    )
    state = {"decision": decision}
    updated_state = response_builder_node(state)

    assert updated_state["status"] == ConversationStatus.RESOLVED
    assert "shipped" in updated_state["final_response"]
    assert any(log.node_name == "response_builder" for log in updated_state["trace"])

def test_response_builder_node_escalate():
    decision = ReasonerDecision(
        action_type=ActionType.ESCALATE,
        rationale="Needs human attention",
    )
    state = {"decision": decision}
    updated_state = response_builder_node(state)

    assert updated_state["status"] == ConversationStatus.ESCALATED
    assert "human agent" in updated_state["final_response"]


def test_response_builder_instructional_guidance_no_robotic_prefix():
    from src.agent.state import ActionResult
    action_result = ActionResult(
        tool_name="report_damaged_item",
        success=False,
        output="Please provide photographic evidence of the damaged or defective item. Once photos are submitted, we will immediately process your replacement or refund.",
        requires_escalation=False,
    )
    state = {
        "decision": ReasonerDecision(action_type=ActionType.TOOL_CALL, rationale="Checking damage"),
        "action_result": action_result,
    }
    updated_state = response_builder_node(state)
    assert updated_state["status"] == ConversationStatus.RESOLVED
    assert updated_state["final_response"].startswith("Please provide photographic evidence")
    assert "Unable to complete request:" not in updated_state["final_response"]

def test_full_graph_execution_order_status_escalation_flow():
    """When order is not found, MockLLM escalates to human."""
    llm = MockLLMClient()
    app = build_graph(llm)

    initial_state = {
        "conversation_id": "test-conv-1",
        "customer_id": "cust-1",
        "messages": [{"role": "user", "content": "Where is my order?"}],
    }

    result = app.invoke(initial_state)

    assert result["intent"] == "order_status"
    assert result["status"] == ConversationStatus.ESCALATED
    assert "human agent" in result["final_response"]

    # Verify all 4 nodes logged their steps in the trace
    trace_nodes = [log.node_name for log in result["trace"]]
    assert trace_nodes == ["router", "retrieval", "reasoner", "response_builder"]
    for log in result["trace"]:
        assert log.latency_ms >= 0
