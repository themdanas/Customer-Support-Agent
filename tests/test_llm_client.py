from __future__ import annotations
import os
import pytest
from unittest.mock import MagicMock, patch

from src.agent.llm_client import MockLLMClient, GroqLLMClient, get_llm_client
from src.agent.state import ActionType


def test_mock_llm_client_intent():
    client = MockLLMClient()
    assert client.classify_intent("Where is my package?", ["order_status", "refund_request"]) == "order_status"
    assert client.classify_intent("I want my money back", ["order_status", "refund_request"]) == "refund_request"
    assert client.classify_intent("What is your address?", ["order_status", "shipping_question"]) == "shipping_question"


def test_mock_llm_client_decision_escalates_on_anger():
    client = MockLLMClient()
    decision = client.decide(
        message="I am furious with this service!",
        order_context={"status": "shipped"},
        policy_context=[],
    )
    assert decision.action_type == ActionType.ESCALATE


def test_groq_client_initialization_with_env_key():
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_test_key"}):
        with patch("groq.Groq") as mock_groq:
            client = GroqLLMClient()
            assert client is not None
            mock_groq.assert_called_once_with(api_key="fake_test_key")


def test_groq_client_structured_decision_parsing():
    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_test_key"}):
        with patch("groq.Groq") as mock_groq:
            mock_instance = MagicMock()
            mock_groq.return_value = mock_instance

            # Mock JSON response from Groq
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(
                        content='''{
                            "action_type": "tool_call",
                            "rationale": "Customer requested refund under policy limit.",
                            "tool_name": "initiate_refund",
                            "tool_args": {"order_id": "test-order-1", "amount": "45.00"},
                            "draft_response": null
                        }'''
                    )
                )
            ]
            mock_instance.chat.completions.create.return_value = mock_response

            client = GroqLLMClient()
            decision = client.decide(
                message="Refund order test-order-1 for $45",
                order_context={"order_id": "test-order-1", "status": "delivered"},
                policy_context=["Refunds under $75 auto-approve."],
            )

            assert decision.action_type == ActionType.TOOL_CALL
            assert decision.tool_name == "initiate_refund"
            assert decision.tool_args["amount"] == 45.0
            assert decision.tool_args["order_id"] == "test-order-1"

