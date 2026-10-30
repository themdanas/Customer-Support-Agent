from __future__ import annotations
from abc import ABC, abstractmethod

from src.agent.state import ActionType, ReasonerDecision

class LLMClient(ABC):
    @abstractmethod
    def classify_intent(self, message: str, labels: list[str]) -> str: ...

    @abstractmethod
    def decide(self, message: str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision: ...

class MockLLMClient(LLMClient):
    # for checking the graph is working 
    def classify_intent(self, message: str, labels: list[str]) -> str:
        text = message.lower()
        if any(w in text for w in ["where", "status", "track", "arrive"]):
            return "order_status"
        if any(w in text for w in ["refund", "money_back", "return"]):
            return "refund_request"
        if any(w in text for w in ["ship", "delivery", "address"]):
            return "shipping_question"
        return "general_question"

    def decide(self, message: str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision:
        if order_context is not None:
            return ReasonerDecision(
                action_type=ActionType.ANSWER,
                rationale="Order context available, answering directly from status",
                draft_response=(
                    f"Your order is currently '{order_context['status']}'. "
                    f"Let me know if you'd like more detail."
                ),
            )

        return ReasonerDecision(
            action_type=ActionType.ESCALATE,
            rationale="No order context found and mock client has no fallback reasoning.",
        )

class MLLMClient(LLMClient):

    def __init__(self, model: str = ""):
        # for llm client api
        self.model = model

    def classify_intent(self, message: str, labels: list[str]):
        prompt = (
            f"Classify this customer message into exactly one label: {labels}.\n"
            f"Message: {message}\n Respond with only the label."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )

        return resp.content[0].text.strip()

    def decide(self, message: str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision:
        raise NotImplementedError("wire this up once tool schemas are finalized in action.py")
