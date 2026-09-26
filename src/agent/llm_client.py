from __future__ import annotations
from abc import ABC, abstractmethod

from state import ActionType, ReasonerDecision

class LLMClient(ABC):
    @abstractmethod
    def classify_intent(self, message: str, labels: list[str]) -> str: ...

    @abstractmethod
    def decide(self, message: str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision: ...

class MockLLMClient(LLMClient):
    # for checking the graph is working 
    def classify_intent(self, message: str, labels: list[str]) -> str:
        text = message.lower()
        if any ( w in text for w in ["where","status", "track", "arrive"]):
            return "order_status"
        if any (w in text for w in ["refund","money_back", "return"]):
            return "refund request"
        if any (w in text for w in ["ship", "delivery", "address"]):
            return "shipping_question"
        return "general_question"

    def decide(self, message: str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision:
        if order_context is not None:
            return ReasonerDecision(
                action_typr=ActionType.ANSWER,
                rationale="Order context available, answering directly from status",
                draft_response=(
                    f"Your order is currently '{order_context['status']}"
                    f"Let me know if you'd like more detail"
                ),

            )

        return ReasonerDecision(
            action_typr=ActionType.ESCALATE,
            rationale="No order context found and mock client has no fallback reasoning.",
        )

class MLLMClient(LLMClient):

    def __init__(self, model:str = ""):
        # for llm client api
        ...

    def classify_intent(self, message: str, labels: list[str]):
        propmt = (
            f"Classify this customber message into exactly one labes: {labels}.\n"
            f"Message: {message}\n Respond with only the label."
        )
        resp = self.client.message.create(
            model = self.model, max_token=20,
            message=[{"role": "user", "content": propmt}],
        )

        return resp.content[0].text.strip()

    def decide(self, message:str, order_context: dict | None, policy_context: list[str]) -> ReasonerDecision:
        raise NotImplementedError("wire this up once tool schemas are finalized in action.py")
