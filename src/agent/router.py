from __future__ import annotations
from typing import Optional
from src.agent.observability.logger import with_logging
from src.agent.llm_client import LLMClient
from src.agent.state import ConversationStatus

INTENT_LABELS = [
    "order_status",
    "refund_request",
    "shipping_question",
    "general_question",
]

ACTIONABLE_INTENTS = {"order_status", "refund_request", "shipping_question"}


def _build_classification_input(messages: list[dict], prior_intent: str | None) -> str:
    """
    Builds a context-enriched string for multi-turn intent classification
    with the current user message positioned first to ensure prompt precedence.
    """
    if not messages:
        return ""

    last_msg = messages[-1]
    last_message = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
    if len(messages) <= 1:
        return last_message

    recent_history = []
    for msg in messages[-4:-1]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if content:
            recent_history.append(f"{role}: {content}")

    history_str = " | ".join(recent_history)
    prefix = f"[Prior Intent: {prior_intent}] " if prior_intent else ""
    return f"Current message: {last_message}\n{prefix}[Recent Context: {history_str}]"


def make_router_node(llm: LLMClient):
    @with_logging("router")
    def router_node(state: dict) -> dict:
        messages = state.get("messages") or []
        last_msg_obj = messages[-1] if messages else {}
        last_message = (
            last_msg_obj.get("content", "") if isinstance(last_msg_obj, dict) else str(last_msg_obj)
        )
        prior_intent = state.get("intent")

        # 1. Primary classification on latest customer message
        # This guarantees new customer requests (e.g. asking for a refund in Turn 2)
        # take priority over prior turn context in MockLLMClient and production models.
        current_intent = llm.classify_intent(last_message, labels=INTENT_LABELS)

        if current_intent in ACTIONABLE_INTENTS:
            intent = current_intent
        elif prior_intent in ACTIONABLE_INTENTS and len(last_message.split()) <= 8:
            # Short follow-ups ("yes please", "here is the photo", "can you?") maintain prior actionable flow
            intent = prior_intent
        elif len(messages) > 1:
            # Use enriched context for ambiguous messages
            context_input = _build_classification_input(messages, prior_intent)
            context_intent = llm.classify_intent(context_input, labels=INTENT_LABELS)
            intent = context_intent if context_intent != "general_question" else (prior_intent or "general_question")
        else:
            intent = current_intent

        # 2. Promote image source if available on latest message or already in state
        image_source = (
            last_msg_obj.get("image_source") if isinstance(last_msg_obj, dict) else None
        ) or state.get("image_source")

        # 3. Return shallow copy of state merged with updated fields for dual-compatibility
        return {
            **state,
            "intent": intent,
            "image_source": image_source,
            "status": ConversationStatus.IN_PROGRESS,
            "decision": None,
            "action_result": None,
            "escalation": None,
            "approval_decision": None,
            "final_response": None,
        }

    return router_node
