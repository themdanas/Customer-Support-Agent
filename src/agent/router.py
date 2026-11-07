from __future__ import annotations
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
    Building a context-aware string for llm.classify_intent(): Should the short 
    follow-up messages (like providing an order ID from a photo on turn 2, not losing turn 1)
    in `if` there is only one message pass the raw message string so the single-turn unit test matches
    """
    if not messages:
        return ""
    
    last_message = messages[-1].get("content", "")
    if len(messages) == 1:
        return last_message

    recent_history = []
    for msg in messages[-4:-1]:
        role = msg.get("role","user")
        content = msg.get("content", "")
        if content:
            recent_history.append(f"{role}:{content}")
    
    history_str = "|".join(recent_history)
    prefix = f"[previous Intent: {prior_intent}]" if prior_intent else ""
    return f"{prefix}[Recent Context: {history_str}] Current user message: {last_message}"


def make_router_node(llm: LLMClient):
    @with_logging("router")
    def router_node(state: dict) -> dict:
        messages = state.get("messages") or []
        last_msg_obj = messages[-1] if messages else {}
        last_message = last_msg_obj.get("content", "") if isinstance(last_msg_obj, dict) else ""
        prior_intent = state.get("intent") 


        # buiding the multi turn aware input for intent classification
        classification_input = _build_classification_input(messages, prior_intent)
        intent = llm.classify_intent(
            classification_input,
            labels=INTENT_LABELS,
        )

        #2. STRICY INTENT GUARAILS 
        if (
            prior_intent in ACTIONABLE_INTENTS
            and intent == "general_question"
            and len(last_message.split()) <= 6
        ):
            intent = prior_intent
        
        # 3. promote the image source if available on the latest message dict
        image_source = (
            last_msg_obj.get("image_source") if isinstance(last_msg_obj, dict) else None
        ) or state.get("image_source")

        return {
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