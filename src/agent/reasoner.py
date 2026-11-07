from __future__ import annotations
from src.agent.observability.logger import with_logging
from src.agent.llm_client import LLMClient
from typing import Optional

def _resolve_session_image(state:dict, messages: list[dict]) -> Optional[str]:
    """
    Finds the active image_source from top-level state or any prior message in the thread
    so an image uploaded in Turn 1 isn't forgotten when the user gives their Order ID in Turn 2.   
    """
    if state.get("image_source"):
        return state["image_source"]
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("image_source"):
            return msg["image_source"]
    return None
        
def _build_reasoner_message(state: dict, messages: list[dict], image_source: Optional[str]) -> str:
    if not messages:
        return ""
    last_message = messages[-1].get("content","") if isinstance(messages[-1], dict) else str(messages[-1])

    if len(messages) <= 1:
        return last_message

    history_lines = []
    for msg in messages[-9,-1]:
        if not isinstance(msg, dict):
            continue
        role = "Customer" if msg.get("role") == "user" else "Agent"
        content = msg.get("content","").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    order_ctx = state.get("order_contex")
    order_id = (
        order_ctx.get("order_id")
        if isinstance(order_ctx, dict)
        else getattr(order_ctx, "order_id", None)
    ) if order_ctx else None
    
    context_header = [
        "=== CONVERSATION MEMORY & PINNED FACTS (DO NOT ASK AGAIN IF PRESENT) ===",
        f"Active Intent: {state.get('intent') or 'unknown'}",
        f"Pinned Order ID: {order_id or 'Not yet provided'}",
        f"Ticket Count on Order: {state.get('ticket_count', 0)}",
        f"Image Evidence in Session: {image_source or 'None'}",
        "--- Recent Conversation History ---",
        *history_lines,
        "========================================================================",
        f"Current Customer Message: {last_message}",
    ]
    return "\n".join(context_header)



def make_reasoner_node(llm: LLMClient):
    @with_logging("reasoner")
    def reasoner_node(state: dict) -> dict:
        messages = state.get("messages", [])
        image_source = _resolve_session_image(state, messages)
        reasoner_input = _build_reasoner_message(state, messages, image_source)

        if not image_source and messages:
            image_source = messages[-1].get("image_source")

        policy_texts = [
            c.text if hasattr(c, "text") else c.get("text", "")
            for c in state.get("policy_context", [])
        ]
        try:
            decision = llm.decide(
                message=reasoner_input,
                order_context=state.get("order_context"),
                policy_context=policy_texts,
                image_source=image_source,
            )
        except TypeError:
            decision = llm.decide(
                message=reasoner_input,
                order_context=state.get("order_context"),
                policy_context=policy_texts,
            )
        state["decision"] = decision
        return state
    return reasoner_node

# Backwards compatibility alias
make_resaoner_node = make_reasoner_node
                    