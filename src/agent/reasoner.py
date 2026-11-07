from __future__ import annotations
from typing import Optional, Any
from src.agent.observability.logger import with_logging
from src.agent.llm_client import LLMClient
from src.agent.retrieval import _get_order_id_from_record


def _resolve_session_image(state: dict, messages: list[dict]) -> Optional[str]:
    """
    Finds the active image_source from top-level state or any prior message in the thread
    so an image uploaded in Turn 1 isn't forgotten when the user provides an Order ID in Turn 2.
    """
    if state.get("image_source"):
        return state["image_source"]
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("image_source"):
            return msg["image_source"]
    return None


def _build_reasoner_message(state: dict, messages: list[dict], image_source: Optional[str]) -> str:
    """
    Constructs a structured reasoner input combining pinned session facts, recent conversation
    history, and the current customer message.
    """
    if not messages:
        return ""

    last_msg = messages[-1]
    last_message = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)

    if len(messages) <= 1:
        return last_message

    history_lines = []
    for msg in messages[-9:-1]:
        if not isinstance(msg, dict):
            continue
        role = "Customer" if msg.get("role") == "user" else "Agent"
        content = msg.get("content", "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    order_ctx = state.get("order_context")
    order_id = _get_order_id_from_record(order_ctx)

    context_header = [
        "=== CONVERSATION MEMORY & PINNED FACTS ===",
        f"Active Intent: {state.get('intent') or 'unknown'}",
        f"Pinned Order ID: {order_id or 'Not yet provided'}",
        f"Ticket Count on Order: {state.get('ticket_count', 0)}",
        f"Image Evidence in Session: {image_source or 'None'}",
        "--- Recent Conversation History ---",
        *history_lines,
        "==========================================",
        f"Current Customer Message: {last_message}",
    ]
    return "\n".join(context_header)


def make_reasoner_node(llm: LLMClient):
    @with_logging("reasoner")
    def reasoner_node(state: dict) -> dict:
        messages = state.get("messages") or []
        image_source = _resolve_session_image(state, messages)
        reasoner_input = _build_reasoner_message(state, messages, image_source)

        if not image_source and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                image_source = last_msg.get("image_source")

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

        # Return shallow copy of state merged with updated fields for dual-compatibility
        return {
            **state,
            "decision": decision,
        }

    return reasoner_node


# Backwards compatibility alias
make_resaoner_node = make_reasoner_node
