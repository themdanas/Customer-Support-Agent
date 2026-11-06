from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from src.agent.observability.logger import with_logging
from src.agent.tools.order_tools import (
    extract_order_id,
    check_order_status,
    most_recent_order_for_customer,
)

_POLICY_STORE = None


def _get_policy_store():
    global _POLICY_STORE
    if _POLICY_STORE is None:
        idx_path = Path("policy_index.pkl")
        if idx_path.exists():
            try:
                from src.agent.tools.policy_rag.vector_store import PolicyVectorStore
                _POLICY_STORE = PolicyVectorStore.load(str(idx_path))
            except Exception:
                _POLICY_STORE = None
    return _POLICY_STORE


@with_logging("retrieval")
def retrieval_node(state: dict) -> dict:
    """
    Enriches the graph state by fetching:
    1. Order context (from UUID in message or customer's most recent order).
    2. Repeat contact history (ticket count for the order to enforce escalation rules).
    3. Grounding policy chunks from the vector store based on semantic query.
    """
    intent = state.get("intent")
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if messages else ""

    # 1. Order Context Retrieval
    order_id = extract_order_id(last_message)
    order = None

    if order_id:
        try:
            order = check_order_status(order_id)
        except Exception:
            order = None

    # Multi-turn preservation: Retain existing order_context from previous turns
    if order is None and state.get("order_context"):
        order = state.get("order_context")

    # Multi-turn search: Look backward through earlier messages if order was mentioned earlier
    if order is None and messages:
        for prev in reversed(messages[:-1]):
            prev_content = prev.get("content", "")
            prev_id = extract_order_id(prev_content)
            if prev_id:
                try:
                    order = check_order_status(prev_id)
                    if order:
                        break
                except Exception:
                    pass

    # Fallback to customer's most recent order when intent implies an order or refund inquiry
    if order is None and intent in ("order_status", "refund_request") and state.get("customer_id"):
        try:
            order = most_recent_order_for_customer(state["customer_id"])
        except Exception:
            order = None

    state["order_context"] = order

    # 2. Contact History Enrichment (Repeat Contact Escalation Rule)
    resolved_order_id = (
        order.get("order_id") if isinstance(order, dict) else getattr(order, "order_id", None)
    )
    if resolved_order_id:
        try:
            from src.agent.tools.customer_tools import get_ticket_count_for_order
            state["ticket_count"] = get_ticket_count_for_order(str(resolved_order_id))
        except Exception:
            state["ticket_count"] = 0
    else:
        state["ticket_count"] = 0

    # 3. Policy Context Retrieval
    store = _get_policy_store()
    if store and last_message:
        try:
            chunks = store.query(last_message, k=2)
            state["policy_context"] = chunks
        except Exception:
            state["policy_context"] = []
    else:
        state.setdefault("policy_context", [])

    return state
