from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Any

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


def _get_order_id_from_record(order: Any) -> Optional[str]:
    """Safely extracts order_id from either a dict or an OrderRecord Pydantic model."""
    if not order:
        return None
    if isinstance(order, dict):
        val = order.get("order_id")
        return str(val) if val else None
    val = getattr(order, "order_id", None)
    return str(val) if val else None


def _get_order_status_from_record(order: Any) -> Optional[str]:
    """Safely extracts status string from either a dict or an OrderRecord Pydantic model."""
    if not order:
        return None
    status = order.get("status") if isinstance(order, dict) else getattr(order, "status", None)
    return getattr(status, "value", str(status)) if status else None


ORDER_INTENTS = {
    "order_status",
    "refund_request",
    "shipping_question",
    "general_question",
}


@with_logging("retrieval")
def retrieval_node(state: dict) -> dict:
    """
    Enriches the graph state by fetching:
    1. Order context (from UUID in message, existing pinned state, or customer's most recent order).
    2. Repeat contact history (ticket count for the order to enforce escalation rules).
    3. Grounding policy chunks from the vector store based on semantic query.
    """
    intent = state.get("intent")
    messages = state.get("messages") or []
    last_msg_obj = messages[-1] if messages else {}
    last_message = (
        last_msg_obj.get("content", "") if isinstance(last_msg_obj, dict) else str(last_msg_obj)
    )
    customer_id = state.get("customer_id")
    existing_order = state.get("order_context")

    # 1. Order Context Retrieval
    explicit_order_id = extract_order_id(last_message) if last_message else None
    pinned_order_id = _get_order_id_from_record(existing_order)

    # Search backwards through earlier conversation messages if no order id in current message
    if not explicit_order_id and not pinned_order_id and messages:
        for prev in reversed(messages[:-1]):
            prev_content = prev.get("content", "") if isinstance(prev, dict) else ""
            extracted = extract_order_id(prev_content)
            if extracted:
                explicit_order_id = extracted
                break

    target_order_id = explicit_order_id or pinned_order_id
    order = None

    if target_order_id:
        try:
            order = check_order_status(target_order_id)
        except Exception:
            order = None

        # If DB lookup fails or is unavailable (e.g. mock order or offline test), preserve existing order
        if order is None and existing_order is not None:
            order = existing_order

    # Fallback to customer's most recent order when intent implies an order inquiry
    if order is None and not explicit_order_id and intent in ("order_status", "refund_request") and customer_id:
        try:
            order = most_recent_order_for_customer(customer_id)
        except Exception:
            order = None

    # 2. Contact History and Customer Profile Enrichment
    resolved_order_id = _get_order_id_from_record(order)
    ticket_count = 0
    if resolved_order_id:
        try:
            from src.agent.tools.customer_tools import get_ticket_count_for_order
            ticket_count = get_ticket_count_for_order(resolved_order_id)
        except Exception:
            ticket_count = state.get("ticket_count") or 0
    else:
        ticket_count = state.get("ticket_count") or 0

    customer_profile = state.get("customer_profile")
    if customer_profile is None and customer_id:
        try:
            from src.agent.tools.customer_tools import get_customer_profile
            customer_profile = get_customer_profile(customer_id)
        except Exception:
            customer_profile = None

    # 3. Context-Aware Policy RAG Retrieval
    policy_context = list(state.get("policy_context") or [])
    store = _get_policy_store()
    if store and last_message:
        order_status_str = _get_order_status_from_record(order)
        query_parts = []
        if intent:
            query_parts.append(f"intent: {intent}")
        if order_status_str:
            query_parts.append(f"order status: {order_status_str}")
        query_parts.append(last_message)
        enriched_query = " | ".join(query_parts)

        try:
            policy_context = store.query(enriched_query, k=2)
        except Exception:
            try:
                policy_context = store.query(last_message, k=2)
            except Exception:
                pass

    # Return shallow copy of state merged with updated fields for dual-compatibility
    return {
        **state,
        "order_context": order,
        "ticket_count": ticket_count,
        "customer_profile": customer_profile,
        "policy_context": policy_context,
    }
