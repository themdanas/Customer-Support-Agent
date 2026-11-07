
from __future__ import annotations
from httpx import get
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

def _get_order_id_from_record(order: Any) -> Optional[list]:
    if not order:
        return None
    if isinstance(order, dict):
        return str(order.get("order_id")) if order.get("order_id") else None
    return str(getattr(order, "order_id", None)) if getattr(order, "order_id", None) else None

def _get_order_status_from_record(order: Any) -> Optional[str]:
    if not order:
        return None
    status = order.get("status") if isinstance(order, dict) else getattr(order, "status", None)
    return getattr(status, "value", str(status)) if status else None

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
    last_message = messages[-1].get("content", "") if messages else ""
    customer_id = state.get("customer_id")
    existing_order = state.get("order_context")

    # 1. Order Context Retrieval
    explicit_order_id = extract_order_id(last_message) if last_message else None
    pinned_order_id = _get_order_id_from_record(existing_order)

    order = None
    target_order_id = explicit_order_id or pinned_order_id

    if target_order_id:
        try:
            order = check_order_status(target_order_id)
        except Exception:
            order = None
        
        #if db look up fails, still keep the order id for the next turn
        if order is None and not explicit_order_id and existing_order is None:
            order = existing_order

    ORDER_INTENTS = {
        "order_status",
        "refund_request",
        "damaged_item",
        "address_change",
        "lost_package",
        }

    if order is None and not explicit_order_id and intent in ORDER_INTENTS and customer_id:
        try:
            order = most_recent_order_for_customer(customer_id)
        except Exception:
            order = None

   #2. Context history and customber profile enrichment
   # pyrefly: ignore[parse-error]
   resolved_order_id = _get_order_id_from_record(order)
   # pyrefly: ignore[parse-error]
   ticket_count = 0
   if resolved_order_id:
    try:
        from src.agent.tools.customer_tools import get_ticket_count_for_order
        ticket_count = get_ticket_count_for_order(resolved_order_id)
    except Exception:
        ticket_count = state.get("ticket_count") or 0
    
    customer_profile = state.get("customer_profile")
    if customer_profile is None and customer_id:
        try:
            from src.agent.tools.customer_tools import get_customer_profile
            customer_profile = get_customer_profile(customer_id)
        except Exception:
            customer_profile = None

    #context aware policy rag
    policy_context = state.get("policy_context") or []
    store = _get_policy_store()
    if store and last_message:
        # Enrich short follow-up replies ("Yes, here is the photo", "Can I change it?")
        # with the classified intent and order status so vector search stays grounded.
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
            policy_context = []

    # Return ONLY the updated state keys (Delta), NEVER `return state`
    return {
        "order_context": order,
        "ticket_count": ticket_count,
        "customer_profile": customer_profile,
        "policy_context": policy_context,
    }

# pyrefly: ignore[parse-error]
