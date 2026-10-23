import os
from pathlib import Path
from src.agent.observability.logger import with_logging
from src.agent.tools.order_tools import extract_order_id, check_order_status, most_recent_order_for_customer

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
    intent = state.get("intent")
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if messages else ""

    # 1. Order Context Retrieval
    if intent == "order_status":
        order_id = extract_order_id(last_message)
        order = None
        if order_id:
            try:
                order = check_order_status(order_id)
            except Exception:
                order = None

        if order is None and state.get("customer_id"):
            try:
                order = most_recent_order_for_customer(state["customer_id"])
            except Exception:
                order = None

        state["order_context"] = order
    else:
        state["order_context"] = None

    # 2. Policy Context Retrieval
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
