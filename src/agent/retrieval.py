from __future__ import annotations
from src.agent.observability.logger import with_logging
from src.agent.tools.order_tools import extract_order_id, check_order_status, most_recent_order_for_customer

@with_logging("retrieval")
def retrieval_node(state: dict) -> dict:
    if state.get("intent") != "order_status":
        state["order_context"] = None
        return state

    last_message = state["messages"][-1]["content"]
    order_id = extract_order_id(last_message)

    order = check_order_status(order_id) if order_id else None
    if order is None:
        order = most_recent_order_for_customer(state["customer_id"])

    state["order_context"] = order
    return state
