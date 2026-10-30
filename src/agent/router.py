from __future__ import annotations
from src.agent.observability.logger import with_logging
from src.agent.llm_client import LLMClient

def make_router_node(llm: LLMClient):
    @with_logging("router")
    def router_node(state: dict) -> dict:
        messages = state.get("messages", [])
        last_message = messages[-1]["content"] if messages else ""
        intent = llm.classify_intent(
            last_message,
            labels=["order_status", "refund_request", "shipping_question", "general_question"]
        )
        state["intent"] = intent
        return state
    return router_node
