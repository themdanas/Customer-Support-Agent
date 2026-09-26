from __future__ import annotations
from src.agent.observability.logger import with_logging
from llm_client import LLMClient

def make_router_node(llm: LLMClient):
    @with_logging("router")
    def router_node(state:dict) -> dict:
        last_message = state["messages"][-1]["content"]
        intent = llm.classify_intent(
            last_message,
            labels=["order_status", "refund_request","shipping_question", "general_question"]
        )
        state["intent"] = intent
        return state
    return router_node
