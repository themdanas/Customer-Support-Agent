from __future__ import annotations
from src.agent.observability.logger import with_logging
from llm_client import LLMClient

def make_resaoner_node(llm: LLMClient):
    @with_logging("reasoner")
    def reasoner_node(state: dict) -> dict:
        last_message = state["messages"][-1]["content"]
        decision = llm.decide(
            message=last_message,
            order_context=state.get("order_context"),
            policy_context=[c.text for c in state.get("policy_context",[])]
        )
        state["decision"] = decision
        return state
    return reasoner_node
                    