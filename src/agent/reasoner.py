from __future__ import annotations
from src.agent.observability.logger import with_logging
from src.agent.llm_client import LLMClient

def make_reasoner_node(llm: LLMClient):
    @with_logging("reasoner")
    def reasoner_node(state: dict) -> dict:
        messages = state.get("messages", [])
        last_message = messages[-1]["content"] if messages else ""
        image_source = state.get("image_source")
        if not image_source and messages:
            image_source = messages[-1].get("image_source")

        policy_texts = [
            c.text if hasattr(c, "text") else c.get("text", "")
            for c in state.get("policy_context", [])
        ]
        try:
            decision = llm.decide(
                message=last_message,
                order_context=state.get("order_context"),
                policy_context=policy_texts,
                image_source=image_source,
            )
        except TypeError:
            decision = llm.decide(
                message=last_message,
                order_context=state.get("order_context"),
                policy_context=policy_texts,
            )
        state["decision"] = decision
        return state
    return reasoner_node

# Backwards compatibility alias
make_resaoner_node = make_reasoner_node
                    