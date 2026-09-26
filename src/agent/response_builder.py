from __future__ import annotations
from src.agent.observability.logger import with_logging
from state import ActionType, ConversationStatus

@with_logging
def response_builder_node(state:dict) -> dict:
    decision = state["decision"]
    if decision.action.type == ActionType.ANSWER:
        state["final_response"] = decision.draft._response
        state["status"] = ConversationStatus.RESOLVED
    elif decision.action.type == ActionType.ESCALATE:
        state["final_response"] = (
            "I'm not able to resolve this directly — I've flagged it for a "
            "human agent to follow up with you shortly."
        )
        state["status"] = ConversationStatus.ESCALATED
    else: 
        state["final_response"] = "This action isn't wires up yet in this phase"
        state["status"] = ConversationStatus.ESCALATED

    return state
