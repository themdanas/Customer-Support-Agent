from __future__ import annotations
from src.agent.observability.logger import with_logging
try:
    from src.agent.state import ActionType, ConversationStatus
except ImportError:
    from state import ActionType, ConversationStatus

@with_logging("response_builder")
def response_builder_node(state: dict) -> dict:
    decision = state.get("decision")
    if not decision:
        state["final_response"] = "I apologize, but I could not process your request at this time."
        state["status"] = ConversationStatus.ESCALATED
        return state

    action_type = decision.action_type if hasattr(decision, "action_type") else decision.get("action_type")

    if action_type == ActionType.ANSWER:
        draft = decision.draft_response if hasattr(decision, "draft_response") else decision.get("draft_response")
        state["final_response"] = draft or "Your inquiry has been processed."
        state["status"] = ConversationStatus.RESOLVED
    elif action_type == ActionType.ESCALATE:
        state["final_response"] = (
            "I'm not able to resolve this directly — I've flagged it for a "
            "human agent to follow up with you shortly."
        )
        state["status"] = ConversationStatus.ESCALATED
    else:
        state["final_response"] = "This action isn't wired up yet in this phase."
        state["status"] = ConversationStatus.ESCALATED

    return state
