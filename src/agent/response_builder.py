from __future__ import annotations
from src.agent.observability.logger import with_logging
from src.agent.state import ActionType, ConversationStatus

def _finalize(state: dict, final_response: str, status:ConversationStatus) -> dict:
    """Appends assistance response to the conversations messages
    (avoiding the duplicates) and returns states updated delta"""
    existing_messages = list(state.get("messages") or [])
    assistant_msg = {"role": "assistant", "content": final_response}

    if not existing_messages or existing_messages[-1] != assistant_msg:
        updated_messages = existing_messages + [assistant_msg]
    else:
        updated_messages = existing_messages

    return {
        "final_response" : final_response,
        "status": status,
        "messages": updated_messages,
    }


@with_logging("response_builder")
def response_builder_node(state: dict) -> dict:
    decision = state.get("decision")
    if not decision:
        return _finalize(
            state,
            "I apologize, but I could not process your request at this time.",
            ConversationStatus.ESCALATED,
        )

    approval = state.get("approval_decision")
    action_result = state.get("action_result")

    if approval:
        if approval.approved:
            return _finalize(
                state,
                f"Your request requires human approval: {action_result.escalation_reason or 'Policy threshold exceeded.'}",
                ConversationStatus.PENDING_APPROVAL,
            )
        elif action_result.success:
            return _finalize(state, action_result.output, ConversationStatus.RESOLVED)
        else:
            output = action_result.output
            if output.startswith("Please ") or "under company policy" in output.lower():
                final_text = output
            else:
                final_text = f"I was unable to complete that request: {output}"
            return _finalize(state, final_text, ConversationStatus.RESOLVED)

    action_type = (
        decision.action_type
        if hasattr(decision, "action_type")
        else decision.get("action_type")
    )

    if action_type == ActionType.ANSWER:
        draft = (
            decision.draft_response
            if hasattr(decision, "draft_response")
            else decision.get("draft_response")
        )
        return _finalize(
            state,
            draft or "Your inquiry has been processed.",
            ConversationStatus.RESOLVED,
        )
    elif action_type == ActionType.ESCALATE:
        return _finalize(
            state,
            "I'm not able to resolve this directly — I've flagged it for a "
            "human agent to follow up with you shortly.",
            ConversationStatus.ESCALATED,
        )
    else:
        return _finalize(
            state,
            "This action isn't wired up yet in this phase.",
            ConversationStatus.ESCALATED,
        )