from __future__ import annotations
from typing import Any
from src.agent.observability.logger import with_logging
from src.agent.state import ActionType, ConversationStatus


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to safely access fields on both dicts and Pydantic models."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _finalize(state: dict, final_response: str, status: ConversationStatus) -> dict:
    """Appends assistant response to conversation messages and returns updated state."""
    existing_messages = list(state.get("messages") or [])
    assistant_msg = {"role": "assistant", "content": final_response}

    if not existing_messages or existing_messages[-1] != assistant_msg:
        updated_messages = existing_messages + [assistant_msg]
    else:
        updated_messages = existing_messages

    return {
        **state,
        "final_response": final_response,
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

    # 1. Resumed from human supervisor approval flow
    if approval:
        approved = _get_val(approval, "approved", False)
        if approved:
            extra = _get_val(action_result, "output", "") if action_result else ""
            msg = f"Your request has been approved and processed. {extra}".strip()
            return _finalize(state, msg, ConversationStatus.RESOLVED)
        else:
            reason = _get_val(approval, "reason") or "Policy limit exceeded"
            msg = f"Your request was reviewed and could not be approved: {reason}."
            return _finalize(state, msg, ConversationStatus.RESOLVED)

    # 2. Tool action execution result
    if action_result:
        requires_escalation = _get_val(action_result, "requires_escalation", False)
        escalation_reason = _get_val(action_result, "escalation_reason", None)
        success = _get_val(action_result, "success", False)
        output = _get_val(action_result, "output", "")

        if requires_escalation:
            msg = f"Your request requires human approval: {escalation_reason or 'Policy threshold exceeded.'}"
            return _finalize(state, msg, ConversationStatus.PENDING_APPROVAL)
        elif success:
            return _finalize(state, output, ConversationStatus.RESOLVED)
        else:
            if output.startswith("Please ") or "under company policy" in output.lower():
                msg = output
            else:
                msg = f"I was unable to complete that request: {output}"
            return _finalize(state, msg, ConversationStatus.RESOLVED)

    # 3. Direct answer or reasoner decision
    action_type = _get_val(decision, "action_type")

    if action_type == ActionType.ANSWER:
        draft = _get_val(decision, "draft_response")
        return _finalize(
            state,
            draft or "Your inquiry has been processed.",
            ConversationStatus.RESOLVED,
        )
    elif action_type == ActionType.ESCALATE:
        return _finalize(
            state,
            "I'm not able to resolve this directly — I've flagged it for a human agent to follow up with you shortly.",
            ConversationStatus.ESCALATED,
        )
    else:
        return _finalize(
            state,
            "This action isn't wired up yet in this phase.",
            ConversationStatus.ESCALATED,
        )
