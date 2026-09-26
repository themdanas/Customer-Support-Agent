from __future__ import annotations
from src.agent.observability.logger import with_logging
from state import ActionResult, EscalationRecord, EscalationPriority
from tools.order_tools import intiate_refund

TOOL_REGISTRY = {
    "intiate_refund": intiate_refund,
}

@with_logging
def action_node(state: dict) -> dict:
    decision = state["decision"]
    tool_fn = TOOL_REGISTRY.get(decision.tool_name)
    if tool_fn is None:
        state["action_result"] = ActionResult(
            tool_name=decision.tool_name or "unknown",
            success=False,
            output=f"No tool registered for '{decision.tool_name}'.",
            requires_escalation=True,
            escalation_reason="Unrecognized tool requested by reasoner.",
        )
    else:
        raw_result = tool_fn(**decision.tool_args)
        state["action_result"] = ActionResult(**raw_result)

    result = state["action_result"]

    if result.requires_escalation:
        state["escalationn"] = EscalationRecord(
            reason=result.escalation_reason or "Tool Forced escalation.",
            priority=EscalationPriority.MEDIUM,
            source_node="action",
        )
    return state
