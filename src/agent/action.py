from __future__ import annotations
from src.agent.observability.logger import with_logging
try:
    from src.agent.state import ActionResult, EscalationRecord, EscalationPriority
except ImportError:
    from state import ActionResult, EscalationRecord, EscalationPriority

try:
    from src.agent.tools.order_tools import initiate_refund
except ImportError:
    from tools.order_tools import initiate_refund

TOOL_REGISTRY = {
    "initiate_refund": initiate_refund,
    "intiate_refund": initiate_refund,
}

@with_logging("action")
def action_node(state: dict) -> dict:
    decision = state.get("decision")
    if not decision:
        return state

    tool_name = getattr(decision, "tool_name", None) or (decision.get("tool_name") if isinstance(decision, dict) else None)
    tool_args = getattr(decision, "tool_args", {}) or (decision.get("tool_args", {}) if isinstance(decision, dict) else {})

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        state["action_result"] = ActionResult(
            tool_name=tool_name or "unknown",
            success=False,
            output=f"No tool registered for '{tool_name}'.",
            requires_escalation=True,
            escalation_reason="Unrecognized tool requested by reasoner.",
        )
    else:
        raw_result = tool_fn(**tool_args)
        state["action_result"] = ActionResult(**raw_result)

    result = state["action_result"]

    if result.requires_escalation:
        state["escalation"] = EscalationRecord(
            reason=result.escalation_reason or "Tool forced escalation.",
            priority=EscalationPriority.MEDIUM,
            source_node="action",
        )
    return state
