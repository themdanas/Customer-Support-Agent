from __future__ import annotations
from src.agent.observability.logger import with_logging
from src.agent.state import ActionResult, EscalationRecord, EscalationPriority
from src.agent.tools.order_tools import (
    initiate_refund,
    change_shipping_address,
    report_lost_package,
    report_damaged_item,
    check_order_status,
)
from src.agent.tools.customer_tools import (
    get_customer_profile,
    get_customer_orders,
    check_repeat_contact_escalation,
)

TOOL_REGISTRY = {
    "initiate_refund": initiate_refund,
    "intiate_refund": initiate_refund,  # Backwards compatibility alias
    "change_shipping_address": change_shipping_address,
    "report_lost_package": report_lost_package,
    "report_damaged_item": report_damaged_item,
    "check_order_status": check_order_status,
    "get_customer_profile": get_customer_profile,
    "get_customer_orders": get_customer_orders,
    "check_repeat_contact_escalation": check_repeat_contact_escalation,
}


@with_logging("action")
def action_node(state: dict) -> dict:
    """
    Executes tool calls specified by the reasoner node with deterministic guardrails.
    If the tool execution requires human escalation (e.g. policy violations, high refund amounts,
    lost package claims), flags the state with an EscalationRecord so the conditional router
    routes the thread into the human_approval node.
    """
    decision = state.get("decision")
    if not decision:
        return state

    tool_name = (
        getattr(decision, "tool_name", None)
        or (decision.get("tool_name") if isinstance(decision, dict) else None)
    )
    tool_args = (
        getattr(decision, "tool_args", {})
        or (decision.get("tool_args", {}) if isinstance(decision, dict) else {})
    )

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        state["action_result"] = ActionResult(
            tool_name=tool_name or "unknown",
            success=False,
            output=f"No tool registered for '{tool_name}'.",
            requires_escalation=True,
            escalation_reason=f"Unrecognized tool requested by reasoner: '{tool_name}'.",
        )
    else:
        try:
            raw_result = tool_fn(**tool_args)
            if isinstance(raw_result, dict):
                state["action_result"] = ActionResult(**raw_result)
            else:
                state["action_result"] = ActionResult(
                    tool_name=tool_name,
                    success=True,
                    output=str(raw_result),
                    requires_escalation=False,
                )
        except Exception as e:
            state["action_result"] = ActionResult(
                tool_name=tool_name,
                success=False,
                output=f"Error executing tool '{tool_name}': {str(e)}",
                requires_escalation=True,
                escalation_reason=f"Tool exception: {str(e)}",
            )

    result = state["action_result"]

    if result.requires_escalation:
        state["escalation"] = EscalationRecord(
            reason=result.escalation_reason or "Tool forced escalation.",
            priority=EscalationPriority.MEDIUM,
            source_node="action",
        )
    return state
