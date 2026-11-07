from __future__ import annotations
from typing import Any
from src.agent.observability.logger import with_logging
from src.agent.state import ActionResult, EscalationRecord, EscalationPriority
from src.agent.retrieval import _get_order_id_from_record
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
from src.agent.tools.vision_tools import inspect_damage_photo

TOOL_REGISTRY = {
    "initiate_refund": initiate_refund,
    "intiate_refund": initiate_refund,  # Backwards compatibility alias
    "change_shipping_address": change_shipping_address,
    "report_lost_package": report_lost_package,
    "report_damaged_item": report_damaged_item,
    "inspect_damage_photo": inspect_damage_photo,
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
        return {**state}

    tool_name = (
        getattr(decision, "tool_name", None)
        or (decision.get("tool_name") if isinstance(decision, dict) else None)
    )
    tool_args = dict(
        getattr(decision, "tool_args", {})
        or (decision.get("tool_args", {}) if isinstance(decision, dict) else {})
    )

    # Automatic parameter enrichment from state context
    order_ctx = state.get("order_context")
    resolved_id = _get_order_id_from_record(order_ctx)
    if "order_id" not in tool_args and resolved_id:
        tool_args["order_id"] = str(resolved_id)

    if tool_name == "report_damaged_item":
        if not tool_args.get("image_source") and state.get("image_source"):
            tool_args["image_source"] = state["image_source"]
            tool_args["photo_provided"] = True

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        action_result = ActionResult(
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
                action_result = ActionResult(**raw_result)
            else:
                action_result = ActionResult(
                    tool_name=tool_name,
                    success=True,
                    output=str(raw_result),
                    requires_escalation=False,
                )
        except Exception as e:
            action_result = ActionResult(
                tool_name=tool_name,
                success=False,
                output=f"Error executing tool '{tool_name}': {str(e)}",
                requires_escalation=True,
                escalation_reason=f"Tool exception: {str(e)}",
            )

    updates: dict[str, Any] = {
        "action_result": action_result,
    }

    if action_result.requires_escalation:
        updates["escalation"] = EscalationRecord(
            reason=action_result.escalation_reason or "Tool forced escalation.",
            priority=EscalationPriority.MEDIUM,
            source_node="action",
        )

    # Refresh order_context if state-changing tool was executed
    target_order_id = tool_args.get("order_id") or resolved_id
    if target_order_id and tool_name in ("change_shipping_address", "initiate_refund", "report_damaged_item"):
        try:
            refreshed = check_order_status(str(target_order_id))
            if refreshed:
                updates["order_context"] = refreshed
        except Exception:
            pass

    return {
        **state,
        **updates,
    }
