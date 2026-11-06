from __future__ import annotations
import os
import re
from datetime import datetime, timezone
from typing import Optional, Any
import psycopg2
import psycopg2.extras

AUTO_APPROVE_MAX_AMOUNT = 75.00
AUTO_APPROVE_MAX_RATIO = 0.80

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/support_agent"


def _get_db_conn():
    return psycopg2.connect(os.environ.get("DATABASE_URL", DEFAULT_DB_URL))


def extract_order_id(message: str) -> str | None:
    match = UUID_RE.search(message)
    return match.group(0) if match else None


def check_order_status(order_id: str) -> dict | None:
    """Returns order record as a plain dict if found."""
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT order_id::text, customer_id::text, status, total_amount,
                   placed_at, delivered_at
            FROM orders
            WHERE order_id = %s
            """,
            (order_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def most_recent_order_for_customer(customer_id: str) -> dict | None:
    """Fallback lookup when the customer does not provide an explicit order ID."""
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT order_id::text, customer_id::text, status, total_amount,
                   placed_at, delivered_at
            FROM orders
            WHERE customer_id = %s
            ORDER BY placed_at DESC
            LIMIT 1
            """,
            (customer_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def initiate_refund(order_id: str, amount: float, reason: str = "") -> dict[str, Any]:
    """
    Enforces Policy (Refund Policy):
    - Refund amount <= $75.00
    - Refund amount <= 80% of total order value
    Auto-approves if both conditions met; escalates to human approval otherwise.
    """
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT total_amount FROM orders WHERE order_id = %s", (order_id,))
        row = cur.fetchone()
        if row is None:
            return {
                "tool_name": "initiate_refund",
                "success": False,
                "output": f"No order found with id {order_id}",
                "requires_escalation": False,
            }

        total_amount = float(row["total_amount"])
        within_amount_limit = amount <= AUTO_APPROVE_MAX_AMOUNT
        within_ratio_limit = amount <= total_amount * AUTO_APPROVE_MAX_RATIO
        auto_approved = within_amount_limit and within_ratio_limit

        status = "processed" if auto_approved else "requested"
        processed_at = datetime.now(timezone.utc) if auto_approved else None

        cur.execute(
            """
            INSERT INTO refunds (order_id, amount, status, reason, processed_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING refund_id
            """,
            (order_id, amount, status, reason, processed_at),
        )
        refund_id = cur.fetchone()["refund_id"]
        conn.commit()

        if auto_approved:
            return {
                "tool_name": "initiate_refund",
                "success": True,
                "output": f"Refund of ${amount:.2f} approved and processed (refund_id={refund_id})",
                "requires_escalation": False,
            }
        else:
            reasons = []
            if not within_amount_limit:
                reasons.append(f"amount ${amount:.2f} exceeds ${AUTO_APPROVE_MAX_AMOUNT:.2f} threshold")
            if not within_ratio_limit:
                reasons.append(f"amount is more than {AUTO_APPROVE_MAX_RATIO:.0%} of order total (${total_amount:.2f})")

            return {
                "tool_name": "initiate_refund",
                "success": True,
                "output": f"Refund of ${amount:.2f} recorded as pending human approval (${total_amount:.2f} total)",
                "requires_escalation": True,
                "escalation_reason": "; ".join(reasons),
            }
    finally:
        conn.close()


def change_shipping_address(order_id: str, new_address: str) -> dict[str, Any]:
    """
    Enforces Policy (Shipping Policy - Address Changes):
    Address changes are allowed ONLY if order status is 'pending'.
    If order is already 'shipped' or 'delivered', policy prohibits modification.
    """
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT status FROM orders WHERE order_id = %s", (order_id,))
        row = cur.fetchone()
        if row is None:
            return {
                "tool_name": "change_shipping_address",
                "success": False,
                "output": f"No order found with id {order_id}.",
                "requires_escalation": False,
            }

        status = row["status"]
        if status == "pending":
            return {
                "tool_name": "change_shipping_address",
                "success": True,
                "output": f"Shipping address for order {order_id} has been updated to: '{new_address}'.",
                "requires_escalation": False,
            }
        else:
            return {
                "tool_name": "change_shipping_address",
                "success": False,
                "output": (
                    f"Order {order_id} is currently '{status}'. Under company policy, "
                    "shipping addresses can only be modified while the order is 'pending'. "
                    "Because it has already processed/shipped, please contact the carrier directly "
                    "or refuse delivery so the package returns to sender."
                ),
                "requires_escalation": False,
            }
    finally:
        conn.close()


def report_lost_package(order_id: str, notes: str = "") -> dict[str, Any]:
    """
    Enforces Policy (Escalation Rules & Shipping Policy - Lost Packages):
    All lost package claims must be escalated to a human agent for carrier investigation
    rather than automatically refunded.
    """
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT status FROM orders WHERE order_id = %s", (order_id,))
        row = cur.fetchone()
        if row is None:
            return {
                "tool_name": "report_lost_package",
                "success": False,
                "output": f"No order found with id {order_id}.",
                "requires_escalation": False,
            }

        return {
            "tool_name": "report_lost_package",
            "success": True,
            "output": f"Lost package claim recorded for order {order_id}. Escalated for carrier investigation.",
            "requires_escalation": True,
            "escalation_reason": "Lost package claim requires carrier investigation per shipping policy.",
        }
    finally:
        conn.close()


def report_damaged_item(
    order_id: str,
    photo_provided: bool = False,
    image_source: Optional[str] = None,
    notes: str = "",
) -> dict[str, Any]:
    """
    Enforces Policy (Refund Policy - Damaged Items):
    If an item arrives damaged, photographic evidence is required.
    Supports automated vision verification via inspect_damage_photo.
    Reports within 14 days of delivery qualify for replacement or refund.
    Reports beyond 14 days require human supervisor escalation.
    """
    # 1. Automated Vision Inspection if image source is provided
    if image_source:
        try:
            from src.agent.tools.vision_tools import inspect_damage_photo
            vision_result = inspect_damage_photo(image_source)
            if vision_result.get("verified", False):
                photo_provided = True
                notes = f"Vision inspection confirmed damage: {vision_result.get('damage_description')}"
            else:
                return {
                    "tool_name": "report_damaged_item",
                    "success": False,
                    "output": (
                        f"Photo was reviewed, but visible damage could not be clearly confirmed "
                        f"({vision_result.get('damage_description')}). Flagged for human supervisor review."
                    ),
                    "requires_escalation": True,
                    "escalation_reason": "Visual damage verification unconfirmed; requires human inspection.",
                }
        except Exception as e:
            photo_provided = False

    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT status, delivered_at FROM orders WHERE order_id = %s", (order_id,))
        row = cur.fetchone()
        if row is None:
            return {
                "tool_name": "report_damaged_item",
                "success": False,
                "output": f"No order found with id {order_id}.",
                "requires_escalation": False,
            }

        delivered_at = row["delivered_at"]
        if not photo_provided:
            return {
                "tool_name": "report_damaged_item",
                "success": False,
                "output": (
                    "Please provide photographic evidence of the damaged or defective item. "
                    "Once photos are submitted, we will immediately process your replacement or refund."
                ),
                "requires_escalation": False,
            }

        is_within_14_days = True
        if delivered_at:
            delta = datetime.now(timezone.utc) - delivered_at
            if delta.days > 14:
                is_within_14_days = False

        if is_within_14_days:
            output_msg = f"Damaged item claim verified for order {order_id}. Full replacement or refund authorized."
            if notes:
                output_msg += f" ({notes})"
            return {
                "tool_name": "report_damaged_item",
                "success": True,
                "output": output_msg,
                "requires_escalation": False,
            }
        else:
            return {
                "tool_name": "report_damaged_item",
                "success": True,
                "output": f"Damaged item for order {order_id} reported after 14-day window. Escalated for supervisor review.",
                "requires_escalation": True,
                "escalation_reason": "Damaged item reported > 14 days after delivery requires human review.",
            }
    finally:
        conn.close()
