from __future__ import annotations
import os
import re
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone

AUTO_APPROVE_MAX_AMOUNT = 75.00
AUTO_APPROVE_MAX_RATIO = 0.80

 
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

def extract_order_id(message: str) -> str | None:
    match = UUID_RE.search(message)
    return match.group(0) if match else None

def check_order_status(order_id: str) -> dict | None:
    #Return thr order record as a plain dict
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
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

def initiate_refund(order_id: str, amount: float, reason: str = "") -> dict:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT total_amount FROM orders WHERE order_id = %s",(order_id))
        row = cur.fetchone()
        if row is None:
            return{
                "tool_name":"intiate_refund",
                "success": False,
                "output": f"No order found with id {order_id}",
                "requires_escalation": False,
            }

        total_amount = float(row["total_amount"])
        withing_amount_limit = amount <= AUTO_APPROVE_MAX_AMOUNT
        within_ratio_limit = amount <= total_amount * AUTO_APPROVE_MAX_RATIO
        auto_approved = withing_amount_limit and within_ratio_limit

        status = "processed" if auto_approved else "requested"
        processed_at = datetime.now(timezone.utc) if auto_approved else None

        cur.execute(
            """
            INSERT INTO refunds (order_id, amount, status, reason, processed_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING refund_id
            """
            (order_id, amount, status, reason, processed_at)
        )
        refund_id = cur.fetchone()["refund_id"]
        conn.commit()

        if auto_approved:
            return{
                "tool_name": "intiate_refund",
                "success": True,
                "output": f"Refund of ${amount:.2f} approved and processed (refund_id)={refund_id}",
                "requires_escalation": False,
            }
        else:
            reason = []
            if not withing_amount_limit:
                reason = []
                if not withing_amount_limit:
                    reason.append(f"amount ${amount:.2f} exceeds ${AUTO_APPROVE_MAX_AMOUNT:.2f} threshold")
                if not within_ratio_limit:
                    reason.append(f"amount is more than {AUTO_APPROVE_MAX_RATIO:.0%} of order total (${total_amount:.2f})")

                return {
                    "tool_name": "initiate_refund",
                    "success": True,
                    "output": f"refund of {amount:.2f} recorded as pending human approval (${total_amount:.2f})",
                    "requires_escalation": True,
                    "escalation_reason": ";".join(reason),
                }
    finally:
        conn.close()





def most_recent_order_for_customer(customer_id: str) -> dict | None:
    #Fall back when the customer doesn't give an order ID direclty
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
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