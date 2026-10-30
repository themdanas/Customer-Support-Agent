from __future__ import annotations
import os
from typing import Optional, Any
import psycopg2
import psycopg2.extras

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/support_agent"


def _get_db_conn(database_url: Optional[str] = None):
    url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL)
    return psycopg2.connect(url)


def get_customer_profile(customer_id: str) -> dict[str, Any] | None:
    """
    Fetches customer account details and lifetime statistics.
    Useful for customer verification and VIP/tier-based reasoning.
    """
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT 
                c.customer_id::text,
                c.full_name,
                c.email,
                c.created_at,
                COUNT(o.order_id) AS total_orders,
                COALESCE(SUM(o.total_amount), 0.0) AS lifetime_spend
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            WHERE c.customer_id = %s
            GROUP BY c.customer_id, c.full_name, c.email, c.created_at
            """,
            (customer_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        res = dict(row)
        res["lifetime_spend"] = float(res["lifetime_spend"])
        return res
    finally:
        conn.close()


def get_customer_orders(customer_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Retrieves the customer's most recent orders sorted by placement date.
    Helps the agent disambiguate queries when the customer doesn't specify an order ID.
    """
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT 
                order_id::text,
                customer_id::text,
                status,
                total_amount,
                placed_at,
                delivered_at
            FROM orders
            WHERE customer_id = %s
            ORDER BY placed_at DESC
            LIMIT %s
            """,
            (customer_id, limit),
        )
        rows = cur.fetchall()
        orders = []
        for r in rows:
            d = dict(r)
            d["total_amount"] = float(d["total_amount"])
            orders.append(d)
        return orders
    finally:
        conn.close()


def get_ticket_count_for_order(order_id: str) -> int:
    """
    Counts existing support tickets logged against a specific order.
    Directly feeds into repeat-contact escalation detection.
    """
    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM support_tickets WHERE order_id = %s",
            (order_id,),
        )
        row = cur.fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def check_repeat_contact_escalation(order_id: str) -> dict[str, Any]:
    """
    Enforces Policy (Escalation Rules - Repeat Contact):
    If a customer has contacted support more than twice about the same order without resolution,
    escalate to a human agent regardless of auto-approval thresholds.
    """
    ticket_count = get_ticket_count_for_order(order_id)
    # If ticket_count >= 2, this conversation represents the 3rd or subsequent touchpoint
    requires_escalation = ticket_count >= 2

    return {
        "tool_name": "check_repeat_contact_escalation",
        "order_id": order_id,
        "ticket_count": ticket_count,
        "requires_escalation": requires_escalation,
        "escalation_reason": (
            f"Customer has contacted support {ticket_count} times regarding order {order_id}. "
            "Policy requires human escalation after more than 2 unresolved contacts."
            if requires_escalation
            else None
        ),
    }


def create_support_ticket(
    customer_id: str,
    subject: str,
    order_id: Optional[str] = None,
    status: str = "open",
) -> dict[str, Any]:
    """
    Logs a support ticket in the database to maintain audit trail and contact history.
    """
    conn = _get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            INSERT INTO support_tickets (customer_id, order_id, subject, status)
            VALUES (%s, %s, %s, %s::ticket_status)
            RETURNING ticket_id, customer_id::text, order_id::text, subject, status, created_at
            """,
            (customer_id, order_id, subject, status),
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row)
    finally:
        conn.close()

