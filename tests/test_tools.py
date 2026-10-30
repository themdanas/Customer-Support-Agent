from __future__ import annotations
import os
import uuid
from datetime import datetime, timezone, timedelta
import pytest
import psycopg2
import psycopg2.extras

from src.agent.tools.customer_tools import (
    get_customer_profile,
    get_customer_orders,
    get_ticket_count_for_order,
    check_repeat_contact_escalation,
    create_support_ticket,
)
from src.agent.tools.order_tools import (
    change_shipping_address,
    report_lost_package,
    report_damaged_item,
)


@pytest.fixture
def db_conn():
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_agent"
    )
    conn = psycopg2.connect(db_url)
    yield conn
    conn.close()


def create_customer(conn) -> str:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "INSERT INTO customers (full_name, email) VALUES (%s, %s) RETURNING customer_id",
        (f"Tool User {uuid.uuid4()}", f"tool-{uuid.uuid4()}@example.com"),
    )
    cust_id = str(cur.fetchone()["customer_id"])
    conn.commit()
    return cust_id


def create_order(
    conn,
    customer_id: str,
    status: str = "pending",
    total_amount: float = 100.0,
    delivered_at: datetime | None = None,
) -> str:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    placed_at = (
        delivered_at - timedelta(days=3)
        if delivered_at
        else datetime.now(timezone.utc)
    )
    cur.execute(
        """
        INSERT INTO orders (customer_id, status, total_amount, placed_at, delivered_at)
        VALUES (%s, %s::order_status, %s, %s, %s)
        RETURNING order_id
        """,
        (customer_id, status, total_amount, placed_at, delivered_at),
    )
    order_id = str(cur.fetchone()["order_id"])
    conn.commit()
    return order_id


# =========================================================================
# Customer Tools Tests
# =========================================================================

def test_get_customer_profile(db_conn):
    cust_id = create_customer(db_conn)
    create_order(db_conn, cust_id, total_amount=150.0)
    create_order(db_conn, cust_id, total_amount=50.0)

    profile = get_customer_profile(cust_id)
    assert profile is not None
    assert profile["customer_id"] == cust_id
    assert profile["total_orders"] == 2
    assert profile["lifetime_spend"] == 200.0


def test_get_customer_orders_limit(db_conn):
    cust_id = create_customer(db_conn)
    for i in range(4):
        create_order(db_conn, cust_id, total_amount=10.0 * (i + 1))

    orders = get_customer_orders(cust_id, limit=2)
    assert len(orders) == 2
    assert orders[0]["customer_id"] == cust_id


def test_repeat_contact_escalation_triggers_on_third_contact(db_conn):
    cust_id = create_customer(db_conn)
    order_id = create_order(db_conn, cust_id)

    # 0 prior tickets: No escalation
    res0 = check_repeat_contact_escalation(order_id)
    assert res0["requires_escalation"] is False
    assert res0["ticket_count"] == 0

    # 1st ticket logged: Still under policy limit (<= 2 contacts)
    create_support_ticket(cust_id, subject="Where is my order?", order_id=order_id)
    res1 = check_repeat_contact_escalation(order_id)
    assert res1["requires_escalation"] is False
    assert res1["ticket_count"] == 1

    # 2nd ticket logged: At threshold -> triggers escalation on next contact
    create_support_ticket(cust_id, subject="Still waiting for order", order_id=order_id)
    res2 = check_repeat_contact_escalation(order_id)
    assert res2["requires_escalation"] is True
    assert res2["ticket_count"] == 2
    assert "human escalation" in res2["escalation_reason"]


# =========================================================================
# Order Tools Policy Tests
# =========================================================================

def test_change_shipping_address_allowed_when_pending(db_conn):
    cust_id = create_customer(db_conn)
    order_id = create_order(db_conn, cust_id, status="pending")

    result = change_shipping_address(order_id, new_address="456 Elm St, New York, NY")
    assert result["success"] is True
    assert result["requires_escalation"] is False
    assert "updated" in result["output"]


def test_change_shipping_address_rejected_when_shipped(db_conn):
    cust_id = create_customer(db_conn)
    order_id = create_order(db_conn, cust_id, status="shipped")

    result = change_shipping_address(order_id, new_address="789 Pine St, Boston, MA")
    assert result["success"] is False
    assert result["requires_escalation"] is False
    assert "shipped" in result["output"]
    assert "refuse delivery" in result["output"]


def test_report_lost_package_always_escalates_to_human(db_conn):
    cust_id = create_customer(db_conn)
    order_id = create_order(db_conn, cust_id, status="shipped")

    result = report_lost_package(order_id)
    assert result["success"] is True
    assert result["requires_escalation"] is True
    assert "carrier investigation" in result["escalation_reason"]


def test_report_damaged_item_requires_photo(db_conn):
    cust_id = create_customer(db_conn)
    order_id = create_order(db_conn, cust_id, status="delivered")

    result = report_damaged_item(order_id, photo_provided=False)
    assert result["success"] is False
    assert result["requires_escalation"] is False
    assert "photographic evidence" in result["output"]


def test_report_damaged_item_with_photo_within_14_days(db_conn):
    cust_id = create_customer(db_conn)
    recent_delivery = datetime.now(timezone.utc) - timedelta(days=5)
    order_id = create_order(db_conn, cust_id, status="delivered", delivered_at=recent_delivery)

    result = report_damaged_item(order_id, photo_provided=True)
    assert result["success"] is True
    assert result["requires_escalation"] is False
    assert "authorized" in result["output"]


def test_report_damaged_item_with_photo_after_14_days_escalates(db_conn):
    cust_id = create_customer(db_conn)
    old_delivery = datetime.now(timezone.utc) - timedelta(days=20)
    order_id = create_order(db_conn, cust_id, status="delivered", delivered_at=old_delivery)

    result = report_damaged_item(order_id, photo_provided=True)
    assert result["success"] is True
    assert result["requires_escalation"] is True
    assert "> 14 days" in result["escalation_reason"]
