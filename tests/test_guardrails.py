from __future__ import annotations
import os
import uuid
import pytest
import psycopg2
import psycopg2.extras

from src.agent.tools.order_tools import initiate_refund, AUTO_APPROVE_MAX_AMOUNT, AUTO_APPROVE_MAX_RATIO

@pytest.fixture
def db_conn():
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/support_agent")
    conn = psycopg2.connect(db_url)
    yield conn
    conn.close()

def make_order(conn, total_amount: float) -> str:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "INSERT INTO customers (full_name, email) VALUES (%s, %s) RETURNING customer_id",
        (f"Test Customer {uuid.uuid4()}", f"test-{uuid.uuid4()}@example.com"),
    )
    customer_id = cur.fetchone()["customer_id"]
    cur.execute(
        "INSERT INTO orders (customer_id, status, total_amount) VALUES (%s, 'delivered', %s) RETURNING order_id",
        (customer_id, total_amount),
    )
    order_id = cur.fetchone()["order_id"]
    conn.commit()
    return str(order_id)

def test_small_refund_auto_approves(db_conn):
    """$20 refund on a $500 order: under both the $75 cap and the 80% ratio."""
    order_id = make_order(db_conn, total_amount=500.00)
    result = initiate_refund(order_id=order_id, amount=20.00, reason="test")
    assert result["success"] is True
    assert result["requires_escalation"] is False
    assert "processed" in result["output"]

def test_refund_over_dollar_cap_escalates(db_conn):
    """$100 refun on a $500 order: under 80% ration, but over the 75% cap."""
    order_id = make_order(db_conn, total_amount=500.00)
    result = initiate_refund(order_id=order_id, amount=100.00, reason="test")
    assert result["success"] is True
    assert "threshold" in result["escalation_reason"]

def test_refund_over_ratio_escalates(db_conn):
    """$100 refund on a $500 order: under 80% ratio, but over the $75 cap."""
    order_id = make_order(db_conn, total_amount=500.00)
    result = initiate_refund(order_id=order_id, amount=100.00, reason="test")
    assert result["requires_escalation"] is True
    assert "threshold" in result["escalation_reason"]


def test_refund_over_ratio_escalates_even_under_dollar_cap(db_conn):
    """
    The important edge case: a refund can be UNDER the $75 cap and still
    require escalation if it's too large a fraction of a small order.
    $45 refund on a $50 order = 90% of order value, over the 80% ratio,
    even though $45 alone is well under the $75 cap.
    """
    order_id = make_order(db_conn, total_amount=50.00)
    result = initiate_refund(order_id=order_id, amount=45.00, reason="test")
    assert result["requires_escalation"] is True
    assert "80%" in result["escalation_reason"] or "ratio" not in result["escalation_reason"]
    assert "more than" in result["escalation_reason"]

def test_refund_over_ratio_escalates_even_under_dollar_cap(db_conn):
    """
    The important edge case: a refund can be UNDER the $75 cap and still
    require escalation if it's too large a fraction of a small order.
    $45 refund on a $50 order = 90% of order value, over the 80% ratio,
    even though $45 alone is well under the $75 cap.
    """
    order_id = make_order(db_conn, total_amount=50.00)
    result = initiate_refund(order_id=order_id, amount=45.00, reason="test")
    assert result["requires_escalation"] is True
    assert "80%" in result["escalation_reason"] or "ratio" not in result["escalation_reason"]
    assert "more than" in result["escalation_reason"]

def test_ratio_boundary_exactly_at_limit_auto_approves(db_conn):
    """Exactly 80% of order total should auto-approve (limit is inclusive: <=)."""
    order_id = make_order(db_conn, total_amount=50.00)
    boundary_amount = round(50.00 * AUTO_APPROVE_MAX_RATIO, 2)  # 40.00
    result = initiate_refund(order_id=order_id, amount=boundary_amount, reason="test")
    assert result["requires_escalation"] is False

def test_ratio_boundary_just_over_limit_escalates(db_conn):
    """One cent over 80% of order total should escalate."""
    order_id = make_order(db_conn, total_amount=50.00)
    over_boundary = round(50.00 * AUTO_APPROVE_MAX_RATIO, 2) + 0.01  # 40.01
    result = initiate_refund(order_id=order_id, amount=over_boundary, reason="test")
    assert result["requires_escalation"] is True
 
 
def test_dollar_cap_boundary_exactly_at_limit_auto_approves(db_conn):
    """Exactly $75 on a large order (ratio not a factor) should auto-approve."""
    order_id = make_order(db_conn, total_amount=1000.00)
    result = initiate_refund(order_id=order_id, amount=AUTO_APPROVE_MAX_AMOUNT, reason="test")
    assert result["requires_escalation"] is False
 
 
def test_nonexistent_order_fails_cleanly(db_conn):
    fake_order_id = str(uuid.uuid4())
    result = initiate_refund(order_id=fake_order_id, amount=10.00, reason="test")
    assert result["success"] is False
    assert result["requires_escalation"] is False
    assert "No order found" in result["output"]
 
 
def test_auto_approved_refund_is_persisted_as_processed(db_conn):
    order_id = make_order(db_conn, total_amount=500.00)
    initiate_refund(order_id=order_id, amount=20.00, reason="test")
    cur = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT status, processed_at FROM refunds WHERE order_id = %s", (order_id,))
    row = cur.fetchone()
    assert row["status"] == "processed"
    assert row["processed_at"] is not None
 
 
def test_escalated_refund_is_persisted_as_requested_not_processed(db_conn):
    order_id = make_order(db_conn, total_amount=500.00)
    initiate_refund(order_id=order_id, amount=200.00, reason="test")
    cur = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT status, processed_at FROM refunds WHERE order_id = %s", (order_id,))
    row = cur.fetchone()
    assert row["status"] == "requested"
    assert row["processed_at"] is None
 