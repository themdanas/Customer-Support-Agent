from __future__ import annotations
import argparse
import random
from datetime import datetime, timedelta


import psycopg2
from psycopg2.extras import execute_values
from faker import Faker
import os


fake = Faker()
 
ORDER_STATUS_WEIGHTS = {
    "pending": 0.10,
    "shipped": 0.15,
    "delivered": 0.55,
    "returned": 0.10,
    "refunded": 0.10,
}
 
TICKET_SUBJECTS = [
    "Where is my order?",
    "Item arrived damaged",
    "Wrong item received",
    "Requesting a refund",
    "Order never arrived",
    "Need to change shipping address",
    "Question about return policy",
]
 
PRODUCT_CATALOG = [
    ("Wireless Mouse", 19.99),
    ("Mechanical Keyboard", 79.99),
    ("USB-C Hub", 34.50),
    ("Laptop Stand", 45.00),
    ("Noise Cancelling Headphones", 129.99),
    ("Webcam 1080p", 39.99),
    ("Desk Lamp", 24.99),
    ("Phone Case", 14.99),
    ("Portable SSD 1TB", 89.99),
    ("Bluetooth Speaker", 54.99),
]

def weighted_status() -> str:
    statuses, weights = zip(*ORDER_STATUS_WEIGHTS.items())
    return random.choices(statuses, weights=weights, k=1)[0]
 
 
def build_customers(n: int) -> list[tuple]:
    return [(fake.name(), fake.unique.email()) for _ in range(n)]
 
 
def build_order(customer_id: str) -> tuple[dict, list[dict]]:
    status = weighted_status()
    placed_at = fake.date_time_between(start_date="-120d", end_date="-1d")
    delivered_at = None
    if status in ("delivered", "returned", "refunded"):
        delivered_at = placed_at + timedelta(days=random.randint(2, 10))
 
    n_items = random.randint(1, 4)
    items = []
    total = 0.0
    for _ in range(n_items):
        name, price = random.choice(PRODUCT_CATALOG)
        qty = random.randint(1, 3)
        items.append({"product_name": name, "quantity": qty, "unit_price": price})
        total += price * qty
 
    order = {
        "customer_id": customer_id,
        "status": status,
        "total_amount": round(total, 2),
        "placed_at": placed_at,
        "delivered_at": delivered_at,
    }
    return order, items
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
 
    random.seed(args.seed)
    Faker.seed(args.seed)
 
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise SystemExit("Set DATABASE_URL, e.g. postgresql://postgres:postgres@localhost:5432/support_agent")
 
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
 
    # --- customers ---
    # NOTE: execute_values batches inserts in pages (default page_size=100).
    # Without fetch=True, cur.fetchall() only returns the LAST batch's rows,
    # silently dropping ids from earlier batches once customers > 100.
    customer_rows = build_customers(args.customers)
    returned = execute_values(
        cur,
        "INSERT INTO customers (full_name, email) VALUES %s RETURNING customer_id",
        customer_rows,
        fetch=True,
    )
    customer_ids = [row[0] for row in returned]
 
    # --- orders + order_items ---
    order_ids_by_customer: dict[str, list[str]] = {}
    for customer_id in customer_ids:
        n_orders = random.randint(0, 5)
        order_ids_by_customer[customer_id] = []
        for _ in range(n_orders):
            order, items = build_order(customer_id)
            cur.execute(
                """
                INSERT INTO orders (customer_id, status, total_amount, placed_at, delivered_at)
                VALUES (%(customer_id)s, %(status)s, %(total_amount)s, %(placed_at)s, %(delivered_at)s)
                RETURNING order_id
                """,
                order,
            )
            order_id = cur.fetchone()[0]
            order_ids_by_customer[customer_id].append(order_id)
 
            item_rows = [(order_id, i["product_name"], i["quantity"], i["unit_price"]) for i in items]
            execute_values(
                cur,
                "INSERT INTO order_items (order_id, product_name, quantity, unit_price) VALUES %s",
                item_rows,
            )
 
    # --- support_tickets (only ~30% of orders generate one) ---
    ticket_ids_by_order: dict[str, str] = {}
    for customer_id, order_ids in order_ids_by_customer.items():
        for order_id in order_ids:
            if random.random() < 0.3:
                cur.execute(
                    """
                    INSERT INTO support_tickets (customer_id, order_id, subject, status, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING ticket_id
                    """,
                    (
                        customer_id,
                        order_id,
                        random.choice(TICKET_SUBJECTS),
                        random.choice(["open", "pending_customer", "pending_agent", "resolved"]),
                        fake.date_time_between(start_date="-60d", end_date="now"),
                    ),
                )
                ticket_ids_by_order[order_id] = cur.fetchone()[0]
 
    # --- refunds (only ~40% of tickets escalate to a refund request) ---
    refund_count = 0
    escalation_count = 0
    cur.execute("SELECT order_id, total_amount FROM orders")
    order_totals = dict(cur.fetchall())
 
    for order_id, ticket_id in ticket_ids_by_order.items():
        if random.random() < 0.4:
            order_total = float(order_totals[order_id])
            refund_amount = round(order_total * random.uniform(0.2, 1.0), 2)
            cur.execute(
                """
                INSERT INTO refunds (order_id, ticket_id, amount, status, reason, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING refund_id
                """,
                (
                    order_id,
                    ticket_id,
                    refund_amount,
                    random.choice(["requested", "approved", "processed"]),
                    random.choice(["damaged item", "wrong item", "never arrived", "changed mind"]),
                    fake.date_time_between(start_date="-30d", end_date="now"),
                ),
            )
            refund_count += 1
 
            # refunds over $75 or over 80% of order value simulate the kind
            # of thing that should have been escalated for human approval
            if refund_amount > 75 or refund_amount > order_total * 0.8:
                cur.execute(
                    """
                    INSERT INTO escalations (ticket_id, reason, priority, source_node)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        ticket_id,
                        "Refund amount exceeds auto-approval threshold",
                        random.choice(["medium", "high"]),
                        "action",
                    ),
                )
                escalation_count += 1
 
    conn.commit()
    cur.close()
    conn.close()
 
    print(f"Inserted {len(customer_ids)} customers")
    print(f"Inserted {sum(len(v) for v in order_ids_by_customer.values())} orders")
    print(f"Inserted {len(ticket_ids_by_order)} tickets")
    print(f"Inserted {refund_count} refunds")
    print(f"Inserted {escalation_count} escalations")
 
 
if __name__ == "__main__":
    main()
