from __future__ import annotations
import argparse
from src.agent.tools.policy_rag.vector_store import PolicyVectorStore

#Each entry: (query, expected_policy_id)
EVAL_SET = [
    ("Can the customer get a refund for an item bought 40 days ago?", "refund_policy::chunk_1"),
    ("A customer wants an $80 refund, does this need human approval?", "refund_policy::chunk_2"),
    ("The item that arrived is broken, what should I offer the customer?", "refund_policy::chunk_3"),
    ("Customer says they got the wrong product entirely", "refund_policy::chunk_4"),
    ("Can we refund a gift card purchase?", "refund_policy::chunk_5"),
    ("How long until an approved refund shows up on the customer's card?", "refund_policy::chunk_6"),
    ("How many days does normal delivery usually take?", "shipping_policy::chunk_1"),
    ("Order hasn't moved in tracking for 4 days, what do we do?", "shipping_policy::chunk_2"),
    ("Customer wants to change their delivery address, order already shipped", "shipping_policy::chunk_3"),
    ("Tracking hasn't updated in 12 days and the package is way overdue", "shipping_policy::chunk_4"),
    ("Does customs delay on an international order count for a refund?", "shipping_policy::chunk_5"),
    ("When does a refund amount need to go to a human instead of auto-approving?", "escalation_rules::chunk_1"),
    ("Customer has messaged support three times about the same order", "escalation_rules::chunk_2"),
    ("The customer is furious and threatening to leave a bad review", "escalation_rules::chunk_3"),
    ("The situation doesn't match anything in our policy docs", "escalation_rules::chunk_4"),
    ("A package is confirmed lost, can we just refund it automatically?", "escalation_rules::chunk_5"),
    ("Customer insists their final-sale item should still be refundable", "escalation_rules::chunk_6"),
]

def recall_at_k(store: PolicyVectorStore, k:int) -> tuple[float, list]:
    hits = 0
    misses = []
    for query, expected_id in EVAL_SET:
        results = store.query(query, k=k)
        retrieved_ids = [r.policy_id for r in results]
        if expected_id in retrieved_ids:
            hits += 1
        else:
            misses.append((query, expected_id, retrieved_ids))
    return hits/ len(EVAL_SET), misses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="policy_index.pkl")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    store = PolicyVectorStore.load(args.index)
    recall, misses = recall_at_k(store, args.k)

    print(f"recall@{args.k}: {recall:.2%} ({len(EVAL_SET) - len(misses)}/{len(EVAL_SET)})")

    if misses:
        print("\nMisses:")
        for query, expected_id, retrieved_ids in misses:
            print(f"  Q: {query}")
            print(f"     expected: {expected_id}")
            print(f"     got:      {retrieved_ids}")

if __name__ == "__main__":
    main()

