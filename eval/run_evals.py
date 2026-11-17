#!/usr/bin/env python3
"""
Automated Multi-Turn Evaluation & Benchmarking Harness for the AI Customer Support Agent.

Evaluates:
  1. Multi-Turn Context Retention (Order ID & Image persistence across turns, no re-asking)
  2. Intent Classification & Tool Call Accuracy (tool_name + backfilled tool_args)
  3. Deterministic Policy & Financial Guardrails ($75 refund cap, shipped address change block)
  4. Turn & Node Latency Metrics (Mean, P50, P95, Max in ms)

Usage:
  python src/evals/run_evals.py                # Runs against live GroqLLMClient (if GROQ_API_KEY set)
  python src/evals/run_evals.py --mock         # Runs against deterministic MockLLMClient
  python src/evals/run_evals.py --delay 1.5    # Adds 1.5s delay between turns for Groq rate limits
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.checkpoint.memory import MemorySaver
from src.agent.graph import build_graph
from src.agent.llm_client import get_llm_client, MockLLMClient
from src.agent.persistence import get_thread_config
from src.agent.state import (
    ActionResult,
    ActionType,
    ConversationStatus,
    OrderRecord,
    OrderStatus,
)

# =====================================================================
# 1. DETERMINISTIC GOLDEN FIXTURES (Sandbox Order Store)
# =====================================================================

NOW = datetime.now(timezone.utc)

GOLDEN_ORDERS: dict[str, OrderRecord] = {
    # Delivered 3 days ago, $45.00 (Eligible for auto-refund <= $75 and <= 80% / damage claim)
    "11111111-1111-1111-1111-111111111111": OrderRecord(
        order_id="11111111-1111-1111-1111-111111111111",
        customer_id="cust-eval-01",
        status=OrderStatus.DELIVERED,
        total_amount=45.00,
        placed_at=NOW - timedelta(days=6),
        delivered_at=NOW - timedelta(days=3),
        items=["Wireless Mechanical Keyboard"],
    ),
    # Delivered 2 days ago, $200.00 (Exceeds $75 guardrail -> Must trigger PENDING_APPROVAL)
    "22222222-2222-2222-2222-222222222222": OrderRecord(
        order_id="22222222-2222-2222-2222-222222222222",
        customer_id="cust-eval-02",
        status=OrderStatus.DELIVERED,
        total_amount=200.00,
        placed_at=NOW - timedelta(days=5),
        delivered_at=NOW - timedelta(days=2),
        items=["27-inch 4K IPS Monitor"],
    ),
    # Pending order, $60.00 (Eligible for shipping address change)
    "33333333-3333-3333-3333-333333333333": OrderRecord(
        order_id="33333333-3333-3333-3333-333333333333",
        customer_id="cust-eval-03",
        status=OrderStatus.PENDING,
        total_amount=60.00,
        placed_at=NOW - timedelta(days=1),
        delivered_at=None,
        items=["USB-C Docking Station"],
    ),
    # Shipped order, $85.00 (Address change prohibited by guardrail)
    "44444444-4444-4444-4444-444444444444": OrderRecord(
        order_id="44444444-4444-4444-4444-444444444444",
        customer_id="cust-eval-04",
        status=OrderStatus.SHIPPED,
        total_amount=85.00,
        placed_at=NOW - timedelta(days=3),
        delivered_at=None,
        items=["Noise-Cancelling Headphones"],
    ),
}


# =====================================================================
# 2. MULTI-TURN GOLDEN EVALUATION SUITE
# =====================================================================

EVAL_SCENARIOS = [
    {
        "id": "EVAL-01",
        "name": "Multi-Turn Order Lookup -> Follow-Up Refund (No Order ID in Turn 2)",
        "customer_id": "cust-eval-01",
        "turns": [
            {
                "user": "Hi, can you check the status of my order 11111111-1111-1111-1111-111111111111?",
                "expect_order_id": "11111111-1111-1111-1111-111111111111",
                "expect_intent": "order_status",
                "expect_status": ConversationStatus.RESOLVED,
            },
            {
                "user": "Actually, I don't need it anymore. Please give me a refund of $30 for it.",
                "expect_order_id": "11111111-1111-1111-1111-111111111111",
                "expect_intent": "refund_request",
                "expect_tool": "initiate_refund",
                "forbid_reasking_order_id": True,
                "expect_status": ConversationStatus.RESOLVED,
            },
        ],
    },
    {
        "id": "EVAL-02",
        "name": "Photo Attached First -> Bare 'order_id <uuid>' Fragment in Turn 2 (JSON Crash Regression Test)",
        "customer_id": "cust-eval-01",
        "turns": [
            {
                "user": "I got a broken keyboard, the screen on it is cracked and damaged!",
                "image_source": "/tmp/cracked_keyboard.png",
                "expect_image_source": "/tmp/cracked_keyboard.png",
            },
            {
                "user": "order_id 11111111-1111-1111-1111-111111111111",
                "expect_order_id": "11111111-1111-1111-1111-111111111111",
                "expect_image_source": "/tmp/cracked_keyboard.png",
                "expect_tool": "report_damaged_item",
                "forbid_reasking_order_id": True,
            },
        ],
    },
    {
        "id": "EVAL-03",
        "name": "Order Provided in Turn 1 -> Photo Attached in Turn 2 (Cross-Turn Vision Claim)",
        "customer_id": "cust-eval-01",
        "turns": [
            {
                "user": "My order 11111111-1111-1111-1111-111111111111 arrived completely shattered.",
                "expect_order_id": "11111111-1111-1111-1111-111111111111",
                # In Turn 1 no photo is attached yet, so agent should answer & ask for a photo
                "expect_action_type": ActionType.ANSWER,
            },
            {
                "user": "Here is the photo of the broken item.",
                "image_source": "/tmp/broken_item.jpg",
                "expect_order_id": "11111111-1111-1111-1111-111111111111",
                "expect_image_source": "/tmp/broken_item.jpg",
                "expect_tool": "report_damaged_item",
                "forbid_reasking_order_id": True,
            },
        ],
    },
    {
        "id": "EVAL-04",
        "name": "High-Value Refund Guardrail ($150 > $75 Threshold -> Pauses for Human Approval)",
        "customer_id": "cust-eval-02",
        "turns": [
            {
                "user": "Check my order 22222222-2222-2222-2222-222222222222 please.",
                "expect_order_id": "22222222-2222-2222-2222-222222222222",
            },
            {
                "user": "I want a refund of $150.00 for this order.",
                "expect_order_id": "22222222-2222-2222-2222-222222222222",
                "expect_tool": "initiate_refund",
                "expect_status": ConversationStatus.PENDING_APPROVAL,
                "forbid_reasking_order_id": True,
            },
        ],
    },
    {
        "id": "EVAL-05",
        "name": "Address Change on Pending Order -> Multi-Turn Context Preservation",
        "customer_id": "cust-eval-03",
        "turns": [
            {
                "user": "Where is my order 33333333-3333-3333-3333-333333333333?",
                "expect_order_id": "33333333-3333-3333-3333-333333333333",
            },
            {
                "user": "Since it is still pending, please change the shipping address to 742 Evergreen Terrace, Springfield.",
                "expect_order_id": "33333333-3333-3333-3333-333333333333",
                "expect_tool": "change_shipping_address",
                "forbid_reasking_order_id": True,
            },
        ],
    },
    {
        "id": "EVAL-06",
        "name": "Hostile / Legal Threat Escalation Guardrail",
        "customer_id": "cust-eval-04",
        "turns": [
            {
                "user": "This service is terrible! I am furious and I am going to sue your company immediately!",
                "expect_action_type": ActionType.ESCALATE,
                "expect_status": ConversationStatus.ESCALATED,
            },
        ],
    },
]


# =====================================================================
# 3. SANDBOX TOOL MOCKS (Deterministic DB Isolation)
# =====================================================================

def _sandbox_check_order_status(order_id: str) -> Optional[OrderRecord]:
    clean_id = str(order_id).strip()
    if clean_id in GOLDEN_ORDERS:
        return GOLDEN_ORDERS[clean_id].model_copy(deep=True)
    return None


def _sandbox_most_recent_order(customer_id: str) -> Optional[OrderRecord]:
    for order in GOLDEN_ORDERS.values():
        if order.customer_id == customer_id:
            return order.model_copy(deep=True)
    return None


# =====================================================================
# 4. EVALUATION RUNNER & SCORER
# =====================================================================

def _extract_order_id_from_state(order_ctx: Any) -> Optional[str]:
    if not order_ctx:
        return None
    if isinstance(order_ctx, dict):
        return str(order_ctx.get("order_id")) if order_ctx.get("order_id") else None
    return str(getattr(order_ctx, "order_id", None))


def _asks_for_order_id(response_text: str) -> bool:
    """Detects if the agent hallucinated a request for an already-known Order ID."""
    lower = response_text.lower()
    triggers = [
        "provide your order id",
        "share your order id",
        "what is your order id",
        "need your order id",
        "provide the order number",
        "share the order number",
    ]
    return any(t in lower for t in triggers)


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = min(len(sorted_data) - 1, max(0, int(round((pct / 100.0) * (len(sorted_data) - 1)))))
    return round(sorted_data[idx], 2)


def run_evaluation(use_mock: bool = False, delay_sec: float = 0.5) -> dict:
    llm = MockLLMClient() if use_mock else get_llm_client()
    llm_name = type(llm).__name__
    is_mock = isinstance(llm, MockLLMClient)

    print("=" * 72)
    print(f" 🧪 AI CUSTOMER SUPPORT AGENT — MULTI-TURN EVAL HARNESS")
    print(f"    LLM Provider : {llm_name} ({'Offline Mock Mode' if is_mock else 'Live Groq Inference'})")
    print(f"    Scenarios    : {len(EVAL_SCENARIOS)} multi-turn workflows")
    print("=" * 72)

    total_turns = 0
    context_checks = 0
    context_passed = 0
    decision_checks = 0
    decision_passed = 0
    turn_latencies_ms: list[float] = []
    node_latencies_ms: dict[str, list[float]] = {}
    scenario_results = []

    # Patch retrieval DB lookups with our deterministic GOLDEN_ORDERS sandbox
    with patch("src.agent.retrieval.check_order_status", side_effect=_sandbox_check_order_status), \
         patch("src.agent.retrieval.most_recent_order_for_customer", side_effect=_sandbox_most_recent_order), \
         patch("src.agent.tools.order_tools.check_order_status", side_effect=_sandbox_check_order_status):

        for scenario in EVAL_SCENARIOS:
            checkpointer = MemorySaver()
            app = build_graph(llm, checkpointer=checkpointer)
            thread_id = f"eval-{scenario['id'].lower()}-{uuid.uuid4().hex[:6]}"
            config = get_thread_config(thread_id)

            print(f"\n▶ [{scenario['id']}] {scenario['name']}")
            messages = []
            scenario_passed = True
            turn_logs = []

            for turn_idx, turn in enumerate(scenario["turns"], start=1):
                total_turns += 1
                user_text = turn["user"]
                img = turn.get("image_source")

                user_content = f"[Image attached: {img}] {user_text}".strip() if img else user_text
                msg_entry = {"role": "user", "content": user_content}
                if img:
                    msg_entry["image_source"] = img
                messages.append(msg_entry)

                state_input: dict[str, Any] = {
                    "conversation_id": thread_id,
                    "customer_id": scenario["customer_id"],
                    "messages": messages,
                }
                if img:
                    state_input["image_source"] = img

                t0 = time.perf_counter()
                error_msg = None
                try:
                    result = app.invoke(state_input, config=config)
                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    result = {}
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                turn_latencies_ms.append(elapsed_ms)

                # Sync messages from graph state
                if result.get("messages"):
                    messages = list(result["messages"])
                elif result.get("final_response"):
                    messages.append({"role": "assistant", "content": result["final_response"]})

                # Collect per-node trace latencies
                for step in (result.get("trace") or [])[-6:]:
                    n_name = getattr(step, "node_name", "unknown")
                    n_lat = getattr(step, "latency_ms", 0.0)
                    node_latencies_ms.setdefault(n_name, []).append(n_lat)

                # --- Assertions ---
                failures: list[str] = []
                if error_msg:
                    failures.append(f"Exception raised: {error_msg}")

                actual_order_id = _extract_order_id_from_state(result.get("order_context"))
                actual_img = result.get("image_source")
                actual_intent = result.get("intent")
                actual_status = result.get("status")
                final_resp = result.get("final_response") or ""
                decision = result.get("decision")
                actual_action_type = getattr(decision, "action_type", None) if decision else None
                actual_tool = getattr(decision, "tool_name", None) if decision else None

                # 1. Context Retention Checks
                if "expect_order_id" in turn:
                    context_checks += 1
                    if actual_order_id == turn["expect_order_id"]:
                        context_passed += 1
                    else:
                        failures.append(
                            f"Order ID mismatch: expected {turn['expect_order_id']}, got {actual_order_id}"
                        )

                if "expect_image_source" in turn:
                    context_checks += 1
                    if actual_img == turn["expect_image_source"]:
                        context_passed += 1
                    else:
                        failures.append(
                            f"Image source lost: expected {turn['expect_image_source']}, got {actual_img}"
                        )

                if turn.get("forbid_reasking_order_id"):
                    context_checks += 1
                    if not _asks_for_order_id(final_resp):
                        context_passed += 1
                    else:
                        failures.append(f"Agent re-asked for Order ID in response: '{final_resp}'")

                # 2. Decision & Tool Accuracy Checks (Evaluated on Live LLM or supported Mock paths)
                if "expect_action_type" in turn:
                    decision_checks += 1
                    if actual_action_type == turn["expect_action_type"]:
                        decision_passed += 1
                    else:
                        failures.append(
                            f"ActionType mismatch: expected {turn['expect_action_type']}, got {actual_action_type}"
                        )

                if "expect_tool" in turn and (not is_mock or turn["expect_tool"] == "report_damaged_item"):
                    decision_checks += 1
                    if actual_tool == turn["expect_tool"]:
                        decision_passed += 1
                    else:
                        failures.append(
                            f"Tool mismatch: expected '{turn['expect_tool']}', got '{actual_tool}' (action_type={actual_action_type})"
                        )

                if "expect_status" in turn and (not is_mock or turn["expect_status"] != ConversationStatus.PENDING_APPROVAL):
                    decision_checks += 1
                    if actual_status == turn["expect_status"]:
                        decision_passed += 1
                    else:
                        failures.append(
                            f"Status mismatch: expected {turn['expect_status']}, got {actual_status}"
                        )

                turn_ok = len(failures) == 0
                if not turn_ok:
                    scenario_passed = False

                badge = "✅ PASS" if turn_ok else "❌ FAIL"
                print(
                    f"   Turn {turn_idx}: {badge} ({elapsed_ms}ms) | "
                    f"intent={actual_intent} | order_pinned={actual_order_id is not None} | "
                    f"tool={actual_tool or 'none'} | status={getattr(actual_status, 'value', actual_status)}"
                )
                for f in failures:
                    print(f"      ⚠️  {f}")

                turn_logs.append({
                    "turn": turn_idx,
                    "user": user_text,
                    "passed": turn_ok,
                    "latency_ms": elapsed_ms,
                    "intent": actual_intent,
                    "order_id": actual_order_id,
                    "tool_called": actual_tool,
                    "status": getattr(actual_status, "value", str(actual_status)),
                    "final_response": final_resp,
                    "failures": failures,
                })

                if not is_mock and delay_sec > 0:
                    time.sleep(delay_sec)

            scenario_results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "passed": scenario_passed,
                "turns": turn_logs,
            })

    # =====================================================================
    # 5. AGGREGATE SCORECARD & LATENCY SUMMARY
    # =====================================================================
    ctx_rate = round((context_passed / context_checks) * 100, 1) if context_checks else 100.0
    dec_rate = round((decision_passed / decision_checks) * 100, 1) if decision_checks else 100.0
    scenarios_passed = sum(1 for s in scenario_results if s["passed"])

    p50_ms = percentile(turn_latencies_ms, 50)
    p95_ms = percentile(turn_latencies_ms, 95)
    mean_ms = round(statistics.mean(turn_latencies_ms), 2) if turn_latencies_ms else 0.0

    node_summary = {
        node: {
            "mean_ms": round(statistics.mean(lats), 2),
            "p95_ms": percentile(lats, 95),
        }
        for node, lats in node_latencies_ms.items()
        if lats
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm_provider": llm_name,
        "summary": {
            "scenarios_total": len(EVAL_SCENARIOS),
            "scenarios_passed": scenarios_passed,
            "total_turns": total_turns,
            "context_retention_rate_pct": ctx_rate,
            "decision_accuracy_rate_pct": dec_rate,
            "turn_latency_mean_ms": mean_ms,
            "turn_latency_p50_ms": p50_ms,
            "turn_latency_p95_ms": p95_ms,
        },
        "node_latencies": node_summary,
        "scenarios": scenario_results,
    }

    report_path = PROJECT_ROOT / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(" 📊 FINAL EVALUATION SCORECARD")
    print("=" * 72)
    print(f"  Scenarios Passed         : {scenarios_passed}/{len(EVAL_SCENARIOS)}")
    print(f"  Context Retention Rate   : {ctx_rate}% ({context_passed}/{context_checks} checks)")
    print(f"  Decision & Tool Accuracy : {dec_rate}% ({decision_passed}/{decision_checks} checks)")
    print(f"  End-to-End Turn Latency  : Mean={mean_ms}ms | P50={p50_ms}ms | P95={p95_ms}ms")
    if node_summary:
        print("  Per-Node Latency Breakdown:")
        for n_name, stats in node_summary.items():
            print(f"    - {n_name:<18}: Mean={stats['mean_ms']}ms | P95={stats['p95_ms']}ms")
    print(f"\n  💾 Full JSON report saved to: {report_path}")
    print("=" * 72)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Turn Agent Evaluations")
    parser.add_argument("--mock", action="store_true", help="Force offline MockLLMClient")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay (seconds) between live LLM turns")
    args = parser.parse_args()
    run_evaluation(use_mock=args.mock, delay_sec=args.delay)