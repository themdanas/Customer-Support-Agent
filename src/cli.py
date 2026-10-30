#!/usr/bin/env python3
"""
Interactive Command-Line Interface (CLI) for the Customer Support Agent.
Allows developers and users to converse with the agent, inspect step-by-step
graph traces, and simulate human supervisor approval workflows in real time.
"""
from __future__ import annotations
import sys
from pathlib import Path
import uuid

# Ensure repository root is on sys.path regardless of execution method
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import build_graph
from src.agent.llm_client import MockLLMClient
from src.agent.persistence import get_checkpointer, get_thread_config, close_connection_pool
from src.agent.human_approval import resume_approval
from src.agent.state import ConversationStatus


def format_trace_step(log) -> str:
    return f"  > [{log.node_name.upper()} ({log.latency_ms}ms)]: {log.output_summary}"


def main():
    print("=" * 65)
    print(" 🤖 Customer Support Agent - Interactive Debug CLI")
    print("=" * 65)
    print("Type 'exit' or 'quit' to end the session.\n")

    # 1. Initialize checkpointer (Postgres with MemorySaver fallback)
    checkpointer = get_checkpointer(use_postgres=True)
    llm = MockLLMClient()
    app = build_graph(llm, checkpointer=checkpointer)

    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    customer_id = f"cust-{uuid.uuid4().hex[:8]}"
    config = get_thread_config(conversation_id)

    messages = []
    print(f"Session started. Thread ID: {conversation_id} | Customer ID: {customer_id}\n")

    try:
        while True:
            try:
                user_input = input("Customer: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting session. Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting session. Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})

            current_state = {
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "messages": messages,
            }

            # Invoke the LangGraph workflow
            result = app.invoke(current_state, config=config)

            # Print detailed execution trace
            trace = result.get("trace", [])
            if trace:
                print("\n--- Execution Trace ---")
                for log in trace:
                    print(format_trace_step(log))
                print("-----------------------")

            status = result.get("status")

            # Handle Human-in-the-Loop interrupt
            if status == ConversationStatus.PENDING_APPROVAL:
                action_result = result.get("action_result")
                reason = getattr(action_result, "escalation_reason", "High-risk action requires review.")
                print(f"\n[⚠️ ESCALATION] Request paused for supervisor review: {reason}")
                
                try:
                    decision = input("Supervisor Action: (a)pprove / (r)eject? [a/r]: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    break

                approved = decision.startswith("a")
                notes = "Approved via CLI" if approved else "Rejected via CLI"
                
                # Resume graph execution with supervisor command
                result = resume_approval(
                    app=app,
                    thread_id=conversation_id,
                    approved=approved,
                    reviewer="cli_supervisor",
                    notes=notes,
                )
                print("[✅ Resumed] Graph resumed with supervisor decision.")

            final_response = result.get(
                "final_response", "I apologize, but I am unable to process your request."
            )
            messages.append({"role": "assistant", "content": final_response})
            print(f"\nAgent: {final_response}\n")
    finally:
        close_connection_pool()


if __name__ == "__main__":
    main()
