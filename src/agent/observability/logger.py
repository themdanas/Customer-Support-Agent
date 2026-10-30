from __future__ import annotations
import time
import functools
from datetime import datetime, timezone

from src.agent.state import StepLog

def with_logging(node_name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            started = datetime.now(timezone.utc)
            t0 = time.perf_counter()
            input_summary = f"intent={state.get('intent')}, messages={len(state.get('messages', []))}"
            error = None
            try:
                result = fn(state)
            except Exception as e:
                error = str(e)
                result = state
                raise
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000
                output_summary = _summarize(result, node_name)
                log = StepLog(
                    node_name=node_name,
                    started_at=started,
                    latency_ms=round(latency_ms, 2),
                    input_summary=input_summary,
                    output_summary=output_summary,
                    error=error,
                )
                result.setdefault("trace", [])
                result["trace"].append(log)

            return result
        return wrapper
    return decorator

def _summarize(state: dict, node_name: str) -> str:
    if node_name == "router":
        return f"intent={state.get('intent')}"
    if node_name == "retrieval":
        order = state.get("order_context")
        return f"order_found={order is not None}"
    if node_name == "reasoner":
        d = state.get("decision")
        action_type = getattr(d, "action_type", None) or (d.get("action_type") if isinstance(d, dict) else None)
        return f"decision={action_type}"
    if node_name == "action":
        r = state.get("action_result")
        success = getattr(r, "success", None) or (r.get("success") if isinstance(r, dict) else None)
        return f"success={success}"
    if node_name == "human_approval":
        appr = state.get("approval_decision")
        approved = getattr(appr, "approved", None) if appr else None
        return f"approved={approved}"
    if node_name == "response_builder":
        return f"status={state.get('status')}"
    return "ok"

