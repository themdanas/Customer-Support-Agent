from __future__ import annotations
import io
from unittest.mock import patch
from src.cli import main, format_trace_step
from src.agent.state import StepLog
from datetime import datetime, timezone


def test_format_trace_step():
    log = StepLog(
        node_name="router",
        started_at=datetime.now(timezone.utc),
        latency_ms=12.34,
        input_summary="intent=None",
        output_summary="intent=order_status",
    )
    formatted = format_trace_step(log)
    assert "[ROUTER (12.34ms)]" in formatted
    assert "intent=order_status" in formatted


def test_cli_simple_exit(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
    main()
    captured = capsys.readouterr()
    assert "Customer Support Agent" in captured.out
    assert "Goodbye!" in captured.out


def test_cli_single_turn_interaction(monkeypatch, capsys):
    inputs = iter(["Where is my order?", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    main()
    captured = capsys.readouterr()
    assert "Execution Trace" in captured.out
    assert "ROUTER" in captured.out
    assert "Agent:" in captured.out

