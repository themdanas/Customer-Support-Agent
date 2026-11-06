from __future__ import annotations
import io
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

from src.cli import (
    main,
    format_trace_step,
    clean_path,
    is_image_path_or_url,
    parse_user_cli_input,
)
from src.agent.state import StepLog
from src.agent.llm_client import MockLLMClient


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


def test_clean_path():
    assert clean_path("'/path/to/img.png'") == "/path/to/img.png"
    assert clean_path('"/path/to/my img.png"') == "/path/to/my img.png"
    assert clean_path(r"/path/to/escaped\ space.png") == "/path/to/escaped space.png"


def test_is_image_path_or_url(tmp_path):
    assert is_image_path_or_url("https://example.com/damaged.jpg") is True
    assert is_image_path_or_url("http://cdn.store.com/item.png") is True
    assert is_image_path_or_url("data:image/png;base64,abc123") is True

    img_file = tmp_path / "damage.jpg"
    img_file.write_bytes(b"dummy")
    assert is_image_path_or_url(str(img_file)) is True
    assert is_image_path_or_url("just a normal question") is False


def test_parse_user_cli_input():
    # 1. /image with path and notes
    msg, img = parse_user_cli_input("/image /tmp/broken.png The item arrived cracked")
    assert img == "/tmp/broken.png"
    assert msg == "The item arrived cracked"

    # 2. Direct path
    msg, img = parse_user_cli_input("/tmp/cracked_screen.jpg")
    assert img == "/tmp/cracked_screen.jpg"
    assert "photo" in msg

    # 3. Embedded path
    msg, img = parse_user_cli_input("Here is the damaged box: https://example.com/box.png please refund")
    assert img == "https://example.com/box.png"
    assert "Here is the damaged box" in msg

    # 4. Normal text without image
    msg, img = parse_user_cli_input("Where is my package?")
    assert img is None
    assert msg == "Where is my package?"


def test_parse_user_cli_input_interactive(monkeypatch):
    inputs = iter(["/tmp/interactive.png", "Item is totally crushed"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    msg, img = parse_user_cli_input("/image")
    assert img == "/tmp/interactive.png"
    assert msg == "Item is totally crushed"


def test_cli_simple_exit(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
    main()
    captured = capsys.readouterr()
    assert "Customer Support Agent" in captured.out
    assert "Goodbye!" in captured.out


def test_cli_help_command(monkeypatch, capsys):
    inputs = iter(["/help", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    main()
    captured = capsys.readouterr()
    assert "Supported Commands & Tips" in captured.out
    assert "/image" in captured.out


def test_cli_single_turn_interaction(monkeypatch, capsys):
    monkeypatch.setattr("src.cli.get_llm_client", lambda: MockLLMClient())
    inputs = iter(["Where is my order?", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    main()
    captured = capsys.readouterr()
    assert "Execution Trace" in captured.out
    assert "ROUTER" in captured.out
    assert "Agent:" in captured.out


def test_cli_image_input_interaction(monkeypatch, capsys, tmp_path):
    img_file = tmp_path / "broken_phone.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")

    monkeypatch.setattr("src.cli.get_llm_client", lambda: MockLLMClient())
    inputs = iter([f"/image {img_file} Order is broken", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    main()
    captured = capsys.readouterr()
    assert "Photo Attached" in captured.out
    assert "Execution Trace" in captured.out
