#!/usr/bin/env python3
"""
Interactive Command-Line Interface (CLI) for the Customer Support Agent.
Allows developers and users to converse with the agent, inspect step-by-step
graph traces, attach images for vision inspection, and simulate human supervisor
approval workflows in real time.
"""
from __future__ import annotations
import sys
import re
import shlex
import uuid
from pathlib import Path
from typing import Optional

# Ensure repository root is on sys.path regardless of execution method
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import build_graph
from src.agent.llm_client import get_llm_client
from src.agent.persistence import get_checkpointer, get_thread_config, close_connection_pool
from src.agent.human_approval import resume_approval
from src.agent.state import ConversationStatus

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def clean_path(path_str: str) -> str:
    """Strips quotes, normalizes mac terminal escaped spaces, and expands ~."""
    s = path_str.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()
    # Unescape spaces from macOS terminal drag-and-drop (e.g. "path/to/my\ file.png")
    s = s.replace(r"\ ", " ")
    return s


def is_image_path_or_url(candidate: str) -> bool:
    """Checks if a string is solely an image URL, data URI, or local image file."""
    c = clean_path(candidate)
    if not c:
        return False
    try:
        p = Path(c).expanduser()
        if p.is_file():
            return True
    except Exception:
        pass

    # If it contains spaces and is not an existing file on disk, treat as a message with potential embedded image
    if " " in c:
        return False

    if c.startswith(("http://", "https://", "data:image/")):
        return True

    try:
        p = Path(c).expanduser()
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    except Exception:
        pass
    return False


def parse_user_cli_input(raw: str) -> tuple[str, Optional[str]]:
    """
    Parses user input from the CLI.
    Supports:
      1. /image <path_or_url> [optional message]
      2. /image (prompts interactively for image path and optional message)
      3. Direct file path or URL pasted as the message
      4. Message containing an image path or URL
      5. Standard text message
    Returns:
      (message_text, image_source)
    """
    raw = raw.strip()
    if not raw:
        return "", None

    # 1. /image command
    if raw.startswith("/image"):
        rest = raw[6:].strip()
        if not rest:
            # Interactive prompt mode
            print("📸 [Image Attachment Mode]")
            try:
                img_path = input("   Enter image file path or URL: ").strip()
                img_path = clean_path(img_path)
                if not img_path:
                    print("   No image path provided. Cancelling image attachment.")
                    return "", None
                user_msg = input("   Message or notes (press Enter to skip): ").strip()
            except (KeyboardInterrupt, EOFError):
                return "", None
            return user_msg or "Here is the photo of the item.", img_path

        # Split rest into image path and optional message
        try:
            parts = shlex.split(rest)
        except Exception:
            parts = rest.split(maxsplit=1)

        if parts:
            img_path = clean_path(parts[0])
            user_msg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
            return user_msg or "Here is the photo of the item.", img_path

    # 2. Direct image path or URL pasted directly into prompt
    if is_image_path_or_url(raw):
        img_path = clean_path(raw)
        return "Here is the photo of the item.", img_path

    # 3. Text containing an embedded image path or URL
    img_match = re.search(
        r"(https?://[^\s]+\.(?:png|jpg|jpeg|webp|gif|bmp)|data:image/[^\s]+|/[^\s]+\.(?:png|jpg|jpeg|webp|gif|bmp)|~[^\s]+\.(?:png|jpg|jpeg|webp|gif|bmp))",
        raw,
        re.IGNORECASE,
    )
    if img_match:
        matched_path = clean_path(img_match.group(0))
        return raw, matched_path

    return raw, None


def format_trace_step(log) -> str:
    return f"  > [{log.node_name.upper()} ({log.latency_ms}ms)]: {log.output_summary}"


def main():
    print("=" * 65)
    print(" 🤖 Customer Support Agent - Interactive Debug CLI")
    print("=" * 65)
    print("Commands:")
    print("  - Type 'exit' or 'quit' to end session.")
    print("  - Type '/image <path_or_url> [message]' to attach a photo.")
    print("  - Or paste an image path/URL directly into the chat prompt.")
    print("  - Type '/help' for command details.\n")

    # 1. Initialize checkpointer (Postgres with MemorySaver fallback)
    checkpointer = get_checkpointer(use_postgres=True)
    llm = get_llm_client()
    print(f"Loaded LLM Provider: {type(llm).__name__}")
    app = build_graph(llm, checkpointer=checkpointer)

    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    customer_id = f"cust-{uuid.uuid4().hex[:8]}"
    config = get_thread_config(conversation_id)

    messages = []
    last_assistant_response = ""
    print(f"Session started. Thread ID: {conversation_id} | Customer ID: {customer_id}\n")

    try:
        while True:
            # If the agent just requested photo evidence, display a contextual prompt hint
            if any(term in (last_assistant_response or "").lower() for term in ("photo", "photographic", "picture", "evidence")):
                print("💡 [Tip: Attach your photo with '/image <path_or_url>' or paste the file path directly]")

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
            if user_input.lower() in ("/help", "help"):
                print("\n📖 Supported Commands & Tips:")
                print("  /image <path_or_url> [message] - Attach an image (e.g. photo of damaged item)")
                print("  /image                         - Interactively prompt for an image path/URL")
                print("  /help                          - Show this help menu")
                print("  exit / quit                    - End the support session")
                print("  You can also paste a file path (or drag & drop from Finder) or URL directly.\n")
                continue

            user_text, image_source = parse_user_cli_input(user_input)
            if not user_text and not image_source:
                continue

            if image_source:
                # Check local file existence if not a remote URL
                if not image_source.startswith(("http://", "https://", "data:")):
                    p = Path(image_source).expanduser()
                    if not p.is_file():
                        print(f"⚠️  [Notice] File '{image_source}' was not found on disk. Proceeding anyway.")
                print(f"📎 [Photo Attached]: {image_source}")
                user_content = f"[Image attached: {image_source}] {user_text}".strip()
            else:
                user_content = user_text

            new_user_msg = {"role": "user", "content": user_content}
            if image_source:
                new_user_msg["image_source"] = image_source
            messages.append(new_user_msg)

            current_state = {
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "messages": messages,
            }
            # Only pass image_source when a new image is attached so we don't overwrite
            # an image attached in an earlier turn with None!
            if image_source:
                current_state["image_source"] = image_source

            # Invoke the LangGraph workflow with automatic recovery on transient DB restarts
            try:
                result = app.invoke(current_state, config=config)
            except Exception as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ("adminshutdown", "terminating connection", "closed connection", "broken pipe")):
                    print("\n[🔄 Reconnecting] Database connection was interrupted (e.g. server restart). Re-establishing pool...")
                    close_connection_pool()
                    checkpointer = get_checkpointer(use_postgres=True, force_refresh=True)
                    app = build_graph(llm, checkpointer=checkpointer)
                    result = app.invoke(current_state, config=config)
                else:
                    raise

            # Print detailed execution trace (only the steps from the latest turn)
            trace = result.get("trace", [])
            if trace:
                print("\n--- Execution Trace ---")
                for log in trace[-6:]:
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

            final_response = (
                result.get("final_response")
                or "I apologize, but I am unable to process your request."
            )
            last_assistant_response = str(final_response or "")
            # Sync local messages list with graph's authoritative messages list
            if result.get("messages"):
                messages = list(result["messages"])
            else:
                messages.append({"role": "assistant", "content": final_response})
            print(f"\nAgent: {final_response}\n")
    finally:
        close_connection_pool()


if __name__ == "__main__":
    main()
