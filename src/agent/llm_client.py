from __future__ import annotations
import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any

from src.agent.state import ActionType, ReasonerDecision

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen/qwen3.8-27b"
# Fallback model in case the requested model is unavailable
FALLBACK_MODEL = "openai/gpt-oss-20b"


def load_env_file(path: str = ".env") -> None:
    """Lightweight .env loader that populates os.environ without requiring external packages."""
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        p = Path(__file__).resolve().parent.parent.parent / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env_file()


def _normalize_order_context(order_context: Any) -> Optional[dict]:
    """Converts OrderRecord Pydantic models or dicts into a clean, JSON-serializable dict."""
    if order_context is None:
        return None
    if isinstance(order_context, dict):
        return order_context
    if hasattr(order_context, "model_dump"):
        return order_context.model_dump(mode="json")
    if hasattr(order_context, "dict"):
        return order_context.dict()
    return {
        "order_id": getattr(order_context, "order_id", None),
        "customer_id": getattr(order_context, "customer_id", None),
        "status": getattr(order_context, "status", None),
        "total_amount": getattr(order_context, "total_amount", None),
        "placed_at": str(getattr(order_context, "placed_at", "")),
        "delivered_at": str(getattr(order_context, "delivered_at", "")),
        "items": getattr(order_context, "items", []),
    }


class LLMClient(ABC):
    """Abstract interface for LLM client providers."""

    @abstractmethod
    def classify_intent(self, message: str, labels: list[str]) -> str:
        """Classify a customer message into one of the allowed intent labels."""
        ...

    @abstractmethod
    def decide(
        self,
        message: str,
        order_context: dict | None,
        policy_context: list[str],
        image_source: Optional[str] = None,
    ) -> ReasonerDecision:
        """Analyze message, policy, and order state to decide on the next graph action."""
        ...


class MockLLMClient(LLMClient):
    """Deterministic mock client used for unit testing and offline development."""

    def classify_intent(self, message: str, labels: list[str]) -> str:
        text = message.lower()
        if any(w in text for w in ["where", "status", "track", "arrive", "package"]):
            return "order_status"
        if any(w in text for w in ["refund", "money_back", "money back", "return", "cancel"]):
            return "refund_request"
        if any(w in text for w in ["ship", "delivery", "address"]):
            return "shipping_question"
        return "general_question"

    def decide(
        self,
        message: str,
        order_context: dict | None,
        policy_context: list[str],
        image_source: Optional[str] = None,
    ) -> ReasonerDecision:
        text = message.lower()
        if "sue" in text or "furious" in text or "terrible" in text:
            return ReasonerDecision(
                action_type=ActionType.ESCALATE,
                rationale="Detected angry sentiment or legal threat; escalating per policy.",
            )

        if ("damage" in text or "broken" in text or "defect" in text) and (
            image_source or "image" in text or "photo" in text
        ):
            resolved_order_id = (
                order_context.get("order_id")
                if isinstance(order_context, dict)
                else getattr(order_context, "order_id", "mock-order-id")
            ) if order_context else "mock-order-id"
            return ReasonerDecision(
                action_type=ActionType.TOOL_CALL,
                tool_name="report_damaged_item",
                tool_args={
                    "order_id": str(resolved_order_id),
                    "photo_provided": True,
                    "image_source": image_source or "mock_photo.png",
                    "notes": "Customer submitted photo of damaged item.",
                },
                rationale="Customer reported damaged item with photo evidence.",
            )

        if order_context is not None:
            status = (
                order_context.get("status")
                if isinstance(order_context, dict)
                else getattr(order_context, "status", "shipped")
            )
            return ReasonerDecision(
                action_type=ActionType.ANSWER,
                rationale="Order context available, answering directly from status.",
                draft_response=(
                    f"Your order is currently '{status}'. Let me know if you would like more detail."
                ),
            )

        return ReasonerDecision(
            action_type=ActionType.ESCALATE,
            rationale="No order context found and mock client has no fallback reasoning.",
        )


class GroqLLMClient(LLMClient):
    """
    Production LLM client connecting to Groq's high-speed inference engine.
    Supports structured JSON reasoning, tool calling, and automatic model fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        from groq import Groq

        # Support both GROQ_API_KEY and GROK_API_KEY from environment
        resolved_key = (
            api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("GROK_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY or GROK_API_KEY in your .env file."
            )

        self.client = Groq(api_key=resolved_key)
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self._active_model = self.model

    def _call_with_fallback(self, **kwargs) -> Any:
        """Invokes Groq API with automatic fallback if the requested model is decommissioned or not found."""
        try:
            return self.client.chat.completions.create(model=self._active_model, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "404" in err_msg or "decommissioned" in err_msg:
                logger.warning(
                    f"Model '{self._active_model}' is unavailable on Groq. "
                    f"Falling back to '{FALLBACK_MODEL}'."
                )
                self._active_model = FALLBACK_MODEL
                return self.client.chat.completions.create(model=self._active_model, **kwargs)
            raise

    def classify_intent(self, message: str, labels: list[str]) -> str:
        """Uses Groq to classify message intent into exactly one category."""
        labels_str = ", ".join(f'"{label}"' for label in labels)
        prompt = (
            f"You are a customer support routing classifier.\n"
            f"Allowed labels: [{labels_str}]\n"
            f"Classify the customer's current message into exactly one of the allowed labels.\n"
            f"If the message is a short follow-up (like providing an Order ID, photo, or confirmation), "
            f"preserve the active intent from the recent context.\n"
            f"Input: \"{message}\"\n\n"
            f"Reply with ONLY the exact label name, with no quotes or extra text."
        )

        resp = self._call_with_fallback(
            messages=[
                {"role": "system", "content": "You are a concise intent classification engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )

        raw_label = resp.choices[0].message.content.strip().lower()
        for label in labels:
            if label.lower() in raw_label:
                return label
        return "general_question"

    def decide(
        self,
        message: str,
        order_context: dict | None,
        policy_context: list[str],
        image_source: Optional[str] = None,
    ) -> ReasonerDecision:
        """
        Synthesizes policies, multi-turn history, and order details to choose an action,
        invoking tools or escalating when required.
        """
        policies_text = (
            "\n---\n".join(policy_context)
            if policy_context
            else "No specific policy retrieved for this query."
        )
        normalized_order = _normalize_order_context(order_context)
        order_text = json.dumps(normalized_order, default=str) if normalized_order else "None"

        # Check if image_source was provided or embedded in message text
        if not image_source:
            m = re.search(r"\[Image attached:\s*([^\]]+)\]", message)
            if m:
                image_source = m.group(1).strip()

        image_info = ""
        if image_source:
            image_info = (
                f"\n### Customer Attached Photo Evidence:\n"
                f"- Image Source: {image_source}\n"
                f"- Photo Status: PROVIDED by customer in this session.\n"
                f"INSTRUCTION: If the customer is reporting or following up on a damaged/defective item "
                f"and an Order ID is available in Customer Order Context, call report_damaged_item with "
                f"order_id, photo_provided=True, and image_source='{image_source}'.\n"
            )

        system_prompt = f"""You are an expert, stateful customer support agent for an e-commerce company.
Your role is to reason through customer inquiries strictly according to company policies, conversation history, and verified order data.

### Relevant Company Policies:
{policies_text}

### Customer Order Context (VERIFIED IN DATABASE):
{order_text}
{image_info}
### Operational Tools Available:
- initiate_refund(order_id: str, amount: float, reason: str): Process or request a refund.
- change_shipping_address(order_id: str, new_address: str): Change shipping destination (Allowed ONLY if order status is 'pending').
- report_lost_package(order_id: str, notes: str): Flag package as lost for carrier investigation.
- report_damaged_item(order_id: str, photo_provided: bool, image_source: str, notes: str): Report damaged/broken goods. Automated vision will inspect image_source if provided.
- check_order_status(order_id: str): Query tracking information.

### Critical Multi-Turn Memory & Grounding Rules:
1. **NEVER RE-ASK FOR KNOWN DATA**: If `Customer Order Context` above is not `None`, you ALREADY have the customer's `order_id`, `status`, `total_amount`, and `items`. NEVER ask the customer for their Order ID again. Use the `order_id` from `Customer Order Context` directly in your `tool_args` or `draft_response`.
2. **CONNECT FOLLOW-UP REPLIES TO PRIOR TURNS**: Read the `Recent Conversation History` in the user prompt carefully.
   - If the customer asked for a refund/address change/damage claim in Turn 1, and in Turn 2 they provide their Order ID or photo, immediately execute the requested `tool_call` now that you have the missing piece.
   - If the customer asks for a refund without specifying a dollar amount, default `amount` to `total_amount` from `Customer Order Context`.
3. **TOOL CALL**: Call an operational tool when the customer requests an actionable task and prerequisites are met:
   - For refunds (non-damage): call `initiate_refund`.
   - For address changes: call `change_shipping_address`.
   - For lost packages: call `report_lost_package`.
   - For damaged items: ONLY call `report_damaged_item` if `image_source` (photo evidence) has been provided in this turn or an earlier turn AND an order is known. If the customer reports damage but HAS NOT provided a photo yet, set `action_type="answer"` and warmly ask for a photo. If they provided a photo but `Customer Order Context` is `None`, set `action_type="answer"` and ask for their Order ID.
4. **ANSWER**: If the customer asks an informational question or needs to provide missing required info (either Order ID when `Customer Order Context` is `None`, or a photo for a damage claim), set `action_type="answer"` and provide a warm, concise `draft_response`.
5. **ESCALATE**: If the customer is furious, threatens legal action, or cannot be resolved under policy, set `action_type="escalate"`.

You MUST reply with a valid JSON object matching this exact schema:
{{
  "action_type": "answer" | "tool_call" | "escalate",
  "rationale": "Brief explanation of your reasoning",
  "tool_name": "string or null",
  "tool_args": {{}},
  "draft_response": "Polite draft response to the customer (required if action_type is answer)"
}}"""

        resp = self._call_with_fallback(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=350,
        )

        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ReasonerDecision(
                action_type=ActionType.ANSWER,
                rationale="Could not parse LLM output as JSON; defaulting to raw text.",
                draft_response=content,
            )

        raw_action_type = str(parsed.get("action_type", "answer")).lower()
        if raw_action_type == "tool_call":
            action_type = ActionType.TOOL_CALL
        elif raw_action_type == "escalate":
            action_type = ActionType.ESCALATE
        else:
            action_type = ActionType.ANSWER

        tool_name = parsed.get("tool_name")
        tool_args = parsed.get("tool_args") or {}

        # Deterministic backfill from verified session state so the LLM never drops pinned args
        if action_type == ActionType.TOOL_CALL and normalized_order:
            pinned_order_id = str(normalized_order.get("order_id", ""))
            if pinned_order_id and not tool_args.get("order_id"):
                tool_args["order_id"] = pinned_order_id

            if tool_name == "initiate_refund":
                raw_amt = tool_args.get("amount")
                try:
                    amt_val = float(raw_amt) if raw_amt is not None else 0.0
                except (ValueError, TypeError):
                    amt_val = 0.0
                # If customer didn't specify a partial amount, default to full order total_amount
                if amt_val <= 0.0 and normalized_order.get("total_amount"):
                    amt_val = float(normalized_order["total_amount"])
                tool_args["amount"] = amt_val
                tool_args.setdefault("reason", "Customer requested refund")

            elif tool_name == "report_damaged_item" and image_source:
                tool_args.setdefault("photo_provided", True)
                tool_args.setdefault("image_source", image_source)
                tool_args.setdefault("notes", "Customer submitted photo evidence of damage.")

        # Normalize numeric types in tool arguments even if order_context was None
        if "amount" in tool_args:
            try:
                tool_args["amount"] = float(tool_args["amount"])
            except (ValueError, TypeError):
                tool_args["amount"] = 0.0

        return ReasonerDecision(
            action_type=action_type,
            rationale=parsed.get("rationale", "LLM reasoning complete."),
            tool_name=tool_name,
            tool_args=tool_args,
            draft_response=parsed.get("draft_response"),
        )


def get_llm_client() -> LLMClient:
    """
    Factory function returning GroqLLMClient if an API key is available in environment,
    or MockLLMClient for offline testing.
    """
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY")
    if api_key:
        try:
            return GroqLLMClient()
        except Exception as e:
            logger.warning(f"Could not initialize GroqLLMClient: {e}. Falling back to MockLLMClient.")
    return MockLLMClient()