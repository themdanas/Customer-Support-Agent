from __future__ import annotations
import os
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
    ) -> ReasonerDecision:
        text = message.lower()
        if "sue" in text or "furious" in text or "terrible" in text:
            return ReasonerDecision(
                action_type=ActionType.ESCALATE,
                rationale="Detected angry sentiment or legal threat; escalating per policy.",
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
            f"Classify the customer's message into exactly one of the allowed labels.\n"
            f"Customer message: \"{message}\"\n\n"
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
    ) -> ReasonerDecision:
        """
        Synthesizes policies and order details to choose an action, invoking tools or escalating when required.
        """
        policies_text = (
            "\n---\n".join(policy_context)
            if policy_context
            else "No specific policy retrieved for this query."
        )
        order_text = json.dumps(order_context, default=str) if order_context else "None"

        system_prompt = f"""You are an expert customer support agent for an e-commerce company.
Your role is to reason through customer inquiries strictly according to company policies and order data.

### Relevant Company Policies:
{policies_text}

### Customer Order Context:
{order_text}

### Operational Tools Available:
- initiate_refund(order_id: str, amount: float, reason: str): Process or request a refund.
- change_shipping_address(order_id: str, new_address: str): Change shipping destination (Allowed ONLY if order status is 'pending').
- report_lost_package(order_id: str, notes: str): Flag package as lost for carrier investigation.
- report_damaged_item(order_id: str, photo_provided: bool, notes: str): Report damaged/broken goods.
- check_order_status(order_id: str): Query tracking information.

### Decision Guidelines:
1. **TOOL CALL**: If customer requests an actionable task (e.g., refund, address change, report issue), set action_type="tool_call", specify tool_name and valid tool_args.
2. **ANSWER**: If customer asks an informational question (e.g. return policy, tracking explanation), set action_type="answer" and provide a helpful, polite draft_response grounded in the policy context.
3. **ESCALATE**: If customer is furious, threatens bad reviews or legal action, or if their situation cannot be resolved under policy, set action_type="escalate".

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

        raw_action_type = parsed.get("action_type", "answer").lower()
        if raw_action_type == "tool_call":
            action_type = ActionType.TOOL_CALL
        elif raw_action_type == "escalate":
            action_type = ActionType.ESCALATE
        else:
            action_type = ActionType.ANSWER

        tool_args = parsed.get("tool_args") or {}
        # Normalize numeric types in tool arguments (e.g. refund amount)
        if "amount" in tool_args:
            try:
                tool_args["amount"] = float(tool_args["amount"])
            except (ValueError, TypeError):
                tool_args["amount"] = 0.0

        return ReasonerDecision(
            action_type=action_type,
            rationale=parsed.get("rationale", "LLM reasoning complete."),
            tool_name=parsed.get("tool_name"),
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
