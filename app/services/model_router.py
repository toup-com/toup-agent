"""
Model Router — simple key-based model selection.

No classifier, no tiers. Just picks the best model based on available keys:
  - Anthropic (with or without OpenAI): Claude Opus 4.6 for everything
  - OpenAI only: GPT-5.4 for everything
  - Rate limit / auth fallback: cross to the other provider

Voice:
  - Both keys: OpenAI realtime (default)
  - OpenAI only: OpenAI realtime
  - Anthropic only: Anthropic voice model
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def _is_claude_model(model: str) -> bool:
    """Check if a model name refers to an Anthropic Claude model."""
    return model.startswith("claude-")


@dataclass
class RoutingDecision:
    """Result of model selection."""
    tier: str                       # always "default" now
    model: str                      # actual model ID to use
    label: str                      # human-readable label
    confidence: float               # always 1.0
    reason: str                     # short explanation
    signals: Dict[str, float] = field(default_factory=dict)


def classify_request(
    user_message: str,
    conversation_history: Optional[List[Dict]] = None,
    has_media: bool = False,
    preferred_provider: Optional[str] = None,
) -> RoutingDecision:
    """
    Select the model based on configured preference + available providers.

    Priority:
      1. Bundle mode: respect `preferred_provider` (anthropic/openai). Both
         providers are reachable via the proxy regardless of local keys.
      2. BYOK Anthropic key → Claude Opus 4.7
      3. BYOK OpenAI key    → GPT-5.4
      4. No keys + no bundle → Claude Opus 4.7 (sane default; will fail)

    Bundle subscribers always have BOTH providers available through the
    Toup proxy, so the local key_provider check (which sees no keys for
    bundle agents) would always fall through to the default — not what
    the user wants. The preferred_provider arg lets the caller (the chat
    handler) pass the user's choice from agent_configs.preferred_provider.
    """
    from app.services.key_provider import keys
    from app.config import settings

    # Bundle mode: trust preferred_provider, fall back to anthropic
    if settings.llm_mode == "bundle" and settings.toup_token:
        choice = (preferred_provider or "anthropic").lower()
        if choice == "openai":
            model = "gpt-5.4"
            label = "GPT-5.4"
            reason = f"bundle mode + preferred=openai"
        else:
            model = "claude-opus-4-7"
            label = "Claude Opus 4.7"
            reason = f"bundle mode + preferred=anthropic"
        logger.info(f"[ROUTER] {reason}")
        return RoutingDecision(
            tier="default", model=model, label=label, confidence=1.0, reason=reason,
        )

    # BYOK paths
    if keys.has_anthropic:
        model = "claude-opus-4-7"
        label = "Claude Opus 4.7"
        reason = "anthropic key available → opus"
    elif keys.has_openai:
        model = "gpt-5.4"
        label = "GPT-5.4"
        reason = "openai key only → gpt-5.4"
    else:
        model = "claude-opus-4-7"
        label = "Claude Opus 4.7"
        reason = "no keys → default opus"

    logger.info(f"[ROUTER] {reason}")

    return RoutingDecision(
        tier="default",
        model=model,
        label=label,
        confidence=1.0,
        reason=reason,
    )


def get_model_for_auto(
    user_message: str,
    conversation_history: Optional[List[Dict]] = None,
    has_media: bool = False,
) -> str:
    """Convenience function: return just the model ID."""
    decision = classify_request(user_message, conversation_history, has_media)
    return decision.model
