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
) -> RoutingDecision:
    """
    Select the model based on available API keys. No complexity classification.

    Priority:
      1. Anthropic key available → Claude Opus 4.6
      2. OpenAI key available    → GPT-5.4
      3. No keys                 → Claude Opus 4.6 (will fail, but sane default)
    """
    from app.services.key_provider import keys

    if keys.has_anthropic:
        model = "claude-opus-4-6"
        label = "Claude Opus 4.6"
        reason = "anthropic key available → opus"
    elif keys.has_openai:
        model = "gpt-5.4"
        label = "GPT-5.4"
        reason = "openai key only → gpt-5.4"
    else:
        model = "claude-opus-4-6"
        label = "Claude Opus 4.6"
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
