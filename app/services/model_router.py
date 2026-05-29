"""
Model Router — simple key-based model selection.

No classifier, no tiers. Just picks the best model based on available keys:
  - Anthropic key available: agent default (resolved through model_resolver)
  - OpenAI key only:         cross-provider fallback (also resolved)
  - No keys:                 agent default (will fail, but sane default)

All literal model strings live in `app.services.model_resolver`. To
upgrade the platform-wide default, change `settings.agent_model`
(env var `AGENT_MODEL`).

Voice:
  - Both keys: OpenAI realtime (default)
  - OpenAI only: OpenAI realtime
  - Anthropic only: Anthropic voice model
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import settings
from app.services.model_resolver import (
    default_model,
    default_fallback_model,
    is_claude_model,
)

logger = logging.getLogger(__name__)


def _is_claude_model(model: str) -> bool:
    """Backwards-compatible local alias for resolver's classifier."""
    return is_claude_model(model)


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
         providers are reachable via the proxy regardless of local keys;
         the model id is resolved through `model_resolver`.
      2. BYOK Anthropic key → agent default model (model_resolver)
      3. BYOK OpenAI key    → cross-provider fallback (model_resolver)
      4. No keys + no bundle → agent default (will fail, but sane default)

    Bundle subscribers always have BOTH providers available through the
    Toup proxy, so the local key_provider check (which sees no keys for
    bundle agents) would always fall through to the default — not what
    the user wants. The preferred_provider arg lets the caller (the chat
    handler) pass the user's choice from agent_configs.preferred_provider.

    All literal model strings live in `app.services.model_resolver`. To
    upgrade the platform-wide default, change `settings.agent_model`
    (env var `AGENT_MODEL`).
    """
    from app.services.key_provider import keys
    from app.config import settings

    anthropic_on = getattr(settings, "anthropic_enabled", True)

    def _openai_model() -> str:
        """A guaranteed-OpenAI model id. Uses the resolver's
        default_openai_model() which returns the agent default when it's
        already OpenAI (e.g. gpt-5.5) and otherwise falls back to the
        canonical OpenAI model — never a Claude id, even if both the
        configured default AND fallback happen to be Claude (e.g. a dev
        env with AGENT_MODEL pinned to Claude)."""
        from app.services.model_resolver import default_openai_model
        return default_openai_model()

    # Anthropic deactivated platform-wide → always OpenAI, regardless of
    # preferred_provider / available keys. This is the router-level half of
    # the kill switch (the proxy enforces the other half as a hard backstop).
    if not anthropic_on:
        model = _openai_model()
        reason = f"anthropic disabled → openai {model}"
        logger.info(f"[ROUTER] {reason}")
        return RoutingDecision(
            tier="default", model=model, label=model, confidence=1.0, reason=reason,
        )

    # Bundle mode: trust preferred_provider. Default to OPENAI now (was
    # anthropic) — each user has their own per-tenant OpenAI key, whereas
    # Anthropic rides one shared platform account. Only an explicit
    # preferred=anthropic opts into Claude.
    if settings.llm_mode == "bundle" and settings.toup_token:
        choice = (preferred_provider or "openai").lower()
        if choice == "anthropic":
            from app.services.model_resolver import default_anthropic_model
            model = default_anthropic_model()
            label = model
            reason = f"bundle mode + preferred=anthropic → {model}"
        else:
            model = _openai_model()
            label = model
            reason = f"bundle mode + preferred=openai → {model}"
        logger.info(f"[ROUTER] {reason}")
        return RoutingDecision(
            tier="default", model=model, label=label, confidence=1.0, reason=reason,
        )

    # BYOK paths
    if keys.has_anthropic:
        model = default_model()
        reason = f"anthropic key available → {model}"
    elif keys.has_openai:
        model = default_fallback_model()
        reason = f"openai key only → {model}"
    else:
        model = default_model()
        reason = f"no keys → default {model}"

    label = model  # downstream UI prettifies; logs use id directly
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
