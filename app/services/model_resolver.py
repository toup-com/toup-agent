"""
Model resolver — single source of truth for default LLM model selection.

Every default model identifier in the agent + auto-builder resolves through
the functions in this module. To upgrade the platform-wide default, change
`settings.agent_model` (env var `AGENT_MODEL`). No other file should
hardcode a model string for *default* selection (intentional inline pins
for cheap-summary / vision / TTS jobs are exempt — they're per-task pins,
not defaults).

Resolution chain for every `default_*_model()` function:

    1. Per-tenant override — `agent_config.<column>` if non-null.
    2. Settings env override — `settings.<field>` if set (env wins).
    3. Hard-coded canonical default — the private constants below.

The canonical constants are the *only* literal model strings outside the
data registries (context window, pricing) and the intentional inline
pins. Phase 4 grep proves this.

See:
- docs/llm-model-refactor-plan.md for the architecture.
- docs/llm-model-audit.md for the original drift inventory.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


# ── Canonical defaults ────────────────────────────────────────────────
# Module-private. The ONLY literal model strings outside data tables.
# Settings + per-tenant overrides take precedence; these are the
# last-resort fallback when nothing else is configured.
_CANONICAL_AGENT_MODEL = "claude-opus-4-7"
_CANONICAL_FALLBACK_MODEL = "gpt-5.5"
_CANONICAL_ANTHROPIC_MODEL = "claude-opus-4-7"
_CANONICAL_OPENAI_MODEL = "gpt-5.5"


# ── Default-resolution functions ──────────────────────────────────────


def default_model(agent_config: object | None = None) -> str:
    """The primary agent model.

    Resolves: per-tenant `agent_config.agent_model` →
              `settings.agent_model` →
              canonical default.
    """
    return (
        _cfg_attr(agent_config, "agent_model")
        or _settings_attr("agent_model")
        or _CANONICAL_AGENT_MODEL
    )


def default_fallback_model(agent_config: object | None = None) -> str:
    """The cross-provider fallback model.

    Used by `agent_runner` when the primary call fails AND the user has
    a key for the other provider configured.
    """
    return (
        _settings_attr("agent_fallback_model")
        or _CANONICAL_FALLBACK_MODEL
    )


def app_builder_planner_model(agent_config: object | None = None) -> str:
    """Auto-builder Planner sub-agent model.

    Used for conversational reasoning (research, requirements gathering,
    plan synthesis, modify-app analysis). Resolves via per-tenant column
    → settings env → falls through to `default_model`.

    Null at every layer means "share the agent's default model" — this is
    the documented and tested behaviour, not an oversight.
    """
    return (
        _cfg_attr(agent_config, "app_builder_planner_model")
        or _settings_attr("app_builder_planner_model")
        or default_model(agent_config)
    )


def app_builder_builder_model(agent_config: object | None = None) -> str:
    """Auto-builder Builder sub-agent model.

    Used for code generation, file writes, and JSON repair. Resolves via
    per-tenant column → settings env → falls through to `default_model`.
    """
    return (
        _cfg_attr(agent_config, "app_builder_builder_model")
        or _settings_attr("app_builder_builder_model")
        or default_model(agent_config)
    )


def default_anthropic_model(agent_config: object | None = None) -> str:
    """The Anthropic-specific default.

    If `default_model` resolves to a Claude model, that wins. Otherwise
    fall back to the canonical Claude default — used by the cross-provider
    fallback path when the primary is OpenAI and we need a Claude fallback.
    """
    primary = default_model(agent_config)
    if is_claude_model(primary):
        return primary
    return (
        _settings_attr("anthropic_model")  # deprecated but kept for back-compat
        or _CANONICAL_ANTHROPIC_MODEL
    )


def default_openai_model(agent_config: object | None = None) -> str:
    """The OpenAI-specific default.

    Symmetric to `default_anthropic_model`. If the primary is OpenAI, use
    it; otherwise fall back to the canonical OpenAI choice (used as the
    cross-provider fallback when primary is Claude).
    """
    primary = default_model(agent_config)
    if is_openai_model(primary):
        return primary
    return _CANONICAL_OPENAI_MODEL


# ── Per-model facts (registry helpers) ────────────────────────────────


def context_window_for(model: str) -> int:
    """Token-count budget for a given model.

    Wraps the canonical `MODEL_CONTEXT_WINDOWS` registry in
    `app.agent.context_manager`. Returns the registry's default for
    unknown models (with a warning) so callers don't have to handle the
    miss case.
    """
    if not model:
        return _safe_default_window()

    try:
        from app.agent.context_manager import (
            MODEL_CONTEXT_WINDOWS,
            DEFAULT_CONTEXT_WINDOW,
        )
    except ImportError:
        return _safe_default_window()

    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]

    logger.warning(
        "context_window_for: unknown model %r, returning default %d",
        model, DEFAULT_CONTEXT_WINDOW,
    )
    return DEFAULT_CONTEXT_WINDOW


def pricing_for(model: str) -> Optional[Tuple[float, float]]:
    """Per-1k-token pricing for a model: returns (input_per_1k, output_per_1k)
    in USD, or None if the model isn't in the pricing registry.

    The pricing dict lives in `settings.pricing_per_1k`. Three duplicate
    pricing dicts exist in the codebase (config / token_tracker /
    model_session) — those will be consolidated in the post-refactor
    cleanup commit.
    """
    if not model:
        return None
    table = getattr(settings, "pricing_per_1k", {}) or {}
    entry = table.get(model)
    if not entry:
        return None
    try:
        return float(entry["input"]), float(entry["output"])
    except (KeyError, TypeError, ValueError):
        return None


# ── Provider classification ───────────────────────────────────────────


def is_claude_model(model: str | None) -> bool:
    """True when the model id belongs to Anthropic Claude.

    Matches by prefix so dated model IDs (e.g., `claude-sonnet-4-5-20250929`)
    classify correctly.
    """
    if not model:
        return False
    return model.lower().startswith("claude")


def is_openai_model(model: str | None) -> bool:
    """True when the model id belongs to OpenAI.

    GPT family + reasoning models (o1, o3, o4) are all OpenAI.
    """
    if not model:
        return False
    m = model.lower()
    return m.startswith(("gpt", "o1-", "o3-", "o4-")) or m in {"o1", "o3", "o4"}


def supports_custom_temperature(model: str | None) -> bool:
    """False for OpenAI reasoning models + gpt-5.x family — they only accept
    temperature=1 (the default) and reject any explicit value with HTTP 400.
    Callers should omit the `temperature` kwarg from the API request entirely
    when this returns False.
    """
    if not model:
        return True
    m = model.lower()
    if m.startswith(("o1", "o3", "o4")):
        return False
    if m.startswith("gpt-5"):
        return False
    return True


# ── OAuth token detection ─────────────────────────────────────────────


def is_oauth_anthropic_token(key: str | None) -> bool:
    """True for Claude Code OAuth tokens (`sk-ant-oat*`).

    Used by `bundle_client.make_anthropic_client` to decide whether to
    add the Claude Code beta + CLI identity headers. Bare API keys
    (`sk-ant-api*`) and OpenAI keys (`sk-proj-*`, `sk-*`) all return False.
    """
    if not key:
        return False
    return "sk-ant-oat" in key


# ── Internal helpers ──────────────────────────────────────────────────


def _cfg_attr(agent_config: object | None, name: str) -> str | None:
    """Read an attribute from an `AgentConfig`-like object, returning None
    when missing or empty.

    Uses `getattr` with a default so the resolver works even if the
    attribute hasn't been added to the SQLAlchemy model yet (e.g., during
    the transitional deploy window between commit A and commit E).
    """
    if agent_config is None:
        return None
    value = getattr(agent_config, name, None)
    if not value:
        return None
    return str(value).strip() or None


def _settings_attr(name: str) -> str | None:
    """Read a string field from settings, returning None when absent or
    empty. Mirrors `_cfg_attr` so settings-not-yet-added cases don't
    explode during the transitional period."""
    value = getattr(settings, name, None)
    if not value:
        return None
    return str(value).strip() or None


def _safe_default_window() -> int:
    """Used when context_manager isn't importable for whatever reason
    (early-boot, test environment without full app context)."""
    return 128_000
