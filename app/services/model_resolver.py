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
# Canonical default is OpenAI: bundle Anthropic calls share one platform
# Claude account (single point of failure when its credit runs out), while
# OpenAI calls use each user's own per-tenant project key. See the
# `agent_model` comment in config.py for the 2026-05-29 incident.
# `_CANONICAL_ANTHROPIC_MODEL` stays Claude — it's the explicit-Claude
# fallback, only reached when a user/operator deliberately picks Anthropic.
#
# 2026-08-07: moved gpt-5.5 → gpt-5.6-terra. G1 passed on measured OpenAI
# organization billing (docs/audits/2026-08-g1-cost-and-latency.md §8):
# terra is 50-55% cheaper per token on every token class, p50 -17.5%,
# p95 -54.1%, and marginally more reliable (2.1% vs 2.6% error rate).
# `_CANONICAL_FALLBACK_MODEL` deliberately stays gpt-4o: it is the
# cross-provider fallback and keeps its long-proven chat wire (see
# `wire_api_for` below).
_CANONICAL_AGENT_MODEL = "gpt-5.6-terra"
_CANONICAL_FALLBACK_MODEL = "gpt-4o"
_CANONICAL_ANTHROPIC_MODEL = "claude-opus-4-7"
_CANONICAL_OPENAI_MODEL = "gpt-5.6-terra"


# ── G1 chat-model guard (no-cached-rate tier) ─────────────────────────
# OpenAI's "-pro" reasoning tier has NO cached-input rate (gpt-5.5-pro is
# $30/$180 flat) — routing the chat/agent hot path there forfeits the
# entire prompt-cache discount the stable-prefix layout exists to earn.
# The gate (docs/audits/2026-07-g1-model-gate.md) bars these ids from
# ever resolving as the chat/agent model: `default_model` logs and falls
# through to the next candidate instead of honouring the override.
_NO_CACHED_INPUT_RATE_MODELS = frozenset({"gpt-5.5-pro", "o1-pro", "o3-pro"})


def has_cached_input_rate(model: str | None) -> bool:
    """False for models the provider bills with NO cached-input discount.

    Matches the explicit denylist plus any OpenAI id ending in "-pro" so
    future pro tiers (e.g. a gpt-5.6-pro) stay barred without a code
    change. Unknown/Claude/empty ids return True — this guard exists only
    to keep the known no-cache tier off the hot chat path, not to
    validate arbitrary model ids.
    """
    if not model:
        return True
    m = model.lower().strip()
    if m in _NO_CACHED_INPUT_RATE_MODELS:
        return False
    if is_openai_model(m) and m.endswith("-pro"):
        return False
    return True


# ── Wire selection (derived, NOT an independent operator choice) ───────
# gpt-5.6-* returns 400 on /v1/chat/completions whenever function tools
# are present ("use /v1/responses or set reasoning_effort to 'none'" —
# canary abort 2026-07-28). Every agent turn sends function tools, so for
# that family the wire is not a preference: it is the only wire that works.
#
# `settings.openai_wire_api` used to be the sole selector, which meant the
# model and the wire were two independent settings that had to be flipped
# together. Setting one without the other hard-400s every turn on that
# container, and nothing about a half-flipped container looks unhealthy —
# it just fails every request. This codebase has lost fleets to exactly
# that shape before (the four bind gates, the append-only bridge env).
#
# So the wire is DERIVED from the model. A container can no longer land in
# the broken half-flipped state, whether the model arrives from the code
# default, a Railway env var, per-container bridge env, or a per-tenant
# agent_config column. The setting still governs every other family, so an
# operator can still move gpt-4o/gpt-5.5 onto the Responses wire
# deliberately.
_RESPONSES_ONLY_PREFIXES = ("gpt-5.6",)


def requires_responses_wire(model: str | None) -> bool:
    """True for OpenAI families that reject /v1/chat/completions outright
    when function tools are present. Not a preference — a hard provider
    constraint."""
    if not model:
        return False
    return model.lower().strip().startswith(_RESPONSES_ONLY_PREFIXES)


def wire_api_for(model: str | None) -> str:
    """Which OpenAI wire to use for `model`: 'chat' | 'responses'.

    Families that mandate the Responses API get it regardless of settings.
    Everything else honours `settings.openai_wire_api` (default 'chat'),
    so the gpt-4o fallback path keeps the wire it has always used.
    """
    if requires_responses_wire(model):
        return "responses"
    configured = (_settings_attr("openai_wire_api") or "chat")
    return str(configured).strip().lower() or "chat"


# ── Default-resolution functions ──────────────────────────────────────


def default_model(agent_config: object | None = None) -> str:
    """The primary agent model.

    Resolves: per-tenant `agent_config.agent_model` →
              `settings.agent_model` →
              canonical default.

    G1 guard: a candidate with no cached-input rate (gpt-5.5-pro et al.)
    is never honoured — it is logged and resolution falls through to the
    next layer (ultimately the canonical default, which always has a
    cached rate).
    """
    for source, candidate in (
        ("agent_config.agent_model", _cfg_attr(agent_config, "agent_model")),
        ("settings.agent_model", _settings_attr("agent_model")),
    ):
        if not candidate:
            continue
        if not has_cached_input_rate(candidate):
            logger.error(
                "default_model: %r (from %s) has no cached-input rate and is "
                "barred as the chat/agent model (G1 gate) — falling back",
                candidate, source,
            )
            continue
        return candidate
    return _CANONICAL_AGENT_MODEL


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
    """False for OpenAI reasoning models + gpt-5.x family (incl. the
    gpt-5.6-* tiers) — they only accept
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


def uses_max_completion_tokens(model: str | None) -> bool:
    """True for OpenAI reasoning models + gpt-5.x family (incl. the
    gpt-5.6-* tiers) — they reject the
    classic `max_tokens` parameter with HTTP 400 ("Unsupported parameter:
    'max_tokens' is not supported with this model. Use 'max_completion_tokens'
    instead.") and require `max_completion_tokens` instead.

    Callers should pass the OpenAI client kwarg conditionally:
        kwarg = "max_completion_tokens" if uses_max_completion_tokens(model)
                else "max_tokens"
    """
    if not model:
        return False
    m = model.lower()
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith("gpt-5"):
        return True
    return False


def is_reasoning_model(model: str | None) -> bool:
    """True for OpenAI models that do internal chain-of-thought reasoning
    (o-series + gpt-5 family, incl. the gpt-5.6-* tiers).

    These models silently spend output-token budget on hidden reasoning
    tokens BEFORE emitting any visible text. Burned the routine path: a
    13k-token email summary with `max_completion_tokens=1200` exhausted
    its entire budget on reasoning and returned an empty `message.content`
    — the call logged status=ok with 1200 output tokens but zero visible
    output. Callers running cheap summarisation jobs should pass
    `reasoning_effort="minimal"` to keep the reasoning portion tiny and
    leave the budget for the actual summary."""
    if not model:
        return False
    m = model.lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5"))


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
