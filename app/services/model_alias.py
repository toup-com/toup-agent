"""Public model-identity aliasing — defense-in-depth for the white-label story.

The agent is presented to users as *their* agent, never as "Claude" or "GPT"
(see agent_runner.py identity_anchor + docs/security/audit-2026.md MI-2). But
the real provider model id leaks onto several user-facing surfaces (usage
summary, job cards, message payloads, Telegram command output) because it is
serialized straight from ``Message.model_used`` / ``job.model``.

This module gives those serializers ONE place to convert a real model id to a
neutral tier label, and to scrub provider names out of free text. It is a
FILTER-layer defense: it does not (and cannot) stop a tenant with shell access
to their own container from reading the real model — it closes the casual and
external-observer disclosure only.

Callers gate on ``settings.security_leak_filter`` so this is a no-op until the
operator flips the flag (the raw value still exists behind it for admins).
"""
from __future__ import annotations

import re

# Neutral, user-facing tier labels. Deliberately vendor-free.
TIER_FAST = "Fast"
TIER_BALANCED = "Balanced"
TIER_DEEP = "Deep"

# Substrings that map a real model id to a tier. Order matters: first hit wins.
# Kept broad so unseen future ids still land on a sane tier rather than leaking.
_DEEP = ("opus", "gpt-5.5", "gpt-5.4", "o1", "o3", "-pro", "reasoning")
_FAST = ("haiku", "mini", "nano", "flash", "gpt-4o-mini", "small", "lite", "turbo")

# Provider / model tokens to scrub from free-text output (case-insensitive).
# Word-boundaried so we don't mangle unrelated words.
_PROVIDER_TOKENS = [
    r"claude(?:[\s-]?[\w.]+)?",
    r"gpt-[\w.]+",
    r"\bgpt\b",
    r"\bo[134](?:-[\w.]+)?\b",
    r"sonnet",
    r"opus",
    r"haiku",
    r"anthropic",
    r"openai",
    r"gemini",
    r"mistral",
    r"deepseek",
    r"grok",
    r"\bxai\b",
    r"llama",
]
_PROVIDER_RE = re.compile("|".join(_PROVIDER_TOKENS), re.IGNORECASE)


def public_model_label(real_id: str | None) -> str:
    """Map a real provider model id to a neutral tier label.

    Returns ``TIER_BALANCED`` for unknown/empty ids so nothing ever falls
    through to a raw provider name. Idempotent: an already-neutral label
    (Fast/Balanced/Deep) is returned unchanged.
    """
    if not real_id:
        return TIER_BALANCED
    s = str(real_id).strip()
    if s in (TIER_FAST, TIER_BALANCED, TIER_DEEP):
        return s
    low = s.lower()
    if any(tok in low for tok in _DEEP):
        return TIER_DEEP
    if any(tok in low for tok in _FAST):
        return TIER_FAST
    return TIER_BALANCED


def public_provider_label(_provider: str | None) -> str:
    """Collapse a real provider name (anthropic/openai/…) to a neutral bucket.

    We surface a single neutral label rather than the vendor. Numbers/costs
    that accompany it in usage dashboards are unaffected — only identity is
    neutralized.
    """
    return "Toup"


def scrub_provider_names(text: str | None) -> str:
    """Replace provider/model tokens in free text with a neutral placeholder.

    For command output / error bodies where a full re-map isn't practical.
    Returns the input unchanged when it's empty.
    """
    if not text:
        return text or ""
    return _PROVIDER_RE.sub("the model", text)


# Infrastructure / stack tokens to neutralize in diagnostic text that reaches
# the model (e.g. the `doctor` tool). These are the "how Toup is built" details
# the identity anchor says are proprietary (docs/security/audit-2026.md MI-5).
# Word-boundaried, case-insensitive; replaced with a neutral "[component]".
_STACK_TOKENS = [
    r"docker", r"fastapi", r"uvicorn", r"sqlalchemy", r"alembic", r"pydantic",
    r"playwright", r"patchright", r"chromium", r"pgvector", r"postgresql",
    r"postgres", r"sqlite", r"redis", r"nginx", r"caddy", r"railway",
    r"supabase", r"baileys", r"libreoffice", r"hermes", r"expo",
]
_STACK_RE = re.compile(r"\b(?:" + "|".join(_STACK_TOKENS) + r")\b", re.IGNORECASE)


def scrub_stack_terms(text: str | None) -> str:
    """Neutralize infrastructure/stack names in free text with "[component]".

    Companion to scrub_provider_names for diagnostic output (the `doctor`
    tool) that would otherwise recite the architecture into model context.
    Returns the input unchanged when empty.
    """
    if not text:
        return text or ""
    return _STACK_RE.sub("[component]", text)
