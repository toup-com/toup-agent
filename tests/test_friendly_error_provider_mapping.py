"""Regression: _friendly_error must not mis-map an Anthropic credit error
to the OpenAI billing message.

Incident (2026-05-29): the platform's shared Claude account ran out of
credit. Every bundle user calling Claude (the old default model
claude-opus-4-7) got Anthropic's hard 400:

    "Your credit balance is too low to access the Anthropic API.
     Please go to Plans & Billing to upgrade or purchase credits."

_friendly_error's OpenAI branch matched the words "credit"/"balance"
and rendered "Add more credits at platform.openai.com/account/billing
• Or add a different API key in Settings" — wrong provider AND wrong
actor (a bundle user can't fix the platform's billing). These tests
pin the Anthropic-specific branch so the OpenAI copy never leaks for
an Anthropic credit error again.
"""
from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "test")


class _Err(Exception):
    pass


def test_anthropic_credit_error_does_not_show_openai_billing():
    from app.api.ws_chat import _friendly_error
    msg = _friendly_error(_Err(
        "anthropic upstream 400: Your credit balance is too low to access "
        "the Anthropic API. Please go to Plans & Billing to upgrade or "
        "purchase credits."
    ))
    # Must NOT route to the OpenAI billing copy.
    assert "platform.openai.com" not in msg
    assert "add a different API key" not in msg.lower()
    # Should signal a platform-side, transient condition.
    assert "temporarily unavailable" in msg.lower()


def test_openai_quota_error_still_maps_to_openai_billing():
    """The OpenAI branch must keep working for genuine OpenAI quota
    errors — the Anthropic guard sits ABOVE it and must not swallow them."""
    from app.api.ws_chat import _friendly_error
    msg = _friendly_error(_Err(
        "Error code: 429 - insufficient_quota: You exceeded your current "
        "quota, please check your plan and billing details."
    ))
    assert "platform.openai.com/account/billing" in msg


def test_claude_subscription_oauth_exhaustion_unchanged():
    """The pre-existing Claude *subscription* (OAuth) exhaustion branch
    must still fire for the CLI-token wording — distinct from the API
    console 'credit balance too low' billing error."""
    from app.api.ws_chat import _friendly_error
    msg = _friendly_error(_Err(
        "You're out of extra usage. Add more at claude.ai/settings/usage"
    ))
    assert "claude.ai/settings/usage" in msg


def test_default_canonical_agent_model_is_openai():
    """The canonical last-resort default model must be an OpenAI model so
    a brand-new tenant with no per-user override and no AGENT_MODEL env
    doesn't fall back onto the shared-Anthropic single point of failure."""
    from app.services import model_resolver
    assert model_resolver.is_openai_model(model_resolver._CANONICAL_AGENT_MODEL)
    # Anthropic stays available as the explicit-Claude fallback constant.
    assert model_resolver.is_claude_model(model_resolver._CANONICAL_ANTHROPIC_MODEL)


def test_default_model_resolves_to_openai_when_no_overrides(monkeypatch):
    """With no per-tenant column and settings.agent_model cleared, the
    resolver must return an OpenAI model (the new canonical default)."""
    from app.services import model_resolver
    from app.config import settings
    # Clear the settings override (local .env / env var) so the canonical
    # constant is what answers.
    monkeypatch.setattr(settings, "agent_model", "", raising=False)
    resolved = model_resolver.default_model(None)
    assert model_resolver.is_openai_model(resolved), (
        f"expected an OpenAI default, got {resolved!r}"
    )
