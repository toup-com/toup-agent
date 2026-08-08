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


def test_openai_quota_error_reaches_the_openai_branch():
    """The OpenAI branch must keep working for genuine OpenAI quota
    errors — the Anthropic guard sits ABOVE it and must not swallow them.

    This asserted `platform.openai.com/account/billing` until 77a835e4
    ("no external purchase links in client-facing errors, App Review
    3.1.1") deliberately replaced that copy with the in-app Credits
    pointer. The assertion was never updated, so it has been red on main
    ever since — testing copy that the product had already, correctly,
    removed. What the test is actually FOR is the branch routing, so
    assert that instead, and pin the compliance property separately below.
    """
    from app.api.ws_chat import _friendly_error
    msg = _friendly_error(_Err(
        "Error code: 429 - insufficient_quota: You exceeded your current "
        "quota, please check your plan and billing details."
    ))
    # Reached the OpenAI/credits branch...
    assert "Toup credits" in msg
    # ...and NOT the Anthropic platform-outage branch above it, which is
    # the mis-routing this whole module exists to prevent.
    assert "temporarily unavailable" not in msg.lower()
    # ...and NOT the generic 429 branch below it, which would swallow a
    # billing error as a transient rate limit and tell the user to wait.
    assert "rate limit" not in msg.lower()


def test_no_client_facing_error_links_to_an_external_purchase_page():
    """App Review 3.1.1 anti-steering: on iOS the only legitimate purchase
    surface is the in-app Credits screen, so a billing/quota error must
    never point at the provider's console. This is the requirement that
    77a835e4 introduced and that the stale assertion above was silently
    working against.

    Swept across every branch that could plausibly mention money, because
    the regression is a one-line copy edit in any one of them.
    """
    from app.api.ws_chat import _friendly_error

    money_errors = [
        "Error code: 429 - insufficient_quota: You exceeded your current "
        "quota, please check your plan and billing details.",
        "openai quota exceeded",
        "Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits.",
        "billing hard limit reached",
    ]
    for raw in money_errors:
        msg = _friendly_error(_Err(raw))
        assert "platform.openai.com" not in msg, raw
        assert "console.anthropic.com" not in msg, raw
        assert "plans & billing" not in msg.lower(), raw


def test_control_the_subscription_branch_may_still_link_out():
    """Anti-vacuity control for the sweep above. The Claude *subscription*
    branch legitimately links to claude.ai/settings/usage — it is not a
    purchase surface for Toup credits, and it is how a user with their own
    Claude subscription unblocks themselves. If the sweep had been written
    as a blanket "no URLs anywhere" it would pass while asserting nothing
    about the actual compliance rule."""
    from app.api.ws_chat import _friendly_error
    msg = _friendly_error(_Err(
        "You're out of extra usage. Add more at claude.ai/settings/usage"
    ))
    assert "claude.ai/settings/usage" in msg


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


def test_bundle_auth_gate_shows_still_finishing_setup():
    """A brand-new signup whose container is reachable a hair before LLM
    activation lands gets the proxy's 'Missing token'/'Invalid token' (401)
    or 'Bundle subscription is not active' (403). _friendly_error must surface
    the honest, recoverable 'still finishing setup' state — NOT the misleading
    'Your API key is invalid' dead-end (the user has no key to fix)."""
    from app.api.ws_chat import _friendly_error
    for raw in (
        "proxy 401: Missing token",
        "proxy 401: Invalid token",
        "proxy 403: Bundle subscription is not active",
    ):
        msg = _friendly_error(_Err(raw))
        assert "finishing setup" in msg.lower(), f"expected setup copy for {raw!r}, got {msg!r}"
        assert "api key is invalid" not in msg.lower()


def test_genuine_byok_bad_key_still_maps_to_invalid_key():
    """The bundle-auth-gate branch must NOT swallow a real BYOK bad-key 401 —
    that should still tell the user to fix their key in Settings."""
    from app.api.ws_chat import _friendly_error

    class _AuthErr(Exception):
        status_code = 401

    msg = _friendly_error(_AuthErr("authentication_error: invalid x-api-key"))
    assert "api key is invalid" in msg.lower()
    assert "finishing setup" not in msg.lower()
