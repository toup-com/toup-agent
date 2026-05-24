"""
PR onboarding-v2 hotfix #2 — `activate_free_tier` runs on "Wake your agent up".

Validates the load-bearing wiring that makes the Free-tier flow actually
work end-to-end:

  1. The shared `activate_free_tier` service exists and is importable.
  2. The `managed_agents.provision` endpoint calls it BEFORE the
     legacy `bundle_status` gate (otherwise Free users would 402).
  3. The provision call uses `recreate=True` so the existing container
     actually picks up the fresh env after activation.

Source-level checks — no DB required. The full round-trip against a
live container is exercised by the staging smoke test in the PR
description.
"""

from __future__ import annotations

import inspect

from app.services import free_tier_activation
from app.api import managed_agents


def test_activate_free_tier_service_exists():
    assert hasattr(free_tier_activation, "activate_free_tier"), (
        "Hotfix moved the activation logic into a shared service so the "
        "provision endpoint can re-use it. The function must still exist."
    )


def test_activate_free_tier_signature():
    sig = inspect.signature(free_tier_activation.activate_free_tier)
    params = set(sig.parameters)
    assert "db" in params
    assert "user_id" in params
    # force_env_push is the knob the provision endpoint uses to opt OUT
    # of the redundant env push (recreate=True below handles env refresh).
    assert "force_env_push" in params


def test_activate_free_tier_mints_connect_token_idempotently():
    src = inspect.getsource(free_tier_activation.activate_free_tier)
    # connect_token only minted when NULL — re-runs preserve the
    # existing token so the agent's TOUP_TOKEN doesn't drift.
    assert "if not cfg.connect_token" in src
    assert "f\"toup_ct_{secrets.token_urlsafe(32)}\"" in src


def test_activate_free_tier_does_not_clobber_paid_users():
    src = inspect.getsource(free_tier_activation.activate_free_tier)
    # bundle_status flip is GUARDED on `not was_already_active` so
    # users on a paid plan ('active' or 'cancelling') keep their state.
    assert "was_already_active" in src
    assert "if not was_already_active" in src


def test_provision_calls_activate_before_gate():
    src = inspect.getsource(managed_agents.provision)
    # The activate call must appear in the source BEFORE the
    # "Bundle subscription is not active" 402 raise — otherwise Free
    # users would always 402 because their bundle_status is 'none'
    # at the moment provision_container is called.
    activate_idx = src.find("activate_free_tier(")
    gate_idx = src.find("Bundle subscription is not active")
    assert activate_idx != -1, "provision endpoint must import activate_free_tier"
    assert gate_idx != -1, "the 402 gate should still be in place for safety"
    assert activate_idx < gate_idx, (
        "activate_free_tier must run BEFORE the bundle_status gate or "
        "Free users will 402."
    )


def test_provision_uses_recreate_true():
    src = inspect.getsource(managed_agents.provision)
    # recreate=True is the load-bearing fix — without it, provision's
    # idempotency early-returns and the agent keeps stale env.
    assert "recreate=True" in src, (
        "provision_container must be called with recreate=True so the "
        "existing prewarm container picks up the fresh env after "
        "activate_free_tier flips llm_mode + connect_token."
    )


def test_provision_force_env_push_false():
    # The provision endpoint calls activate_free_tier with
    # `force_env_push=False` because it's about to recreate the
    # container itself a few lines below — a redundant push would
    # waste ~200ms and introduce an extra failure point.
    src = inspect.getsource(managed_agents.provision)
    assert "force_env_push=False" in src


def test_provision_does_not_gate_activation_on_credit_balance_existence():
    """Regression guard for the 2026-05-23 → 2026-05-24 production bug.

    PR #102 originally guarded activate_free_tier on the presence of a
    credit_balances row. That row is lazy-created on first credit
    charge — NOT at signup — so every fresh Free user fell through the
    gate and provisioned a container with empty TOUP_TOKEN. The chat
    UI surfaced "Error: Something went wrong" on every message.

    The current gate must drive off ``credit_balances.plan_id``
    (paid vs. free), NOT off the row existing at all. This test fails
    if anyone re-introduces the original existence-check pattern.
    """
    src = inspect.getsource(managed_agents.provision)
    # The buggy pattern: `on_credit_system = ... scalar_one_or_none() is not None`
    # followed by `if on_credit_system:` gating the activation call.
    assert "on_credit_system" not in src, (
        "provision endpoint must NOT gate activate_free_tier on the "
        "existence of a credit_balances row — that row is lazy-created "
        "on first charge, not at signup. Use credit_balances.plan_id "
        "to distinguish Free vs. paid-awaiting-Stripe."
    )
    # The correct pattern keys on plan_id.
    assert "plan_id" in src and "free" in src, (
        "provision endpoint must distinguish Free users from paid-plan "
        "users via credit_balances.plan_id."
    )


def test_activate_free_tier_provisions_openai_project():
    """Every Free signup must get their own OpenAI project + service-account
    key — same per-user attribution the paid-bundle path has always given.

    Regression for the 2026-05-24 finding: post-credit-system rollout, only
    paid Stripe-active users had `_provision_openai_project_if_needed`
    wired in (billing.py + vps.py webhook handlers). Free signups were
    silently using the platform master OpenAI key with no per-project
    usage attribution in the operator's OpenAI dashboard.

    The activation service is the single funnel point for the Free path,
    so the provisioning must happen here.
    """
    src = inspect.getsource(free_tier_activation.activate_free_tier)
    assert "_provision_openai_project_if_needed" in src, (
        "activate_free_tier must call _provision_openai_project_if_needed "
        "so Free signups get a per-user OpenAI project + key, matching "
        "what paid users get on Stripe activation."
    )


def test_provision_skips_activation_for_paid_plan_awaiting_stripe():
    """The 402 gate from the 2026-04-27 Arshia incident MUST still
    fire for users with a paid plan selected (credit_balances.plan_id
    != 'free') but bundle_status not yet active. Auto-activating them
    as Free would bypass Stripe and let them provision without paying.
    """
    src = inspect.getsource(managed_agents.provision)
    # The new logic computes a paid-plan signal and skips activation
    # when it's true. The exact variable name is part of the contract
    # so the gate is greppable.
    assert "is_paid_plan_awaiting_stripe" in src, (
        "provision endpoint must compute a paid-plan signal so users "
        "with a non-free plan_id awaiting Stripe activation are NOT "
        "auto-activated as Free."
    )
