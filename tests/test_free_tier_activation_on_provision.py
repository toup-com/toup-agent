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
