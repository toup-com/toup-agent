"""Regression for the 2026-05-24 ("Wake your agent up still 403's chat") bug.

The OnboardingShell's chromeless wake button at
``AgentPage.tsx``~3453 fires ``saveConfig({setup_completed:true, ...})``
+ ``navigate('/chat')``. It does NOT call ``/managed-agent/provision``.
The PUT ``/api/agent-setup/config`` handler therefore needs to be the
load-bearing place where Free-tier activation happens — otherwise the
env-sync inside that handler pushes the bridge an empty ``TOUP_TOKEN``
and every chat 403s.

Tests:
  1. Source-invariant: PUT /config must call ``activate_free_tier``
     (in the same function body) AFTER the WAKE-CLICK log marker and
     BEFORE the env-sync ``update_container_env`` call — so the
     minted ``connect_token`` is included in the env push.

  2. Functional (DB-backed): a fresh Free signup with
     ``bundle_status='none'`` + ``connect_token=NULL`` who hits PUT
     /config with ``setup_completed=true`` must end the request with
     ``bundle_status='active'`` and a minted ``connect_token``. Fails
     on main; passes with the fix.
"""

from __future__ import annotations

import inspect

import pytest
from httpx import AsyncClient
from sqlalchemy import select


def test_put_config_calls_activate_free_tier_before_env_sync():
    from app.api import agent_setup

    src = inspect.getsource(agent_setup.update_config)

    activate_idx = src.find("await activate_free_tier(")
    wake_log_idx = src.find("[WAKE-CLICK]")
    # The env-sync CALL site — `update_container_env(` matches the
    # invocation, not the import at the top of the function body.
    env_sync_idx = src.find("await update_container_env(")

    assert activate_idx != -1, (
        "PUT /config must call activate_free_tier when setup_completed "
        "flips True — otherwise the bridge env push goes out with empty "
        "TOUP_TOKEN for fresh Free signups."
    )
    assert wake_log_idx != -1, "[WAKE-CLICK] log marker missing"
    assert env_sync_idx != -1, "update_container_env call missing"

    assert wake_log_idx < activate_idx, (
        "activate_free_tier must run AFTER the WAKE-CLICK log so the "
        "log fires for every Wake click (telemetry contract)."
    )
    assert activate_idx < env_sync_idx, (
        "activate_free_tier must run BEFORE update_container_env so the "
        "freshly-minted connect_token is included in the env push."
    )


def test_put_config_skips_activation_for_paid_plan_awaiting_stripe():
    from app.api import agent_setup
    src = inspect.getsource(agent_setup.update_config)
    # Same defense-in-depth signal as the provision endpoint —
    # users with a non-free credit_balances.plan_id who haven't yet
    # completed Stripe must NOT get auto-activated as Free.
    assert "is_paid_plan_awaiting_stripe" in src, (
        "PUT /config must compute the paid-plan signal so users with a "
        "non-free plan awaiting Stripe activation are NOT auto-activated."
    )


def test_put_config_skips_activation_when_already_active():
    from app.api import agent_setup
    src = inspect.getsource(agent_setup.update_config)
    # The branch guard — activate only runs when bundle_status is
    # NOT already active/cancelling. Users on a paid sub or already-
    # activated free should be a no-op.
    assert 'bundle_status not in ("active", "cancelling")' in src, (
        "PUT /config must short-circuit activation for users whose "
        "bundle_status is already active or cancelling."
    )


@pytest.mark.asyncio
async def test_put_config_activates_free_tier_on_setup_completed(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_user_id: str,
):
    """Fresh Free signup hitting PUT /config with setup_completed=True
    must end the request with bundle_status='active' and a minted
    connect_token + llm_token_hash. Fails on main (no activation in
    the handler); passes with the fix.

    update_container_env is mocked because the test app has no bridge.
    """
    from app.db import AgentConfig, async_session_maker
    from app.config import settings
    from unittest.mock import patch
    from app.services import docker_host_service

    # Fresh AgentConfig — what a Free signup looks like just before the
    # Wake click: managed hosting, default llm_mode, bundle_status='none',
    # no connect_token.
    async with async_session_maker() as db:
        cfg = AgentConfig(
            user_id=test_user_id,
            hosting_mode="managed",
            llm_mode="manual",
            bundle_status="none",
            setup_step=4,
            setup_type="auto",
            setup_completed=False,
        )
        db.add(cfg)
        await db.commit()

    # The handler will try to push env to the bridge — mock the call
    # so the test stays deterministic. The DB state changes we care
    # about happen BEFORE update_container_env runs, so the mock can
    # return None without affecting the assertion.
    async def _fake_update_container_env(*_a, **_kw):
        return None

    settings.managed_hosting_enabled = True
    try:
        with patch.object(
            docker_host_service, "update_container_env",
            side_effect=_fake_update_container_env,
        ):
            resp = await client.put(
                "/api/agent-setup/config",
                headers=auth_headers,
                json={
                    "setup_step": 4,
                    "setup_type": "auto",
                    "hosting_mode": "managed",
                    "db_mode": "local_postgres",
                    "llm_mode": "bundle",
                    "setup_completed": True,
                    "onboarding_completed": True,
                },
            )
        assert resp.status_code == 200, (
            f"PUT /config failed: {resp.status_code}: {resp.text}"
        )
    finally:
        settings.managed_hosting_enabled = False

    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == test_user_id)
        )).scalar_one()

    assert cfg.bundle_status == "active", (
        f"activate_free_tier did not flip bundle_status; "
        f"got {cfg.bundle_status!r} — fresh Free signups will chat into "
        f"a 403 from the bundle proxy."
    )
    assert cfg.connect_token, "connect_token must be minted on Wake click"
    assert cfg.llm_token_hash, (
        "llm_token_hash must be set — the bundle proxy gate keys on it"
    )
    assert cfg.llm_mode == "bundle"
