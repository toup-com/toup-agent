"""
T0b — Agent API key rotation tests.

Two layers, mirroring T0c:

  1. Service-level unit tests against `rotate_agent_api_key` directly.
     The bridge call is monkey-patched (we don't want a real Contabo
     here), as is the `_verify_new_key` health poll. Covers the atomic
     contract: bridge fail → DB rollback, verify fail → DB rollback +
     bridge restore, success → audit row + new key returned.

  2. Endpoint-level tests against `POST /api/agent/rotate-key`. Covers
     the auth chain (JWT + sensitive-action token + replay defense),
     the feature flag, and the run_mode=agent guard.

Both layers skip the parts that need real Postgres (User.is_canary
partial unique index — see test-postgres-fixture follow-up). Locally
on SQLite the unit tests skip; on CI Postgres they all run.
"""

from __future__ import annotations

import os
import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    AgentConfig,
    AgentSecurityEvent,
    EVENT_AGENT_KEY_ROTATED,
    EVENT_AGENT_KEY_ROTATION_FAILED,
    ManagedContainer,
    User,
)
from app.services.agent_key_rotation import (
    AgentKeyRotationError,
    rotate_agent_api_key,
)


# ─── SQLite is fine here — every test seeds at most ONE user, so the
# `is_canary` partial-unique-index that bit T0c does not collide.


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def alice_with_managed_container() -> dict:
    """Seed a User + AgentConfig + ManagedContainer for one tenant.

    Returns: {"user_id", "agent_key", "agent_url"}.
    """
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid,
            email=f"alice-{uid[:8]}@example.com",
            hashed_password="x",
            name="Alice",
        ))
        await db.flush()
        cfg = AgentConfig(
            user_id=uid,
            hosting_mode="managed",
            agent_api_key="old_key_" + uuid.uuid4().hex,
            agent_url=f"https://agent-{uid[:8]}.agents.toup.ai",
        )
        db.add(cfg)
        container = ManagedContainer(
            id=str(uuid.uuid4()),
            user_id=uid,
            container_name=f"toup-agent-{uid[:8]}",
            host_port=8001,  # NOT NULL on the model — placeholder for tests
            db_name=f"toup_agent_{uid[:8]}",  # NOT NULL on the model
            status="running",
        )
        db.add(container)
        await db.commit()
        return {
            "user_id": uid,
            "agent_key": cfg.agent_api_key,
            "agent_url": cfg.agent_url,
        }


@pytest.fixture
def patch_bridge_ok(monkeypatch):
    """Replace `provision_container` with a no-op that succeeds. Used by
    the happy-path and verify-fail tests."""
    calls = []

    async def fake_provision(db, user_id, agent_config=None, recreate=False):
        calls.append({
            "user_id": user_id,
            "recreate": recreate,
            "key_at_call": agent_config.agent_api_key if agent_config else None,
        })
        return None

    monkeypatch.setattr(
        "app.services.agent_key_rotation.provision_container",
        fake_provision,
        raising=False,
    )
    # Also patch the import-from-inside path (the service uses
    # `from app.services.docker_host_service import provision_container`
    # at call time, so we have to patch that target too).
    monkeypatch.setattr(
        "app.services.docker_host_service.provision_container",
        fake_provision,
    )
    return calls


@pytest.fixture
def patch_bridge_fails(monkeypatch):
    """Make `provision_container` raise. The rotation flow must roll
    back the DB and attempt to restore the old env on the bridge."""
    calls = []

    async def fake_provision(db, user_id, agent_config=None, recreate=False):
        calls.append({
            "user_id": user_id,
            "recreate": recreate,
            "key_at_call": agent_config.agent_api_key if agent_config else None,
        })
        # First call (rotation) fails; subsequent calls (rollback)
        # succeed so the user is not stranded.
        if len(calls) == 1:
            raise RuntimeError("simulated bridge 502")
        return None

    monkeypatch.setattr(
        "app.services.docker_host_service.provision_container",
        fake_provision,
    )
    return calls


@pytest.fixture
def patch_verify_ok(monkeypatch):
    """Verify-health succeeds immediately (happy path)."""
    async def fake_verify(agent_url, new_key):
        return True

    monkeypatch.setattr(
        "app.services.agent_key_rotation._verify_new_key", fake_verify,
    )


@pytest.fixture
def patch_verify_timeout(monkeypatch):
    """Verify-health times out — should trigger DB rollback + restore."""
    async def fake_verify(agent_url, new_key):
        return False

    monkeypatch.setattr(
        "app.services.agent_key_rotation._verify_new_key", fake_verify,
    )


# ─── Service: happy path ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rotation_happy_path(
    alice_with_managed_container, patch_bridge_ok, patch_verify_ok,
):
    """End-to-end success: new key is returned, DB has new key, audit
    row is `agent_key_rotated`."""
    alice = alice_with_managed_container
    old_key = alice["agent_key"]

    async with async_session_maker() as db:
        result = await rotate_agent_api_key(db, alice["user_id"])

    assert result.new_agent_api_key != old_key
    assert len(result.new_agent_api_key) >= 32, "key must satisfy min-length contract"
    assert result.key_prefix == result.new_agent_api_key[:8]

    # DB committed with new key
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == alice["user_id"])
        )).scalar_one()
        assert cfg.agent_api_key == result.new_agent_api_key

        # Exactly one audit row, type=agent_key_rotated
        events = (await db.execute(
            select(AgentSecurityEvent)
            .where(AgentSecurityEvent.user_id == alice["user_id"])
        )).scalars().all()
        assert len(events) == 1
        assert events[0].event_type == EVENT_AGENT_KEY_ROTATED
        assert events[0].metadata_json is not None
        assert old_key[:8] in events[0].metadata_json
        assert result.new_agent_api_key[:8] in events[0].metadata_json

    # Bridge was called exactly once with recreate=True
    assert len(patch_bridge_ok) == 1
    assert patch_bridge_ok[0]["recreate"] is True
    # And the bridge saw the NEW key (DB write happened before bridge call)
    assert patch_bridge_ok[0]["key_at_call"] == result.new_agent_api_key


# ─── Service: bridge failure rolls back ───────────────────────────────


@pytest.mark.asyncio
async def test_bridge_failure_rolls_back_old_key(
    alice_with_managed_container, patch_bridge_fails, patch_verify_ok,
):
    """Bridge recreate fails → DB must reflect old key, audit row is
    `agent_key_rotation_failed`, bridge gets a second call to restore."""
    alice = alice_with_managed_container
    old_key = alice["agent_key"]

    async with async_session_maker() as db:
        with pytest.raises(AgentKeyRotationError, match="Bridge recreate failed"):
            await rotate_agent_api_key(db, alice["user_id"])

    # DB still has the OLD key (rollback worked)
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == alice["user_id"])
        )).scalar_one()
        assert cfg.agent_api_key == old_key, (
            "rollback must restore the old key"
        )

        # Audit row is the failure type
        events = (await db.execute(
            select(AgentSecurityEvent)
            .where(AgentSecurityEvent.user_id == alice["user_id"])
        )).scalars().all()
        assert len(events) == 1
        assert events[0].event_type == EVENT_AGENT_KEY_ROTATION_FAILED

    # Bridge called twice: once for rotation (failed), once for rollback
    # restore (succeeded).
    assert len(patch_bridge_fails) == 2
    # First call carries the new key (the rotation attempt).
    # Second call carries the OLD key (the restore).
    assert patch_bridge_fails[1]["key_at_call"] == old_key


# ─── Service: verify timeout rolls back ───────────────────────────────


@pytest.mark.asyncio
async def test_verify_timeout_rolls_back_old_key(
    alice_with_managed_container, patch_bridge_ok, patch_verify_timeout,
):
    """Bridge succeeds but new container never responds to /agent/health
    → rollback. Old key remains valid, failure audit row written."""
    alice = alice_with_managed_container
    old_key = alice["agent_key"]

    async with async_session_maker() as db:
        with pytest.raises(AgentKeyRotationError, match="did not respond"):
            await rotate_agent_api_key(db, alice["user_id"])

    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == alice["user_id"])
        )).scalar_one()
        assert cfg.agent_api_key == old_key

        events = (await db.execute(
            select(AgentSecurityEvent)
            .where(AgentSecurityEvent.user_id == alice["user_id"])
        )).scalars().all()
        assert len(events) == 1
        assert events[0].event_type == EVENT_AGENT_KEY_ROTATION_FAILED

    # Bridge called once for rotation (succeeded but verify failed),
    # then again to restore old env.
    assert len(patch_bridge_ok) == 2


# ─── Service: pre-flight rejections ───────────────────────────────────


@pytest.mark.asyncio
async def test_rotation_rejects_user_with_no_agent_config(
    patch_bridge_ok, patch_verify_ok,
):
    """User with no AgentConfig row → AgentKeyRotationError, no DB
    writes."""
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid, email=f"{uid[:8]}@x.com", hashed_password="x", name="x",
        ))
        await db.commit()

        with pytest.raises(AgentKeyRotationError, match="No agent_config"):
            await rotate_agent_api_key(db, uid)

    # No bridge call
    assert len(patch_bridge_ok) == 0


@pytest.mark.asyncio
async def test_rotation_rejects_self_hosted_user(
    patch_bridge_ok, patch_verify_ok,
):
    """hosting_mode='self-hosted' → AgentKeyRotationError, no bridge call."""
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid, email=f"{uid[:8]}@x.com", hashed_password="x", name="x",
        ))
        await db.flush()
        db.add(AgentConfig(
            user_id=uid,
            hosting_mode="self-hosted",
            agent_api_key="some_key",
        ))
        await db.commit()

        with pytest.raises(AgentKeyRotationError, match="managed-mode"):
            await rotate_agent_api_key(db, uid)

    assert len(patch_bridge_ok) == 0


# ─── Endpoint: full HTTP integration ──────────────────────────────────


@pytest_asyncio.fixture
async def rotation_test_app() -> AsyncIterator[FastAPI]:
    """Minimal FastAPI app with auth + agent routers for endpoint tests.
    Skips if FastAPI/Starlette pin mismatch (same caveat as T0c).

    Note: the router imports themselves trigger FastAPI's APIRouter
    constructor, so they sit inside the try/except too — without that,
    the pin mismatch fires before we get to skip.
    """
    try:
        from app.api.auth import router as auth_router
        from app.api.agent import router as agent_router

        app = FastAPI()
        app.include_router(auth_router, prefix=settings.api_prefix)
        app.include_router(agent_router, prefix=settings.api_prefix)
    except TypeError as e:
        pytest.skip(f"FastAPI wiring failed (local env pin mismatch): {e}")
    yield app


@pytest_asyncio.fixture
async def http_client(rotation_test_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=rotation_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def alice_jwt(alice_with_managed_container) -> tuple[dict, str]:
    """Mint a real JWT for alice so the endpoint's get_current_user
    accepts it. Returns (alice_dict, bearer_token)."""
    from app.services.auth_service import create_access_token
    token = create_access_token(alice_with_managed_container["user_id"])
    return alice_with_managed_container, token


@pytest_asyncio.fixture
async def enable_rotation():
    prev = settings.enable_agent_key_rotation
    settings.enable_agent_key_rotation = True
    try:
        yield
    finally:
        settings.enable_agent_key_rotation = prev


@pytest.mark.asyncio
async def test_endpoint_503_when_flag_disabled(http_client, alice_jwt):
    """Flag off → 503, no DB or bridge action."""
    settings.enable_agent_key_rotation = False
    alice, token = alice_jwt
    r = await http_client.post(
        "/api/agent/rotate-key",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 503
    assert "not enabled" in r.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_endpoint_401_when_no_sensitive_token(
    http_client, alice_jwt, enable_rotation,
):
    """JWT alone is insufficient — sensitive-action token required."""
    alice, token = alice_jwt
    r = await http_client.post(
        "/api/agent/rotate-key",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_endpoint_401_when_wrong_purpose_token(
    http_client, alice_jwt, enable_rotation,
):
    """Sensitive-action token minted for delete_account must NOT be
    accepted for rotate-key — purpose mismatch."""
    from app.services.sensitive_action_token import (
        SensitiveActionPurpose, issue_sensitive_action_token,
    )
    alice, token = alice_jwt
    wrong_token, _ = issue_sensitive_action_token(
        user_id=alice["user_id"],
        purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    r = await http_client.post(
        "/api/agent/rotate-key",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Sensitive-Action-Token": wrong_token,
        },
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_endpoint_happy_path(
    http_client, alice_jwt, enable_rotation, patch_bridge_ok, patch_verify_ok,
):
    """Mint a ROTATE_AGENT_KEY token via /auth/reauth, then call
    /api/agent/rotate-key with it. Returns 200 with new key."""
    alice, token = alice_jwt

    # Mint the sensitive-action token via the real /auth/reauth endpoint
    # so we exercise the body-purpose plumbing too.
    reauth_r = await http_client.post(
        "/api/auth/reauth",
        headers={"Authorization": f"Bearer {token}"},
        json={"purpose": "rotate_agent_key"},
    )
    assert reauth_r.status_code == 200, reauth_r.text
    sat = reauth_r.json()["sensitive_action_token"]
    assert reauth_r.json()["purpose"] == "rotate_agent_key"

    # Rotate.
    r = await http_client.post(
        "/api/agent/rotate-key",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Sensitive-Action-Token": sat,
        },
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["new_agent_api_key"] != alice["agent_key"]
    assert payload["key_prefix"] == payload["new_agent_api_key"][:8]
    assert payload["rotated_at"].endswith("Z")


@pytest.mark.asyncio
async def test_endpoint_401_on_token_replay(
    http_client, alice_jwt, enable_rotation, patch_bridge_ok, patch_verify_ok,
):
    """A successfully-redeemed sensitive-action token cannot be replayed."""
    alice, token = alice_jwt
    reauth_r = await http_client.post(
        "/api/auth/reauth",
        headers={"Authorization": f"Bearer {token}"},
        json={"purpose": "rotate_agent_key"},
    )
    sat = reauth_r.json()["sensitive_action_token"]

    # First call succeeds.
    r1 = await http_client.post(
        "/api/agent/rotate-key",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Sensitive-Action-Token": sat,
        },
    )
    assert r1.status_code == 200

    # Second call with the SAME token must 401 (already redeemed).
    r2 = await http_client.post(
        "/api/agent/rotate-key",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Sensitive-Action-Token": sat,
        },
    )
    assert r2.status_code == 401


@pytest.mark.asyncio
async def test_endpoint_404_on_agent_run_mode(
    http_client, alice_jwt, enable_rotation,
):
    """The agent_router is mounted on tenant containers too. Rotation
    must be invisible there — return 404, not 503 — so probes can't
    distinguish a tenant container from a non-existent endpoint."""
    prev = settings.run_mode
    settings.run_mode = "agent"
    try:
        alice, token = alice_jwt
        r = await http_client.post(
            "/api/agent/rotate-key",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404
    finally:
        settings.run_mode = prev


@pytest.mark.asyncio
async def test_last_rotated_endpoint(
    http_client, alice_jwt, enable_rotation, patch_bridge_ok, patch_verify_ok,
):
    """Before rotation: last_rotated_at is None. After: it's the
    timestamp of the most recent agent_key_rotated event."""
    alice, token = alice_jwt

    pre = await http_client.get(
        "/api/agent/key-last-rotated",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert pre.status_code == 200
    assert pre.json()["last_rotated_at"] is None

    # Rotate.
    sat_r = await http_client.post(
        "/api/auth/reauth",
        headers={"Authorization": f"Bearer {token}"},
        json={"purpose": "rotate_agent_key"},
    )
    sat = sat_r.json()["sensitive_action_token"]
    rotate_r = await http_client.post(
        "/api/agent/rotate-key",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Sensitive-Action-Token": sat,
        },
    )
    assert rotate_r.status_code == 200
    rotated_at = rotate_r.json()["rotated_at"]

    post = await http_client.get(
        "/api/agent/key-last-rotated",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert post.status_code == 200
    # Compare without microseconds — timestamps round-trip with Python
    # default precision but we don't need to assert on that.
    assert post.json()["last_rotated_at"] is not None
    assert post.json()["last_rotated_at"][:19] == rotated_at[:19]
