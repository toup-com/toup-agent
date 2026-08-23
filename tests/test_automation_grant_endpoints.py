"""Platform HTTP surface for automations (Round 26): grant-request
lifecycle, templates, the agent RPC auth wall, and the dark-launch 404.

PLATFORM lane. The app under test mounts the two real routers; the
`automations` flag is flipped per-test via the real
feature_flags.set_rollout_pct (DB row wins over env, same as prod).
"""

import json
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig, AutomationGrant, AutomationTemplate


@pytest_asyncio.fixture
async def app() -> FastAPI:
    from app.api.auth import router as auth_router
    from app.api.automations_proxy import router as proxy_router
    from app.api.automations_platform import router as rpc_router
    a = FastAPI()
    a.include_router(auth_router, prefix=settings.api_prefix)
    a.include_router(proxy_router, prefix=settings.api_prefix)
    a.include_router(rpc_router, prefix=settings.api_prefix)
    return a


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


async def _flag_on():
    from app.services import feature_flags
    async with async_session_maker() as db:
        await feature_flags.set_rollout_pct(db, "automations", 100)


async def _mk_agent_config(user_id: str) -> str:
    key = f"agent-key-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=user_id,
            agent_url="http://agent.invalid:8001",
            agent_api_key=key,
            deploy_status="active",
        ))
        await db.commit()
    return key


async def _mk_pending_grant(user_id: str, **over) -> str:
    row = AutomationGrant(
        user_id=user_id,
        connector_id="slack",
        tool_name="slack__send_message",
        target_json=json.dumps({"kind": "channel", "id": "C1",
                                "label": "#eng"}),
        mode="auto",
        summary="post to #eng",
        status=over.get("status", "pending"),
        expires_at=over.get(
            "expires_at", datetime.utcnow() + timedelta(hours=1),
        ),
    )
    async with async_session_maker() as db:
        db.add(row)
        await db.commit()
        return row.id


@pytest.mark.asyncio
async def test_dark_launch_every_route_404s(client, auth_headers):
    r = await client.get("/api/automations/templates", headers=auth_headers)
    assert r.status_code == 404
    r = await client.get("/api/automations/grant-requests/nope",
                         headers=auth_headers)
    assert r.status_code == 404
    r = await client.get("/api/automations", headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_templates_list(client, auth_headers):
    await _flag_on()
    async with async_session_maker() as db:
        db.add(AutomationTemplate(
            slug="jira-to-slack", name="Jira → Slack",
            description="demo", icon="jira",
            connectors_json='["jira", "slack"]',
            spec_json='{"name": "Jira \\u2192 Slack"}',
        ))
        await db.commit()
    r = await client.get("/api/automations/templates", headers=auth_headers)
    assert r.status_code == 200
    templates = r.json()["templates"]
    assert [t["slug"] for t in templates] == ["jira-to-slack"]
    assert templates[0]["connectors"] == ["jira", "slack"]
    assert templates[0]["spec"]["name"] == "Jira → Slack"


@pytest.mark.asyncio
async def test_grant_approve_claims_exactly_once(
    client, auth_headers, test_user_id,
):
    await _flag_on()
    gid = await _mk_pending_grant(test_user_id)
    r = await client.post(
        f"/api/automations/grant-requests/{gid}/approve",
        json={"decided_via": "web"}, headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved"
    assert body["action"] == "slack__send_message"
    assert body["target"]["id"] == "C1"
    # Double-tap: the claim is single-winner.
    r2 = await client.post(
        f"/api/automations/grant-requests/{gid}/approve",
        json={}, headers=auth_headers,
    )
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_expired_grant_approve_is_410(
    client, auth_headers, test_user_id,
):
    await _flag_on()
    gid = await _mk_pending_grant(
        test_user_id,
        expires_at=datetime.utcnow() - timedelta(minutes=1),
    )
    r = await client.post(
        f"/api/automations/grant-requests/{gid}/approve",
        json={}, headers=auth_headers,
    )
    assert r.status_code == 410
    r2 = await client.get(f"/api/automations/grant-requests/{gid}",
                          headers=auth_headers)
    assert r2.json()["status"] == "expired"


@pytest.mark.asyncio
async def test_reject_is_double_tap_safe(client, auth_headers, test_user_id):
    await _flag_on()
    gid = await _mk_pending_grant(test_user_id)
    r = await client.post(
        f"/api/automations/grant-requests/{gid}/reject",
        json={"decided_via": "app"}, headers=auth_headers,
    )
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    r2 = await client.post(
        f"/api/automations/grant-requests/{gid}/reject",
        json={}, headers=auth_headers,
    )
    assert r2.status_code == 200 and r2.json()["status"] == "rejected"


@pytest.mark.asyncio
async def test_revoke_kills_an_approved_grant(
    client, auth_headers, test_user_id,
):
    await _flag_on()
    gid = await _mk_pending_grant(test_user_id, status="approved")
    r = await client.post(
        f"/api/automations/grant-requests/{gid}/revoke",
        headers=auth_headers,
    )
    assert r.status_code == 200 and r.json()["status"] == "revoked"
    # And the dispatcher now refuses it (fail closed downstream).
    async with async_session_maker() as db:
        row = await db.get(AutomationGrant, gid)
        assert row.status == "revoked"


@pytest.mark.asyncio
async def test_other_users_grant_is_a_404_not_403(
    client, auth_headers, test_user_id,
):
    await _flag_on()
    other = str(uuid.uuid4())
    from app.db.models import User
    async with async_session_maker() as db:
        db.add(User(id=other, email=f"{other[:8]}@example.com",
                    hashed_password="x", name="Other"))
        await db.commit()
    gid = await _mk_pending_grant(other)
    r = await client.get(f"/api/automations/grant-requests/{gid}",
                         headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_rpc_wall_rejects_bad_agent_key(client, test_user_id):
    await _flag_on()
    await _mk_agent_config(test_user_id)
    r = await client.get(
        "/api/v1/automations/registry",
        headers={"X-Agent-Key": "wrong", "X-Agent-User-Id": test_user_id},
    )
    assert r.status_code == 401
    r = await client.get("/api/v1/automations/registry")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_rpc_registry_and_grant_request_roundtrip(
    client, test_user_id,
):
    await _flag_on()
    key = await _mk_agent_config(test_user_id)
    headers = {"X-Agent-Key": key, "X-Agent-User-Id": test_user_id}

    from app.services.connector_registry import get_registry
    if not get_registry().automation_registry():
        get_registry().load_all(include_experimental=True)

    r = await client.get("/api/v1/automations/registry", headers=headers)
    assert r.status_code == 200
    ids = {c["connector_id"] for c in r.json()["connectors"]}
    assert {"jira", "slack", "gmail"} <= ids

    r = await client.post(
        "/api/v1/automations/grant-requests",
        json={
            "connector_id": "slack",
            "tool_name": "slack__send_message",
            "target": {"kind": "channel", "id": "C9", "label": "#ops"},
            "mode": "auto",
            "summary": "post to #ops",
        },
        headers=headers,
    )
    assert r.status_code == 200
    grant = r.json()["grant"]
    assert grant["status"] == "pending"

    # An ungrantable action (read tool) is refused at request time.
    r = await client.post(
        "/api/v1/automations/grant-requests",
        json={
            "connector_id": "slack",
            "tool_name": "slack__list_channels",
            "target": {"kind": "channel", "id": "C9"},
            "summary": "nope",
        },
        headers=headers,
    )
    assert r.status_code == 422

    # grant-status returns the authoritative row.
    r = await client.get(
        "/api/v1/automations/grant-status",
        params={"grant_id": grant["id"]}, headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["grant"]["id"] == grant["id"]
