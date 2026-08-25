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


@pytest.mark.asyncio
async def test_rpc_connections_disclose_the_bound_account(
    client, test_user_id,
):
    """R28 connector disclosure: /connections names the account the
    connector is bound to (provider_account_id) — the Gmail address
    where we have it, None where the provider never told us (Outlook
    has no backfill), so the setup skill can say WHICH inbox it is
    about to automate."""
    await _flag_on()
    key = await _mk_agent_config(test_user_id)
    headers = {"X-Agent-Key": key, "X-Agent-User-Id": test_user_id}

    from app.db.models.connectors import ConnectorIdentity
    async with async_session_maker() as db:
        db.add(ConnectorIdentity(
            user_id=test_user_id, connector_id="gmail", status="active",
            provider_account_id="person@gmail.com",
            scopes_json=json.dumps(["gmail.readonly"]),
        ))
        db.add(ConnectorIdentity(
            user_id=test_user_id, connector_id="outlook", status="active",
        ))
        await db.commit()

    r = await client.get("/api/v1/automations/connections", headers=headers)
    assert r.status_code == 200
    by_id = {c["connector_id"]: c for c in r.json()["connections"]}
    assert by_id["gmail"]["account"] == "person@gmail.com"
    assert by_id["gmail"]["connected"] is True
    assert by_id["outlook"]["account"] is None


# ── Round 29: grants on the Overview, mode consent, scope truth ──────


async def _mk_approved_grant(user_id: str, automation_id: str) -> str:
    row = AutomationGrant(
        user_id=user_id,
        automation_id=automation_id,
        connector_id="slack",
        tool_name="slack__send_message",
        target_json=json.dumps({"kind": "channel", "id": "C1",
                                "label": "#eng"}),
        mode="auto",
        summary="post to #eng",
        status="approved",
        decided_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=1),
    )
    async with async_session_maker() as db:
        db.add(row)
        await db.commit()
        return row.id


@pytest.mark.asyncio
async def test_grants_list_scoped_to_owner_and_automation(
    client, auth_headers, test_user_id,
):
    await _flag_on()
    aid = str(uuid.uuid4())
    gid = await _mk_approved_grant(test_user_id, aid)
    await _mk_pending_grant(test_user_id)          # no automation_id
    await _mk_approved_grant(test_user_id, str(uuid.uuid4()))  # other one

    r = await client.get(f"/api/automations/{aid}/grants",
                         headers=auth_headers)
    assert r.status_code == 200
    grants = r.json()["grants"]
    assert [g["id"] for g in grants] == [gid]
    assert grants[0]["granted_at"] is not None
    assert grants[0]["action_label"] == "send message"


@pytest.mark.asyncio
async def test_nested_revoke_checks_the_automation_and_notifies(
    client, auth_headers, test_user_id, monkeypatch,
):
    await _flag_on()
    aid = str(uuid.uuid4())
    gid = await _mk_approved_grant(test_user_id, aid)

    decided = []

    async def _capture(db, uid, row):
        decided.append(row.status)

    monkeypatch.setattr(
        "app.api.automations_proxy._notify_agent_grant_decided", _capture)

    r = await client.post(
        f"/api/automations/{str(uuid.uuid4())}/grants/{gid}/revoke",
        headers=auth_headers,
    )
    assert r.status_code == 404, "wrong automation must not find the grant"

    r = await client.post(
        f"/api/automations/{aid}/grants/{gid}/revoke", headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "revoked"
    assert decided == ["revoked"], "the agent hook heard about it"

    r = await client.post(
        f"/api/automations/{aid}/grants/{gid}/revoke", headers=auth_headers,
    )
    assert r.status_code == 409, "revoke is one-way"


@pytest.mark.asyncio
async def test_mode_patch_flips_spec_then_grants(
    client, auth_headers, test_user_id, monkeypatch,
):
    """§3.3: the user-JWT route is the ONLY door to a grant's mode —
    it proxies the spec flip to the agent first, then moves every
    approved grant, stamping mode_changed_at."""
    from fastapi import Response

    await _flag_on()
    aid = str(uuid.uuid4())
    gid = await _mk_approved_grant(test_user_id, aid)

    proxied = []

    async def _fake_proxy(request, sub_path, *, current_user, db):
        proxied.append(sub_path)
        return Response(content=b'{"automation": {}}', status_code=200,
                        media_type="application/json")

    monkeypatch.setattr("app.api.automations_proxy._proxy", _fake_proxy)

    r = await client.patch(
        f"/api/automations/{aid}/mode", json={"mode": "confirm"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert proxied == [f"/{aid}/mode"], "spec flip proxied to the agent"
    async with async_session_maker() as db:
        row = await db.get(AutomationGrant, gid)
        assert row.mode == "confirm"
        assert row.mode_changed_at is not None

    # An agent 409 (e.g. draft automation mid-edit) leaves grants alone.
    async def _refusing_proxy(request, sub_path, *, current_user, db):
        return Response(content=b'{"detail": "no"}', status_code=409,
                        media_type="application/json")

    monkeypatch.setattr("app.api.automations_proxy._proxy", _refusing_proxy)
    r = await client.patch(
        f"/api/automations/{aid}/mode", json={"mode": "auto"},
        headers=auth_headers,
    )
    assert r.status_code == 409
    async with async_session_maker() as db:
        row = await db.get(AutomationGrant, gid)
        assert row.mode == "confirm", "a refused spec flip moves no grant"


@pytest.mark.asyncio
async def test_outlook_draft_grant_needs_the_reconnect(
    client, test_user_id,
):
    """§5 scope truth: a pre-R29 Outlook connection (no Mail.ReadWrite)
    gets the stable 409 scope_missing at grant-request time; a
    reconnected one proceeds; a connection with NO recorded scopes is
    let through (dispatch fails honestly instead)."""
    await _flag_on()
    key = await _mk_agent_config(test_user_id)
    headers = {"X-Agent-Key": key, "X-Agent-User-Id": test_user_id}

    from app.services.connector_registry import get_registry
    if not get_registry().automation_registry():
        get_registry().load_all(include_experimental=True)

    from app.db.models.connectors import ConnectorIdentity
    async with async_session_maker() as db:
        db.add(ConnectorIdentity(
            user_id=test_user_id, connector_id="outlook", status="active",
            scopes_json=json.dumps([
                "https://graph.microsoft.com/Mail.Read",
                "https://graph.microsoft.com/Mail.Send",
            ]),
        ))
        await db.commit()

    body = {
        "connector_id": "outlook",
        "tool_name": "outlook__create_draft",
        "target": {"kind": "recipient", "id": "boss@corp.com",
                   "label": "boss@corp.com"},
        "summary": "draft replies to the boss",
    }
    r = await client.post("/api/v1/automations/grant-requests",
                          json=body, headers=headers)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "scope_missing"
    assert detail["connector_id"] == "outlook"
    assert detail["reconnect"] is True
    assert "Mail.ReadWrite" in detail["needed_scope"]

    # Reconnected: the scope is on the identity now.
    from sqlalchemy import update as sa_update
    async with async_session_maker() as db:
        await db.execute(
            sa_update(ConnectorIdentity)
            .where(ConnectorIdentity.user_id == test_user_id)
            .where(ConnectorIdentity.connector_id == "outlook")
            .values(scopes_json=json.dumps([
                "https://graph.microsoft.com/Mail.Read",
                "https://graph.microsoft.com/Mail.ReadWrite",
            ]))
        )
        await db.commit()
    r = await client.post("/api/v1/automations/grant-requests",
                          json=body, headers=headers)
    assert r.status_code == 200
    assert r.json()["grant"]["status"] == "pending"
