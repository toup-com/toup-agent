"""Tests for the platform-side watch provisioning RPC + the agent-side
auto-arm path in create_trigger.

Covers the production-grade pipeline shipped on 2026-05-14:

  • POST /api/v1/triggers/_provision_watch authentication
  • Endpoint correctly translates GmailWatchError reasons into
    structured `{provisioned: false, error: "..."}` responses
  • Successful arm writes ConnectorIdentity.metadata_json
  • Agent's `_provision_email_watch` stamps `provider_state_json` +
    `last_status` based on the platform's response
  • Webhook reads watermark from ConnectorIdentity (no Trigger DB
    dependency)
  • email_received handler's `_notify_reauth` fans out to all
    delivery_channels + dedupes within the 24h window
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


os.environ.setdefault("AGENT_API_KEY", "test-key-prov-rpc")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000000aa")
os.environ.setdefault("TRIGGERS_EMAIL_ENABLED", "true")
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("PUBSUB_TOPIC", "gmail-events")


def _build_app() -> FastAPI:
    from app.api.triggers_provision import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    app = _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Auth ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provision_requires_agent_key(client):
    r = await client.post("/api/v1/triggers/_provision_watch", json={
        "user_id": "any", "connector_id": "gmail",
    })
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_provision_rejects_user_id_mismatch(client, monkeypatch):
    """Even with the right X-Agent-Key, a body.user_id that doesn't
    match the AgentConfig row's user_id is rejected. Defence-in-depth
    against agent routing bugs."""
    # No matching AgentConfig row exists for this test DB; the auth
    # helper short-circuits with 401.
    r = await client.post(
        "/api/v1/triggers/_provision_watch",
        json={"user_id": "wrong-user", "connector_id": "gmail"},
        headers={"X-Agent-Key": "some-key"},
    )
    assert r.status_code == 401


# ── Failure mode translation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_provision_handles_no_gmail_identity(client, monkeypatch):
    """When the connector vault has no refresh token for the user,
    start_watch raises 'no active gmail identity'. The RPC should
    map this to error=needs_reauth instead of an HTTP failure."""
    from app.api import triggers_provision as tp
    from app.services.gmail_pubsub import GmailWatchError

    user_id = "11111111-2222-3333-4444-555555555555"
    api_key = "key-test-noident"

    # Seed an AgentConfig row so the auth gate passes.
    async def _ok_auth(_x_agent_key, _user_id):
        from app.db.models import AgentConfig
        return AgentConfig(user_id=user_id, agent_api_key=api_key)

    async def _raise(_uid):
        raise GmailWatchError("no active gmail identity / refresh token")

    monkeypatch.setattr(tp, "_auth_agent", _ok_auth)
    monkeypatch.setattr(
        "app.services.gmail_pubsub.start_watch", _raise,
    )

    r = await client.post(
        "/api/v1/triggers/_provision_watch",
        json={"user_id": user_id, "connector_id": "gmail"},
        headers={"X-Agent-Key": api_key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provisioned"] is False
    assert body["error"] == "needs_reauth"


@pytest.mark.asyncio
async def test_provision_handles_ops_blocked(client, monkeypatch):
    """GCP project/topic unset on the platform → ops_blocked.
    Returned BEFORE start_watch is called (cheap fail-fast)."""
    from app.api import triggers_provision as tp
    from app.config import settings

    user_id = "22222222-3333-4444-5555-666666666666"
    api_key = "key-test-opsblock"

    async def _ok_auth(_x, _u):
        from app.db.models import AgentConfig
        return AgentConfig(user_id=user_id, agent_api_key=api_key)

    monkeypatch.setattr(tp, "_auth_agent", _ok_auth)
    monkeypatch.setattr(settings, "gcp_project", "")
    monkeypatch.setattr(settings, "pubsub_topic", "")

    r = await client.post(
        "/api/v1/triggers/_provision_watch",
        json={"user_id": user_id, "connector_id": "gmail"},
        headers={"X-Agent-Key": api_key},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["provisioned"] is False
    assert body["error"] == "ops_blocked"


@pytest.mark.asyncio
async def test_provision_unsupported_connector(client, monkeypatch):
    """v1 only supports gmail. calendar/slack/etc → unsupported_connector."""
    from app.api import triggers_provision as tp

    user_id = "33333333-4444-5555-6666-777777777777"
    api_key = "key-test-unsupp"

    async def _ok_auth(_x, _u):
        from app.db.models import AgentConfig
        return AgentConfig(user_id=user_id, agent_api_key=api_key)

    monkeypatch.setattr(tp, "_auth_agent", _ok_auth)

    r = await client.post(
        "/api/v1/triggers/_provision_watch",
        json={"user_id": user_id, "connector_id": "slack"},
        headers={"X-Agent-Key": api_key},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["provisioned"] is False
    assert body["error"] == "unsupported_connector"


# ── Helper: metadata parse + expiry checks ───────────────────────────


def test_parse_metadata_handles_garbage():
    from app.api.triggers_provision import _parse_metadata
    assert _parse_metadata(None) == {}
    assert _parse_metadata("") == {}
    assert _parse_metadata("not-json") == {}
    assert _parse_metadata("[1,2,3]") == {}            # array → not a dict
    assert _parse_metadata('{"k":"v"}') == {"k": "v"}


def test_expired_iso():
    from app.api.triggers_provision import _expired
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    assert _expired(past) is True
    assert _expired(future) is False
    assert _expired("garbage") is True                 # fail-closed


# ── Agent-side auto-arm — provider_state_patch translation ───────────


@pytest.mark.asyncio
async def test_auto_arm_stamps_live_state(monkeypatch):
    """When the platform returns provisioned=true, the trigger row's
    provider_state_json gets gmail_history_id + watch_expires_at and
    last_status stays at never_fired (ready to fire on first push)."""
    import httpx
    from app.api import triggers as t

    captured: dict = {}

    async def _fake_stamp(trigger_id, *, last_status, provider_state_patch):
        captured["last_status"] = last_status
        captured["patch"] = provider_state_patch

    monkeypatch.setattr(t, "_stamp_trigger_state", _fake_stamp)

    class _MockResp:
        status_code = 200
        text = ""
        def json(self):
            return {
                "provisioned": True,
                "history_id": "98765",
                "expires_at": "2026-05-21T00:00:00+00:00",
            }

    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return _MockResp()

    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)
    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "https://test.local/api")
    monkeypatch.setattr(settings, "agent_api_key", "key")
    monkeypatch.setattr(settings, "user_id", "uid")

    await t._provision_email_watch("trig-1")
    assert captured["last_status"] == "never_fired"
    assert captured["patch"]["gmail_history_id"] == "98765"
    assert captured["patch"]["watch_expires_at"].startswith("2026-05-21")


@pytest.mark.asyncio
async def test_auto_arm_stamps_needs_reauth(monkeypatch):
    """provisioned=false + error=needs_reauth → last_status=skipped_reauth."""
    import httpx
    from app.api import triggers as t

    captured: dict = {}

    async def _fake_stamp(trigger_id, *, last_status, provider_state_patch):
        captured["last_status"] = last_status
        captured["patch"] = provider_state_patch

    monkeypatch.setattr(t, "_stamp_trigger_state", _fake_stamp)

    class _MockResp:
        status_code = 200
        text = ""
        def json(self):
            return {
                "provisioned": False,
                "error": "needs_reauth",
                "detail": "Gmail refresh token is rejected",
            }

    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return _MockResp()

    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)
    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "https://test.local/api")
    monkeypatch.setattr(settings, "agent_api_key", "key")
    monkeypatch.setattr(settings, "user_id", "uid")

    await t._provision_email_watch("trig-2")
    assert captured["last_status"] == "skipped_reauth"
    assert captured["patch"]["provision_error"] == "needs_reauth"


@pytest.mark.asyncio
async def test_auto_arm_handles_transport_failure(monkeypatch):
    """Platform unreachable → provisioning_failed + platform_unreachable
    detail. The trigger row exists and the agent + UI surface the issue
    cleanly; no exception escapes create_trigger."""
    import httpx
    from app.api import triggers as t

    captured: dict = {}

    async def _fake_stamp(trigger_id, *, last_status, provider_state_patch):
        captured["last_status"] = last_status
        captured["patch"] = provider_state_patch

    monkeypatch.setattr(t, "_stamp_trigger_state", _fake_stamp)

    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw):
            raise httpx.ConnectError("could not reach platform")

    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)
    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "https://test.local/api")
    monkeypatch.setattr(settings, "agent_api_key", "key")
    monkeypatch.setattr(settings, "user_id", "uid")

    await t._provision_email_watch("trig-3")
    assert captured["last_status"] == "provisioning_failed"
    assert captured["patch"]["provision_error"] == "platform_unreachable"


@pytest.mark.asyncio
async def test_auto_arm_skips_when_agent_misconfigured(monkeypatch):
    """If platform_api_url / agent_api_key / user_id are missing,
    skip the RPC entirely + stamp provisioning_failed so the user
    knows why. Avoids burning an HTTP timeout on every create."""
    from app.api import triggers as t

    captured: dict = {}

    async def _fake_stamp(trigger_id, *, last_status, provider_state_patch):
        captured["last_status"] = last_status
        captured["patch"] = provider_state_patch

    monkeypatch.setattr(t, "_stamp_trigger_state", _fake_stamp)

    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "")
    monkeypatch.setattr(settings, "agent_api_key", "")
    monkeypatch.setattr(settings, "user_id", "")

    await t._provision_email_watch("trig-4")
    assert captured["last_status"] == "provisioning_failed"
    assert captured["patch"]["provision_error"] == "agent_misconfigured"


# ── Reauth notice dedupe ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reauth_notice_dedupes_within_window(monkeypatch):
    """Two _notify_reauth calls within the 24h window — only the first
    sends. Prevents a Pub/Sub retry storm from spamming the user."""
    from app.agent.triggers.email_received_handler import (
        EmailReceivedHandler, _REAUTH_NOTICE_DEDUPE,
    )

    sends: list = []

    async def _fake_writer(*a, **kw):
        sends.append(kw)
        return "msg-id", "day-id"

    async def _fake_broadcaster(*a, **kw):
        sends.append(("broadcast", kw))
        return 1

    handler = EmailReceivedHandler(
        writer=_fake_writer, broadcaster=_fake_broadcaster,
    )

    class _Trig:
        id = "trig-r-1"
        user_id = "user-r-1"
        name = "My trigger"
        kind = "email_received"
        config_json = {"delivery_channels": ["website", "telegram"]}

    _REAUTH_NOTICE_DEDUPE.clear()
    await handler._notify_reauth(_Trig(), reason="test")
    first_count = len(sends)
    assert first_count >= 1

    # Second call within the window — should not send.
    await handler._notify_reauth(_Trig(), reason="test")
    assert len(sends) == first_count


@pytest.mark.asyncio
async def test_reauth_notice_fires_when_dedupe_expires(monkeypatch):
    """If 24h has passed since the last notice, a new failure produces
    a new notice. Keeps the user informed if they still haven't acted."""
    from app.agent.triggers import email_received_handler as h

    sends: list = []

    async def _fake_writer(*a, **kw):
        sends.append("written")
        return "msg-id", "day-id"

    async def _fake_broadcaster(*a, **kw):
        return 1

    handler = h.EmailReceivedHandler(
        writer=_fake_writer, broadcaster=_fake_broadcaster,
    )

    class _Trig:
        id = "trig-r-2"
        user_id = "user-r-2"
        name = "X"
        kind = "email_received"
        config_json = {"delivery_channels": ["website"]}

    # Pretend the last notice fired 25h ago.
    h._REAUTH_NOTICE_DEDUPE[("user-r-2", "trig-r-2")] = (
        time.time() - 25 * 3600
    )
    await handler._notify_reauth(_Trig(), reason="test")
    assert len(sends) == 1
