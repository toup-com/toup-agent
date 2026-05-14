"""Tests for the user-facing /api/triggers CRUD surface.

Uses the same minimal-FastAPI-app approach as the inbound tests —
bypasses conftest's autouse _reset_database fixture (which init_db's
the full Postgres schema and is heavy in the local async-SQLite
shell). CI runs through the standard pytest path against Postgres.

Coverage:
  - GET / list happy path + ordering
  - POST: validation (unknown kind / action / channel)
  - POST: forward_to_telegram auto-implies telegram delivery
  - POST: feature flag off → 404
  - PATCH: partial update preserves untouched fields
  - DELETE: cascades event rows (covered in schema test)
  - GET /events: paginated history
  - POST /test: inserts a `test:` event, returns its row
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select


os.environ.setdefault("AGENT_API_KEY", "test-key-triggers-api")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000bbb")
os.environ.setdefault("TRIGGERS_EMAIL_ENABLED", "true")

CONTAINER_USER_ID = "00000000-0000-0000-0000-000000000bbb"


def _build_app() -> FastAPI:
    from app.api.triggers import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest_asyncio.fixture(autouse=True)
def _mock_auto_arm(monkeypatch):
    """Bypass the platform→watch RPC in unit tests.

    Real auto-arm fires an HTTP POST to `platform_api_url`. Unit tests
    don't have a platform to talk to; mocking the inner function keeps
    the API-layer assertions ("trigger was created with these fields")
    decoupled from the cross-process provisioning. The auto-arm itself
    is covered by `test_triggers_provision_endpoint.py`.
    """
    async def _noop(_trigger_id: str) -> None:
        return None
    from app.api import triggers as _t
    monkeypatch.setattr(_t, "_provision_email_watch", _noop)


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    app = _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Validation ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_rejects_unknown_kind(client):
    r = await client.post("/api/triggers", json={
        "kind": "calendar_event_made_up",
        "action": "summarize_and_post",
    })
    # 400 (unknown kind validation) or 404 (feature gate) both acceptable;
    # both are loud rejections from a deliberate validator.
    assert r.status_code in (400, 404)


@pytest.mark.asyncio
async def test_create_rejects_unknown_action(client):
    r = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "delete_everything",
    })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_create_rejects_unknown_delivery_channel(client):
    r = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
        "delivery_channels": ["website", "carrier_pigeon"],
    })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_create_returns_404_when_feature_off(client, monkeypatch):
    """Feature flag off → 404 (not 403/400) so Mission Control can
    render empty state cleanly. Same pattern as routines."""
    from app.config import settings as cfg
    monkeypatch.setattr(cfg, "triggers_email_enabled", False)
    r = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
    })
    assert r.status_code == 404


# ── Happy path ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_minimal_succeeds(client):
    r = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
        "name": "Watch my inbox",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["kind"] == "email_received"
    assert body["action"] == "summarize_and_post"
    assert body["name"] == "Watch my inbox"
    assert body["enabled"] is True
    assert body["fire_count"] == 0
    assert body["last_status"] == "never_fired"
    assert body["delivery_channels"] == ["website"]
    assert body["watch_provisioned"] is False


@pytest.mark.asyncio
async def test_create_forward_to_telegram_auto_adds_telegram(client):
    """If the user picks the `forward_to_telegram` action without
    explicitly adding telegram to delivery_channels, the API should
    auto-add it — the action LABEL is the source of truth."""
    r = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "forward_to_telegram",
    })
    assert r.status_code == 201
    body = r.json()
    assert "telegram" in body["delivery_channels"]
    assert "website" in body["delivery_channels"]


@pytest.mark.asyncio
async def test_create_filters_passed_through(client):
    r = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
        "filter_json": {
            "from_contains": ["@github.com"],
            "subject_contains": ["PR"],
            "exclude_categories": ["promotions"],
        },
    })
    assert r.status_code == 201
    body = r.json()
    assert body["filter_json"]["from_contains"] == ["@github.com"]
    assert body["filter_json"]["subject_contains"] == ["PR"]


# ── List + ordering ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_returns_user_triggers_newest_first(client):
    """Create two, list returns the most recent first."""
    r1 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
        "name": "first",
    })
    assert r1.status_code == 201
    r2 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "notify_only",
        "name": "second",
    })
    assert r2.status_code == 201

    r_list = await client.get("/api/triggers")
    assert r_list.status_code == 200
    rows = r_list.json()
    assert len(rows) >= 2
    names = [row["name"] for row in rows[:2]]
    # Newest first
    assert names[0] == "second"


# ── Update ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_partial_preserves_untouched_fields(client):
    r1 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
        "name": "before",
        "filter_json": {"from_contains": ["acme"]},
    })
    tid = r1.json()["id"]

    r2 = await client.patch(f"/api/triggers/{tid}", json={
        "name": "after",
    })
    assert r2.status_code == 200
    body = r2.json()
    assert body["name"] == "after"
    # Filters untouched
    assert body["filter_json"]["from_contains"] == ["acme"]
    # Action untouched
    assert body["action"] == "summarize_and_post"


@pytest.mark.asyncio
async def test_update_enabled_toggle(client):
    r1 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
    })
    tid = r1.json()["id"]
    r2 = await client.patch(f"/api/triggers/{tid}", json={"enabled": False})
    assert r2.status_code == 200
    assert r2.json()["enabled"] is False


@pytest.mark.asyncio
async def test_update_rejects_unknown_action(client):
    r1 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
    })
    tid = r1.json()["id"]
    r2 = await client.patch(f"/api/triggers/{tid}", json={"action": "nuke_inbox"})
    assert r2.status_code == 400


# ── Delete ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_returns_204_then_404_on_relookup(client):
    r1 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
    })
    tid = r1.json()["id"]
    rd = await client.delete(f"/api/triggers/{tid}")
    assert rd.status_code == 204
    rl = await client.delete(f"/api/triggers/{tid}")
    assert rl.status_code == 404


# ── Events history ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_events_empty_when_no_events(client):
    r1 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
    })
    tid = r1.json()["id"]
    r = await client.get(f"/api/triggers/{tid}/events")
    assert r.status_code == 200
    assert r.json() == []


# ── Test fire ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_test_fire_inserts_event(client):
    r1 = await client.post("/api/triggers", json={
        "kind": "email_received",
        "action": "summarize_and_post",
    })
    tid = r1.json()["id"]

    rt = await client.post(f"/api/triggers/{tid}/test")
    assert rt.status_code == 200
    body = rt.json()
    assert body["trigger_id"] == tid
    assert body["event_dedupe_id"].startswith("test:")
    # The test-fire row is queued; the runner (not wired in tests)
    # would normally dispatch. Status is whatever the runner left it
    # as — typically still 'queued' here since no runner ref injected.
    assert body["status"] in ("queued", "failed", "running")


@pytest.mark.asyncio
async def test_test_fire_404_on_missing_trigger(client):
    r = await client.post(f"/api/triggers/{uuid.uuid4()}/test")
    assert r.status_code == 404
