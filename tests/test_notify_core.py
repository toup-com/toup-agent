"""Notification core (Autopilot arc PR2) — push devices, agent-notify
ingest, extended notification preferences.

Covers the contracts the rest of the arc builds on:
- push_devices / notification_queue are PLATFORM_ONLY (split-DB rule).
- POST /api/agent/notify: X-Agent-Key + user binding (credits.py
  recipe), idempotent on (user_id, idempotency_key) — the agent outbox
  (PR4) flushes at-least-once against this.
- Device registration upserts on expo_push_token; unregister revokes.
- /api/account/preferences round-trips the new autopilot/quiet-hours
  keys AND no longer wipes unknown JSONB keys (extension_memory_optout
  regression).
"""

from __future__ import annotations

import uuid

import pytest


# ── Split-DB categorization ───────────────────────────────────────


def test_new_tables_are_platform_only():
    from app.db.models.base import (
        AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, SHARED_TABLES,
    )

    for t in ("push_devices", "notification_queue"):
        assert t in PLATFORM_ONLY_TABLES, f"{t} must be PLATFORM_ONLY"
        assert t not in AGENT_ONLY_TABLES
        assert t not in SHARED_TABLES


def test_platform_main_mounts_notify_routers():
    """Pin the mounts source-grep style (test_api_no_html_invariant.py
    pattern) — an unmounted platform route surfaces as 405/404."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "platform_main.py").read_text()
    assert "from app.api.push_devices import router as push_devices_router" in src
    assert "push_devices_router, prefix=settings.api_prefix" in src
    assert "from app.api.agent_notify import router as agent_notify_router" in src
    assert "agent_notify_router, prefix=settings.api_prefix" in src


# ── Push-device registry ──────────────────────────────────────────


def _token() -> str:
    return f"ExponentPushToken[{uuid.uuid4().hex[:22]}]"


@pytest.mark.asyncio
async def test_register_list_unregister_push_device(client, auth_headers):
    token = _token()
    res = await client.post(
        "/api/devices/push",
        json={"token": token, "platform": "ios", "device_name": "iPhone 17"},
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    device_id = res.json()["id"]

    # Re-registration is an upsert, not a duplicate.
    res = await client.post(
        "/api/devices/push",
        json={"token": token, "platform": "ios", "app_version": "1.2.3"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    assert res.json()["id"] == device_id
    assert res.json()["app_version"] == "1.2.3"

    res = await client.get("/api/devices/push", headers=auth_headers)
    assert res.status_code == 200
    devices = res.json()
    assert len(devices) == 1 and devices[0]["id"] == device_id

    # Sign-out flow revokes by token.
    res = await client.post(
        "/api/devices/push/unregister",
        json={"token": token},
        headers=auth_headers,
    )
    assert res.status_code == 200 and res.json()["revoked"] is True

    res = await client.get("/api/devices/push", headers=auth_headers)
    assert res.json() == []


@pytest.mark.asyncio
async def test_register_rejects_non_expo_token(client, auth_headers):
    res = await client.post(
        "/api/devices/push",
        json={"token": "d" * 64, "platform": "ios"},  # raw APNs-style token
        headers=auth_headers,
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_push_device_requires_auth(client):
    res = await client.post(
        "/api/devices/push", json={"token": _token(), "platform": "ios"},
    )
    assert res.status_code in (401, 403)


# ── Agent-notify ingest ───────────────────────────────────────────


async def _make_agent_config(user_id: str) -> str:
    from app.db import async_session_maker
    from app.db.models import AgentConfig

    key = f"tk-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=user_id, agent_api_key=key))
        await db.commit()
    return key


def _notify_body(user_id: str, **overrides) -> dict:
    body = {
        "user_id": user_id,
        "idempotency_key": f"outbox-{uuid.uuid4().hex}",
        "event_kind": "mission_completed",
        "title": "Mission finished",
        "body": "Booked the dentist appointment for Tuesday 10:00.",
        "data": {"route": "mission-control", "mission_id": "m1"},
        "priority": "default",
        "dedup_key": "m1:mission_completed",
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_agent_notify_requires_key(client, test_user_id):
    res = await client.post("/api/agent/notify", json=_notify_body(test_user_id))
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_agent_notify_rejects_wrong_key(client, test_user_id):
    await _make_agent_config(test_user_id)
    res = await client.post(
        "/api/agent/notify",
        json=_notify_body(test_user_id),
        headers={"X-Agent-Key": "not-the-key"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_agent_notify_queues_and_replays_idempotently(client, test_user_id):
    key = await _make_agent_config(test_user_id)
    body = _notify_body(test_user_id)

    res = await client.post(
        "/api/agent/notify", json=body, headers={"X-Agent-Key": key},
    )
    assert res.status_code == 200, res.text
    first = res.json()
    assert first["status"] == "queued" and first["id"]

    # At-least-once outbox replay → duplicate, same row, no second insert.
    res = await client.post(
        "/api/agent/notify", json=body, headers={"X-Agent-Key": key},
    )
    assert res.status_code == 200
    assert res.json() == {"status": "duplicate", "id": first["id"]}

    from sqlalchemy import func, select
    from app.db import async_session_maker
    from app.db.models import NotificationQueue

    async with async_session_maker() as db:
        count = await db.scalar(
            select(func.count()).select_from(NotificationQueue).where(
                NotificationQueue.user_id == test_user_id,
            )
        )
    assert count == 1


@pytest.mark.asyncio
async def test_agent_notify_rejects_unknown_kind_and_rich_data(client, test_user_id):
    key = await _make_agent_config(test_user_id)
    res = await client.post(
        "/api/agent/notify",
        json=_notify_body(test_user_id, event_kind="totally_new_kind"),
        headers={"X-Agent-Key": key},
    )
    assert res.status_code == 422

    res = await client.post(
        "/api/agent/notify",
        json=_notify_body(test_user_id, data={"nested": {"not": "scalar"}}),
        headers={"X-Agent-Key": key},
    )
    assert res.status_code == 422


# ── Notification preferences ──────────────────────────────────────


@pytest.mark.asyncio
async def test_preferences_include_autopilot_defaults(client, auth_headers):
    res = await client.get("/api/account/preferences", headers=auth_headers)
    assert res.status_code == 200, res.text
    prefs = res.json()
    assert prefs["autopilot_push"] is True
    assert prefs["autopilot_channel_fallback"] is True
    assert prefs["quiet_hours"] == {"enabled": True, "start": "22:00", "end": "08:00"}
    assert prefs["daily_push_cap"] == 10


@pytest.mark.asyncio
async def test_preferences_patch_new_keys_and_validation(client, auth_headers):
    res = await client.patch(
        "/api/account/preferences",
        json={
            "autopilot_push": False,
            "quiet_hours": {"enabled": True, "start": "23:30", "end": "07:15"},
            "daily_push_cap": 3,
        },
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text

    res = await client.get("/api/account/preferences", headers=auth_headers)
    prefs = res.json()
    assert prefs["autopilot_push"] is False
    assert prefs["quiet_hours"]["start"] == "23:30"
    assert prefs["daily_push_cap"] == 3
    # Untouched keys keep their values.
    assert prefs["security_alerts"] is True

    # Bad HH:MM must 422, not corrupt state.
    res = await client.patch(
        "/api/account/preferences",
        json={"quiet_hours": {"enabled": True, "start": "25:99", "end": "08:00"}},
        headers=auth_headers,
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_preferences_patch_preserves_unknown_keys(client, auth_headers, test_user_id):
    """Regression: the extension writes extension_memory_optout into
    the same JSONB via raw SQL; the old PATCH wiped it."""
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        user = (await db.execute(
            select(User).where(User.id == test_user_id)
        )).scalar_one()
        user.notification_preferences = {
            "morning_briefing": True,
            "extension_memory_optout": True,
        }
        await db.commit()

    res = await client.patch(
        "/api/account/preferences",
        json={"product_updates": True},
        headers=auth_headers,
    )
    assert res.status_code == 200

    async with async_session_maker() as db:
        user = (await db.execute(
            select(User).where(User.id == test_user_id)
        )).scalar_one()
        stored = user.notification_preferences
    assert stored["extension_memory_optout"] is True, "unknown key was wiped"
    assert stored["morning_briefing"] is True
    assert stored["product_updates"] is True
