"""iOS Live Activity lane — payload contract, registration API,
dispatcher routing (Autopilot phone surface).

The APNs HTTP seam is monkeypatched at
``live_activity_service.apns_push.send_live_activity`` — no network.

The payload-builder tests pin the Swift Codable contract: the widget
extension decodes ``content-state`` with the exact keys of
``LiveActivityAttributes.ContentState`` (expo-live-activity 0.4.2) and
``attributes-type`` must be the literal type name. Renaming anything
here breaks phones in the field silently — these tests are the tripwire.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED, LiveActivity, LiveActivityDevice,
    NotificationQueue, User, NQ_QUEUED, NQ_SENT, NQ_SUPPRESSED,
)
from app.services import apns_push
from app.services import live_activity_service as las
from app.services import notification_dispatcher as nd


# ── Payload builders: the Swift contract ──────────────────────────


def test_start_payload_matches_swift_contract():
    p = apns_push.build_start_payload(
        mission_id="m-123", title="Research CRMs", subtitle="Starting…",
        progress=0.0, alert_title="🚀 Autopilot engaged",
        alert_body="Research CRMs", timestamp=1_752_000_000,
    )
    aps = p["aps"]
    assert aps["event"] == "start"
    assert aps["timestamp"] == 1_752_000_000
    # iOS 18+ token reuse — the force-quit guarantee.
    assert aps["input-push-token"] == 1
    assert aps["attributes-type"] == "LiveActivityAttributes"
    assert aps["attributes"]["name"] == "m-123"
    # ContentState Codable keys — exact match required.
    cs = aps["content-state"]
    assert set(cs) <= {
        "title", "subtitle", "timerEndDateInMilliseconds",
        "progress", "imageName", "dynamicIslandImageName",
    }
    assert cs["title"] == "Research CRMs"
    assert cs["progress"] == 0.0
    assert aps["alert"]["title"] == "🚀 Autopilot engaged"
    assert aps["alert"]["sound"] == "default"


def test_start_payload_always_carries_alert_config():
    """iOS 26 drops start events with no alert configuration
    (liveactivitiesd SessionCore, observed on-device 2026-07-18) — a
    'silent' start must still ship a synthesized, SOUNDLESS alert or
    the card never renders while APNs returns 200."""
    p = apns_push.build_start_payload(
        mission_id="m-1", title="⏰ Stretch", subtitle="Time to stretch",
        timer_end_ms=1_752_000_000_000, timestamp=1,
    )
    alert = p["aps"]["alert"]
    assert alert["title"] == "⏰ Stretch"
    assert alert["body"] == "Time to stretch"
    assert "sound" not in alert


def test_update_payload_clamps_progress_and_omits_none():
    p = apns_push.build_update_payload(title="T", progress=1.7, timestamp=1)
    cs = p["aps"]["content-state"]
    assert cs["progress"] == 1.0
    assert "subtitle" not in cs
    assert "alert" not in p["aps"]
    assert p["aps"]["event"] == "update"


def test_end_payload_carries_final_state_and_optional_dismissal():
    p = apns_push.build_end_payload(
        title="T", subtitle="Completed ✓", progress=1.0,
        alert_title="Done", dismissal_date=123, timestamp=1,
    )
    assert p["aps"]["event"] == "end"
    assert p["aps"]["dismissal-date"] == 123
    assert p["aps"]["content-state"]["subtitle"] == "Completed ✓"


def test_is_token_dead_classification():
    assert apns_push.is_token_dead(410, "Unregistered")
    assert apns_push.is_token_dead(400, "BadDeviceToken")
    assert not apns_push.is_token_dead(500, "InternalServerError")
    assert not apns_push.is_token_dead(429, "TooManyRequests")


# ── Registration API ──────────────────────────────────────────────


_HEX_TOKEN = "ab" * 32  # 64-char hex


@pytest.mark.asyncio
async def test_register_and_list_live_activity_device(client, auth_headers):
    resp = await client.post(
        "/api/devices/live-activity",
        json={"token": _HEX_TOKEN, "environment": "development",
              "device_name": "iPhone 15 Pro Max"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["environment"] == "development"

    # Re-registration upserts (same id back, last_seen bumped).
    resp2 = await client.post(
        "/api/devices/live-activity",
        json={"token": _HEX_TOKEN, "environment": "development"},
        headers=auth_headers,
    )
    assert resp2.json()["id"] == body["id"]

    listed = await client.get("/api/devices/live-activity", headers=auth_headers)
    assert [d["id"] for d in listed.json()] == [body["id"]]


@pytest.mark.asyncio
async def test_concurrent_registration_of_same_token_never_500s(client, auth_headers):
    """Live-caught bug (2026-07-10): the app can fire several parallel
    registrations of the same token (retry loop + dev fast-refresh
    stacking listeners). Losers of the unique-constraint race must
    adopt the winner's row, not 500."""
    import asyncio

    token = "fe" * 32
    responses = await asyncio.gather(*[
        client.post(
            "/api/devices/live-activity",
            json={"token": token, "environment": "development"},
            headers=auth_headers,
        )
        for _ in range(4)
    ])
    assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]

    # Exactly one device row persisted. (Don't assert on the response
    # ids: the sqlite test harness runs every session on ONE shared
    # connection, so a losing request's rollback can erase a winner's
    # uncommitted insert and the ids returned are phantoms — a harness
    # artifact; postgres gives each request its own connection.)
    from sqlalchemy import func, select
    from app.db import async_session_maker

    async with async_session_maker() as db:
        count = await db.scalar(
            select(func.count()).select_from(LiveActivityDevice).where(
                LiveActivityDevice.push_to_start_token == token
            )
        )
    assert count == 1


@pytest.mark.asyncio
async def test_register_rejects_non_hex_and_bad_environment(client, auth_headers):
    bad_token = await client.post(
        "/api/devices/live-activity",
        json={"token": "ExponentPushToken[abcdefghij1234567890abcdef]"},
        headers=auth_headers,
    )
    assert bad_token.status_code == 422
    bad_env = await client.post(
        "/api/devices/live-activity",
        json={"token": _HEX_TOKEN, "environment": "staging"},
        headers=auth_headers,
    )
    assert bad_env.status_code == 422


@pytest.mark.asyncio
async def test_register_requires_auth(client):
    resp = await client.post(
        "/api/devices/live-activity", json={"token": _HEX_TOKEN},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_activity_token_report_updates_started_rows(
    client, auth_headers, test_user_id,
):
    from app.db import async_session_maker

    device_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivityDevice(
            id=device_id, user_id=test_user_id,
            push_to_start_token="cd" * 32, created_at=datetime.utcnow(),
        ))
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id="m-9",
            device_id=device_id, status=LA_STARTED,
            started_at=datetime.utcnow(),
        ))
        await db.commit()

    resp = await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "m-9", "activity_push_token": "ef" * 32},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["updated"] == 1

    # Local (non-mission) activities report unknown names → no-op.
    resp2 = await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "ExpoLiveActivity", "activity_push_token": "ef" * 32},
        headers=auth_headers,
    )
    assert resp2.json()["updated"] == 0
    assert "adopted" not in resp2.json()


# ── install_id token lifecycle ────────────────────────────────────
#
# Reinstalls rotate the push-to-start token. Without a stable install
# identity, every reinstall accretes a new device row while the stale
# sibling keeps its dead token — which APNs accepts with 200 forever,
# so half the user's cards go into the void.


@pytest.mark.asyncio
async def test_install_id_token_rotation_updates_row_in_place(client, auth_headers):
    install = "install-rotation-1"
    r1 = await client.post(
        "/api/devices/live-activity",
        json={"token": "a1" * 32, "environment": "development",
              "install_id": install},
        headers=auth_headers,
    )
    assert r1.status_code == 200, r1.text
    device_id = r1.json()["id"]

    # Reinstall: same install_id, rotated token → SAME row, new token.
    r2 = await client.post(
        "/api/devices/live-activity",
        json={"token": "a2" * 32, "environment": "development",
              "install_id": install},
        headers=auth_headers,
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["id"] == device_id

    from sqlalchemy import select
    from app.db import async_session_maker

    async with async_session_maker() as db:
        device = await db.get(LiveActivityDevice, device_id)
        assert device.push_to_start_token == "a2" * 32
        assert device.revoked_at is None
        count = len((await db.execute(
            select(LiveActivityDevice).where(
                LiveActivityDevice.install_id == install
            )
        )).scalars().all())
        assert count == 1  # no sibling accreted


@pytest.mark.asyncio
async def test_install_id_registration_revokes_null_install_sibling(
    client, auth_headers,
):
    """A stale row from an app build that predates install_id (NULL)
    with a different token is a reinstall leftover — revoke it."""
    # Old-build registration: no install_id.
    r_old = await client.post(
        "/api/devices/live-activity",
        json={"token": "b1" * 32, "environment": "development"},
        headers=auth_headers,
    )
    old_id = r_old.json()["id"]

    # New-build registration after reinstall: install_id + new token.
    r_new = await client.post(
        "/api/devices/live-activity",
        json={"token": "b2" * 32, "environment": "development",
              "install_id": "install-sweep-1"},
        headers=auth_headers,
    )
    new_id = r_new.json()["id"]
    assert new_id != old_id

    from app.db import async_session_maker

    async with async_session_maker() as db:
        old = await db.get(LiveActivityDevice, old_id)
        new = await db.get(LiveActivityDevice, new_id)
        assert old.revoked_at is not None  # superseded reinstall leftover
        assert new.revoked_at is None

    # A wrongly-swept second REAL device self-heals: its next launch
    # re-registers its token and the upsert un-revokes the row.
    r_back = await client.post(
        "/api/devices/live-activity",
        json={"token": "b1" * 32, "environment": "development"},
        headers=auth_headers,
    )
    assert r_back.json()["id"] == old_id
    async with async_session_maker() as db:
        old = await db.get(LiveActivityDevice, old_id)
        assert old.revoked_at is None


# ── activity-token adoption (locally-started chat-turn cards) ─────


@pytest.mark.asyncio
async def test_activity_token_adopts_locally_started_chatturn(
    client, auth_headers, test_user_id,
):
    await client.post(
        "/api/devices/live-activity",
        json={"token": "c1" * 32, "environment": "production",
              "install_id": "install-adopt-1"},
        headers=auth_headers,
    )

    turn = "chatturn:cafe01234567"
    resp = await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": turn, "activity_push_token": "dd" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("adopted") is True

    from sqlalchemy import select
    from app.db import async_session_maker

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == turn)
        )).scalars().one()
        assert la.status == LA_STARTED
        assert la.activity_push_token == "dd" * 32
        assert la.apns_environment == "production"
        assert la.user_id == test_user_id


@pytest.mark.asyncio
async def test_activity_token_does_not_adopt_non_chatturn_missions(
    client, auth_headers,
):
    await client.post(
        "/api/devices/live-activity",
        json={"token": "c2" * 32, "environment": "development"},
        headers=auth_headers,
    )
    resp = await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "m-not-a-turn", "activity_push_token": "ee" * 32},
        headers=auth_headers,
    )
    body = resp.json()
    assert body["updated"] == 0
    assert "adopted" not in body

    from sqlalchemy import select
    from app.db import async_session_maker

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "m-not-a-turn")
        )).scalars().all()
        assert rows == []


# ── Dispatcher routing ────────────────────────────────────────────


async def _mk_user() -> str:
    from app.db import async_session_maker
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"la-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="LA Test",
            timezone="America/Toronto",
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


async def _mk_la_device(
    user_id: str, token: str = None, environment: str = "development",
) -> str:
    from app.db import async_session_maker

    device_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivityDevice(
            id=device_id, user_id=user_id,
            push_to_start_token=token or uuid.uuid4().hex + uuid.uuid4().hex,
            apns_environment=environment,
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return device_id


async def _enqueue(user_id: str, **overrides) -> str:
    from app.db import async_session_maker

    row_id = str(uuid.uuid4())
    fields = dict(
        id=row_id, user_id=user_id, source="agent",
        event_kind="progress", title="Autopilot: T", body="working",
        priority="low", idempotency_key=f"idem-{row_id}",
        status=NQ_QUEUED, created_at=datetime.utcnow(),
        data_json={"mission_id": "m-1", "mission_title": "T", "progress": 40},
    )
    fields.update(overrides)
    async with async_session_maker() as db:
        db.add(NotificationQueue(**fields))
        await db.commit()
    return row_id


async def _claim_and_dispatch(row_id: str) -> str:
    from app.db import async_session_maker

    now = datetime.utcnow()
    async with async_session_maker() as db:
        claimed = await nd._claim_batch(db, now)
        assert row_id in claimed
        return await nd._dispatch_row(db, row_id, now)


def _patch_apns(monkeypatch, sent: list, status: int = 200, reason: str = ""):
    async def fake_send(token, payload, *, environment="development", priority=10):
        sent.append({"token": token, "payload": payload,
                     "environment": environment, "priority": priority})
        return status, reason

    monkeypatch.setattr(las.apns_push, "send_live_activity", fake_send)
    monkeypatch.setattr(settings, "apns_key_b64", "eA==")
    monkeypatch.setattr(settings, "apns_key_id", "KEY123")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123")


@pytest.mark.asyncio
async def test_mission_started_starts_activity_on_each_device(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _mk_la_device(user_id)
    row_id = await _enqueue(
        user_id, event_kind="mission_started",
        title="🚀 Autopilot engaged: T", priority="default",
    )

    result = await _claim_and_dispatch(row_id)
    assert result == "sent"
    assert len(sent) == 2
    assert all(s["payload"]["aps"]["event"] == "start" for s in sent)
    assert all(s["priority"] == 10 for s in sent)

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "m-1")
        )).scalars().all()
        assert len(rows) == 2
        assert all(r.status == LA_STARTED for r in rows)


@pytest.mark.asyncio
async def test_mission_started_retry_does_not_duplicate_activities(monkeypatch):
    """At-least-once redelivery of mission_started must not spawn a
    second activity on a device that already has one."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    first = await _enqueue(user_id, event_kind="mission_started")
    assert await _claim_and_dispatch(first) == "sent"
    assert len(sent) == 1

    replay = await _enqueue(
        user_id, event_kind="mission_started", dedup_key=None,
    )
    result = await _claim_and_dispatch(replay)
    assert len(sent) == 1  # no second start push
    assert result.startswith("suppressed")


@pytest.mark.asyncio
async def test_mission_started_without_devices_suppresses_quietly(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    row_id = await _enqueue(user_id, event_kind="mission_started")

    result = await _claim_and_dispatch(row_id)
    assert result == "suppressed:live_activity_unavailable"
    assert sent == []


@pytest.mark.asyncio
async def test_progress_updates_activity_and_stays_suppressed(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    started = await _enqueue(user_id, event_kind="mission_started")
    await _claim_and_dispatch(started)
    sent.clear()

    row_id = await _enqueue(user_id)  # kind=progress, progress=40
    result = await _claim_and_dispatch(row_id)

    # The policy invariant holds — progress never becomes an alert push…
    assert result == "suppressed:progress_in_app_only"
    from app.db import async_session_maker
    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        assert row.status == NQ_SUPPRESSED
        assert row.channels_json["live_activity"]["delivered"] is True
    # …but the Live Activity moved, at the unbudgeted priority.
    assert len(sent) == 1
    assert sent[0]["payload"]["aps"]["event"] == "update"
    assert sent[0]["payload"]["aps"]["content-state"]["progress"] == 0.4
    assert sent[0]["priority"] == 5


@pytest.mark.asyncio
async def test_progress_never_moves_backwards(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(user_id, event_kind="mission_started"))
    await _claim_and_dispatch(await _enqueue(
        user_id, data_json={"mission_id": "m-1", "progress": 60},
    ))
    sent.clear()

    await _claim_and_dispatch(await _enqueue(
        user_id, data_json={"mission_id": "m-1", "progress": 30},
    ))
    assert sent[0]["payload"]["aps"]["content-state"]["progress"] == 0.6


@pytest.mark.asyncio
async def test_completed_ends_activity_and_counts_as_delivery(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    monkeypatch.setattr(
        nd.expo_push, "send_push_messages",
        lambda msgs: (_ for _ in ()).throw(AssertionError("no expo devices")),
    )
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(user_id, event_kind="mission_started"))
    sent.clear()

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    row_id = await _enqueue(
        user_id, event_kind="mission_completed",
        title="✅ “T” is done", priority="default",
        data_json={"mission_id": "m-1", "mission_title": "T", "progress": 100},
    )
    result = await _claim_and_dispatch(row_id)

    # APNs acceptance = delivery: no retry loop, row lands sent even
    # with zero Expo devices and no Telegram.
    assert result == "sent"
    assert sent[0]["payload"]["aps"]["event"] == "end"
    assert sent[0]["payload"]["aps"]["content-state"]["progress"] == 1.0
    assert sent[0]["payload"]["aps"]["alert"]["title"] == "✅ “T” is done"

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "m-1")
        )).scalars().one()
        assert la.status == LA_ENDED


@pytest.mark.asyncio
async def test_needs_input_updates_with_alert_but_keeps_activity_alive(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(user_id, event_kind="mission_started"))
    sent.clear()

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    row_id = await _enqueue(
        user_id, event_kind="needs_input", title="“T” needs you",
        priority="high",
        data_json={"mission_id": "m-1", "progress": 50},
    )
    result = await _claim_and_dispatch(row_id)
    assert result == "sent"
    aps = sent[0]["payload"]["aps"]
    assert aps["event"] == "update"
    assert aps["alert"]["title"] == "“T” needs you"
    assert sent[0]["priority"] == 10

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "m-1")
        )).scalars().one()
        assert la.status == LA_STARTED  # mission is waiting, card stays live


@pytest.mark.asyncio
async def test_dead_token_on_start_revokes_device(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent, status=410, reason="Unregistered")
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    row_id = await _enqueue(user_id, event_kind="mission_started")

    await _claim_and_dispatch(row_id)

    from app.db import async_session_maker
    async with async_session_maker() as db:
        device = await db.get(LiveActivityDevice, device_id)
        assert device.revoked_at is not None


@pytest.mark.asyncio
async def test_apns_unconfigured_skips_lane_gracefully(monkeypatch):
    monkeypatch.setattr(settings, "apns_key_b64", None)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    row_id = await _enqueue(user_id)  # progress

    result = await _claim_and_dispatch(row_id)
    assert result == "suppressed:progress_in_app_only"
    from app.db import async_session_maker
    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        assert row.channels_json["live_activity"]["reason"] == "apns_not_configured"


@pytest.mark.asyncio
async def test_non_mission_rows_bypass_lane_entirely(monkeypatch):
    called: list = []

    async def spy(db, row, now):
        called.append(row.id)
        return None

    monkeypatch.setattr(las, "handle_notification_row", spy)
    user_id = await _mk_user()

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)
    monkeypatch.setattr(settings, "notification_max_attempts", 1)

    row_id = await _enqueue(
        user_id, event_kind="generic", title="hi", priority="default",
        data_json=None,
    )
    await _claim_and_dispatch(row_id)
    assert called == [row_id]  # lane consulted, declined via None


# ── One-activity-per-device: preemption + self-healing ───────────


@pytest.mark.asyncio
async def test_new_start_preempts_previous_activity_on_device(monkeypatch):
    """Apple leaves multi-activity routing on a shared push-to-start
    token UNDEFINED — starting task B must first end task A's card."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": "m-A", "mission_title": "Task A", "progress": 0},
    ))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": "m-B", "mission_title": "Task B", "progress": 0},
    ))
    assert result == "sent"
    # First push ends A (immediate dismissal), second starts B.
    assert sent[0]["payload"]["aps"]["event"] == "end"
    assert sent[0]["payload"]["aps"]["dismissal-date"] <= int(datetime.utcnow().timestamp())
    assert sent[1]["payload"]["aps"]["event"] == "start"
    assert sent[1]["payload"]["aps"]["attributes"]["name"] == "m-B"

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        rows = {r.mission_id: r.status for r in (await db.execute(
            select(LiveActivity).where(LiveActivity.user_id == user_id)
        )).scalars().all()}
    assert rows == {"m-A": LA_ENDED, "m-B": LA_STARTED}


@pytest.mark.asyncio
async def test_progress_silently_restarts_preempted_card(monkeypatch):
    """A mission preempted by a quick job self-heals: its next progress
    heartbeat re-starts the card WITHOUT an alert banner."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": "m-A", "mission_title": "Task A", "progress": 10},
    ))
    # Quick job B preempts A.
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": "job-B", "mission_title": "Quick job"},
    ))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id,  # kind=progress
        data_json={"mission_id": "m-A", "mission_title": "Task A", "progress": 40},
    ))
    assert result == "suppressed:progress_in_app_only"
    # Job B's card was ended, then A restarted — quiet start: iOS 26
    # requires an alert config on every start, so it is synthesized
    # from the card content with no sound.
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end", "start"]
    assert "sound" not in sent[1]["payload"]["aps"]["alert"]
    assert sent[1]["payload"]["aps"]["attributes"]["name"] == "m-A"
    assert sent[1]["payload"]["aps"]["content-state"]["progress"] == 0.4

    # The (device, mission) row was REUSED (unique constraint) and is
    # started again.
    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        row = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "m-A")
        )).scalars().one()
        assert row.status == LA_STARTED
        assert row.last_progress == 40


async def _mk_started_rows(user_id: str, device_id: str, mission_ids, *, started_at=None):
    from app.db import async_session_maker

    async with async_session_maker() as db:
        for mid in mission_ids:
            db.add(LiveActivity(
                id=str(uuid.uuid4()), user_id=user_id, mission_id=mid,
                device_id=device_id, status=LA_STARTED,
                started_at=started_at or datetime.utcnow(),
            ))
        await db.commit()


@pytest.mark.asyncio
async def test_shared_token_ambiguity_resolved_by_preemption(monkeypatch):
    """2026-07-16 incident: a replica race left TWO started rows on one
    device; every tokenless send then skipped 'ambiguous_shared_token'
    FOREVER (nothing mutates the rows) and the alert retried to failure.
    The lane must now RESOLVE: preempt-end every other row (the shared
    token becomes unambiguous), then deliver the send."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(user_id, device_id, ("m-X", "m-Y"))

    await _claim_and_dispatch(await _enqueue(
        user_id, data_json={"mission_id": "m-X", "progress": 50},
    ))
    # End push for m-Y on the shared token, then m-X's update delivers.
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end", "update"]
    assert sent[1]["payload"]["aps"]["content-state"]["progress"] == 0.5

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        rows = {r.mission_id: r.status for r in (await db.execute(
            select(LiveActivity).where(LiveActivity.user_id == user_id)
        )).scalars().all()}
        assert rows == {"m-X": LA_STARTED, "m-Y": LA_ENDED}

        row = await db.get(NotificationQueue, (await db.execute(
            __import__("sqlalchemy").select(NotificationQueue.id).order_by(
                NotificationQueue.created_at.desc()).limit(1)
        )).scalar_one())
        frag = row.channels_json["live_activity"]["devices"][device_id]
        assert frag == {"status": "ok", "preempted": 1}


@pytest.mark.asyncio
async def test_terminal_send_resolves_ambiguity_end_to_end(monkeypatch):
    """The exact founder failure shape: two started rows + a terminal
    row for one of them. Must deliver (row → sent), and BOTH LiveActivity
    rows end — one preempted, one terminally."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(user_id, device_id, ("m-X", "m-Y"))

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    row_id = await _enqueue(
        user_id, event_kind="mission_completed", title="✅ Done: X",
        priority="default",
        data_json={"mission_id": "m-X", "mission_title": "X", "progress": 100},
    )
    result = await _claim_and_dispatch(row_id)
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end", "end"]  # preempt m-Y, then terminal end for m-X
    assert sent[1]["payload"]["aps"]["alert"]["title"] == "✅ Done: X"

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        assert row.status == NQ_SENT
        assert row.channels_json["live_activity"]["delivered"] is True
        rows = {r.mission_id: r.status for r in (await db.execute(
            select(LiveActivity).where(LiveActivity.user_id == user_id)
        )).scalars().all()}
        assert rows == {"m-X": LA_ENDED, "m-Y": LA_ENDED}


@pytest.mark.asyncio
async def test_stale_started_rows_swept_at_lane_entry_without_pushes(monkeypatch):
    """Apple hard-caps Live Activities at 8h — a 9h-old started row is
    dead on-device. The lane-entry GC must end it DB-only (zero pushes
    for it) and leave fresh rows untouched."""
    from datetime import timedelta

    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(
        user_id, device_id, ("m-old",),
        started_at=datetime.utcnow() - timedelta(hours=9),
    )
    await _mk_started_rows(user_id, device_id, ("m-new",))

    await _claim_and_dispatch(await _enqueue(
        user_id, data_json={"mission_id": "m-new", "progress": 20},
    ))
    # The GC beat the preemption path: only m-new's update went out —
    # no end push was wasted on the long-dead m-old card.
    assert [s["payload"]["aps"]["event"] for s in sent] == ["update"]

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        rows = {r.mission_id: r for r in (await db.execute(
            select(LiveActivity).where(LiveActivity.user_id == user_id)
        )).scalars().all()}
        assert rows["m-old"].status == LA_ENDED
        assert rows["m-old"].ended_at is not None
        assert rows["m-new"].status == LA_STARTED


@pytest.mark.asyncio
async def test_requeue_stuck_sweeps_stale_rows_for_all_users(monkeypatch):
    """Idle devices converge too: the dispatcher's _requeue_stuck pass
    runs the same 8h GC across ALL users — no queued row required."""
    from datetime import timedelta

    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(
        user_id, device_id, ("m-idle",),
        started_at=datetime.utcnow() - timedelta(hours=9),
    )

    from app.db import async_session_maker
    async with async_session_maker() as db:
        await nd._requeue_stuck(db, datetime.utcnow())

    from sqlalchemy import select
    async with async_session_maker() as db:
        row = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "m-idle")
        )).scalars().one()
        assert row.status == LA_ENDED
    assert sent == []  # DB-only, zero pushes


@pytest.mark.asyncio
async def test_terminal_restart_delivers_despite_foreign_started_row(monkeypatch):
    """Founder scenario regression (2026-07-16): a terminal row for a
    never-started chat turn while a FOREIGN mission's tokenless started
    row occupies the device. The _START_IF_MISSING restart must preempt
    the foreign card, start the turn card, and the fall-through end must
    deliver the alert."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(user_id, device_id, ("m-foreign",))

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    turn = "chatturn:deadbeef1234"
    row_id = await _enqueue(
        user_id, event_kind="mission_completed", title="Answer ready",
        priority="default",
        data_json={"mission_id": turn, "mission_title": "Answer",
                   "route": "chat", "kind": "chat_turn"},
    )
    result = await _claim_and_dispatch(row_id)
    assert result == "sent"
    # Preempt the foreign card, quietly start the turn card (iOS 26
    # requires an alert config on every start — synthesized, soundless),
    # then the end push carries the real banner.
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end", "start", "end"]
    assert "sound" not in sent[1]["payload"]["aps"]["alert"]  # quiet restart
    assert sent[2]["payload"]["aps"]["alert"]["title"] == "Answer ready"

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        assert row.status == NQ_SENT
        rows = {r.mission_id: r.status for r in (await db.execute(
            select(LiveActivity).where(LiveActivity.user_id == user_id)
        )).scalars().all()}
        assert rows == {"m-foreign": LA_ENDED, turn: LA_ENDED}


@pytest.mark.asyncio
async def test_job_timer_and_dismissal_payloads(monkeypatch):
    """Quick jobs: timer-driven bar in the start payload, and completed
    cards dismiss after data.dismiss_after_s."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    timer_ms = int((datetime.utcnow().timestamp() + 300) * 1000)

    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started", title="🛠 Working on: PM tools",
        data_json={"mission_id": "job-1", "mission_title": "PM tools",
                   "timer_end_ms": timer_ms, "urgent": True},
    ))
    cs = sent[0]["payload"]["aps"]["content-state"]
    assert cs["timerEndDateInMilliseconds"] == timer_ms
    assert "progress" not in cs  # timer wins
    sent.clear()

    now_ts = int(datetime.utcnow().timestamp())
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="✅ Done: PM tools",
        data_json={"mission_id": "job-1", "mission_title": "PM tools",
                   "progress": 100, "dismiss_after_s": 900, "urgent": True},
    ))
    aps = sent[0]["payload"]["aps"]
    assert aps["event"] == "end"
    assert aps["content-state"]["progress"] == 1.0
    assert now_ts + 800 <= aps["dismissal-date"] <= now_ts + 1000


# ── Agent-side producer ───────────────────────────────────────────


def test_parse_progress_marker_variants():
    from app.agent.routines.autopilot_handler import (
        parse_progress_value, parse_tick_markers,
    )

    m = parse_tick_markers(
        "did things\nAUTOPILOT_STATUS: working\n"
        "AUTOPILOT_PROGRESS: 45%\nAUTOPILOT_NOTE: n"
    )
    assert parse_progress_value(m.get("progress")) == 45
    assert parse_progress_value("120") == 100
    assert parse_progress_value("-5") == 0
    assert parse_progress_value("60 percent done") == 60
    assert parse_progress_value("about half") is None
    assert parse_progress_value(None) is None


def test_mission_started_kind_is_known():
    from app.db.models import KNOWN_NOTIFY_KINDS
    assert "mission_started" in KNOWN_NOTIFY_KINDS


# ── APNs environment self-heal ────────────────────────────────────
#
# BadDeviceToken is indistinguishable from a sandbox/production
# mismatch. Seen live 2026-07-16: a dev-provisioned Release build
# registered environment='production'; the first production-host send
# got BadDeviceToken and the device was WRONGLY revoked — every card
# after that died with no_live_activity_devices. The lane must retry
# the flipped host before declaring a token dead, and persist the
# heal on success.


def _patch_apns_env_sensitive(monkeypatch, sent: list, alive_env: str):
    """APNs double that only accepts pushes on `alive_env` — the other
    host answers 400 BadDeviceToken (the mismatch signature)."""
    async def fake_send(token, payload, *, environment="development", priority=10):
        sent.append({"token": token, "environment": environment,
                     "payload": payload, "priority": priority})
        if environment == alive_env:
            return 200, ""
        return 400, "BadDeviceToken"

    monkeypatch.setattr(las.apns_push, "send_live_activity", fake_send)
    monkeypatch.setattr(settings, "apns_key_b64", "eA==")
    monkeypatch.setattr(settings, "apns_key_id", "KEY123")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123")


@pytest.mark.asyncio
async def test_env_mismatch_selfheals_and_delivers(monkeypatch):
    sent: list = []
    _patch_apns_env_sensitive(monkeypatch, sent, alive_env="development")
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id, environment="production")
    row_id = await _enqueue(
        user_id, event_kind="mission_started",
        title="🚀 Autopilot engaged: T", priority="default",
    )

    result = await _claim_and_dispatch(row_id)
    assert result == "sent"
    # First try on the registered (wrong) env, retry on the flipped one.
    assert [s["environment"] for s in sent] == ["production", "development"]

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        device = await db.get(LiveActivityDevice, device_id)
        assert device.revoked_at is None, "mismatch must not revoke"
        assert device.apns_environment == "development", "heal must persist"
        rows = (await db.execute(
            select(LiveActivity).where(LiveActivity.device_id == device_id)
        )).scalars().all()
        assert len(rows) == 1 and rows[0].status == LA_STARTED


@pytest.mark.asyncio
async def test_dead_on_both_hosts_still_revokes(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent, status=410, reason="Unregistered")
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id, environment="development")
    row_id = await _enqueue(
        user_id, event_kind="mission_started",
        title="🚀 Autopilot engaged: T", priority="default",
    )

    await _claim_and_dispatch(row_id)
    # Tried both hosts, then revoked — a genuinely dead token must not
    # survive via the self-heal path.
    assert len(sent) == 2
    from app.db import async_session_maker
    async with async_session_maker() as db:
        device = await db.get(LiveActivityDevice, device_id)
        assert device.revoked_at is not None
