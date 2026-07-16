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
    # Job B's card was ended, then A restarted — silently (no alert).
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end", "start"]
    assert "alert" not in sent[1]["payload"]["aps"]
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


@pytest.mark.asyncio
async def test_shared_token_fallback_refused_when_ambiguous(monkeypatch):
    """Belt for the invariant: if two started rows ever coexist on a
    device (constructed here directly), updates must refuse the shared
    push-to-start token rather than hit an undefined card."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)

    from app.db import async_session_maker
    async with async_session_maker() as db:
        for mid in ("m-X", "m-Y"):
            db.add(LiveActivity(
                id=str(uuid.uuid4()), user_id=user_id, mission_id=mid,
                device_id=device_id, status=LA_STARTED,
                started_at=datetime.utcnow(),
            ))
        await db.commit()

    await _claim_and_dispatch(await _enqueue(
        user_id, data_json={"mission_id": "m-X", "progress": 50},
    ))
    assert sent == []  # nothing sent — ambiguous target

    from app.db import async_session_maker as asm
    async with asm() as db:
        row = await db.get(NotificationQueue, (await db.execute(
            __import__("sqlalchemy").select(NotificationQueue.id).order_by(
                NotificationQueue.created_at.desc()).limit(1)
        )).scalar_one())
        frag = row.channels_json["live_activity"]["devices"][device_id]
        assert frag == {"status": "skipped", "reason": "ambiguous_shared_token"}


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
