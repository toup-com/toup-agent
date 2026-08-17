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
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED, LiveActivity, LiveActivityDevice,
    NotificationQueue, User, NQ_QUEUED, NQ_SENDING, NQ_SENT,
    NQ_SUPPRESSED, NQ_FAILED,
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


def test_start_payload_orb_color_strict_hex_only():
    """attributes.orbColor carries the user's agent color to the
    widget face; the widget's hex parser assumes exactly '#RRGGBB', so
    anything looser is dropped (widget falls back to brand default)
    and a valid color also tints the progress bar to match."""
    p = apns_push.build_start_payload(
        mission_id="m-1", title="T", timestamp=1, orb_color="#2ECC71",
    )
    assert p["aps"]["attributes"]["orbColor"] == "#2ECC71"
    assert p["aps"]["attributes"]["progressViewTint"] == "#2ECC71"

    for bad in (None, "", "2ECC71", "#2ECC7", "#2ECC711", "#GGGGGG", "green"):
        p = apns_push.build_start_payload(
            mission_id="m-1", title="T", timestamp=1, orb_color=bad,
        )
        assert "orbColor" not in p["aps"]["attributes"], bad
        # Unset/invalid color keeps the fixed default tint.
        assert p["aps"]["attributes"]["progressViewTint"] == "#3B82F6"


@pytest.mark.asyncio
async def test_start_push_carries_users_agent_color(monkeypatch):
    """_send_start reads the LIVE agent_configs.agent_color (the same
    source the in-app orb renders) at send time."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    from app.db import async_session_maker
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=user_id, agent_color="#2ECC71"))
        await db.commit()

    await _claim_and_dispatch(await _enqueue(user_id, event_kind="mission_started"))
    attrs = sent[0]["payload"]["aps"]["attributes"]
    assert attrs["orbColor"] == "#2ECC71"
    assert attrs["progressViewTint"] == "#2ECC71"


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
    # with zero Expo devices and no Telegram. The banner+sound ride an
    # alerting update (documented surface); the end closes bannerless.
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update", "end"]
    assert sent[0]["payload"]["aps"]["alert"]["title"] == "✅ “T” is done"
    assert sent[0]["payload"]["aps"]["alert"]["sound"] == "default"
    assert sent[0]["payload"]["aps"]["content-state"]["progress"] == 1.0
    assert sent[1]["payload"]["aps"]["event"] == "end"
    assert sent[1]["payload"]["aps"]["content-state"]["progress"] == 1.0
    assert "alert" not in sent[1]["payload"]["aps"]

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
    # Preempt m-Y (end), alerting update for m-X on the now-unambiguous
    # shared token, then the bannerless terminal end.
    assert events == ["end", "update", "end"]
    assert sent[1]["payload"]["aps"]["alert"]["title"] == "✅ Done: X"
    assert sent[1]["payload"]["aps"]["alert"]["sound"] == "default"
    assert "alert" not in sent[2]["payload"]["aps"]

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
    # Preempt the foreign card, LOUD-start the turn card (the start
    # alert with sound is the surface iOS 26 provably renders), then a
    # bannerless end closes it.
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end", "start", "end"]
    assert sent[1]["payload"]["aps"]["alert"]["title"] == "Answer ready"
    assert sent[1]["payload"]["aps"]["alert"]["sound"] == "default"
    assert "alert" not in sent[2]["payload"]["aps"]

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
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update", "end"]  # alerting update, bannerless end
    aps = sent[1]["payload"]["aps"]
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


# ── P3 lifecycle: attempts cap, exception containment, ack, wedge ──


@pytest.mark.asyncio
async def test_requeue_stuck_fails_rows_at_the_attempts_cap(monkeypatch):
    """The 2026-07-18 incident class: a row whose every attempt raises
    never completes a dispatch, so the in-dispatch cap checks never
    run. _requeue_stuck must fail such rows at the cap instead of
    resurrecting them every 10 minutes forever (attempts=48 over 8h)."""
    from app.db import async_session_maker

    user_id = await _mk_user()
    stale_claim = datetime.utcnow() - timedelta(minutes=30)
    capped_id = await _enqueue(
        user_id, status=NQ_SENDING, attempts=settings.notification_max_attempts,
        claimed_at=stale_claim,
    )
    fresh_id = await _enqueue(
        user_id, status=NQ_SENDING, attempts=1, claimed_at=stale_claim,
        idempotency_key=f"idem-fresh-{uuid.uuid4()}",
    )

    async with async_session_maker() as db:
        await nd._requeue_stuck(db, datetime.utcnow())

    async with async_session_maker() as db:
        capped = await db.get(NotificationQueue, capped_id)
        fresh = await db.get(NotificationQueue, fresh_id)
        assert capped.status == NQ_FAILED
        assert capped.last_error == "stuck_requeue_exhausted"
        assert fresh.status == NQ_QUEUED  # under the cap: normal requeue


@pytest.mark.asyncio
async def test_dispatch_exception_requeues_row_with_recorded_error(monkeypatch):
    """An exception mid-dispatch must not strand the row in 'sending'
    with a NULL last_error — it re-queues on the normal backoff with
    the exception recorded, so the cap applies and the incident is
    readable off the row."""
    from app.db import async_session_maker

    user_id = await _mk_user()
    await _mk_la_device(user_id)
    row_id = await _enqueue(user_id)  # kind=progress → LA lane runs first

    async def boom(db, row, now):
        raise RuntimeError("simulated wedge")

    monkeypatch.setattr(nd.live_activity_service, "handle_notification_row", boom)
    monkeypatch.setattr(settings, "notification_dispatch_enabled", True)

    stats = await nd.run_notification_dispatch()
    assert stats.get("errored") == 1

    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        assert row.status == NQ_QUEUED
        assert row.last_error.startswith("dispatch_exception: RuntimeError")
        assert row.scheduled_for is not None  # backoff applied


@pytest.mark.asyncio
async def test_chatturn_rows_swept_after_30_minutes():
    """A chatturn row still 'started' after 30 min is a wedge (turns
    live minutes; local cards carry a 10-min staleDate) — the sweep
    must end it long before Apple's 8h cap, without touching younger
    turn rows or ordinary missions."""
    from app.db import async_session_maker
    from app.services.live_activity_service import sweep_stale_activities

    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    old = datetime.utcnow() - timedelta(minutes=31)
    await _mk_started_rows(user_id, device_id, ("chatturn:aged00000001",),
                           started_at=old)
    await _mk_started_rows(user_id, device_id, ("m-long-mission",),
                           started_at=old)
    await _mk_started_rows(user_id, device_id, ("chatturn:fresh0000001",))

    async with async_session_maker() as db:
        swept = await sweep_stale_activities(db, datetime.utcnow(), user_id=user_id)
    assert swept == 1

    async with async_session_maker() as db:
        rows = {r.mission_id: r.status for r in (await db.execute(
            select(LiveActivity).where(LiveActivity.user_id == user_id)
        )).scalars().all()}
    assert rows["chatturn:aged00000001"] == LA_ENDED
    assert rows["m-long-mission"] == LA_STARTED  # 8h rule untouched
    assert rows["chatturn:fresh0000001"] == LA_STARTED


@pytest.mark.asyncio
async def test_activity_token_does_not_revive_ended_turn(
    client, auth_headers, test_user_id,
):
    """The 8h-wedge shape: the platform already ENDED the turn, then
    the app's token report arrives. The token is stored (post-hoc ack
    pushes can still reach the card) but the row must NOT be revived —
    nothing ever ends a revived row for a finished turn."""
    from app.db import async_session_maker

    await client.post(
        "/api/devices/live-activity",
        json={"token": "c9" * 32, "environment": "development",
              "install_id": "install-wedge-1"},
        headers=auth_headers,
    )
    turn = "chatturn:wedge0000001"
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id=turn,
            device_id=device_id, status=LA_ENDED,
            started_at=datetime.utcnow(), ended_at=datetime.utcnow(),
        ))
        await db.commit()

    resp = await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": turn, "activity_push_token": "df" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("already_ended") is True

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == turn)
        )).scalars().one()
        assert la.status == LA_ENDED  # NOT revived
        assert la.activity_push_token == "df" * 32  # token stored


@pytest.mark.asyncio
async def test_ack_ends_cards_and_suppresses_pending_rows(
    client, auth_headers, test_user_id, monkeypatch,
):
    """Tap-ack: ends the mission's cards (push + DB) and suppresses its
    pending completed/progress rows; needs_input stays — seen is not
    answered."""
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)

    await client.post(
        "/api/devices/live-activity",
        json={"token": "ca" * 32, "environment": "development",
              "install_id": "install-ack-1"},
        headers=auth_headers,
    )
    mission = "reminder:ack00000001"
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id=mission,
            device_id=device_id, status=LA_STARTED,
            started_at=datetime.utcnow(),
        ))
        for kind, idem in (("mission_completed", "ack-c"), ("needs_input", "ack-n")):
            db.add(NotificationQueue(
                id=str(uuid.uuid4()), user_id=test_user_id, source="agent",
                event_kind=kind, title="t", priority="high",
                idempotency_key=idem, status=NQ_QUEUED,
                created_at=datetime.utcnow(),
                data_json={"mission_id": mission},
            ))
        await db.commit()

    resp = await client.post(
        "/api/devices/live-activity/ack",
        json={"mission_id": mission},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ended"] == 1
    assert body["suppressed"] == 1
    # The end push is immediate-dismissal, bannerless.
    assert sent[-1]["payload"]["aps"]["event"] == "end"
    assert "alert" not in sent[-1]["payload"]["aps"]

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_ENDED
        rows = {r.idempotency_key: r.status for r in (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.user_id == test_user_id)
        )).scalars().all()}
        assert rows["ack-c"] == NQ_SUPPRESSED
        assert rows["ack-n"] == NQ_QUEUED  # needs_input untouched


@pytest.mark.asyncio
async def test_active_missions_lists_started_only(
    client, auth_headers, test_user_id,
):
    """The foreground-reconcile contract: exactly the user's STARTED
    mission ids, nothing ended, nobody else's."""
    from app.db import async_session_maker

    await client.post(
        "/api/devices/live-activity",
        json={"token": "cb" * 32, "environment": "development",
              "install_id": "install-recon-1"},
        headers=auth_headers,
    )
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id,
            mission_id="reminder:recon-live", device_id=device_id,
            status=LA_STARTED, started_at=datetime.utcnow(),
        ))
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id,
            mission_id="reminder:recon-done", device_id=device_id,
            status=LA_ENDED, started_at=datetime.utcnow(),
            ended_at=datetime.utcnow(),
        ))
        await db.commit()

    resp = await client.get(
        "/api/devices/live-activity/active-missions", headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    missions = resp.json()["missions"]
    assert "reminder:recon-live" in missions
    assert "reminder:recon-done" not in missions


# ── P4 parity: update_only progress lane ──────────────────────────


@pytest.mark.asyncio
async def test_update_only_progress_never_starts_a_card(monkeypatch):
    """Turn/job status beacons emit on EVERY turn now — safe only
    because update_only rows refresh an existing card and never
    silently start one (no card on ordinary foreground turns)."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    result = await _claim_and_dispatch(await _enqueue(
        user_id,  # kind=progress, no card started
        data_json={"mission_id": "chatturn:p4nocard0001", "mission_title": "T",
                   "progress": 40, "update_only": True},
    ))
    assert result == "suppressed:progress_in_app_only"
    assert sent == []  # no restart, no start push — nothing


@pytest.mark.asyncio
async def test_update_only_progress_updates_existing_card(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(user_id, device_id, ("chatturn:p4card00001",))

    result = await _claim_and_dispatch(await _enqueue(
        user_id,
        data_json={"mission_id": "chatturn:p4card00001",
                   "mission_title": "Find flights to Lisbon",
                   "progress": 40, "update_only": True},
        body="Searching the web…",
    ))
    assert result == "suppressed:progress_in_app_only"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update"]
    cs = sent[0]["payload"]["aps"]["content-state"]
    assert cs["title"] == "Find flights to Lisbon"
    assert cs["subtitle"] == "Searching the web…"
    assert cs["progress"] == 0.4


@pytest.mark.asyncio
async def test_plain_progress_still_restarts_preempted_card(monkeypatch):
    """Autopilot mission markers (no update_only) keep the self-heal:
    a preempted mission card comes back on the next heartbeat."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    result = await _claim_and_dispatch(await _enqueue(
        user_id,
        data_json={"mission_id": "m-heal", "mission_title": "T", "progress": 40},
    ))
    assert result == "suppressed:progress_in_app_only"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["start"]  # silent self-heal restart preserved


# ── Alarm-class rows: ring-until-acked chain ──────────────────────


@pytest.mark.asyncio
async def test_alarm_fire_rings_open_card_and_chains_next_ring(monkeypatch):
    """Ring 1 of a reminder fire: alerting update with the DEFAULT
    tone (iOS has never honored a named sound on a Live Activity push
    alert — the producer's file must never reach the wire), card left
    OPEN, and the next ring booked as an LA-only row one gap out."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    monkeypatch.setattr(
        nd.expo_push, "send_push_messages",
        lambda msgs: (_ for _ in ()).throw(AssertionError("no expo devices")),
    )

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": "reminder:r1", "mission_title": "⏰ Ring",
                   "silent": True},
    ))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="⏰ Ring — now",
        body="Deep probe", priority="high",
        data_json={"mission_id": "reminder:r1", "mission_title": "⏰ Ring",
                   "sound": "toup_alarm.caf", "dismiss_after_s": 120,
                   "subtitle": "Deep probe"},
    ))
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update"]  # card stays OPEN for ring 2 / the ack
    aps = sent[0]["payload"]["aps"]
    assert aps["alert"]["title"] == "⏰ Ring — now"
    assert aps["alert"]["sound"] == "default"

    from app.db import async_session_maker
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "reminder:r1")
        )).scalars().one()
        assert la.status == LA_STARTED
        ring2 = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key == "alarm-ring:reminder:r1:2",
            )
        )).scalars().one()
        assert ring2.data_json["realert_seq"] == 2
        assert ring2.data_json["la_only"] is True
        assert ring2.data_json["no_agent_fallback"] is True
        assert ring2.scheduled_for is not None
        assert ring2.event_kind == "mission_completed"
        assert ring2.title == "⏰ Ring — now"


@pytest.mark.asyncio
async def test_alarm_last_ring_ends_card_and_stops_chain(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": "reminder:r1", "mission_title": "⏰ Ring",
                   "silent": True},
    ))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="⏰ Ring — now",
        priority="high",
        data_json={"mission_id": "reminder:r1", "mission_title": "⏰ Ring",
                   "sound": "toup_alarm.caf", "dismiss_after_s": 120,
                   "realert_seq": 3, "la_only": True,
                   "no_agent_fallback": True},
    ))
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update", "end"]  # final ring closes the card
    assert sent[0]["payload"]["aps"]["alert"]["sound"] == "default"
    assert "alert" not in sent[1]["payload"]["aps"]

    from app.db import async_session_maker
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "reminder:r1")
        )).scalars().one()
        assert la.status == LA_ENDED
        ring4 = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key == "alarm-ring:reminder:r1:4",
            )
        )).scalars().all()
        assert ring4 == []


@pytest.mark.asyncio
async def test_realert_after_ack_stays_silent(monkeypatch):
    """The ack (or a user swipe) ended the card between rings: a
    chained ring must NOT loud-restart it, must NOT fall back to
    Expo/chat, and must NOT book another ring — suppressed quietly."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    monkeypatch.setattr(
        nd.expo_push, "send_push_messages",
        lambda msgs: (_ for _ in ()).throw(AssertionError("expo must not fire")),
    )

    async def fail_fallback(db, row):
        raise AssertionError("agent fallback must not fire")

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fail_fallback)

    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": "reminder:r1", "mission_title": "⏰ Ring",
                   "silent": True},
    ))
    from app.db import async_session_maker
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "reminder:r1")
        )).scalars().one()
        la.status = LA_ENDED
        la.ended_at = datetime.utcnow()
        await db.commit()
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="⏰ Ring — now",
        priority="high",
        data_json={"mission_id": "reminder:r1", "mission_title": "⏰ Ring",
                   "sound": "toup_alarm.caf", "realert_seq": 2,
                   "la_only": True, "no_agent_fallback": True},
    ))
    assert result == "suppressed:la_only_undeliverable"
    assert sent == []

    async with async_session_maker() as db:
        ring3 = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key == "alarm-ring:reminder:r1:3",
            )
        )).scalars().all()
        assert ring3 == []


@pytest.mark.asyncio
async def test_plain_completed_does_not_chain_rings(monkeypatch):
    """Non-alarm terminal rows keep the classic single-bang contract:
    alerting update + bannerless end, no follow-up ring rows."""
    sent: list = []
    _patch_apns(monkeypatch, sent)

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(user_id, event_kind="mission_started"))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="✅ Done",
        priority="default",
        data_json={"mission_id": "m-1", "mission_title": "T", "progress": 100},
    ))
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update", "end"]

    from app.db import async_session_maker
    async with async_session_maker() as db:
        rings = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key.like("alarm-ring:%"),
            )
        )).scalars().all()
        assert rings == []


# ── 2026-07-22 reminder pipeline incident: REMINDER WINS in the
#    restart lane, pre-fire tap exemption, fired state, alarm-owned ──


@pytest.mark.asyncio
async def test_answer_restart_yields_to_live_countdown(monkeypatch):
    """The 'Answer ready' chat-turn terminal row must NOT push-to-start
    a card over a live reminder countdown (it stole the Dynamic Island
    for its 5-min linger and its preempt orphaned the countdown row —
    founder repro 2026-07-22)."""
    sent: list = []
    _patch_apns(monkeypatch, sent)

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(user_id, device_id, ("reminder:cd0001",))
    sent.clear()

    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="Answer ready",
        body="Done — I'll remind you at 3:50 PM.", priority="high",
        data_json={"mission_id": "chatturn:beefbeef0001",
                   "mission_title": "Remind me in 4 min",
                   "kind": "chat_turn", "progress": 100,
                   "dismiss_after_s": 300},
    ))
    # No start push — the countdown keeps the island.
    assert [s["payload"]["aps"]["event"] for s in sent] == []

    from app.db import async_session_maker
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(
                LiveActivity.mission_id == "reminder:cd0001")
        )).scalars().one()
        assert la.status == LA_STARTED  # countdown row untouched


@pytest.mark.asyncio
async def test_preempt_refuses_to_end_reminder_for_chatturn():
    """Belt-and-braces: even a direct preempt call must never end a
    reminder:* row to make room for a chat-turn card."""
    from app.db import async_session_maker

    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    await _mk_started_rows(user_id, device_id, ("reminder:cd0002",))

    async with async_session_maker() as db:
        device = (await db.execute(
            select(LiveActivityDevice).where(
                LiveActivityDevice.id == device_id)
        )).scalars().one()
        preempted = await las._preempt_device(
            db, device, "chatturn:cafecafe0001", datetime.utcnow(),
        )
        await db.commit()
        assert preempted == 0
        la = (await db.execute(
            select(LiveActivity).where(
                LiveActivity.mission_id == "reminder:cd0002")
        )).scalars().one()
        assert la.status == LA_STARTED


@pytest.mark.asyncio
async def test_prefire_countdown_tap_is_navigation_only(
    client, auth_headers, test_user_id, monkeypatch,
):
    """A tap on a still-counting reminder card deep-links WITHOUT
    acking: card stays, nothing suppressed (the countdown and fired
    cards share the same ?mission= link — phase comes from the newest
    arm row's timer)."""
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)

    await client.post(
        "/api/devices/live-activity",
        json={"token": "cc" * 32, "environment": "development",
              "install_id": "install-prefire-1"},
        headers=auth_headers,
    )
    mission = "reminder:prefire00001"
    future_ms = int((datetime.utcnow().timestamp() + 240) * 1000)
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id=mission,
            device_id=device_id, status=LA_STARTED,
            started_at=datetime.utcnow(),
        ))
        db.add(NotificationQueue(
            id=str(uuid.uuid4()), user_id=test_user_id, source="agent",
            event_kind="mission_started", title="⏰ TV", priority="default",
            idempotency_key="prefire-arm", status="sent",
            created_at=datetime.utcnow(),
            data_json={"mission_id": mission, "timer_end_ms": future_ms,
                       "silent": True},
        ))
        await db.commit()

    resp = await client.post(
        "/api/devices/live-activity/ack",
        json={"mission_id": mission},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("pre_fire") is True
    assert body["ended"] == 0 and body["suppressed"] == 0
    assert sent == []  # no end push

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_STARTED


@pytest.mark.asyncio
async def test_postfire_tap_acks_and_future_nonring_rows_survive(
    client, auth_headers, test_user_id, monkeypatch,
):
    """After the timer passed, the tap keeps full ack semantics: cards
    end, queued rings (future-scheduled by design) are suppressed —
    but a future-scheduled NON-ring row for the mission survives."""
    from datetime import timedelta as _td
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)

    await client.post(
        "/api/devices/live-activity",
        json={"token": "cd" * 32, "environment": "development",
              "install_id": "install-postfire-1"},
        headers=auth_headers,
    )
    mission = "reminder:postfire0001"
    past_ms = int((datetime.utcnow().timestamp() - 30) * 1000)
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id=mission,
            device_id=device_id, status=LA_STARTED,
            started_at=datetime.utcnow(),
        ))
        db.add(NotificationQueue(
            id=str(uuid.uuid4()), user_id=test_user_id, source="agent",
            event_kind="mission_started", title="⏰ TV", priority="default",
            idempotency_key="postfire-arm", status="sent",
            created_at=datetime.utcnow(),
            data_json={"mission_id": mission, "timer_end_ms": past_ms,
                       "silent": True},
        ))
        db.add(NotificationQueue(  # queued ring 2 — MUST be suppressed
            id=str(uuid.uuid4()), user_id=test_user_id, source="platform",
            event_kind="mission_completed", title="⏰ TV — now",
            priority="high", idempotency_key=f"alarm-ring:{mission}:2",
            status=NQ_QUEUED, created_at=datetime.utcnow(),
            scheduled_for=datetime.utcnow() + _td(seconds=20),
            data_json={"mission_id": mission, "realert_seq": 2,
                       "la_only": True, "sound": "toup_alarm.caf"},
        ))
        db.add(NotificationQueue(  # future ordinary row — MUST survive
            id=str(uuid.uuid4()), user_id=test_user_id, source="agent",
            event_kind="mission_completed", title="tomorrow",
            priority="high", idempotency_key="postfire-future",
            status=NQ_QUEUED, created_at=datetime.utcnow(),
            scheduled_for=datetime.utcnow() + _td(hours=20),
            data_json={"mission_id": mission},
        ))
        await db.commit()

    resp = await client.post(
        "/api/devices/live-activity/ack",
        json={"mission_id": mission},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("pre_fire") is None
    assert body["ended"] == 1
    assert body["suppressed"] == 1  # the ring; not the future row

    async with async_session_maker() as db:
        rows = {r.idempotency_key: r.status for r in (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.user_id == test_user_id)
        )).scalars().all()}
        assert rows[f"alarm-ring:{mission}:2"] == NQ_SUPPRESSED
        assert rows["postfire-future"] == NQ_QUEUED


@pytest.mark.asyncio
async def test_alarm_fire_carries_fired_state_not_progress(monkeypatch):
    """Fire pushes render the RINGING presentation: content-state.fired
    on the alerting update (card present) and on the loud restart (card
    missing) — never a 0%/100% progress bar."""
    sent: list = []
    _patch_apns(monkeypatch, sent)

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    # Card present → alerting update carries fired, no progress.
    await _mk_started_rows(user_id, device_id, ("reminder:fired001",))
    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="⏰ TV — now",
        priority="high",
        data_json={"mission_id": "reminder:fired001", "mission_title": "⏰ TV",
                   "sound": "toup_alarm.caf", "subtitle": "Turn off the TV."},
    ))
    upd_cs = sent[0]["payload"]["aps"]["content-state"]
    assert upd_cs.get("fired") is True
    assert "progress" not in upd_cs and "timerEndDateInMilliseconds" not in upd_cs

    # Card missing → ring-1 loud restart start payload carries fired.
    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_completed", title="⏰ TV — now",
        priority="high",
        data_json={"mission_id": "reminder:fired002", "mission_title": "⏰ TV",
                   "sound": "toup_alarm.caf", "subtitle": "Turn off the TV."},
    ))
    start = next(s for s in sent
                 if s["payload"]["aps"]["event"] == "start")
    start_cs = start["payload"]["aps"]["content-state"]
    assert start_cs.get("fired") is True
    assert "progress" not in start_cs


@pytest.mark.asyncio
async def test_alarm_owned_fire_stays_quiet_and_chains_nothing(
    client, auth_headers, test_user_id, monkeypatch,
):
    """Once the app reports AlarmKit ownership (retiring the countdown
    card), the fire lane must not loud-restart or ring — the device
    alarm is already ringing through silent/Focus."""
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    await client.post(
        "/api/devices/live-activity",
        json={"token": "ce" * 32, "environment": "development",
              "install_id": "install-owned-1"},
        headers=auth_headers,
    )
    mission = "reminder:owned0001"
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id=mission,
            device_id=device_id, status=LA_STARTED,
            started_at=datetime.utcnow(),
        ))
        await db.commit()

    # The app arms its device alarm and reports ownership. The row
    # STAYS started — it is the anchor every REMINDER-WINS guard keys
    # off (ending it would let the next chat turn start a card over
    # the AlarmKit countdown); only the flag flips.
    resp = await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["updated"] == 1

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_STARTED  # guard anchor survives
        assert la.alarm_owned_at is not None

    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        test_user_id, event_kind="mission_completed", title="⏰ TV — now",
        priority="high",
        data_json={"mission_id": mission, "mission_title": "⏰ TV",
                   "sound": "toup_alarm.caf", "subtitle": "Turn off the TV."},
    ))
    # AlarmKit owns the fire: the ghost row is closed QUIETLY — a
    # bannerless fired end, never a loud restart or alerting update.
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end"]
    assert "alert" not in sent[0]["payload"]["aps"]
    assert sent[0]["payload"]["aps"]["content-state"].get("fired") is True

    async with async_session_maker() as db:
        rings = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key.like(
                    f"alarm-ring:{mission}:%"),
            )
        )).scalars().all()
        assert rings == []  # chain never started


@pytest.mark.asyncio
async def test_fresh_countdown_start_clears_alarm_ownership(monkeypatch):
    """A new countdown cycle re-arms from scratch: the platform start
    resets alarm_owned_at so a stale ownership flag can never mute a
    later fire the device no longer owns."""
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    device_id = await _mk_la_device(user_id)
    mission = "reminder:recycle01"
    async with async_session_maker() as db:
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=user_id, mission_id=mission,
            device_id=device_id, status=LA_ENDED,
            started_at=datetime.utcnow(), ended_at=datetime.utcnow(),
            alarm_owned_at=datetime.utcnow(),
        ))
        await db.commit()

    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started",
        data_json={"mission_id": mission, "mission_title": "⏰ TV",
                   "silent": True, "timer_end_ms":
                   int((datetime.utcnow().timestamp() + 300) * 1000)},
    ))
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_STARTED
        assert la.alarm_owned_at is None


@pytest.mark.asyncio
async def test_register_stores_alarm_observability(
    client, auth_headers, test_user_id,
):
    """alarm_auth/alarms_armed ride the normal registration — the
    one-query answer to 'why was the fire silent on this phone'."""
    from app.db import async_session_maker

    resp = await client.post(
        "/api/devices/live-activity",
        json={"token": "cf" * 32, "environment": "development",
              "install_id": "install-obs-1",
              "alarm_auth": "authorized", "alarms_armed": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    async with async_session_maker() as db:
        device = (await db.execute(
            select(LiveActivityDevice).where(
                LiveActivityDevice.push_to_start_token == "cf" * 32)
        )).scalars().one()
        assert device.alarm_auth == "authorized"
        assert device.alarms_armed == 3


@pytest.mark.asyncio
async def test_owned_fire_consumes_ownership_and_next_fire_rings_loud(
    client, auth_headers, test_user_id, monkeypatch,
):
    """F1 regression (review 2026-07-23): the fire consumes the
    alarm-ownership cycle. A later fire of the same mission that never
    re-armed (reschedule beyond the countdown window — no card, no
    re-report) must ring LOUD, not be muted by the stale flag."""
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    await client.post(
        "/api/devices/live-activity",
        json={"token": "d0" * 32, "environment": "development",
              "install_id": "install-cycle-1"},
        headers=auth_headers,
    )
    mission = "reminder:cycle0001"
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id=mission,
            device_id=device_id, status=LA_STARTED,
            started_at=datetime.utcnow(),
        ))
        await db.commit()
    resp = await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission}, headers=auth_headers,
    )
    assert resp.json()["updated"] == 1

    # Fire 1: quiet owned end — and the flag is CONSUMED.
    await _claim_and_dispatch(await _enqueue(
        test_user_id, event_kind="mission_completed", title="⏰ R — now",
        priority="high",
        data_json={"mission_id": mission, "mission_title": "⏰ R",
                   "sound": "toup_alarm.caf", "subtitle": "ring"},
    ))
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_ENDED
        assert la.alarm_owned_at is None  # cycle consumed

    # Fire 2 (rescheduled far out, never re-armed): must restart LOUD.
    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        test_user_id, event_kind="mission_completed", title="⏰ R — now",
        priority="high",
        data_json={"mission_id": mission, "mission_title": "⏰ R",
                   "sound": "toup_alarm.caf", "subtitle": "ring again"},
    ))
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert "start" in events  # loud restart happened
    start = next(s for s in sent if s["payload"]["aps"]["event"] == "start")
    assert "alert" in start["payload"]["aps"]

    async with async_session_maker() as db:
        ring2 = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key == f"alarm-ring:{mission}:2",
            )
        )).scalars().all()
        assert len(ring2) == 1  # chain armed again


@pytest.mark.asyncio
async def test_postfire_ack_suppresses_retrying_fire_row(
    client, auth_headers, test_user_id, monkeypatch,
):
    """F2 regression (review 2026-07-23): a ring-1 fire row sitting in
    retry backoff (attempts>=1, future scheduled_for) is part of the
    in-progress alert — a post-fire tap must suppress it, or it
    loud-restarts minutes after the user already acknowledged."""
    from datetime import timedelta as _td
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)

    await client.post(
        "/api/devices/live-activity",
        json={"token": "d1" * 32, "environment": "development",
              "install_id": "install-retryack-1"},
        headers=auth_headers,
    )
    mission = "reminder:retryack01"
    past_ms = int((datetime.utcnow().timestamp() - 30) * 1000)
    async with async_session_maker() as db:
        device_id = (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.user_id == test_user_id)
        )).scalars().first()
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id, mission_id=mission,
            device_id=device_id, status=LA_STARTED,
            started_at=datetime.utcnow(),
        ))
        db.add(NotificationQueue(
            id=str(uuid.uuid4()), user_id=test_user_id, source="agent",
            event_kind="mission_started", title="⏰ R", priority="default",
            idempotency_key="retryack-arm", status="sent",
            created_at=datetime.utcnow(),
            data_json={"mission_id": mission, "timer_end_ms": past_ms,
                       "silent": True},
        ))
        db.add(NotificationQueue(  # ring-1 fire row in retry backoff
            id=str(uuid.uuid4()), user_id=test_user_id, source="agent",
            event_kind="mission_completed", title="⏰ R — now",
            priority="high", idempotency_key="retryack-fire",
            status=NQ_QUEUED, attempts=1,
            created_at=datetime.utcnow(),
            scheduled_for=datetime.utcnow() + _td(minutes=8),
            data_json={"mission_id": mission, "sound": "toup_alarm.caf"},
        ))
        await db.commit()

    resp = await client.post(
        "/api/devices/live-activity/ack",
        json={"mission_id": mission}, headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["suppressed"] == 1

    async with async_session_maker() as db:
        fire = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key == "retryack-fire")
        )).scalars().one()
        assert fire.status == NQ_SUPPRESSED


# ── 2026-07-23 round 4: pre-start alarm-owned MARKER rows ─────────
# The app arms the AlarmKit countdown ~1-2s after the reminder tool
# runs and reports ownership token-scoped BEFORE the platform's own
# countdown push dispatches — the marker makes that push a no-op so
# the user never sees a duplicate card flash.


async def _register_and_get_device(client, auth_headers, test_user_id, token):
    await client.post(
        "/api/devices/live-activity",
        json={"token": token, "environment": "development",
              "install_id": f"install-{token[:8]}"},
        headers=auth_headers,
    )
    from app.db import async_session_maker
    async with async_session_maker() as db:
        return (await db.execute(
            select(LiveActivityDevice.id).where(
                LiveActivityDevice.push_to_start_token == token)
        )).scalar_one()


@pytest.mark.asyncio
async def test_alarm_owned_pre_start_inserts_marker_and_dedups_start(
    client, auth_headers, test_user_id, monkeypatch,
):
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)
    token = "e0" * 32
    await _register_and_get_device(client, auth_headers, test_user_id, token)

    mission = "reminder:marker0001"
    resp = await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission, "token": token},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("marker") == "inserted"

    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_STARTED
        assert la.alarm_owned_at is not None

    # The platform's own countdown push now dedups per-device: no card.
    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        test_user_id, event_kind="mission_started",
        data_json={"mission_id": mission, "mission_title": "⏰ M",
                   "silent": True, "timer_end_ms":
                   int((datetime.utcnow().timestamp() + 120) * 1000)},
    ))
    assert sent == []  # zero APNs sends — the marker owns the slot
    async with async_session_maker() as db:
        count = len((await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().all())
        assert count == 1  # no duplicate row either


@pytest.mark.asyncio
async def test_alarm_owned_marker_anchors_reminder_wins(
    client, auth_headers, test_user_id, monkeypatch,
):
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)
    token = "e1" * 32
    await _register_and_get_device(client, auth_headers, test_user_id, token)
    mission = "reminder:marker0002"
    await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission, "token": token},
        headers=auth_headers,
    )

    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        test_user_id, event_kind="mission_started",
        data_json={"mission_id": "chatturn:markerturn01",
                   "mission_title": "Working", "kind": "chat_turn"},
    ))
    assert sent == []  # chat turn yields to the marker — no preempt/start
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_STARTED  # marker untouched


@pytest.mark.asyncio
async def test_alarm_owned_conflict_with_started_foreign_row_is_flag_only(
    client, auth_headers, test_user_id, monkeypatch,
):
    from app.db import async_session_maker

    token = "e2" * 32
    device_id = await _register_and_get_device(
        client, auth_headers, test_user_id, token)
    async with async_session_maker() as db:
        db.add(LiveActivity(
            id=str(uuid.uuid4()), user_id=test_user_id,
            mission_id="chatturn:occupies01", device_id=device_id,
            status=LA_STARTED, started_at=datetime.utcnow(),
        ))
        await db.commit()

    mission = "reminder:marker0003"
    resp = await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission, "token": token},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("marker") == "skipped"
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().all()
        assert rows == []  # nothing inserted; slot respected


@pytest.mark.asyncio
async def test_alarm_owned_false_ends_started_marker(
    client, auth_headers, test_user_id,
):
    from app.db import async_session_maker

    token = "e3" * 32
    await _register_and_get_device(client, auth_headers, test_user_id, token)
    mission = "reminder:marker0004"
    await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission, "token": token},
        headers=auth_headers,
    )
    resp = await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission, "token": token, "owned": False},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_ENDED  # slot freed
        assert la.alarm_owned_at is None


@pytest.mark.asyncio
async def test_fire_lane_quiet_ends_marker_without_ring_chain(
    client, auth_headers, test_user_id, monkeypatch,
):
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    token = "e4" * 32
    await _register_and_get_device(client, auth_headers, test_user_id, token)
    mission = "reminder:marker0005"
    await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission, "token": token},
        headers=auth_headers,
    )

    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        test_user_id, event_kind="mission_completed", title="⏰ M — now",
        priority="high",
        data_json={"mission_id": mission, "mission_title": "⏰ M",
                   "sound": "toup_alarm.caf", "subtitle": "ring"},
    ))
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["end"]  # quiet fired end, no loud restart/update
    assert "alert" not in sent[0]["payload"]["aps"]
    assert sent[0]["payload"]["aps"]["content-state"].get("fired") is True
    async with async_session_maker() as db:
        rings = (await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.idempotency_key.like(
                    f"alarm-ring:{mission}:%"),
            )
        )).scalars().all()
        assert rings == []


@pytest.mark.asyncio
async def test_reminder_cancel_silently_ends_marker(
    client, auth_headers, test_user_id, monkeypatch,
):
    from app.db import async_session_maker

    sent: list = []
    _patch_apns(monkeypatch, sent)
    token = "e5" * 32
    await _register_and_get_device(client, auth_headers, test_user_id, token)
    mission = "reminder:marker0006"
    await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission, "token": token},
        headers=auth_headers,
    )

    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        test_user_id, event_kind="mission_completed", title="Reminder canceled",
        data_json={"mission_id": mission, "mission_title": "⏰ M",
                   "silent": True, "dismiss_after_s": 0},
    ))
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().one()
        assert la.status == LA_ENDED  # cancel consumed the marker


@pytest.mark.asyncio
async def test_alarm_owned_report_without_token_never_inserts(
    client, auth_headers, test_user_id,
):
    from app.db import async_session_maker

    mission = "reminder:marker0007"
    resp = await client.post(
        "/api/devices/live-activity/alarm-owned",
        json={"mission_id": mission},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["updated"] == 0
    assert "marker" not in resp.json() or resp.json().get("marker") is None
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == mission)
        )).scalars().all()
        assert rows == []  # external-channel wake vector preserved


@pytest.mark.asyncio
async def test_reminder_card_deeplink_carries_seed_params(monkeypatch):
    """INSTANT-OPEN contract: reminder cards' tap URL carries the
    reminder text (rtext, clipped+encoded) and the fire instant (rat)
    so the app renders the message before any network round-trip.
    Chat-turn cards carry neither."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    end_ms = int((datetime.utcnow().timestamp() + 240) * 1000)
    # Production shape: countdown arm rows carry the reminder text as
    # row.body, NOT data.subtitle (routines._reminder_countdown_notify)
    # — the body fallback is what makes the countdown card seed.
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started", title="⏰ Stretch",
        body="Time to stretch & breathe", priority="default",
        data_json={"mission_id": "reminder:seed0001", "silent": True,
                   "kind": "reminder", "route": "chat",
                   "timer_end_ms": end_ms},
    ))
    url = sent[0]["payload"]["aps"]["attributes"]["deepLinkUrl"]
    assert url.startswith("toup://chat?mission=reminder:seed0001&rtext=")
    assert "Time%20to%20stretch%20%26%20breathe" in url
    assert f"&rat={end_ms}" in url

    # Chat turns: no seed params. Fresh user — the live countdown above
    # would otherwise make this yield (REMINDER WINS) and send nothing.
    user2 = await _mk_user()
    await _mk_la_device(user2)
    sent.clear()
    await _claim_and_dispatch(await _enqueue(
        user2, event_kind="mission_started", title="Working",
        data_json={"mission_id": "chatturn:seedless01", "kind": "chat_turn",
                   "route": "chat", "mission_title": "Working"},
    ))
    url2 = sent[0]["payload"]["aps"]["attributes"]["deepLinkUrl"]
    assert "rtext" not in url2 and "rat" not in url2


@pytest.mark.asyncio
async def test_announcement_start_carries_no_progress_surface(monkeypatch):
    """An operator's message must reach the Dynamic Island with NO
    percentage and NO bar.

    Both client surfaces bind a PRESENT optional
    (`else if let progress = contentState.progress`), so a substituted 0.0
    does not read as "no progress" — it reads as zero percent, and renders
    `Text("0%")` (LiveActivityWidget.swift:407) beside an empty
    `ProgressView` (:430, LiveActivityView.swift:253). Founder report
    2026-08-13: an operator's message under a progress bar for a job that
    does not exist. `nil` renders neither, so the assertion is on the KEY
    being absent, not on its value.
    """
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="announcement", source="platform",
        title="A word from the team", body="We shipped something.",
        # Exactly what admin_dispatch_worker._ensure_notification writes:
        # no `progress` key anywhere in it.
        data_json={"mission_id": "admin:d-1", "dispatch_id": "d-1",
                   "mode": "once", "kind": "announcement",
                   "deep_link": "toup://chat?mission=admin:d-1"},
    ))
    cs = sent[0]["payload"]["aps"]["content-state"]
    assert "progress" not in cs, (
        "an announcement narrates no work — a 0.0 here is rendered as '0%' "
        f"and an empty bar, not as absent. Got: {cs}"
    )
    assert "timerEndDateInMilliseconds" not in cs
    # The operator's words ARE the card: the subtitle override must survive.
    assert cs["subtitle"] == "We shipped something."


@pytest.mark.asyncio
async def test_non_announcement_start_still_carries_zero_progress(monkeypatch):
    """The companion, and the reason this pair is worth two tests: the fix
    must be SCOPED.

    A mission genuinely starting at zero percent is a true statement and its
    bar is wanted — so `progress: 0.0` has to survive here. Without this
    test, deleting the substitution outright would leave the announcement
    test green while silently removing every start-of-mission bar.
    """
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started", title="Working",
        data_json={"mission_id": "m-zero", "mission_title": "Working"},
    ))
    cs = sent[0]["payload"]["aps"]["content-state"]
    assert cs.get("progress") == 0.0


# ── voice call cards: adoption + the disconnect ender ─────────────────
# The 2026-08-16 force-quit repro: the app dies with the island claiming
# "Listening…", no app code left to end it. The platform must (a) HOLD a
# routable token for the locally-started voice card — adoption — and
# (b) end it from ws_realtime's disconnect path — end_voice_activities.


@pytest.mark.asyncio
async def test_activity_token_adopts_voice_call(client, auth_headers, test_user_id):
    await client.post(
        "/api/devices/live-activity",
        json={"token": "c7" * 32, "environment": "production",
              "install_id": "install-voice-1"},
        headers=auth_headers,
    )
    resp = await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "voice:adopt1234", "activity_push_token": "fa" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("adopted") is True

    from app.db import async_session_maker
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "voice:adopt1234")
        )).scalars().one()
        assert la.status == LA_STARTED
        assert la.activity_push_token == "fa" * 32


@pytest.mark.asyncio
async def test_end_voice_activities_pushes_end_and_ends_the_row(
    monkeypatch, client, auth_headers, test_user_id,
):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    await client.post(
        "/api/devices/live-activity",
        json={"token": "c8" * 32, "environment": "production",
              "install_id": "install-voice-2"},
        headers=auth_headers,
    )
    await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "voice:endme5678", "activity_push_token": "fb" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )

    # The disconnect path names the call it is ending — a session that
    # cannot name its card ends nothing (a web session's close must not
    # kill the phone's live call card).
    assert await las.end_voice_activities(test_user_id, None) == 0
    ended = await las.end_voice_activities(test_user_id, "voice:endme5678")
    assert ended == 1
    assert len(sent) == 1
    assert sent[0]["token"] == "fb" * 32
    assert sent[0]["payload"]["aps"]["event"] == "end"
    # Immediate dismissal: the card leaves the Lock Screen now, not at
    # the system's leisurely default.
    assert sent[0]["payload"]["aps"]["dismissal-date"] <= int(datetime.utcnow().timestamp())

    from app.db import async_session_maker
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "voice:endme5678")
        )).scalars().one()
        assert la.status == LA_ENDED

    # Idempotent: a second end finds nothing.
    assert await las.end_voice_activities(test_user_id, "voice:endme5678") == 0


@pytest.mark.asyncio
async def test_voice_sweep_never_touches_chat_turn_cards(
    monkeypatch, client, auth_headers, test_user_id,
):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    await client.post(
        "/api/devices/live-activity",
        json={"token": "c9" * 32, "environment": "production",
              "install_id": "install-voice-3"},
        headers=auth_headers,
    )
    await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "chatturn:bystander1", "activity_push_token": "fc" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )

    assert await las.end_voice_activities(test_user_id, "chatturn:bystander1") == 0
    assert sent == []

    from app.db import async_session_maker
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "chatturn:bystander1")
        )).scalars().one()
        assert la.status == LA_STARTED


@pytest.mark.asyncio
async def test_voice_adoption_leaves_other_cards_running(
    client, auth_headers, test_user_id,
):
    """A voice call COEXISTS: adopting its card must not force-end a live
    reminder countdown's row (the chat-turn one-per-device semantics do not
    apply to a call)."""
    await client.post(
        "/api/devices/live-activity",
        json={"token": "ca" * 32, "environment": "production",
              "install_id": "install-voice-4"},
        headers=auth_headers,
    )
    # A countdown card is live on the device (adopted via the chat-turn path
    # would end others, so plant it directly).
    await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "chatturn:preexisting", "activity_push_token": "fd" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )
    await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "voice:coexist1", "activity_push_token": "fe" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )
    from app.db import async_session_maker
    async with async_session_maker() as db:
        chat = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "chatturn:preexisting")
        )).scalars().one()
        voice = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "voice:coexist1")
        )).scalars().one()
        assert chat.status == LA_STARTED   # the call did not evict the turn card
        assert voice.status == LA_STARTED


@pytest.mark.asyncio
async def test_preempt_never_ends_a_voice_card(monkeypatch, client, auth_headers, test_user_id):
    """A mission/reminder start on the device must not end the island
    presence of a call in progress."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    await client.post(
        "/api/devices/live-activity",
        json={"token": "cb" * 32, "environment": "production",
              "install_id": "install-voice-5"},
        headers=auth_headers,
    )
    await client.post(
        "/api/devices/live-activity/activity-token",
        json={"mission_id": "voice:precious1", "activity_push_token": "ff" * 32,
              "source": "local_start"},
        headers=auth_headers,
    )
    from app.db import async_session_maker
    async with async_session_maker() as db:
        device = (await db.execute(
            select(LiveActivityDevice).where(
                LiveActivityDevice.install_id == "install-voice-5")
        )).scalars().one()
        preempted = await las._preempt_device(
            db, device, "mission-new-start", datetime.utcnow())
        await db.commit()
        assert preempted == 0
        voice = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "voice:precious1")
        )).scalars().one()
        assert voice.status == LA_STARTED
    assert sent == []   # and no end push went anywhere near it
