"""Feature 3 — reminder countdown Live Activity.

Lifecycle under test:
  creation (near-term one-shot) → silent mission_started with
  timer_end_ms (on-device countdown, digital island timer) →
  fire → mission_completed with the SAME mission_id flips the card into
  the urgent "now" alert (restart-if-missing guarantees the banner even
  when the countdown was preempted / never armed) →
  cancel / disable / reschedule / delete → silent end + immediate
  dismissal.

APNs is monkeypatched at live_activity_service.apns_push — no network.
notify() is monkeypatched at app.services.agent_notify_client — the
producers late-import it, so the module attr is the seam.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone as dt_timezone
from types import SimpleNamespace

import pytest

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED, LiveActivity, LiveActivityDevice,
    NotificationQueue, User, NQ_QUEUED,
)
from app.services import live_activity_service as las
from app.services import notification_dispatcher as nd


# ── Fixtures / harness (mirrors test_live_activity) ───────────────


@pytest.fixture
def notify_calls(monkeypatch):
    calls = []

    async def fake_notify(**kwargs):
        calls.append(kwargs)
        return "nid"

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", fake_notify)
    return calls


async def _mk_user(tz: str = "UTC") -> str:
    from app.db import async_session_maker
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"rc-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="RC Test",
            timezone=tz,
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


async def _mk_la_device(user_id: str) -> str:
    from app.db import async_session_maker

    device_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(LiveActivityDevice(
            id=device_id, user_id=user_id,
            push_to_start_token=uuid.uuid4().hex + uuid.uuid4().hex,
            apns_environment="development",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return device_id


async def _enqueue(user_id: str, **overrides) -> str:
    from app.db import async_session_maker

    row_id = str(uuid.uuid4())
    fields = dict(
        id=row_id, user_id=user_id, source="agent",
        event_kind="mission_completed", title="⏰ Stretch — now",
        body="Time to stretch", priority="high",
        idempotency_key=f"idem-{row_id}",
        status=NQ_QUEUED, created_at=datetime.utcnow(),
        data_json={"mission_id": "reminder:r-1"},
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


# ── LA lane: silent countdown start ───────────────────────────────


@pytest.mark.asyncio
async def test_silent_countdown_start_no_banner_timer_and_digital_island(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    timer_ms = int((datetime.utcnow().timestamp() + 1800) * 1000)

    row_id = await _enqueue(
        user_id, event_kind="mission_started", title="⏰ Stretch",
        body="Time to stretch", priority="default",
        data_json={"mission_id": "reminder:r-1", "mission_title": "⏰ Stretch",
                   "kind": "reminder", "route": "chat",
                   "timer_end_ms": timer_ms, "timer_type": "digital",
                   "silent": True, "urgent": True, "progress": 0},
    )
    assert await _claim_and_dispatch(row_id) == "sent"

    aps = sent[0]["payload"]["aps"]
    assert aps["event"] == "start"
    # iOS 26 requires an alert config on every start: the arm carries a
    # synthesized, soundless banner (which doubles as set-confirmation).
    assert "sound" not in aps["alert"]
    assert aps["content-state"]["timerEndDateInMilliseconds"] == timer_ms
    assert aps["content-state"]["subtitle"] == "Time to stretch"
    assert aps["attributes"]["timerType"] == "digital"
    # The mission id rides the tap URL so the app can ack the tap.
    assert aps["attributes"]["deepLinkUrl"] == "toup://chat?mission=reminder:r-1"
    # Zombie backstop: the card self-marks stale 2 min after fire time.
    assert aps["stale-date"] == int(timer_ms / 1000) + 120


# ── LA lane: fire flips the same-mission card ─────────────────────


@pytest.mark.asyncio
async def test_fire_flips_countdown_card_with_subtitle_override(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)
    timer_ms = int((datetime.utcnow().timestamp() + 600) * 1000)

    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started", title="⏰ Stretch",
        body="Time to stretch", priority="default",
        data_json={"mission_id": "reminder:r-1", "silent": True,
                   "timer_end_ms": timer_ms, "kind": "reminder",
                   "route": "chat", "urgent": True},
    ))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id,
        data_json={"mission_id": "reminder:r-1", "mission_title": "⏰ Stretch",
                   "kind": "reminder", "route": "chat",
                   "subtitle": "Time to stretch", "urgent": True,
                   "cap_exempt": True, "dismiss_after_s": 900,
                   "no_agent_fallback": True},
    ))
    assert result == "sent"
    # Audible fire = alerting UPDATE (documented alert surface, with
    # sound) followed by a bannerless end that closes the card.
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update", "end"]
    upd = sent[0]["payload"]["aps"]
    assert upd["alert"]["title"] == "⏰ Stretch — now"
    assert upd["alert"]["sound"] == "default"
    # Producer subtitle replaces the hard-coded 'Completed ✓' on BOTH
    # pushes — the update already shows the fired state, so a lost end
    # can no longer leave a live-looking card.
    assert upd["content-state"]["subtitle"] == "Time to stretch"
    aps = sent[1]["payload"]["aps"]
    assert aps["event"] == "end"
    assert "alert" not in aps
    assert aps["content-state"]["subtitle"] == "Time to stretch"
    now_ts = int(datetime.utcnow().timestamp())
    assert now_ts + 800 <= aps["dismissal-date"] <= now_ts + 1000

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "reminder:r-1")
        )).scalars().one()
        assert la.status == LA_ENDED


@pytest.mark.asyncio
async def test_fire_with_no_card_restarts_then_alerts(monkeypatch):
    """The preempted/never-armed path: a terminal alert row with zero
    started activities restarts the card LOUD — the start alert (with
    sound) is the surface iOS 26 provably renders — then the end goes
    bannerless (this is what makes the unconditional ws_chat 'Answer
    ready' audible too)."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    result = await _claim_and_dispatch(await _enqueue(
        user_id,
        data_json={"mission_id": "reminder:r-9", "mission_title": "⏰ Pills",
                   "kind": "reminder", "route": "chat", "subtitle": "Take pills",
                   "urgent": True, "dismiss_after_s": 900,
                   "no_agent_fallback": True},
    ))
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["start", "end"]
    start_alert = sent[0]["payload"]["aps"]["alert"]
    assert start_alert["title"] == "⏰ Stretch — now"  # loud restart
    assert start_alert["sound"] == "default"
    assert "alert" not in sent[1]["payload"]["aps"]  # bannerless close

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        row_id = (await db.execute(
            select(NotificationQueue.id).order_by(
                NotificationQueue.created_at.desc()).limit(1)
        )).scalar_one()
        row = await db.get(NotificationQueue, row_id)
        assert row.channels_json["live_activity"]["restarted"] is True


@pytest.mark.asyncio
async def test_needs_input_with_no_card_restarts_then_alerts(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    async def fake_fallback(db, row):
        return {"status": "skipped", "reason": "no_active_agent"}

    monkeypatch.setattr(nd, "_request_agent_channel_delivery", fake_fallback)

    result = await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="needs_input", title="“T” needs you",
        data_json={"mission_id": "m-77", "mission_title": "T",
                   "urgent": True, "progress": 50},
    ))
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["start", "update"]
    # Loud restart: the start carries the banner+sound; the follow-up
    # update goes bannerless (no double-bang).
    assert sent[0]["payload"]["aps"]["alert"]["title"] == "“T” needs you"
    assert sent[0]["payload"]["aps"]["alert"]["sound"] == "default"
    assert "alert" not in sent[1]["payload"]["aps"]

    # The card stays alive — the task is waiting for the user.
    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        la = (await db.execute(
            select(LiveActivity).where(LiveActivity.mission_id == "m-77")
        )).scalars().one()
        assert la.status == LA_STARTED


# ── LA lane: silent cancel end ────────────────────────────────────


@pytest.mark.asyncio
async def test_fire_counts_delivered_when_banner_lands_but_close_fails(monkeypatch):
    """The alert is the job; the card close is hygiene. If the alerting
    update 200s but the follow-up end fails, the row must land sent —
    NOT retry (a re-alert would double-bang the user)."""
    sent: list = []

    async def fake_send(token, payload, *, environment="development", priority=10):
        sent.append({"token": token, "payload": payload,
                     "environment": environment, "priority": priority})
        if payload["aps"]["event"] == "end":
            return 500, "InternalServerError"
        return 200, ""

    monkeypatch.setattr(las.apns_push, "send_live_activity", fake_send)
    monkeypatch.setattr(settings, "apns_key_b64", "eA==")
    monkeypatch.setattr(settings, "apns_key_id", "KEY123")
    monkeypatch.setattr(settings, "apns_team_id", "TEAM123")

    user_id = await _mk_user()
    await _mk_la_device(user_id)
    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started", title="⏰ Stretch",
        body="Time to stretch", priority="default",
        data_json={"mission_id": "reminder:r-5", "silent": True,
                   "kind": "reminder", "urgent": True,
                   "timer_end_ms": int((datetime.utcnow().timestamp() + 120) * 1000)},
    ))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id,
        data_json={"mission_id": "reminder:r-5", "mission_title": "⏰ Stretch",
                   "kind": "reminder", "route": "chat",
                   "subtitle": "Time to stretch", "urgent": True,
                   "dismiss_after_s": 900, "no_agent_fallback": True},
    ))
    assert result == "sent"
    events = [s["payload"]["aps"]["event"] for s in sent]
    assert events == ["update", "end"]
    assert sent[0]["payload"]["aps"]["alert"]["sound"] == "default"

    from app.db import async_session_maker
    from sqlalchemy import select
    async with async_session_maker() as db:
        row_id = (await db.execute(
            select(NotificationQueue.id).order_by(
                NotificationQueue.created_at.desc()).limit(1)
        )).scalar_one()
        row = await db.get(NotificationQueue, row_id)
        frag = row.channels_json["live_activity"]
        assert frag["delivered"] is True
        device_frag = list(frag["devices"].values())[0]
        assert device_frag["alerted"] is True


@pytest.mark.asyncio
async def test_silent_cancel_ends_card_without_banner(monkeypatch):
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    await _claim_and_dispatch(await _enqueue(
        user_id, event_kind="mission_started", title="⏰ Stretch",
        body="Time to stretch", priority="default",
        data_json={"mission_id": "reminder:r-2", "silent": True,
                   "kind": "reminder", "urgent": True,
                   "timer_end_ms": int((datetime.utcnow().timestamp() + 900) * 1000)},
    ))
    sent.clear()

    result = await _claim_and_dispatch(await _enqueue(
        user_id, title="Reminder canceled", body=None, priority="low",
        data_json={"mission_id": "reminder:r-2", "kind": "reminder",
                   "silent": True, "urgent": True, "cap_exempt": True,
                   "dismiss_after_s": 0, "no_agent_fallback": True},
    ))
    assert result == "sent"
    aps = sent[0]["payload"]["aps"]
    assert aps["event"] == "end"
    assert "alert" not in aps  # silent end — the card just vanishes
    assert aps["dismissal-date"] <= int(datetime.utcnow().timestamp()) + 1


@pytest.mark.asyncio
async def test_silent_cancel_with_no_card_does_not_restart(monkeypatch):
    """Cancel for a card that isn't live must NOT flash a start+end —
    silent rows skip the restart-if-missing path entirely."""
    sent: list = []
    _patch_apns(monkeypatch, sent)
    user_id = await _mk_user()
    await _mk_la_device(user_id)

    row_id = await _enqueue(
        user_id, title="Reminder canceled", body=None, priority="low",
        data_json={"mission_id": "reminder:r-3", "kind": "reminder",
                   "silent": True, "urgent": True, "cap_exempt": True,
                   "dismiss_after_s": 0, "no_agent_fallback": True},
    )
    await _claim_and_dispatch(row_id)
    assert sent == []

    from app.db import async_session_maker
    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        assert row.channels_json["live_activity"] == {
            "status": "skipped", "reason": "no_active_activity",
        }


# ── Producer: create/update/delete hooks in app.api.routines ──────


async def _create_reminder(minutes_out: int = 30, **overrides):
    from app.api.routines import RoutineCreate, create_routine

    payload = dict(
        kind="reminder", schedule_kind="at",
        schedule_at=datetime.utcnow() + timedelta(minutes=minutes_out),
        reminder_text="Time to stretch", name="Stretch",
    )
    payload.update(overrides)
    return await create_routine(RoutineCreate(**payload))


@pytest.fixture
def _as_user(monkeypatch):
    async def seed():
        user_id = await _mk_user(tz="UTC")
        monkeypatch.setattr(settings, "user_id", user_id)
        return user_id
    return seed


@pytest.mark.asyncio
async def test_create_near_term_reminder_arms_countdown(notify_calls, _as_user):
    await _as_user()
    resp = await _create_reminder(minutes_out=30)

    assert len(notify_calls) == 1
    call = notify_calls[0]
    assert call["event_kind"] == "mission_started"
    assert call["title"] == "⏰ Stretch"
    data = call["data"]
    assert data["mission_id"] == f"reminder:{resp.id}"
    assert data["silent"] is True and data["urgent"] is True
    assert data["kind"] == "reminder" and data["route"] == "chat"
    assert data["timer_type"] == "digital"
    # schedule_at is UTC-naive — the epoch must be computed via utc, not
    # the container's local offset.
    expected_ms = int(
        resp.schedule_at.replace(tzinfo=dt_timezone.utc).timestamp() * 1000
    )
    assert data["timer_end_ms"] == expected_ms


@pytest.mark.asyncio
async def test_create_beyond_window_or_recurring_arms_nothing(notify_calls, _as_user):
    await _as_user()
    # 9h out — beyond the 8h ActivityKit window.
    await _create_reminder(minutes_out=9 * 60)
    # Recurring daily reminder — a perpetual card would hog the
    # one-activity-per-device slot.
    from app.api.routines import RoutineCreate, create_routine
    await create_routine(RoutineCreate(
        kind="reminder", schedule_kind="cron", schedule_cron_local="30 7 * * *",
        reminder_text="Drink water", name="Water",
    ))
    assert notify_calls == []


@pytest.mark.asyncio
async def test_create_flag_off_arms_nothing(notify_calls, _as_user, monkeypatch):
    await _as_user()
    monkeypatch.setattr(settings, "reminder_countdown_live_activity_enabled", False)
    await _create_reminder(minutes_out=30)
    assert notify_calls == []


@pytest.mark.asyncio
async def test_disable_cancels_countdown(notify_calls, _as_user):
    from app.api.routines import RoutineUpdate, update_routine

    await _as_user()
    resp = await _create_reminder(minutes_out=30)
    notify_calls.clear()

    await update_routine(resp.id, RoutineUpdate(enabled=False))

    assert [c["event_kind"] for c in notify_calls] == ["mission_completed"]
    data = notify_calls[0]["data"]
    assert data["mission_id"] == f"reminder:{resp.id}"
    assert data["silent"] is True
    assert data["dismiss_after_s"] == 0
    assert data["no_agent_fallback"] is True


@pytest.mark.asyncio
async def test_reschedule_cancels_then_rearms(notify_calls, _as_user):
    from app.api.routines import RoutineUpdate, update_routine

    await _as_user()
    resp = await _create_reminder(minutes_out=30)
    notify_calls.clear()

    new_at = datetime.utcnow() + timedelta(hours=1)
    await update_routine(resp.id, RoutineUpdate(schedule_at=new_at))

    # Cancel-end first, fresh countdown second (dispatcher claims rows
    # in created_at order, so the ordering here is load-bearing).
    assert [c["event_kind"] for c in notify_calls] == [
        "mission_completed", "mission_started",
    ]
    rearm = notify_calls[1]["data"]
    assert rearm["mission_id"] == f"reminder:{resp.id}"
    assert rearm["timer_end_ms"] == int(
        new_at.replace(tzinfo=dt_timezone.utc).timestamp() * 1000
    )


@pytest.mark.asyncio
async def test_reschedule_beyond_window_only_cancels(notify_calls, _as_user):
    from app.api.routines import RoutineUpdate, update_routine

    await _as_user()
    resp = await _create_reminder(minutes_out=30)
    notify_calls.clear()

    await update_routine(
        resp.id, RoutineUpdate(schedule_at=datetime.utcnow() + timedelta(hours=12)),
    )
    assert [c["event_kind"] for c in notify_calls] == ["mission_completed"]


@pytest.mark.asyncio
async def test_delete_cancels_countdown(notify_calls, _as_user):
    from app.api.routines import delete_routine

    await _as_user()
    resp = await _create_reminder(minutes_out=30)
    notify_calls.clear()

    await delete_routine(resp.id)

    assert [c["event_kind"] for c in notify_calls] == ["mission_completed"]
    assert notify_calls[0]["data"]["mission_id"] == f"reminder:{resp.id}"


# ── Producer: fire-time notify in ReminderHandler ─────────────────


def _routine_ns(**kw):
    return SimpleNamespace(
        id="r-77", user_id="u-77", name=kw.pop("name", "Stretch"),
        reminder_text=kw.pop("reminder_text", "Time to stretch"),
        config_json=None, **kw,
    )


async def _stub_writer(db, **kw):
    return "m-1", "d-1"


@pytest.mark.asyncio
async def test_reminder_fire_notifies_now_alert(notify_calls):
    from app.agent.routines.reminder_handler import ReminderHandler

    handler = ReminderHandler(writer=_stub_writer)
    result = await handler.execute(
        _routine_ns(), SimpleNamespace(id="run-1"), None,
    )

    assert result.status == "success"
    assert len(notify_calls) == 1
    call = notify_calls[0]
    assert call["event_kind"] == "mission_completed"
    assert call["title"] == "⏰ Stretch — now"
    assert call["priority"] == "high"
    assert call["dedup_key"] == "reminder:r-77:fire:run-1"
    data = call["data"]
    assert data["mission_id"] == "reminder:r-77"
    assert data["subtitle"] == "Time to stretch"
    assert data["urgent"] is True
    assert data["cap_exempt"] is True
    assert data["no_agent_fallback"] is True
    assert data["dismiss_after_s"] == 900


@pytest.mark.asyncio
async def test_reminder_fire_notify_failure_never_fails_the_run(monkeypatch):
    from app.agent.routines.reminder_handler import ReminderHandler

    async def broken_notify(**kwargs):
        raise RuntimeError("outbox down")

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", broken_notify)

    handler = ReminderHandler(writer=_stub_writer)
    result = await handler.execute(
        _routine_ns(), SimpleNamespace(id="run-2"), None,
    )
    assert result.status == "success"


# ── Ingest contract pin ───────────────────────────────────────────


def test_countdown_payloads_pass_ingest_validation():
    from app.api.agent_notify import AgentNotifyRequest

    rid = str(uuid.uuid4())
    armed = AgentNotifyRequest(
        user_id="u" * 36, idempotency_key="row-" + uuid.uuid4().hex[:12],
        event_kind="mission_started", title="⏰ Stretch", body="Time to stretch",
        data={"mission_id": f"reminder:{rid}", "mission_title": "⏰ Stretch",
              "kind": "reminder", "route": "chat",
              "timer_end_ms": 1_784_140_000_000, "timer_type": "digital",
              "silent": True, "urgent": True, "progress": 0},
        priority="default",
    )
    # mission_id stays under the LA lane's 64-char cap.
    assert len(armed.data["mission_id"]) <= 64

    fired = AgentNotifyRequest(
        user_id="u" * 36, idempotency_key="row-" + uuid.uuid4().hex[:12],
        event_kind="mission_completed", title="⏰ Stretch — now",
        body="Time to stretch",
        data={"mission_id": f"reminder:{rid}", "mission_title": "⏰ Stretch",
              "kind": "reminder", "route": "chat", "subtitle": "Time to stretch",
              "urgent": True, "cap_exempt": True, "dismiss_after_s": 900,
              "no_agent_fallback": True},
        priority="high", dedup_key=f"reminder:{rid}:fire:run-1",
    )
    assert fired.data["subtitle"] == "Time to stretch"
