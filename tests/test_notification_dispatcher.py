"""Notification dispatcher (Autopilot PR3) — claim CAS, policy, send,
receipts.

The replica-safety contract is the headline test: two concurrent
claimers over the same queued rows must partition them (no row claimed
twice) because platform-api runs the scheduler on 2 replicas.

Expo HTTP is monkeypatched at the module seam
(notification_dispatcher.expo_push.*) — no network in tests.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest

from app.db.models import (
    NotificationQueue, PushDevice, User,
    NQ_QUEUED, NQ_SENT, NQ_SUPPRESSED, NQ_FAILED,
)
from app.services import notification_dispatcher as nd


# ── Helpers ───────────────────────────────────────────────────────


async def _mk_user(tz: str | None = None, prefs: dict | None = None) -> str:
    from app.db import async_session_maker
    from app.services.auth_service import get_password_hash

    if prefs is None:
        # Deterministic dispatch tests: the DEFAULT prefs enable quiet
        # hours 22:00-08:00, which made every send-path test fail when
        # the suite ran at night UTC (found by the 2026-07-10 health
        # sweep). Tests that exercise quiet hours pass explicit prefs
        # or drive the pure policy function with an injected now.
        prefs = {"quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"}}
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"nd-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12),
            name="ND Test",
            timezone=tz,
            notification_preferences=prefs,
        ))
        await db.commit()
    return user_id


async def _enqueue(user_id: str, **overrides) -> str:
    from app.db import async_session_maker

    row_id = str(uuid.uuid4())
    fields = dict(
        id=row_id,
        user_id=user_id,
        source="agent",
        event_kind="mission_completed",
        title="Done",
        body="body",
        priority="default",
        idempotency_key=f"idem-{row_id}",
        status=NQ_QUEUED,
        created_at=datetime.utcnow(),
    )
    fields.update(overrides)
    async with async_session_maker() as db:
        db.add(NotificationQueue(**fields))
        await db.commit()
    return row_id


async def _add_device(user_id: str) -> str:
    from app.db import async_session_maker

    device_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(PushDevice(
            id=device_id,
            user_id=user_id,
            expo_push_token=f"ExponentPushToken[{uuid.uuid4().hex[:20]}]",
            platform="ios",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return device_id


async def _row(row_id: str) -> NotificationQueue:
    from app.db import async_session_maker

    async with async_session_maker() as db:
        row = await db.get(NotificationQueue, row_id)
        db.expunge(row)
        return row


def _ok_tickets(messages):
    return [{"status": "ok", "id": f"t-{i}"} for i, _ in enumerate(messages)]


# ── Claim CAS: the replica-safety contract ────────────────────────


@pytest.mark.asyncio
async def test_concurrent_claimers_partition_rows():
    from app.db import async_session_maker

    user_id = await _mk_user()
    ids = [await _enqueue(user_id) for _ in range(8)]
    now = datetime.utcnow()

    async def claim():
        async with async_session_maker() as db:
            return await nd._claim_batch(db, now, limit=20)

    a, b = await asyncio.gather(claim(), claim())
    # Every row claimed exactly once across the two "replicas".
    assert set(a).isdisjoint(set(b))
    assert set(a) | set(b) == set(ids)


@pytest.mark.asyncio
async def test_stuck_sending_rows_requeue():
    from app.db import async_session_maker

    user_id = await _mk_user()
    row_id = await _enqueue(
        user_id,
        status="sending",
        claimed_at=datetime.utcnow() - timedelta(minutes=30),
    )
    async with async_session_maker() as db:
        requeued = await nd._requeue_stuck(db, datetime.utcnow())
    assert requeued == 1
    assert (await _row(row_id)).status == NQ_QUEUED


# ── Policy ────────────────────────────────────────────────────────


def _prefs(**overrides):
    from app.api.account import _merged_prefs
    base = _merged_prefs(None)
    base.update(overrides)
    return base


def test_policy_progress_never_pushes():
    row = NotificationQueue(event_kind="progress", priority="default")
    decision, reason = nd.evaluate_policy(
        row, _prefs(), "UTC", datetime.utcnow(), 0, False,
    )
    assert (decision, reason) == ("suppress", "progress_in_app_only")


def test_policy_respects_autopilot_push_toggle():
    row = NotificationQueue(event_kind="mission_completed", priority="default")
    decision, reason = nd.evaluate_policy(
        row, _prefs(autopilot_push=False), "UTC", datetime.utcnow(), 0, False,
    )
    assert (decision, reason) == ("suppress", "autopilot_push_disabled")


def test_policy_daily_cap_with_needs_input_bypass():
    # Fixed daytime instant — utcnow() at night trips the default
    # quiet-hours window and turns 'send' into 'defer'.
    now = datetime(2026, 7, 8, 15, 0)
    capped = NotificationQueue(event_kind="mission_completed", priority="default")
    decision, reason = nd.evaluate_policy(capped, _prefs(daily_push_cap=3), "UTC", now, 3, False)
    assert (decision, reason) == ("suppress", "daily_cap")

    urgent = NotificationQueue(event_kind="needs_input", priority="high")
    decision, _ = nd.evaluate_policy(urgent, _prefs(daily_push_cap=3), "UTC", now, 99, False)
    assert decision == "send"


def test_policy_dedup_window():
    row = NotificationQueue(
        event_kind="mission_completed", priority="default", dedup_key="m1:done",
    )
    decision, reason = nd.evaluate_policy(
        row, _prefs(), "UTC", datetime.utcnow(), 0, True,
    )
    assert (decision, reason) == ("suppress", "dedup_window")


def test_policy_quiet_hours_defer_and_urgent_bypass():
    # 23:00 Toronto = inside a 22:00-08:00 window.
    now_utc = datetime(2026, 7, 8, 3, 0)  # 23:00 EDT previous day
    prefs = _prefs(quiet_hours={"enabled": True, "start": "22:00", "end": "08:00"})

    row = NotificationQueue(event_kind="mission_completed", priority="default")
    decision, until = nd.evaluate_policy(row, prefs, "America/Toronto", now_utc, 0, False)
    assert decision == "defer"
    # 08:00 EDT = 12:00 UTC same day.
    assert until == datetime(2026, 7, 8, 12, 0)

    urgent = NotificationQueue(
        event_kind="needs_approval", priority="high", data_json={"urgent": True},
    )
    decision, _ = nd.evaluate_policy(urgent, prefs, "America/Toronto", now_utc, 0, False)
    assert decision == "send"


def test_policy_quiet_hours_disabled_without_timezone():
    now_utc = datetime(2026, 7, 8, 3, 0)
    prefs = _prefs(quiet_hours={"enabled": True, "start": "22:00", "end": "08:00"})
    row = NotificationQueue(event_kind="mission_completed", priority="default")
    decision, _ = nd.evaluate_policy(row, prefs, None, now_utc, 0, False)
    assert decision == "send"


# ── End-to-end dispatch over the DB (Expo mocked) ─────────────────


@pytest.mark.asyncio
async def test_dispatch_sends_and_records_tickets(monkeypatch):
    sent_batches = []

    async def fake_send(messages):
        sent_batches.append(messages)
        return _ok_tickets(messages)

    monkeypatch.setattr(nd.expo_push, "send_push_messages", fake_send)

    user_id = await _mk_user(tz="UTC")
    await _add_device(user_id)
    row_id = await _enqueue(user_id)

    stats = await nd.run_notification_dispatch()
    assert stats["claimed"] == 1 and stats.get("sent") == 1

    row = await _row(row_id)
    assert row.status == nd.NQ_SENT_PENDING_RECEIPT
    assert row.sent_at is not None
    expo = row.channels_json["expo"]
    assert list(expo.values())[0]["ticket"] == "t-0"
    # Deep-link payload rides in data.
    assert sent_batches[0][0]["data"]["notification_id"] == row_id


@pytest.mark.asyncio
async def test_dispatch_device_not_registered_revokes_token(monkeypatch):
    async def fake_send(messages):
        return [{
            "status": "error", "message": "gone",
            "details": {"error": "DeviceNotRegistered"},
        }]

    monkeypatch.setattr(nd.expo_push, "send_push_messages", fake_send)

    user_id = await _mk_user(tz="UTC")
    device_id = await _add_device(user_id)
    row_id = await _enqueue(user_id, event_kind="digest", priority="low")

    await nd.run_notification_dispatch()

    from app.db import async_session_maker
    async with async_session_maker() as db:
        device = await db.get(PushDevice, device_id)
        assert device.revoked_at is not None

    # digest is not a fallback kind → nothing delivered → retry queued.
    row = await _row(row_id)
    assert row.status == NQ_QUEUED and row.scheduled_for is not None


@pytest.mark.asyncio
async def test_dispatch_no_devices_no_fallback_exhausts_to_failed(monkeypatch):
    user_id = await _mk_user(
        tz="UTC",
        prefs={
            "autopilot_channel_fallback": False,
            "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
        },
    )
    row_id = await _enqueue(user_id)

    from app.config import settings
    monkeypatch.setattr(settings, "notification_max_attempts", 1)

    await nd.run_notification_dispatch()
    row = await _row(row_id)
    assert row.status == NQ_FAILED
    assert row.channels_json["expo"]["reason"] == "no_active_devices"


@pytest.mark.asyncio
async def test_dispatch_suppressed_rows_are_terminal(monkeypatch):
    user_id = await _mk_user(tz="UTC", prefs={"autopilot_push": False})
    row_id = await _enqueue(user_id)

    await nd.run_notification_dispatch()
    row = await _row(row_id)
    assert row.status == NQ_SUPPRESSED
    assert row.channels_json["policy"]["suppressed"] == "autopilot_push_disabled"

    # Second tick must not touch it again.
    stats = await nd.run_notification_dispatch()
    assert stats["claimed"] == 0


@pytest.mark.asyncio
async def test_receipt_check_prunes_dead_tokens(monkeypatch):
    async def fake_send(messages):
        return _ok_tickets(messages)

    async def fake_receipts(ids):
        return {
            tid: {
                "status": "error", "message": "gone",
                "details": {"error": "DeviceNotRegistered"},
            }
            for tid in ids
        }

    monkeypatch.setattr(nd.expo_push, "send_push_messages", fake_send)
    monkeypatch.setattr(nd.expo_push, "get_push_receipts", fake_receipts)

    user_id = await _mk_user(tz="UTC")
    device_id = await _add_device(user_id)
    row_id = await _enqueue(user_id)
    await nd.run_notification_dispatch()

    # Age the send past the receipt delay.
    from app.db import async_session_maker
    from sqlalchemy import update as sa_update
    async with async_session_maker() as db:
        await db.execute(
            sa_update(NotificationQueue)
            .where(NotificationQueue.id == row_id)
            .values(sent_at=datetime.utcnow() - timedelta(minutes=20))
        )
        await db.commit()

    stats = await nd.run_notification_receipt_check()
    assert stats == {"checked": 1, "pruned": 1}

    row = await _row(row_id)
    assert row.status == NQ_SENT
    async with async_session_maker() as db:
        device = await db.get(PushDevice, device_id)
        assert device.revoked_at is not None


def test_dispatch_loop_wired_into_platform_lifespan():
    """Source-grep pin (test_api_no_html_invariant.py style). The
    dispatcher runs as a lifespan asyncio loop, NOT under the
    ENABLE_SCHEDULER-gated APScheduler — that scheduler never runs on
    the Railway deployment (2026-07-10 live find), which silently
    killed all notification dispatch."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pm = (root / "platform_main.py").read_text()
    assert "notification_dispatch_loop" in pm
    assert "settings.notification_dispatch_enabled" in pm
    nd = (root / "app" / "services" / "notification_dispatcher.py").read_text()
    assert "async def notification_dispatch_loop" in nd
    st = (root / "app" / "scripts" / "scheduled_tasks.py").read_text()
    assert 'id="notification_dispatch"' not in st, (
        "dispatcher must not ALSO be registered on the APScheduler — "
        "the lifespan loop supersedes it"
    )


@pytest.mark.asyncio
async def test_silent_row_with_only_skips_suppresses_instead_of_retrying():
    """A silent row (quiet card bookkeeping — e.g. the Answer-ready
    push of a reminder-creating turn while the user is in-app) whose
    every channel merely SKIPPED must suppress, not churn the retry
    backoff ladder: the next attempt sees the same world and silence
    IS the intended outcome (2026-07-23 probe: such a row sat at
    attempts=3, last_error=no_channel_delivered)."""
    from app.db import async_session_maker
    from sqlalchemy import select

    user_id = await _mk_user()  # no push devices, no LA devices
    row_id = await _enqueue(
        user_id,
        event_kind="mission_completed",
        title="Answer ready",
        data_json={"mission_id": "chatturn:beef00000001", "kind": "chat_turn",
                   "silent": True, "progress": 100},
    )
    now = datetime.utcnow()
    async with async_session_maker() as db:
        claimed = await nd._claim_batch(db, now)
        assert row_id in claimed
        result = await nd._dispatch_row(db, row_id, now)
    assert result == "suppressed:silent_undeliverable"

    async with async_session_maker() as db:
        row = (await db.execute(
            select(NotificationQueue).where(NotificationQueue.id == row_id)
        )).scalars().one()
        assert row.status == NQ_SUPPRESSED
        assert row.channels_json["policy"]["suppressed"] == "silent_undeliverable"
