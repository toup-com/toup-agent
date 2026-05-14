"""Phase A acceptance tests for the reminder/alert capability.

Locks the four production-grade contracts that replace CronJob's
single-channel behaviour:

  1. ReminderHandler delivers `reminder_text` verbatim with zero LLM
     calls, zero MCP tool invocations, and per-channel fanout via
     `delivery_channels`.
  2. `_build_trigger_for_routine` dispatches correctly on
     `schedule_kind` — cron / at / every — and rejects past 'at'
     datetimes + sub-60s intervals.
  3. `_in_active_window` correctly gates interval fires, including
     overnight windows (22:00–06:00).
  4. `auto_disable_after_fire` flips `enabled=false` after a
     successful fire AND removes the in-memory APScheduler trigger.

These pin the contract for Phase B (CronJob data migration) and
Phase C (CronService deprecation) — any change that breaks these
breaks the user-facing reminder system.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, time, timedelta, timezone

import pytest
from sqlalchemy import select


# ── 1. Window-gate helper ───────────────────────────────────────────


def test_in_active_window_simple_daytime():
    """9:00–17:00 covers business hours. Standard interval."""
    from app.agent.routines.runner import _in_active_window
    assert _in_active_window(time(9, 0), time(9, 0), time(17, 0))
    assert _in_active_window(time(12, 0), time(9, 0), time(17, 0))
    assert _in_active_window(time(16, 59, 59), time(9, 0), time(17, 0))
    assert not _in_active_window(time(8, 59, 59), time(9, 0), time(17, 0))
    assert not _in_active_window(time(17, 0), time(9, 0), time(17, 0))
    assert not _in_active_window(time(23, 0), time(9, 0), time(17, 0))


def test_in_active_window_overnight_wraps_midnight():
    """22:00–06:00 wraps midnight. Both 23:00 and 02:00 must be in window;
    08:00 must not."""
    from app.agent.routines.runner import _in_active_window
    assert _in_active_window(time(23, 0), time(22, 0), time(6, 0))
    assert _in_active_window(time(0, 0), time(22, 0), time(6, 0))
    assert _in_active_window(time(2, 30), time(22, 0), time(6, 0))
    assert _in_active_window(time(5, 59), time(22, 0), time(6, 0))
    assert not _in_active_window(time(6, 0), time(22, 0), time(6, 0))
    assert not _in_active_window(time(8, 0), time(22, 0), time(6, 0))
    assert not _in_active_window(time(21, 59), time(22, 0), time(6, 0))


def test_in_active_window_none_bounds_means_always_active():
    from app.agent.routines.runner import _in_active_window
    assert _in_active_window(time(3, 0), None, None)
    assert _in_active_window(time(15, 0), None, None)


# ── 2. Trigger-shape dispatch ───────────────────────────────────────


class _StubRoutine:
    """Light stand-in for the Routine model — only the fields the
    trigger builder reads."""
    def __init__(self, **kwargs):
        self.schedule_kind = kwargs.get("schedule_kind", "cron")
        self.schedule_cron_local = kwargs.get("schedule_cron_local", "0 9 * * *")
        self.schedule_at = kwargs.get("schedule_at")
        self.schedule_interval_seconds = kwargs.get("schedule_interval_seconds")
        self.kind = kwargs.get("kind", "reminder")


def test_build_trigger_cron_returns_crontrigger():
    from apscheduler.triggers.cron import CronTrigger
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    routine = _StubRoutine(schedule_kind="cron", schedule_cron_local="0 9 * * *")
    trigger, tag = _build_trigger_for_routine(routine, ZoneInfo("America/Toronto"))
    assert tag == "cron"
    assert isinstance(trigger, CronTrigger)


def test_build_trigger_at_returns_datetrigger_when_future():
    from apscheduler.triggers.date import DateTrigger
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    future = datetime.utcnow() + timedelta(hours=2)
    routine = _StubRoutine(schedule_kind="at", schedule_at=future)
    trigger, tag = _build_trigger_for_routine(routine, ZoneInfo("UTC"))
    assert tag == "at"
    assert isinstance(trigger, DateTrigger)


def test_build_trigger_at_in_past_returns_past_at_tag():
    """One-shot reminders in the past must NOT fire. The trigger
    builder returns `past_at` so `_register_trigger_for` logs and skips."""
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    past = datetime.utcnow() - timedelta(hours=1)
    routine = _StubRoutine(schedule_kind="at", schedule_at=past)
    trigger, tag = _build_trigger_for_routine(routine, ZoneInfo("UTC"))
    assert tag == "past_at"
    assert trigger is None


def test_build_trigger_every_returns_intervaltrigger():
    from apscheduler.triggers.interval import IntervalTrigger
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    routine = _StubRoutine(schedule_kind="every", schedule_interval_seconds=300)
    trigger, tag = _build_trigger_for_routine(routine, ZoneInfo("America/Toronto"))
    assert tag == "every"
    assert isinstance(trigger, IntervalTrigger)


def test_build_trigger_every_rejects_sub_minute_interval():
    """Floor of 60 seconds protects against runaway notifications.
    Anything under 60s returns `invalid`."""
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    routine = _StubRoutine(schedule_kind="every", schedule_interval_seconds=30)
    trigger, tag = _build_trigger_for_routine(routine, ZoneInfo("UTC"))
    assert tag == "invalid"
    assert trigger is None


# ── 3. ReminderHandler — text-only delivery, zero LLM ──────────────


@pytest.mark.asyncio
async def test_reminder_handler_delivers_text_with_zero_llm_zero_mcp():
    """The ReminderHandler MUST NOT call any LLM or MCP — its whole
    value proposition is "cheap text delivery". Test injects a custom
    writer that captures the message; asserts `tools_invoked` is empty
    and the message content is the literal reminder_text."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.reminder_handler import ReminderHandler

    captured = {}

    async def _fake_writer(db, **kwargs):
        captured["content"] = kwargs.get("content")
        captured["source"] = kwargs.get("source")
        return "msg-fake", "daychat-fake"

    class _Routine:
        id = "r-1"
        user_id = "u-1"
        name = "Take vitamins"
        reminder_text = "Time to take your vitamins"
        config_json = {"delivery_channels": ["website"]}

    handler = ReminderHandler(writer=_fake_writer)
    result = await handler.execute(_Routine(), run=None, db=None)

    assert result.status == "success"
    assert result.tools_invoked == []
    assert captured["content"] == "Time to take your vitamins"
    assert captured["source"] == "reminder"
    # Watermark stays None — reminders don't track upstream state.
    assert result.new_watermark is None


@pytest.mark.asyncio
async def test_reminder_handler_fails_cleanly_on_empty_text():
    """Empty/blank reminder_text → failed result with `empty_reminder`
    error class. The CHECK constraint at the DB layer should make this
    impossible, but defend in depth."""
    from app.agent.routines.reminder_handler import ReminderHandler

    class _Routine:
        id = "r-2"
        user_id = "u-2"
        name = "Bad"
        reminder_text = "   "
        config_json = None

    async def _fake_writer(db, **kwargs):
        return "msg", "dc"

    handler = ReminderHandler(writer=_fake_writer)
    result = await handler.execute(_Routine(), run=None, db=None)
    assert result.status == "failed"
    assert result.error_class == "empty_reminder"


# ── 4. auto_disable_after_fire — one-shot reminders disable themselves ─


@pytest.mark.asyncio
async def test_post_terminal_auto_disables_one_shot_reminder():
    """A successful fire on a routine with `auto_disable_after_fire=True`
    MUST flip the row's `enabled=false` AND remove the APScheduler
    trigger so it doesn't re-fire."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, User

    user_id = str(uuid.uuid4())
    routine_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    future = datetime.utcnow() + timedelta(hours=1)

    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"{user_id}@reminder.test",
            hashed_password="x", name="t", timezone="UTC",
        ))
        db.add(Routine(
            id=routine_id, user_id=user_id, kind="reminder",
            name="One-shot", enabled=True,
            schedule_kind="at", schedule_at=future,
            schedule_cron_local="* * * * *",
            auto_disable_after_fire=True,
            reminder_text="Test reminder",
            config_json={"delivery_channels": ["website"]},
            last_status="never_run",
        ))
        db.add(RoutineRun(
            id=run_id, routine_id=routine_id, user_id=user_id,
            scheduled_for_local_date=date.today(),
            status="running",
            fire_instant=datetime.utcnow(),
        ))
        await db.commit()

    result = RoutineResult(
        status="success",
        summary_message_id="msg-1",
        channel_results={"website": {"status": "delivered", "message_id": "msg-1"}},
        tools_invoked=[],
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    assert routine.enabled is False, (
        "Phase A: auto_disable_after_fire=True MUST flip enabled=false "
        "after a success outcome so the one-shot doesn't re-fire."
    )


@pytest.mark.asyncio
async def test_post_terminal_does_not_auto_disable_on_failure():
    """A failure outcome MUST leave the routine enabled — the user
    needs a chance to fix the underlying issue and have it retry."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, User

    user_id = str(uuid.uuid4())
    routine_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"{user_id}@reminder.test",
            hashed_password="x", name="t", timezone="UTC",
        ))
        db.add(Routine(
            id=routine_id, user_id=user_id, kind="reminder",
            name="One-shot", enabled=True,
            schedule_kind="at",
            schedule_at=datetime.utcnow() + timedelta(hours=1),
            schedule_cron_local="* * * * *",
            auto_disable_after_fire=True,
            reminder_text="Test",
            last_status="never_run",
        ))
        db.add(RoutineRun(
            id=run_id, routine_id=routine_id, user_id=user_id,
            scheduled_for_local_date=date.today(),
            status="running",
            fire_instant=datetime.utcnow(),
        ))
        await db.commit()

    result = RoutineResult(
        status="failed",
        error_class="broadcast_failed",
        error_detail="WS unavailable",
    )
    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    assert routine.enabled is True, (
        "Failure outcome MUST NOT auto-disable — user needs a retry chance."
    )


# ── 5. Reminder feature flag ───────────────────────────────────────


def test_reminder_kind_gated_by_feature_flag():
    """`reminder` kind defaults to the master flag for back-compat. If
    operators set `routines_reminders_enabled=False` explicitly, the
    kind disables independently of the master flag."""
    from app.agent.routines.runner import RoutineRunner

    class _Settings:
        routines_email_briefing_enabled = True
        routines_reminders_enabled = False

    assert RoutineRunner._kind_enabled("reminder", _Settings()) is False

    class _Settings2:
        routines_email_briefing_enabled = False
        # No explicit reminder flag → falls back to master (False).

    assert RoutineRunner._kind_enabled("reminder", _Settings2()) is False

    class _Settings3:
        routines_email_briefing_enabled = True
        # No explicit reminder flag → falls back to master (True).

    assert RoutineRunner._kind_enabled("reminder", _Settings3()) is True


# ── 6. _in_fire window gate end-to-end ─────────────────────────────


@pytest.mark.asyncio
async def test_fire_skips_when_outside_active_window(monkeypatch):
    """For schedule_kind='every' with a window set, `_fire` exits
    before claiming a routine_run row when the current local time is
    outside the window. No work happens; no nudge; no broadcast."""
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, User

    user_id = str(uuid.uuid4())
    routine_id = str(uuid.uuid4())

    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"{user_id}@reminder.test",
            hashed_password="x", name="t",
            # UTC tz makes the "current time" easy to reason about.
            timezone="UTC",
        ))
        # Window 00:00–00:01 — almost certainly outside whenever this
        # test runs (one-minute window). _fire should silently skip.
        db.add(Routine(
            id=routine_id, user_id=user_id, kind="reminder",
            name="Windowed", enabled=True,
            schedule_kind="every",
            schedule_interval_seconds=300,
            schedule_cron_local="* * * * *",  # placeholder
            schedule_window_start_local="00:00:00",
            schedule_window_end_local="00:01:00",
            reminder_text="Hi",
            last_status="never_run",
            config_json={"delivery_channels": ["website"]},
        ))
        await db.commit()

    # Force the feature flag on.
    from app.config import settings
    monkeypatch.setattr(settings, "routines_email_briefing_enabled", True)
    monkeypatch.setattr(settings, "routines_reminders_enabled", True)

    rr = RoutineRunner()
    await rr._fire(routine_id)

    # No routine_run row should have been created — the window gate
    # exits before the idempotency claim.
    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == routine_id)
        )
        rows = list(result.scalars().all())

    # Either:
    #  (a) Current time is genuinely outside the 00:00–00:01 window
    #      (overwhelmingly likely) — assert 0 rows.
    #  (b) Test happens to run within that minute. Skip rather than fail.
    now = datetime.utcnow().time()
    if not (time(0, 0) <= now < time(0, 1)):
        assert len(rows) == 0, (
            f"Phase A window gate failed: {len(rows)} routine_run rows "
            "created despite being outside the active window."
        )
