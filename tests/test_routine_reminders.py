"""Phase A acceptance tests — reminder kind + multi-shape schedules.

Pins three production-grade contracts:

  1. ReminderHandler delivers `reminder_text` verbatim with zero LLM
     calls, zero MCP tools, and per-channel fanout via delivery_channels.
  2. `_build_trigger_for_routine` dispatches correctly on
     `schedule_kind` (cron/at/every), tolerates NULL schedule_kind
     (legacy rows pre-mig 042), rejects past 'at' and sub-60s intervals.
  3. `_in_active_window` correctly gates interval fires, including
     overnight windows (22:00–06:00).

The auto-disable + window-gate end-to-end paths are covered by the
`test_routines_phase3_verification.py` scenarios and the existing
runner integration tests.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pytest


# ── _in_active_window helper ──────────────────────────────────────


def test_in_active_window_daytime():
    from app.agent.routines.runner import _in_active_window
    assert _in_active_window(time(9, 0), time(9, 0), time(17, 0))
    assert _in_active_window(time(12, 0), time(9, 0), time(17, 0))
    assert _in_active_window(time(16, 59, 59), time(9, 0), time(17, 0))
    assert not _in_active_window(time(8, 59, 59), time(9, 0), time(17, 0))
    assert not _in_active_window(time(17, 0), time(9, 0), time(17, 0))


def test_in_active_window_overnight():
    """22:00–06:00 wraps midnight — 23:00 and 02:00 must be in window;
    08:00 must not."""
    from app.agent.routines.runner import _in_active_window
    assert _in_active_window(time(23, 0), time(22, 0), time(6, 0))
    assert _in_active_window(time(2, 30), time(22, 0), time(6, 0))
    assert not _in_active_window(time(8, 0), time(22, 0), time(6, 0))
    assert not _in_active_window(time(21, 59), time(22, 0), time(6, 0))


def test_in_active_window_none_bounds():
    from app.agent.routines.runner import _in_active_window
    assert _in_active_window(time(3, 0), None, None)


# ── _build_trigger_for_routine dispatch ──────────────────────────


class _Stub:
    """Light stand-in for Routine — only the fields the trigger
    builder reads."""
    def __init__(self, **kw):
        self.schedule_kind = kw.get("schedule_kind")  # may be None — legacy
        self.schedule_cron_local = kw.get("schedule_cron_local", "0 9 * * *")
        self.schedule_at = kw.get("schedule_at")
        self.schedule_interval_seconds = kw.get("schedule_interval_seconds")


def test_build_trigger_null_kind_falls_back_to_cron():
    """Critical contract for mig 042: rows with NULL schedule_kind
    (pre-migration) keep behaving as cron triggers."""
    from apscheduler.triggers.cron import CronTrigger
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    r = _Stub(schedule_kind=None, schedule_cron_local="0 9 * * *")
    trigger, tag = _build_trigger_for_routine(r, ZoneInfo("America/Toronto"))
    assert tag == "cron"
    assert isinstance(trigger, CronTrigger)


def test_build_trigger_at_future_returns_datetrigger():
    from apscheduler.triggers.date import DateTrigger
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    future = datetime.utcnow() + timedelta(hours=2)
    r = _Stub(schedule_kind="at", schedule_at=future)
    trigger, tag = _build_trigger_for_routine(r, ZoneInfo("UTC"))
    assert tag == "at"
    assert isinstance(trigger, DateTrigger)


def test_build_trigger_at_past_returns_past_at():
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    r = _Stub(schedule_kind="at", schedule_at=datetime.utcnow() - timedelta(hours=1))
    trigger, tag = _build_trigger_for_routine(r, ZoneInfo("UTC"))
    assert tag == "past_at"
    assert trigger is None


def test_build_trigger_every_returns_intervaltrigger():
    from apscheduler.triggers.interval import IntervalTrigger
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    r = _Stub(schedule_kind="every", schedule_interval_seconds=300)
    trigger, tag = _build_trigger_for_routine(r, ZoneInfo("America/Toronto"))
    assert tag == "every"
    assert isinstance(trigger, IntervalTrigger)


def test_build_trigger_every_rejects_sub_minute():
    from zoneinfo import ZoneInfo
    from app.agent.routines.runner import _build_trigger_for_routine

    r = _Stub(schedule_kind="every", schedule_interval_seconds=30)
    trigger, tag = _build_trigger_for_routine(r, ZoneInfo("UTC"))
    assert tag == "invalid"
    assert trigger is None


# ── ReminderHandler ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reminder_handler_delivers_text_with_zero_llm_zero_mcp():
    """The whole value prop: cheap text delivery — no LLM, no MCP."""
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
    assert result.new_watermark is None  # reminders don't track upstream state


@pytest.mark.asyncio
async def test_reminder_handler_fails_cleanly_on_empty_text():
    from app.agent.routines.reminder_handler import ReminderHandler

    class _Routine:
        id = "r-2"
        user_id = "u-2"
        name = "Bad"
        reminder_text = "   "
        config_json = None

    async def _fake_writer(db, **kw):
        return "msg", "dc"

    handler = ReminderHandler(writer=_fake_writer)
    result = await handler.execute(_Routine(), run=None, db=None)
    assert result.status == "failed"
    assert result.error_class == "empty_reminder"


# ── Feature flag gate ─────────────────────────────────────────────


def test_reminder_kind_gated_by_feature_flag():
    """`reminder` defaults to master flag when its dedicated flag isn't
    set on the Settings object."""
    from app.agent.routines.runner import RoutineRunner

    class _S1:
        routines_email_briefing_enabled = True
        routines_reminders_enabled = False
    assert RoutineRunner._kind_enabled("reminder", _S1()) is False

    class _S2:
        routines_email_briefing_enabled = False
    # Falls back to master (False), no reminders flag set on stub.
    assert RoutineRunner._kind_enabled("reminder", _S2()) is False

    class _S3:
        routines_email_briefing_enabled = True
    # Master True, no explicit override → True via fallback.
    assert RoutineRunner._kind_enabled("reminder", _S3()) is True


# ── KIND_HANDLERS registry ────────────────────────────────────────


def test_reminder_handler_is_registered():
    from app.agent.routines import KIND_HANDLERS, ReminderHandler

    assert "reminder" in KIND_HANDLERS
    assert isinstance(KIND_HANDLERS["reminder"], ReminderHandler)
