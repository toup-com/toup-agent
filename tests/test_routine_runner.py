"""Gate 1 tests for the RoutineRunner — lifecycle, idempotency, restart
sweep, tz fallback, reload idempotency, /_runner_status endpoint, and
coexistence with CronService on the same event loop.

No Gmail, no LLM, no Message writes — those land in Gate 2.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, timedelta

import pytest
import pytest_asyncio
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select


# ── Helpers ──────────────────────────────────────────────────────────


async def _make_user(timezone: str = "UTC") -> str:
    """Seed a User row and return its id. The runner reads timezone from
    this row at fire time."""
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"{user_id}@routine-test.local",
                hashed_password="x",
                name="Routine Test User",
                timezone=timezone,
            )
        )
        await db.commit()
    return user_id


async def _make_routine(
    user_id: str,
    *,
    kind: str = "_smoke",
    cron: str = "0 7 * * *",
    enabled: bool = True,
) -> str:
    """Seed a Routine row. `_smoke` kind bypasses the production feature
    flag (see RoutineRunner._kind_enabled)."""
    from app.db import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid,
                user_id=user_id,
                kind=kind,
                enabled=enabled,
                schedule_cron_local=cron,
                last_status="never_run",
            )
        )
        await db.commit()
    return rid


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runner_starts_and_stops_with_zero_routines():
    """The bare lifecycle path — no routines in DB, runner still boots and
    shuts down cleanly. Reveals any "what if the list is empty" bugs."""
    from app.agent.routines import RoutineRunner

    rr = RoutineRunner()
    await rr.start()
    assert rr.scheduler.running is True
    snap = rr.status_snapshot()
    # Snapshot must carry the baseline lifecycle fields. Additional
    # observability counters (missed_fires_total, reload_failures_total,
    # reconcile_runs_total, reconcile_active) are present in current
    # implementations — checked individually rather than equality so this
    # test doesn't break every time we add a counter.
    assert snap["running"] is True
    assert snap["routines_registered"] == 0
    assert snap["next_fire_at"] is None
    await rr.stop()
    assert rr.scheduler.running is False


@pytest.mark.asyncio
async def test_restart_sweep_marks_orphaned_running_rows():
    """A 'running' run row older than 10min from a previous boot must
    be flipped to failed/agent_restarted on next start()."""
    from app.agent.routines import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import RoutineRun

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)
    orphan_id = str(uuid.uuid4())
    fresh_id = str(uuid.uuid4())

    async with async_session_maker() as db:
        # Orphaned — should be swept
        db.add(
            RoutineRun(
                id=orphan_id,
                routine_id=rid,
                user_id=user_id,
                scheduled_for_local_date=date.today() - timedelta(days=1),
                started_at=datetime.utcnow() - timedelta(minutes=15),
                status="running",
            )
        )
        # Fresh — under threshold, must NOT be touched (a parallel run
        # might still legitimately be in progress)
        db.add(
            RoutineRun(
                id=fresh_id,
                routine_id=rid,
                user_id=user_id,
                scheduled_for_local_date=date.today(),
                started_at=datetime.utcnow() - timedelta(minutes=2),
                status="running",
            )
        )
        await db.commit()

    rr = RoutineRunner()
    await rr.start()
    try:
        async with async_session_maker() as db:
            orphan = (await db.execute(select(RoutineRun).where(RoutineRun.id == orphan_id))).scalar_one()
            fresh = (await db.execute(select(RoutineRun).where(RoutineRun.id == fresh_id))).scalar_one()
        assert orphan.status == "failed"
        assert orphan.error_class == "agent_restarted"
        assert orphan.finished_at is not None
        assert fresh.status == "running"
        assert fresh.error_class is None
    finally:
        await rr.stop()


@pytest.mark.asyncio
async def test_invalid_timezone_falls_back_to_utc_with_log(caplog):
    """An invalid User.timezone string must not crash. Registration falls
    back to UTC and emits a structured tz_fallback warning so existing
    log-grep tooling catches it."""
    from app.agent.routines import RoutineRunner

    user_id = await _make_user(timezone="Mars/Olympus_Mons")
    rid = await _make_routine(user_id)

    caplog.set_level(logging.WARNING, logger="app.agent.routines.runner")
    rr = RoutineRunner()
    await rr.start()
    try:
        assert rid in rr._jobs
        # Match the agent_runner.py:2103 log shape.
        assert "tz_fallback source=invalid_tz" in caplog.text
        assert "Mars/Olympus_Mons" in caplog.text
        snap = rr.status_snapshot()
        assert snap["routines_registered"] == 1
        assert snap["next_fire_at"] is not None
    finally:
        await rr.stop()


@pytest.mark.asyncio
async def test_reload_routine_is_idempotent():
    """Calling reload_routine() twice with the same id must leave exactly
    one APScheduler job. Catches the "double-register" footgun the API
    will hit when the user toggles enable+disable rapidly."""
    from app.agent.routines import RoutineRunner

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)

    rr = RoutineRunner()
    await rr.start()
    try:
        baseline = len(rr.scheduler.get_jobs())
        await rr.reload_routine(rid)
        await rr.reload_routine(rid)
        await rr.reload_routine(rid)
        assert len(rr.scheduler.get_jobs()) == baseline
    finally:
        await rr.stop()


@pytest.mark.asyncio
async def test_reload_routine_removes_when_disabled():
    """If a routine flips to enabled=False, reload_routine() must
    unregister its trigger (the toggle-off path from the API)."""
    from app.agent.routines import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)

    rr = RoutineRunner()
    await rr.start()
    try:
        assert rid in rr._jobs
        async with async_session_maker() as db:
            routine = await db.get(Routine, rid)
            routine.enabled = False
            await db.commit()
        await rr.reload_routine(rid)
        assert rid not in rr._jobs
    finally:
        await rr.stop()


@pytest.mark.asyncio
async def test_idempotency_collision_on_second_fire_same_day():
    """Two _fire() calls on the same (routine_id, today's local_date) —
    second hits the UNIQUE and is a silent no-op. Exactly one routine_runs
    row is left."""
    from app.agent.routines import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import RoutineRun

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)

    rr = RoutineRunner()
    # Don't start the scheduler — call _fire() directly.
    await rr._fire(rid)
    await rr._fire(rid)

    async with async_session_maker() as db:
        rows = (await db.execute(select(RoutineRun).where(RoutineRun.routine_id == rid))).scalars().all()
    assert len(rows) == 1
    assert rows[0].status in ("success", "running")  # Gate 1 finalizes as success


@pytest.mark.asyncio
async def test_runner_status_snapshot_shape():
    """`RoutineRunner.status_snapshot()` is the data layer behind the
    `/api/routines/_runner_status` HTTP endpoint. Test the snapshot
    directly here; the HTTP-level integration test that wires it
    through the FastAPI app lives in Gate 3 (with the full CRUD surface)."""
    from app.agent.routines import RoutineRunner

    # Down case: scheduler not started
    rr_down = RoutineRunner()
    snap = rr_down.status_snapshot()
    assert snap["running"] is False
    assert snap["routines_registered"] == 0
    assert snap["next_fire_at"] is None

    # Up case with one registered routine
    user_id = await _make_user("UTC")
    await _make_routine(user_id)
    rr = RoutineRunner()
    await rr.start()
    try:
        snap = rr.status_snapshot()
        assert snap["running"] is True
        assert snap["routines_registered"] == 1
        assert isinstance(snap["next_fire_at"], str)  # iso8601
    finally:
        await rr.stop()


@pytest.mark.asyncio
async def test_feature_flag_off_skips_email_briefing_kind():
    """A routine with kind='email_briefing' must NOT register if the
    per-tenant feature flag is off (the default)."""
    from app.agent.routines import RoutineRunner
    from app.config import settings

    assert settings.routines_email_briefing_enabled is False
    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id, kind="email_briefing")

    rr = RoutineRunner()
    await rr.start()
    try:
        assert rid not in rr._jobs
        assert rr.status_snapshot()["routines_registered"] == 0
    finally:
        await rr.stop()


@pytest.mark.asyncio
async def test_coexistence_routine_runner_and_cron_service():
    """RoutineRunner.scheduler and CronService.scheduler fire independently
    on the same event loop. Smoke: 1s interval each, ~6s wall-clock, both
    must accumulate fires.

    Gate evidence requires 60s; this in-pytest variant is capped at ~6s so
    CI doesn't drag. The full 60s capture is in the runner_smoke.py CLI.
    """
    from app.agent.routines import RoutineRunner
    from app.agent.cron_service import CronService

    rr = RoutineRunner()
    cs = CronService()
    await rr.start()
    await cs.start()

    rr_fires: list[datetime] = []
    cs_fires: list[datetime] = []

    async def rr_tick():
        rr_fires.append(datetime.utcnow())

    async def cs_tick():
        cs_fires.append(datetime.utcnow())

    rr.scheduler.add_job(rr_tick, trigger=IntervalTrigger(seconds=1), id="rr_coexist", replace_existing=True)
    cs.scheduler.add_job(cs_tick, trigger=IntervalTrigger(seconds=1), id="cs_coexist", replace_existing=True)

    await asyncio.sleep(5.5)

    await rr.stop()
    await cs.stop()

    assert len(rr_fires) >= 3, f"RoutineRunner fired {len(rr_fires)} times in 5.5s (expected ≥3)"
    assert len(cs_fires) >= 3, f"CronService fired {len(cs_fires)} times in 5.5s (expected ≥3)"
    # Cross-check independence: the gap between consecutive fires on each
    # scheduler should stay near 1s — if one were blocking the other, the
    # gaps would balloon.
    rr_gaps = [(rr_fires[i + 1] - rr_fires[i]).total_seconds() for i in range(len(rr_fires) - 1)]
    cs_gaps = [(cs_fires[i + 1] - cs_fires[i]).total_seconds() for i in range(len(cs_fires) - 1)]
    assert all(g < 2.0 for g in rr_gaps), f"RoutineRunner blocked: gaps={rr_gaps}"
    assert all(g < 2.0 for g in cs_gaps), f"CronService blocked: gaps={cs_gaps}"


# ── Catch-up: never-fired one-shot reminders whose schedule_at is in the past ──

@pytest.mark.asyncio
async def test_past_reminder_never_fired_catches_up_immediately():
    """A `kind=reminder` row with schedule_kind='at', schedule_at in the
    past, and last_run_at=NULL is a never-delivered one-shot. Without
    catch-up the user never sees the reminder — the runner silently
    drops it as "past_at, already delivered or expired". Production bug
    reported 2026-05-21: agent restarted (image upgrade) at 01:35 UTC
    while the reminder was scheduled for 01:28 UTC; the reminder stayed
    `last_status='never_run'` forever.

    Expected: _register_trigger_for reschedules the routine for
    ~now+5s with a DateTrigger so the user gets the (slightly-late)
    reminder. auto_disable_after_fire keeps the one-shot semantic
    intact — the rescheduled fire still won't repeat.
    """
    from app.agent.routines import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user(timezone="UTC")

    # Seed a reminder scheduled 2 minutes in the past, never fired.
    rid = str(uuid.uuid4())
    past_at = datetime.utcnow() - timedelta(minutes=2)
    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid,
                user_id=user_id,
                kind="reminder",
                enabled=True,
                schedule_kind="at",
                schedule_at=past_at,
                schedule_cron_local="* * * * *",  # placeholder, ignored for kind=at
                reminder_text="catch-up smoke test",
                auto_disable_after_fire=True,
                last_status="never_run",
                last_run_at=None,
            )
        )
        await db.commit()

    # Enable the reminder kind so _load_enabled_routines doesn't filter it.
    from app.config import settings
    settings.routines_reminders_enabled = True

    runner = RoutineRunner()
    await runner.start()
    try:
        # After start(), the catch-up branch should have registered the
        # routine with APScheduler instead of dropping it.
        job = runner.scheduler.get_job(rid)
        assert job is not None, (
            "past-scheduled reminder with last_run_at=None was silently "
            "dropped — catch-up branch didn't fire"
        )
        # The reschedule lands ~5s from now. Wide window so test isn't
        # flaky under CI load.
        now = datetime.utcnow().replace(tzinfo=job.next_run_time.tzinfo)
        delta = (job.next_run_time - now).total_seconds()
        assert -2 < delta < 30, (
            f"catch-up fire was scheduled outside expected window: "
            f"next_run_time={job.next_run_time} now={now} delta={delta:.1f}s"
        )
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_past_reminder_already_fired_is_NOT_re_caught_up():
    """If the reminder DID fire previously (last_run_at IS NOT NULL),
    catch-up MUST NOT re-fire it. Otherwise every restart would re-send
    every old one-shot — spam in chat / Telegram / WhatsApp.
    """
    from app.agent.routines import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user(timezone="UTC")

    rid = str(uuid.uuid4())
    past_at = datetime.utcnow() - timedelta(hours=2)
    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid,
                user_id=user_id,
                kind="reminder",
                enabled=True,
                schedule_kind="at",
                schedule_at=past_at,
                schedule_cron_local="* * * * *",
                reminder_text="already delivered",
                auto_disable_after_fire=True,
                last_status="success",
                last_run_at=past_at,  # already fired at scheduled time
            )
        )
        await db.commit()

    # Enable the reminder kind so _load_enabled_routines doesn't filter it.
    from app.config import settings
    settings.routines_reminders_enabled = True

    runner = RoutineRunner()
    await runner.start()
    try:
        # Already delivered → silently skipped, no APScheduler job.
        assert runner.scheduler.get_job(rid) is None, (
            "already-delivered reminder got re-registered — would re-spam the user"
        )
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_past_non_reminder_at_routine_is_NOT_caught_up():
    """Catch-up is reminder-specific. A past `at` schedule for any other
    kind (email_briefing, agent_task, etc.) keeps the old skip behavior
    — those aren't user-visible "ping me" one-shots and re-firing
    them after a restart could be expensive (LLM calls) or wrong.
    """
    from app.agent.routines import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user(timezone="UTC")

    rid = str(uuid.uuid4())
    past_at = datetime.utcnow() - timedelta(minutes=2)
    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid,
                user_id=user_id,
                kind="agent_task",  # NOT reminder
                enabled=True,
                schedule_kind="at",
                schedule_at=past_at,
                schedule_cron_local="* * * * *",
                prompt_text="some task",
                last_status="never_run",
                last_run_at=None,
            )
        )
        await db.commit()

    # Enable the reminder kind so _load_enabled_routines doesn't filter it.
    from app.config import settings
    settings.routines_reminders_enabled = True

    runner = RoutineRunner()
    await runner.start()
    try:
        # Past `at` for non-reminder is still skipped.
        assert runner.scheduler.get_job(rid) is None, (
            "past-scheduled agent_task got caught up — catch-up should "
            "only apply to kind=reminder one-shots"
        )
    finally:
        await runner.stop()
