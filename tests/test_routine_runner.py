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
async def test_routine_runner_two_jobs_independent_ticks():
    """Phase D — CronService deleted; this test used to verify
    coexistence with it. Now we verify two independent APScheduler
    jobs (interval ticks) on RoutineRunner's scheduler fire cleanly
    on the same event loop — the same property the old test pinned,
    but scoped to the single surviving scheduler.

    Smoke: 1s interval each, ~6s wall-clock, both accumulate fires
    with sub-2s gaps (no event-loop blocking).
    """
    from app.agent.routines import RoutineRunner

    rr = RoutineRunner()
    await rr.start()

    a_fires: list[datetime] = []
    b_fires: list[datetime] = []

    async def a_tick():
        a_fires.append(datetime.utcnow())

    async def b_tick():
        b_fires.append(datetime.utcnow())

    rr.scheduler.add_job(a_tick, trigger=IntervalTrigger(seconds=1), id="a_coexist", replace_existing=True)
    rr.scheduler.add_job(b_tick, trigger=IntervalTrigger(seconds=1), id="b_coexist", replace_existing=True)

    await asyncio.sleep(5.5)
    await rr.stop()

    assert len(a_fires) >= 3, f"job A fired {len(a_fires)} times in 5.5s (expected ≥3)"
    assert len(b_fires) >= 3, f"job B fired {len(b_fires)} times in 5.5s (expected ≥3)"
    a_gaps = [(a_fires[i + 1] - a_fires[i]).total_seconds() for i in range(len(a_fires) - 1)]
    b_gaps = [(b_fires[i + 1] - b_fires[i]).total_seconds() for i in range(len(b_fires) - 1)]
    assert all(g < 2.0 for g in a_gaps), f"job A blocked: gaps={a_gaps}"
    assert all(g < 2.0 for g in b_gaps), f"job B blocked: gaps={b_gaps}"
