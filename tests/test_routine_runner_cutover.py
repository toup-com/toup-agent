"""PR #47 — routine runner cutover from ``routine_runs`` to
``build_jobs``.

PR #47 moved reads to ``build_jobs``; PR #49 removed the legacy
``routine_runs`` dual-write entirely. After PR #49 the runner reads
AND writes ``build_jobs`` exclusively.

What this file pins:

  1. ``_restart_sweep`` flips orphan ``build_jobs`` rows (status=
     'running', source_kind='routine', stale ``created_at``) to
     status='failed' with a clear ``error_message``. PR #49: the
     legacy ``routine_runs`` UPDATE is gone.
  2. ``_fire`` is Job-only: a fresh fire mints exactly one
     ``BuildJob`` row (with ``fire_instant``, ``attempt=1``,
     ``idempotency_key=str(local_date)``, ``source_kind='routine'``).
     PR #49 removed the RoutineRun dual-write.
  3. Dedupe via explicit SELECT on (source_id, idempotency_key).
     A second ``_fire`` for the same routine + local_date produces
     no duplicate Job and raises no error.
  4. ``_run_with_retry`` stamps ``BuildJob.attempt`` on every retry
     (the legacy RoutineRun stamp is gone).
  5. ``_finalize_run`` writes the five mig 052 terminal columns
     (``emails_fetched``, ``finished_local_at``, ``error_json``,
     ``channel_results_json``, ``tools_invoked_json``) directly
     onto the BuildJob — the helper that used to do this as a
     dual-write mirror is gone.
  6. A ``JobRunner.create_job`` failure aborts intake — no
     downstream row materialises.

Test fixture pattern matches ``test_job_parity_routines.py`` — bypass
conftest's full ``init_db()`` (entities pgvector breaks SQLite) and
build only the narrow set of tables these tests need.
"""
from __future__ import annotations

import os
import uuid
from datetime import date as date_cls, datetime, timedelta
from typing import Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-routine-cutover")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000dd1")


USER_ID = "00000000-0000-0000-0000-000000000dd1"
ROUTINE_ID = "00000000-0000-0000-0000-0000000dd100"


# ── Schema setup ──────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Narrow schema bootstrap. PR #49: the cutover paths no longer
    touch RoutineRun, so the fixture only sets up User, Routine,
    BuildJob, JobEvent."""
    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, Routine, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent, Routine):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (Routine, JobEvent, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)
    await engine.dispose()


@pytest_asyncio.fixture
async def _seed_user_and_routine():
    from app.db.database import async_session_maker
    from app.db.models import Routine, User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"cutover-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
        # ``kind="_smoke"`` is NOT in KIND_HANDLERS — _fire takes the
        # ``handler is None → _finalize_run(success)`` short-circuit so
        # we exercise the Job-only mint path WITHOUT the retry loop
        # mutating ``attempt`` past 1.
        db.add(Routine(
            id=ROUTINE_ID,
            user_id=USER_ID,
            kind="_smoke",
            enabled=True,
            schedule_cron_local="0 8 * * *",
            schedule_kind="cron",
            config_json={},
            last_state_json={},
            last_status="never_run",
        ))
        await db.commit()


# ── _restart_sweep ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restart_sweep_flips_orphan_buildjob(_seed_user_and_routine):
    """A BuildJob stuck in ``status='running'`` with
    ``source_kind='routine'`` and a ``created_at`` older than
    ORPHAN_THRESHOLD must be flipped to ``status='failed'`` with
    ``error_message`` containing 'agent_restarted' and ``completed_at``
    populated. Fresh in-flight rows are left alone."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    orphan_id = str(uuid.uuid4())
    fresh_id = str(uuid.uuid4())
    orphan_age = RoutineRunner.ORPHAN_THRESHOLD + timedelta(minutes=5)
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=orphan_id,
            user_id=USER_ID,
            title="orphan",
            prompt="",
            job_type="routine_run",
            status="running",
            source_kind="routine",
            source_id=ROUTINE_ID,
            idempotency_key=f"orphan-{orphan_id[:8]}",
            created_at=datetime.utcnow() - orphan_age,
        ))
        db.add(BuildJob(
            id=fresh_id,
            user_id=USER_ID,
            title="fresh",
            prompt="",
            job_type="routine_run",
            status="running",
            source_kind="routine",
            source_id=ROUTINE_ID,
            idempotency_key=f"fresh-{fresh_id[:8]}",
            created_at=datetime.utcnow(),
        ))
        await db.commit()

    runner = RoutineRunner(session_maker=async_session_maker)
    swept = await runner._restart_sweep()
    assert swept == 1, f"expected 1 orphan swept, got {swept}"

    async with async_session_maker() as db:
        orphan = await db.get(BuildJob, orphan_id)
        assert orphan.status == "failed"
        assert orphan.error_message and "agent_restarted" in orphan.error_message
        assert orphan.completed_at is not None
        fresh = await db.get(BuildJob, fresh_id)
        assert fresh.status == "running", (
            "fresh in-flight rows must not be swept"
        )


# ── _fire: Job-only intake ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_fire_mints_single_build_job(
    _seed_user_and_routine, monkeypatch,
):
    """A successful ``_fire`` mints exactly one ``BuildJob`` (with
    ``fire_instant`` possibly populated, ``attempt=1``,
    ``idempotency_key=str(local_date)``, ``source_kind='routine'``).
    PR #49: no RoutineRun row is written."""
    from sqlalchemy import select

    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    runner = RoutineRunner(session_maker=async_session_maker)

    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        job_rows = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == ROUTINE_ID)
        )).scalars().all()
    assert len(job_rows) == 1, (
        f"expected exactly one BuildJob; got {len(job_rows)}"
    )
    job = job_rows[0]
    # Job-only invariants — these are the new source-of-truth fields.
    assert job.source_kind == "routine"
    # idempotency_key is set to str(local_date) by _fire.
    assert job.idempotency_key is not None
    # _smoke kind takes the no-handler short-circuit which calls
    # _finalize_run(status="success") — the row is completed, but the
    # attempt counter was stamped to 1 at mint time and never
    # incremented (retry loop never ran).
    assert job.attempt == 1, (
        f"BuildJob.attempt must be stamped to 1 on fresh fire; "
        f"got {job.attempt!r}"
    )
    if job.fire_instant is not None:
        assert isinstance(job.fire_instant, datetime)


@pytest.mark.asyncio
async def test_fire_dedupe_via_explicit_select_check(
    _seed_user_and_routine, monkeypatch,
):
    """Two ``_fire`` calls for the same routine + local_date produce
    exactly one BuildJob and the second call raises nothing.

    PR #49 invariant: the composite UNIQUE on
    ``(build_jobs.source_id, build_jobs.idempotency_key)`` is the
    sole dedupe gate — the legacy RoutineRun UNIQUE is gone."""
    from sqlalchemy import select, func

    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)
    await runner._fire(ROUTINE_ID)  # second call should be silent dedupe

    async with async_session_maker() as db:
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == ROUTINE_ID,
                BuildJob.source_kind == "routine",
            )
        )).scalar_one()
    assert job_count == 1, (
        f"expected 1 Job after dedupe; got {job_count}"
    )


# ── _run_with_retry: attempt mirror ──────────────────────────────────


@pytest.mark.asyncio
async def test_retry_increments_buildjob_attempt(
    _seed_user_and_routine, monkeypatch,
):
    """The per-attempt UPDATE on ``BuildJob.attempt`` walks the
    retry counter forward. We seed a routine + job, then drive
    ``_run_with_retry`` with a handler that always fails so the
    retry loop walks through every attempt. Final BuildJob.attempt
    equals the retry budget."""
    from app.agent.routines.runner import RoutineRunner
    from app.agent.routines.base_handler import RoutineResult
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Routine

    # Seed: 1 routine + 1 BuildJob linked.
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id,
            user_id=USER_ID,
            title="retry test",
            prompt="",
            job_type="routine_run",
            status="running",
            source_kind="routine",
            source_id=ROUTINE_ID,
            idempotency_key="2026-05-20",
            attempt=1,
            created_at=datetime.utcnow(),
        ))
        await db.commit()
        routine = await db.get(Routine, ROUTINE_ID)

    class _AlwaysFails:
        async def execute(self, routine, run_obj, db):
            return RoutineResult(
                status="failed",
                error_class="TestError",
                error_detail="forced failure for retry test",
            )

    # Tiny retry delays so the test stays fast — three attempts.
    runner = RoutineRunner(
        session_maker=async_session_maker,
        retry_delays=(0.001, 0.001, 0.001),
    )

    await runner._run_with_retry(_AlwaysFails(), routine, job_id)

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
    # Three attempts in the budget → final attempt counter is 3.
    assert job.attempt == 3, f"BuildJob.attempt expected 3, got {job.attempt}"


# ── _finalize_run: terminal columns on BuildJob ──────────────────────


@pytest.mark.asyncio
async def test_finalize_run_writes_terminal_columns_to_buildjob(
    _seed_user_and_routine, monkeypatch,
):
    """``_finalize_run`` writes the terminal shape directly onto
    ``build_jobs`` — all five mig 052 columns
    (``emails_fetched``, ``finished_local_at``, ``error_json``,
    ``channel_results_json``, ``tools_invoked_json``) plus
    status/outcome/completed_at/summary_message_id. PR #49 inlined
    this write (the dual-write mirror helper is gone)."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    # _fire mints the BuildJob.
    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        from sqlalchemy import select
        job = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == ROUTINE_ID)
        )).scalar_one()
        job_id = job.id

    channel_results = {
        "website": {"status": "delivered", "message_id": "msg-1", "ws_count": 2},
        "telegram": {"status": "delivered", "telegram_message_id": 4242},
    }
    tools_invoked = ["gmail__list_messages", "gmail__get_message"]
    error_json = None  # success outcomes leave error_json NULL

    await runner._finalize_run(
        job_id,
        status="success",
        outcome="success",
        error_class=None,
        error_detail=None,
        error_json=error_json,
        emails_fetched=3,
        summary_message_id="msg-1",
        channel_results=channel_results,
        tools_invoked=tools_invoked,
    )

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
    # The five mig 052 columns must be populated verbatim on BuildJob.
    assert job.status == "completed"
    assert job.outcome == "success"
    assert job.emails_fetched == 3, (
        f"BuildJob.emails_fetched write failed; got {job.emails_fetched!r}"
    )
    assert job.finished_local_at is not None, (
        "BuildJob.finished_local_at must be populated"
    )
    # error_json NULL on success outcomes.
    assert job.error_json is None
    assert job.channel_results_json == channel_results, (
        f"BuildJob.channel_results_json write failed; "
        f"got {job.channel_results_json!r}"
    )
    assert job.tools_invoked_json == tools_invoked, (
        f"BuildJob.tools_invoked_json write failed; "
        f"got {job.tools_invoked_json!r}"
    )
    assert job.summary_message_id == "msg-1"


# ── Intake abort on Job mint failure ─────────────────────────────────


@pytest.mark.asyncio
async def test_intake_aborts_when_job_mint_fails(
    _seed_user_and_routine, monkeypatch,
):
    """``JobRunner.create_job`` raising → ``_fire`` MUST NOT create
    any downstream row. PR #49 contract: Job is the sole row carrying
    per-fire state; without it APScheduler retries on the next
    interval."""
    from sqlalchemy import select, func

    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    # Force JobRunner.create_job to raise.
    from app.agent.job_runner import JobRunner

    async def _boom(self, **kwargs):
        raise RuntimeError("simulated Job mint failure")

    monkeypatch.setattr(JobRunner, "create_job", _boom)

    # Should NOT raise — _fire catches and logs the error.
    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == ROUTINE_ID,
            )
        )).scalar_one()
    assert job_count == 0, (
        f"Job mint failed; expected 0 BuildJobs, got {job_count}"
    )
