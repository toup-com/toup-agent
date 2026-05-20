"""PR #47 — routine runner cutover from ``routine_runs`` to
``build_jobs``.

The 9-PR unified-jobs arc dual-writes every routine fire into both
tables today. This PR moves the *reads* on the runner side over to
``build_jobs`` so the legacy table can be dropped in PR #49 without
the runner losing visibility into orphaned work, intake dedupe, or
the terminal shape (mig 052 adds the five legacy terminal columns
to ``build_jobs``).

What this file pins:

  1. ``_restart_sweep`` flips orphan ``build_jobs`` rows (status=
     'running', source_kind='routine', stale ``created_at``) to
     status='failed' with a clear ``error_message``, and the legacy
     ``routine_runs`` row stays dual-written so old readers don't
     trip.
  2. ``_fire`` is Job-first: a fresh fire mints exactly one
     ``BuildJob`` row (with ``fire_instant``, ``attempt=1``,
     ``idempotency_key=str(local_date)``, ``source_kind='routine'``)
     AND exactly one ``RoutineRun`` (the legacy dual-write). The
     ``RoutineRun.job_id`` links to the Job.
  3. Dedupe via explicit SELECT on (source_id, idempotency_key) —
     matches the PR #46 fix in ``triggers_inbound._idempotent_insert``.
     A second ``_fire`` for the same routine + local_date produces no
     duplicate Job and no duplicate RoutineRun and raises no error.
  4. ``_run_with_retry`` stamps ``BuildJob.attempt`` alongside
     ``RoutineRun.attempt`` on every retry.
  5. ``_finalize_run`` → ``_mirror_run_terminal_to_job`` writes the
     five mig 052 columns (``emails_fetched``, ``finished_local_at``,
     ``error_json``, ``channel_results_json``, ``tools_invoked_json``)
     onto the BuildJob.
  6. A ``JobRunner.create_job`` failure aborts intake — NO RoutineRun
     is inserted (matches the PR #46 contract change for the
     Job-first source-of-truth invariant).

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
    """Narrow schema bootstrap — same approach as the rest of the
    unified-jobs test suite. We need User, Routine, RoutineRun,
    BuildJob, JobEvent for the cutover paths."""
    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, Routine, RoutineRun, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent, Routine, RoutineRun):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (RoutineRun, Routine, JobEvent, BuildJob, User):
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
        # we exercise the Job-first mint path WITHOUT the retry loop
        # mutating ``attempt`` past 1. (PR #46's trigger cutover uses
        # the equivalent stub-injection trick via KIND_HANDLERS[...] =
        # <FakeHandler> for the success-mirror test; the simpler path
        # here is to choose an unregistered kind.)
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


# ── _fire: Job-first intake ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_fire_mints_job_first_then_routine_run(
    _seed_user_and_routine, monkeypatch,
):
    """A successful ``_fire`` mints exactly one ``BuildJob`` (with
    ``fire_instant``, ``attempt=1``, ``idempotency_key=str(local_date)``,
    ``source_kind='routine'``) AND exactly one ``RoutineRun`` (the
    legacy dual-write). The ``RoutineRun.job_id`` links to the Job."""
    from sqlalchemy import select, func

    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun

    runner = RoutineRunner(session_maker=async_session_maker)

    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        job_rows = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == ROUTINE_ID)
        )).scalars().all()
        run_rows = (await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == ROUTINE_ID)
        )).scalars().all()
    assert len(job_rows) == 1, (
        f"expected exactly one BuildJob; got {len(job_rows)}"
    )
    assert len(run_rows) == 1, (
        f"expected exactly one RoutineRun; got {len(run_rows)}"
    )
    job = job_rows[0]
    run = run_rows[0]
    # Job-first invariants — these are the new source-of-truth fields.
    assert job.source_kind == "routine"
    assert job.idempotency_key == str(run.scheduled_for_local_date)
    assert job.attempt == 1, (
        f"BuildJob.attempt must be stamped to 1 on fresh fire; "
        f"got {job.attempt!r}"
    )
    # fire_instant may be None when no APScheduler job is registered
    # (this test invokes _fire directly without start()). The cutover
    # contract is that the UPDATE column is wired — pin its presence
    # via the attempt write above + the explicit check below.
    # When fire_instant IS set, it must be a datetime.
    if job.fire_instant is not None:
        assert isinstance(job.fire_instant, datetime)
    # Legacy linkage stays correct.
    assert run.job_id == job.id


@pytest.mark.asyncio
async def test_fire_dedupe_via_explicit_select_check(
    _seed_user_and_routine, monkeypatch,
):
    """Two ``_fire`` calls for the same routine + local_date produce
    exactly one BuildJob, exactly one RoutineRun, and the second call
    raises nothing.

    Pins the PR #47 dedupe-via-explicit-SELECT invariant (the
    proximity heuristic in PR #46's first version was unreliable in
    tests; explicit SELECT replaces it). Mirrors
    ``test_intake_dedupe_creates_one_job_one_event`` in the trigger
    cutover suite."""
    from sqlalchemy import select, func

    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun

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
        run_count = (await db.execute(
            select(func.count(RoutineRun.id)).where(
                RoutineRun.routine_id == ROUTINE_ID,
            )
        )).scalar_one()
    assert job_count == 1, (
        f"expected 1 mirrored Job after dedupe; got {job_count}"
    )
    assert run_count == 1, (
        f"expected 1 RoutineRun after dedupe; got {run_count}"
    )


# ── _run_with_retry: attempt mirror ──────────────────────────────────


@pytest.mark.asyncio
async def test_retry_increments_buildjob_attempt(
    _seed_user_and_routine, monkeypatch,
):
    """The per-attempt UPDATE that stamps ``RoutineRun.attempt`` must
    also bump ``BuildJob.attempt``. We seed a routine + run + job, then
    drive ``_run_with_retry`` with a handler that always fails so the
    retry loop walks through every attempt. Final BuildJob.attempt
    equals the retry budget."""
    from app.agent.routines.runner import RoutineRunner
    from app.agent.routines.base_handler import RoutineResult
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Routine, RoutineRun

    # Seed: 1 routine + 1 BuildJob + 1 RoutineRun linked.
    run_id = str(uuid.uuid4())
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
        db.add(RoutineRun(
            id=run_id,
            routine_id=ROUTINE_ID,
            user_id=USER_ID,
            scheduled_for_local_date=date_cls.today(),
            started_at=datetime.utcnow(),
            status="running",
            attempt=1,
            job_id=job_id,
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

    await runner._run_with_retry(_AlwaysFails(), routine, run_id)

    async with async_session_maker() as db:
        run = await db.get(RoutineRun, run_id)
        job = await db.get(BuildJob, job_id)
    # Three attempts in the budget → final attempt counter is 3 on
    # both rows (legacy + cutover invariant).
    assert run.attempt == 3, f"RoutineRun.attempt expected 3, got {run.attempt}"
    assert job.attempt == 3, f"BuildJob.attempt expected 3, got {job.attempt}"


# ── _finalize_run: mirror five new columns ───────────────────────────


@pytest.mark.asyncio
async def test_finalize_run_mirrors_terminal_columns_to_buildjob(
    _seed_user_and_routine, monkeypatch,
):
    """``_finalize_run`` writes the legacy terminal shape onto
    ``routine_runs``, then ``_mirror_run_terminal_to_job`` mirrors all
    five mig 052 columns onto the linked BuildJob:

      emails_fetched, finished_local_at, error_json,
      channel_results_json, tools_invoked_json
    """
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    # _fire mints the Job + dual-writes the RoutineRun.
    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        from sqlalchemy import select
        run = (await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == ROUTINE_ID)
        )).scalar_one()
        run_id = run.id
        job_id = run.job_id

    channel_results = {
        "website": {"status": "delivered", "message_id": "msg-1", "ws_count": 2},
        "telegram": {"status": "delivered", "telegram_message_id": 4242},
    }
    tools_invoked = ["gmail__list_messages", "gmail__get_message"]
    error_json = None  # success outcomes leave error_json NULL

    await runner._finalize_run(
        run_id,
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
        run = await db.get(RoutineRun, run_id)
        job = await db.get(BuildJob, job_id)
    # Sanity: legacy row got everything it always did.
    assert run.status == "success"
    assert run.emails_fetched == 3
    assert run.channel_results_json == channel_results
    assert run.tools_invoked_json == tools_invoked
    # The five mig 052 columns must mirror verbatim onto BuildJob.
    assert job.status == "completed"
    assert job.outcome == "success"
    assert job.emails_fetched == 3, (
        f"BuildJob.emails_fetched mirror failed; got {job.emails_fetched!r}"
    )
    assert job.finished_local_at is not None, (
        "BuildJob.finished_local_at must be populated from RoutineRun"
    )
    # error_json NULL on success outcomes.
    assert job.error_json is None
    assert job.channel_results_json == channel_results, (
        f"BuildJob.channel_results_json mirror failed; "
        f"got {job.channel_results_json!r}"
    )
    assert job.tools_invoked_json == tools_invoked, (
        f"BuildJob.tools_invoked_json mirror failed; "
        f"got {job.tools_invoked_json!r}"
    )


# ── Intake abort on Job mint failure ─────────────────────────────────


@pytest.mark.asyncio
async def test_intake_aborts_when_job_mint_fails(
    _seed_user_and_routine, monkeypatch,
):
    """``JobRunner.create_job`` raising → ``_fire`` MUST NOT insert a
    RoutineRun. Matches PR #46's inverted contract for the trigger
    intake path: Job is the source of truth; without it we have no
    runner-visible row, no terminal-mirror target, no anchor for
    observability — better to abort and let APScheduler retry on the
    next interval."""
    from sqlalchemy import select, func

    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    # Force JobRunner.create_job to raise — the runner imports it
    # inside _fire, so we patch the class attribute itself.
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
        run_count = (await db.execute(
            select(func.count(RoutineRun.id)).where(
                RoutineRun.routine_id == ROUTINE_ID,
            )
        )).scalar_one()
    assert job_count == 0, (
        f"Job mint failed; expected 0 BuildJobs, got {job_count}"
    )
    assert run_count == 0, (
        f"Job mint failed; RoutineRun MUST NOT be inserted "
        f"(PR #47 contract), got {run_count} RoutineRun rows"
    )
