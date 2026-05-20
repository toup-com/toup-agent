"""Job parity — routine fires must materialise mirrored BuildJob rows.

PR 4b of the unified-jobs arc. Sibling of test_job_parity_triggers.py.

What we pin:

  1. ``RoutineRunner._fire`` mints a mirrored BuildJob immediately
     after the RoutineRun INSERT, in the same transaction lifecycle.
     The RoutineRun's ``job_id`` is stamped; the Job's ``source_id``
     points back at the routine; ``idempotency_key`` is the user-
     local-date discriminator so the composite UNIQUE on
     ``(source_id, idempotency_key)`` collapses to the same dedupe
     gate as ``UNIQUE (routine_id, scheduled_for_local_date)``.

  2. Each RoutineRun terminal status mirrors onto the linked Job
     via ``_finalize_run`` → ``_mirror_run_terminal_to_job``:
        success            → status='completed', outcome=<derived>
        partial            → status='completed', outcome='partial'
        skipped_reauth     → status='completed', outcome='skipped_reauth'
        failed             → status='failed',    outcome=NULL

  3. The Ticket-2.1 ``derived outcome`` (success / success_empty /
     partial / tool_error / failure) survives on ``BuildJob.outcome``
     so Mission Control can distinguish e.g. "no new emails today"
     from "real summary delivered."

  4. A Job mint failure at fire-time must NOT roll back the
     RoutineRun insert (best-effort dual-write).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, date as date_cls
from typing import Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-job-parity-routines")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000cc1")


USER_ID = "00000000-0000-0000-0000-000000000cc1"
ROUTINE_ID = "00000000-0000-0000-0000-0000000cc100"


# ── Schema setup ──────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Bypass conftest's init_db autouse fixture (entities pgvector
    breaks SQLite). Build only the tables we need."""
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
                email=f"parity-routine-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
        db.add(Routine(
            id=ROUTINE_ID,
            user_id=USER_ID,
            kind="email_briefing",
            enabled=True,
            schedule_cron_local="0 7 * * *",
            schedule_kind="cron",
            config_json={},
            last_state_json={},
            last_status="never_run",
        ))
        await db.commit()


# ── _fire dual-write ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fire_creates_mirrored_job_after_idempotency_claim(
    _seed_user_and_routine, monkeypatch,
):
    """A successful ``_fire`` writes a RoutineRun AND a mirrored
    BuildJob. The Job carries source_id=routine.id,
    idempotency_key=str(local_date), job_type='routine_run'.
    RoutineRun.job_id links back."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun
    from sqlalchemy import select

    runner = RoutineRunner(session_maker=async_session_maker)

    # Stub the user-tz resolver so we don't hit the agent_config row.
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    # No registered handler → _fire takes the "_finalize_run success"
    # short-circuit. That's fine for this parity test — we're asserting
    # the create-side dual-write, not handler dispatch.
    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        runs = (await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == ROUTINE_ID)
        )).scalars().all()
        assert len(runs) == 1
        run = runs[0]
        assert run.job_id is not None, "RoutineRun.job_id must be stamped"

        job = await db.get(BuildJob, run.job_id)
        assert job is not None
        assert job.job_type == "routine_run"
        assert job.source_kind == "routine"
        assert job.source_id == ROUTINE_ID
        assert job.idempotency_key == str(run.scheduled_for_local_date)
        assert job.user_id == USER_ID


@pytest.mark.asyncio
async def test_two_fires_same_day_produce_one_job(
    _seed_user_and_routine, monkeypatch,
):
    """Two ``_fire`` calls for the same local-date hit the
    ``UNIQUE (routine_id, scheduled_for_local_date)`` constraint on
    the second call. Exactly ONE Job exists. The Job's
    idempotency_key prevents a duplicate at the unified layer too."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select, func

    runner = RoutineRunner(session_maker=async_session_maker)

    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)
    await runner._fire(ROUTINE_ID)  # idempotency_collision

    async with async_session_maker() as db:
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == ROUTINE_ID,
                BuildJob.source_kind == "routine",
            )
        )).scalar_one()
        assert job_count == 1, (
            f"expected 1 mirrored Job for the dedupe pair, got {job_count}"
        )


# ── _finalize_run mirror ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_run_success_mirrors_to_job(
    _seed_user_and_routine, monkeypatch,
):
    """When ``_finalize_run`` writes a success outcome, the mirror
    helper updates the linked Job: status='completed',
    outcome=<the derived Ticket-2.1 outcome value>, completed_at set,
    summary_message_id mirrored."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        from sqlalchemy import select
        run = (await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == ROUTINE_ID)
        )).scalar_one()
        run_id = run.id
        job_id = run.job_id

    # Drive _finalize_run directly with a synthesised terminal write.
    await runner._finalize_run(
        run_id,
        status="success",
        outcome="success",
        emails_fetched=3,
        summary_message_id="msg-abc",
    )

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "completed"
        assert job.outcome == "success"
        assert job.summary_message_id == "msg-abc"
        assert job.completed_at is not None


@pytest.mark.asyncio
async def test_finalize_run_failed_mirrors_to_job(
    _seed_user_and_routine, monkeypatch,
):
    """``status='failed'`` mirrors as ``Job.status='failed'`` with
    ``error_message`` set and ``outcome=NULL``."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun
    from sqlalchemy import select

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)
    async with async_session_maker() as db:
        run = (await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == ROUTINE_ID)
        )).scalar_one()
        run_id = run.id
        job_id = run.job_id

    await runner._finalize_run(
        run_id,
        status="failed",
        error_class="ProviderTimeout",
        error_detail="upstream gmail.googleapis.com timed out after 30s",
    )

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "failed"
        assert job.outcome is None, (
            f"failed routines must leave Job.outcome NULL; got {job.outcome!r}"
        )
        assert job.error_message and "upstream gmail" in job.error_message


@pytest.mark.asyncio
async def test_finalize_run_skipped_reauth_mirrors_to_job(
    _seed_user_and_routine, monkeypatch,
):
    """``status='skipped_reauth'`` mirrors as ``Job.status='completed'``
    + ``outcome='skipped_reauth'`` (closed BuildJob enum doesn't
    include skipped_reauth; granular state lives on outcome)."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun
    from sqlalchemy import select

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)
    async with async_session_maker() as db:
        run = (await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == ROUTINE_ID)
        )).scalar_one()
        run_id = run.id
        job_id = run.job_id

    await runner._finalize_run(
        run_id,
        status="skipped_reauth",
        outcome="tool_error",
        error_class="GmailNeedsReauth",
        error_detail="user revoked OAuth scope",
    )

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "completed"
        # Derived outcome 'tool_error' takes precedence over raw status.
        assert job.outcome == "tool_error"


@pytest.mark.asyncio
async def test_intake_aborts_when_job_mint_fails(
    _seed_user_and_routine, monkeypatch,
):
    """Contract changed in PR #47: with the Job-first inversion, a
    failed ``JobRunner.create_job`` aborts the fire path entirely —
    no legacy RoutineRun row is created, no BuildJob row exists,
    APScheduler's next tick is the natural retry.

    The previous test name (``test_dual_write_robust_to_job_failure``)
    asserted the legacy fire path survived a Job mint failure. That
    contract is intentionally inverted: ``build_jobs`` is now the
    source of truth and the dedupe gate, so we must NOT create an
    orphan RoutineRun without a Job to mirror against."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, RoutineRun
    from sqlalchemy import select, func

    async def _explode(*a, **kw):
        raise RuntimeError("simulated JobRunner.create_job crash")

    import app.agent.job_runner as jr
    monkeypatch.setattr(jr.JobRunner, "create_job", _explode)

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    # _fire must NOT raise — best-effort wrapper logs and bails.
    await runner._fire(ROUTINE_ID)

    # Neither table got a row — APScheduler retries on next tick.
    async with async_session_maker() as db:
        run_count = (await db.execute(
            select(func.count(RoutineRun.id)).where(
                RoutineRun.routine_id == ROUTINE_ID,
            )
        )).scalar_one()
        assert run_count == 0, (
            "Job mint failure must NOT leave an orphan RoutineRun — "
            "Job is the source of truth, RoutineRun is the mirror"
        )
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == ROUTINE_ID,
            )
        )).scalar_one()
        assert job_count == 0
