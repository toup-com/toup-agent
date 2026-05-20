"""Job parity — routine fires materialise BuildJob rows.

Originally PR 4b of the unified-jobs arc, this suite pinned a
dual-write between ``routine_runs`` (legacy) and ``build_jobs``
(new). PR #49 of the cutover arc removed the legacy dual-write —
``build_jobs`` is now the sole source of truth.

What we pin:

  1. ``RoutineRunner._fire`` mints ONE BuildJob row per
     (routine, local_date) pair. ``source_id`` points at the
     routine; ``idempotency_key=str(local_date)``;
     ``job_type='routine_run'``.

  2. Each terminal status maps onto the Job via the inline
     ``_finalize_run`` write:
        success            → status='completed', outcome=<derived>
        partial            → status='completed', outcome='partial'
        skipped_reauth     → status='completed', outcome='skipped_reauth'
        failed             → status='failed',    outcome=NULL

  3. The Ticket-2.1 ``derived outcome`` (success / success_empty /
     partial / tool_error / failure) survives on ``BuildJob.outcome``.

  4. A Job mint failure at fire-time aborts the intake — no
     downstream row exists.
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
    breaks SQLite). PR #49: cutover paths no longer touch RoutineRun,
    so the fixture only sets up User, Routine, BuildJob, JobEvent."""
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


# ── _fire mint ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fire_creates_job_after_idempotency_claim(
    _seed_user_and_routine, monkeypatch,
):
    """A successful ``_fire`` writes ONE BuildJob row.
    The Job carries source_id=routine.id,
    idempotency_key=str(local_date), job_type='routine_run'."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select

    runner = RoutineRunner(session_maker=async_session_maker)

    # Stub the user-tz resolver so we don't hit the agent_config row.
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    # No registered handler → _fire takes the "_finalize_run success"
    # short-circuit. That's fine for this parity test — we're asserting
    # the Job mint, not handler dispatch.
    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        jobs = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == ROUTINE_ID)
        )).scalars().all()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.job_type == "routine_run"
        assert job.source_kind == "routine"
        assert job.source_id == ROUTINE_ID
        # idempotency_key is str(local_date); we can't pin the exact
        # date here (depends on wall-clock), but the field must be set.
        assert job.idempotency_key is not None
        assert job.user_id == USER_ID


@pytest.mark.asyncio
async def test_two_fires_same_day_produce_one_job(
    _seed_user_and_routine, monkeypatch,
):
    """Two ``_fire`` calls for the same local-date hit the explicit
    SELECT dedupe gate on the second call. Exactly ONE Job exists."""
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
            f"expected 1 Job for the dedupe pair, got {job_count}"
        )


# ── _finalize_run terminal writes ────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_run_success_writes_to_job(
    _seed_user_and_routine, monkeypatch,
):
    """When ``_finalize_run`` writes a success outcome, the Job is
    updated to status='completed', outcome=<the derived Ticket-2.1
    outcome value>, completed_at set, summary_message_id set."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)

    async with async_session_maker() as db:
        from sqlalchemy import select
        job = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == ROUTINE_ID)
        )).scalar_one()
        job_id = job.id

    # Drive _finalize_run directly with a synthesised terminal write.
    await runner._finalize_run(
        job_id,
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
async def test_finalize_run_failed_writes_to_job(
    _seed_user_and_routine, monkeypatch,
):
    """``status='failed'`` writes ``Job.status='failed'`` with
    ``error_message`` set and ``outcome=NULL``."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)
    async with async_session_maker() as db:
        job = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == ROUTINE_ID)
        )).scalar_one()
        job_id = job.id

    await runner._finalize_run(
        job_id,
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
async def test_finalize_run_skipped_reauth_writes_to_job(
    _seed_user_and_routine, monkeypatch,
):
    """``status='skipped_reauth'`` writes ``Job.status='completed'``
    + ``outcome='tool_error'`` (derived outcome takes precedence)."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select

    runner = RoutineRunner(session_maker=async_session_maker)
    async def _tz(uid):
        return "UTC"
    monkeypatch.setattr(runner, "_user_tz_async", _tz)

    await runner._fire(ROUTINE_ID)
    async with async_session_maker() as db:
        job = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == ROUTINE_ID)
        )).scalar_one()
        job_id = job.id

    await runner._finalize_run(
        job_id,
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
    """Contract from PR #47 (preserved through PR #49): a failed
    ``JobRunner.create_job`` aborts the fire path entirely — no
    BuildJob row exists, APScheduler's next tick is the natural
    retry."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
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

    async with async_session_maker() as db:
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == ROUTINE_ID,
            )
        )).scalar_one()
        assert job_count == 0
