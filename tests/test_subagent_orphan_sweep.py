"""Sub-agent orphan sweep — Phase 5 crash-recovery path.

Mirrors ``test_trigger_runner_failure_handler.py`` and the existing
``_restart_sweep`` patterns. Pin:

  - Rows older than the threshold flip from 'running' to 'failed'
    with a clear error_message + outcome='failed'.
  - Rows younger than the threshold are LEFT ALONE.
  - Terminal rows (completed / failed / cancelled / timeout) are
    NEVER overwritten by the sweep.
  - Non-subagent job_type rows (agent_task, routine_run, trigger_run,
    auto_builder) are NEVER touched.
  - The sweep is idempotent — a second call after a clean sweep
    returns 0 and changes nothing.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio


async def _seed_user(db, user_id: str):
    from app.db.models import User
    from sqlalchemy import select
    existing = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if existing is None:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Test",
        ))
        await db.commit()


async def _make_subagent(
    db,
    *,
    user_id: str,
    status: str = "running",
    created_offset_minutes: int = 0,
    job_type: str = "subagent",
) -> str:
    """Insert a row with a fabricated ``created_at`` so we can put
    it on either side of the orphan threshold without waiting."""
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    job = BuildJob(
        id=jid,
        user_id=user_id,
        title="t",
        prompt="p",
        job_type=job_type,
        status=status,
        config_json={"label": f"L-{jid[:6]}"} if job_type == "subagent" else None,
    )
    job.created_at = datetime.utcnow() - timedelta(minutes=created_offset_minutes)
    db.add(job)
    await db.commit()
    return jid


@pytest_asyncio.fixture
async def db():
    from app.db.database import async_session_maker
    async with async_session_maker() as s:
        yield s


@pytest.mark.asyncio
async def test_old_running_row_swept_to_failed(db, monkeypatch):
    from app.agent.subagent_orchestrator import orphan_sweep_on_boot
    from app.db.models import BuildJob
    from app.config import settings
    from sqlalchemy import select

    # Lower threshold to 5m so we can put a 6m-old row "over the line"
    monkeypatch.setattr(settings, "subagent_orphan_sweep_threshold_minutes", 5)

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)
    jid = await _make_subagent(db, user_id=uid, status="running", created_offset_minutes=10)

    swept = await orphan_sweep_on_boot()
    assert swept >= 1

    row = (await db.execute(select(BuildJob).where(BuildJob.id == jid))).scalar_one()
    assert row.status == "failed"
    assert row.outcome == "failed"
    assert "orphaned" in (row.error_message or "")
    assert row.completed_at is not None


@pytest.mark.asyncio
async def test_fresh_running_row_left_alone(db, monkeypatch):
    """A row that hasn't crossed the threshold yet must NOT be
    swept — it's probably still legitimately running."""
    from app.agent.subagent_orchestrator import orphan_sweep_on_boot
    from app.db.models import BuildJob
    from app.config import settings
    from sqlalchemy import select

    monkeypatch.setattr(settings, "subagent_orphan_sweep_threshold_minutes", 10)

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)
    jid = await _make_subagent(db, user_id=uid, status="running", created_offset_minutes=3)

    await orphan_sweep_on_boot()
    row = (await db.execute(select(BuildJob).where(BuildJob.id == jid))).scalar_one()
    assert row.status == "running"  # untouched
    assert row.completed_at is None


@pytest.mark.asyncio
async def test_terminal_rows_never_swept(db, monkeypatch):
    """Old completed/failed/cancelled rows are terminal — the sweep
    must leave them alone even if they predate the threshold."""
    from app.agent.subagent_orchestrator import orphan_sweep_on_boot
    from app.db.models import BuildJob
    from app.config import settings
    from sqlalchemy import select

    monkeypatch.setattr(settings, "subagent_orphan_sweep_threshold_minutes", 1)

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)
    old_completed = await _make_subagent(
        db, user_id=uid, status="completed", created_offset_minutes=60,
    )
    old_failed = await _make_subagent(
        db, user_id=uid, status="failed", created_offset_minutes=60,
    )

    await orphan_sweep_on_boot()
    rows = {
        r.id: r for r in (await db.execute(
            select(BuildJob).where(BuildJob.id.in_([old_completed, old_failed]))
        )).scalars().all()
    }
    assert rows[old_completed].status == "completed"
    assert rows[old_failed].status == "failed"
    assert rows[old_failed].error_message is None  # was None when inserted


@pytest.mark.asyncio
async def test_non_subagent_job_type_not_touched(db, monkeypatch):
    """The sweep is type-scoped to subagent — other job types use
    their own runners' sweeps (TriggerRunner, RoutineRunner)."""
    from app.agent.subagent_orchestrator import orphan_sweep_on_boot
    from app.db.models import BuildJob
    from app.config import settings
    from sqlalchemy import select

    monkeypatch.setattr(settings, "subagent_orphan_sweep_threshold_minutes", 1)

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)
    other = await _make_subagent(
        db, user_id=uid, status="running",
        created_offset_minutes=60, job_type="agent_task",
    )
    trigger = await _make_subagent(
        db, user_id=uid, status="running",
        created_offset_minutes=60, job_type="trigger_run",
    )

    await orphan_sweep_on_boot()
    rows = {
        r.id: r for r in (await db.execute(
            select(BuildJob).where(BuildJob.id.in_([other, trigger]))
        )).scalars().all()
    }
    assert rows[other].status == "running"
    assert rows[trigger].status == "running"


@pytest.mark.asyncio
async def test_idempotent_second_call_is_zero(db, monkeypatch):
    """After a clean sweep, a second sweep on the same DB should
    find nothing to do."""
    from app.agent.subagent_orchestrator import orphan_sweep_on_boot
    from app.config import settings

    monkeypatch.setattr(settings, "subagent_orphan_sweep_threshold_minutes", 1)

    uid = str(uuid.uuid4())
    await _seed_user(db, uid)
    await _make_subagent(db, user_id=uid, status="running", created_offset_minutes=10)

    first = await orphan_sweep_on_boot()
    second = await orphan_sweep_on_boot()
    assert first >= 1
    assert second == 0


@pytest.mark.asyncio
async def test_zero_count_when_no_rows(db, monkeypatch):
    from app.agent.subagent_orchestrator import orphan_sweep_on_boot
    from app.config import settings

    monkeypatch.setattr(settings, "subagent_orphan_sweep_threshold_minutes", 1)
    swept = await orphan_sweep_on_boot()
    assert swept == 0
