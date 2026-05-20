"""Unit tests for ``backend/scripts/check_job_parity.py``.

The script is the operator's pre-flight gate before flipping
``ALLOW_LEGACY_JOB_TABLES_DROP=true`` (migration 050). If the script
ever falsely reports "parity-green" while orphans exist, the
operator would authorise a destructive drop on inconsistent data.
That's the exact failure mode worth pinning hard against.

What we pin:

  1. **Green path** — every TriggerEvent and RoutineRun row has a
     populated ``job_id``; the script's verdict is
     ``parity-green-safe-to-drop`` and exit code is 0.

  2. **Trigger gap** — a TriggerEvent with ``job_id=NULL`` produces
     ``parity-gap-N-orphan-rows`` and exit code 1.

  3. **Routine gap** — same but on the routine side.

  4. **Schema-not-ready** — when required tables don't exist
     (e.g. agent on a pre-PR-2 branch), exit code is 2 and the
     verdict is ``schema-not-ready``.

  5. **Already-dropped** — when both legacy tables are absent, the
     script reports ``verdict=already-dropped`` and exit code 0 —
     the operator can re-run safely without confusion.

These tests do NOT shell out — they import ``_amain`` and call it.
Faster and gives us the live exit code via the return value, not a
subprocess exit interpretation.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-parity")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000prt01")


USER_ID = "00000000-0000-0000-0000-0000000prt01"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    from app.db.database import rebind_database
    from app.config import settings

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import (
        BuildJob, JobEvent, Routine, RoutineRun, Trigger,
        TriggerEvent, User,
    )

    async with engine.begin() as conn:
        for model_cls in (
            User, BuildJob, JobEvent, Trigger, TriggerEvent,
            Routine, RoutineRun,
        ):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (
            RoutineRun, Routine, TriggerEvent, Trigger,
            JobEvent, BuildJob, User,
        ):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


async def _seed_user():
    from app.db.database import async_session_maker
    from app.db.models import User
    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"parity-{USER_ID[:8]}@example.com",
                hashed_password="x", timezone="UTC",
            ))
            await db.commit()


async def _seed_trigger_event(job_id: str | None) -> str:
    """Insert a TriggerEvent with the given job_id (None or a uuid)."""
    from app.db.database import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    trigger_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Trigger(
            id=trigger_id, user_id=USER_ID, kind="email_received",
            action="agent_handle", enabled=True, config_json={},
        ))
        db.add(TriggerEvent(
            id=event_id, trigger_id=trigger_id, user_id=USER_ID,
            event_dedupe_id=f"dedupe-{event_id[:8]}",
            received_at=datetime.utcnow(),
            status="success",
            job_id=job_id,
        ))
        await db.commit()
    return event_id


async def _seed_routine_run(job_id: str | None) -> str:
    from app.db.database import async_session_maker
    from app.db.models import Routine, RoutineRun

    routine_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Routine(
            id=routine_id, user_id=USER_ID, kind="email_briefing",
            enabled=True, config_json={},
            schedule_cron_local="0 8 * * *",
        ))
        db.add(RoutineRun(
            id=run_id, routine_id=routine_id, user_id=USER_ID,
            scheduled_for_local_date=date.today(),
            started_at=datetime.utcnow(),
            status="success",
            job_id=job_id,
        ))
        await db.commit()
    return run_id


async def _seed_build_job(with_event: bool = False) -> str:
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, JobEvent

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=USER_ID, title="parity test",
            prompt="", job_type="trigger_run", status="completed",
            source_kind="trigger", source_id=job_id,
            created_at=datetime.utcnow(),
        ))
        if with_event:
            db.add(JobEvent(
                id=str(uuid.uuid4()),
                job_id=job_id, user_id=USER_ID,
                ts=datetime.utcnow(),
                kind="info", label="event", level="info",
            ))
        await db.commit()
    return job_id


async def _arun() -> tuple[int, str]:
    """Async helper — invoke ``_amain()`` and capture its print()
    output. We load the script via ``importlib`` because the
    ``backend/scripts`` directory has no ``__init__.py`` (operator
    scripts run directly, not via package import) — so a plain
    ``from scripts.check_job_parity import ...`` may resolve as
    a namespace package on some Python configs and fail on others.
    importlib.util.spec_from_file_location is the explicit form."""
    import io
    import contextlib
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "check_job_parity.py"
    spec = importlib.util.spec_from_file_location(
        "_parity_script_under_test", script_path,
    )
    assert spec is not None and spec.loader is not None, (
        f"could not load parity script at {script_path}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exit_code = await module._amain()
    return exit_code, buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# 1. Green path: every legacy row has a job_id.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_green_path_emits_safe_to_drop_verdict():
    """All legacy rows linked to a BuildJob → verdict
    'parity-green-safe-to-drop' and exit code 0."""
    await _seed_user()
    bj = await _seed_build_job(with_event=True)
    await _seed_trigger_event(job_id=bj)
    await _seed_routine_run(job_id=bj)

    exit_code, output = await _arun()
    assert exit_code == 0, (
        f"green path should exit 0; got {exit_code}.\nOutput:\n{output}"
    )
    assert "verdict=parity-green-safe-to-drop" in output, (
        f"expected green verdict; got output:\n{output}"
    )
    # Sanity-check the counts.
    assert "trigger_events_orphans=0" in output
    assert "routine_runs_orphans=0" in output


# ──────────────────────────────────────────────────────────────────────
# 2. Trigger gap surfaces.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_orphan_produces_gap_verdict():
    """A TriggerEvent with job_id=NULL produces a parity-gap verdict
    and exit code 1."""
    await _seed_user()
    bj = await _seed_build_job()
    # One linked, two orphans.
    await _seed_trigger_event(job_id=bj)
    await _seed_trigger_event(job_id=None)
    await _seed_trigger_event(job_id=None)
    # No routine rows.

    exit_code, output = await _arun()
    assert exit_code == 1, (
        f"trigger orphan should exit 1; got {exit_code}.\nOutput:\n{output}"
    )
    assert "trigger_events_orphans=2" in output, (
        f"orphan count should be 2; got:\n{output}"
    )
    assert "verdict=parity-gap-" in output


# ──────────────────────────────────────────────────────────────────────
# 3. Routine gap surfaces independently.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_routine_orphan_produces_gap_verdict():
    """Same as trigger but on the routine side. Pin separately so a
    regression that only counts one side surfaces."""
    await _seed_user()
    bj = await _seed_build_job()
    # No trigger rows.
    await _seed_routine_run(job_id=bj)
    await _seed_routine_run(job_id=None)

    exit_code, output = await _arun()
    assert exit_code == 1
    assert "routine_runs_orphans=1" in output
    assert "verdict=parity-gap-" in output


# ──────────────────────────────────────────────────────────────────────
# 4. Already-dropped path — both legacy tables absent.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_already_dropped_returns_zero():
    """If migration 050 has already run, trigger_events and
    routine_runs are gone. The script must report
    ``verdict=already-dropped`` and exit 0 — operators can re-run
    safely. Drop the tables manually to simulate."""
    await _seed_user()
    await _seed_build_job()
    # Drop the two legacy tables AFTER autouse created them.
    from app.db.database import engine
    from app.db.models import RoutineRun, Routine, TriggerEvent, Trigger
    async with engine.begin() as conn:
        for m in (RoutineRun, Routine, TriggerEvent, Trigger):
            await conn.run_sync(m.__table__.drop, checkfirst=True)

    exit_code, output = await _arun()
    assert exit_code == 0, (
        f"already-dropped should exit 0; got {exit_code}.\nOutput:\n{output}"
    )
    assert "verdict=already-dropped" in output


# ──────────────────────────────────────────────────────────────────────
# 5. Schema-not-ready — build_jobs / job_events missing.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_schema_not_ready_returns_two():
    """Pre-unified-arc DB has no ``job_events`` table. The script
    must exit 2 (not 0 or 1) so the operator's deploy script can
    branch on "agent not on the right branch yet" vs "agent on the
    right branch but has gaps"."""
    await _seed_user()
    # Drop the unified-arc table required by the script.
    from app.db.database import engine
    from app.db.models import JobEvent
    async with engine.begin() as conn:
        await conn.run_sync(JobEvent.__table__.drop, checkfirst=True)

    exit_code, output = await _arun()
    assert exit_code == 2, (
        f"schema-not-ready should exit 2; got {exit_code}.\nOutput:\n{output}"
    )
    assert "verdict=schema-not-ready" in output


# ──────────────────────────────────────────────────────────────────────
# 6. Mixed gaps sum to the right total.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mixed_gaps_sum_correctly():
    """3 trigger orphans + 2 routine orphans = exit 1 with
    ``verdict=parity-gap-5-orphan-rows``. Pin the counting math so a
    regression that double-counts or drops one side surfaces."""
    await _seed_user()
    bj = await _seed_build_job()
    for _ in range(3):
        await _seed_trigger_event(job_id=None)
    await _seed_trigger_event(job_id=bj)
    for _ in range(2):
        await _seed_routine_run(job_id=None)
    await _seed_routine_run(job_id=bj)

    exit_code, output = await _arun()
    assert exit_code == 1
    assert "trigger_events_orphans=3" in output
    assert "routine_runs_orphans=2" in output
    assert "verdict=parity-gap-5-orphan-rows" in output, (
        f"3+2 orphans should be summed; got:\n{output}"
    )
