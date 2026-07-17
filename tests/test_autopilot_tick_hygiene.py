"""Autopilot ticks are engine internals, not user tasks.

Each ~5-min mission heartbeat mints a BuildJob (full job lifecycle for
retry/observability), but task surfaces must not show them — the
MISSION is the user-visible unit. Founder bug 2026-07-16: anonymous
'Routine fire: autopilot <date>' rows in Agent Tasks while the mission
itself appeared nowhere.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient


async def _mk_user() -> str:
    from app.db import User, async_session_maker
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"tick-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12), name="Tick Test",
        ))
        await db.commit()
    return user_id


async def _mk_job(user_id: str, *, job_type: str, title: str) -> str:
    from app.db import async_session_maker
    from app.db.models import BuildJob

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=user_id, title=title, prompt="p",
            job_type=job_type, status="completed",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return job_id


@pytest.mark.asyncio
async def test_jobs_listing_hides_autopilot_ticks(monkeypatch):
    from fastapi import FastAPI
    from app.config import settings
    from app.api.apps import router as apps_router

    user_id = await _mk_user()
    monkeypatch.setattr(settings, "user_id", user_id)

    task_id = await _mk_job(
        user_id, job_type="agent_task", title="Top 10 CRM comparison",
    )
    tick_id = await _mk_job(
        user_id, job_type="autopilot_tick",
        title="Autopilot: UI/UX design tools research — tick 3",
    )

    app = FastAPI()
    app.include_router(apps_router, prefix="/api")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        default = await ac.get("/api/apps/jobs/")
        assert default.status_code == 200, default.text
        ids = [j["id"] for j in default.json()]
        assert task_id in ids
        assert tick_id not in ids, "ticks must be hidden by default"

        debug = await ac.get("/api/apps/jobs/?include_ticks=true")
        ids = [j["id"] for j in debug.json()]
        assert task_id in ids and tick_id in ids


# ──────────────────────────────────────────────────────────────────────
# Backfill: legacy tick rows minted before the autopilot_tick job_type
# existed (job_type='routine_run', title='Routine fire: autopilot …')
# must flip to 'autopilot_tick' so they disappear from task surfaces.
# The statements live in init_db._alter_statements (agent boot);
# SQLite ≥3.33 supports UPDATE…FROM so we execute them directly here.
# ──────────────────────────────────────────────────────────────────────

_BACKFILL_STATEMENTS = [
    "UPDATE build_jobs SET job_type = 'autopilot_tick' "
    "FROM routines r WHERE build_jobs.source_id = r.id "
    "AND r.kind = 'autopilot' AND build_jobs.job_type = 'routine_run'",
    "UPDATE build_jobs SET job_type = 'autopilot_tick' "
    "WHERE job_type = 'routine_run' "
    "AND title LIKE 'Routine fire: autopilot %'",
]


def test_backfill_statements_are_registered_in_init_db():
    """Pin that the exact statements ship in database.py so the boot
    path (init_db._alter_statements) actually runs them per tenant."""
    import inspect
    from app.db import database as db_mod

    src = inspect.getsource(db_mod)
    assert "AND r.kind = 'autopilot' AND build_jobs.job_type = 'routine_run'" in src
    assert "AND title LIKE 'Routine fire: autopilot %'" in src


@pytest.mark.asyncio
async def test_backfill_flips_legacy_autopilot_ticks_only():
    from sqlalchemy import text
    from app.db import async_session_maker
    from app.db.models import BuildJob, Routine

    user_id = await _mk_user()

    routine_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Routine(
            id=routine_id, user_id=user_id, kind="autopilot",
            name="UI/UX design tools research", enabled=False,
            schedule_cron_local="* * * * *", schedule_kind="every",
            schedule_interval_seconds=300,
        ))
        await db.commit()

    async def _mk_legacy(title: str, source_id: str | None) -> str:
        job_id = str(uuid.uuid4())
        async with async_session_maker() as db:
            db.add(BuildJob(
                id=job_id, user_id=user_id, title=title, prompt="p",
                job_type="routine_run", status="completed",
                source_kind="routine", source_id=source_id,
                created_at=datetime.utcnow(),
            ))
            await db.commit()
        return job_id

    # Joined to a live autopilot routine → flipped by statement 1.
    tick_joined = await _mk_legacy("Routine fire: autopilot 2026-07-16", routine_id)
    # Parent routine deleted → caught by the LIKE fallback.
    tick_orphan = await _mk_legacy("Routine fire: autopilot 2026-07-15", str(uuid.uuid4()))
    # A genuine recurring routine fire must NOT flip.
    control = await _mk_legacy("Routine fire: email_briefing 2026-07-16", None)

    async def _run_backfill():
        async with async_session_maker() as db:
            for stmt in _BACKFILL_STATEMENTS:
                await db.execute(text(stmt))
            await db.commit()

    await _run_backfill()
    await _run_backfill()  # idempotency: second run is a no-op

    async with async_session_maker() as db:
        rows = {
            j.id: j.job_type
            for j in (await db.execute(
                __import__("sqlalchemy").select(BuildJob).where(BuildJob.user_id == user_id)
            )).scalars()
        }
    assert rows[tick_joined] == "autopilot_tick"
    assert rows[tick_orphan] == "autopilot_tick"
    assert rows[control] == "routine_run"
