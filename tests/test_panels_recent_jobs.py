"""Triggers + Routines panels surface their mirrored Job rows.

PR 7 of the unified-jobs arc.

What we pin:

  1. ``GET /api/triggers`` augments each Trigger row with a
     ``recent_jobs`` field — the last 5 ``build_jobs`` rows where
     ``source_kind='trigger' AND source_id=trigger.id``. PR 4a's
     dual-write produces these rows; this PR exposes them.

  2. ``GET /api/routines`` does the same for routines:
     ``source_kind='routine' AND source_id=routine.id``. PR 4b's
     dual-write produces these.

  3. The compact projection (``TriggerRecentJob`` /
     ``RoutineRecentJob``) carries id + title + status + outcome
     + created_at + completed_at — enough for the dashboard card
     to render a "Last fires" list with link-through to the Job
     detail drawer.

  4. Tenants pre-dating PR 4a/4b (no mirrored Jobs) get
     ``recent_jobs=[]`` — the card hides the section gracefully.

These tests invoke the route functions directly to dodge the
pytest-asyncio + StaticPool + ``:memory:`` SQLite quirk
documented in test_jobs_events_endpoint.py.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-panels-recent-jobs")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000abe01")


USER_ID = "00000000-0000-0000-0000-0000000abe01"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Bypass conftest's init_db autouse fixture. Build only the
    tables these tests touch. ``rebind_database`` is called first
    so the engine is bound to the current event loop (see
    test_jobs_events_endpoint.py for the full story)."""
    from app.config import settings
    from app.db.database import rebind_database

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import (
        BuildJob, JobEvent, Routine, RoutineRun, Trigger, TriggerEvent, User,
    )

    async with engine.begin() as conn:
        for model_cls in (
            User, BuildJob, JobEvent,
            Trigger, TriggerEvent,
            Routine, RoutineRun,
        ):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (
            JobEvent, BuildJob,
            TriggerEvent, Trigger,
            RoutineRun, Routine,
            User,
        ):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


@pytest_asyncio.fixture
async def _seed_user():
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"panels-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
            await db.commit()


async def _make_trigger(name: str = "Test trigger") -> str:
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Trigger(
            id=tid,
            user_id=USER_ID,
            kind="email_received",
            action="summarize_and_post",
            name=name,
            enabled=True,
            filter_json={},
            config_json={},
            provider_state_json={},
            last_status="never_fired",
            fire_count=0,
        ))
        await db.commit()
    return tid


async def _make_routine(name: str = "Test routine") -> str:
    from app.db.database import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Routine(
            id=rid,
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
    return rid


async def _add_mirrored_job(
    source_id: str,
    *,
    source_kind: str,
    title: str,
    status: str = "completed",
    outcome: str | None = "success",
    created_at_offset_minutes: int = 0,
) -> str:
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    now = datetime.utcnow() + timedelta(minutes=created_at_offset_minutes)
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=jid,
            user_id=USER_ID,
            title=title,
            prompt="",
            job_type=("trigger_run" if source_kind == "trigger" else "routine_run"),
            status=status,
            outcome=outcome,
            source_kind=source_kind,
            source_id=source_id,
            created_at=now,
            completed_at=now if status in ("completed", "failed") else None,
        ))
        await db.commit()
    return jid


# ──────────────────────────────────────────────────────────────────────
# Triggers
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_triggers_surfaces_recent_jobs(_seed_user):
    """``GET /api/triggers`` should return each Trigger row with a
    ``recent_jobs`` array of up to 5 mirrored Jobs, newest first."""
    from app.api.triggers import list_triggers

    tid = await _make_trigger(name="Watch inbox")
    # Seed 6 jobs so we can verify the limit of 5.
    job_ids = []
    for i in range(6):
        jid = await _add_mirrored_job(
            tid,
            source_kind="trigger",
            title=f"Trigger fire {i}",
            created_at_offset_minutes=i,
        )
        job_ids.append(jid)

    triggers = await list_triggers()
    assert len(triggers) == 1
    t = triggers[0]
    assert t.id == tid
    assert len(t.recent_jobs) == 5, (
        f"expected 5 most-recent jobs, got {len(t.recent_jobs)}"
    )
    # Newest first by created_at — the index 5 (the latest one)
    # leads, the index 1 trails. Index 0 is dropped (limit hit).
    assert t.recent_jobs[0].title == "Trigger fire 5"
    assert t.recent_jobs[-1].title == "Trigger fire 1"
    # Compact projection carries status + outcome.
    assert t.recent_jobs[0].status == "completed"
    assert t.recent_jobs[0].outcome == "success"


@pytest.mark.asyncio
async def test_list_triggers_no_fires_yet_returns_empty_recent_jobs(_seed_user):
    """A freshly-created Trigger with zero fires should return
    ``recent_jobs=[]`` — the dashboard card hides the section
    gracefully, no JOIN explosion."""
    from app.api.triggers import list_triggers

    await _make_trigger(name="Brand new")

    triggers = await list_triggers()
    assert len(triggers) == 1
    assert triggers[0].recent_jobs == []


@pytest.mark.asyncio
async def test_list_triggers_isolates_recent_jobs_per_trigger(_seed_user):
    """Two triggers, each with their own Jobs. Each row's
    ``recent_jobs`` must contain only THIS trigger's mirrored
    Jobs — no cross-contamination from sibling triggers."""
    from app.api.triggers import list_triggers

    t1 = await _make_trigger(name="Trigger A")
    t2 = await _make_trigger(name="Trigger B")
    await _add_mirrored_job(t1, source_kind="trigger", title="A-fire-1")
    await _add_mirrored_job(t1, source_kind="trigger", title="A-fire-2")
    await _add_mirrored_job(t2, source_kind="trigger", title="B-fire-1")

    triggers = await list_triggers()
    by_name = {t.name: t for t in triggers}
    a, b = by_name["Trigger A"], by_name["Trigger B"]
    a_titles = {j.title for j in a.recent_jobs}
    b_titles = {j.title for j in b.recent_jobs}
    assert a_titles == {"A-fire-1", "A-fire-2"}
    assert b_titles == {"B-fire-1"}


# ──────────────────────────────────────────────────────────────────────
# Routines
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_routines_surfaces_recent_jobs(_seed_user):
    """``GET /api/routines`` mirrors the trigger surface: up to
    5 mirrored Jobs per routine, newest first."""
    from app.api.routines import list_routines

    rid = await _make_routine(name="Morning briefing")
    for i in range(6):
        await _add_mirrored_job(
            rid,
            source_kind="routine",
            title=f"Briefing {i}",
            created_at_offset_minutes=i,
        )

    routines = await list_routines()
    assert len(routines) == 1
    r = routines[0]
    assert len(r.recent_jobs) == 5
    assert r.recent_jobs[0].title == "Briefing 5"
    assert r.recent_jobs[-1].title == "Briefing 1"


@pytest.mark.asyncio
async def test_list_routines_no_fires_yet_returns_empty_recent_jobs(_seed_user):
    """A routine with no fires returns ``recent_jobs=[]``."""
    from app.api.routines import list_routines

    await _make_routine(name="Brand new")
    routines = await list_routines()
    assert len(routines) == 1
    assert routines[0].recent_jobs == []


@pytest.mark.asyncio
async def test_list_routines_isolates_recent_jobs_per_routine(_seed_user):
    """Per-routine isolation, same property as the trigger test."""
    from app.api.routines import list_routines

    r1 = await _make_routine(name="Routine A")
    r2 = await _make_routine(name="Routine B")
    await _add_mirrored_job(r1, source_kind="routine", title="A-run-1")
    await _add_mirrored_job(r2, source_kind="routine", title="B-run-1")

    routines = await list_routines()
    by_name = {r.name: r for r in routines}
    # Note: routine.name comes from the seed; both have None unless
    # we set it explicitly. Re-look up by ID instead.
    by_id = {r.id: r for r in routines}
    a_titles = {j.title for j in by_id[r1].recent_jobs}
    b_titles = {j.title for j in by_id[r2].recent_jobs}
    assert a_titles == {"A-run-1"}
    assert b_titles == {"B-run-1"}
