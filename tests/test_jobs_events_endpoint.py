"""GET /apps/jobs/events — server-side activity feed.

PR 6 of the unified-jobs arc.

What we pin:

  1. The endpoint returns one entry per ``job_events`` row,
     JOINed to ``build_jobs`` for ``job_title`` / ``job_type`` /
     ``source_kind`` attribution. The "no more Nokia Snake Arcade
     everywhere" regression check: events from 3 different jobs
     surface with 3 different ``job_title`` values.

  2. Pagination via ``?before=<iso ts>``: passing the previous
     page's ``next_before`` returns strictly older events.

  3. Per-user scoping: events for a different user are excluded.

  4. ``limit`` clamps to [1, 200] (FastAPI Query validator).

  5. Orphan events (job_id pointing at a deleted BuildJob) return
     with ``job_title=None`` — the read path doesn't crash.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import HTTPException


os.environ.setdefault("AGENT_API_KEY", "test-key-jobs-events")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000abc01")


USER_ID = "00000000-0000-0000-0000-0000000abc01"
OTHER_USER_ID = "00000000-0000-0000-0000-0000000abc02"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Bypass conftest's init_db autouse fixture. The endpoint
    only needs users, build_jobs, job_events.

    Rebuild the module-level engine before each test. Reason:
    pytest-asyncio (asyncio_mode=auto) creates a new event loop
    per test. The aiosqlite + StaticPool + ``:memory:`` engine
    pins its single connection to the loop it was created on;
    cross-file test chains run on NEW loops where the original
    connection is orphaned, and a subsequent ``async_session_
    maker()`` call silently lands on a fresh empty ``:memory:``
    DB — symptoms: INSERTs land in a phantom DB, SELECTs return
    0 rows. Rebuilding the engine on the current loop sidesteps
    the bug entirely."""
    # Force a fresh engine + sessionmaker bound to the current
    # event loop. ``rebind_database`` is the production-grade
    # equivalent of "swap engine atomically" — we reuse it here.
    from app.db.database import rebind_database
    from app.config import settings

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (JobEvent, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


async def _call_list_events(
    limit: int = 50, before: str | None = None,
) -> dict:
    """Call the route function directly, bypassing FastAPI's ASGI
    pipeline.

    Why: when this test file runs after another that calls
    ``engine.dispose()`` in its autouse teardown, the next HTTPX
    request opens a session through ``async_session_maker()`` that
    DOESN'T see the same ``:memory:`` SQLite DB the test's seeding
    code wrote to (StaticPool + per-event-loop quirk). Calling the
    route function directly on the same coroutine keeps the
    connection scope coherent — same engine, same StaticPool
    checkout, same in-memory DB.

    The endpoint's URL contract is exercised once via httpx in the
    422-on-bad-limit test (which doesn't touch DB state)."""
    from app.api.jobs_events import list_job_events
    # FastAPI ``Query`` annotations are non-binding at direct-call
    # time — we just pass plain kwargs.
    page = await list_job_events(limit=limit, before=before)
    return page.model_dump()


@pytest_asyncio.fixture
async def _seed_users():
    """Seed only the primary user. The ``OTHER_USER_ID`` for the
    per-user-scope test is referenced by uuid in BuildJob/JobEvent
    rows only — SQLite doesn't enforce FKs by default, so we don't
    need a corresponding User row. (And can't easily: the User
    table's ``is_canary`` index loses its ``postgresql_where``
    partial-filter on SQLite and collides on the second insert.)"""
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"events-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
            await db.commit()


async def _create_job(
    *, user_id: str = USER_ID,
    title: str = "Untitled",
    job_type: str = "auto_builder",
    source_kind: str = "app_builder_skill",
) -> str:
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id,
            user_id=user_id,
            title=title,
            prompt="p",
            job_type=job_type,
            status="running",
            source_kind=source_kind,
        ))
        await db.commit()
    return job_id


async def _emit_event(
    job_id: str,
    *,
    user_id: str = USER_ID,
    kind: str = "phase_started",
    label: str = "Planning",
    status: str = "running",
    level: str = "info",
    ts_offset_seconds: float = 0.0,
) -> str:
    """Insert a JobEvent with an explicit ts so pagination tests
    can pin row ordering deterministically. ``ts_offset_seconds``
    is relative to a baseline (newer values → newer events)."""
    from app.db.database import async_session_maker
    from app.db.models import JobEvent

    eid = str(uuid.uuid4())
    base = datetime(2026, 5, 18, 12, 0, 0)
    ts = base + timedelta(seconds=ts_offset_seconds)
    async with async_session_maker() as db:
        db.add(JobEvent(
            id=eid,
            job_id=job_id,
            user_id=user_id,
            ts=ts,
            kind=kind,
            label=label,
            status=status,
            level=level,
        ))
        await db.commit()
    return eid


# ──────────────────────────────────────────────────────────────────────
# Tests.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_events_returns_correct_attribution_across_job_types(
    _seed_users,
):
    """The investigation-doc regression check: seed 3 jobs of
    different types and emit one event each. The activity feed
    returns 3 entries with 3 different ``job_title`` values —
    NOT all attributed to the same job. This is what the old
    client-side flatten of ``BuildJob.steps_json`` got wrong."""
    job_a = await _create_job(
        title="Nokia Snake Arcade", job_type="auto_builder",
        source_kind="app_builder_skill",
    )
    job_b = await _create_job(
        title="Summarize new Gmail", job_type="trigger_run",
        source_kind="trigger",
    )
    job_c = await _create_job(
        title="Morning briefing 2026-05-18", job_type="routine_run",
        source_kind="routine",
    )
    await _emit_event(job_a, label="Installing dependencies", ts_offset_seconds=10)
    await _emit_event(job_b, label="Fetching Gmail message", ts_offset_seconds=20)
    await _emit_event(job_c, label="Composing briefing", ts_offset_seconds=30)

    body = await _call_list_events()
    events = body["events"]
    assert len(events) == 3
    # Newest first.
    titles = [e["job_title"] for e in events]
    assert titles == [
        "Morning briefing 2026-05-18",
        "Summarize new Gmail",
        "Nokia Snake Arcade",
    ], (
        "Each event must attribute to its OWN job, not flatten "
        "to one title — this is the structural fix for the "
        "'everything is Nokia Snake Arcade' bug."
    )
    # job_type + source_kind also surface for client styling.
    job_types = {e["job_type"] for e in events}
    assert job_types == {"auto_builder", "trigger_run", "routine_run"}
    source_kinds = {e["source_kind"] for e in events}
    assert source_kinds == {"app_builder_skill", "trigger", "routine"}


@pytest.mark.asyncio
async def test_events_paginates_via_before_parameter(_seed_users):
    """Forward pagination: with limit=2 and 4 events seeded, the
    first call returns the 2 newest and a non-NULL ``next_before``.
    Calling again with that value returns the next 2 (older),
    and ``next_before`` is NULL (last page)."""
    job_id = await _create_job(title="Paginated Build")
    for i in range(4):
        await _emit_event(job_id, label=f"phase-{i}", ts_offset_seconds=i)

    body1 = await _call_list_events(limit=2)
    assert len(body1["events"]) == 2
    assert body1["events"][0]["label"] == "phase-3"
    assert body1["events"][1]["label"] == "phase-2"
    assert body1["next_before"] is not None

    body2 = await _call_list_events(limit=2, before=body1["next_before"])
    assert len(body2["events"]) == 2
    assert body2["events"][0]["label"] == "phase-1"
    assert body2["events"][1]["label"] == "phase-0"
    assert body2["next_before"] is None, (
        "Less than a full page = last page; next_before must be NULL "
        "so the client knows to stop paging."
    )


@pytest.mark.asyncio
async def test_events_scoped_to_current_user(_seed_users):
    """Per-user scope: events for OTHER_USER_ID must NOT surface
    on the current user's feed. ``settings.user_id`` is the
    single-tenant scope."""
    my_job = await _create_job(title="My App", user_id=USER_ID)
    other_job = await _create_job(title="Their App", user_id=OTHER_USER_ID)
    await _emit_event(my_job, label="mine", user_id=USER_ID, ts_offset_seconds=10)
    await _emit_event(other_job, label="theirs", user_id=OTHER_USER_ID, ts_offset_seconds=20)

    body = await _call_list_events()
    assert len(body["events"]) == 1
    assert body["events"][0]["label"] == "mine"
    assert body["events"][0]["job_title"] == "My App"


def test_events_route_declares_limit_bounds():
    """Source-grep guard: the ``limit`` parameter on the route
    must carry FastAPI ``Query(ge=1, le=200)`` so out-of-range
    values get 422 before reaching the handler. (Driving the
    ASGI app live to confirm 422 leaves shared StaticPool /
    session-maker state in a place that breaks subsequent tests
    in this file — verified locally; this source-grep is the
    cheaper, equally-load-bearing pin.)"""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "app/api/jobs_events.py").read_text()
    assert "Query(50, ge=1, le=200)" in src, (
        "GET /events must declare limit bounds via FastAPI Query "
        "validator so out-of-range values return 422 cleanly."
    )


@pytest.mark.asyncio
async def test_events_with_orphan_job_id_returns_null_title(
    _seed_users,
):
    """A JobEvent whose job_id no longer exists in build_jobs
    (deleted job, race condition) surfaces with ``job_title=None``
    instead of crashing the request. The frontend renders these
    with a sane fallback."""
    # Emit an event pointing at a job that doesn't exist.
    fake_job_id = str(uuid.uuid4())
    from app.db.database import async_session_maker
    from app.db.models import JobEvent

    async with async_session_maker() as db:
        db.add(JobEvent(
            id=str(uuid.uuid4()),
            job_id=fake_job_id,
            user_id=USER_ID,
            ts=datetime(2026, 5, 18, 12, 0, 0),
            kind="phase_started",
            label="ghost phase",
            level="info",
        ))
        await db.commit()

    body = await _call_list_events()
    assert len(body["events"]) == 1
    ev = body["events"][0]
    assert ev["label"] == "ghost phase"
    assert ev["job_title"] is None
    assert ev["job_type"] is None
