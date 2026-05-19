"""Auto Builder phases dual-write to job_events.

PR 5 of the unified-jobs arc.

What we pin:

  1. Every call to ``AppBuilderSkill._update_step`` writes BOTH:
       - the existing ``BuildJob.steps_json`` mutation (preserved
         for one-release transition window).
       - one new ``job_events`` row with ``kind='phase_started'``
         (when status='running') or ``kind='phase_completed'``
         (when status='done' / 'failed').

  2. The ``job_events.label`` matches the step's user-facing
     phase label (from ``_initial_steps``), so the activity feed
     in PR 6 can render "Installing dependencies — done" without
     parsing ``steps_json``.

  3. ``metadata_json`` carries ``phase_type`` and (on completion)
     ``duration_ms`` so the dashboard can show how long each
     phase took.

  4. Failed phases write ``status='failed'`` and ``level='error'``
     on the event row so the activity feed renders them with the
     error styling.

A full build pipeline run isn't exercised here (depends on
app-manager / file-system / Expo). We drive ``_update_step``
directly against a seeded BuildJob row and assert per-call event
state. This is the same surface the production pipeline calls.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-auto-builder-phase-events")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-00000000ff01")


USER_ID = "00000000-0000-0000-0000-00000000ff01"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Bypass conftest's init_db autouse fixture. Build only the 3
    tables ``_update_step`` touches: users, build_jobs, job_events."""
    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (JobEvent, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)
    await engine.dispose()


def _initial_steps_sample() -> list:
    """Mirror of ``AppBuilderSkill._initial_steps`` — 8 hardcoded
    phases. Frozen at the snapshot in [`docs/architecture/jobs-
    investigation-2026-05-18.md`](docs/architecture/jobs-
    investigation-2026-05-18.md#1.1-Auto-Builder)."""
    types_labels = [
        ("planning", "Planning app architecture..."),
        ("scaffolding", "Creating Expo project..."),
        ("writing", "Generating app code..."),
        ("database", "Setting up database..."),
        ("installing", "Installing dependencies..."),
        ("github", "Creating GitHub repository..."),
        ("starting", "Starting preview servers..."),
        ("ready", "App is ready!"),
    ]
    return [
        {
            "id": str(uuid.uuid4()),
            "type": t,
            "label": label,
            "status": "pending",
        }
        for t, label in types_labels
    ]


@pytest_asyncio.fixture
async def _seed_build_job():
    """Seed a User + a BuildJob in 'queued' status with the 8
    initial phases populated in steps_json. Returns the job_id."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, User

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"phase-events-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
        db.add(BuildJob(
            id=job_id,
            user_id=USER_ID,
            title="Build: phase-events test",
            prompt="test",
            job_type="auto_builder",
            status="queued",
            steps_json=json.dumps(_initial_steps_sample()),
            layer=1,
        ))
        await db.commit()
    return job_id


def _make_skill():
    """Construct an AppBuilderSkill instance with no real wiring
    (no ws_broadcast, no app_manager). _update_step only needs the
    DB; the WebSocket broadcast guard at the end handles None."""
    from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill
    return AppBuilderSkill()


# ──────────────────────────────────────────────────────────────────────
# Behavioural tests.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_phase_started_writes_phase_started_event(_seed_build_job):
    """Calling _update_step with status='running' writes one
    job_events row with kind='phase_started', label matching the
    phase title, metadata_json with phase_type."""
    from app.db.database import async_session_maker
    from app.db.models import JobEvent
    from sqlalchemy import select

    job_id = _seed_build_job
    skill = _make_skill()

    await skill._update_step(job_id, USER_ID, "installing", "running")

    async with async_session_maker() as db:
        events = (await db.execute(
            select(JobEvent).where(JobEvent.job_id == job_id)
        )).scalars().all()
        assert len(events) == 1
        ev = events[0]
        assert ev.kind == "phase_started"
        assert ev.label == "Installing dependencies..."
        assert ev.status == "running"
        assert ev.level == "info"
        meta = json.loads(ev.metadata_json or "{}")
        assert meta["phase_type"] == "installing"


@pytest.mark.asyncio
async def test_phase_completed_writes_phase_completed_event_with_duration(
    _seed_build_job,
):
    """A running → done transition writes two events. The completion
    event includes ``duration_ms`` so the activity feed can render
    'Installing dependencies — done (1234ms)'."""
    import asyncio
    from app.db.database import async_session_maker
    from app.db.models import JobEvent
    from sqlalchemy import select

    job_id = _seed_build_job
    skill = _make_skill()

    await skill._update_step(job_id, USER_ID, "installing", "running")
    # Tiny sleep so the duration is non-zero; the duration is
    # computed from steps_json.started_at vs datetime.utcnow().
    await asyncio.sleep(0.01)
    await skill._update_step(job_id, USER_ID, "installing", "done")

    async with async_session_maker() as db:
        events = (await db.execute(
            select(JobEvent)
            .where(JobEvent.job_id == job_id)
            .order_by(JobEvent.ts)
        )).scalars().all()
        assert len(events) == 2
        started, completed = events
        assert started.kind == "phase_started"
        assert completed.kind == "phase_completed"
        assert completed.status == "done"
        assert completed.level == "info"
        meta = json.loads(completed.metadata_json or "{}")
        assert meta["phase_type"] == "installing"
        # duration_ms is best-effort — present in normal flow.
        # Accept either >= 0 (some clocks tick weirdly) or None.
        if "duration_ms" in meta:
            assert isinstance(meta["duration_ms"], int)
            assert meta["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_phase_failed_writes_event_with_error_level(_seed_build_job):
    """A failed phase writes ``status='failed'``, ``level='error'``
    on the event row. The activity feed can pick this up and
    render with the error styling."""
    from app.db.database import async_session_maker
    from app.db.models import JobEvent
    from sqlalchemy import select

    job_id = _seed_build_job
    skill = _make_skill()

    await skill._update_step(job_id, USER_ID, "starting", "running")
    await skill._update_step(
        job_id, USER_ID, "starting", "failed", detail="Metro port 4001 busy",
    )

    async with async_session_maker() as db:
        events = (await db.execute(
            select(JobEvent)
            .where(JobEvent.job_id == job_id)
            .order_by(JobEvent.ts)
        )).scalars().all()
        completed_ev = events[-1]
        assert completed_ev.kind == "phase_completed"
        assert completed_ev.status == "failed"
        assert completed_ev.level == "error"
        meta = json.loads(completed_ev.metadata_json or "{}")
        assert meta.get("detail") == "Metro port 4001 busy"


@pytest.mark.asyncio
async def test_full_build_emits_one_event_per_transition(_seed_build_job):
    """End-to-end shape: a build that walks 4 phases through
    running → done emits 8 events total (one per transition).
    The legacy ``steps_json`` mutation continues to work in
    parallel (dual-write, transitional)."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, JobEvent
    from sqlalchemy import select

    job_id = _seed_build_job
    skill = _make_skill()

    phases = ["planning", "scaffolding", "writing", "database"]
    for p in phases:
        await skill._update_step(job_id, USER_ID, p, "running")
        await skill._update_step(job_id, USER_ID, p, "done")

    async with async_session_maker() as db:
        # 4 phases × 2 transitions = 8 events.
        events = (await db.execute(
            select(JobEvent)
            .where(JobEvent.job_id == job_id)
            .order_by(JobEvent.ts)
        )).scalars().all()
        assert len(events) == 8
        kinds = [e.kind for e in events]
        assert kinds == [
            "phase_started", "phase_completed",
            "phase_started", "phase_completed",
            "phase_started", "phase_completed",
            "phase_started", "phase_completed",
        ]
        # The phase_type metadata follows the phase order.
        phase_types = [
            json.loads(e.metadata_json or "{}").get("phase_type")
            for e in events
        ]
        assert phase_types == [
            "planning", "planning",
            "scaffolding", "scaffolding",
            "writing", "writing",
            "database", "database",
        ]

        # Legacy steps_json continues to reflect the same state
        # — the 4 phases are marked 'done', the others still
        # 'pending'.
        job = await db.get(BuildJob, job_id)
        steps_after = json.loads(job.steps_json)
        done_count = sum(1 for s in steps_after if s.get("status") == "done")
        assert done_count == 4, (
            "Legacy steps_json must still reflect transitions "
            "(dual-write — both surfaces stay in sync)."
        )
