"""Job parity — Auto Builder intake repointed through JobRunner.

PR 4d of the unified-jobs arc — the final intake-side repoint.

Auto Builder's intake (``_exec_build_app`` at
``backend/app/agent/skills/builtins/app_builder/skill.py``) creates
both an ``apps`` row and a ``build_jobs`` row. Pre-PR-4d those
were committed in a single transaction with a slug-collision retry
loop. PR 4d splits them: App with retry, then JobRunner.create_job
for the BuildJob carrying the unified-arc columns.

What we pin:

  - The split preserves the slug-collision retry: a unique-slug
    conflict on the App row triggers a fresh slug + app_id, and
    the BuildJob is minted only after the App commits.
  - The BuildJob carries source_kind='app_builder_skill',
    source_id=app.id, idempotency_key=slug.
  - status='queued' (legitimate — the background _build_app
    task picks the row up and runs phases sequentially).
  - The 8 initial phase steps are populated in steps_json (PR 5
    will start mirroring them to job_events).

The test does NOT drive ``_exec_build_app`` end-to-end — that
method depends on a full app-manager / file-system / Expo
boot setup that's out of scope here. We use source-grep guards
plus a behavioural test of the new JobRunner contract that the
caller relies on (``app_id=``, ``idempotency_key=``,
``status='queued'``).
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-job-parity-auto-builder")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000ee1")


USER_ID = "00000000-0000-0000-0000-000000000ee1"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    from app.db.database import engine
    from app.db.models import App, BuildJob, JobEvent, User

    async with engine.begin() as conn:
        for model_cls in (User, App, BuildJob, JobEvent):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (JobEvent, BuildJob, App, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)
    await engine.dispose()


@pytest_asyncio.fixture
async def _seed_user():
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"parity-builder-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
            await db.commit()


# ──────────────────────────────────────────────────────────────────────
# Source-grep guards — pin the Auto Builder intake repoint.
# ──────────────────────────────────────────────────────────────────────


_BACKEND = Path(__file__).resolve().parent.parent
_SKILL_SRC = (
    _BACKEND / "app/agent/skills/builtins/app_builder/skill.py"
).read_text()


def test_auto_builder_intake_uses_job_runner():
    """The ``_exec_build_app`` method must mint the BuildJob via
    ``JobRunner.create_job`` rather than inline ``BuildJob(...)``."""
    func_start = _SKILL_SRC.find("async def _exec_build_app")
    assert func_start != -1
    func_end = _SKILL_SRC.find("\n    async def _exec_get_status", func_start)
    func_body = _SKILL_SRC[func_start:func_end if func_end > -1 else func_start + 8000]
    assert "JobRunner" in func_body and "create_job" in func_body, (
        "_exec_build_app must call JobRunner.create_job; the inline "
        "BuildJob(...) construction is the pre-PR-4d pattern."
    )


def test_auto_builder_intake_passes_unified_arc_kwargs():
    """The JobRunner.create_job call inside ``_exec_build_app`` must
    pass source_kind='app_builder_skill', source_id=app_id,
    idempotency_key=slug, and app_id=app_id so the activity feed
    in PR 6 can attribute Auto Builder jobs back to their App."""
    func_start = _SKILL_SRC.find("async def _exec_build_app")
    func_end = _SKILL_SRC.find("\n    async def _exec_get_status", func_start)
    func_body = _SKILL_SRC[func_start:func_end if func_end > -1 else func_start + 8000]
    assert 'source_kind="app_builder_skill"' in func_body
    assert "source_id=app_id" in func_body
    assert "idempotency_key=slug" in func_body
    assert "app_id=app_id" in func_body


def test_auto_builder_preserves_initial_steps_in_steps_json():
    """Auto Builder pre-declares 8 initial phases via
    ``_initial_steps``. PR 4d must continue passing those into
    BuildJob.steps_json (via JobRunner's new steps_json= kwarg)
    so PR 5 can emit them as phase_started / phase_completed
    job_events without a separate migration."""
    func_start = _SKILL_SRC.find("async def _exec_build_app")
    func_end = _SKILL_SRC.find("\n    async def _exec_get_status", func_start)
    func_body = _SKILL_SRC[func_start:func_end if func_end > -1 else func_start + 8000]
    assert "steps_json=json.dumps(self._initial_steps())" in func_body, (
        "_exec_build_app must pass the 8 initial phase rows through "
        "JobRunner.create_job(steps_json=...). Otherwise the dashboard "
        "activity feed loses the build's phase timeline."
    )


def test_auto_builder_keeps_slug_collision_retry():
    """The slug-collision retry loop must survive PR 4d's refactor
    — concurrent builds of the same name still need to deconflict.
    Source-grep the retry loop's invariants."""
    func_start = _SKILL_SRC.find("async def _exec_build_app")
    func_end = _SKILL_SRC.find("\n    async def _exec_get_status", func_start)
    func_body = _SKILL_SRC[func_start:func_end if func_end > -1 else func_start + 8000]
    assert "_slug_retry" in func_body, "retry loop variable missing"
    assert "IntegrityError" in func_body, (
        "retry loop must catch IntegrityError on the App unique-slug "
        "constraint"
    )
    assert "Slug collision" in func_body, (
        "retry loop must log on collision so operators see when it fires"
    )


# ──────────────────────────────────────────────────────────────────────
# Behavioural — exercise the JobRunner contract Auto Builder relies on.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_builder_jobrunner_contract(_seed_user):
    """Directly exercise the JobRunner.create_job kwargs that
    ``_exec_build_app`` passes. The full ``_exec_build_app`` path
    depends on app-manager / file-system setup that's out of
    scope; this test pins the contract the caller is committing
    to."""
    import json
    from app.agent.job_runner import JobRunner, TaskSpec
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    app_id = str(uuid.uuid4())
    slug = f"my-app-{uuid.uuid4().hex[:6]}"

    spec = TaskSpec(
        user_id=USER_ID,
        channel="app_builder",
        source_kind="app_builder_skill",
        source_id=app_id,
    )
    initial_steps = [
        {"id": str(uuid.uuid4()), "type": "planning", "label": "Plan", "status": "pending"},
        {"id": str(uuid.uuid4()), "type": "writing", "label": "Code", "status": "pending"},
    ]
    job = await JobRunner().create_job(
        job_type="auto_builder",
        spec=spec,
        title="Build: My App",
        prompt="a simple app",
        status="queued",
        steps_json=json.dumps(initial_steps),
        app_id=app_id,
        idempotency_key=slug,
        layer=1,
    )

    async with async_session_maker() as db:
        row = await db.get(BuildJob, job.id)
        assert row.job_type == "auto_builder"
        assert row.status == "queued"
        assert row.layer == 1
        assert row.app_id == app_id
        assert row.source_kind == "app_builder_skill"
        assert row.source_id == app_id
        assert row.idempotency_key == slug
        # Initial phases survive on the row.
        loaded = json.loads(row.steps_json)
        assert len(loaded) == 2
        assert loaded[0]["type"] == "planning"


@pytest.mark.asyncio
async def test_auto_builder_idempotency_returns_existing_job(_seed_user):
    """Two ``create_job`` calls with the same (app_id, slug) — e.g.
    the user double-clicks "Build" — return the existing Job. The
    background _build_app task only runs once."""
    from app.agent.job_runner import JobRunner, TaskSpec
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select, func

    app_id = str(uuid.uuid4())
    slug = "my-app-double-click"
    spec = TaskSpec(
        user_id=USER_ID,
        channel="app_builder",
        source_kind="app_builder_skill",
        source_id=app_id,
    )

    first = await JobRunner().create_job(
        job_type="auto_builder",
        spec=spec,
        title="Build: My App",
        status="queued",
        app_id=app_id,
        idempotency_key=slug,
    )
    second = await JobRunner().create_job(
        job_type="auto_builder",
        spec=spec,
        title="Build: My App (retry click)",
        status="queued",
        app_id=app_id,
        idempotency_key=slug,
    )
    assert first.id == second.id

    async with async_session_maker() as db:
        count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == app_id,
                BuildJob.idempotency_key == slug,
            )
        )).scalar_one()
        assert count == 1
