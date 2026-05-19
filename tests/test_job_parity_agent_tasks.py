"""Job parity — agent-task intake paths populate the unified-arc columns.

PR 4c of the unified-jobs arc. Three intake paths land in this PR:

  1. Dashboard POST ``/apps/jobs/`` (``apps.py:create_job``)
  2. Chat-intent regex (``ws_chat.py:_detect_and_create_task``)
  3. ``create_job`` agent tool (``tool_executor.py:_tool_create_job``)

Unlike PR 4a/4b where the legacy "fire" tables coexist with the
mirrored Job, agent-task paths only ever wrote to ``build_jobs``.
So this is a pure repoint through ``JobRunner.create_job`` — no
dual-write, no status mirror. The behaviour we're pinning:

  - ``job_type='agent_task'``, ``status='running'``, ``layer=0``
    (preserved from pre-PR-4c, dashboard expects these).
  - NEW: ``source_kind`` correctly distinguishes between
    dashboard / chat / agent-authored.
  - NEW: ``source_id`` and ``conversation_id`` link back to the
    spawning conversation when applicable.

PR 4c does NOT change execute behaviour — these tasks still run
inline via ``asyncio.create_task`` against the existing handlers.
PR 5+ migrates execute through JobRunner.execute.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-job-parity-agent")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000dd1")


USER_ID = "00000000-0000-0000-0000-000000000dd1"


# ── Schema setup ──────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Bypass conftest's init_db autouse fixture. Build only
    ``users`` and ``build_jobs``; both intake paths only touch
    these two tables."""
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


@pytest_asyncio.fixture
async def _seed_user():
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"parity-agent-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
            await db.commit()


# ──────────────────────────────────────────────────────────────────────
# Source-grep guards — pin the repoints.
# ──────────────────────────────────────────────────────────────────────


from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_APPS_SRC = (_BACKEND / "app/api/apps.py").read_text()
_WS_CHAT_SRC = (_BACKEND / "app/api/ws_chat.py").read_text()
_TOOL_EXEC_SRC = (_BACKEND / "app/agent/tool_executor.py").read_text()


def test_dashboard_post_uses_job_runner():
    """The dashboard POST handler must call JobRunner.create_job
    rather than constructing BuildJob inline. A future refactor that
    drops the indirection without writing source_kind / source_id
    would silently regress the activity feed in PR 6."""
    assert "JobRunner().create_job" in _APPS_SRC or "JobRunner.create_job" in _APPS_SRC, (
        "apps.py POST handler must go through JobRunner.create_job. "
        "Inline BuildJob(...) construction is the pre-PR-4c pattern."
    )
    # Find the create_job function and confirm source_kind is set.
    func_start = _APPS_SRC.find("async def create_job(req")
    func_end = _APPS_SRC.find("\nclass UpdateJobStatusRequest", func_start)
    if func_end == -1:
        func_end = func_start + 4000
    func_body = _APPS_SRC[func_start:func_end]
    assert 'source_kind="manual"' in func_body, (
        "dashboard POST must set source_kind='manual' on the TaskSpec"
    )


def test_chat_intent_uses_job_runner_with_chat_intent_source():
    """``_detect_and_create_task`` must use source_kind='chat_intent'
    and pass the session_id through as both source_id AND
    conversation_id so Mission Control can attribute the task to
    the chat that spawned it."""
    assert "_detect_and_create_task" in _WS_CHAT_SRC
    func_start = _WS_CHAT_SRC.find("async def _detect_and_create_task")
    func_end = _WS_CHAT_SRC.find("\nasync def ", func_start + 1)
    if func_end == -1:
        func_end = func_start + 3000
    func_body = _WS_CHAT_SRC[func_start:func_end]
    assert "JobRunner" in func_body and "create_job" in func_body
    assert 'source_kind="chat_intent"' in func_body
    assert "source_id=session_id" in func_body
    assert "conversation_id=session_id" in func_body


def test_create_job_tool_uses_job_runner_with_steps_json():
    """The agent-authored ``create_job`` tool must use JobRunner
    AND pass through the agent-supplied ``steps`` array on
    BuildJob.steps_json (PR 5 will start writing those phases to
    job_events; we need them on the row first)."""
    func_start = _TOOL_EXEC_SRC.find("async def _tool_create_job")
    func_end = _TOOL_EXEC_SRC.find("\n    async def ", func_start + 1)
    if func_end == -1:
        func_end = func_start + 3000
    func_body = _TOOL_EXEC_SRC[func_start:func_end]
    assert "JobRunner" in func_body and "create_job" in func_body
    assert "steps_json=" in func_body, (
        "create_job tool must pass steps_json= when spawning a Job — "
        "the agent's step plan needs to land on the row."
    )
    assert 'source_kind="manual"' in func_body


# ──────────────────────────────────────────────────────────────────────
# Behavioural tests — call the intake paths and assert row state.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detect_and_create_task_populates_unified_arc_columns(
    _seed_user,
):
    """Drive ``_detect_and_create_task`` directly. The created
    BuildJob carries source_kind='chat_intent',
    source_id=conversation_id=session_id, status='running',
    layer=0, job_type='agent_task'."""
    import asyncio
    from app.api.ws_chat import _detect_and_create_task
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    session_id = "conv-" + uuid.uuid4().hex[:12]
    bq: asyncio.Queue = asyncio.Queue()

    job_id = await _detect_and_create_task(
        "research the best Postgres-on-NVMe setup",
        USER_ID,
        session_id,
        bq,
    )
    assert job_id is not None

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job is not None
        assert job.job_type == "agent_task"
        assert job.status == "running"
        assert job.layer == 0
        assert job.source_kind == "chat_intent"
        assert job.source_id == session_id
        assert job.conversation_id == session_id
        assert "research" in (job.title or "").lower()


@pytest.mark.asyncio
async def test_detect_and_create_task_non_task_returns_none(_seed_user):
    """Non-task text (no intent verb at the start) still bypasses
    the intake path — regression check that the repoint didn't
    widen the trigger pattern."""
    import asyncio
    from app.api.ws_chat import _detect_and_create_task

    bq: asyncio.Queue = asyncio.Queue()
    job_id = await _detect_and_create_task(
        "hey can you say hi",
        USER_ID,
        "any-session",
        bq,
    )
    assert job_id is None


@pytest.mark.asyncio
async def test_detect_and_create_task_handles_null_session_id(_seed_user):
    """When session_id is None (rare — should always be set in
    practice but the type permits it), the task still gets created
    with source_kind='chat_intent' and NULL source_id / NULL
    conversation_id. Don't crash on the optional input."""
    import asyncio
    from app.api.ws_chat import _detect_and_create_task
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    bq: asyncio.Queue = asyncio.Queue()
    job_id = await _detect_and_create_task(
        "investigate the crash report",
        USER_ID,
        None,
        bq,
    )
    assert job_id is not None
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.source_kind == "chat_intent"
        assert job.source_id is None
        assert job.conversation_id is None
