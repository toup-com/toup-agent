"""Job parity — agent-task intake paths populate the unified-arc columns.

PR 4c of the unified-jobs arc. Three intake paths landed in that PR:

  1. Dashboard POST ``/apps/jobs/`` (``apps.py:create_job``)
  2. Chat-intent regex (``ws_chat.py:_detect_and_create_task``) — RETIRED in
     the voice round of 2026-10-05 (D-P4): the runner mints a chat turn's
     card itself (``voice_jobs.TurnJob``); its tests left with it.
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
