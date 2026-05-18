"""JobRunner — unified Job envelope + dispatch + idempotency.

PR 3 of the unified jobs/tasks/logs arc.

What we pin:

  1. ``JobRunner.create_job`` writes a ``BuildJob`` with the
     unified-arc columns populated correctly (source_kind,
     source_id, conversation_id, idempotency_key).
  2. Idempotency: calling ``create_job`` twice with the same
     ``(source_id, idempotency_key)`` returns the *existing* row,
     not a duplicate.
  3. Dispatch: ``execute`` looks up the handler in
     ``JobRunner.HANDLERS`` by ``job_type`` and calls it.
  4. Missing handler: ``execute`` marks the row failed with a
     diagnostic ``error_message`` and returns without raising.
  5. Handler crash: ``execute`` marks the row failed with the
     exception repr and re-raises.
  6. Handler-set status preserved: if a handler sets terminal
     status (e.g. ``skipped_filter``) before returning normally,
     the runner does NOT clobber it with ``completed``.
  7. ``JobLogger`` dual-writes to ``job_events`` in addition to
     ``build_logs_json``.

PR 3 does NOT wire any production handlers — the registry starts
empty. PR 4 connects ``auto_builder`` / ``agent_task`` /
``trigger_run`` / ``routine_run`` / ``vibe_code`` against their
respective code paths.
"""
from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import Mock

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-job-runner")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000aa1")


USER_ID = "00000000-0000-0000-0000-000000000aa1"


# ── Schema setup ──────────────────────────────────────────────────────
#
# The conftest's autouse ``_reset_database`` calls ``init_db()`` which
# fails under SQLite for the ``entities.embedding`` pgvector column.
# Override with a narrow fixture that creates only what these tests
# need: users, build_jobs, job_events. Same pattern as
# test_trigger_runner_failure_handler.py.


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
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
                email=f"job-runner-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
            await db.commit()
    return USER_ID


@pytest_asyncio.fixture
async def runner():
    """Fresh JobRunner per test, plus a clean HANDLERS registry so
    one test's registrations don't leak into another."""
    from app.agent.job_runner import JobRunner

    original = dict(JobRunner.HANDLERS)
    JobRunner.HANDLERS.clear()
    yield JobRunner()
    JobRunner.HANDLERS.clear()
    JobRunner.HANDLERS.update(original)


def _spec(**overrides) -> Any:
    """Build a TaskSpec with sensible defaults."""
    from app.agent.job_runner import TaskSpec

    base = dict(
        user_id=USER_ID,
        channel="web",
        source_kind="manual",
        source_id=None,
        prompt=None,
        config_json=None,
        conversation_id=None,
    )
    base.update(overrides)
    return TaskSpec(**base)


# ──────────────────────────────────────────────────────────────────────
# create_job
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_job_populates_unified_arc_columns(runner, _seed_user):
    """A freshly-created job row carries source_kind, source_id,
    conversation_id, and idempotency_key from the spec. The legacy
    fields (status, job_type, title, prompt) are populated too."""
    spec = _spec(
        channel="trigger",
        source_kind="trigger",
        source_id="trigger-abc",
        conversation_id="conv-xyz",
        prompt="Summarize this email",
    )
    job = await runner.create_job(
        job_type="trigger_run",
        spec=spec,
        title="Summarize incoming email",
        idempotency_key="gmail-msg-id-1",
    )
    assert job.job_type == "trigger_run"
    assert job.status == "queued"
    assert job.title == "Summarize incoming email"
    assert job.prompt == "Summarize this email"
    assert job.source_kind == "trigger"
    assert job.source_id == "trigger-abc"
    assert job.conversation_id == "conv-xyz"
    assert job.idempotency_key == "gmail-msg-id-1"
    assert job.user_id == USER_ID
    assert job.id  # uuid populated


@pytest.mark.asyncio
async def test_create_job_idempotency_returns_existing_row(runner, _seed_user):
    """Two ``create_job`` calls with the same
    ``(source_id, idempotency_key)`` return the same row — the second
    is a no-op. Preserves the existing UNIQUE semantics from
    routine_runs (routine_id, scheduled_for_local_date) and
    trigger_events (trigger_id, event_dedupe_id)."""
    spec = _spec(
        source_kind="routine",
        source_id="routine-1",
    )
    first = await runner.create_job(
        job_type="routine_run",
        spec=spec,
        title="Morning briefing",
        idempotency_key="2026-05-18",
    )
    second = await runner.create_job(
        job_type="routine_run",
        spec=spec,
        title="Morning briefing (retry call)",
        idempotency_key="2026-05-18",
    )
    assert first.id == second.id, (
        "second create_job call must return the existing row, not "
        "insert a duplicate"
    )
    # And the title from the first call wins — the second call's
    # title is silently ignored because we returned the existing
    # row.
    assert second.title == "Morning briefing"


@pytest.mark.asyncio
async def test_create_job_no_idempotency_key_allows_duplicates(
    runner, _seed_user,
):
    """Without an idempotency_key, every call creates a fresh row.
    The composite UNIQUE is partial — ``WHERE idempotency_key IS NOT
    NULL`` — so NULL doesn't collide with NULL."""
    spec = _spec()
    first = await runner.create_job(
        job_type="agent_task",
        spec=spec,
        title="Task 1",
    )
    second = await runner.create_job(
        job_type="agent_task",
        spec=spec,
        title="Task 2",
    )
    assert first.id != second.id


# ──────────────────────────────────────────────────────────────────────
# execute
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_dispatches_to_registered_handler(runner, _seed_user):
    """``execute`` looks up the handler in HANDLERS by job_type and
    calls it. The handler receives the job + spec it was created
    with."""
    from app.agent.job_runner import JobRunner

    received = Mock()

    async def handler(job, spec, db_unused):
        received(job_id=job.id, job_type=job.job_type, source_id=spec.source_id)

    JobRunner.register("agent_task", handler)

    spec = _spec(source_kind="chat_intent", source_id="conv-1")
    job = await runner.create_job(
        job_type="agent_task",
        spec=spec,
        title="Investigate failed build",
    )
    await runner.execute(job, spec)

    received.assert_called_once_with(
        job_id=job.id, job_type="agent_task", source_id="conv-1",
    )


@pytest.mark.asyncio
async def test_execute_no_handler_marks_failed_without_raising(
    runner, _seed_user,
):
    """If no handler is registered for the job_type, ``execute``
    marks the row failed with a diagnostic error_message and returns
    without raising. Same fail-loud-but-don't-crash pattern as the
    trigger runner's missing-handler path."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    spec = _spec()
    job = await runner.create_job(
        job_type="unknown_kind",
        spec=spec,
        title="orphan job",
    )

    # Must not raise.
    await runner.execute(job, spec)

    async with async_session_maker() as db:
        fresh = await db.get(BuildJob, job.id)
        assert fresh.status == "failed"
        assert fresh.error_message and "unknown_kind" in fresh.error_message
        assert fresh.completed_at is not None


@pytest.mark.asyncio
async def test_execute_handler_crash_marks_failed_and_reraises(
    runner, _seed_user,
):
    """When the handler raises, ``execute`` writes the exception
    repr into ``error_message``, marks the row failed, and re-raises
    so the caller knows."""
    from app.agent.job_runner import JobRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    async def crashing(job, spec, db_unused):
        raise RuntimeError("boom from handler")

    JobRunner.register("agent_task", crashing)

    spec = _spec()
    job = await runner.create_job(
        job_type="agent_task",
        spec=spec,
        title="will crash",
    )

    with pytest.raises(RuntimeError, match="boom from handler"):
        await runner.execute(job, spec)

    async with async_session_maker() as db:
        fresh = await db.get(BuildJob, job.id)
        assert fresh.status == "failed"
        assert "boom from handler" in (fresh.error_message or "")
        assert fresh.completed_at is not None


@pytest.mark.asyncio
async def test_execute_preserves_terminal_status_set_by_handler(
    runner, _seed_user,
):
    """If the handler sets a terminal status before returning
    normally (e.g. for a skipped/coalesced job), the runner must NOT
    overwrite it with ``failed`` or ``completed``. Production
    handlers (trigger email handler, routine handlers) rely on this
    to communicate fine-grained outcomes."""
    from app.agent.job_runner import JobRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    async def handler_sets_skipped(job, spec, db_unused):
        from datetime import datetime as _dt
        async with async_session_maker() as db:
            row = await db.get(BuildJob, job.id)
            row.status = "completed"
            row.outcome = "skipped_filter"
            row.completed_at = _dt.utcnow()
            await db.commit()

    JobRunner.register("trigger_run", handler_sets_skipped)

    spec = _spec(source_kind="trigger", source_id="t-1")
    job = await runner.create_job(
        job_type="trigger_run",
        spec=spec,
        title="trigger that gets filtered",
    )
    await runner.execute(job, spec)

    async with async_session_maker() as db:
        fresh = await db.get(BuildJob, job.id)
        assert fresh.status == "completed", (
            f"runner clobbered handler-set status. got {fresh.status!r}"
        )
        assert fresh.outcome == "skipped_filter"


# ──────────────────────────────────────────────────────────────────────
# JobLogger dual-write to job_events
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_job_logger_dual_writes_to_job_events(_seed_user):
    """Each ``JobLogger.info/tool/edit/error`` call writes BOTH:
      - one entry to ``build_logs_json`` (the per-job verbose log)
      - one row to ``job_events`` (the cross-job activity feed)

    Pins the dual-write contract introduced in PR 3 so the activity
    feed query in PR 6 has rows to JOIN against."""
    import json
    from sqlalchemy import select
    from app.agent.job_logger import JobLogger
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, JobEvent

    # Seed a job row first — JobEvent.job_id has a CASCADE FK so the
    # parent must exist before we can write events.
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id,
            user_id=USER_ID,
            title="logger test",
            prompt="p",
            job_type="agent_task",
            status="running",
        ))
        await db.commit()

    jl = JobLogger(job_id=job_id, user_id=USER_ID)
    await jl.info("started planning", detail="phase=planning")
    await jl.tool("called gmail__get_message", meta={"msg_id": "abc"})
    await jl.edit("edited routes.py")
    await jl.error("OOM in build", detail="container OOMKilled")

    # In-memory: 4 entries on the logger.
    assert len(jl._logs) == 4
    await jl.persist()

    # On-disk: build_logs_json carries all 4.
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        logs = json.loads(job.build_logs_json or "[]")
        assert len(logs) == 4
        levels_in_blob = [e["level"] for e in logs]
        assert levels_in_blob == ["info", "tool", "edit", "error"]

        # And: job_events table carries one row per log call, with
        # the kind enum mapped per the contract.
        events = (await db.execute(
            select(JobEvent)
            .where(JobEvent.job_id == job_id)
            .order_by(JobEvent.ts)
        )).scalars().all()
        assert len(events) == 4, (
            f"expected 4 job_events rows, got {len(events)}"
        )
        kinds = [e.kind for e in events]
        # info → info, tool → tool_call, edit → tool_call, error → error
        assert kinds == ["info", "tool_call", "tool_call", "error"]

        # And: labels match the messages, levels are preserved.
        labels = [e.label for e in events]
        assert labels[0] == "started planning"
        assert labels[1] == "called gmail__get_message"
        assert labels[2] == "edited routes.py"
        assert labels[3] == "OOM in build"

        # Each event's user_id is denormalized correctly.
        assert all(e.user_id == USER_ID for e in events)
