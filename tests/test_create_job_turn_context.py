"""BEHAVIOURAL tests for the per-turn context the `create_job` tool depends on.

These exist because a source-grep test suite let a completely INERT fix ship.
The turn-end finalizer in `AgentRunner.run()` originally selected jobs with
`BuildJob.conversation_id == session_id`, but the producer wrote
`conversation_id=getattr(self, "_current_session_id", None)` — an attribute
NOTHING in the codebase ever assigned. So the column was always NULL, the
predicate never matched a row, and every structural test still passed because
the source text looked exactly right.

The lesson: assert on the DATABASE ROW and on the recorded ids, not on the
source text. Each test below fails if that class of bug returns.

Two invariants under test:
  1. `_tool_create_job` stamps the real conversation id onto the row and
     records the job id for the finalizer to close.
  2. That state is per-asyncio-task (ContextVars), because ONE ToolExecutor is
     shared across concurrent turns and by spawned sub-agents. An instance
     attribute here would let one user's turn close another's jobs.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest


def _executor(tmp_path):
    from app.agent.tool_executor import ToolExecutor
    return ToolExecutor(workspace=str(tmp_path))


async def _create_job(te, user_id: str, title: str = "Research the newest model"):
    te.set_user_id(user_id)
    out = await te._tool_create_job({
        "title": title,
        "description": "check release date and name",
        "steps": ["find the announcement", "confirm the name", "summarise"],
    })
    return json.loads(out)


# ── Invariant 1: the row and the recorded ids are real ──────────────────

@pytest.mark.asyncio
async def test_create_job_stamps_the_conversation_id_on_the_row(tmp_path, test_user_id):
    """THE REGRESSION TEST. conversation_id must be the session we set, not NULL.

    A NULL here is what made the finalizer inert, and it also silently broke the
    documented Mission Control affordance ("spawned from chat with ___")."""
    from app.db import async_session_maker
    from app.db.models import BuildJob

    te = _executor(tmp_path)
    session_id = f"sess-{uuid.uuid4()}"
    te.set_session_id(session_id)

    res = await _create_job(te, test_user_id)

    async with async_session_maker() as db:
        row = await db.get(BuildJob, res["job_id"])
    assert row is not None, "create_job must persist a BuildJob"
    assert row.conversation_id == session_id, (
        f"conversation_id is {row.conversation_id!r}; a NULL here is the exact "
        "bug that made the turn-end finalizer a no-op"
    )
    assert row.status == "running"
    assert row.source_kind == "manual"
    assert row.job_type == "agent_task"


@pytest.mark.asyncio
async def test_created_job_ids_are_recorded_then_drained(tmp_path, test_user_id):
    """The finalizer closes EXACT ids, so they must be recorded, and draining
    must be idempotent (a second turn must not re-close the first turn's jobs)."""
    te = _executor(tmp_path)
    te.set_session_id(f"sess-{uuid.uuid4()}")

    a = await _create_job(te, test_user_id, "first")
    b = await _create_job(te, test_user_id, "second")

    ids = te.take_created_job_ids()
    assert list(ids) == [a["job_id"], b["job_id"]], "both ids, in order"
    assert te.take_created_job_ids() == (), "draining must reset"


@pytest.mark.asyncio
async def test_set_session_id_clears_the_previous_turns_ids(tmp_path, test_user_id):
    """A new turn on the same shared executor must not inherit stale ids —
    otherwise turn N+1 would close turn N's jobs."""
    te = _executor(tmp_path)
    te.set_session_id("sess-one")
    await _create_job(te, test_user_id, "from the first turn")

    te.set_session_id("sess-two")          # next turn begins
    assert te.take_created_job_ids() == ()


# ── Invariant 2: per-task isolation (the concurrency contract) ──────────

@pytest.mark.asyncio
async def test_concurrent_turns_do_not_see_each_others_jobs(tmp_path, test_user_id):
    """GATE. One ToolExecutor is shared across concurrent turns. If this state
    were an instance attribute, whichever turn finished last would close the
    other turn's job too. ContextVars are per-asyncio-task, so each turn must
    see only its own id."""
    te = _executor(tmp_path)
    seen: dict[str, tuple] = {}

    async def turn(tag: str) -> None:
        te.set_session_id(f"sess-{tag}")
        res = await _create_job(te, test_user_id, f"job for {tag}")
        await asyncio.sleep(0.05)          # interleave the two turns
        seen[tag] = (res["job_id"], te.take_created_job_ids())

    await asyncio.gather(turn("A"), turn("B"))

    id_a, drained_a = seen["A"]
    id_b, drained_b = seen["B"]
    assert id_a != id_b
    assert list(drained_a) == [id_a], f"turn A leaked: {drained_a}"
    assert list(drained_b) == [id_b], f"turn B leaked: {drained_b}"


@pytest.mark.asyncio
async def test_concurrent_turns_stamp_their_own_conversation_id(tmp_path, test_user_id):
    """Same isolation, verified on the persisted rows rather than in memory."""
    from app.db import async_session_maker
    from app.db.models import BuildJob

    te = _executor(tmp_path)
    out: dict[str, str] = {}

    async def turn(tag: str) -> None:
        te.set_session_id(f"sess-{tag}")
        res = await _create_job(te, test_user_id, f"job for {tag}")
        await asyncio.sleep(0.05)
        out[tag] = res["job_id"]

    await asyncio.gather(turn("A"), turn("B"))

    async with async_session_maker() as db:
        row_a = await db.get(BuildJob, out["A"])
        row_b = await db.get(BuildJob, out["B"])
    assert row_a.conversation_id == "sess-A"
    assert row_b.conversation_id == "sess-B"


# ── The finalizer's write semantics, on real rows ───────────────────────

@pytest.mark.asyncio
async def test_guarded_update_refuses_to_reopen_a_terminal_job(tmp_path, test_user_id):
    """The finalizer's UPDATE re-checks status='running' at write time so it
    cannot clobber a row that `update_job` or the reaper already drove
    terminal. Exercised here with the same statement shape the finalizer uses."""
    from sqlalchemy import update
    from app.db import async_session_maker
    from app.db.models import BuildJob
    from datetime import datetime

    te = _executor(tmp_path)
    te.set_session_id(f"sess-{uuid.uuid4()}")
    res = await _create_job(te, test_user_id)
    job_id = res["job_id"]

    # A concurrent writer fails it first (what the reaper does).
    async with async_session_maker() as db:
        row = await db.get(BuildJob, job_id)
        row.status = "failed"
        row.error_message = "hit an error"
        await db.commit()

    # The finalizer now runs; its WHERE must match nothing.
    async with async_session_maker() as db:
        r = await db.execute(
            update(BuildJob)
            .where(BuildJob.id == job_id,
                   BuildJob.user_id == test_user_id,
                   BuildJob.status == "running")
            .values(status="completed", completed_at=datetime.utcnow())
            .returning(BuildJob.id)
        )
        assert r.first() is None, "must not resurrect a terminal job"
        await db.commit()

    async with async_session_maker() as db:
        row = await db.get(BuildJob, job_id)
    assert row.status == "failed", "the concurrent terminal write must survive"
    assert row.error_message == "hit an error"
