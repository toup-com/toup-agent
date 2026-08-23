"""Round 25 — the card you come back to is the card you left.

Backgrounding the app during a build and returning showed a "Didn't finish"
snapshot with arithmetic that could not be true, beside a duplicate
placeholder. Two backend causes, and neither is the copy.

**The live frame and the REST payload were computed by different rules.**
``job_update`` carries ``completed_steps``/``total_steps`` — counted by
``step_counts``, which excludes skipped rows from BOTH numbers — while
``JobResponse`` carried the raw ``steps`` array plus the unrelated
``progress_step``/``progress_total`` columns, and left the client to work the
rest out. So the card a user watched live and the card they came back to were
never the same computation, and one of the two was always a guess. They now
come from one function.

**A build killed by a dying turn kept its rows frozen.**
``_close_interrupted_jobs`` terminalises jobs in a single bulk UPDATE of
``status``/``error_class``/``user_message`` and never touches ``steps_json``,
so an app build cancelled mid-flight kept every ``running`` and ``pending`` row
exactly as it stood. The card read "In progress · 4/7 steps" underneath a
status that was already terminal — an in-progress row under a dead job, which
is precisely the impossible snapshot.

SCOPE, stated honestly: the specific "4/7 = 97%" pairing is NOT produced
anywhere in this backend. No percent was computed server-side at all before
this round, and the app build sends no Live Activity progress that could
supply one. That number is client-derived. What the backend can do — and now
does — is publish one authoritative, monotonic percent so there is nothing
left for a client to derive.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.agent.skills.builtins.app_html import steps as steps_mod
from app.db.database import async_session_maker
from app.db.models import BuildJob


@pytest.fixture
def frames(monkeypatch) -> List[Dict[str, Any]]:
    captured: List[Dict[str, Any]] = []

    async def _capture(user_id, payload):
        captured.append(payload)

    async def _no_push(**_k):
        return None

    monkeypatch.setattr(steps_mod, "_broadcast", _capture)
    monkeypatch.setattr(steps_mod, "_push_live_activity", _no_push)
    return captured


async def test_the_rest_payload_agrees_with_the_live_frame(test_user_id, frames):
    """The headline. Whatever the last live frame said, a client that comes
    back and re-reads the job must be told the same thing."""
    from app.api.apps import _job_to_response

    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    for t in ("create", "verify"):
        await steps_mod.emit_step(
            user_id=test_user_id, job_id=job_id, step_type=t, status="done",
        )
    live = frames[-1]

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        rest = _job_to_response(job)

    assert rest.completed_steps == live["completed_steps"]
    assert rest.total_steps == live["total_steps"]
    assert rest.percent == live["percent"]
    assert rest.status == live["status"]
    assert [s["type"] for s in rest.steps] == [
        s["type"] for s in live["steps"]
    ], "the two transports disagreed about which rows exist"


async def test_the_rest_payload_never_undercuts_what_was_already_shown(
    test_user_id, frames,
):
    """It reads the persisted high-water mark, so returning cannot show a bar
    lower than the one the user was looking at when they left."""
    from app.api.apps import _job_to_response

    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="done",
    )
    peak = frames[-1]["percent"]
    # A later row regresses (a forced skip, a refused re-check).
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="review",
        status="running",
    )
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        rest = _job_to_response(job)
    assert rest.percent >= peak


async def test_reading_a_job_does_not_advance_it(test_user_id, frames):
    """A GET must not write. Two clients polling would otherwise race each
    other's high-water marks."""
    from app.api.apps import _job_to_response

    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="done",
    )
    async with async_session_maker() as db:
        before = json.dumps((await db.get(BuildJob, job_id)).config_json or {},
                            sort_keys=True)
    async with async_session_maker() as db:
        _job_to_response(await db.get(BuildJob, job_id))
    async with async_session_maker() as db:
        after = json.dumps((await db.get(BuildJob, job_id)).config_json or {},
                           sort_keys=True)
    assert before == after


async def test_a_non_build_job_gets_no_build_arithmetic(test_user_id):
    """`JobResponse` serves agent tasks and routines too. A job with no step
    plan must not acquire a 0% bar it never had."""
    from app.agent.job_runner import JobRunner, TaskSpec
    from app.api.apps import _job_to_response

    job = await JobRunner().create_job(
        job_type="agent_task",
        spec=TaskSpec(user_id=test_user_id, channel="agent_task",
                      source_kind="manual"),
        title="Research something", prompt="x", status="running", layer=0,
        steps_json=json.dumps([]),
    )
    async with async_session_maker() as db:
        rest = _job_to_response(await db.get(BuildJob, job.id))
    assert rest.total_steps is None
    assert rest.percent is None


# ── A killed turn leaves no in-progress rows behind ───────────────────

async def test_an_interrupted_build_has_no_rows_still_running(
    test_user_id, frames,
):
    """`_close_interrupted_jobs` flips status in a bulk UPDATE and never
    touched `steps_json`, so the rows stayed exactly as the dying turn left
    them: "In progress · 4/7" under a terminal status."""
    from app.agent.agent_runner import _settle_build_steps

    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="done",
    )
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="verify",
        status="running",
    )

    async with async_session_maker() as db:
        await _settle_build_steps(db, job_id)
        await db.commit()

    async with async_session_maker() as db:
        rows = json.loads((await db.get(BuildJob, job_id)).steps_json)
    live = [s["status"] for s in rows if s["status"] in ("running", "pending")]
    assert not live, f"rows left mid-flight on a dead build: {live}"
    # And the work that DID complete is still counted — settling must not
    # un-count a finished phase.
    done, total = steps_mod.step_counts(rows)
    assert done >= 1
    assert done <= total


async def test_settling_keeps_a_phase_that_had_already_completed(
    test_user_id, frames,
):
    """A row mid-RETRY when the turn died completed once. Its last reported
    result stands — skipping it is how a card reading 2/5 died reading 1/5."""
    from app.agent.agent_runner import _settle_build_steps

    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="verify", status="done",
    )
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="verify",
        status="running",
    )
    async with async_session_maker() as db:
        await _settle_build_steps(db, job_id)
        await db.commit()

    async with async_session_maker() as db:
        rows = json.loads((await db.get(BuildJob, job_id)).steps_json)
    verify = [s for s in rows if s["type"] == "verify"][0]
    assert verify["status"] == "done", verify


def test_settle_is_idempotent():
    """It runs from three different close paths and they can overlap."""
    rows = [{"type": "create", "status": "done"},
            {"type": "verify", "status": "running"},
            {"type": "look", "status": "pending"}]
    once = json.dumps(steps_mod.settle_steps(list(rows), detail="x"))
    twice = json.dumps(
        steps_mod.settle_steps(json.loads(once), detail="x")
    )
    assert once == twice
