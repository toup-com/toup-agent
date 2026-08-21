"""Round 15 — one build, one card.

A single "build me a snake game" turn produced TWO cards in the chat:

    [ Building your app          · running 1/4 ]   ← the model's create_job
    [ Build: Nokia Snake Classic · running 0/5 ]   ← the pipeline's own job

Both are legitimate on their own. Together they are one build rendered twice,
with two progress bars that disagree and two things to close. The pipeline now
ADOPTS the job the model already opened this turn instead of adding a second
one beside it, and it is retyped to an app build so every surface that
discriminates on `job_type` renders it as one.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import select

from app.agent.skills.builtins.app_html import steps as steps_mod
from app.agent.tool_executor import _created_job_registry
from app.db.database import async_session_maker
from app.db.models import BuildJob


@pytest.fixture(autouse=True)
def _clean_registry():
    """The per-turn create_job registry is a list in a ContextVar; clear it in
    place (never `.set()` — see _CREATED_JOB_IDS_CTX)."""
    reg = _created_job_registry()
    del reg[:]
    yield
    reg = _created_job_registry()
    del reg[:]


async def _model_created_job(user_id: str, title: str = "Building your app") -> str:
    """What the `create_job` TOOL produces: an agent_task at layer 0 with the
    model's own steps and no app of its own."""
    from app.agent.job_runner import JobRunner, TaskSpec
    job = await JobRunner().create_job(
        job_type="agent_task",
        spec=TaskSpec(user_id=user_id, channel="agent_task", source_kind="manual"),
        title=title,
        prompt="Build a snake game",
        status="running",
        layer=0,
        steps_json=json.dumps([
            {"id": str(uuid.uuid4()), "type": "step_0",
             "label": "Designing the game", "status": "running"},
        ]),
    )
    _created_job_registry().append(job.id)
    return job.id


async def _jobs_for(user_id: str) -> list[BuildJob]:
    async with async_session_maker() as db:
        return list((await db.execute(
            select(BuildJob).where(BuildJob.user_id == user_id)
        )).scalars().all())


async def test_the_pipeline_adopts_the_turns_job_instead_of_adding_one(test_user_id):
    jid = await _model_created_job(test_user_id)

    got = await steps_mod.ensure_job(test_user_id, "nokia-snake-classic",
                                     "Nokia Snake Classic")

    assert got == jid, "a second card was opened beside the model's"
    assert len(await _jobs_for(test_user_id)) == 1

    async with async_session_maker() as db:
        job = await db.get(BuildJob, jid)
    assert job.title == "Build: Nokia Snake Classic"
    # The column every surface discriminates on — the Jobs tab's Build panel,
    # the chat card, the Live Activity lane. Left as `agent_task`, the adopted
    # row would render as a task with no build in it.
    assert job.job_type == "auto_builder"
    assert job.layer == 1
    assert job.app_id == steps_mod.app_id_for(test_user_id, "nokia-snake-classic")
    steps = json.loads(job.steps_json)
    assert [s["type"] for s in steps] == [t for t, _ in steps_mod.STEP_TYPES]
    assert (job.config_json or {}).get("adopted_by") == "app_html"


async def test_every_tool_in_the_turn_lands_on_that_one_card(test_user_id):
    """create → edit → present is three `ensure_job` calls. All three must be
    the same card, or the user watches three cards race."""
    jid = await _model_created_job(test_user_id)
    ids = [
        await steps_mod.ensure_job(test_user_id, "snake", "Snake"),
        await steps_mod.ensure_job(test_user_id, "snake", "Snake"),
        await steps_mod.ensure_job(test_user_id, "snake", "Snake"),
    ]
    assert ids == [jid, jid, jid]
    assert len(await _jobs_for(test_user_id)) == 1


async def test_a_later_turn_reuses_the_apps_existing_card(test_user_id):
    """An edit three turns later updates the card the app already has — it
    does not adopt that turn's unrelated job and it does not open a new one."""
    first = await steps_mod.ensure_job(test_user_id, "snake", "Snake")

    reg = _created_job_registry()
    del reg[:]
    other = await _model_created_job(test_user_id, "Researching something else")

    again = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    assert again == first != other

    async with async_session_maker() as db:
        untouched = await db.get(BuildJob, other)
    assert untouched.job_type == "agent_task" and untouched.app_id is None


async def test_a_job_that_already_drives_an_app_is_not_stolen(test_user_id):
    """Two apps in one turn: the second must not hijack the first's card."""
    first = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    _created_job_registry().append(first)

    second = await steps_mod.ensure_job(test_user_id, "tetris", "Tetris")
    assert second != first
    assert len(await _jobs_for(test_user_id)) == 2


async def test_a_turn_tracking_two_things_is_left_alone(test_user_id):
    """"Build me a snake game and research X" opens two jobs. Guessing which
    one is the build is worse than a second card, so the pipeline opens its
    own and neither of the model's is retitled."""
    a = await _model_created_job(test_user_id, "Researching competitors")
    b = await _model_created_job(test_user_id, "Building your app")

    got = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    assert got not in (a, b)
    assert len(await _jobs_for(test_user_id)) == 3

    async with async_session_maker() as db:
        for jid in (a, b):
            job = await db.get(BuildJob, jid)
            assert job.job_type == "agent_task" and job.app_id is None


async def test_a_job_with_recorded_progress_is_not_overwritten(test_user_id):
    """Replacing the steps of a job the model has been ticking off would
    erase work the user watched happen."""
    jid = await _model_created_job(test_user_id, "Researching competitors")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, jid)
        steps = json.loads(job.steps_json)
        steps[0]["status"] = "done"
        job.steps_json = json.dumps(steps)
        await db.commit()

    got = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    assert got != jid

    async with async_session_maker() as db:
        untouched = await db.get(BuildJob, jid)
    assert untouched.title == "Researching competitors"
    assert json.loads(untouched.steps_json)[0]["status"] == "done"


async def test_no_turn_job_means_the_pipeline_opens_its_own(test_user_id):
    """The ordinary case — the model went straight to the tools."""
    jid = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    assert jid
    async with async_session_maker() as db:
        job = await db.get(BuildJob, jid)
    assert job.job_type == "auto_builder" and job.title == "Build: Snake"
    assert (job.config_json or {}).get("adopted_by") is None


async def test_a_finished_job_is_never_adopted(test_user_id):
    """A card the user has already seen close must not spring back to life as
    something else's progress bar."""
    jid = await _model_created_job(test_user_id)
    async with async_session_maker() as db:
        job = await db.get(BuildJob, jid)
        job.status = "completed"
        await db.commit()

    got = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    assert got != jid
    assert len(await _jobs_for(test_user_id)) == 2
