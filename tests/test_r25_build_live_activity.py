"""Round 25 — the app build reaches the phone.

The app builder was the one long-running job with no phone surface of its own.
``grep -rn "notify"`` over the whole ``app_html`` package returned nothing: the
pipeline's entire external effect was a WebSocket frame, so a build that took
two minutes showed the user nothing at all once they left the app. No Dynamic
Island, no lock-screen card, no completion.

None of the machinery was missing. ``_notify_job_event`` already carries a
title, a step name, ``steps_done``/``steps_total``, a 0-100 ``progress`` and
the deep-link ids, and the Live Activity lane already renders all of it. It had
simply never been called from here.

There is one real trap, and it is why this is not a two-line change. A Live
Activity is addressed by ``job_mission_id(job_id, chat_id)``, which falls back
to the RAW JOB ID when there is no chat id — and a build job opened by this
pipeline had ``conversation_id = NULL``, because ``ensure_job`` builds its own
``TaskSpec``. A push addressed that way does not land on the conversation's
``chatjob:<sid>`` card at all; it opens a second, orphaned one. So the address
has to come from the turn's context, and the job row has to carry it too, or
every OTHER consumer that derives an address from the job (the interrupted-job
sweep's ``mission_failed``) keeps missing.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.agent.skills.builtins.app_html import steps as steps_mod
from app.agent.subagent_orchestrator import job_mission_id


@pytest.fixture
def pushes(monkeypatch) -> List[Dict[str, Any]]:
    """Every Live Activity event the build would have sent."""
    sent: List[Dict[str, Any]] = []

    async def _fake(**kwargs):
        sent.append(kwargs)

    import app.agent.subagent_orchestrator as orch
    monkeypatch.setattr(orch, "_notify_job_event", _fake)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(steps_mod, "_broadcast", _noop)
    return sent


@pytest.fixture
def in_a_conversation(monkeypatch):
    """A turn that has a session — the normal case, and the one where the
    card must be keyed to the conversation."""
    from app.agent import tool_executor as te
    token = te._SESSION_ID_CTX.set("sess-42")
    yield "sess-42"
    te._SESSION_ID_CTX.reset(token)


async def test_a_build_puts_a_card_on_the_phone(test_user_id, pushes):
    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create",
        status="running",
    )
    assert pushes, (
        "the build sent nothing to the phone — this is the whole of item 7"
    )
    first = pushes[0]
    assert first["kind"] == "mission_started"
    assert first["job_type"] == "auto_builder"
    assert first["steps_total"] >= 1
    assert first["progress"] is not None


async def test_the_card_is_keyed_to_the_conversation_not_the_job(
    test_user_id, pushes, in_a_conversation,
):
    """The trap. `job_mission_id` falls back to the raw job id with no chat
    id, which addresses a card nobody is looking at."""
    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create",
        status="running",
    )
    push = pushes[0]
    assert push["chat_id"] == in_a_conversation
    assert push["route"] == "chat", "the card must deep-link back to the chat"
    assert job_mission_id(job_id, push["chat_id"]) == f"chatjob:{in_a_conversation}"
    assert job_mission_id(job_id, push["chat_id"]) != job_id


async def test_the_job_row_carries_the_conversation_too(
    test_user_id, pushes, in_a_conversation,
):
    """Not only the push. Every other consumer derives the address from the
    ROW — the interrupted-job sweep pushes `mission_failed` with
    `chat_id=job.conversation_id` — so a NULL there misses the same card."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    job_id = await steps_mod.ensure_job(test_user_id, "tetris", "Tetris")
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
    assert job.conversation_id == in_a_conversation


async def test_only_the_first_transition_starts_the_card(test_user_id, pushes):
    """A second `mission_started` is dropped as an at-least-once duplicate,
    and a `progress` for a card that was never started updates nothing."""
    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    for step_type, status in (("create", "running"), ("create", "done"),
                              ("verify", "running")):
        await steps_mod.emit_step(
            user_id=test_user_id, job_id=job_id, step_type=step_type,
            status=status,
        )
    kinds = [p["kind"] for p in pushes]
    assert kinds[0] == "mission_started"
    assert kinds.count("mission_started") == 1, kinds
    assert set(kinds[1:]) == {"progress"}, kinds


async def test_the_island_says_fixing_it_during_a_repair(test_user_id, pushes):
    """The phone and the chat card must not disagree about what is happening.
    Both read the same headline."""
    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="done",
    )
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="verify",
        status="failed", detail="1 problem", recoverable=True,
    )
    assert pushes[-1]["step_name"] == steps_mod.FIXING_LABEL
    # And it is NOT a failure on the phone either, for the same reason it is
    # not red in the chat: the build is still going.
    assert pushes[-1]["kind"] == "progress"


async def test_the_percent_on_the_phone_is_the_percent_on_the_card(
    test_user_id, pushes,
):
    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    for t in ("create", "verify", "look"):
        await steps_mod.emit_step(
            user_id=test_user_id, job_id=job_id, step_type=t, status="done",
        )
    percents = [p["progress"] for p in pushes]
    assert percents == sorted(percents), f"the island's bar regressed: {percents}"
    last = pushes[-1]
    assert last["progress"] == steps_mod.percent_for(
        last["steps_done"], last["steps_total"]
    )


async def test_a_finished_build_reports_success_and_ends(test_user_id, pushes):
    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="done",
    )
    await steps_mod.finish_job(test_user_id, job_id)

    final = pushes[-1]
    assert final["kind"] == "mission_completed"
    assert final["progress"] == 100
    assert "ready to open" in (final.get("body") or "")


async def test_a_dead_build_reports_failure(test_user_id, pushes):
    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create",
        status="failed", detail="disk full",
    )
    await steps_mod.finish_job(test_user_id, job_id)
    assert pushes[-1]["kind"] == "mission_failed"


async def test_a_phone_outage_never_fails_a_build(test_user_id, monkeypatch):
    """The module is fail-open by contract. A push that raises must not
    propagate out of `emit_step` — a lock-screen card is never worth a build."""
    async def _boom(**_k):
        raise RuntimeError("APNs is down")

    import app.agent.subagent_orchestrator as orch
    monkeypatch.setattr(orch, "_notify_job_event", _boom)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(steps_mod, "_broadcast", _noop)

    job_id = await steps_mod.ensure_job(test_user_id, "snake", "Snake")
    await steps_mod.emit_step(
        user_id=test_user_id, job_id=job_id, step_type="create", status="done",
    )
    assert await steps_mod.finish_job(test_user_id, job_id) == "completed"
