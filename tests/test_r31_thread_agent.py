# agent-mode: automations/automation_threads/_turns/build_jobs are AGENT_ONLY.
"""R31 §4.9 — the thread answers for itself, executed.

`thread_agent.py` is 418 lines implementing the headline of §4.1, and
before this file NOTHING referenced it. Both of its two entry points
raised on their first line of real work, so the feature could never have
worked for anybody:

  open_question_run  TypeError: TaskSpec.__init__() got an unexpected
                     keyword argument 'title'
  answer_in_thread   KeyError: slice(None, 20, None)

The route catches both and writes "Something went wrong answering that.
Ask me again and I will try once more." into the thread — which is what
the founder photographed on 2026-08-26 after pressing `Run it now`.

Neither is subtle. Both are the kind of mistake that a single execution
finds and that no amount of reading does, which is the whole argument
for this file existing.
"""

import json
import uuid

import pytest

from app.db.database import async_session_maker
from app.db.models import Automation, User


async def _mk(automation_steps=True) -> tuple[str, str]:
    uid = str(uuid.uuid4())
    aid = str(uuid.uuid4())
    spec = {
        "version": 2, "name": "Boss email -> draft reply", "mode": "auto",
        "trigger": {"sources": [{"id": "t", "connector_id": "gmail",
                                 "event": "message_received", "mode": "poll"}]},
        "steps": ([{"id": "read", "connector_id": "gmail",
                    "tool": "gmail__search", "params": {},
                    "on_error": "continue"}] if automation_steps else []),
    }
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Thread"))
        db.add(Automation(
            id=aid, user_id=uid, name="Boss email -> draft reply",
            status="armed", spec_json=json.dumps(spec), trigger_mode="poll",
        ))
        await db.commit()
    return uid, aid


@pytest.mark.asyncio
async def test_a_fresh_read_question_opens_a_run_instead_of_raising(monkeypatch):
    """`Run it now` → `needs_fresh_read` → `open_question_run`.

    It built its job as `create_job(db, TaskSpec(title=..., job_type=...))`.
    `create_job` is keyword-only and takes no `db`; `job_type` and `title`
    are ITS arguments and TaskSpec has neither — wrong on three counts, so
    this raised before reaching the model every single time.
    """
    from app.agent.automations import thread_agent, ledger

    monkeypatch.setattr(thread_agent, "_complete",
                        lambda *a, **k: _async("Checked your mail. Nothing new."))
    uid, aid = await _mk()
    assert thread_agent.needs_fresh_read("Run it now") is True

    async with async_session_maker() as db:
        automation = await db.get(Automation, aid)
        thread = await ledger.ensure_thread(db, user_id=uid, automation_id=aid)
        await db.commit()
        run_id = await thread_agent.open_question_run(
            db, automation=automation, thread=thread, user_text="Run it now",
        )
    assert run_id, "a fresh-read question produced no run"


@pytest.mark.asyncio
async def test_a_past_question_is_answered_instead_of_raising(monkeypatch):
    """A past-tense question → `answer_in_thread` → `_grounding`.

    `memory_v2_service.recall` answers {"facts": [...], "episodes": [...]}
    — a DICT — and `_facts_for` returned it whole while its annotation said
    list[dict]. `_grounding` then sliced it: `facts[:20]` on a dict is
    `KeyError: slice(None, 20, None)`, on every past-tense question.
    """
    from app.agent.automations import thread_agent, ledger

    monkeypatch.setattr(thread_agent, "_complete",
                        lambda *a, **k: _async("Yesterday I checked your mail."))
    uid, aid = await _mk()
    assert thread_agent.needs_fresh_read("what did you do yesterday") is False

    async with async_session_maker() as db:
        automation = await db.get(Automation, aid)
        thread = await ledger.ensure_thread(db, user_id=uid, automation_id=aid)
        await db.commit()
        turn = await thread_agent.answer_in_thread(
            db, automation=automation, thread=thread,
            user_text="what did you do yesterday",
        )
    assert turn and turn.get("kind") == "agent", turn


@pytest.mark.asyncio
async def test_facts_for_always_answers_a_list(monkeypatch):
    """The narrow pin for the defect above: whatever `recall` hands back,
    `_facts_for` answers something `_grounding` can slice."""
    from app.agent.automations import thread_agent

    uid, aid = await _mk()
    async with async_session_maker() as db:
        automation = await db.get(Automation, aid)

        for shape in ({"facts": [{"text": "a"}], "episodes": []},
                      {"facts": []},
                      {},
                      [{"text": "b"}],
                      None):
            monkeypatch.setattr(
                "app.services.memory_v2_service.recall",
                lambda *a, _s=shape, **k: _async(_s),
            )
            facts = await thread_agent._facts_for(db, automation)
            assert isinstance(facts, list), (shape, facts)
            # The slice `_grounding` performs must not raise.
            assert facts[:20] == facts[:20]


def _async(value):
    async def _run():
        return value
    return _run()
