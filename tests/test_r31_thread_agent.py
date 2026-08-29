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
async def test_a_thread_question_runs_the_agent_loop_on_its_own_channel(monkeypatch):
    """Round 33 item 8: a thread question uses the MAIN CHAT's tool loop.

    R31 answered it with a toolless completion, and its "needs new
    reading" branch resolved a read tool from the capability registry's
    `source_tool` — which Gmail and Slack do not declare. So "give me my
    last five gmail" was answered "I could not read Gmail" in the thread
    and answered correctly in the main chat a minute later, on the same
    account. This pins the three things that make the two agree: the
    runner is used, the channel is `automation_thread` (which is what
    withholds the writers), and NEITHER side of the turn is persisted as
    a day-chat row.
    """
    from app.agent.automations import thread_agent, ledger

    seen = {}

    class _FakeRunner:
        async def run(self, **kw):
            seen.update(kw)
            chunk = kw.get("on_text_chunk")
            if chunk:
                await chunk("Here are your five latest emails.")

            class _R:
                text = "Here are your five latest emails."
            return _R()

    monkeypatch.setattr(thread_agent, "_runner", lambda: _FakeRunner())
    uid, aid = await _mk()

    async with async_session_maker() as db:
        automation = await db.get(Automation, aid)
        thread = await ledger.ensure_thread(db, user_id=uid, automation_id=aid)
        await db.commit()
        turn = await thread_agent.answer_in_thread(
            db, automation=automation, thread=thread,
            user_text="Give me my last five gmail",
        )

    assert turn and turn.get("kind") == "agent", turn
    assert turn.get("text") == "Here are your five latest emails."
    assert seen.get("channel") == "automation_thread", seen.get("channel")
    assert seen.get("save_user_message") is False
    assert seen.get("save_assistant_message") is False
    assert seen.get("disable_post_processing") is True


@pytest.mark.asyncio
async def test_the_thread_channel_withholds_every_writer():
    """The bound is the CHANNEL, not the absence of tools. A thread turn
    may read and answer; it may not defer work, write memory, or mutate
    the user's schedule."""
    from app.agent.prompt_profile import disabled_tools_for_channel

    off = disabled_tools_for_channel("automation_thread")
    for name in ("create_job", "update_job", "spawn", "start_mission",
                 "memory_store", "routines__create", "routines__remind",
                 "triggers__create"):
        assert name in off, name
    # …and it keeps the reads, which is the whole point of the change.
    assert "web_search" not in off
    assert "gmail__list_messages" not in off


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


@pytest.mark.asyncio
async def test_the_thread_channel_is_named_rather_than_clamped():
    """`automation_thread` is a REGISTERED channel, and it keeps the deny.

    Before this, the string existed only in the agent: `mcp_auth` had
    never heard of it, so every connector call a thread made hit
    `_UNKNOWN_CHANNEL_CLAMP` and arrived at the dispatcher as
    "background". The POLICY that produced was right — reads pass,
    mutating connectors deny — but it was right by accident, it wrote a
    `logger.warning` on every tool call a thread ever made, and the day
    the clamp target moves the thread's permissions move with it,
    silently. Both halves are asserted because either one alone is the
    bug: named-but-not-denied would hand an unattended-shaped surface
    the write tools, and denied-but-not-named is where we started.
    """
    from app.mcp_auth import _KNOWN_CHANNELS, _UNKNOWN_CHANNEL_CLAMP
    from app.services.connector_dispatcher import (
        _MUTATES_UNATTENDED_DENY_CHANNELS,
    )

    assert "automation_thread" in _KNOWN_CHANNELS
    assert "automation_thread" in _MUTATES_UNATTENDED_DENY_CHANNELS
    # The clamp target is still in the deny set, so the two agree and a
    # channel we forget to register stays fail-closed.
    assert _UNKNOWN_CHANNEL_CLAMP in _MUTATES_UNATTENDED_DENY_CHANNELS


@pytest.mark.asyncio
async def test_a_deleted_question_run_stops_speaking_for_the_automation():
    """R31's question runs wrote failures that never happened.

    `open_question_run` called every account with an empty tool input,
    and for gmail — whose manifest declares no `source_tool` — it never
    called anything: `_default_read_tool` returned None and the ledger
    got "Could not reach Gmail / I could not tell why". Deleting that
    code stops NEW ones; the rows already written are durable, and the
    grounding replays the newest turns verbatim, so the thread would go
    on answering "the last run did not read any account data" with a
    fabricated read for as long as they stayed in the window.

    A REAL run's failure is kept — that is the thread's whole job.
    """
    from app.agent.automations import thread_agent, ledger
    from app.db.models import BuildJob

    uid, aid = await _mk()
    q_id, real_id = str(uuid.uuid4()), str(uuid.uuid4())

    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid, automation_id=aid)
        for jid, kind in ((q_id, "question"), (real_id, "scheduled")):
            db.add(BuildJob(
                id=jid, user_id=uid, title="Morning work brief", prompt="",
                job_type="automation_run", status="completed",
                config_json={"run_kind": kind},
            ))
        await db.commit()

        async def _tool(run_id: str, action: str):
            await ledger.append_turn(
                db, user_id=uid, thread=thread, run_id=run_id, kind="tool",
                payload={"account_id": "gmail", "tool_kind": "read",
                         "action": action, "detail": "I could not tell why",
                         "ok": False, "ms": 3, "steps": [], "items": [],
                         "write_ids": [], "rest": ""},
                broadcast=False,
            )

        await _tool(q_id, "Could not reach Gmail")
        await _tool(real_id, "Could not reach Gmail")
        await ledger.append_turn(
            db, user_id=uid, thread=thread, run_id=q_id, kind="agent",
            payload={"text": "Looking at your 4 accounts now."},
            broadcast=False,
        )
        await db.commit()

        legacy = await thread_agent._legacy_question_run_ids(
            db, [{"run_id": q_id}, {"run_id": real_id}],
        )
        assert legacy == {q_id}, legacy

        turns = await thread_agent._recent_turns(db, thread.id)

    runs_of = lambda k: [t.get("run_id") for t in turns if t.get("kind") == k]
    assert runs_of("tool") == [real_id], runs_of("tool")
    # The sentence the user actually saw is not rewritten out from under
    # them — only the claim about an account goes.
    assert q_id in runs_of("agent"), runs_of("agent")

    automation_name = "Boss email -> draft reply"
    grounding = thread_agent._grounding(
        type("A", (), {"name": automation_name, "id": aid,
                       "user_id": uid, "rules_json": "[]",
                       "spec_json": "{}"})(),
        turns, [],
    )
    assert grounding.count("Could not reach Gmail") == 1, grounding


def _async(value):
    async def _run():
        return value
    return _run()
