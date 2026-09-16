"""A durable voice task belongs to the DAY, not to the socket that started it.

`voice_tasks.list()` filtered on `BuildJob.conversation_id == session_id`, and
the voice session id was minted per SOCKET — so any reconnect, any second call,
and any device switch made the work the user started a minute ago vanish from
the resume snapshot while it was still running, and `voice_task status/steer/
cancel` answered "This task is not available in this conversation".

The task's Conversation already knows its day. Scope by that instead, and carry
the scope on the wire (`scope_key`) so the relay can widen its own check without
guessing. Both halves degrade: a task submitted before the day stamp existed, or
an agent image that does not send `scope_key`, keeps behaving exactly as it does
today.
"""
from __future__ import annotations

import uuid
from datetime import date as _date

import pytest

from app.agent.voice_tasks import SOURCE_KIND, VoiceTaskService
from app.db import async_session_maker
from app.db.models import BuildJob, Conversation, DayChat, User


async def _seed(day_id: str, *, n_convs: int = 1):
    user_id = str(uuid.uuid4())
    conv_ids = [str(uuid.uuid4()) for _ in range(n_convs)]
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"rebind-{user_id[:8]}@example.test",
            hashed_password="x", name="Rebind Test",
        ))
        db.add(DayChat(id=day_id, user_id=user_id, local_date=_date(2026, 9, 15)))
        for cid in conv_ids:
            db.add(Conversation(
                id=cid, user_id=user_id, title="Voice", channel="voice",
                day_chat_id=day_id, message_count=0, total_tokens=0,
            ))
        await db.commit()
    return user_id, conv_ids


async def _seed_job(user_id: str, conversation_id: str, *, day_chat_id: str | None):
    task_id = str(uuid.uuid4())
    state = {
        "request_id": task_id, "fingerprint": "seed",
        "original_prompt": "Do the work", "execution_prompt": "Do the work",
        "run_number": 0, "generation": 0, "steering": [], "progress": [],
        "sources": [], "artifacts": [], "pending_actions": [],
        "resolved_operations": [],
    }
    if day_chat_id is not None:
        state["day_chat_id"] = day_chat_id
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=task_id, user_id=user_id, title="Do the work", prompt="Do the work",
            job_type="agent_task", source_kind=SOURCE_KIND,
            source_id=conversation_id, conversation_id=conversation_id,
            status="running", config_json={"voice": state},
            state_revision=1, layer=0, steps_json="[]",
        ))
        await db.commit()
    return task_id


def _service():
    class _NeverRunner:
        async def run(self, **_kw):  # pragma: no cover
            raise AssertionError("no work should be claimed here")

    svc = VoiceTaskService(_NeverRunner(), session_maker=async_session_maker)
    # `list()` awaits start(); the supervisor is not what is under test.
    svc.start = _noop_start.__get__(svc, VoiceTaskService)
    return svc


async def _noop_start(self):
    return None


@pytest.mark.asyncio
async def test_a_task_started_on_another_socket_is_still_listed():
    """THE DEFECT. Two Conversations, one day — a reconnect makes the second."""
    day = str(uuid.uuid4())
    user_id, (first, second) = await _seed(day, n_convs=2)
    task_id = await _seed_job(user_id, first, day_chat_id=day)

    out = await _service().list(user_id=user_id, session_id=second)
    assert [t["task_id"] for t in out["tasks"]] == [task_id]
    assert out["scope_key"] == day


@pytest.mark.asyncio
async def test_the_scope_key_is_on_the_wire():
    day = str(uuid.uuid4())
    user_id, (conv,) = await _seed(day)
    await _seed_job(user_id, conv, day_chat_id=day)

    out = await _service().list(user_id=user_id, session_id=conv)
    assert out["tasks"][0]["scope_key"] == day
    # …and the session id is still there, because an older client compares that.
    assert out["tasks"][0]["session_id"] == conv


@pytest.mark.asyncio
async def test_a_task_from_another_day_is_not_listed():
    """Widening the scope must stop at the day; a week of tasks resuming into
    one call is a different bug."""
    other_day = str(uuid.uuid4())
    day = str(uuid.uuid4())
    user_id, (conv,) = await _seed(day)
    async with async_session_maker() as db:
        other_conv = str(uuid.uuid4())
        db.add(DayChat(id=other_day, user_id=user_id, local_date=_date(2026, 9, 14)))
        db.add(Conversation(
            id=other_conv, user_id=user_id, title="Voice", channel="voice",
            day_chat_id=other_day, message_count=0, total_tokens=0,
        ))
        await db.commit()
    await _seed_job(user_id, other_conv, day_chat_id=other_day)

    out = await _service().list(user_id=user_id, session_id=conv)
    assert out["tasks"] == []


@pytest.mark.asyncio
async def test_another_users_task_is_never_listed():
    """The only way widening the scope could leak: a second account whose
    Conversation carries the same day id. A DayChat belongs to one user, so this
    is a corrupted row rather than a reachable state — which is exactly why the
    user filter has to hold on BOTH sides of the join rather than be implied."""
    day = str(uuid.uuid4())
    mine_user, (mine,) = await _seed(day)

    theirs_user = str(uuid.uuid4())
    theirs = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=theirs_user, email=f"rebind-{theirs_user[:8]}@example.test",
            hashed_password="x", name="Other",
        ))
        db.add(Conversation(
            id=theirs, user_id=theirs_user, title="Voice", channel="voice",
            day_chat_id=day, message_count=0, total_tokens=0,
        ))
        await db.commit()
    await _seed_job(theirs_user, theirs, day_chat_id=day)

    out = await _service().list(user_id=mine_user, session_id=mine)
    assert out["tasks"] == []


@pytest.mark.asyncio
async def test_a_task_with_no_day_stamp_still_resolves_on_its_own_socket():
    """Rows written before the stamp existed keep working — the fallback is the
    exact pre-R46 predicate."""
    day = str(uuid.uuid4())
    user_id, (conv,) = await _seed(day)
    task_id = await _seed_job(user_id, conv, day_chat_id=None)

    out = await _service().list(user_id=user_id, session_id=conv)
    assert [t["task_id"] for t in out["tasks"]] == [task_id]
    assert out["tasks"][0]["scope_key"] == conv, "falls back to the conversation id"


# ── The relay's half ───────────────────────────────────────────────────────

def test_the_relay_widens_by_scope_key_and_never_narrows():
    from app.api.voice_task_relay import VoiceTaskRelay

    relay = VoiceTaskRelay(None, None, None, None)
    # Before any sync the relay knows no scope: the session check is the floor.
    assert relay._in_scope({"session_id": "s1"}, "s1") is True
    assert relay._in_scope({"session_id": "s0", "scope_key": "day-1"}, "s1") is False

    relay._scope = "day-1"
    assert relay._in_scope({"session_id": "s0", "scope_key": "day-1"}, "s1") is True
    assert relay._in_scope({"session_id": "s0", "scope_key": "day-2"}, "s1") is False
    # An agent image that sends no scope_key is unaffected in both directions.
    assert relay._in_scope({"session_id": "s1"}, "s1") is True
    assert relay._in_scope({"session_id": "s0"}, "s1") is False


@pytest.mark.asyncio
async def test_the_relay_LEARNS_the_scope_from_the_list_response():
    """The producer, not just the predicate.

    `_in_scope` was tested by hand-setting `relay._scope`; the single line that
    POPULATES it (`sync`'s `if isinstance(result.get("scope_key"), str) …`) was
    executed by nothing, so deleting it left every voice test green — and with
    `_scope` permanently None `_in_scope` degrades to raw session equality,
    i.e. straight back to "This task is not available in this conversation" for
    work the user started two minutes ago.
    """
    from app.api.voice_task_relay import VoiceTaskRelay

    async def _request(method, path, *, body=None, params=None):
        return {
            "tasks": [{
                "task_id": "t1", "session_id": "socket-0", "scope_key": "day-1",
                "revision": 1, "generation": 0, "status": "running",
                "delivery_revision": 0, "receipt_revision": 0, "spoken_revision": 0,
            }],
            "scope_key": "day-1",
        }

    async def _emit(_frame):
        return None

    async def _session():
        return "socket-1"  # a RECONNECT: a different conversation, same day

    class _Coordinator:
        """`_context` pushes the restored task ids into the model session."""

        def __init__(self):
            self.events: list = []

        async def send_event(self, event):
            self.events.append(event)

    relay = VoiceTaskRelay(_request, _emit, _Coordinator(), _session)
    await relay.sync()

    assert relay._scope == "day-1", "the scope on the wire was never ingested"
    assert "t1" in relay.tasks, (
        "a task started on the previous socket was filtered out of its own day"
    )


@pytest.mark.asyncio
async def test_a_conversation_with_no_day_stamp_does_not_widen_to_every_task():
    """The fallback branch of `list()`'s day scoping. Its failure mode is a
    cross-day LEAK, not a missing row — replacing the else body with `pass`
    makes a day-less conversation match every voice task the user owns, on
    every day — so it is silent in exactly the direction that matters."""
    day = str(uuid.uuid4())
    user_id, (dated,) = await _seed(day)
    await _seed_job(user_id, dated, day_chat_id=day)

    dayless = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(
            id=dayless, user_id=user_id, title="Voice", channel="voice",
            day_chat_id=None, message_count=0, total_tokens=0,
        ))
        await db.commit()
    mine = await _seed_job(user_id, dayless, day_chat_id=None)

    out = await _service().list(user_id=user_id, session_id=dayless)
    assert [t["task_id"] for t in out["tasks"]] == [mine], (
        "a conversation with no day matched tasks from other conversations"
    )
    assert out["scope_key"] == dayless, "falls back to the conversation id"
