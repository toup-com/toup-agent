"""A voice turn that calls a tool must answer, and must not end the session.

The 2026-08-20 P0. Production, four consecutive sessions on 871bac24:

    [REALTIME] Function call: think({'task': 'سرچ کن ببین ...'})
    [REALTIME] openai_to_client error: cannot access local variable
               'turn_tool_events' where it is not associated with a value
    [REALTIME] Session ended for user 871bac24

`turn_tool_events` is created in `realtime_voice_ws` and rebound inside
`openai_to_client` (``turn_tool_events = []``) with no ``nonlocal``, so Python
binds it local for the whole of that function and every READ of it — the
assistant persist, and the `think` dispatch — raises UnboundLocalError. The
`think` read is not inside a try, so it unwound the relay loop: the user saw
their words transcribed, then a generic error, and the agent never searched,
never answered, never spoke. Since V2 gates voice down to `think` +
`navigate_to` + `play_media`, that is EVERY real request.

Two things are pinned here, and the second is the one that matters in a year:

1. The binding itself — a think turn completes and the session stays up.
2. The blast radius — a tool that raises must degrade to a failed tool RESULT
   the model can talk about, never a dead relay. The dispatch block had no
   try/except at all, so any raise from `_think` / `_play_media_direct` /
   `_execute_tool` took the whole call down with it.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

pytestmark = pytest.mark.asyncio


# ── Fakes ────────────────────────────────────────────────────────────────

class FakeClientWS:
    """The phone. Collects what the relay sends; never speaks."""

    def __init__(self):
        self.sent: list = []
        self.closed = False

    async def send_json(self, payload):
        if self.closed:
            raise RuntimeError("client gone")
        self.sent.append(payload)

    async def receive_text(self):
        # The user is listening, not typing. Block until the OpenAI side
        # finishes and the endpoint cancels us.
        await asyncio.Event().wait()

    async def close(self, code=1000):
        self.closed = True

    def frames(self, ftype):
        return [f for f in self.sent if f.get("type") == ftype]


class FakeOpenAIWS:
    """The Realtime API. Yields a scripted event list, then ends the stream."""

    def __init__(self, events, tail_delay: float = 0.0):
        self._events = list(events)
        self._tail_delay = tail_delay
        self.sent: list = []
        self.closed = False

    def __aiter__(self):
        async def gen():
            for ev in self._events:
                yield json.dumps(ev)
                await asyncio.sleep(0)
            # Ending the stream tears the session down (FIRST_COMPLETED). A
            # test that asserts on something the CLIENT loop does needs that
            # loop to get a turn first.
            if self._tail_delay:
                await asyncio.sleep(self._tail_delay)
        return gen()

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def close(self):
        self.closed = True


def _function_call(name, arguments, call_id="call_1"):
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "name": name,
            "call_id": call_id,
            "arguments": json.dumps(arguments),
        },
    }


def _in_voice_runtime(task) -> bool:
    """True if a task's coroutine lives in the voice runtime (ws_realtime or
    voice_tasks) — the files whose fire-and-forget work outlives a driven
    socket and must not touch the DB after a test's schema is dropped."""
    coro = task.get_coro()
    fn = getattr(coro, "cr_code", None)
    path = getattr(fn, "co_filename", "") if fn is not None else ""
    return "ws_realtime" in path or "voice_tasks" in path


@pytest.fixture
async def relay(monkeypatch):
    """Everything outside the relay loop stubbed; the loop itself is real."""
    from app.api import ws_realtime as rt
    from app.api import _ws_auth_helpers as auth

    async def _accept(ws):
        return "tok"
    monkeypatch.setattr(auth, "accept_with_subprotocol_auth", _accept)

    async def _auth(token):
        return "user-1"
    monkeypatch.setattr(rt, "_authenticate_ws", _auth)

    async def _key(user_id):
        # is_byok=True → no credit pre-flight, no metering task.
        return ("sk-test", True)
    monkeypatch.setattr(rt, "_get_user_openai_key_ex", _key)

    async def _sess(user_id, session_id):
        return "sess-1"
    monkeypatch.setattr(rt, "_get_or_create_voice_session", _sess)

    async def _instr(user_id, onboarding=False, now_utc=None):
        return "You are a voice agent."
    monkeypatch.setattr(rt, "build_realtime_instructions", _instr)

    async def _lang(user_id):
        return None
    monkeypatch.setattr(rt, "resolve_voice_language", _lang)
    monkeypatch.setattr(rt, "_cached_voice_language", lambda uid: None)

    async def _ensure(user_id):
        return None
    monkeypatch.setattr(rt, "_ensure_vps_user", _ensure)

    async def _save(*a, **kw):
        return None
    monkeypatch.setattr(rt, "_save_voice_messages", _save)

    yield rt

    # The real endpoint spawns fire-and-forget tasks on socket close — chiefly
    # `_defer_voice_la_end`, which sleeps a 6 s grace and then touches the DB to
    # end the voice card (a deliberate cross-request backstop; see the comment
    # on `_voice_session_owner`). The endpoint returns before they run, so under
    # CI's shared in-memory DB one wakes during teardown and hits a database
    # being dropped ("Cannot operate on a closed database"). Cancelling is not
    # enough on its own — a bare `.cancel()` only SCHEDULES cancellation, and an
    # already-started DB call still races drop_db — so we AWAIT every cancelled
    # task here, before conftest's autouse `_reset_database` (which tears down
    # after this module-level fixture) drops the schema. Production keeps the
    # grace; this only cleans up after driving the real socket. Same idea as
    # test_signup_trace_wiring draining its discovery loops.
    import asyncio as _asyncio
    _leaked = [
        _t for _t in _asyncio.all_tasks()
        if not _t.done() and _t is not _asyncio.current_task()
        and _in_voice_runtime(_t)
    ]
    for _t in _leaked:
        _t.cancel()
    if _leaked:
        await _asyncio.gather(*_leaked, return_exceptions=True)


async def _drive(rt, monkeypatch, events, tail_delay: float = 0.0):
    """Run the real endpoint against a scripted OpenAI event stream."""
    client = FakeClientWS()
    openai_ws = FakeOpenAIWS(events, tail_delay=tail_delay)

    async def _connect(*a, **kw):
        return openai_ws
    monkeypatch.setattr(rt.websockets, "connect", _connect)

    await asyncio.wait_for(
        rt.realtime_voice_ws(client, token="tok", session_id=None, onboarding=False),
        timeout=20,
    )
    return client, openai_ws


# ── 1. The P0 itself ─────────────────────────────────────────────────────

async def test_a_think_turn_completes_and_the_session_survives(relay, monkeypatch):
    rt = relay

    async def _think(user_id, task, session_id, relay=None, out=None):
        return ("Here is what I found.", "gpt-5.6-terra")
    monkeypatch.setattr(rt, "_think", _think)

    client, openai_ws = await _drive(rt, monkeypatch, [
        {"type": "session.created"},
        _function_call("think", {"task": "search what companies invest in AI"}),
    ])

    errors = client.frames("error")
    assert not errors, (
        f"a think turn killed the session: {errors} — this is the production "
        "UnboundLocalError on turn_tool_events"
    )

    done = client.frames("tool_call.completed")
    assert done, "the phone never saw the tool finish"
    assert done[0]["name"] == "think" and done[0]["ok"] is True

    # The answer has to reach OpenAI, or the agent stays mute.
    outputs = [s for s in openai_ws.sent
               if s.get("item", {}).get("type") == "function_call_output"]
    assert outputs and outputs[0]["item"]["output"] == "Here is what I found.", (
        "the tool result never went back to the model"
    )


# ── 2. The blast radius ──────────────────────────────────────────────────

async def test_a_raising_tool_degrades_to_a_result_not_a_dead_session(relay, monkeypatch):
    """A tool that throws must not take the call down with it."""
    rt = relay

    async def _boom(user_id, task, session_id, relay=None, out=None):
        raise RuntimeError("agent container replaced mid-turn")
    monkeypatch.setattr(rt, "_think", _boom)

    client, openai_ws = await _drive(rt, monkeypatch, [
        {"type": "session.created"},
        _function_call("think", {"task": "anything"}),
        {"type": "session.updated"},
    ])

    assert not client.frames("error"), (
        "one throwing tool ended the whole voice session"
    )
    done = client.frames("tool_call.completed")
    assert done and done[0]["ok"] is False, (
        "a failed tool must report failure, not vanish"
    )
    outputs = [s for s in openai_ws.sent
               if s.get("item", {}).get("type") == "function_call_output"]
    assert outputs, "the model was left waiting on a call that never returned"
    assert outputs[0]["item"]["output"].upper().startswith("ERROR")


async def test_the_relay_keeps_serving_after_a_tool_failure(relay, monkeypatch):
    """The turn after a failed tool still works — the loop is still alive."""
    rt = relay
    calls = []

    async def _flaky(user_id, task, session_id, relay=None, out=None):
        calls.append(task)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return ("Second time worked.", "gpt-5.6-terra")
    monkeypatch.setattr(rt, "_think", _flaky)

    client, openai_ws = await _drive(rt, monkeypatch, [
        _function_call("think", {"task": "first"}, call_id="c1"),
        _function_call("think", {"task": "second"}, call_id="c2"),
    ])

    assert len(calls) == 2, f"the relay died before the second turn: {calls}"
    done = client.frames("tool_call.completed")
    assert [d["ok"] for d in done] == [False, True]


# ── 3. What the turn PRODUCED reaches the row that persists it ────────────

def _spoken_done(text, response_id="resp_1"):
    return {
        "type": "response.done",
        "response": {
            "id": response_id, "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "audio", "transcript": text},
            ]}],
        },
    }


def _created(response_id="resp_1"):
    return {"type": "response.created", "response": {"id": response_id}}


def _transcript(text, item_id="item_1"):
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": text, "item_id": item_id,
    }


@pytest.fixture
def persisted(relay, monkeypatch):
    """Capture what the persistence worker hands `_save_voice_messages`.

    The middle hop of the attachment fix — `_think(out=…)` →
    `pending_attachments` → `_enqueue_persist` → `_save_voice_messages` — had no
    test: hop 1 (`ChatResponse`) and hop 3 (the request body) were both covered
    and deleting the two lines between them left every voice test green.
    """
    calls: list = []

    async def _save(*a, **kw):
        # The worker passes user_id/session_id/user_text/assistant_text
        # POSITIONALLY; everything else is keyword. Flattened so a test reads
        # one dict.
        rec = dict(kw)
        rec["user_text"] = a[2] if len(a) > 2 else ""
        rec["assistant_text"] = a[3] if len(a) > 3 else ""
        calls.append(rec)
        return None
    monkeypatch.setattr(relay, "_save_voice_messages", _save)
    return calls


async def test_a_file_made_during_a_call_rides_the_row_that_persists_the_reply(
    relay, monkeypatch, persisted,
):
    rt = relay

    async def _think(user_id, task, session_id, relay=None, out=None):
        # Exactly what `/internal/agent-turn` hands back for a save=False turn.
        out["attachments"] = [{"id": "a1", "filename": "report.pdf"}]
        out["app_artifact"] = {"slug": "budget"}
        return ("Here is the report.", "gpt-5.6-terra")
    monkeypatch.setattr(rt, "_think", _think)

    await _drive(rt, monkeypatch, [
        {"type": "session.created"},
        _function_call("think", {"task": "make me a report"}),
        # The function-call response.done comes FIRST and must not consume them.
        {"type": "response.done", "response": {
            "id": "resp_fn", "status": "completed",
            "output": [{"type": "function_call", "call_id": "call_1", "name": "think"}],
        }},
        _created("resp_1"),
        _spoken_done("Here is the report."),
    ], tail_delay=0.35)

    assert persisted, "the spoken reply was never persisted"
    kw = persisted[-1]
    assert kw["attachments"] and kw["attachments"][0]["id"] == "a1", (
        "the file the turn produced never reached the persist call — "
        "GET /api/files/{message_id}/{aid} has no row to authorize against"
    )
    assert kw["app_artifact"] == {"slug": "budget"}
    assert kw["assistant_text"] == "Here is the report."


async def test_a_second_spoken_turn_does_not_inherit_the_first_turns_files(
    relay, monkeypatch, persisted,
):
    """`pending_*` are per-turn. Left set, the NEXT reply's row is stamped with
    the previous turn's file and app — the documented `_last_media` class."""
    rt = relay

    async def _think(user_id, task, session_id, relay=None, out=None):
        out["attachments"] = [{"id": "a1"}]
        return ("Done.", "gpt-5.6-terra")
    monkeypatch.setattr(rt, "_think", _think)

    await _drive(rt, monkeypatch, [
        _function_call("think", {"task": "first"}, call_id="c1"),
        {"type": "response.done", "response": {
            "id": "resp_fn", "status": "completed",
            "output": [{"type": "function_call", "call_id": "c1", "name": "think"}],
        }},
        _created("resp_1"),
        _spoken_done("Done.", "resp_1"),
        _created("resp_2"),
        _spoken_done("Anything else?", "resp_2"),
    ], tail_delay=0.35)

    assert len(persisted) >= 2, persisted
    assert persisted[-1]["assistant_text"] == "Anything else?"
    assert not persisted[-1]["attachments"], (
        "the second reply inherited the first turn's file"
    )


# ── 4. The question is stamped before the answer it caused ────────────────

async def test_the_question_is_stamped_before_the_answer_even_when_it_arrives_after(
    relay, monkeypatch, persisted,
):
    """The two halves are enqueued from two INDEPENDENT provider events and
    realtime input transcription routinely finalises AFTER the reply. Two
    independent clock reads therefore inverted the pair — and the thread now
    sorts on `COALESCE(occurred_at, created_at)`, so the inversion is what the
    user sees. Scripted in the inverted order on purpose.

    The previous version of this assertion lived in `test_voice_turn_ordering`
    and handed `_save_voice_messages` both stamps itself, so it proved only
    that the request builder forwards two literals: gutting the real stamping
    left it green.
    """
    rt = relay

    await _drive(rt, monkeypatch, [
        {"type": "input_audio_buffer.speech_stopped"},
        _created("resp_1"),
        _spoken_done("Forty dollars."),
        _transcript("what is my balance"),
    ], tail_delay=0.35)

    by_role = {}
    for kw in persisted:
        if kw.get("user_text"):
            by_role["user"] = kw["user_occurred_at"]
        if kw.get("assistant_text"):
            by_role["assistant"] = kw["assistant_occurred_at"]
    assert "user" in by_role and "assistant" in by_role, persisted
    assert by_role["user"] < by_role["assistant"], (
        "the answer is stamped before the question it answers"
    )


async def test_two_utterances_stopping_before_either_transcript_keep_their_order(
    relay, monkeypatch, persisted,
):
    """A single `speech_stopped` slot was correct only while at most one
    utterance was outstanding. On a quick follow-up both stops land first, the
    second overwrites the first, the first transcript consumes it and empties
    the slot, and the SECOND question then falls back to `now` — after the
    reply it caused."""
    rt = relay

    await _drive(rt, monkeypatch, [
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "input_audio_buffer.speech_stopped"},
        _created("resp_1"),
        _spoken_done("Forty dollars.", "resp_1"),
        _transcript("what is my balance", "item_1"),
        _created("resp_2"),
        _spoken_done("Yes, twice.", "resp_2"),
        _transcript("and is that confirmed", "item_2"),
    ], tail_delay=0.45)

    users = [kw["user_occurred_at"] for kw in persisted if kw.get("user_text")]
    assistants = [kw["assistant_occurred_at"] for kw in persisted if kw.get("assistant_text")]
    assert len(users) == 2 and len(assistants) == 2, persisted
    assert users[1] < assistants[1], (
        "the second question sorts below the answer it caused"
    )
    assert users[0] <= users[1], "the two questions inverted against each other"


async def test_a_cancelled_reply_does_not_leave_its_files_for_the_next_one(
    relay, monkeypatch, persisted,
):
    """Barge-in is the ordinary case this file spends whole functions on. The
    clears used to live INSIDE the `full_text and response_visible and status
    != 'cancelled'` guard, so a barged-in spoken reply left `pending_*` set and
    the NEXT reply's row was stamped with the previous turn's file and app."""
    rt = relay

    async def _think(user_id, task, session_id, relay=None, out=None):
        out["attachments"] = [{"id": "a1"}]
        out["app_artifact"] = {"slug": "budget"}
        return ("Done.", "gpt-5.6-terra")
    monkeypatch.setattr(rt, "_think", _think)

    await _drive(rt, monkeypatch, [
        _function_call("think", {"task": "first"}, call_id="c1"),
        {"type": "response.done", "response": {
            "id": "resp_fn", "status": "completed",
            "output": [{"type": "function_call", "call_id": "c1", "name": "think"}],
        }},
        _created("resp_1"),
        # The user talked over it: a spoken response that ends `cancelled`.
        {"type": "response.done", "response": {
            "id": "resp_1", "status": "cancelled",
            "output": [{"type": "message", "content": [
                {"type": "audio", "transcript": "Do"},
            ]}],
        }},
        _created("resp_2"),
        _spoken_done("Anything else?", "resp_2"),
    ], tail_delay=0.35)

    spoken = [kw for kw in persisted if kw["assistant_text"]]
    assert spoken and spoken[-1]["assistant_text"] == "Anything else?", persisted
    assert not spoken[-1]["attachments"], (
        "the cancelled turn's file was stapled onto an unrelated reply"
    )
    assert not spoken[-1]["app_artifact"]


async def test_the_reply_is_stamped_after_the_question_even_at_the_same_instant(
    relay, monkeypatch, persisted,
):
    """With the clock frozen — a coarse clock, or two enqueues inside one tick
    — both halves read the same instant and `COALESCE(occurred_at, created_at)`
    would order them by id. The floor is what makes "the reply is authored
    after the question" true rather than usually true."""
    import datetime as _dt

    rt = relay
    fixed = _dt.datetime(2026, 9, 15, 12, 0, 0, tzinfo=_dt.timezone.utc)

    class _Frozen(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is not None else fixed.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return fixed.replace(tzinfo=None)

    monkeypatch.setattr(rt, "datetime", _Frozen)

    await _drive(rt, monkeypatch, [
        {"type": "input_audio_buffer.speech_stopped"},
        _transcript("what is my balance"),
        _created("resp_1"),
        _spoken_done("Forty dollars."),
    ], tail_delay=0.35)

    user = [kw["user_occurred_at"] for kw in persisted if kw["user_text"]]
    asst = [kw["assistant_occurred_at"] for kw in persisted if kw["assistant_text"]]
    assert user and asst, persisted
    assert asst[0] > user[0], "both halves carry the same stamp"
