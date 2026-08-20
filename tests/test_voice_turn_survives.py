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


@pytest.fixture
def relay(monkeypatch):
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

    return rt


async def _drive(rt, monkeypatch, events):
    """Run the real endpoint against a scripted OpenAI event stream."""
    client = FakeClientWS()
    openai_ws = FakeOpenAIWS(events)

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

    async def _think(user_id, task, session_id, relay=None):
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

    async def _boom(user_id, task, session_id, relay=None):
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

    async def _flaky(user_id, task, session_id, relay=None):
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
