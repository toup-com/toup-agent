"""Every distinct voice failure needs a name of its own.

The phone deliberately renders none of the server's error text — wording is the
client's job, and it keys off a `code`. That is right, but it means `relay_error`
is ONE string standing in for a dozen unrelated causes: a dead agent container, a
422 from an empty tool call, an unbound local. On 2026-08-20 voice was down for
every real request and the only server-side evidence was a single WARNING line;
there was no way to tell how many users, how often, or which cause.

`_vcount` gives each cause a greppable `[VOICE_METRIC]` line and a process-local
counter. The names ARE the taxonomy.
"""
from __future__ import annotations

import asyncio
import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from test_voice_turn_survives import (  # noqa: E402,F401
    FakeClientWS, FakeOpenAIWS, _function_call, relay,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def clean_counters():
    from app.api.ws_realtime import _VOICE_COUNTERS
    _VOICE_COUNTERS.clear()
    yield
    _VOICE_COUNTERS.clear()


def test_a_counter_never_raises():
    """Instrumentation that can break a call is worse than none."""
    from app.api.ws_realtime import _vcount

    class Nasty:
        def __str__(self):
            raise RuntimeError("boom")

    _vcount("x", None, detail=Nasty())  # must not raise
    _vcount("y", "user-1")
    from app.api.ws_realtime import voice_counter_snapshot
    assert voice_counter_snapshot()["y"] == 1


async def test_a_healthy_session_records_open_and_turn_outcomes(relay, monkeypatch):
    from app.api import ws_realtime as rt
    from app.api.ws_realtime import voice_counter_snapshot

    async def _think(user_id, task, session_id, relay=None):
        return ("Found it.", "gpt-5.6-terra")
    monkeypatch.setattr(rt, "_think", _think)

    client = FakeClientWS()
    openai_ws = FakeOpenAIWS([
        _function_call("think", {"task": "search"}),
        {"type": "response.done", "response": {"output": [
            {"type": "message", "content": [
                {"type": "audio", "transcript": "Here you go."}]},
        ]}},
    ])

    async def _connect(*a, **kw):
        return openai_ws
    monkeypatch.setattr(rt.websockets, "connect", _connect)
    await asyncio.wait_for(
        rt.realtime_voice_ws(client, token="tok", session_id=None, onboarding=False),
        timeout=20,
    )

    snap = voice_counter_snapshot()
    assert snap.get("session_open_attempt") == 1
    assert snap.get("session_open_ok") == 1
    assert snap.get("voice_turn_completed") == 1
    assert not snap.get("voice_turn_zero_tool_calls"), (
        "a turn that DID call a tool must not be counted as a silent one"
    )
    assert not snap.get("relay_loop_died")


async def test_a_spoken_turn_with_no_tool_call_is_counted(relay, monkeypatch):
    """The quiet failure: the model answers from its own head — no search, no
    sources — and every other signal looks healthy."""
    from app.api import ws_realtime as rt
    from app.api.ws_realtime import voice_counter_snapshot

    client = FakeClientWS()
    openai_ws = FakeOpenAIWS([
        {"type": "response.done", "response": {"output": [
            {"type": "message", "content": [
                {"type": "audio", "transcript": "I think it's probably X."}]},
        ]}},
    ])

    async def _connect(*a, **kw):
        return openai_ws
    monkeypatch.setattr(rt.websockets, "connect", _connect)
    await asyncio.wait_for(
        rt.realtime_voice_ws(client, token="tok", session_id=None, onboarding=False),
        timeout=20,
    )

    assert voice_counter_snapshot().get("voice_turn_zero_tool_calls") == 1


async def test_a_failed_auth_is_counted_separately_from_a_failed_open(monkeypatch):
    from app.api import ws_realtime as rt
    from app.api import _ws_auth_helpers as auth
    from app.api.ws_realtime import voice_counter_snapshot

    async def _accept(ws):
        return "tok"
    monkeypatch.setattr(auth, "accept_with_subprotocol_auth", _accept)

    async def _no_user(token):
        return None
    monkeypatch.setattr(rt, "_authenticate_ws", _no_user)

    async def _safe_close(ws, code=None, message=None):
        return None
    monkeypatch.setattr(auth, "safe_send_close_ws", _safe_close)

    class NoAuthClient(FakeClientWS):
        # The endpoint's last resort is an inline auth message; answer it with
        # something unparseable so the 10s wait does not dominate the test.
        async def receive_text(self):
            return "not json"

    client = NoAuthClient()
    await asyncio.wait_for(
        rt.realtime_voice_ws(client, token="bad", session_id=None, onboarding=False),
        timeout=10,
    )

    snap = voice_counter_snapshot()
    assert snap.get("session_open_failed_auth") == 1
    assert not snap.get("session_open_attempt"), (
        "an unauthenticated socket never became a session"
    )


async def test_the_client_can_report_a_failure_the_server_cannot_see(relay, monkeypatch):
    """A mic that never started is invisible from here — the call looks open
    and silent. The client says so; we count it."""
    from app.api import ws_realtime as rt
    from app.api.ws_realtime import voice_counter_snapshot

    class ReportingClient(FakeClientWS):
        def __init__(self):
            super().__init__()
            self._queue = [
                '{"type":"client_event","event":"audio_start_failed",'
                '"detail":"Could not start the audio engine"}',
            ]

        async def receive_text(self):
            if self._queue:
                return self._queue.pop(0)
            await asyncio.Event().wait()

    client = ReportingClient()
    openai_ws = FakeOpenAIWS([{"type": "session.created"}], tail_delay=0.2)

    async def _connect(*a, **kw):
        return openai_ws
    monkeypatch.setattr(rt.websockets, "connect", _connect)
    await asyncio.wait_for(
        rt.realtime_voice_ws(client, token="tok", session_id=None, onboarding=False),
        timeout=20,
    )

    assert voice_counter_snapshot().get("client_audio_start_failed") == 1
