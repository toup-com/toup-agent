"""The one-active-response rule, pinned.

The 2026-08-16 recording: a `think` continuation fired `response.create` into a
VAD-created response, OpenAI answered "Conversation already has an active
response in progress: resp_EDWtl…", the relay forwarded that string verbatim,
and the phone rendered it as a terminal error with the mic disabled.

Two module-level units now own that boundary:
- `_ResponseGate` — serializes every relay-sent response.create against the
  responses OpenAI creates on its own (VAD), defers instead of colliding, and
  self-heals the one unwinnable ordering via `on_conflict`.
- `classify_realtime_error` — no raw upstream text ever crosses the WS
  boundary; benign turn-taking noise is silenced, billing keeps its dedicated
  copy, everything else carries a code + a recoverable bit.

These tests assert BEHAVIOUR (what gets sent, what the user sees), not the
presence of a diff — see test_voice_tz's note on why that distinction matters.
"""

import asyncio
import json

import pytest

from app.api.ws_realtime import (
    _ResponseGate,
    classify_realtime_error,
)


class Wire:
    """Captures what the gate actually sends."""

    def __init__(self):
        self.sent: list = []

    async def send(self, payload: str):
        self.sent.append(json.loads(payload))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def wire():
    return Wire()


@pytest.fixture()
def gate(wire):
    return _ResponseGate(wire.send)


# ── The plain path ────────────────────────────────────────────────────────

def test_idle_create_sends_immediately(gate, wire):
    run(gate.create())
    assert wire.sent == [{"type": "response.create"}]
    assert gate.active is True


def test_create_while_active_defers_and_sends_nothing(gate, wire):
    gate.on_created()                    # VAD opened a response
    run(gate.create())                   # tool continuation arrives
    assert wire.sent == []               # no collision
    assert gate.deferred is True


def test_done_reports_deferred_exactly_once(gate, wire):
    gate.on_created()
    run(gate.create())
    assert run(gate.on_done()) is True   # replay wanted
    assert run(gate.on_done()) is False  # and only once


def test_done_without_deferred_reports_nothing(gate):
    gate.on_created()
    assert run(gate.on_done()) is False


# ── The recording's race, end to end ─────────────────────────────────────
# During a long tool run the reader is blocked, so the function-call response's
# response.done AND a VAD response.created are queued unread. The continuation
# fires between them, collides, and must come back on its own.

def test_tool_continuation_race_self_heals(gate, wire):
    # Function-call response is active; tool result ready mid-response.
    gate.on_created()
    run(gate.create())                   # deferred
    # Queue drains: the function-call response finishes...
    assert run(gate.on_done()) is True
    run(gate.create())                   # replay — sends
    assert len(wire.sent) == 1
    # ...but a VAD response was ALSO queued: our create collides upstream.
    gate.on_conflict()                   # error event arrives
    assert gate.active and gate.deferred
    run(gate.create())                   # any further asks still defer
    assert len(wire.sent) == 1
    # VAD response finishes → the deferred continuation finally runs.
    assert run(gate.on_done()) is True
    run(gate.create())
    assert len(wire.sent) == 2
    # Terminates: nothing active, nothing deferred.
    assert run(gate.on_done()) is False


def test_concurrent_creators_produce_one_send(gate, wire):
    async def storm():
        await asyncio.gather(*(gate.create() for _ in range(5)))
    run(storm())
    assert len(wire.sent) == 1           # one sent, four deferred
    assert gate.deferred is True


# ── The WS boundary: what the user is allowed to see ─────────────────────

def test_benign_turn_taking_noise_is_silenced():
    for code in ("response_cancel_not_active",
                 "input_audio_buffer_commit_empty",
                 "item_truncate_audio_end_ms_too_large"):
        assert classify_realtime_error(code, "whatever upstream says") is None


def test_raw_upstream_text_never_reaches_the_frame():
    raw = "Conversation already has an active response in progress: resp_EDWtl"
    frame = classify_realtime_error("some_new_code", raw)
    assert frame is not None
    assert raw not in json.dumps(frame)
    assert "resp_" not in json.dumps(frame)


def test_unknown_errors_are_recoverable_with_a_code():
    frame = classify_realtime_error("weird_new_thing", "msg")
    assert frame["recoverable"] is True
    assert frame["code"] == "weird_new_thing"
    assert frame["type"] == "error"


def test_missing_code_still_yields_a_stable_code():
    frame = classify_realtime_error("", "mystery")
    assert frame["code"] == "voice_error"
    assert frame["recoverable"] is True


def test_session_death_is_fatal():
    for code in ("session_expired", "invalid_session_state", "auth_failed"):
        frame = classify_realtime_error(code, "msg")
        assert frame["recoverable"] is False


def test_billing_keeps_its_dedicated_copy_and_no_link():
    frame = classify_realtime_error("insufficient_quota", "You exceeded your current quota")
    assert frame["billing"] is True
    assert "http" not in json.dumps(frame)   # never the platform's billing page
