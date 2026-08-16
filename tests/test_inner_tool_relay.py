"""`_InnerToolRelay` — the voice relay's inner-step row lifecycle.

Context: on the realtime voice channel the whole agent turn is handed to one
`think` tool, and the steps the user actually cares about are relayed out of it
as `tool.intent` / `tool.start` / `tool.end`. `tool.intent` fires at the model's
tool_use_start, BEFORE the arguments have finished streaming — so it can name
the step a second or two before `tool.start` can describe it.

It used to open no row at all, and the phone showed nothing for that window.
Opening a PROVISIONAL row buys the head start, but only if adoption is exact:
the failure modes are a duplicated step and — worse — a spinner that never
resolves, because the phone only closes a row whose call_id it has seen.

These tests pin the adoption rules. They exercise the relay directly against a
fake socket; nothing here needs a database, an agent or an event loop beyond
asyncio itself.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.ws_realtime import _InnerToolRelay  # noqa: E402


class FakeWS:
    """Collects the frames the relay would have sent to the phone."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, frame: dict) -> None:
        self.sent.append(frame)


def drive(events: list[dict], close: bool = False) -> list[dict]:
    ws = FakeWS()
    relay = _InnerToolRelay(ws, "outer1")

    async def run() -> None:
        for ev in events:
            await relay.on_event(ev)
        if close:
            await relay.close_open()

    asyncio.run(run())
    return ws.sent


def started(frames: list[dict]) -> list[dict]:
    return [f for f in frames if f.get("type") == "tool_call.started"]


def completed(frames: list[dict]) -> list[dict]:
    return [f for f in frames if f.get("type") == "tool_call.completed"]


def test_intent_then_matching_start_is_one_row() -> None:
    """The whole point: the row opens early and the start FILLS it.

    Two `tool_call.started` frames are correct and expected — the client treats
    a repeated call_id as an update — but they must carry the SAME call_id, or
    the step is drawn twice.
    """
    frames = drive([
        {"type": "tool.intent", "name": "web_search"},
        {"type": "tool.start", "call_id": "c1", "name": "web_search",
         "args": {"query": "grok bot"}},
        {"type": "tool.end", "call_id": "c1", "name": "web_search", "ok": True},
    ])
    starts = started(frames)
    assert len(starts) == 2, starts
    assert starts[0]["call_id"] == starts[1]["call_id"]
    # The intent frame names the step but cannot describe it yet.
    assert starts[0]["detail"] == ""
    assert starts[1]["detail"]
    # And the completion lands on that same row.
    ends = completed(frames)
    assert len(ends) == 1
    assert ends[0]["call_id"] == starts[0]["call_id"]


def test_intent_adopted_only_by_the_same_tool() -> None:
    """A start for a DIFFERENT tool must not silently inherit the row."""
    frames = drive([
        {"type": "tool.intent", "name": "web_search"},
        {"type": "tool.start", "call_id": "c1", "name": "recall_day", "args": {}},
    ])
    starts = started(frames)
    assert len(starts) == 2
    assert starts[0]["name"] == "web_search"
    assert starts[1]["name"] == "recall_day"
    assert starts[0]["call_id"] != starts[1]["call_id"]
    # The abandoned provisional row is CLOSED, not left spinning.
    ends = completed(frames)
    assert [e["call_id"] for e in ends] == [starts[0]["call_id"]]
    assert ends[0]["ok"] is False


def test_second_intent_falls_back_to_the_coarse_flag() -> None:
    """Two tool_use blocks opened before either's arguments land.

    Only one provisional row may be outstanding: guessing which start belongs
    to which intent is exactly how a step gets attributed to the wrong tool.
    The second intent gives up the head start instead.
    """
    frames = drive([
        {"type": "tool.intent", "name": "web_search"},
        {"type": "tool.intent", "name": "recall_day"},
    ])
    assert len(started(frames)) == 1
    assert frames[-1] == {"type": "state", "state": "tool_use"}


def test_unnamed_intent_opens_no_row() -> None:
    frames = drive([{"type": "tool.intent", "name": ""}])
    assert started(frames) == []
    assert frames == [{"type": "state", "state": "tool_use"}]


def test_close_open_closes_a_pending_provisional_row() -> None:
    """A turn that ends between tool_use_start and the arguments landing.

    Without this the phone keeps a named, spinning step for the rest of the
    call — the exact "never leaves a spinner" guarantee `close_open` exists for.
    """
    frames = drive([{"type": "tool.intent", "name": "web_search"}], close=True)
    starts, ends = started(frames), completed(frames)
    assert len(starts) == 1 and len(ends) == 1
    assert ends[0]["call_id"] == starts[0]["call_id"]
    assert ends[0]["ok"] is False


def test_end_without_a_start_is_dropped() -> None:
    """Pre-existing contract: never orphan a completion onto an unopened row."""
    frames = drive([{"type": "tool.end", "call_id": "nope", "name": "x", "ok": True}])
    assert completed(frames) == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
