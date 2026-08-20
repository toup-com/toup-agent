"""A voice run must land in the day chat as a RUN, not as a bare sentence.

The 2026-08-20 recording: a voice question read nineteen pages across four
searches, and the thread it wrote into showed two plain bubbles with mic icons.
Everything the agent did — the steps, the actions, the sites, the sources —
existed only as frames on the audio socket and was discarded when the call
ended. A typed run in the same thread carries all of it, because agent_runner
persists `tool_events` into the message's metadata_json.

So the relay now records what it relays, and the record rides the assistant row
through the SAME key a chat turn uses. These tests pin the record's shape (the
clients read it directly), that it survives a phone that has gone away, and
that the two metadata keys no longer evict each other.
"""
import asyncio
import json

import pytest

from app.api.sessions import _build_metadata, _clean_tool_events
from app.api.ws_realtime import (
    _PERSIST_SOURCES_MAX,
    _InnerToolRelay,
    _message_payload,
)


class _DeadSocket:
    """A phone that has hung up. Every send raises, exactly like a closed WS."""

    async def send_json(self, frame):
        raise RuntimeError("phone gone")


class _Socket:
    def __init__(self):
        self.frames = []

    async def send_json(self, frame):
        self.frames.append(frame)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


SOURCES = [
    {"title": "Best Video Generation AI Models in 2026 | Pinggy Blog",
     "url": "https://pinggy.io/blog/best_ai_video_models_2026/", "domain": "pinggy.io"},
    {"title": "AI Video Generation Models Compared", "url": "https://digen.ai/compare",
     "domain": "digen.ai"},
]


async def _search_turn(sink, ws=None):
    relay = _InnerToolRelay(ws or _Socket(), "outer-1", sink=sink)
    await relay.on_event({"type": "tool.start", "call_id": "c1", "name": "web_search",
                          "args": {"query": "strongest video model"}})
    await relay.on_event({"type": "tool.end", "call_id": "c1", "name": "web_search",
                          "ok": True, "elapsed_ms": 4000,
                          "preview": "1. Best Video Generation AI Models",
                          "sources": SOURCES})
    return relay


# ── The record the clients read ───────────────────────────────────────────

def test_the_relay_records_a_client_shaped_tool_event():
    sink = []
    _run(_search_turn(sink))
    assert len(sink) == 1
    rec = sink[0]
    # The two fields day_chats._serialize_tool_events REQUIRES, or the record
    # is dropped at read time and the run is invisible again.
    assert rec["tool"] == "web_search"
    assert isinstance(rec["started_at_ms"], int)
    assert rec["completed_at_ms"] - rec["started_at_ms"] == 4000
    assert rec["summary"].startswith("1. Best Video Generation")


def test_the_record_carries_the_sites_the_favicons_are_drawn_from():
    sink = []
    _run(_search_turn(sink))
    rec = sink[0]
    # `domains` is what the clients' favicon resolver reads (round four).
    assert rec["domains"] == ["pinggy.io", "digen.ai"]
    assert rec["urls"][0].startswith("https://pinggy.io/")


def test_the_record_carries_the_TITLES_too():
    """`domains` alone reduces the source list to bare hostnames.

    The call showed titled cards; the thread has to be able to show the same
    list, which means the titles have to survive the socket.
    """
    sink = []
    _run(_search_turn(sink))
    srcs = sink[0]["sources"]
    assert len(srcs) == 2
    assert srcs[0]["title"].startswith("Best Video Generation AI Models")
    assert srcs[0]["domain"] == "pinggy.io"


def test_sources_are_capped_on_the_persisted_record():
    sink = []
    many = [{"title": f"t{i}", "url": f"https://s{i}.com/a", "domain": f"s{i}.com"}
            for i in range(20)]

    async def go():
        relay = _InnerToolRelay(_Socket(), "outer-1", sink=sink)
        await relay.on_event({"type": "tool.start", "call_id": "c1", "name": "web_search", "args": {}})
        await relay.on_event({"type": "tool.end", "call_id": "c1", "name": "web_search",
                              "ok": True, "elapsed_ms": 10, "preview": "p", "sources": many})
    _run(go())
    assert len(sink[0]["sources"]) <= _PERSIST_SOURCES_MAX


def test_a_hung_up_phone_still_gets_its_run_recorded():
    """`_send` gives up the moment the socket dies, and the call the user
    walked away from is exactly the one whose record has to survive."""
    sink = []
    _run(_search_turn(sink, ws=_DeadSocket()))
    assert len(sink) == 1 and sink[0]["domains"] == ["pinggy.io", "digen.ai"]


def test_a_failed_step_is_recorded_as_failed():
    sink = []

    async def go():
        relay = _InnerToolRelay(_Socket(), "outer-1", sink=sink)
        await relay.on_event({"type": "tool.start", "call_id": "c1", "name": "web_search", "args": {}})
        await relay.on_event({"type": "tool.end", "call_id": "c1", "name": "web_search",
                              "ok": False, "elapsed_ms": 200, "preview": ""})
    _run(go())
    assert sink[0]["ok"] is False


def test_no_sink_is_a_no_op():
    """Every other caller constructs the relay without one."""
    sink_free = _InnerToolRelay(_Socket(), "outer-1")
    _run(sink_free.on_event({"type": "tool.start", "call_id": "c1", "name": "web_search", "args": {}}))
    _run(sink_free.on_event({"type": "tool.end", "call_id": "c1", "name": "web_search",
                             "ok": True, "elapsed_ms": 1, "preview": "p", "sources": SOURCES}))
    assert sink_free._rec == {}


def test_several_steps_stay_in_call_order():
    sink = []

    async def go():
        relay = _InnerToolRelay(_Socket(), "outer-1", sink=sink)
        for i, name in enumerate(["recall_day", "web_search", "web_search"]):
            await relay.on_event({"type": "tool.start", "call_id": f"c{i}", "name": name, "args": {}})
            await relay.on_event({"type": "tool.end", "call_id": f"c{i}", "name": name,
                                  "ok": True, "elapsed_ms": 100, "preview": f"p{i}"})
    _run(go())
    assert [r["tool"] for r in sink] == ["recall_day", "web_search", "web_search"]


def test_a_provisional_intent_row_is_recorded_once_not_twice():
    """`tool.intent` opens a row and `tool.start` ADOPTS it. If both opened a
    record the thread would show every step twice."""
    sink = []

    async def go():
        relay = _InnerToolRelay(_Socket(), "outer-1", sink=sink)
        await relay.on_event({"type": "tool.intent", "name": "web_search"})
        await relay.on_event({"type": "tool.start", "call_id": "c1", "name": "web_search",
                              "args": {"query": "q"}})
        await relay.on_event({"type": "tool.end", "call_id": "c1", "name": "web_search",
                              "ok": True, "elapsed_ms": 50, "preview": "p"})
    _run(go())
    assert len(sink) == 1, "the adopted row must not be recorded a second time"
    assert "completed_at_ms" in sink[0], "the completion must reach the adopted record"


# ── The wire ──────────────────────────────────────────────────────────────

def test_tool_events_ride_the_BODY_and_drop_the_query_shim():
    body, params = _message_payload("assistant", "hello", "gpt-realtime",
                                    None, [{"tool": "web_search", "started_at_ms": 1}])
    assert body["tool_events"][0]["tool"] == "web_search"
    assert params is None, "a list of objects cannot ride the query shim"


def test_media_and_tool_events_can_ride_the_same_row():
    body, params = _message_payload("assistant", "hi", "m", {"video_id": "x"},
                                    [{"tool": "play_media", "started_at_ms": 1}])
    assert body["media"] == {"video_id": "x"}
    assert body["tool_events"]
    assert params is None


def test_no_tool_events_leaves_the_payload_exactly_as_it_was():
    body, params = _message_payload("user", "hello")
    assert "tool_events" not in body
    assert params == {"role": "user", "content": "hello"}


# ── The agent's side ──────────────────────────────────────────────────────

def test_clean_tool_events_keeps_only_what_the_clients_read():
    got = _clean_tool_events([{
        "tool": "web_search", "started_at_ms": 1, "completed_at_ms": 2,
        "summary": "s", "domains": ["a.com"], "sources": [{"title": "t"}],
        "raw_result": "SHOULD NOT BE STORED", "api_key": "nope",
    }])
    assert set(got[0]) == {"tool", "started_at_ms", "completed_at_ms",
                           "summary", "domains", "sources"}


def test_clean_tool_events_drops_records_the_reader_would_drop_anyway():
    assert _clean_tool_events([{"started_at_ms": 1}]) is None      # no tool
    assert _clean_tool_events([{"tool": "web_search"}]) is None    # no start
    assert _clean_tool_events([]) is None
    assert _clean_tool_events(None) is None
    assert _clean_tool_events("not a list") is None


def test_clean_tool_events_is_bounded():
    many = [{"tool": "t", "started_at_ms": i} for i in range(500)]
    assert len(_clean_tool_events(many)) <= 40


def test_metadata_no_longer_lets_the_two_keys_evict_each_other():
    """Media used to REPLACE the whole blob, so a voice turn that played a
    song and used a tool could persist only one of them."""
    both = json.loads(_build_metadata({"video_id": "x"}, [{"tool": "t", "started_at_ms": 1}]))
    assert both["media"] == {"video_id": "x"}
    assert both["tool_events"][0]["tool"] == "t"
    assert json.loads(_build_metadata({"video_id": "x"}, None)) == {"media": {"video_id": "x"}}
    assert _build_metadata(None, None) is None


def test_the_persisted_record_survives_the_readers_own_filter():
    """End to end through the function that reads it back.

    `_serialize_tool_events` drops records missing `tool`/`started_at_ms`, so
    a shape that looks fine on write can still be invisible on read.
    """
    from app.api.day_chats import _serialize_tool_events

    sink = []
    _run(_search_turn(sink))
    cleaned = _clean_tool_events(sink)

    class _Msg:
        metadata_json = _build_metadata(None, cleaned)

    out = _serialize_tool_events(_Msg())
    assert out and out[0]["tool"] == "web_search"
    assert out[0]["domains"] == ["pinggy.io", "digen.ai"]
    assert out[0]["sources"][0]["title"].startswith("Best Video Generation")
