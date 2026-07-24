"""In-flight turn registry — the reconnect/resume contract (ws_chat).

Founder repro 2026-07-23: "I left the app while it was showing the thinking
animation; when I returned the chat rendered empty." The turn kept running
headless (that part already worked) but nothing ever TOLD the returning
client that work was still in progress, so the app had no choice but to
show a dead thread.

These pins cover the signal that fixes it:
  • a running turn is announced to a reconnecting socket (`turn_active`),
  • it is narrated to that socket as it advances (`turn_status`), never to
    the socket that owns the turn (which gets the real frames),
  • it is retired exactly once, and a stale entry can never be announced.
"""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.fixture(autouse=True)
def _clean_registry():
    from app.api import ws_chat
    ws_chat._active_turns.clear()
    ws_chat._user_ws_queues.clear()
    yield
    ws_chat._active_turns.clear()
    ws_chat._user_ws_queues.clear()


def test_active_turn_roundtrip_and_frame_shape():
    from app.api import ws_chat

    now = time.time()
    ws_chat._set_active_turn(
        "user-1", mission_id="chatturn:abc", title="Remind me to call my father",
        stage="thinking", tool=None, started_at=now,
    )
    entry = ws_chat._get_active_turn("user-1")
    assert entry and entry["mission_id"] == "chatturn:abc"

    frame = ws_chat._turn_frame("turn_active", entry, resumed=True)
    assert frame["type"] == "turn_active"
    assert frame["mission_id"] == "chatturn:abc"
    assert frame["title"] == "Remind me to call my father"
    assert frame["stage"] == "thinking"
    assert frame["tool"] is None
    assert frame["started_at_ms"] == int(now * 1000)
    assert frame["resumed"] is True


def test_stale_turn_is_never_announced():
    """A process killed mid-turn must not strand the next client on a
    working orb that will never resolve."""
    from app.api import ws_chat

    ws_chat._set_active_turn(
        "user-1", mission_id="chatturn:old", title="x",
        stage="thinking", tool=None,
        started_at=time.time() - (ws_chat._TURN_STALE_S + 60),
    )
    assert ws_chat._get_active_turn("user-1") is None
    assert "user-1" not in ws_chat._active_turns


def test_clear_is_mission_scoped():
    """A finished turn must not retire the entry of a NEWER turn the user
    started while it was wrapping up."""
    from app.api import ws_chat

    ws_chat._set_active_turn(
        "user-1", mission_id="chatturn:new", title="second ask",
        stage="thinking", tool=None, started_at=time.time(),
    )
    ws_chat._clear_active_turn("user-1", "chatturn:old")
    assert ws_chat._get_active_turn("user-1") is not None

    ws_chat._clear_active_turn("user-1", "chatturn:new")
    assert ws_chat._get_active_turn("user-1") is None


def test_second_turn_replaces_first_entry_wholesale():
    from app.api import ws_chat

    ws_chat._set_active_turn(
        "user-1", mission_id="chatturn:one", title="first",
        stage="tool", tool="web_search", started_at=time.time(),
    )
    ws_chat._set_active_turn(
        "user-1", mission_id="chatturn:two", title="second",
        stage="thinking", tool=None, started_at=time.time(),
    )
    entry = ws_chat._get_active_turn("user-1")
    # No leakage of the previous turn's tool into the new entry.
    assert entry["mission_id"] == "chatturn:two"
    assert entry["tool"] is None
    assert entry["stage"] == "thinking"


def test_broadcast_excludes_the_owning_socket():
    """The socket running the turn already gets tool_start/status/done — the
    mirror is for every OTHER socket (the phone that just came back)."""
    from app.api import ws_chat

    owner: asyncio.Queue = asyncio.Queue(maxsize=10)
    resumed: asyncio.Queue = asyncio.Queue(maxsize=10)
    ws_chat._register_ws_queue("user-1", owner)
    ws_chat._register_ws_queue("user-1", resumed)

    sent = asyncio.run(ws_chat.broadcast_to_user(
        "user-1", {"type": "turn_status", "stage": "tool"}, exclude=owner,
    ))
    assert sent == 1
    assert owner.empty()
    assert resumed.get_nowait()["type"] == "turn_status"


def test_broadcast_without_exclude_reaches_everyone():
    from app.api import ws_chat

    a: asyncio.Queue = asyncio.Queue(maxsize=10)
    b: asyncio.Queue = asyncio.Queue(maxsize=10)
    ws_chat._register_ws_queue("user-1", a)
    ws_chat._register_ws_queue("user-1", b)

    sent = asyncio.run(ws_chat.broadcast_to_user("user-1", {"type": "message"}))
    assert sent == 2
    assert not a.empty() and not b.empty()
