"""A play asked for BEFORE a stop/pause must not start AFTER it (addendum-2 item 9).

Review F29. The caller says "play some jazz"; the tenant's search takes a few
seconds. Before it returns they say "stop the music": the station goes OFF,
the phone stops and acks, and the relay says "Stopped the music." Then the
older play's search returns and `_tool_play_media` broadcasts a non-auto
`media_play`. The phone reads that as an explicit user play: it lifts the
user-stop fence, starts the stale track and re-seeds the station the stop had
just turned off. The newer intent loses.

The tenant sends that frame, so the tenant decides the order: a play takes a
mark when it is asked for (`internal_play_media` on arrival, the agent's tool
when it starts) and re-reads it right before its broadcast. A voice stop/pause
(`radio.control`) or any radio_toggle OFF (the phone's X, the pill, a web
close) that landed in between drops the play: no frame, no `_last_media`, no
station seed. The endpoint answers `{"ok": false, "reason": "superseded"}`; the
tool answers a truthful non-ERROR line so the model neither announces the play
nor retries it.

Everything below drives the REAL endpoint / tool / control / radio_toggle code
with only the YouTube search (httpx) and the socket fan-out stubbed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.agent.radio import control
from app.agent.radio.session import RadioSessionManager
from app.api import api_v1, ws_chat
from app.api.api_v1 import PlayMediaRequest
from app.config import settings

pytestmark = pytest.mark.asyncio

USER = "owner-stale-play"
VIDEO = "JAZZ0000001"


class _Req:
    headers = {"X-Agent-Key": "stale-play-agent-key"}


async def _until(predicate, timeout: float = 2.0, label: str = "condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.005)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """The tenant with a search that blocks until the test releases it."""
    import httpx

    import app.agent.radio as radio_pkg
    import app.agent.tool_executor as te
    from app.agent import media_resolve
    from app.agent.radio import player
    from app.agent.tool_executor import ToolExecutor

    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "stale-play-agent-key")
    monkeypatch.setattr(settings, "user_id", USER)

    manager = RadioSessionManager()
    seeds: list = []
    real_seed = manager.record_user_seed

    def spy_seed(**kw):
        seeds.append(kw)
        return real_seed(**kw)

    monkeypatch.setattr(manager, "record_user_seed", spy_seed)
    monkeypatch.setattr(radio_pkg, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(control, "get_radio_manager", lambda: manager)
    for name in (
        "_radio_toggle_locks", "_radio_off_generations",
        "_media_ended_locks", "_display_mode_locks",
    ):
        monkeypatch.setattr(ws_chat, name, {})
    monkeypatch.setattr(control, "_pending_acks", type(control._pending_acks)())
    # Absent before the fix; `raising=False` so the test runs (and fails on
    # behaviour) against a tenant without the guard.
    monkeypatch.setattr(control, "_latest_halts", type(control._pending_acks)(), raising=False)

    frames: list[dict] = []

    async def broadcast(user_id, event, exclude=None, **kw):
        frames.append(dict(event))
        return 1

    async def no_swap(*a, **kw):
        return None

    monkeypatch.setattr(ws_chat, "broadcast_to_user", broadcast)
    monkeypatch.setattr(ws_chat, "_check_age_and_swap", no_swap)
    monkeypatch.setattr(player, "warm_audio_cache", lambda *a, **kw: None)

    gate = asyncio.Event()
    searching = asyncio.Event()

    class _SlowSearch:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *a, **kw):
            searching.set()
            await gate.wait()
            return SimpleNamespace(text="<html></html>")

    monkeypatch.setattr(httpx, "AsyncClient", _SlowSearch)
    monkeypatch.setattr(
        media_resolve, "scrape_results",
        lambda html, limit=6: [(VIDEO, "Jazz Mix", "")],
    )
    monkeypatch.setattr(te, "_MEDIA_RESOLVE_CACHE", {})

    tools = ToolExecutor(workspace=str(tmp_path))
    monkeypatch.setattr(api_v1, "_agent_runner", SimpleNamespace(tools=tools))
    return SimpleNamespace(
        frames=frames, seeds=seeds, gate=gate, searching=searching,
        tools=tools, manager=manager,
    )


def _types(frames):
    return [f["type"] for f in frames]


async def test_drake_artist_play_uses_the_existing_tenant_delivery(rig, monkeypatch):
    """The Live fast path's Drake/variety body reaches the real play tool.

    An unrelated Draco hit before the relevant song must not be the item sent
    to the phone. Only catalogue search and socket fan-out are replaced.
    """
    from app.agent.radio import playlist

    searches = []

    def catalogue(action, **kwargs):
        searches.append((action, kwargs.get("query")))
        return [
            {"videoId": "DRACO000001", "title": "Draco Malfoy Theme", "artists": [{"name": "Fan"}]},
            {"videoId": "DRAKE000001", "title": "God's Plan", "artists": [{"name": "Drake"}]},
        ]

    monkeypatch.setattr(playlist, "_yt_remote", catalogue)

    response = await api_v1.internal_play_media(
        PlayMediaRequest(query="Drake", variety=True), _Req(),
    )

    assert (response["ok"], response["title"], response["video_id"]) == (
        True, "Drake - God's Plan", "DRAKE000001",
    )
    assert searches == [("search", "Drake")]
    assert [(f.get("video_id"), f.get("title"), f.get("mode"))
            for f in rig.frames if f.get("type") == "media_play"] == [
        ("DRAKE000001", "Drake - God's Plan", "song"),
    ]


async def _voice_halt(action: str, frames: list, user_id: str = USER):
    """A spoken stop/pause exactly as the relay runs it: the real
    `execute_media_control`, confirmed by the phone's ack."""
    pending = asyncio.create_task(
        control.execute_media_control(user_id, action, "app", ack_timeout_s=2.0),
    )
    frame_type = f"media_{action}"
    await _until(lambda: any(f["type"] == frame_type for f in frames), label=frame_type)
    command_id = [f for f in frames if f["type"] == frame_type][-1]["command_id"]
    done_key = "paused" if action == "pause" else "stopped"
    assert control.deliver_media_ack(user_id, {
        "type": f"{frame_type}_ack", "command_id": command_id,
        done_key: True, "was_playing": True,
    }) is True
    return await asyncio.wait_for(pending, timeout=2.0)


async def _endpoint_play(query: str = "some jazz"):
    return await api_v1.internal_play_media(PlayMediaRequest(query=query), _Req())


# ── The endpoint (the relay's media fast path) ───────────────────────────

async def test_a_voice_stop_during_the_search_drops_the_play(rig):
    play = asyncio.create_task(_endpoint_play())
    await asyncio.wait_for(rig.searching.wait(), timeout=2.0)

    outcome = await _voice_halt("stop", rig.frames)
    assert (outcome.ok, outcome.reason) == (True, "stopped")

    rig.gate.set()
    response = await asyncio.wait_for(play, timeout=2.0)

    assert response["ok"] is False, response
    assert response["reason"] == "superseded"
    assert response["video_id"] == "" and response["title"] == ""
    # Nothing reached the phone after its stop: no media_play to lift the
    # fence, no seed to rebuild the station the stop turned off.
    assert "media_play" not in _types(rig.frames), _types(rig.frames)
    assert _types(rig.frames) == ["radio_state", "media_stop"]
    assert rig.seeds == []
    assert rig.manager.get(USER, "app") is None


async def test_a_voice_pause_during_the_search_drops_the_play(rig):
    play = asyncio.create_task(_endpoint_play())
    await asyncio.wait_for(rig.searching.wait(), timeout=2.0)

    outcome = await _voice_halt("pause", rig.frames)
    assert (outcome.ok, outcome.reason) == (True, "paused")

    rig.gate.set()
    response = await asyncio.wait_for(play, timeout=2.0)

    assert (response["ok"], response["reason"]) == (False, "superseded"), response
    assert "paused" in response["detail"]
    assert _types(rig.frames) == ["media_pause"]
    assert rig.seeds == []


async def test_the_phones_x_during_the_search_drops_the_play(rig):
    """The media-only X: the phone stops natively and sends radio_toggle OFF.
    The relay never hears about it, so only the tenant can drop the play."""
    play = asyncio.create_task(_endpoint_play())
    await asyncio.wait_for(rig.searching.wait(), timeout=2.0)

    await ws_chat._handle_radio_toggle(
        USER, {"type": "radio_toggle", "channel": "app", "enabled": False},
    )

    rig.gate.set()
    response = await asyncio.wait_for(play, timeout=2.0)

    assert (response["ok"], response["reason"]) == (False, "superseded"), response
    assert "media_play" not in _types(rig.frames)
    assert rig.seeds == []


async def test_an_off_on_another_of_the_users_surfaces_also_counts(rig):
    """media_play goes to every socket the user has, so a stop on any of them
    (here the web player's close) makes the older play stale on all."""
    play = asyncio.create_task(_endpoint_play())
    await asyncio.wait_for(rig.searching.wait(), timeout=2.0)

    await ws_chat._handle_radio_toggle(
        USER, {"type": "radio_toggle", "channel": "web", "enabled": False},
    )

    rig.gate.set()
    response = await asyncio.wait_for(play, timeout=2.0)
    assert (response["ok"], response["reason"]) == (False, "superseded"), response
    assert "media_play" not in _types(rig.frames)


async def test_a_stop_before_the_play_was_asked_does_not_block_it(rig):
    """The preserved property: a play asked AFTER the stop still plays."""
    outcome = await _voice_halt("stop", rig.frames)
    assert outcome.ok is True
    await ws_chat._handle_radio_toggle(
        USER, {"type": "radio_toggle", "channel": "app", "enabled": False},
    )

    rig.gate.set()
    response = await asyncio.wait_for(_endpoint_play(), timeout=2.0)

    assert response["ok"] is True, response
    assert response["video_id"] == VIDEO
    assert _types(rig.frames)[-1] == "media_play"
    assert rig.seeds and rig.seeds[0]["seed_track"].video_id == VIDEO


async def test_a_stop_after_the_broadcast_is_not_rewritten_as_a_supersede(rig):
    """The check is at the broadcast. A stop that lands after it sends its
    media_stop after the media_play, so the newer intent still wins on the
    phone, and the play that did start is reported as started."""
    rig.gate.set()
    response = await asyncio.wait_for(_endpoint_play(), timeout=2.0)
    assert response["ok"] is True and response["video_id"] == VIDEO

    outcome = await _voice_halt("stop", rig.frames)
    assert outcome.ok is True
    assert _types(rig.frames) == ["media_play", "radio_state", "media_stop"]


async def test_another_users_stop_does_not_touch_this_play(rig):
    play = asyncio.create_task(_endpoint_play())
    await asyncio.wait_for(rig.searching.wait(), timeout=2.0)

    await _voice_halt("stop", rig.frames, user_id="someone-else")
    await ws_chat._handle_radio_toggle(
        "someone-else", {"type": "radio_toggle", "channel": "app", "enabled": False},
    )

    rig.gate.set()
    response = await asyncio.wait_for(play, timeout=2.0)
    assert response["ok"] is True, response
    assert _types(rig.frames)[-1] == "media_play"


# ── The agent's own play_media tool (a play the fast path did not take) ──

async def test_the_agent_tool_reports_a_superseded_play_truthfully(rig):
    """The voice `think` path: the full agent's play_media runs on channel
    'voice' with no mark from anyone, so it takes its own when it starts."""
    tools = rig.tools
    tools.set_user_id(USER)
    tools.set_channel("voice")
    play = asyncio.create_task(tools._tool_play_media({"query": "some jazz"}))
    await asyncio.wait_for(rig.searching.wait(), timeout=2.0)

    await _voice_halt("stop", rig.frames)
    rig.gate.set()
    result = await asyncio.wait_for(play, timeout=2.0)

    assert result.startswith(getattr(control, "PLAY_SUPERSEDED_PREFIX", "SUPERSEDED:")), result
    # A decision, not a failure: never an ERROR the model would retry, and
    # never a line it could read as "now playing".
    assert not result.upper().startswith("ERROR")
    assert "stopped" in result
    assert "Now playing" not in result and "Jazz Mix" not in result
    assert "Do not call play_media again" in result
    assert "media_play" not in _types(rig.frames)


async def test_the_agent_tool_still_plays_when_nothing_halted(rig):
    tools = rig.tools
    tools.set_user_id(USER)
    tools.set_channel("voice")
    rig.gate.set()
    result = await asyncio.wait_for(
        # A model-supplied value under the private key is not a mark.
        tools._tool_play_media({"query": "some jazz", "_halt_mark": {"halt_seq": -1}}),
        timeout=2.0,
    )
    assert result.startswith("Now playing"), result
    assert _types(rig.frames) == ["media_play"]


# ── The mark itself ──────────────────────────────────────────────────────

async def test_the_mark_moves_only_for_a_newer_halt_of_the_same_user(rig):
    mark = control.media_halt_mark(USER)
    assert control.media_play_superseded(mark) is None
    # A radio ON is not a halt.
    ws_chat._radio_off_generations[(USER, "app")] = 0
    assert control.media_play_superseded(mark) is None

    await _voice_halt("pause", rig.frames)
    assert control.media_play_superseded(mark) == "pause"
    later = control.media_halt_mark(USER)
    assert control.media_play_superseded(later) is None

    await ws_chat._handle_radio_toggle(
        USER, {"type": "radio_toggle", "channel": "app", "enabled": False},
    )
    assert control.media_play_superseded(later) == "stop"
    # Nothing that is not a mark from this module can mark anything.
    assert control.media_play_superseded(None) is None
    assert control.media_play_superseded({"user_id": USER, "halt_seq": -1}) is None
