"""Order-keyed media halts and the fast-path query residue (addendum 6 R6-4
tenant half, R6-6; contract v0.3 §7 "Amendment R6"). The supervisor 00:10
cross-scope TOTAL key this file first pinned is replaced by the R6-4b
skew-safe PAIRWISE rule (writer:r6-tenant-2; the changed pins say so and
`tests/test_media_r6b_tenant.py` holds the R6-4b / R6-9 pins).

Owner requirement 7 / pattern B: the caller's newest media request wins; a
stop halts exactly the plays requested before it; the spoken outcome matches
the device.

The tenant used to order a play and a halt by ARRIVAL at the tenant. The
relay holds a spoken stop for a grace and an agent run that is still thinking
took its mark when it arrived, so 'play Halo' → 'stop the music' → 'play some
jazz' reached the tenant as play(Halo), play(jazz), stop — and the stop
superseded the jazz the caller asked for after it (media-1, media-4); a stop
also silenced a newer song already playing (media-3). Now an ordered request
carries the caller's order (`media_order` / `before_order`, `media_scope`,
`media_scope_started_ms`) and the tenant compares a total key; unordered
requests (phone X, typed stop, a legacy relay) keep the arrival rule.

Everything drives the REAL endpoints, play tool, control and radio_toggle
code; only the YouTube search (httpx + scrape) and the socket fan-out are
stubbed.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from app.agent import media_intent
from app.agent.media_intent import media_request
from app.agent.radio import control
from app.agent.radio.session import RadioSessionManager
from app.api import api_v1, ws_chat
from app.api.api_v1 import ChatRequest, MediaControlRequest, PlayMediaRequest
from app.config import settings

USER = "owner-r6-order"
KEY = "r6-order-agent-key"
SCOPE = "psid-r6-call"
OLD_SCOPE, NEW_SCOPE = "psid-r6-before-reconnect", "psid-r6-after-reconnect"
T_OLD, T_NEW = 1_758_000_000_000, 1_758_000_060_000

VIDEOS = {
    "Halo": ("HALO0000001", "Beyonce - Halo"),
    "some jazz": ("JAZZ0000001", "Jazz Mix"),
    "podcast": ("PODC0000001", "A Podcast"),
}


class _Req:
    headers = {"X-Agent-Key": KEY}


async def _until(predicate, timeout: float = 2.0, label: str = "condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.005)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """The tenant, with a YouTube search a test can hold per query."""
    import httpx

    import app.agent.radio as radio_pkg
    import app.agent.tool_executor as te
    from app.agent import media_resolve
    from app.agent.radio import player
    from app.agent.tool_executor import ToolExecutor

    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", KEY)
    monkeypatch.setattr(settings, "user_id", USER)
    manager = RadioSessionManager()
    monkeypatch.setattr(radio_pkg, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(control, "get_radio_manager", lambda: manager)
    for name in (
        "_radio_toggle_locks", "_radio_off_generations",
        "_media_ended_locks", "_display_mode_locks",
    ):
        monkeypatch.setattr(ws_chat, name, {})
    monkeypatch.setattr(control, "_pending_acks", type(control._pending_acks)())
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

    gates: dict[str, asyncio.Event] = {}
    searching: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
    searches: list[str] = []

    class _Search:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None, **kw):
            query = (params or {}).get("search_query", "")
            searches.append(query)
            searching[query].set()
            if query in gates:
                await gates[query].wait()
            return SimpleNamespace(text=query)

    def scrape(html, limit=6):
        vid, title = VIDEOS.get(html, ("OTHER000001", f"Result for {html}"))
        return [(vid, title, "")]

    monkeypatch.setattr(httpx, "AsyncClient", _Search)
    monkeypatch.setattr(media_resolve, "scrape_results", scrape)
    monkeypatch.setattr(te, "_MEDIA_RESOLVE_CACHE", {})
    tools = ToolExecutor(workspace=str(tmp_path))
    monkeypatch.setattr(api_v1, "_agent_runner", SimpleNamespace(tools=tools))

    def hold(query: str) -> asyncio.Event:
        gates[query] = asyncio.Event()
        return gates[query]

    return SimpleNamespace(
        frames=frames, tools=tools, manager=manager, hold=hold,
        searching=searching, searches=searches,
    )


def _types(frames):
    return [f["type"] for f in frames]


def _played(frames):
    return [f["video_id"] for f in frames if f["type"] == "media_play"]


def _play(query, order=None, scope=None, started=None):
    return asyncio.create_task(api_v1.internal_play_media(PlayMediaRequest(
        query=query, media_order=order, media_scope=scope, media_scope_started_ms=started,
    ), _Req()))


async def _halt(rig, action="stop", before=None, scope=None, started=None):
    """A stop/pause through the real endpoint. When the tenant sends a frame
    the phone acks it (it was playing); when it sends none (newer_playing)
    there is nothing to ack."""
    frame_type = f"media_{action}"
    seen = len([f for f in rig.frames if f["type"] == frame_type])
    task = asyncio.create_task(api_v1.internal_media_control(MediaControlRequest(
        user_id=USER, action=action, channel="app",
        before_order=before, media_scope=scope, media_scope_started_ms=started,
    ), _Req()))
    await _until(
        lambda: task.done() or len([f for f in rig.frames if f["type"] == frame_type]) > seen,
        label=f"{frame_type} or a verdict",
    )
    if not task.done():
        command_id = [f for f in rig.frames if f["type"] == frame_type][-1]["command_id"]
        done_key = "paused" if action == "pause" else "stopped"
        assert control.deliver_media_ack(USER, {
            "type": f"{frame_type}_ack", "command_id": command_id,
            done_key: True, "was_playing": True,
        })
    return await asyncio.wait_for(task, timeout=3.0)


# ══ 1. The fast path (/internal/play-media) ════════════════════════════════

async def test_play_halo_stop_play_jazz_ends_on_jazz_on_the_fast_path(rig):
    """media-1 (fast path): both plays reach the tenant BEFORE the relay's
    stop fires. Old: the stop superseded both by arrival — silence."""
    halo_gate, jazz_gate = rig.hold("Halo"), rig.hold("some jazz")
    halo = _play("Halo", order=1, scope=SCOPE)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    jazz = _play("some jazz", order=3, scope=SCOPE)
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)

    stop = await _halt(rig, before=2, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop

    halo_gate.set()
    jazz_gate.set()
    halo_out = await asyncio.wait_for(halo, timeout=2.0)
    jazz_out = await asyncio.wait_for(jazz, timeout=2.0)

    assert (halo_out["ok"], halo_out["reason"]) == (False, "superseded"), halo_out
    assert jazz_out["ok"] is True and jazz_out["video_id"] == "JAZZ0000001", jazz_out
    assert _played(rig.frames) == ["JAZZ0000001"], rig.frames
    assert _types(rig.frames)[-1] == "media_play", "the device ends on jazz"


async def test_a_newer_ordered_play_already_broadcast_is_not_stopped(rig):
    """media-3: the late stop (turn 2) must not silence jazz (turn 3), which is
    already playing; the older Halo still in flight is dropped. The verdict
    names what plays, so the relay never says "stopped"."""
    halo_gate = rig.hold("Halo")
    halo = _play("Halo", order=1, scope=SCOPE)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    jazz_out = await asyncio.wait_for(_play("some jazz", order=3, scope=SCOPE), timeout=2.0)
    assert jazz_out["ok"] is True
    before = list(rig.frames)

    stop = await _halt(rig, before=2, scope=SCOPE)
    assert stop["ok"] is False and stop["reason"] == "newer_playing", stop
    assert stop["video_id"] == "JAZZ0000001" and stop["title"] == "Jazz Mix"
    assert stop["acked_devices"] == 0 and stop["command_id"] == ""
    assert rig.frames == before, "no station OFF and no media_stop reached the phone"
    station = rig.manager.get(USER, "app")
    assert station is not None and station.seed_track.video_id == "JAZZ0000001"

    halo_gate.set()
    halo_out = await asyncio.wait_for(halo, timeout=2.0)
    assert (halo_out["ok"], halo_out["reason"]) == (False, "superseded"), halo_out
    assert _played(rig.frames) == ["JAZZ0000001"]


async def test_a_pause_asked_before_the_playing_item_leaves_it_too(rig):
    assert (await asyncio.wait_for(_play("some jazz", order=5, scope=SCOPE), timeout=2.0))["ok"]
    pause = await _halt(rig, "pause", before=4, scope=SCOPE)
    assert (pause["ok"], pause["reason"]) == (False, "newer_playing"), pause
    assert "media_pause" not in _types(rig.frames)


async def test_a_stop_asked_after_the_playing_item_stops_it(rig):
    """CONTROL: the plain stop of a playing song."""
    assert (await asyncio.wait_for(_play("some jazz", order=3, scope=SCOPE), timeout=2.0))["ok"]
    stop = await _halt(rig, before=4, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop
    assert _types(rig.frames)[-2:] == ["radio_state", "media_stop"]


async def test_an_ordered_stop_after_a_stopped_newer_item_is_applied(rig):
    """The phone's X stopped the newer item: nothing newer is playing, so a
    late older stop is applied (and the phone answers for itself)."""
    assert (await asyncio.wait_for(_play("some jazz", order=3, scope=SCOPE), timeout=2.0))["ok"]
    await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "app", "enabled": False})
    stop = await _halt(rig, before=2, scope=SCOPE)
    assert stop["reason"] != "newer_playing", stop
    assert "media_stop" in _types(rig.frames)


async def test_an_unordered_item_is_always_stopped_by_an_ordered_stop(rig):
    """A play with no order (a legacy relay, typed chat) is never "newer"."""
    assert (await asyncio.wait_for(_play("some jazz"), timeout=2.0))["ok"]
    stop = await _halt(rig, before=1, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop


# ══ 2. The delegated agent run (/internal/agent-turn[/stream]) ═════════════

class _Runner:
    """The agent: each run 'thinks' until its query is released, then calls
    the REAL play_media tool and reports it through the route's own
    on_tool_event (so tool.start carries `_vs_args`)."""

    def __init__(self, tools):
        self.tools = tools
        self.thinking: dict = defaultdict(asyncio.Event)
        self.release: dict = defaultdict(asyncio.Event)
        self.results: dict = {}

    async def run(self, *, user_message, user_id, on_tool_event=None, **kw):
        query = user_message
        self.thinking[query].set()
        await self.release[query].wait()
        self.tools.set_user_id(user_id)
        self.tools.set_channel("app")
        inp = {"query": query, "mode": "audio"}
        if on_tool_event:
            await on_tool_event({"phase": "start", "name": "play_media", "call_id": f"c-{query}",
                                 "input": inp})
        self.results[query] = result = await self.tools._tool_play_media(dict(inp))
        if on_tool_event:
            await on_tool_event({"phase": "end", "name": "play_media", "call_id": f"c-{query}",
                                 "input": inp, "result": result, "elapsed_ms": 5})
        return SimpleNamespace(
            text="ok", session_id="s", tokens_input=0, tokens_output=0, tokens_total=0,
            model="m", tool_calls=[], processing_time_ms=1, persisted={},
        )


async def _collect(stream):
    frames = []
    async for chunk in stream.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        if chunk.startswith("data: "):
            frames.append(json.loads(chunk[len("data: "):]))
    return frames


def _turn(query, order=None, scope=SCOPE, started=None):
    return ChatRequest(message=query, save=False, media_order=order,
                       media_scope=scope if order is not None else None,
                       media_scope_started_ms=started)


async def test_play_halo_stop_play_jazz_ends_on_jazz_on_the_agent_path(rig, monkeypatch):
    """media-1/media-4 (agent path, the stream route voice takes): both runs
    are thinking when the stop for turn 2 fires."""
    runner = _Runner(rig.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    halo_stream = await api_v1.internal_agent_turn_stream(_turn("Halo", 1), _Req())
    halo = asyncio.create_task(_collect(halo_stream))
    jazz_stream = await api_v1.internal_agent_turn_stream(_turn("some jazz", 3), _Req())
    jazz = asyncio.create_task(_collect(jazz_stream))
    await asyncio.wait_for(runner.thinking["Halo"].wait(), timeout=2.0)
    await asyncio.wait_for(runner.thinking["some jazz"].wait(), timeout=2.0)

    stop = await _halt(rig, before=2, scope=SCOPE)
    assert stop["ok"] is True, stop
    runner.release["Halo"].set()
    runner.release["some jazz"].set()
    halo_frames = await asyncio.wait_for(halo, timeout=5.0)
    jazz_frames = await asyncio.wait_for(jazz, timeout=5.0)

    assert runner.results["Halo"].startswith(control.PLAY_SUPERSEDED_PREFIX), runner.results
    assert runner.results["some jazz"].startswith("Now playing"), runner.results
    assert _played(rig.frames) == ["JAZZ0000001"]
    halo_end = [f for f in halo_frames if f.get("type") == "tool.end"][0]
    jazz_end = [f for f in jazz_frames if f.get("type") == "tool.end"][0]
    assert (halo_end["ok"], halo_end.get("outcome")) == (False, "cancelled")
    assert jazz_end["ok"] is True and "outcome" not in jazz_end
    # R6-6 / media-6: the agent's play is named on the wire.
    starts = [f for f in jazz_frames if f.get("type") == "tool.start"]
    assert starts and starts[0]["args"] == {"query": "some jazz"}, starts


async def test_a_thinking_run_newer_than_the_stop_is_not_superseded(rig, monkeypatch):
    """The run ARRIVED before the stop, but the caller asked for it after:
    the blocking route binds the same order."""
    runner = _Runner(rig.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    turn = asyncio.create_task(api_v1.internal_agent_turn(_turn("some jazz", 3), _Req()))
    await asyncio.wait_for(runner.thinking["some jazz"].wait(), timeout=2.0)

    assert (await _halt(rig, before=2, scope=SCOPE))["ok"] is True
    runner.release["some jazz"].set()
    await asyncio.wait_for(turn, timeout=5.0)
    assert runner.results["some jazz"].startswith("Now playing"), runner.results
    assert _types(rig.frames)[-1] == "media_play"
    assert control.run_halt_mark(USER) is None, "the route leaves no mark bound"


async def test_an_ordered_stop_supersedes_an_older_thinking_run(rig, monkeypatch):
    """CONTROL: a stop the caller asked for AFTER the run still drops it."""
    runner = _Runner(rig.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    stream = await api_v1.internal_agent_turn_stream(_turn("Halo", 1), _Req())
    frames = asyncio.create_task(_collect(stream))
    await asyncio.wait_for(runner.thinking["Halo"].wait(), timeout=2.0)
    assert (await _halt(rig, before=2, scope=SCOPE))["ok"] is True
    runner.release["Halo"].set()
    await asyncio.wait_for(frames, timeout=5.0)
    assert runner.results["Halo"].startswith(control.PLAY_SUPERSEDED_PREFIX)
    assert "media_play" not in _types(rig.frames)


# ══ 3. Across scopes: the total key (supervisor 00:10) ═════════════════════

@pytest.mark.parametrize("started", [True, False], ids=["relay-started-ms", "tenant-first-seen"])
async def test_a_new_scope_stop_fences_an_old_scope_pending_play(rig, started):
    """Ordinals restart after a reconnect: the old scope's turn 7 play is
    older than the new scope's turn 1 stop.

    (writer:r6-tenant-2, R6-4b) Unchanged assertions, new reason for the
    unstamped param: with no stamp the pair is causally incomparable and the
    ARRIVAL rule decides it (the stop landed after the play's mark) — there is
    no tenant first-seen clock any more; the id is kept for continuity."""
    gate = rig.hold("Halo")
    play = _play("Halo", order=7, scope=OLD_SCOPE, started=T_OLD if started else None)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    stop = await _halt(rig, before=1, scope=NEW_SCOPE, started=T_NEW if started else None)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop
    gate.set()
    out = await asyncio.wait_for(play, timeout=2.0)
    assert (out["ok"], out["reason"]) == (False, "superseded"), out
    assert "media_play" not in _types(rig.frames)


async def test_a_late_old_scope_stop_never_supersedes_a_newer_scope_play(rig):
    gate = rig.hold("some jazz")
    play = _play("some jazz", order=1, scope=NEW_SCOPE, started=T_NEW)
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    stop = await _halt(rig, before=9, scope=OLD_SCOPE, started=T_OLD)
    assert stop["ok"] is True, stop
    gate.set()
    out = await asyncio.wait_for(play, timeout=2.0)
    assert out["ok"] is True and out["video_id"] == "JAZZ0000001", out


async def test_newer_playing_across_scopes_in_both_directions(rig):
    # PIN CHANGED (writer:r6-tenant-2, addendum 6 R6-9 T1 + R6-4b): the old
    # pin played an OLD-scope Halo right after a NEW-scope item was already
    # on the device and expected it to broadcast. Under T1 the item on the
    # device fences an ordered play it beats (the new scope's stamp is larger,
    # so the caller asked for jazz after that Halo): the newest request wins.
    # Both gate directions stay pinned, in an order that has no older play
    # overwriting a newer one.
    # An old-scope item is older than a new-scope stop …
    assert (await asyncio.wait_for(
        _play("Halo", order=10, scope=OLD_SCOPE, started=T_OLD), timeout=2.0))["ok"]
    stop = await _halt(rig, before=2, scope=NEW_SCOPE, started=T_NEW)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop
    # … and a new-scope item is newer than a late old-scope stop.
    assert (await asyncio.wait_for(
        _play("some jazz", order=3, scope=NEW_SCOPE, started=T_NEW), timeout=2.0))["ok"]
    late = await _halt(rig, before=9, scope=OLD_SCOPE, started=T_OLD)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late
    assert late["video_id"] == "JAZZ0000001"
    # A late OLD-scope play never replaces the newer-scope item (the new
    # scope's stop and, R6-9 T1, the item on the device both beat it; the
    # T1-only cases are pinned in tests/test_media_r6b_tenant.py).
    halo = await asyncio.wait_for(
        _play("Halo", order=11, scope=OLD_SCOPE, started=T_OLD), timeout=2.0)
    assert (halo["ok"], halo["reason"]) == (False, "superseded"), halo
    assert _played(rig.frames) == ["HALO0000001", "JAZZ0000001"]


def test_the_pairwise_rule_orders_scopes_by_stamp_then_first_seen(monkeypatch):
    """Pure (PIN CHANGED, writer:r6-tenant-2): R6-4b replaces the single
    total key (scope rank, order) this test pinned — and its tenant-clock
    rank — with the PAIRWISE rule. Equal stamps across scopes tie-break on the
    scope the tenant saw first; a scope's first-seen never moves once seen;
    an order never compares across scopes."""
    monkeypatch.setattr(control, "_latest_halts", type(control._pending_acks)())
    a = control.media_halt_mark(USER, media_order=9, media_scope="s-a", media_scope_started_ms=5)
    b = control.media_halt_mark(USER, media_order=1, media_scope="s-b", media_scope_started_ms=5)
    assert a.ordered and b.ordered
    assert control._caller_order(b._event(), a._event()) == 1, "equal stamps: first-seen"
    again = control.media_halt_mark(USER, media_order=2, media_scope="s-a", media_scope_started_ms=5)
    assert again.scope_first_seen == a.scope_first_seen
    assert control._caller_order(again._event(), a._event()) == -1, "same scope: order"
    assert control._caller_order(again._event(), b._event()) == -1
    # Either side unstamped across scopes: no caller order (arrival decides).
    c = control.media_halt_mark(USER, media_order=50, media_scope="s-c")
    assert c.ordered and c.media_scope_started_ms is None
    assert control._caller_order(c._event(), a._event()) is None
    # No usable order or scope: an unordered mark, exactly as before.
    for order, scope in ((None, "s-a"), (3, ""), (True, "s-a"), (-1, "s-a"), ("x", "s-a")):
        assert not control.media_halt_mark(USER, media_order=order, media_scope=scope).ordered


# ══ 4. Unordered requests keep the arrival rule ════════════════════════════

@pytest.mark.parametrize("halt", ["phone_x", "typed_stop_web_close", "legacy_relay_stop"])
async def test_an_unordered_halt_supersedes_an_ordered_play_by_arrival(rig, halt):
    gate = rig.hold("some jazz")
    play = _play("some jazz", order=3, scope=SCOPE)
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    if halt == "phone_x":
        await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "app", "enabled": False})
    elif halt == "typed_stop_web_close":
        await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "web", "enabled": False})
    else:
        assert (await _halt(rig))["ok"] is True
    gate.set()
    out = await asyncio.wait_for(play, timeout=2.0)
    assert (out["ok"], out["reason"]) == (False, "superseded"), out


async def test_the_ordered_stops_own_off_is_not_an_unordered_halt_but_a_later_x_is(rig):
    gate = rig.hold("some jazz")
    play = _play("some jazz", order=3, scope=SCOPE)
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    mark = control.media_halt_mark(USER, media_order=3, media_scope=SCOPE)
    assert (await _halt(rig, before=2, scope=SCOPE))["ok"] is True
    assert "radio_state" in _types(rig.frames), "the stop turned the station OFF"
    assert control.media_play_superseded(mark) is None, "the ordered stop's own OFF"
    await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "app", "enabled": False})
    assert control.media_play_superseded(mark) == "stop", "the phone's X after it"
    gate.set()
    out = await asyncio.wait_for(play, timeout=2.0)
    assert (out["ok"], out["reason"]) == (False, "superseded"), out


async def test_an_ordered_stop_still_supersedes_an_unordered_play_by_arrival(rig):
    """The typed chat's fast path and a legacy relay's play carry no order."""
    gate = rig.hold("some jazz")
    play = _play("some jazz")
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    assert (await _halt(rig, before=2, scope=SCOPE))["ok"] is True
    gate.set()
    out = await asyncio.wait_for(play, timeout=2.0)
    assert (out["ok"], out["reason"]) == (False, "superseded"), out


async def test_a_legacy_relay_with_no_fields_behaves_exactly_as_before(rig):
    gate = rig.hold("some jazz")
    play = _play("some jazz")
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    stop = await _halt(rig)
    assert (stop["ok"], stop["reason"]) == (True, "stopped")
    gate.set()
    assert (await asyncio.wait_for(play, timeout=2.0))["reason"] == "superseded"
    # …and a play asked after that stop still plays.
    assert (await asyncio.wait_for(_play("Halo"), timeout=2.0))["ok"] is True


async def test_an_ordered_stop_before_an_unordered_play_arrives_does_not_block_it(rig):
    assert (await _halt(rig, before=2, scope=SCOPE))["ok"] is True
    assert (await asyncio.wait_for(_play("some jazz"), timeout=2.0))["ok"] is True


def test_the_endpoint_passes_order_fields_only_for_an_ordered_stop_or_pause():
    """A legacy body calls `execute_media_control` exactly as before, and the
    fields are lenient: a malformed value is absent, never a 422."""
    import inspect

    src = inspect.getsource(api_v1.internal_media_control)
    assert "execute_media_control(user_id, req.action, req.channel, **ordered)" in src
    req = MediaControlRequest(user_id="u", action="stop", before_order="x", media_scope=7)
    assert (req.before_order, req.media_scope) == (None, None)
    req = PlayMediaRequest(query="q", media_order=True, media_scope_started_ms=-5)
    assert (req.media_order, req.media_scope_started_ms) == (None, None)
    assert ChatRequest(message="hi").media_order is None


# ══ 5. R6-6: the agent's play is named on tool.start ═══════════════════════

def test_play_media_tool_start_args_carry_the_query():
    assert api_v1._vs_args("play_media", {"query": "Halo Beyonce", "mode": "audio",
                                          "variety": True}) == {"query": "Halo Beyonce"}
    # Other tools' allow-lists are unchanged.
    assert api_v1._VS_ARG_ALLOW["web_search"] == ("query",)
    assert api_v1._VS_ARG_ALLOW["browser"] == ("url", "query")
    assert "exec" not in api_v1._VS_ARG_ALLOW and "write_file" not in api_v1._VS_ARG_ALLOW


# ══ 6. R6-6: the fast-path query residue (media-x1) ════════════════════════

@pytest.mark.parametrize(("text", "query"), [
    # media-x1, verbatim
    ("آره یه پادکست پخش کن", "پادکست"),
    ("آره، یه پادکست هم پخش کن", "پادکست"),
    ("می‌تونی آهنگ هالو از بیانسه رو برام پخش کنی", "هالو بیانسه"),
    # the shapes named for this round
    ("می‌تونی آهنگ بذاری؟", "آهنگ"),
    ("آره پادکست بذار", "پادکست"),
    # the same class and framing in other positions
    ("میتونی یه آهنگ از ابی بذاری؟", "ابی"),
    ("میشه یه آهنگ از ابی بذاری؟", "ابی"),
    ("آهنگ ابی رو می‌تونی بذاری؟", "ابی"),
    ("اوکی، یه آهنگ از ابی بذار", "ابی"),
    ("بله یه آهنگ از شجریان بذار لطفا", "شجریان"),
    ("باشه یه آهنگ بذار مرسی", "آهنگ"),
    ("یه پادکست، آره، پخش کن", "پادکست"),
    ("حتما، آهنگ «ساری گلین» رو پخش کن", "ساری گلین"),
    ("یه آهنگ پخش کن.", "آهنگ"),
    ("can you یه آهنگ از ابی پخش کنی", "ابی"),
])
def test_the_query_is_the_named_content_never_the_framing(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize(("text", "query"), [
    ("یه پادکست پخش کن", "پادکست"),
    ("یه آهنگ پخش کن", "آهنگ"),
    ("play a podcast", "podcast"),
    ("play Bohemian Rhapsody", "Bohemian Rhapsody"),
    # A class word INSIDE the sentence can be a title: only the edges go.
    ("آهنگ سلام آخر رو بذار", "سلام آخر"),
    ("آهنگ سلام رو بذار", "سلام"),
    ("یه آهنگ از ابی پخش کن", "ابی"),
])
def test_controls_keep_their_query(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize("text", [
    # The English arm is the anchored typed-chat grammar, byte for byte: a
    # framed English ask declines to the agent, which resolves the named
    # track — it is never searched as "can you play …".
    "can you play Halo by Beyonce",
    "I want to listen to jazz",
    # Negations never become plays (old: searched «نکن», «نمی», «نه»).
    "آهنگ پخش نکن",
    "نمی‌خوام آهنگ پخش کنی",
    "نه یه پادکست بذار",
    # Nothing but an assent and a verb: nothing to play.
    "آره بذار",
])
def test_these_decline_to_the_agent(text):
    assert media_request(text) is None, text


def test_the_relay_reads_the_same_verdict():
    """One module for both fast paths: the relay's `_media_request` IS
    `media_intent.media_request`."""
    from app.services import live_voice_protocol as lvp

    for text in ("آره یه پادکست پخش کن", "می‌تونی آهنگ هالو از بیانسه رو برام پخش کنی",
                 "can you play Halo by Beyonce"):
        assert lvp._media_request(text) == media_request(text), text


def test_one_class_two_grammars_the_residue_class_is_the_relays():
    """The assent / content-free class the residue drops is the relay's
    closed class, word for word (compared through the relay's own fold)."""
    from app.services import live_voice_protocol as lvp

    def relay_words(text):
        return tuple(lvp._leading_words(text))

    assert {relay_words(w)[0] for w in media_intent.ASSENT_WORDS_TEXT.split()} == set(lvp._ASSENT_TOKENS)
    assert {relay_words(p) for p in media_intent.ASSENT_PHRASES_TEXT} == set(lvp._ASSENT_PHRASES)
    assert {relay_words(w)[0] for w in media_intent.CONTENT_FREE_WORDS_TEXT.split()} \
        == set(lvp._DISCOURSE_FILLERS)


async def test_the_typed_chat_fast_path_searches_the_named_content(rig):
    """The typed half of the same module (`ws_chat._fast_media_check`)."""
    for text, query in (
        ("آره یه پادکست پخش کن", "پادکست"),
        ("می‌تونی آهنگ هالو از بیانسه رو برام پخش کنی", "هالو بیانسه"),
    ):
        q: asyncio.Queue = asyncio.Queue()
        result = await asyncio.wait_for(ws_chat._fast_media_check(text, USER, q), timeout=2.0)
        assert result is not None and result[1] is not None, (text, result)
        assert rig.searches[-1] == query, (text, rig.searches)
    q = asyncio.Queue()
    assert await ws_chat._fast_media_check("can you play Halo by Beyonce", USER, q) is None
