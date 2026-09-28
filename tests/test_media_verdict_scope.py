"""A stop/pause verdict speaks only for the devices that answered (review F27).

Supervisor follow-up to F27. The peer grace ended the 7 s dead interval, but a
verdict built from partial acks could still say "Nothing was playing": one idle
ack ("was_playing": false) beside a chat socket that never answered came back
ok=True / nothing_playing after ~1.55 s. That silent socket may be the player
(an old build, a second device), and radio OFF does not stop a track already
playing — so the claim could be false while the music went on.

The rule now (`radio.control` + `ws_chat` socket evidence):

* `nothing_playing` needs an answer from EVERY reached socket that could be
  playing the phone's channel. With one silent, an all-idle answer is
  `partially_confirmed` (ok=false, additive reason; the station is still OFF).
* A socket proven unable to play that channel is left out — proven only by its
  own frames: it declared channel 'web' / 'vibecoding' (the web ChatPage, which
  drops every channel-'app' frame) or 'extension' (no radio at all). A socket
  that has shown nothing is NOT assumed to be a web tab.
* No new long waits: a device that stopped real playback settles at once; a
  socket that has shown nothing gets the 1.5 s peer grace after an idle answer;
  only a LIVE socket that has answered before on this connection is waited for
  to the deadline (its answer is coming — a late player still wins); the
  budget is unchanged.

Plus the remaining F29 tenant gaps (addendum-2 item 9): the typed fast path's
own media_play, a stop while the agent is still thinking (run-level mark), and
`on_tool_event` naming a superseded play `cancelled`.

Real path throughout: `execute_media_control`, the real `_handle_radio_toggle`,
real `_register_ws_queue` + `broadcast_to_user`, acks through
`ws_chat._dispatch_radio_frame` / `_handle_media_ack`; for plays, the real
`internal_agent_turn_stream` / `internal_agent_turn` / `_tool_play_media` /
`_fast_media_check` with only the YouTube search stubbed.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from types import SimpleNamespace

import pytest

from app.agent.radio import control
from app.agent.radio.session import RadioSessionManager
from app.api import api_v1, ws_chat
from app.config import settings

pytestmark = pytest.mark.asyncio

USER = "owner-verdict-scope"
VIDEO = "JAZZ0000001"


def _note(queue, frame) -> None:
    """A socket's own inbound frame, as the receive loop records it."""
    note = getattr(ws_chat, "_note_socket_frame", None)
    if note is not None:
        note(queue, frame)


async def _ack_via_socket(user, msg, source):
    """An ack arriving on a chat socket: the watcher's routing, with the socket."""
    try:
        await ws_chat._dispatch_radio_frame(user, msg, source=source)
    except TypeError:  # a tenant whose routing cannot say which socket
        await ws_chat._dispatch_radio_frame(user, msg)


# ── Rig: real queues, real broadcast, real OFF ───────────────────────────

@pytest.fixture
def rig(monkeypatch):
    import app.agent.radio as radio

    manager = RadioSessionManager()
    monkeypatch.setattr(radio, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(control, "get_radio_manager", lambda: manager)
    for name in (
        "_media_ended_locks", "_radio_toggle_locks", "_radio_off_generations",
        "_display_mode_locks", "_user_ws_queues", "_recent_broadcasts",
    ):
        monkeypatch.setattr(ws_chat, name, {})
    monkeypatch.setattr(control, "_pending_acks", type(control._pending_acks)())
    captured: list = []
    real_register = control._register_ack

    def spy(command_id, user_id, ack_type):
        pending = real_register(command_id, user_id, ack_type)
        captured.append(pending)
        return pending

    monkeypatch.setattr(control, "_register_ack", spy)
    return SimpleNamespace(pending=captured, manager=manager)


def _socket(user: str = USER) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    ws_chat._register_ws_queue(user, queue)
    return queue


class _Device:
    """A phone-app socket: acts on channel-'app' media_stop / media_pause and
    answers after `delay` (attributed to its socket unless `anonymous`)."""

    def __init__(self, user, queue, *, delay, done=True, was_playing=False, anonymous=False):
        self.user, self.queue, self.delay = user, queue, delay
        self.done, self.was_playing, self.anonymous = done, was_playing, anonymous
        self.seen: list = []
        self.answering: list = []
        self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        while True:
            msg = await self.queue.get()
            if msg.get("channel") not in (None, "app"):
                continue
            self.seen.append(msg.get("type"))
            if msg.get("type") in ("media_stop", "media_pause"):
                self.answering.append(asyncio.create_task(self._answer(msg)))

    async def _answer(self, msg):
        await asyncio.sleep(self.delay)
        key = "stopped" if msg["type"] == "media_stop" else "paused"
        ack = {
            "type": msg["type"] + "_ack", "command_id": msg["command_id"],
            key: self.done, "was_playing": self.was_playing, "channel": "app",
        }
        if self.anonymous:
            await ws_chat._dispatch_radio_frame(self.user, ack)
        else:
            await _ack_via_socket(self.user, ack, self.queue)


class _Silent:
    """A socket that receives everything and never answers (an old build that
    may be the player, a web tab that has sent nothing yet, a stale queue)."""

    def __init__(self, queue, my_channel="app"):
        self.queue, self.my_channel, self.seen = queue, my_channel, []
        self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        while True:
            msg = await self.queue.get()
            if msg.get("channel") not in (None, self.my_channel):
                continue
            self.seen.append(msg.get("type"))


async def _control(action="stop", timeout=None, user=USER):
    loop = asyncio.get_running_loop()
    started = loop.time()
    out = await asyncio.wait_for(
        control.execute_media_control(user, action, "app", ack_timeout_s=timeout),
        timeout=12.0,
    )
    return out, loop.time() - started


def _stop_all(*actors):
    for actor in actors:
        actor.task.cancel()
        for task in getattr(actor, "answering", ()):
            task.cancel()


def _verdict(out):
    return (out.ok, out.reason, out.was_playing)


def _counts(out):
    """(acked, silent) — additive fields; None on a tenant without them."""
    if not hasattr(out, "acked_devices"):
        return None
    return (out.acked_devices, out.silent_devices)


async def _make_answerer(user, *queues):
    """Each socket answers one earlier command on this connection, the way a
    current phone build does — which is what makes it a known answerer."""
    devices = [_Device(user, q, delay=0.01, was_playing=False) for q in queues]
    out, _ = await _control("pause")
    _stop_all(*devices)
    assert out.reason == "nothing_playing", out
    await asyncio.sleep(0)


# ── 1. The supervisor's probe scenario ───────────────────────────────────

@pytest.mark.parametrize("anonymous", [True, False], ids=["probe-routing", "socket-routing"])
async def test_an_idle_ack_beside_a_silent_app_socket_is_never_nothing_playing(rig, anonymous):
    """reverify-w5-tenant `test_D_idle_phone_plus_old_build_app_device_default_budget`,
    at the production budget: one idle ack plus a channel-'app' socket that
    received the stop and never answered. It may be the player — so the
    answer is partial, promptly, and never "Nothing was playing"."""
    assert control._default_ack_timeout_s() > 6.0
    phone = _Device(USER, _socket(), delay=0.05, was_playing=False, anonymous=anonymous)
    old_build = _Silent(_socket(), "app")
    out, waited = await _control("stop")
    _stop_all(phone, old_build)

    assert "media_stop" in old_build.seen, "the silent socket got the command"
    assert (out.ok, out.reason) != (True, "nothing_playing")
    assert _verdict(out) == (False, "partially_confirmed", None)
    assert _counts(out) == (1, 1)
    assert waited < 2.5, f"held {waited:.2f}s"
    # The station is still OFF: the stop happened, only the claim is scoped.
    assert ws_chat._radio_off_generations[(USER, "app")] == 1
    assert control._pending_acks == {}
    assert rig.pending[0].grace is not None and rig.pending[0].grace.cancelled()


async def test_the_same_for_pause(rig):
    phone = _Device(USER, _socket(), delay=0.05, was_playing=False)
    silent = _Silent(_socket(), "app")
    out, waited = await _control("pause")
    _stop_all(phone, silent)
    assert _verdict(out) == (False, "partially_confirmed", None)
    assert waited < 2.5
    assert ws_chat._radio_off_generations == {}, "a pause leaves the station alone"


# ── 2. One device: settles at once ───────────────────────────────────────

@pytest.mark.parametrize("action,reason", [("stop", "stopped"), ("pause", "paused")])
async def test_a_single_device_that_stopped_is_confirmed_immediately(rig, action, reason):
    phone = _Device(USER, _socket(), delay=0.05, was_playing=True)
    out, waited = await _control(action)
    _stop_all(phone)
    assert _verdict(out) == (True, reason, True)
    assert _counts(out) in (None, (1, 0))
    assert waited < 0.5


async def test_a_single_idle_device_is_nothing_playing_immediately(rig):
    phone = _Device(USER, _socket(), delay=0.05, was_playing=False)
    out, waited = await _control("stop")
    _stop_all(phone)
    assert _verdict(out) == (True, "nothing_playing", False)
    assert waited < 0.5


async def test_a_player_beside_a_silent_socket_still_stops_immediately(rig):
    phone = _Device(USER, _socket(), delay=0.05, was_playing=True)
    silent = _Silent(_socket(), "app")
    out, waited = await _control("stop")
    _stop_all(phone, silent)
    assert _verdict(out) == (True, "stopped", True)
    assert waited < 0.5


# ── 3. Two devices, one idle, one the player ─────────────────────────────

async def test_a_known_answerer_that_is_the_player_is_awaited_past_the_grace(rig):
    """Both devices have answered before on this connection. The idle one
    answers at 0.05 s, the player at 1.8 s — after the 1.5 s grace but inside
    the budget. The player's answer is coming, so it is waited for, and the
    verdict is its `stopped`."""
    idle_q, player_q = _socket(), _socket()
    await _make_answerer(USER, idle_q, player_q)

    idle = _Device(USER, idle_q, delay=0.05, was_playing=False)
    player = _Device(USER, player_q, delay=1.8, was_playing=True)
    out, waited = await _control("stop")
    _stop_all(idle, player)

    assert _verdict(out) == (True, "stopped", True)
    assert 1.7 < waited < 3.0
    assert rig.pending[-1].grace is None, "no grace while a known answerer is silent"
    assert control._pending_acks == {}


async def test_an_unproven_player_answering_inside_the_grace_wins(rig):
    idle = _Device(USER, _socket(), delay=0.05, was_playing=False)
    player = _Device(USER, _socket(), delay=0.5, was_playing=True)
    out, waited = await _control("stop")
    _stop_all(idle, player)
    assert _verdict(out) == (True, "stopped", True)
    assert waited < 1.0


async def test_an_unproven_player_answering_after_the_grace_is_never_nothing_playing(rig):
    """The reverify NEW trade-off (player at 1.8 s gave nothing_playing): a
    socket that has shown nothing gets the grace, and when it is still silent
    the verdict says so. Its late answer then settles nothing."""
    idle = _Device(USER, _socket(), delay=0.05, was_playing=False)
    player = _Device(USER, _socket(), delay=1.8, was_playing=True)
    out, waited = await _control("stop")
    assert _verdict(out) == (False, "partially_confirmed", None)
    assert waited < 1.8
    await asyncio.sleep(0.5)  # the player's ack lands on a finished command
    _stop_all(idle, player)
    assert control._pending_acks == {}


# ── 4. Nobody answers ────────────────────────────────────────────────────

async def test_all_silent_is_unacknowledged(rig):
    a, b = _Silent(_socket(), "app"), _Silent(_socket(), "app")
    out, waited = await _control("stop", timeout=0.3)
    _stop_all(a, b)
    assert _verdict(out) == (False, "unacknowledged", None)
    assert _counts(out) in (None, (0, 2))
    assert 0.25 < waited < 0.8
    assert control._pending_acks == {}


# ── 5. Web / extension peers: the capability rule ────────────────────────

@pytest.mark.parametrize("surface", ["web", "vibecoding", "extension", " Web "])
async def test_a_socket_that_declared_a_non_player_surface_is_left_out(rig, surface):
    """The web ChatPage sends its frames on channel 'web' (or 'vibecoding')
    and drops every channel-'app' frame; the extension sends 'extension' and
    handles no radio. Once a socket has said so, it cannot be the phone's
    player: the idle phone's answer is the whole truth, with no grace wait."""
    phone = _Device(USER, _socket(), delay=0.05, was_playing=False)
    web_q = _socket()
    _note(web_q, {"type": "media_ended", "video_id": "x", "channel": surface})
    web = _Silent(web_q, "web")
    out, waited = await _control("stop")
    _stop_all(phone, web)
    assert _verdict(out) == (True, "nothing_playing", False)
    assert waited < 0.5
    assert _counts(out) == (1, 0)


async def test_a_web_only_audience_is_delivery_failed_without_waiting(rig):
    """Only a declared web tab is connected: nothing that can play the
    phone's channel got the command. Said at once (no budget wait), never
    `nothing_playing`, and the station is still turned OFF."""
    web_q = _socket()
    _note(web_q, {"type": "message", "text": "hi", "channel": "web"})
    web = _Silent(web_q, "web")
    out, waited = await _control("stop")
    _stop_all(web)
    assert _verdict(out) == (False, "delivery_failed", None)
    assert waited < 0.5
    assert ws_chat._radio_off_generations[(USER, "app")] == 1


async def test_a_web_tab_that_has_sent_nothing_is_not_assumed_to_be_one(rig):
    """No frame, no evidence: the same socket could be an old build that is
    playing. Partial, promptly."""
    phone = _Device(USER, _socket(), delay=0.05, was_playing=False)
    tab = _Silent(_socket(), "web")
    out, waited = await _control("stop")
    _stop_all(phone, tab)
    assert _verdict(out) == (False, "partially_confirmed", None)
    assert waited < 2.5


@pytest.mark.parametrize("channel", ["app", "mobile", "routine", None, 7])
async def test_frames_on_a_phone_channel_never_exempt_a_socket(rig, channel):
    """'app' is what the phone sends (and the web ChatPage inside an app
    context, which still drops 'app' frames — so it stays unproven, the
    conservative side); 'mobile' is the phone's chat channel."""
    phone = _Device(USER, _socket(), delay=0.05, was_playing=False)
    other_q = _socket()
    _note(other_q, {"type": "radio_toggle", "enabled": True, "channel": channel})
    other = _Silent(other_q, "app")
    out, _ = await _control("stop")
    _stop_all(phone, other)
    assert out.reason == "partially_confirmed"


@pytest.mark.parametrize("was_playing,expect,acked", [
    (False, "nothing_playing", 2), (True, "stopped", 1),
])
async def test_a_declared_web_socket_that_answers_anyway_is_counted(rig, was_playing, expect, acked):
    """Evidence is never allowed to hide an answer: a socket that declared
    'web' but answers before the verdict is a device, and its answer counts
    (a real stop settles at once; an idle one waits for the phone too)."""
    phone = _Device(USER, _socket(), delay=0.1, was_playing=False)
    odd_q = _socket()
    _note(odd_q, {"type": "message", "text": "hi", "channel": "web"})
    odd = _Device(USER, odd_q, delay=0.02, was_playing=was_playing)
    out, waited = await _control("stop")
    _stop_all(phone, odd)
    assert out.reason == expect
    assert waited < 0.5
    assert (_counts(out) or (acked,))[0] == acked


async def test_a_known_answerer_that_went_quiet_is_not_waited_for(rig):
    """A socket that answered once but has sent nothing for longer than the
    phone's keepalive may be half-open (the phone reconnected): it still
    counts as possibly playing, but gets the grace, not the whole budget."""
    stale_q = _socket()
    await _make_answerer(USER, stale_q)
    phone_q = _socket()
    evidence = getattr(ws_chat, "_socket_media_evidence", {}).get(stale_q)
    if evidence is not None:  # a tenant without socket evidence has nothing to age
        evidence.last_inbound = (
            time.monotonic() - getattr(ws_chat, "_MEDIA_ANSWERER_LIVE_S", 40.0) - 5
        )
    phone = _Device(USER, phone_q, delay=0.05, was_playing=False)
    stale = _Silent(stale_q, "app")
    out, waited = await _control("stop")
    _stop_all(phone, stale)
    assert _verdict(out) == (False, "partially_confirmed", None)
    assert waited < 2.5


async def test_a_known_answerer_silent_to_the_deadline_is_not_awaited_again(rig):
    gone_q = _socket()
    await _make_answerer(USER, gone_q)
    phone_q = _socket()
    gone = _Silent(gone_q, "app")
    phone = _Device(USER, phone_q, delay=0.05, was_playing=False)

    out, waited = await _control("stop", timeout=0.6)
    assert _verdict(out) == (False, "partially_confirmed", None)
    assert waited >= 0.55, "a known answerer is waited for to the deadline"
    assert getattr(ws_chat, "_socket_media_evidence", {})[gone_q].answers is False

    out2, waited2 = await _control("stop", timeout=5.0)
    _stop_all(phone, gone)
    assert out2.reason == "partially_confirmed"
    assert waited2 < 2.5, "demoted: only the grace now"


# ── 6. Repeat / replay safety ────────────────────────────────────────────

async def test_one_socket_answering_twice_counts_once(rig):
    phone_q = _socket()
    silent = _Silent(_socket(), "app")

    async def double_ack():
        while True:
            msg = await phone_q.get()
            if msg.get("type") == "media_stop":
                ack = {"type": "media_stop_ack", "command_id": msg["command_id"],
                       "stopped": True, "was_playing": False}
                await _ack_via_socket(USER, ack, phone_q)
                await _ack_via_socket(USER, dict(ack), phone_q)

    phone = asyncio.create_task(double_ack())
    out, _ = await _control("stop")
    phone.cancel()
    _stop_all(silent)
    assert _verdict(out) == (False, "partially_confirmed", None)
    assert _counts(out)[0] == 1


async def test_each_command_is_sent_once_and_leaves_nothing_behind(rig):
    phone = _Device(USER, _socket(), delay=0.02, was_playing=True)
    first, _ = await _control("stop")
    second, _ = await _control("stop")
    _stop_all(phone)
    assert first.command_id != second.command_id
    assert phone.seen.count("media_stop") == 2, phone.seen
    assert control._pending_acks == {}
    # A replayed ack for a finished command settles nothing.
    assert control.deliver_media_ack(USER, {
        "type": "media_stop_ack", "command_id": first.command_id,
        "stopped": True, "was_playing": True,
    }) is False


async def test_a_cancelled_waiter_leaves_no_registry_entry_or_timer(rig):
    phone = _Device(USER, _socket(), delay=0.05, was_playing=False)
    silent = _Silent(_socket(), "app")
    task = asyncio.create_task(control.execute_media_control(USER, "stop", "app"))
    for _ in range(200):
        if rig.pending and rig.pending[0].grace is not None:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _stop_all(phone, silent)
    assert control._pending_acks == {}
    assert rig.pending[0].grace.cancelled()


# ── 7. The socket handler records the evidence ───────────────────────────

def test_the_receive_loop_records_every_frame_and_attributes_acks():
    src = inspect.getsource(ws_chat.ws_chat)
    i_type = src.index('msg_type = msg.get("type", "")')
    i_note = src.index("_note_socket_frame(broadcast_queue, msg)")
    i_ping = src.index('if msg_type == "ping":')
    assert i_type < i_note < i_ping, "recorded before any branch can `continue`"
    assert "_handle_media_ack(user_id, msg, source=broadcast_queue)" in src
    assert "_handle_media_ack(user_id, msg)\n" not in src


def test_an_ack_on_a_real_socket_makes_it_a_known_answerer():
    """Over the in-process ASGI socket: the phone's ack settles the stop and
    the socket's queue is recorded as one that answers."""
    import test_ws_chat_infra_fault_close as wsh
    from app.agent.radio.session import RadioSessionManager as _RSM

    async def go():
        mp = pytest.MonkeyPatch()
        sock = None
        try:
            import app.agent.radio as radio_pkg

            manager = _RSM()
            mp.setattr(radio_pkg, "get_radio_manager", lambda: manager)
            mp.setattr(control, "get_radio_manager", lambda: manager)
            mp.setattr(ws_chat, "_radio_toggle_locks", {})
            mp.setattr(ws_chat, "_radio_off_generations", {})
            mp.setattr(ws_chat, "_media_ended_locks", {})
            mp.setattr(ws_chat, "_user_ws_queues", {})
            mp.setattr(ws_chat, "_recent_broadcasts", {})
            mp.setattr(control, "_pending_acks", type(control._pending_acks)())
            ws_chat._active_turns.pop(wsh.USER_ID, None)

            sock = await wsh._authenticated_socket(mp)
            # The web ChatPage's own frame: a track ended on its 'web' player.
            sock.send_text(json.dumps({"type": "media_ended", "video_id": "v", "channel": "web"}))
            sock.send_text(json.dumps({"type": "ping"}))
            while (await wsh._next_frame(sock)).get("type") != "pong":
                pass
            queues = list(ws_chat._user_ws_queues.get(wsh.USER_ID, []))
            assert len(queues) == 1
            evidence = ws_chat._socket_media_evidence[queues[0]]
            assert evidence.surface == "web" and evidence.answers is False

            pending = asyncio.create_task(control.execute_media_control(
                wsh.USER_ID, "stop", "app", ack_timeout_s=3.0,
            ))
            outcome = await asyncio.wait_for(pending, timeout=2.0)
            # Only a declared web socket was connected: nothing to wait for.
            assert (outcome.ok, outcome.reason) == (False, "delivery_failed")

            # The same socket turns out to answer media commands after all.
            sock.send_text(json.dumps({
                "type": "media_stop_ack", "command_id": "stray", "stopped": True,
            }))
            sock.send_text(json.dumps({"type": "ping"}))
            while (await wsh._next_frame(sock)).get("type") != "pong":
                pass
            assert evidence.answers is True
            assert ws_chat.media_command_audience(wsh.USER_ID)[0][1] == "answerer"
        finally:
            ws_chat._active_turns.pop(wsh.USER_ID, None)
            if sock is not None:
                await sock.dispose()
            mp.undo()

    asyncio.run(go())


# ── 8. F29: the typed fast path's own media_play ─────────────────────────

@pytest.fixture
def slow_search(monkeypatch, tmp_path):
    """The YouTube search blocks until released; everything else is real."""
    import httpx

    import app.agent.radio as radio_pkg
    import app.agent.tool_executor as te
    from app.agent import media_resolve
    from app.agent.radio import player
    from app.agent.tool_executor import ToolExecutor

    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "verdict-agent-key")
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

    frames: list = []

    async def broadcast(user_id, event, exclude=None, **kw):
        frames.append(dict(event))
        return 1

    async def no_swap(*a, **kw):
        return None

    monkeypatch.setattr(ws_chat, "broadcast_to_user", broadcast)
    monkeypatch.setattr(ws_chat, "_check_age_and_swap", no_swap)
    monkeypatch.setattr(player, "warm_audio_cache", lambda *a, **kw: None)

    gate, searching = asyncio.Event(), asyncio.Event()

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
    monkeypatch.setattr(media_resolve, "scrape_results", lambda html, limit=6: [(VIDEO, "Jazz Mix", "")])
    monkeypatch.setattr(te, "_MEDIA_RESOLVE_CACHE", {})
    tools = ToolExecutor(workspace=str(tmp_path))
    return SimpleNamespace(frames=frames, gate=gate, searching=searching, tools=tools, manager=manager)


async def _voice_stop(frames, action="stop"):
    pending = asyncio.create_task(
        control.execute_media_control(USER, action, "app", ack_timeout_s=2.0),
    )
    frame_type = f"media_{action}"
    for _ in range(400):
        if any(f["type"] == frame_type for f in frames):
            break
        await asyncio.sleep(0.005)
    command_id = [f for f in frames if f["type"] == frame_type][-1]["command_id"]
    key = "paused" if action == "pause" else "stopped"
    control.deliver_media_ack(USER, {
        "type": f"{frame_type}_ack", "command_id": command_id, key: True, "was_playing": True,
    })
    return await asyncio.wait_for(pending, timeout=2.0)


def _drain(queue):
    return [queue.get_nowait() for _ in range(queue.qsize())]


@pytest.mark.parametrize("halt", ["voice_stop", "voice_pause", "phone_x"])
async def test_the_typed_fast_path_drops_a_play_stopped_during_its_search(slow_search, halt):
    q: asyncio.Queue = asyncio.Queue()
    play = asyncio.create_task(ws_chat._fast_media_check("play some jazz", USER, q))
    await asyncio.wait_for(slow_search.searching.wait(), timeout=2.0)

    if halt == "phone_x":
        await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "app", "enabled": False})
    else:
        out = await _voice_stop(slow_search.frames, "pause" if halt == "voice_pause" else "stop")
        assert out.ok is True

    slow_search.gate.set()
    result = await asyncio.wait_for(play, timeout=2.0)
    sent = [f["type"] for f in _drain(q)]
    assert "media_play" not in sent, sent
    assert result is not None, "handled: the agent must be told, not handed the play"
    text, meta = result
    assert meta is None
    assert "NOT started" in text and "Do NOT call play_media" in text
    assert "Jazz Mix" not in text


async def test_the_typed_fast_path_honours_the_mark_taken_on_arrival(slow_search):
    """The X tapped between the message arriving and the fast path starting."""
    mark = control.media_halt_mark(USER)
    await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "app", "enabled": False})
    slow_search.gate.set()
    q: asyncio.Queue = asyncio.Queue()
    result = await ws_chat._fast_media_check("play some jazz", USER, q, halt_mark=mark)
    assert "media_play" not in [f["type"] for f in _drain(q)]
    assert result is not None and result[1] is None


async def test_the_typed_fast_path_still_plays_when_nothing_halted(slow_search):
    await ws_chat._handle_radio_toggle(USER, {"type": "radio_toggle", "channel": "app", "enabled": False})
    slow_search.gate.set()
    q: asyncio.Queue = asyncio.Queue()
    text, meta = await ws_chat._fast_media_check("play some jazz", USER, q)
    assert [f["type"] for f in _drain(q)] == ["media_play"]
    assert meta and meta["video_id"] == VIDEO
    assert "Jazz Mix" in text


def test_the_message_branch_marks_on_arrival_and_gates_card_seed_and_station():
    src = inspect.getsource(ws_chat.ws_chat)
    i_text = src.index('text = msg.get("text", "").strip()')
    i_mark = src.index("_play_halt_mark = _media_halt_mark_now(user_id)")
    i_pre = src.index("_pt = _PreTurn(_t_frame)")
    assert i_text < i_mark < i_pre, "the mark is taken when the message arrives"
    assert "halt_mark=_play_halt_mark" in src
    # A dropped play has no meta: no card, no seed, no auto-started station.
    assert "_agent_runner.tools._last_media = _fast_meta" in src
    assert "if _fast_meta:\n" in src
    assert 'if _fast_meta and channel == "mobile":' in src
    assert "preset_media=_fast_meta," in src
    assert "_agent_runner.tools._last_media = _fast_result[1]" not in src
    # The run task inherits the same mark: set before create_task, reset after.
    i_bind = src.index("_halt_ctx_token = _bind_run_halt_mark(_play_halt_mark)")
    i_task = src.index("agent_task = asyncio.create_task(_agent_runner.run(")
    i_reset = src.index("_reset_run_halt_mark(_halt_ctx_token)")
    assert i_bind < i_task < i_reset


async def test_a_typed_run_task_inherits_the_arrival_mark_and_the_handler_does_not_keep_it():
    """The message branch's own pattern: bind, create the run task, reset."""
    mark = control.media_halt_mark(USER)
    token = ws_chat._bind_run_halt_mark(mark)

    async def run():
        await asyncio.sleep(0)
        return control.run_halt_mark(USER)

    task = asyncio.create_task(run())
    ws_chat._reset_run_halt_mark(token)
    assert control.run_halt_mark(USER) is None, "the handler's context is clean"
    assert await task is mark
    assert control.run_halt_mark("someone-else") is None


# ── 9. F29: a stop while the agent is still thinking ─────────────────────

class _Req:
    headers = {"X-Agent-Key": "verdict-agent-key"}


class _ThinkingRunner:
    """The agent: 'thinks' until released, then calls the real play_media tool
    and reports it through the endpoint's own on_tool_event."""

    def __init__(self, tools):
        self.tools = tools
        self.thinking = asyncio.Event()
        self.release = asyncio.Event()
        self.result = None

    async def run(self, *, user_id, on_tool_event=None, **kw):
        self.thinking.set()
        await self.release.wait()
        self.tools.set_user_id(user_id)
        self.tools.set_channel("voice")
        if on_tool_event:
            await on_tool_event({"phase": "start", "name": "play_media", "call_id": "c1",
                                 "input": {"query": "some jazz"}})
        self.result = await self.tools._tool_play_media({"query": "some jazz"})
        if on_tool_event:
            await on_tool_event({"phase": "end", "name": "play_media", "call_id": "c1",
                                 "input": {"query": "some jazz"}, "result": self.result,
                                 "elapsed_ms": 5})
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


async def test_a_stop_while_the_agent_thinks_supersedes_its_later_play(slow_search, monkeypatch):
    runner = _ThinkingRunner(slow_search.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    slow_search.gate.set()  # the search itself is instant: the stop is BEFORE it

    stream = await api_v1.internal_agent_turn_stream(api_v1.ChatRequest(message="play some jazz", save=False), _Req())
    collecting = asyncio.create_task(_collect(stream))
    await asyncio.wait_for(runner.thinking.wait(), timeout=2.0)

    out = await _voice_stop(slow_search.frames)
    assert out.ok is True
    runner.release.set()
    frames = await asyncio.wait_for(collecting, timeout=5.0)

    assert runner.result.startswith(getattr(control, "PLAY_SUPERSEDED_PREFIX", "SUPERSEDED:")), runner.result
    assert [f["type"] for f in slow_search.frames] == ["radio_state", "media_stop"]
    end = [f for f in frames if f.get("type") == "tool.end"]
    assert len(end) == 1 and end[0]["name"] == "play_media"
    assert end[0]["ok"] is False and end[0]["outcome"] == "cancelled"


async def test_a_stop_before_the_request_does_not_block_the_agents_play(slow_search, monkeypatch):
    out = await _voice_stop(slow_search.frames)
    assert out.ok is True
    runner = _ThinkingRunner(slow_search.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    slow_search.gate.set()
    runner.release.set()

    stream = await api_v1.internal_agent_turn_stream(api_v1.ChatRequest(message="play some jazz", save=False), _Req())
    frames = await asyncio.wait_for(_collect(stream), timeout=5.0)

    assert runner.result.startswith("Now playing"), runner.result
    assert slow_search.frames[-1]["type"] == "media_play"
    end = [f for f in frames if f.get("type") == "tool.end"][0]
    assert end["ok"] is True and "outcome" not in end


async def test_the_blocking_think_route_binds_the_same_run_mark(slow_search, monkeypatch):
    runner = _ThinkingRunner(slow_search.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    slow_search.gate.set()

    async def call():
        await api_v1.internal_agent_turn(
            api_v1.ChatRequest(message="play some jazz", save=False), _Req(),
        )
        # Same task as the route: nothing may stay bound after it returns.
        return getattr(control, "run_halt_mark", lambda _u: None)(USER)

    turn = asyncio.create_task(call())
    await asyncio.wait_for(runner.thinking.wait(), timeout=2.0)
    await _voice_stop(slow_search.frames, "pause")
    runner.release.set()
    left_bound = await asyncio.wait_for(turn, timeout=5.0)
    assert runner.result.startswith(getattr(control, "PLAY_SUPERSEDED_PREFIX", "SUPERSEDED:")), runner.result
    assert "media_play" not in [f["type"] for f in slow_search.frames]
    assert left_bound is None, "the route leaves no mark behind"


# ── 10. on_tool_event names a superseded play `cancelled` ────────────────

class _EventRunner:
    def __init__(self, events):
        self.events = events

    async def run(self, *, on_tool_event=None, **kw):
        for ev in self.events:
            await on_tool_event(ev)
        return SimpleNamespace(
            text="", session_id="s", tokens_input=0, tokens_output=0, tokens_total=0,
            model="m", tool_calls=[], processing_time_ms=1, persisted={},
        )


async def test_on_tool_event_outcome_is_scoped_to_a_superseded_play(monkeypatch):
    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "verdict-agent-key")
    monkeypatch.setattr(settings, "user_id", USER)
    prefix = getattr(control, "PLAY_SUPERSEDED_PREFIX", "SUPERSEDED:")
    ends = [
        ("play_media", f"{prefix} not started. The user stopped the music after asking for this."),
        ("play_media", 'Now playing "Jazz Mix"'),
        ("play_media", "ERROR: could not find that track"),
        ("web_search", f"{prefix} a page whose text happens to start like this"),
    ]
    monkeypatch.setattr(api_v1, "_agent_runner", _EventRunner([
        {"phase": "end", "name": name, "call_id": f"c{i}", "input": {}, "result": result}
        for i, (name, result) in enumerate(ends)
    ]))
    stream = await api_v1.internal_agent_turn_stream(api_v1.ChatRequest(message="x", save=False), _Req())
    frames = [f for f in await asyncio.wait_for(_collect(stream), timeout=5.0) if f["type"] == "tool.end"]

    got = [(f["name"], f["ok"], f.get("outcome")) for f in frames]
    assert got == [
        ("play_media", False, "cancelled"),
        ("play_media", True, None),
        ("play_media", False, None),
        ("web_search", True, None),
    ]
