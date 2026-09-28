"""The chat socket carries a stop in both directions, even mid-turn.

Two chat-WS gaps sat under the "music would not stop" recordings (V2+03:17,
V3+00:09; media repair M4/M5):

1. The phone's answer to a tenant `media_stop` / `media_pause`
   (`media_stop_ack` / `media_pause_ack`) had nowhere to go. On an idle socket
   the receive loop answered it "Unknown message type"; during a streaming
   turn the stop-watcher read it and dropped it — so a voice stop could never
   be confirmed while a typed turn happened to be running.
2. The stop-watcher that owns the socket's receive stream during a turn drops
   every frame it does not forward, and `radio_toggle` is not forwarded
   because an ON mid-turn races the agent's own `media_play` into two stations.
   That reason applies to ON only: an OFF (the player's X, a stop) was silently
   eaten and the station kept going.
"""
from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from app.agent.radio import control
from app.agent.radio.session import RadioSession, RadioSessionManager, SeedTrack
from app.agent.radio.playlist import StationTrack
from app.api import ws_chat


USER = "owner-ws-stop"


def _enabled_session(user_id: str = USER) -> RadioSession:
    session = RadioSession(user_id=user_id, channel="app", enabled=True)
    session.seed_track = SeedTrack(video_id="seed", title="Seed")
    session.mark_current_track("seed")
    session.current_station_track = StationTrack(video_id="seed", title="Seed")
    session.playlist = [StationTrack(video_id="next-0", title="Next 0")]
    return session


@pytest.fixture
def radio(monkeypatch):
    import app.agent.radio as radio_pkg

    manager = RadioSessionManager()
    monkeypatch.setattr(radio_pkg, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(control, "get_radio_manager", lambda: manager)
    monkeypatch.setattr(ws_chat, "_radio_toggle_locks", {})
    monkeypatch.setattr(ws_chat, "_radio_off_generations", {})
    monkeypatch.setattr(ws_chat, "_media_ended_locks", {})
    monkeypatch.setattr(control, "_pending_acks", type(control._pending_acks)())
    return manager


@pytest.fixture
def wire(monkeypatch):
    sent: list[dict] = []

    async def broadcast(user_id, event, exclude=None):
        sent.append(dict(event))
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", broadcast)
    return sent


# ── What the stop-watcher forwards ───────────────────────────────────────

def test_radio_off_is_forwarded_while_a_turn_streams():
    assert ws_chat._is_mid_turn_passthrough(
        {"type": "radio_toggle", "enabled": False, "channel": "app"},
    )


def test_radio_on_is_still_not_forwarded_mid_turn():
    # The reason radio_toggle was excluded — ON racing the agent's own
    # media_play into two stations — still holds for ON.
    assert not ws_chat._is_mid_turn_passthrough(
        {"type": "radio_toggle", "enabled": True, "channel": "app"},
    )
    assert "radio_toggle" not in ws_chat._MID_TURN_PASSTHROUGH


@pytest.mark.parametrize("frame_type", ["media_stop_ack", "media_pause_ack"])
def test_media_acks_are_forwarded_mid_turn(frame_type):
    assert ws_chat._is_mid_turn_passthrough({"type": frame_type, "command_id": "c"})


def test_the_socket_and_the_registry_agree_on_the_ack_types():
    # ws_chat spells the pair itself (it does not import the radio package at
    # module load); a drift would route an ack nowhere.
    assert ws_chat._MEDIA_ACK_TYPES == control.MEDIA_ACK_TYPES


@pytest.mark.parametrize("frame_type", [
    "media_ended", "radio_skip_next", "radio_skip_prev", "radio_display_mode",
])
def test_the_existing_passthrough_frames_still_pass(frame_type):
    assert ws_chat._is_mid_turn_passthrough({"type": frame_type})


@pytest.mark.parametrize("frame", [
    {"type": "message", "text": "hi"},
    {"type": "media_stop"},          # tenant → phone only; never inbound
    {},
    ["radio_toggle"],
    "radio_toggle",
    None,
])
def test_everything_else_is_not_forwarded(frame):
    assert not ws_chat._is_mid_turn_passthrough(frame)


def test_the_stop_watcher_forwards_through_the_predicate():
    """`_wait_for_stop` is a closure inside the socket handler, so the wiring
    itself can only be read. The predicate and the dispatcher it feeds are
    executed below."""
    src = inspect.getsource(ws_chat.ws_chat)
    start = src.find("async def _wait_for_stop():")
    end = src.find("except WebSocketDisconnect:", start)
    assert 0 < start < end
    watcher = src[start:end]
    assert "_is_mid_turn_passthrough(m2)" in watcher
    # PIN MOVED (F27 residual): the forwarded ack now names the socket it came
    # on, so one socket answering twice counts once and a known answerer
    # stops holding the verdict (`radio.control.deliver_media_ack`).
    assert "_dispatch_radio_frame(user_id, m2, source=broadcast_queue)" in watcher
    assert "_note_socket_frame(broadcast_queue, m2)" in watcher
    assert "in _MID_TURN_PASSTHROUGH" not in watcher


# ── What the dispatcher does with them ───────────────────────────────────

@pytest.mark.asyncio
async def test_a_mid_turn_radio_off_turns_the_station_off(radio, wire):
    session = _enabled_session()
    radio._sessions[(USER, "app")] = session
    epoch_before = session.station_epoch

    await ws_chat._dispatch_radio_frame(
        USER, {"type": "radio_toggle", "enabled": False, "channel": "app"},
    )

    assert session.enabled is False
    assert session.station_epoch == epoch_before + 1
    assert ws_chat._radio_off_generations[(USER, "app")] == 1
    assert [f for f in wire if f.get("type") == "radio_state"][-1]["enabled"] is False


@pytest.mark.asyncio
async def test_a_mid_turn_radio_on_is_never_dispatched(radio, wire, monkeypatch):
    calls: list = []

    async def record(user_id, msg):
        calls.append(msg)

    monkeypatch.setattr(ws_chat, "_handle_radio_toggle", record)
    await ws_chat._dispatch_radio_frame(
        USER, {"type": "radio_toggle", "enabled": True, "channel": "app"},
    )
    assert calls == []


@pytest.mark.asyncio
async def test_a_mid_turn_stop_ack_confirms_the_pending_stop(radio, wire):
    radio._sessions[(USER, "app")] = _enabled_session()
    pending = asyncio.create_task(
        control.execute_media_control(USER, "stop", "app", ack_timeout_s=2.0),
    )
    for _ in range(200):
        if any(f.get("type") == "media_stop" for f in wire):
            break
        await asyncio.sleep(0.005)
    stop = next(f for f in wire if f.get("type") == "media_stop")

    await ws_chat._dispatch_radio_frame(USER, {
        "type": "media_stop_ack", "command_id": stop["command_id"],
        "stopped": True, "was_playing": True,
    })
    outcome = await asyncio.wait_for(pending, timeout=1.0)
    assert (outcome.ok, outcome.reason) == (True, "stopped")


@pytest.mark.asyncio
async def test_a_stray_ack_is_harmless(radio, wire):
    # No pending command: routed, matched to nothing, never raises.
    await ws_chat._dispatch_radio_frame(USER, {
        "type": "media_pause_ack", "command_id": "gone", "paused": True,
    })
    assert ws_chat._handle_media_ack(USER, {"type": "media_stop_ack"}) is False


# ── On the wire, through the real socket handler ─────────────────────────

def test_a_stop_ack_on_an_idle_socket_confirms_the_stop_and_is_not_an_error():
    """The whole round trip over an in-process ASGI socket: the tenant stop
    reaches the phone as `radio_state` OFF then `media_stop`, the phone's ack
    comes back on the same socket, the command settles `stopped`, and the
    socket does not answer the ack with "Unknown message type"."""
    import test_ws_chat_infra_fault_close as wsh

    async def go():
        mp = pytest.MonkeyPatch()
        sock = None
        try:
            import app.agent.radio as radio_pkg

            manager = RadioSessionManager()
            mp.setattr(radio_pkg, "get_radio_manager", lambda: manager)
            mp.setattr(control, "get_radio_manager", lambda: manager)
            mp.setattr(ws_chat, "_radio_toggle_locks", {})
            mp.setattr(ws_chat, "_radio_off_generations", {})
            mp.setattr(control, "_pending_acks", type(control._pending_acks)())
            ws_chat._active_turns.pop(wsh.USER_ID, None)

            sock = await wsh._authenticated_socket(mp)
            pending = asyncio.create_task(control.execute_media_control(
                wsh.USER_ID, "stop", "app", ack_timeout_s=3.0,
            ))

            seen: list[dict] = []
            while True:
                frame = await wsh._next_frame(sock)
                seen.append(frame)
                if frame.get("type") == "media_stop":
                    break
            types = [f.get("type") for f in seen]
            assert "radio_state" in types, seen
            assert types.index("radio_state") < types.index("media_stop")
            assert seen[types.index("radio_state")]["enabled"] is False
            stop = seen[-1]

            sock.send_text(json.dumps({
                "type": "media_stop_ack", "command_id": stop["command_id"],
                "stopped": True, "was_playing": True,
            }))
            outcome = await asyncio.wait_for(pending, timeout=2.0)
            assert (outcome.ok, outcome.reason) == (True, "stopped")

            # The ack was consumed, not answered as a protocol error.
            sock.send_text(json.dumps({"type": "ping"}))
            reply = await wsh._next_frame(sock)
            assert reply["type"] == "pong", reply
        finally:
            ws_chat._active_turns.pop(wsh.USER_ID, None)
            if sock is not None:
                await sock.dispose()
            mp.undo()

    asyncio.run(go())
