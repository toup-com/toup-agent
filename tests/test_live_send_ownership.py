"""The Live phone socket has one writer across provider and tool tasks."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocket, WebSocketState

from app.api import ws_realtime as rt
from app.services import live_voice_protocol as live


def _session(ws):
    session = live._LiveSession.__new__(live._LiveSession)
    session.ws = ws
    session.client_alive = True
    session.closed = asyncio.Event()
    session._client_send_lock = asyncio.Lock()
    session.provider_session_id = "provider-test"
    session.db_session_id = "session-test"
    session.user_id = "user-test"
    return session


@pytest.mark.asyncio
async def test_tool_and_lifecycle_frames_share_one_connected_asgi_writer():
    active = 0
    peak = 0
    sent = []
    entered = asyncio.Event()

    async def receive():
        return {"type": "websocket.connect"}

    async def asgi_send(message):
        nonlocal active, peak
        if message["type"] != "websocket.send":
            return
        active += 1
        peak = max(peak, active)
        try:
            if active > 1:
                raise AssertionError("ASGI send overlapped")
            entered.set()
            await asyncio.sleep(0.03)
            sent.append(message["text"])
        finally:
            active -= 1

    ws = WebSocket({"type": "websocket"}, receive, asgi_send)
    await ws.accept()
    assert ws.application_state == ws.client_state == WebSocketState.CONNECTED
    session = _session(ws)
    relay = rt._InnerToolRelay(
        ws, "live-delegation:test", frame_sender=session.send,
    )
    lifecycle = asyncio.create_task(session.send({"type": "delegation", "phase": "started"}))
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    tool = asyncio.create_task(relay._send({"type": "tool_call.started", "name": "web_search"}))
    assert await asyncio.gather(lifecycle, tool) == [True, True]
    assert peak == 1 and len(sent) == 2
    assert session.client_alive and not session.closed.is_set()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_poison_socket_writer():
    entered = asyncio.Event()
    release = asyncio.Event()
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def asgi_send(message):
        if message["type"] != "websocket.send":
            return
        if not entered.is_set():
            entered.set()
            await release.wait()
        sent.append(message["text"])

    ws = WebSocket({"type": "websocket"}, receive, asgi_send)
    await ws.accept()
    session = _session(ws)
    first = asyncio.create_task(session.send({"type": "first"}))
    await asyncio.wait_for(entered.wait(), timeout=0.5)
    cancelled = asyncio.create_task(session.send({"type": "cancelled"}))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    assert await first
    assert await session.send({"type": "third"})
    assert len(sent) == 2 and '"cancelled"' not in "".join(sent)
    assert session.client_alive and not session.closed.is_set()


@pytest.mark.asyncio
async def test_send_failure_logs_location_without_exception_text(caplog):
    async def receive():
        return {"type": "websocket.connect"}

    async def asgi_send(message):
        if message["type"] == "websocket.send":
            raise AssertionError("PRIVATE WORDS MUST NOT ENTER LOGS")

    ws = WebSocket({"type": "websocket"}, receive, asgi_send)
    await ws.accept()
    session = _session(ws)
    assert await session.send({"type": "delegation"}) is False
    assert ws.application_state == ws.client_state == WebSocketState.CONNECTED
    assert session.closed.is_set()
    assert "origin=" in caplog.text
    assert "PRIVATE WORDS" not in caplog.text


@pytest.mark.asyncio
async def test_recovered_answer_speaks_after_output_goes_quiet(monkeypatch):
    async def vps(_user_id):
        return "https://tenant.test", "key"

    claims = []

    async def api(_url, _key, method, path, **kwargs):
        assert method == "POST" and path.endswith("/claim-speech")
        claims.append(kwargs["json_body"])
        return {"claimed": True}

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", api)
    session = _session(SimpleNamespace())
    session.utterances = SimpleNamespace(open=None)
    session.adapter = live.LiveClientAdapter("provider-test")
    session.adapter._output_id = "old-output"
    session.adapter._last_delta_monotonic = time.monotonic() - 10.0
    spoken = []

    async def speak(task_id, answer, **kwargs):
        spoken.append((task_id, answer, kwargs))

    session.speak = speak
    await asyncio.wait_for(
        session.speak_recovered_result("task-test", {"turn_id": "turn-test"}, "The answer."),
        timeout=0.5,
    )
    assert claims == [{"session_id": "session-test", "task_id": "task-test"}]
    assert spoken == [("task-test", "The answer.", {"parent_user_turn_id": "turn-test"})]


@pytest.mark.asyncio
async def test_recovered_answer_waits_while_provider_output_is_recent(monkeypatch):
    calls = []

    async def vps(_user_id):
        calls.append("claim")
        return "https://tenant.test", "key"

    async def api(*_args, **_kwargs):
        return {"claimed": True}

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", api)
    session = _session(SimpleNamespace())
    session.utterances = SimpleNamespace(open=None)
    session.adapter = live.LiveClientAdapter("provider-test")
    session.adapter._output_id = "current-output"
    session.adapter._last_delta_monotonic = time.monotonic()

    async def speak(*_args, **_kwargs):
        calls.append("speak")

    session.speak = speak
    task = asyncio.create_task(session.speak_recovered_result(
        "task-test", {"turn_id": "turn-test"}, "The answer.",
    ))
    try:
        await asyncio.sleep(0.05)
        assert not calls
        session.adapter._last_delta_monotonic = time.monotonic() - 10.0
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert calls == ["claim", "speak"]


@pytest.mark.asyncio
async def test_recovered_answer_does_not_interrupt_open_user_turn(monkeypatch):
    claims = []

    async def vps(_user_id):
        claims.append(True)
        return "https://tenant.test", "key"

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    session = _session(SimpleNamespace())
    session.utterances = SimpleNamespace(open=object())
    session.adapter = None
    task = asyncio.create_task(session.speak_recovered_result(
        "task-test", {"turn_id": "turn-test"}, "The answer.",
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not claims
