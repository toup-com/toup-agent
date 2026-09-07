"""The upstream dial, and the client that walked away during it.

Why this file exists
--------------------
`_hold_for_agent` races its readiness poll against `websocket.receive()`, so a
client that leaves mid-hold is noticed at once and nothing it sent is
forwarded. The dial that follows the hold did neither: it awaited
`websockets.connect()` with the client socket unattended, and `browser_to_agent`
then replayed `_pre_read` BEFORE its first `receive_text()` — i.e. before it
could learn the client was gone.

That matters because of what the two ends do next. build 109 arms a one-shot
`TURN_SILENT_GRACE_MS = 12_000` at `ws.send()`; the hold spends up to 9 s of it
and a failed first dial costs `asyncio.sleep(3)` before the retry
(ws_chat_proxy.py, the `for attempt in range(3)` loop). When the grace fires the
client closes the socket and re-sends the SAME `client_msg_id`. The agent takes
delivery of the replayed frame, claims that id in `ProcessedMessage` (deliberately
never released), runs the turn headless — and answers the re-send with a bare
`user_message_persisted{duplicate: true}`. build 109 has no dispatcher case for
that frame, and it stamps `_turnFrameSeen`, so the grace is disarmed; the 45 s
watchdog stays quiet because the pings are ponged. `sendChatMessage` never
resolves: the composer stays locked and the turn is lost until the app is
relaunched. A silent hang, on exactly the connection the hold exists to rescue.

The scenarios and the dedupe agent below are the audit's
(`$SP/audit-wschat/test_audit_wschat.py`, 2026-09-07), promoted into the suite
as falsifiers.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q tests/test_ws_proxy_dial_watch.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from typing import List, Optional

import pytest

import test_ws_proxy_warmup_harness as H
from app.api import ws_chat_proxy as proxy


class DedupAgent:
    """One tenant agent, many upstream sessions — the agent's real receipt
    behaviour (ws_chat.py:3321-3329, 3398-3406): a FIRST delivery of a
    `client_msg_id` is answered with status+done, a duplicate with the bare
    `user_message_persisted{duplicate:true}` ack and nothing else."""

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.seen: set = set()
        self.received: List[tuple] = []          # (virtual_ms, raw, session_no)
        self.sessions: List[H.FakeAgent] = []

    def _vnow(self) -> float:
        return (time.monotonic() - self.t0) / H.SCALE * 1000.0

    def connect(self) -> H.FakeAgent:
        s = H.FakeAgent()
        self.sessions.append(s)
        no = len(self.sessions)
        agent = self

        async def send(raw: str) -> None:
            agent.received.append((agent._vnow(), raw, no))
            cid = json.loads(raw).get("client_msg_id")
            if cid in agent.seen:
                await s.inbox.put(json.dumps({
                    "type": "user_message_persisted", "client_msg_id": cid,
                    "session_id": None, "duplicate": True,
                }))
                return
            agent.seen.add(cid)
            await s.inbox.put(json.dumps({"type": "status", "stage": "received"}))
            await s.inbox.put(json.dumps({"type": "done", "message_id": "m1"}))

        s.send = send  # type: ignore[assignment]
        return s


def _rig(mp, *, ready_at, dial_ms, later_dial_ms=None, pong_every_ms=25_000):
    """The shipped rig, plus a dial of controllable length and a dedupe agent."""
    rig = H.build_rig(mp, ready_at)
    agent = DedupAgent(rig.t0)
    dials = {"n": 0}

    async def fake_connect(url, **kwargs):
        dials["n"] += 1
        d = dial_ms if (dials["n"] == 1 or later_dial_ms is None) else later_dial_ms
        await asyncio.sleep(H.real_s(d))
        s = agent.connect()
        if pong_every_ms:
            # api.ts:3195-3203 — the client pings every 25 s and is ponged, so
            # `_lastFrameAt` stays fresh and the 45 s watchdog never fires.
            async def pongs():
                try:
                    while True:
                        await asyncio.sleep(H.real_s(pong_every_ms))
                        await s.inbox.put(json.dumps({"type": "pong"}))
                except asyncio.CancelledError:
                    pass
            s._pong_task = asyncio.create_task(pongs())  # type: ignore[attr-defined]
        return s

    mp.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=fake_connect))
    return rig, agent


def _stop_pongs(agent: DedupAgent) -> None:
    for s in agent.sessions:
        t = getattr(s, "_pong_task", None)
        if t:
            t.cancel()


class _CloseClock:
    """Virtual time of every client-side close, so a test can prove a forward
    happened AFTER the client left rather than merely near it."""

    def __init__(self, t0) -> None:
        self.t0 = t0
        self.closes: List[float] = []

    def wrap(self, mp) -> None:
        orig = H.AsgiWebSocket.client_close
        clock = self

        def client_close(self_ws, code: int = 1000):
            clock.closes.append((time.monotonic() - clock.t0) / H.SCALE * 1000.0)
            return orig(self_ws, code)

        mp.setattr(H.AsgiWebSocket, "client_close", client_close)


# ══════════════════════════════════════════════════════════════════════
# 1. The falsifier
# ══════════════════════════════════════════════════════════════════════

def test_client_leaving_during_the_dial_forwards_nothing():
    """FALSIFIER. Readiness at poll 8 of 9 and a 4.5 s dial: 12.5 s against the
    client's 12 s grace.

    Before the fix the agent received the frame TWICE — once from the
    `_pre_read` replay onto a socket whose client had closed 0.7 s earlier, once
    from the re-send — and the second delivery was answered only with the
    duplicate ack, so the client hung indefinitely.

    After it: the dial notices the disconnect, forwards nothing, and the
    client's own re-send is a FIRST delivery that completes the turn.
    """
    async def go(mp):
        rig, agent = _rig(mp, ready_at=7_500, dial_ms=4_500, later_dial_ms=500)
        H.configure(mp, hold_virtual_ms=9_000, poll_virtual_ms=1_000,
                    retry_virtual_ms=None, adopt=False)
        clock = _CloseClock(rig.t0)
        clock.wrap(mp)
        try:
            res = await asyncio.wait_for(
                H.Build109Client(rig.app).send("first message"),
                timeout=H.real_s(150_000),
            )
            outcome, attempts = res.outcome, res.attempts
        except asyncio.TimeoutError:
            outcome, attempts = "HUNG>150s", None
        _stop_pongs(agent)

        print("\ndeliveries:", [(round(t), no) for t, _, no in agent.received],
              "closes:", [round(c) for c in clock.closes], "->", outcome)
        assert outcome == "delivered", (outcome, agent.received)
        assert len(agent.received) == 1, agent.received
        # ...and the single delivery is the RE-SEND: it lands after the strike
        # that closed the first socket, on the client's second attempt. The
        # abandoned dial forwarded nothing — it never even completed, so no
        # upstream session was left open behind it.
        assert attempts == 2, attempts
        assert clock.closes and agent.received[0][0] > clock.closes[0], (
            clock.closes, agent.received)
        assert len(agent.sessions) == 1, len(agent.sessions)
        assert json.loads(agent.received[0][1])["text"] == "first message"
    H.run(H._run_with_mp(go))


def test_nothing_is_written_to_a_socket_the_client_already_closed():
    """The dial's own failure paths must not answer a client that has left.

    Same disconnect, but the dial then FAILS: the 4502/4504 frame + close would
    be written into a dead socket, which is an ASGI error on the way out and a
    lie in the log either way.
    """
    async def go(mp):
        rig, agent = _rig(mp, ready_at=7_500, dial_ms=4_500)

        async def never_connects(url, **kwargs):
            await asyncio.sleep(H.real_s(4_500))
            raise OSError("connection refused")

        mp.setitem(sys.modules, "websockets",
                   types.SimpleNamespace(connect=never_connects))
        H.configure(mp, hold_virtual_ms=9_000, poll_virtual_ms=1_000,
                    retry_virtual_ms=None, adopt=False)

        ws = H.AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text(json.dumps({"type": "message", "text": "x",
                                 "client_msg_id": "c1"}))
        await asyncio.sleep(H.real_s(10_000))     # into the dial
        ws.client_close()
        await asyncio.wait_for(asyncio.shield(ws.task), timeout=H.real_s(60_000))
        leftover = []
        while not ws._from_app.empty():
            leftover.append(ws._from_app.get_nowait())
        assert leftover == [], f"wrote into a disconnected socket: {leftover}"
    H.run(H._run_with_mp(go))


# ══════════════════════════════════════════════════════════════════════
# 2. The other half: a client that STAYS must lose nothing
# ══════════════════════════════════════════════════════════════════════

def test_frames_sent_during_the_dial_are_forwarded_once_and_in_order():
    """Watching the socket means READING it, so the frames the watch consumes
    have to reach the agent — in order, exactly once. Two frames land while the
    dial is in flight; both must arrive, and the relay's own receive loop must
    still pick up a third sent after it opens."""
    async def go(mp):
        rig, agent = _rig(mp, ready_at=0, dial_ms=4_000, pong_every_ms=None)
        H.configure(mp, hold_virtual_ms=9_000, poll_virtual_ms=1_000,
                    retry_virtual_ms=None, adopt=False)
        ws = H.AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text(json.dumps({"type": "message", "text": "one",
                                 "client_msg_id": "c1"}))
        await asyncio.sleep(H.real_s(1_000))
        ws.send_text(json.dumps({"type": "message", "text": "two",
                                 "client_msg_id": "c2"}))
        await asyncio.sleep(H.real_s(6_000))      # dial done, relaying
        ws.send_text(json.dumps({"type": "message", "text": "three",
                                 "client_msg_id": "c3"}))
        await asyncio.sleep(H.real_s(2_000))
        texts = [json.loads(raw)["text"] for _, raw, _ in agent.received]
        assert texts == ["one", "two", "three"], texts
        await ws.dispose()
    H.run(H._run_with_mp(go))


def test_warm_agent_still_delivers_in_one_socket():
    """The path 71 of 74 tenants take: agent up, dial instant. One delivery,
    one socket, no hold — the watch must be invisible here."""
    async def go(mp):
        rig, agent = _rig(mp, ready_at=0, dial_ms=0, pong_every_ms=None)
        H.configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        res = await H.Build109Client(rig.app).send("hello")
        assert res.outcome == "delivered", res
        assert res.sockets == 1, res.sockets
        assert len(agent.received) == 1, agent.received
        assert json.loads(agent.received[0][1])["text"] == "hello"
    H.run(H._run_with_mp(go))


def test_a_slow_dial_the_client_waits_out_still_delivers_once():
    """The dial is watched, not shortened. A 6 s dial inside the grace must
    still relay and deliver on the FIRST socket."""
    async def go(mp):
        rig, agent = _rig(mp, ready_at=0, dial_ms=6_000, pong_every_ms=None)
        H.configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None, adopt=False)
        res = await asyncio.wait_for(H.Build109Client(rig.app).send("slow"),
                                     timeout=H.real_s(60_000))
        assert res.outcome == "delivered", res
        assert res.sockets == 1, res.sockets
        assert len(agent.received) == 1, agent.received
        assert agent.received[0][2] == 1, agent.received
    H.run(H._run_with_mp(go))


def test_dial_retry_backoff_is_watched_too():
    """A failed first dial costs `asyncio.sleep(3)` before the retry — three
    quarters of the budget the hold leaves, and the longest single unattended
    await in the handler. A client leaving inside THAT sleep must be noticed
    for the same reason."""
    async def go(mp):
        rig, agent = _rig(mp, ready_at=0, dial_ms=0)
        calls = {"n": 0}

        async def flaky(url, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("refused")          # -> asyncio.sleep(3)
            return agent.connect()

        mp.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=flaky))
        H.configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None, adopt=False)
        ws = H.AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text(json.dumps({"type": "message", "text": "x",
                                 "client_msg_id": "c1"}))
        await asyncio.sleep(0.05)                 # inside the real 3 s backoff
        ws.client_close()
        await asyncio.wait_for(asyncio.shield(ws.task), timeout=10.0)
        assert agent.received == [], agent.received
        assert calls["n"] == 1, "the retry ran after the client had left"
    H.run(H._run_with_mp(go))


# ── kill switch ────────────────────────────────────────────────────────

class _FakeClientWS:
    """The client side of the proxy as `_dial_agent` sees it."""
    def __init__(self, messages):
        self._messages = list(messages); self.receives = 0
    async def receive(self):
        self.receives += 1
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(3600)        # nothing more: stay pending


class _FakeAgentSock:
    def __init__(self): self.closed = False
    async def close(self): self.closed = True


def _fake_websockets(sock):
    async def connect(*a, **k):
        await asyncio.sleep(0.01)
        return sock
    return types.SimpleNamespace(connect=connect)


def test_the_dial_watch_has_a_kill_switch(monkeypatch):
    """AGENT_WS_PROXY_DIAL_WATCH=false must restore the pre-#720 dial: the
    client's socket is not read during the connect, so a disconnect queued
    there is NOT noticed and the relay opens as it always did. With the switch
    on, the same queued disconnect ends the dial with nothing forwarded."""
    async def go():
        gone_msg = {"type": "websocket.disconnect"}
        # OFF: receive() is never awaited, the agent socket is handed back.
        monkeypatch.setattr(proxy.settings, "agent_ws_proxy_dial_watch", False, raising=False)
        ws = _FakeClientWS([gone_msg]); sock = _FakeAgentSock(); pre = []
        agent_ws, failure, client_gone = await proxy._dial_agent(
            ws, _fake_websockets(sock), "ws://x", "k", pre)
        assert agent_ws is sock and failure is None and client_gone is False
        assert ws.receives == 0 and sock.closed is False
        # ON: the disconnect is harvested, the socket is closed, nothing forwarded.
        monkeypatch.setattr(proxy.settings, "agent_ws_proxy_dial_watch", True, raising=False)
        ws = _FakeClientWS([gone_msg]); sock = _FakeAgentSock(); pre = []
        agent_ws, failure, client_gone = await proxy._dial_agent(
            ws, _fake_websockets(sock), "ws://x", "k", pre)
        assert agent_ws is None and client_gone is True and pre == []
        # The disconnect was harvested BEFORE the connect completed, so the
        # abandoned dial was cancelled while pending and there was never a
        # socket to close. (A connect that lands anyway is closed by
        # `_abandon`'s done-callback — covered by the harness tests above.)
        assert ws.receives >= 1 and sock.closed is False
    asyncio.run(go())
