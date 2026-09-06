"""The build-109 warm-up client, ported and driven against the REAL proxy.

Why this file exists
--------------------
The 2026-09-06 incident was not a server bug in isolation and not a client bug
in isolation: it was a server constant (`retry_after_ms`) that the client uses
as an OVERRIDE of its own back-off ladder, changed on the server without anyone
re-deriving the client's budget. `tests/test_ws_proxy_fast_fail.py` — the only
coverage this path had — asserts SOURCE STRINGS. It was green through the whole
incident and would stay green at any value of the integer that caused it. A
source-string test cannot notice that 15 attempts times 2 s misses a p90.

So this harness ports build-109's `sendChatMessage` warm-up state machine from
`src/shared/api.ts` (the exact file every one of the 42 field devices runs) into
Python — same constants, same override rule, same clamp, same give-up rule, same
silent-turn grace — and drives it over a real in-process ASGI WebSocket against
the real `ws_chat_proxy` handler, with a fake agent that becomes ready N seconds
after the first attempt. What is asserted is the thing the user experiences:
did the message reach the agent, after how many attempts, and after how long.

The three build-109 constants that decide everything (api.ts:2300-2309, 3465):

    MAX_WARMUP_ATTEMPTS = 15
    delay = max(500, min(_agentStartingRetryMs ?? ladder, 10000)) + jitter(0..400)
    TURN_SILENT_GRACE_MS = 12_000   # one-shot, armed just before ws.send()

The last one is the ceiling on any server-side hold, and it is not obvious:
build 109 tears the attempt down if no turn-bearing frame arrives within 12 s,
and the SECOND such strike in one send raises a terminal error (`silentStrikes`
is not reset between attempts). A 12 s hold would therefore end a send after
~25 s — worse than the 34 s that produced the incident. `test_hold_above_client_
silent_grace_is_a_regression` pins that, so nobody raises the hold "a bit more".

Time is compressed by `SCALE` (20x): every duration below is written in VIRTUAL
milliseconds — the units the wire and both clients actually use — and converted
at the one boundary where something really sleeps. The server's own hold/poll
settings are set in real ms; `retry_after_ms` stays virtual, which is sound here
because the derivation saturates at the client's 10 000 ms clamp either way
(`test_retry_after_ms_is_derived_and_saturates_at_the_client_clamp` checks the
production settings directly, unscaled).

Run:
    cd backend && PYTHONPATH=. python -m pytest -q tests/test_ws_proxy_warmup_harness.py
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI

from app.api import ws_chat_proxy as proxy
from app.config import settings

# ── Time compression ──────────────────────────────────────────────────
# 1 virtual second = SCALE real seconds. Everything else in this file is
# written in virtual milliseconds.
SCALE = 0.05


def real_s(virtual_ms: float) -> float:
    return (virtual_ms / 1000.0) * SCALE


def real_ms(virtual_ms: float) -> int:
    return max(1, int(round(virtual_ms * SCALE)))


# ── build-109 client constants (src/shared/api.ts, build 2026-08-29) ───
MAX_WARMUP_ATTEMPTS = 15                      # api.ts:2300
WARMUP_BASE_MS = 1000                         # api.ts:2308
WARMUP_MAX_MS = 8000                          # api.ts:2309
MAX_ABNORMAL_PRE_REPLY_RETRIES = 2            # api.ts:2307
TURN_SILENT_GRACE_MS = 12_000                 # api.ts:2337
TURN_SILENCE_TIMEOUT_MS = 45_000              # api.ts:2322
CONNECT_TIMEOUT_MS = 10_000                   # api.ts:3178
CONNECT_RETRY_DELAY_MS = 1_000                # api.ts:3225
AGENT_WARMING_CLOSE_CODES = {4404, 4502, 4503, 4504}   # api.ts:2272
# api.ts:3465-3466 — the delay clamp. `retry_after_ms` is read here and
# ONLY here, and it replaces the ladder rather than adjusting it.
DELAY_FLOOR_MS = 500
DELAY_CEIL_MS = 10_000
JITTER_MS = 400
# Frame types that do NOT stamp `_turnFrameSeen` (api.ts:2971).
NO_STAMP_TYPES = {"pong", "ping", "message", "admin_notice", "admin_notice_retract"}
# ...and the additional ones that do not stamp `_turnReceivedContent`.
NO_CONTENT_TYPES = NO_STAMP_TYPES | {"agent_starting", "status"}


# ── In-process ASGI WebSocket driver ──────────────────────────────────
class AsgiWebSocket:
    """One client-side socket, spoken straight into the ASGI app.

    Deliberately not `TestClient`: the client model needs precise, awaitable
    receive timeouts (it is modelling two timers), and TestClient's session is
    a blocking queue driven from a portal thread.
    """

    def __init__(self, app: Any, path: str = "/api/ws/chat", query: str = "token=t"):
        self._app = app
        self._path = path
        self._query = query
        self._to_app: asyncio.Queue = asyncio.Queue()
        self._from_app: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self.closed = False
        self.close_code: Optional[int] = None

    async def open(self, timeout_ms: float = CONNECT_TIMEOUT_MS) -> None:
        scope = {
            "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "scheme": "ws", "path": self._path,
            "raw_path": self._path.encode(), "query_string": self._query.encode(),
            "root_path": "", "headers": [(b"host", b"testserver")],
            "client": ("127.0.0.1", 51234), "server": ("testserver", 80),
            "subprotocols": [], "state": {},
        }
        self._to_app.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.create_task(
            self._app(scope, self._to_app.get, self._from_app.put)
        )
        msg = await asyncio.wait_for(self._from_app.get(), timeout=real_s(timeout_ms))
        if msg["type"] != "websocket.accept":
            raise ConnectionError(f"handshake refused: {msg}")

    def send_text(self, text: str) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": text})

    async def recv(self, timeout_ms: float) -> Dict[str, Any]:
        """Next server→client ASGI message, or asyncio.TimeoutError.

        A handler that RAISED would otherwise show up as a mysterious timeout
        in every assertion in this file; surface it instead.
        """
        try:
            return await asyncio.wait_for(
                self._from_app.get(), timeout=real_s(timeout_ms),
            )
        except asyncio.TimeoutError:
            if self.task is not None and self.task.done():
                exc = self.task.exception()
                if exc is not None:
                    raise exc
            raise

    def client_close(self, code: int = 1000) -> None:
        """The client walks away (backgrounded app, killed process, dead link)."""
        self.closed = True
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": code})

    async def dispose(self) -> None:
        if not self.closed:
            self.client_close()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=2.0)
            except (asyncio.TimeoutError, Exception):  # noqa: B014
                self.task.cancel()


# ── The build-109 warm-up state machine ───────────────────────────────
@dataclass
class SendResult:
    outcome: str                    # 'delivered' | 'terminal_error' | 'error'
    attempts: int = 0               # warm-up loop iterations entered
    sockets: int = 0                # WebSocket opens (the unit the server logs)
    elapsed_ms: float = 0.0         # virtual ms from the first attempt
    error: Optional[str] = None
    waking_calls: int = 0
    frames: List[dict] = field(default_factory=list)


class Build109Client:
    """`sendChatMessage`'s bounded warm-up retry loop, faithfully.

    Ported from api.ts:3311-3479. Everything that decides an outcome is here:
    the per-attempt settle race (frame dispatcher vs onclose vs silent grace vs
    watchdog), the `_agentStartingRetryMs` override, the clamp, the give-up.
    """

    def __init__(self, app: Any, rng: Optional[random.Random] = None):
        self.app = app
        self.rng = rng or random.Random(1109)
        self.t0 = 0.0

    def _vnow(self) -> float:
        return (time.monotonic() - self.t0) / SCALE * 1000.0

    async def _connect(self, res: SendResult) -> AsgiWebSocket:
        """api.ts connectChat/_connectChatInner — one retry after 1 s."""
        ws = AsgiWebSocket(self.app)
        try:
            await ws.open()
        except Exception:
            res.sockets += 1
            await ws.dispose()
            await asyncio.sleep(real_s(CONNECT_RETRY_DELAY_MS))
            ws = AsgiWebSocket(self.app)
            await ws.open()
        res.sockets += 1
        return ws

    async def send(self, text: str = "hello") -> SendResult:
        res = SendResult(outcome="error")
        self.t0 = time.monotonic()
        frame = json.dumps({"type": "message", "text": text,
                            "client_msg_id": "fixed-id", "channel": "mobile"})
        agent_starting_retry_ms: Optional[int] = None
        abnormal_retries = 0
        silent_strikes = 0

        for attempt in range(MAX_WARMUP_ATTEMPTS):
            res.attempts = attempt + 1
            agent_starting_retry_ms = None
            ws = await self._connect(res)
            outcome, agent_starting_retry_ms, abnormal_retries, silent_strikes = (
                await self._run_attempt(ws, frame, res, abnormal_retries, silent_strikes)
            )
            await ws.dispose()

            if outcome == "done":
                res.outcome = "delivered"
                res.elapsed_ms = self._vnow()
                return res
            if outcome == "error":
                res.elapsed_ms = self._vnow()
                return res

            # outcome == 'warming'
            if attempt + 1 >= MAX_WARMUP_ATTEMPTS:
                break
            res.waking_calls += 1
            ladder = min(WARMUP_BASE_MS * (2 ** attempt), WARMUP_MAX_MS)
            base = agent_starting_retry_ms if agent_starting_retry_ms is not None else ladder
            delay = max(DELAY_FLOOR_MS, min(base, DELAY_CEIL_MS)) + self.rng.randrange(JITTER_MS)
            await asyncio.sleep(real_s(delay))

        # api.ts:3483 — budget exhausted.
        res.outcome = "terminal_error"
        res.error = ("Your agent is taking longer than usual to start. "
                     "Please try again in a moment.")
        res.elapsed_ms = self._vnow()
        return res

    async def _run_attempt(self, ws, frame, res, abnormal_retries, silent_strikes):
        """One attempt's settle race. Returns
        (outcome, agent_starting_retry_ms, abnormal_retries, silent_strikes)."""
        turn_frame_seen = False
        turn_received_content = False
        agent_starting_retry_ms: Optional[int] = None
        sent_at = self._vnow()
        last_frame_at = sent_at
        ws.send_text(frame)

        while True:
            now = self._vnow()
            grace_at = math.inf if turn_frame_seen else sent_at + TURN_SILENT_GRACE_MS
            watchdog_at = max(sent_at, last_frame_at) + TURN_SILENCE_TIMEOUT_MS
            deadline = min(grace_at, watchdog_at)
            try:
                msg = await ws.recv(max(1.0, deadline - now))
            except asyncio.TimeoutError:
                if grace_at <= watchdog_at:
                    # api.ts:3388 silent-grace. Strike one re-sends; strike two
                    # is terminal, and strikes are NOT reset between attempts.
                    silent_strikes += 1
                    if silent_strikes < 2:
                        return "warming", None, abnormal_retries, silent_strikes
                    res.error = "Connection lost — no reply arrived. Try sending again."
                    return "error", None, abnormal_retries, silent_strikes
                # api.ts:3358 turn watchdog.
                if turn_received_content:
                    res.error = "dropped"
                    return "error", None, abnormal_retries, silent_strikes
                return "warming", None, abnormal_retries, silent_strikes

            if msg["type"] == "websocket.close":
                code = msg.get("code", 1000)
                if code in AGENT_WARMING_CLOSE_CODES:
                    return "warming", agent_starting_retry_ms, abnormal_retries, silent_strikes
                if not turn_received_content and abnormal_retries < MAX_ABNORMAL_PRE_REPLY_RETRIES:
                    abnormal_retries += 1
                    return "warming", agent_starting_retry_ms, abnormal_retries, silent_strikes
                res.error = "Connection lost — try sending again"
                return "error", agent_starting_retry_ms, abnormal_retries, silent_strikes

            if msg["type"] != "websocket.send":
                continue

            last_frame_at = self._vnow()
            try:
                data = json.loads(msg.get("text") or "{}")
            except (ValueError, TypeError):
                continue
            res.frames.append(data)
            mtype = data.get("type")
            if mtype not in NO_STAMP_TYPES:
                turn_frame_seen = True
            if mtype not in NO_CONTENT_TYPES:
                turn_received_content = True

            if mtype == "agent_starting":
                # api.ts:3019-3023 — capture the cadence, settle 'warming'.
                rv = data.get("retry_after_ms")
                agent_starting_retry_ms = rv if isinstance(rv, (int, float)) else None
                return "warming", agent_starting_retry_ms, abnormal_retries, silent_strikes
            if mtype == "done":
                return "done", agent_starting_retry_ms, abnormal_retries, silent_strikes
            if mtype == "error":
                message = str(data.get("message") or "")
                warming = data.get("code") == "agent_starting" or any(
                    s in message.lower() for s in (
                        "cannot reach agent", "agent connection timed out",
                        "failed to connect to agent", "agent is waking up",
                        "agent not available", "no active agent", "502", "503", "504",
                    )
                )
                if warming:
                    return "warming", agent_starting_retry_ms, abnormal_retries, silent_strikes
                res.error = message
                return "error", agent_starting_retry_ms, abnormal_retries, silent_strikes


# ── Server-side test doubles ──────────────────────────────────────────
class FakeAgent:
    """The tenant agent. Answers any relayed frame with a `done`."""

    def __init__(self) -> None:
        self.received: List[str] = []
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def send(self, raw: str) -> None:
        self.received.append(raw)
        await self.inbox.put(json.dumps({"type": "done", "message_id": "m1"}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        await self.inbox.put(None)


@dataclass
class Rig:
    app: FastAPI
    agent: FakeAgent
    lookups: List[float]                    # virtual ms of each readiness check
    t0: float


def build_rig(monkeypatch, ready_at_virtual_ms: Optional[float]) -> Rig:
    """Wire the real router with a readiness clock and a fake agent.

    `ready_at_virtual_ms=None` means never ready; `0` means ready before the
    first connect (the warm-user case).
    """
    app = FastAPI()
    app.include_router(proxy.router, prefix="/api")
    agent = FakeAgent()
    t0 = time.monotonic()
    lookups: List[float] = []

    async def fake_auth(token: str) -> str:
        return "aec1977b-1fe0-4565-956a-ae960d06719c"

    async def fake_lookup(user_id: str, quiet: bool = False):
        elapsed_v = (time.monotonic() - t0) / SCALE * 1000.0
        lookups.append(elapsed_v)
        if ready_at_virtual_ms is None or elapsed_v < ready_at_virtual_ms:
            return None
        return ("ws://agent.invalid/api/ws/chat", "k")

    async def fake_connect(url, **kwargs):
        return agent

    # _diag POSTs to the production bridge. Never in a test.
    monkeypatch.setattr(proxy, "_diag", lambda *a, **k: None)
    monkeypatch.setattr(proxy, "_authenticate_ws", fake_auth)
    monkeypatch.setattr(proxy, "_get_agent_ws_info", fake_lookup)
    fake_ws_mod = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "websockets", fake_ws_mod)
    # A fresh semaphore per rig.
    monkeypatch.setattr(proxy, "_hold_sem", None, raising=False)
    monkeypatch.setattr(proxy, "_hold_sem_size", -1, raising=False)
    return Rig(app=app, agent=agent, lookups=lookups, t0=t0)


def configure(monkeypatch, *, fast_fail=True, hold_virtual_ms=0, poll_virtual_ms=1000,
              retry_virtual_ms=None, cap=200, adopt=True):
    """Set the proxy's settings for one scenario.

    `retry_virtual_ms` is expressed as the number the client should receive; it
    is produced through the REAL derivation (`_compute_retry_after_ms`) by
    setting the coverage target, so main's 2000 and PR #710's 6000 are
    reproduced by the shipped code path rather than by a stub.
    """
    monkeypatch.setattr(settings, "agent_ws_proxy_fast_fail", fast_fail)
    monkeypatch.setattr(settings, "agent_ws_proxy_hold_ms", real_ms(hold_virtual_ms)
                        if hold_virtual_ms else 0)
    monkeypatch.setattr(settings, "agent_ws_proxy_hold_poll_ms", real_ms(poll_virtual_ms))
    monkeypatch.setattr(settings, "agent_ws_proxy_hold_max_concurrent", cap)
    monkeypatch.setattr(settings, "agent_ws_proxy_hold_adopts_stranded", adopt)
    monkeypatch.setattr(settings, "agent_ws_proxy_client_attempts", MAX_WARMUP_ATTEMPTS)
    monkeypatch.setattr(settings, "agent_ws_proxy_retry_after_max_ms", DELAY_CEIL_MS)
    if retry_virtual_ms is None:
        monkeypatch.setattr(settings, "agent_ws_proxy_target_coverage_ms", 300_000)
    else:
        # target = attempts*hold + (attempts-1)*retry, with the hold the
        # DERIVATION uses being the real (scaled) one it will actually perform.
        held = int(settings.agent_ws_proxy_hold_ms)
        monkeypatch.setattr(
            settings, "agent_ws_proxy_target_coverage_ms",
            MAX_WARMUP_ATTEMPTS * held + (MAX_WARMUP_ATTEMPTS - 1) * retry_virtual_ms,
        )


def run(coro):
    return asyncio.run(coro)


# ── Scenario driver ───────────────────────────────────────────────────
async def _scenario(monkeypatch, *, ready_at, hold=0, retry=None, cap=200) -> SendResult:
    rig = build_rig(monkeypatch, ready_at)
    configure(monkeypatch, hold_virtual_ms=hold, retry_virtual_ms=retry, cap=cap)
    client = Build109Client(rig.app)
    res = await client.send()
    res.frames = res.frames  # keep
    if res.outcome == "delivered":
        assert rig.agent.received, "delivered but the agent got no frame"
    return res


# ══════════════════════════════════════════════════════════════════════
# 1. The falsifiers: what main and PR #710 do to the incident
# ══════════════════════════════════════════════════════════════════════

def test_main_behaviour_fails_the_incident_at_T74():
    """FALSIFIER. main today: no hold, retry_after_ms=2000.

    This is the incident. The client burns all 15 attempts in ~34 s and shows
    the terminal error 40 s before the agent exists. Measured on the wire for
    the incident user: 15 sockets, 34.66 s (18:18:03.007 → 18:18:37.669).
    """
    res = run(_run_with_mp(_scenario, ready_at=74_000, hold=0, retry=2000))
    assert res.outcome == "terminal_error"
    assert res.attempts == MAX_WARMUP_ATTEMPTS
    assert res.sockets == MAX_WARMUP_ATTEMPTS
    # ~34 s of budget: 15 x (connect) + 14 x (2000..2400)
    assert 28_000 <= res.elapsed_ms <= 42_000, res.elapsed_ms
    assert "taking longer than usual" in (res.error or "")


def test_pr710_6000_rescues_T74_but_still_fails_T150():
    """PR #710 (2000 → 6000) is a real improvement and an incomplete one.

    ~90 s of budget clears the incident (T+74) and the measured p95 of 92.3 s
    by 2 s, but not the p99 tail: the same claim path has produced 367 s, and
    the client's own 10 000 ms clamp caps this lever at ~146 s no matter what
    integer is sent.
    """
    ok = run(_run_with_mp(_scenario, ready_at=74_000, hold=0, retry=6000))
    assert ok.outcome == "delivered", ok
    assert ok.elapsed_ms <= 95_000

    late = run(_run_with_mp(_scenario, ready_at=150_000, hold=0, retry=6000))
    assert late.outcome == "terminal_error", late
    assert late.attempts == MAX_WARMUP_ATTEMPTS


def test_retry_alone_at_the_client_clamp_still_fails_T150():
    """Even 10 000 ms — the largest number build 109 will honour — is not
    enough on its own. This is the argument for holding the socket."""
    res = run(_run_with_mp(_scenario, ready_at=150_000, hold=0, retry=10_000))
    assert res.outcome == "terminal_error", res
    assert 130_000 <= res.elapsed_ms <= 160_000, res.elapsed_ms


# ══════════════════════════════════════════════════════════════════════
# 2. The change: hold + derived retry
# ══════════════════════════════════════════════════════════════════════

def test_hold_delivers_the_incident_T74():
    """PROOF. Same client, same scenario, hold 9 s + derived retry."""
    res = run(_run_with_mp(_scenario, ready_at=74_000, hold=9_000, retry=None))
    assert res.outcome == "delivered", res
    # The hold means the agent is picked up inside an attempt, not only
    # between attempts: far fewer sockets than main burned in half the time.
    assert res.sockets <= 6, res.sockets
    assert res.elapsed_ms <= 90_000, res.elapsed_ms


def test_hold_delivers_at_T150():
    """The case PR #710 cannot reach."""
    res = run(_run_with_mp(_scenario, ready_at=150_000, hold=9_000, retry=None))
    assert res.outcome == "delivered", res
    assert res.elapsed_ms <= 170_000, res.elapsed_ms


def test_hold_total_coverage_exceeds_four_and_a_half_minutes():
    """An agent that never arrives: the client must still be trying well past
    the 34 s main gives it, and must end with the SAME terminal error (no new
    failure mode). ~281 s is the achievable maximum against build 109 — see
    the docstring of test_hold_above_client_silent_grace_is_a_regression for
    why it is not 300."""
    res = run(_run_with_mp(_scenario, ready_at=None, hold=9_000, retry=None))
    assert res.outcome == "terminal_error", res
    assert res.attempts == MAX_WARMUP_ATTEMPTS
    assert res.elapsed_ms >= 270_000, res.elapsed_ms
    assert "taking longer than usual" in (res.error or "")


def test_hold_above_client_silent_grace_is_a_regression():
    """THE CEILING, pinned. A 13 s hold is WORSE than shipping nothing.

    build 109 arms `TURN_SILENT_GRACE_MS = 12_000` just before `ws.send()`.
    A hold longer than that produces no frame in time, so strike one tears the
    attempt down and strike two — `silentStrikes` is not reset between attempts
    — raises a terminal "Connection lost" after ~25 s and TWO attempts. This
    test exists so that "just hold a bit longer" is a red build, not an
    incident.
    """
    res = run(_run_with_mp(_scenario, ready_at=74_000, hold=13_000, retry=None))
    assert res.outcome == "error", res
    assert res.attempts == 2, res.attempts
    assert "Connection lost" in (res.error or "")
    assert res.elapsed_ms < 40_000, res.elapsed_ms


def test_retry_after_ms_is_derived_and_saturates_at_the_client_clamp():
    """Unscaled, against the shipped defaults: the derivation must land on
    10 000 (the client's own ceiling), and must never emit below 500 (its
    floor) even if someone sets a hold that already exceeds the target."""
    assert proxy._compute_retry_after_ms() == 10_000
    assert proxy._compute_retry_after_ms(hold_ms=100_000) == 500
    # 15 x 9000 + 14 x 10000 = 275 s of sleep+hold, before per-attempt
    # connect overhead — the number quoted in the report.
    hold = settings.agent_ws_proxy_hold_ms
    total = MAX_WARMUP_ATTEMPTS * hold + (MAX_WARMUP_ATTEMPTS - 1) * proxy._compute_retry_after_ms()
    assert total >= 270_000, total


# ══════════════════════════════════════════════════════════════════════
# 3. No cost to users whose agent is already up
# ══════════════════════════════════════════════════════════════════════

def test_warm_agent_pays_no_hold_and_one_lookup():
    """The regression that matters most: 71 of 74 tenants are already up.

    A ready agent must never enter the hold, must cost exactly ONE readiness
    check (the same one main does), and must relay in the first attempt.
    """
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=0)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        res = await Build109Client(rig.app).send()
        assert res.outcome == "delivered", res
        assert res.sockets == 1, res.sockets
        assert len(rig.lookups) == 1, rig.lookups
        assert res.elapsed_ms < 5_000, res.elapsed_ms
        assert json.loads(rig.agent.received[0])["text"] == "hello"
    run(_run_with_mp(go))


def test_held_socket_replays_the_clients_frame_to_the_agent():
    """The client sends its chat frame in the tick after `onopen`, so the hold
    necessarily reads it before readiness arrives. It must be forwarded, once,
    in order — otherwise the hold rescues the connection and swallows the
    message it was rescuing."""
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=3_000)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        res = await Build109Client(rig.app).send()
        assert res.outcome == "delivered", res
        assert res.sockets == 1, res.sockets
        assert len(rig.agent.received) == 1, rig.agent.received
        assert json.loads(rig.agent.received[0])["client_msg_id"] == "fixed-id"
    run(_run_with_mp(go))


class _ScaledAsyncio:
    """`asyncio`, with `sleep` on the harness's compressed clock.

    Only used to reach the FAST_FAIL=false decision point without sitting
    through its real 6 x 5 s. Everything else forwards to the real module, so
    the handler's control flow is unchanged.
    """

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, delay, *a, **k):
        return await asyncio.sleep(real_s(delay * 1000.0), *a, **k)


def test_fast_fail_off_path_is_untouched():
    """FAST_FAIL=false keeps the 6x5 s server poll and the 4404 close, and
    NEVER enters the hold.

    Asserted with a spy on `_hold_for_agent` rather than by timing, because
    the flag-off path spends 30 s in its own poll before it would even reach
    the hold — a wall-clock assertion in that window cannot tell the two
    branches apart (a mutation moving the hold out of the fast-fail guard
    survived exactly that test). The second half proves the spy is live.
    """
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        configure(mp, fast_fail=False, hold_virtual_ms=9_000)
        calls: List[str] = []
        real_hold = proxy._hold_for_agent

        async def spy(websocket, user_id, pre_read):
            calls.append(user_id)
            return await real_hold(websocket, user_id, pre_read)

        mp.setattr(proxy, "_hold_for_agent", spy)
        mp.setattr(proxy, "asyncio", _ScaledAsyncio())

        ws = AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text('{"type":"message","text":"x"}')
        frame = await ws.recv(60_000)
        body = json.loads(frame["text"])
        assert body["type"] == "error"
        assert "No active agent found" in body["message"]
        close = await ws.recv(5_000)
        assert close["type"] == "websocket.close" and close["code"] == 4404
        assert calls == [], "the hold ran on the FAST_FAIL=false path"
        assert len(rig.lookups) == 6, rig.lookups     # the original 6 x 5 s
        assert proxy._get_hold_sem()._value == settings.agent_ws_proxy_hold_max_concurrent
        await ws.dispose()

        # ...and with the flag ON the very same spy fires, so its silence
        # above is a fact about the branch, not about the spy.
        mp.setattr(settings, "agent_ws_proxy_fast_fail", True)
        ws2 = AsgiWebSocket(rig.app)
        await ws2.open()
        ws2.send_text('{"type":"message","text":"x"}')
        await ws2.recv(20_000)
        assert len(calls) == 1, calls
        await ws2.dispose()

    run(_run_with_mp(go))


# ══════════════════════════════════════════════════════════════════════
# 4. Bounded concurrency, and no leaked holds
# ══════════════════════════════════════════════════════════════════════

def test_concurrency_cap_degrades_to_fast_fail_not_blocking():
    """Beyond the cap the proxy must answer immediately — today's behaviour —
    rather than queue. A queue would make late arrivals wait behind someone
    else's hold AND their own, blowing the 12 s grace."""
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None, cap=1)

        first = AsgiWebSocket(rig.app)
        await first.open()
        first.send_text('{"type":"message","text":"a"}')
        await asyncio.sleep(real_s(1_500))          # first is now holding

        t = time.monotonic()
        second = AsgiWebSocket(rig.app)
        await second.open()
        second.send_text('{"type":"message","text":"b"}')
        frame = await second.recv(4_000)
        waited_v = (time.monotonic() - t) / SCALE * 1000.0
        assert frame["type"] == "websocket.send"
        body = json.loads(frame["text"])
        assert body["type"] == "agent_starting"
        assert waited_v < 3_000, waited_v      # nowhere near the 9 s hold
        close = await second.recv(2_000)
        assert close["type"] == "websocket.close" and close["code"] == 4503

        # ...and the first connection was genuinely still held meanwhile.
        assert first.task is not None and not first.task.done()
        await first.dispose()
        await second.dispose()
        assert proxy._get_hold_sem()._value == 1, "slot not released"
    run(_run_with_mp(go))


def test_client_disconnect_during_hold_releases_the_slot_immediately():
    """No leak. A client that walks away mid-hold (backgrounded app, dead
    link) must release its semaphore slot at once, not at the hold deadline —
    otherwise a flapping network can pin every slot and the whole mechanism
    degrades to fast-fail for everyone."""
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None, cap=3)
        ws = AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text('{"type":"message","text":"x"}')
        await asyncio.sleep(real_s(1_500))
        assert proxy._get_hold_sem()._value == 2, "hold did not take a slot"

        t = time.monotonic()
        ws.client_close()
        await asyncio.wait_for(asyncio.shield(ws.task), timeout=real_s(4_000))
        freed_v = (time.monotonic() - t) / SCALE * 1000.0
        assert freed_v < 2_000, freed_v            # not the 7.5 s remaining
        assert proxy._get_hold_sem()._value == 3, "slot leaked"
    run(_run_with_mp(go))


def test_hold_does_not_multiply_bridge_breadcrumbs():
    """`_diag` is a live GET at the PROVISIONING BRIDGE, and the bridge was
    timing out a third of its own health probes all through the incident.

    A hold that polls 9 times per connection and breadcrumbs each poll would
    put ~9x the load on the exact component whose slowness it exists to
    survive. The polls must be quiet: one breadcrumb per CONNECTION, from the
    pre-hold lookup and the terminal 4503, and none from the polling.
    """
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        # Re-enable a counting _diag (build_rig stubs it out to protect prod).
        stages: List[str] = []
        mp.setattr(proxy, "_diag", lambda stage, *a, **k: stages.append(stage))
        # ...and a REAL lookup, so its own breadcrumbs are in play.
        mp.setattr(proxy, "_get_agent_ws_info", _never_ready_lookup(rig))

        ws = AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text('{"type":"message","text":"x"}')
        frame = await ws.recv(20_000)
        assert json.loads(frame["text"])["type"] == "agent_starting"
        await ws.dispose()
        assert len(rig.lookups) >= 8, rig.lookups        # it really did poll
        # agent_unresolved + hold_not_ready, and nothing per poll.
        assert len(stages) <= 3, stages
        assert "agent_unresolved" in stages
    run(_run_with_mp(go))


def _never_ready_lookup(rig):
    """The real `_get_agent_ws_info` body's breadcrumb behaviour, without a DB:
    it fires `no_agent_config` exactly when `quiet` is false."""
    async def lookup(user_id: str, quiet: bool = False):
        rig.lookups.append(0.0)
        if not quiet:
            proxy._diag("no_agent_config", user_id)
        return None
    return lookup


def test_frame_landing_during_the_readiness_select_is_not_lost():
    """REVIEW FIX (coordinator, on f0d66f41). The hold's receive is SHIELDED,
    so a `wait_for` timeout leaves it pending — and it can then complete during
    the readiness SELECT on the very iteration that returns `agent_info`.

    The old exit did `recv_task.cancel(); await recv_task` unconditionally.
    `cancel()` on an already-DONE task is a no-op and the `await` hands back the
    message, which was discarded. Since build 109 sends its chat frame the tick
    after `onopen`, the discarded message is THE FIRST MESSAGE OF THE SESSION:
    the hold succeeds, the relay opens, and the agent is handed nothing — a turn
    that never answers, on the exact connection this mechanism exists to rescue.

    The window is made deterministic by injecting the client frame from inside
    the readiness lookup that returns ready, which is precisely the await the
    race lives in.
    """
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        ws = AsgiWebSocket(rig.app)
        await ws.open()
        # Deliberately DON'T send yet — the frame has to land inside the SELECT.
        frame = json.dumps({"type": "message", "text": "in-the-window",
                            "client_msg_id": "race-1"})
        calls = {"n": 0}

        async def lookup(user_id: str, quiet: bool = False):
            calls["n"] += 1
            rig.lookups.append(float(calls["n"]))
            if calls["n"] < 3:          # 1 = the pre-hold lookup, 2 = first poll
                return None
            ws.send_text(frame)
            await asyncio.sleep(0.01)   # let the pending receive() complete
            return ("ws://agent.invalid/api/ws/chat", "k")

        mp.setattr(proxy, "_get_agent_ws_info", lookup)

        done_frame = await ws.recv(20_000)
        assert done_frame["type"] == "websocket.send", done_frame
        assert json.loads(done_frame["text"])["type"] == "done"
        assert len(rig.agent.received) == 1, rig.agent.received
        assert json.loads(rig.agent.received[0])["text"] == "in-the-window"
        await ws.dispose()
    run(_run_with_mp(go))


def test_disconnect_landing_at_the_hold_deadline_is_not_sent_into():
    """The same race on the DEADLINE exit. A slow readiness SELECT can carry
    the hold past its deadline with a completed receive in hand; if that
    receive was a disconnect, the old code discarded it and the caller then
    sent `agent_starting` into a socket that was already gone.

    A text frame arriving here is correctly dropped (we are about to fast-fail
    and the client re-sends, which is what makes the whole retry loop
    exactly-once). A disconnect is not."""
    async def go(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        # adopt off: its first import of pool_service is slow enough to move
        # the deadline, and this test is about WHERE the deadline lands.
        configure(mp, hold_virtual_ms=9_000, poll_virtual_ms=1_000,
                  retry_virtual_ms=None, adopt=False)
        ws = AsgiWebSocket(rig.app)
        await ws.open()
        seen = {"holds": 0}

        async def lookup(user_id: str, quiet: bool = False):
            # `quiet` is set only by the hold's own polls, so this keys off the
            # real call site rather than off a call count.
            if not quiet:
                return None
            seen["holds"] += 1
            rig.lookups.append(float(seen["holds"]))
            if seen["holds"] == 1:
                ws.client_close()
                await asyncio.sleep(0.01)             # receive() resolves: disconnect
                await asyncio.sleep(real_s(15_000))   # ...and we return PAST the deadline
            return None

        mp.setattr(proxy, "_get_agent_ws_info", lookup)
        await asyncio.wait_for(asyncio.shield(ws.task), timeout=real_s(60_000))
        assert seen["holds"] == 1, seen          # the window really was hit
        # Nothing may be written to a socket the client already closed.
        leftover = []
        while not ws._from_app.empty():
            leftover.append(ws._from_app.get_nowait())
        assert leftover == [], f"sent into a disconnected socket: {leftover}"
        assert proxy._get_hold_sem()._value == settings.agent_ws_proxy_hold_max_concurrent
    run(_run_with_mp(go))


def test_stranded_adoption_hook_is_called_once_and_survives_absence():
    """Lane C's `pool_service.try_adopt_stranded` is optional. Called at most
    once per held connection when it exists; a missing hook must not change
    the hold's behaviour at all (this branch has to work standalone)."""
    async def with_hook(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        calls: List[str] = []

        async def hook(db, user_id):
            calls.append(user_id)
            return False

        from app.services import pool_service
        mp.setattr(pool_service, "try_adopt_stranded", hook, raising=False)
        ws = AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text('{"type":"message","text":"x"}')
        await asyncio.sleep(real_s(9_500))
        await ws.dispose()
        assert len(calls) == 1, calls

    async def without_hook(mp):
        rig = build_rig(mp, ready_at_virtual_ms=None)
        configure(mp, hold_virtual_ms=9_000, retry_virtual_ms=None)
        from app.services import pool_service
        if hasattr(pool_service, "try_adopt_stranded"):
            mp.delattr(pool_service, "try_adopt_stranded")
        assert await proxy._try_adopt_stranded("u") is False
        ws = AsgiWebSocket(rig.app)
        await ws.open()
        ws.send_text('{"type":"message","text":"x"}')
        frame = await ws.recv(12_000)
        assert json.loads(frame["text"])["type"] == "agent_starting"
        await ws.dispose()

    run(_run_with_mp(with_hook))
    run(_run_with_mp(without_hook))


# ── monkeypatch plumbing for asyncio.run-based tests ──────────────────
def _run_with_mp(fn, **kwargs):
    """Give an async scenario its own MonkeyPatch, undone afterwards.

    (These tests own an event loop each via asyncio.run, so pytest's function-
    scoped `monkeypatch` fixture cannot be threaded through the coroutine
    boundary cleanly; this is the same object with an explicit lifetime.)
    """
    async def wrapper():
        mp = pytest.MonkeyPatch()
        try:
            return await fn(mp, **kwargs)
        finally:
            mp.undo()
    return wrapper()
