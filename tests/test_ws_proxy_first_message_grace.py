"""The first message survives a booting agent AND a stale key (L4 §6a).

What the founder saw on 2026-09-12, in three server frames:

  19:14:29.6  dial → InvalidStatus → {"type":"error","message":"Cannot reach
              agent: …"} + close 4502
  19:14:38.4  same
  19:14:42.5  dial OK, and the AGENT answered
              {"type":"error","message":"Authentication required"} + close 4001

The third is the screenshot. It carries no `code`, matches none of the mobile
classifier's four classes, and renders "Something went wrong on my end and this
didn't go through." with a Send-again chip that reproduced it for 2 h 28 m.

Three structural facts behind it, and one test class each here:

  1. Every warm-up mechanism — the 9 s hold, `agent_starting`, close 4503, the
     client's 15-attempt budget — was gated on `if not agent_info`, i.e. on the
     user having NO agent row. The observed failure is the OPPOSITE shape: the
     row is present and WRONG. That shape got three dials and a terminal error.
  2. The agent's close CODE never reached the client: `agent_to_browser`'s
     `finally` did a bare `close()`, so 4001 and 4503 both arrived as 1000 and
     `AGENT_WARMING_CLOSE_CODES` could never see an agent-originated code (D4).
  3. `_diag("upstream_error", …, type(exc).__name__)` recorded the exception
     class, so the one externally-readable record of this branch cannot tell a
     404 (no route — an ownership bug) from a 502 (boot race) (D10).

Budget note (L4 §6a): hold + dial must stay under ~9 s per attempt, because
build 109 arms a one-shot TURN_SILENT_GRACE_MS = 12_000 immediately before
`ws.send` and the SECOND silent strike in one send is terminal. The hold and
the dial therefore SHARE one budget rather than stacking.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q \
        tests/test_ws_proxy_first_message_grace.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any, List, Optional

import pytest
from fastapi import FastAPI

from app.api import ws_chat_proxy as proxy
from app.config import settings

from tests.test_ws_proxy_warmup_harness import AsgiWebSocket


def _src(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text()


# ── Doubles ───────────────────────────────────────────────────────────

class FakeInvalidStatus(Exception):
    """`websockets.InvalidStatus` carries `.response.status_code`."""

    def __init__(self, status_code: int):
        super().__init__(f"server rejected WebSocket connection: HTTP {status_code}")
        self.response = types.SimpleNamespace(status_code=status_code)


class ScriptedAgent:
    """An upstream socket that emits a fixed list of frames, then closes."""

    def __init__(self, frames: List[dict], close_code: Optional[int] = 1000):
        self._frames = list(frames)
        self.received: List[str] = []
        self.close_code = close_code
        self._closed = False

    async def send(self, raw: str) -> None:
        self.received.append(raw)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._frames:
            return json.dumps(self._frames.pop(0))
        raise StopAsyncIteration

    async def close(self) -> None:
        self._closed = True


def build_app(monkeypatch, *, connect, lookup, fast_fail=True, hold_ms=900):
    app = FastAPI()
    app.include_router(proxy.router, prefix="/api")

    async def fake_auth(token: str) -> str:
        return "f261b564-0000-0000-0000-000000000000"

    monkeypatch.setattr(proxy, "_diag", lambda *a, **k: None)
    monkeypatch.setattr(proxy, "_authenticate_ws", fake_auth)
    monkeypatch.setattr(proxy, "_get_agent_ws_info", lookup)
    monkeypatch.setitem(sys.modules, "websockets",
                        types.SimpleNamespace(connect=connect))
    monkeypatch.setattr(proxy, "_hold_sem", None, raising=False)
    monkeypatch.setattr(proxy, "_hold_sem_size", -1, raising=False)
    monkeypatch.setattr(settings, "agent_ws_proxy_fast_fail", fast_fail)
    monkeypatch.setattr(settings, "agent_ws_proxy_hold_ms", hold_ms)
    monkeypatch.setattr(settings, "agent_ws_proxy_hold_poll_ms", 100)
    monkeypatch.setattr(settings, "agent_ws_proxy_hold_max_concurrent", 8)
    return app


async def _recv_real(ws, timeout_s: float = 6.0):
    """One server→client message, in REAL seconds.

    Deliberately not `AsgiWebSocket.recv`: the warm-up harness compresses time
    by SCALE=0.05 and writes every duration in virtual ms, which is right for
    modelling the client's timers and wrong here — these tests drive the
    server's own real-clock budget (`agent_ws_proxy_hold_ms`). Mixing the two
    makes an 8 000 "ms" wait 400 real ms and every assertion a timeout.
    """
    try:
        return await asyncio.wait_for(ws._from_app.get(), timeout=timeout_s)
    except asyncio.TimeoutError:
        if ws.task is not None and ws.task.done():
            exc = ws.task.exception()
            if exc is not None:
                raise exc
        raise


async def _drain(ws, n=8, timeout_s: float = 6.0):
    """Collect server→client messages until the socket closes."""
    out = []
    for _ in range(n):
        try:
            msg = await _recv_real(ws, timeout_s)
        except asyncio.TimeoutError:
            break
        out.append(msg)
        if msg["type"] == "websocket.close":
            break
    return out


# ── 1. A row that is present but whose upstream is not ready ──────────

@pytest.mark.asyncio
async def test_a_404_then_a_101_becomes_agent_starting_then_a_relay(monkeypatch):
    """The hold now covers "row present but the dial fails". A 404 from Caddy
    (the tenant route not asserted yet — exactly 19:14:29 on 12 Sep) must be
    retried within the budget, not answered with a terminal error."""
    agent = ScriptedAgent([{"type": "done", "message_id": "m1"}])
    attempts = {"n": 0}

    async def connect(url, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeInvalidStatus(404)
        return agent

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", "k")

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=6000)
    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi","client_msg_id":"c1"}')

    msgs = await _drain(ws)
    kinds = [json.loads(m["text"])["type"] for m in msgs if m["type"] == "websocket.send"]
    assert "done" in kinds, msgs
    assert attempts["n"] == 2, "the 404 must be retried, not surfaced"
    # The client's message survived the retry and reached the agent exactly once.
    assert len(agent.received) == 1
    assert json.loads(agent.received[0])["client_msg_id"] == "c1"
    await ws.dispose()


@pytest.mark.asyncio
async def test_a_dial_that_never_succeeds_answers_agent_starting_not_an_error(monkeypatch):
    """The founder's first two frames. `{"type":"error","message":"Cannot
    reach agent: …"}` + 4502 is replaced by `agent_starting` + a code + a
    retry cadence + 4503, which is what keeps the client's budget alive."""
    async def connect(url, **kw):
        raise FakeInvalidStatus(502)

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", "k")

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi"}')

    msgs = await _drain(ws)
    sent = [json.loads(m["text"]) for m in msgs if m["type"] == "websocket.send"]
    closes = [m for m in msgs if m["type"] == "websocket.close"]
    assert sent, msgs
    assert sent[0]["type"] == "agent_starting"
    assert sent[0]["code"] == "agent_booting"
    assert isinstance(sent[0]["retry_after_ms"], int)
    assert closes and closes[0]["code"] == 4503
    await ws.dispose()


@pytest.mark.asyncio
async def test_a_timeout_is_classed_unreachable_not_booting(monkeypatch):
    async def connect(url, **kw):
        await asyncio.sleep(30)

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", "k")

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi"}')
    msgs = await _drain(ws, timeout_s=25.0)
    sent = [json.loads(m["text"]) for m in msgs if m["type"] == "websocket.send"]
    assert sent and sent[0]["type"] == "agent_starting"
    assert sent[0]["code"] == "agent_unreachable"
    await ws.dispose()


@pytest.mark.asyncio
async def test_the_row_is_re_read_between_dial_attempts(monkeypatch):
    """On 12 Sep the bridge healed the agent's key at 19:17:23 — 2 min 40 s
    after the phone had given up. Nothing on this path ever asked again."""
    agent = ScriptedAgent([{"type": "done"}])
    keys_used: List[str] = []
    lookups = {"n": 0}

    async def connect(url, **kw):
        keys_used.append(kw["additional_headers"]["X-Agent-Key"])
        if len(keys_used) == 1:
            raise FakeInvalidStatus(502)
        return agent

    async def lookup(user_id, quiet=False):
        lookups["n"] += 1
        key = "stale" if lookups["n"] <= 1 else "repaired"
        return ("ws://agent.invalid/api/ws/chat", key)

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=6000)
    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi"}')
    await _drain(ws)
    assert keys_used == ["stale", "repaired"], keys_used
    await ws.dispose()


# ── 2. The agent's identity rejection ─────────────────────────────────

@pytest.mark.asyncio
async def test_authentication_required_triggers_one_repair_and_one_redial(monkeypatch):
    """The exact 19:14:42 frame. It must never reach the client: repair once,
    re-dial once, then `agent_starting` with `code=agent_key_stale`."""
    rejecting = ScriptedAgent(
        [{"type": "error", "message": "Authentication required"}], close_code=4001,
    )
    dials = {"n": 0}
    repairs = {"n": 0}

    async def connect(url, **kw):
        dials["n"] += 1
        return ScriptedAgent(
            [{"type": "error", "message": "Authentication required"}],
            close_code=4001,
        )

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", "platform-key")

    async def fake_probe(agent_url, key):
        return 401

    async def fake_reconcile(user_id):
        repairs["n"] += 1
        return "adopt"

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    monkeypatch.setattr(proxy, "_probe_agent_identity", fake_probe)
    monkeypatch.setattr(
        "app.services.pool_service.reconcile_agent_identity", fake_reconcile,
    )

    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi","client_msg_id":"c1"}')
    msgs = await _drain(ws)

    sent = [json.loads(m["text"]) for m in msgs if m["type"] == "websocket.send"]
    closes = [m for m in msgs if m["type"] == "websocket.close"]

    assert repairs["n"] == 1, "exactly one repair"
    assert dials["n"] == 2, "exactly one re-dial, never two"
    assert not any(
        f.get("message") == "Authentication required" for f in sent
    ), f"the agent's rejection reached the client: {sent}"
    assert sent and sent[-1]["type"] == "agent_starting"
    assert sent[-1]["code"] == "agent_key_stale"
    assert closes and closes[0]["code"] == 4503
    await ws.dispose()


@pytest.mark.asyncio
async def test_a_successful_repair_relays_and_replays_the_message_once(monkeypatch):
    """The wrong-key path EATS the message (the agent's first-frame auth
    reader parses it, finds `type != "auth"` and drops it), so the replay is a
    first delivery — not a duplicate. That is the property the whole retry
    design rests on."""
    good = ScriptedAgent([{"type": "done", "message_id": "m1"}])
    dials = {"n": 0}

    async def connect(url, **kw):
        dials["n"] += 1
        if dials["n"] == 1:
            return ScriptedAgent(
                [{"type": "error", "message": "Authentication required"}],
                close_code=4001,
            )
        return good

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", f"key{dials['n']}")

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    monkeypatch.setattr(proxy, "_probe_agent_identity",
                        lambda *a, **k: _async_value(401))
    monkeypatch.setattr("app.services.pool_service.reconcile_agent_identity",
                        lambda uid: _async_value("adopt"))

    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi","client_msg_id":"c1"}')
    msgs = await _drain(ws)

    kinds = [json.loads(m["text"])["type"] for m in msgs if m["type"] == "websocket.send"]
    assert "done" in kinds, msgs
    assert len(good.received) == 1, good.received
    assert json.loads(good.received[0])["client_msg_id"] == "c1"
    await ws.dispose()


def _async_value(v):
    async def _f(*a, **k):
        return v
    return _f()


@pytest.mark.asyncio
async def test_a_200_probe_means_do_not_repair_on_a_guess(monkeypatch):
    """The rejection came from somewhere other than our key. Repairing on that
    would be acting on a guess — and this repair rewrites a live user's row."""
    repairs = {"n": 0}

    async def connect(url, **kw):
        return ScriptedAgent(
            [{"type": "error", "message": "Authentication required"}],
            close_code=4001,
        )

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", "same-key")

    async def fake_reconcile(user_id):
        repairs["n"] += 1
        return "adopt"

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    monkeypatch.setattr(proxy, "_probe_agent_identity",
                        lambda *a, **k: _async_value(200))
    monkeypatch.setattr("app.services.pool_service.reconcile_agent_identity",
                        fake_reconcile)

    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi"}')
    msgs = await _drain(ws)
    assert repairs["n"] == 0, "a healthy probe must not trigger a rewrite"
    sent = [json.loads(m["text"]) for m in msgs if m["type"] == "websocket.send"]
    assert sent and sent[-1]["type"] == "agent_starting"
    await ws.dispose()


@pytest.mark.asyncio
async def test_a_key_changed_under_us_redials_without_a_repair(monkeypatch):
    """Two replicas hit this at once and the sweep may have fixed it a second
    ago. The cheapest correct action is to dial again with the new key."""
    good = ScriptedAgent([{"type": "done"}])
    dials = {"n": 0}
    repairs = {"n": 0}
    probes = {"n": 0}

    async def connect(url, **kw):
        dials["n"] += 1
        if dials["n"] == 1:
            return ScriptedAgent(
                [{"type": "error", "message": "Authentication required"}],
                close_code=4001,
            )
        return good

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat",
                "old" if dials["n"] < 1 else ("new" if dials["n"] >= 1 else "old"))

    async def fake_probe(*a, **k):
        probes["n"] += 1
        return 401

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    monkeypatch.setattr(proxy, "_probe_agent_identity", fake_probe)
    monkeypatch.setattr("app.services.pool_service.reconcile_agent_identity",
                        lambda uid: _async_value("adopt"))
    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi"}')
    await _drain(ws)
    # The first dial used "old" (the value captured before the relay); the
    # re-read answered "new", so the heal short-circuits before the probe.
    assert probes["n"] == 0, "a changed key must not spend a probe or a repair"
    await ws.dispose()


@pytest.mark.asyncio
async def test_an_identity_rejection_after_real_content_is_not_retried(monkeypatch):
    """Once the thread has content, tearing the socket down to retry is worse
    than the error. The guard is `sent_to_client == 0`."""
    dials = {"n": 0}

    async def connect(url, **kw):
        dials["n"] += 1
        return ScriptedAgent([
            {"type": "text_chunk", "text": "hello"},
            {"type": "error", "message": "Authentication required"},
        ], close_code=4001)

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", "k")

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi"}')
    msgs = await _drain(ws)
    assert dials["n"] == 1, "no re-dial once content has reached the client"
    sent = [json.loads(m["text"]) for m in msgs if m["type"] == "websocket.send"]
    assert any(f["type"] == "text_chunk" for f in sent)
    await ws.dispose()


# ── 3. The close code reaches the client (D4) ─────────────────────────

@pytest.mark.asyncio
async def test_the_upstream_close_code_is_propagated(monkeypatch):
    async def connect(url, **kw):
        return ScriptedAgent([{"type": "done"}], close_code=4503)

    async def lookup(user_id, quiet=False):
        return ("ws://agent.invalid/api/ws/chat", "k")

    app = build_app(monkeypatch, connect=connect, lookup=lookup, hold_ms=600)
    ws = AsgiWebSocket(app)
    await ws.open()
    ws.send_text('{"type":"message","text":"hi"}')
    msgs = await _drain(ws)
    closes = [m for m in msgs if m["type"] == "websocket.close"]
    assert closes and closes[0]["code"] == 4503, msgs
    await ws.dispose()


def test_reserved_close_codes_are_never_put_on_the_wire():
    """1005 (no status) and 1006 (abnormal) are RESERVED — sending either is a
    protocol violation, and `websockets` reports both.

    Round 46: they are no longer laundered to 1000. Mapping the most abnormal
    upstream failure onto the most NORMAL close is what left the client with
    nothing to class on in incident 3 — they now carry 4506
    (`platform_upstream_lost`), which is still a legal code and is honest
    about where the connection broke. 1011 joins them: it is what a future
    uvicorn sends on the same path that gives us 1006 today.
    See tests/test_ws_proxy_fault_honesty.py."""
    assert proxy._safe_close_code(1005) == 4506
    assert proxy._safe_close_code(1006) == 4506
    assert proxy._safe_close_code(1011) == 4506
    assert proxy._safe_close_code(None) == 1000
    assert proxy._safe_close_code(1000) == 1000
    assert proxy._safe_close_code(4503) == 4503
    # 4001 becomes 4503: by the time we propagate, the identity repair has
    # already been attempted, so a retry is safe — and the web client STOPS
    # reconnecting on 4001.
    assert proxy._safe_close_code(4001) == 4503


# ── 4. The breadcrumb records the status (D10) ────────────────────────

def test_exc_status_reads_both_websockets_spellings():
    assert proxy._exc_status(FakeInvalidStatus(404)) == 404
    legacy = Exception("old")
    legacy.status_code = 502
    assert proxy._exc_status(legacy) == 502
    assert proxy._exc_status(Exception("no status")) is None
    assert proxy._exc_status(None) is None


def test_the_upstream_error_breadcrumb_carries_the_status_not_just_the_class():
    src = _src("app/api/ws_chat_proxy.py")
    assert 'type(_exc).__name__}:{_status' in src, (
        "x=InvalidStatus cannot tell a 404 (ownership bug) from a 502 (boot "
        "race), and the breadcrumb is the only externally-readable record"
    )


def test_dial_failure_codes_are_the_three_documented_classes():
    assert set(proxy.AGENT_STARTING_CODES) == {
        "agent_key_stale", "agent_booting", "agent_unreachable",
    }
    assert proxy._dial_failure_code(("error", FakeInvalidStatus(404))) == "agent_booting"
    assert proxy._dial_failure_code(("error", FakeInvalidStatus(502))) == "agent_booting"
    assert proxy._dial_failure_code(("timeout", None)) == "agent_unreachable"
    assert proxy._dial_failure_code(("error", Exception("dns"))) == "agent_unreachable"


# ── 5. The budget, and the direction invariant ────────────────────────

def test_hold_and_dial_share_one_budget():
    """Stacking a dial ON TOP of a 9 s hold would exceed build 109's 12 s
    silent-turn grace and make strike two terminal — worse than the 34 s that
    produced the 2026-09-06 incident. They share `agent_ws_proxy_hold_ms`."""
    src = _src("app/api/ws_chat_proxy.py")
    assert "_budget_deadline" in src
    assert '"deadline": _budget_deadline' in src
    # The hold is handed the budget MINUS a dial reserve, so it can never
    # spend all of it and guarantee a fast-fail.
    assert "_MIN_DIAL_RESERVE_MS" in src
    assert "_budget_left_ms() - _MIN_DIAL_RESERVE_MS" in src
    assert settings.agent_ws_proxy_hold_ms <= 10_000, (
        "config.py's own advice: never past ~10 s, because the client's grace "
        "is 12 s"
    )


def test_the_chat_path_never_pushes_the_platform_key_at_the_route():
    """THE direction invariant (L4 skeptic C4). On 12 Sep the route pointed at
    pool-81 while the platform held the NAMED key. A heal that pushed that key
    at whatever the route hits would have bound the named key onto a live pool
    member — manufacturing the ambiguous-ownership state an operator then had
    to unpick by hand. The proxy may call exactly ONE repair, and that repair
    adopts the routed container INTO the platform row."""
    src = _src("app/api/ws_chat_proxy.py")
    for forbidden in (
        "update_container_env",
        "refresh-config",
        "/admin/bind",
        "claim_for_user",
        "_adopt_discovered_bind",
        "restart_container",
        "/v1/pool/claim",
        "force=True",
    ):
        assert forbidden not in src, (
            f"ws_chat_proxy must not call {forbidden!r} — the chat path's only "
            "repair is pool_service.reconcile_agent_identity, which adopts "
            "bridge truth INTO the platform row"
        )
    assert src.count("reconcile_agent_identity") >= 1


def test_reconcile_agent_identity_never_force_claims():
    """And the repair itself: a force-claim binds an established account to a
    fresh empty slot when the bridge has no bind (R40)."""
    ps = _src("app/services/pool_service.py")
    body = ps[ps.index("async def reconcile_agent_identity"):
              ps.index("async def _adopt_bridge_truth")]
    assert "force=True" not in body
    assert "claim_for_user" not in body
    assert "_adopt_bridge_truth" in body


# ── 6. The agent logs its two rejection branches (D5) ─────────────────

def test_the_agent_logs_both_rejection_branches_with_a_key_fingerprint():
    src = _src("app/api/ws_chat.py")
    assert "REJECT 4001 Authentication required" in src
    assert "REJECT 4503 agent_starting" in src
    assert "presented_key=%s" in src and "expected_key=%s" in src
    # A fingerprint, never the key.
    assert "hexdigest()[:8]" in src
    branch = src[src.index("REJECT 4001"):src.index("REJECT 4001") + 1400]
    assert "settings.agent_api_key," in branch or "_fp(settings.agent_api_key)" in branch
    # The log must sit ABOVE the close that ends the branch.
    assert src.index("REJECT 4001") < src.index(
        'code=4001, message="Authentication required"'
    )
