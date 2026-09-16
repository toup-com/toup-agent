"""The failure-injection matrix: what the user is TOLD when a turn breaks.

Why this file exists
--------------------
2026-09-15, incident 3. PgBouncer died on the VPS; the tenant agent's ws_chat
handler raised `ConnectionRefusedError` on its first database access, which
sat ABOVE the handler's own try/except. uvicorn caught it, logged "Exception
in ASGI application" and closed the transport with no close frame, so the
platform proxy's peer saw **1006**. `_safe_close_code` laundered 1006 into
**1000**: the most abnormal server failure in the system arrived at the phone
as the most normal close, with no code and no frame. The app, having nothing
to class on, matched its own error prose against a regex and rendered "Check
your internet and try again" — four times, to a user whose REST calls were
being answered by the same agent in the same seconds.

Every hop was individually defensible and the composition was a lie. So this
file drives the REAL proxy (`app.api.ws_chat_proxy`) with a `FaultyAgent` and
asserts, for each point at which a turn can break, the two things that decide
what the user sees:

  * the CLOSE CODE the client receives, and
  * whether a coded `{"type":"error", "code": ...}` FRAME preceded it.

The frame matters as much as the code: a close code is lost on several paths
(a browser that never surfaces it, an app build that predates the number),
and both clients already class an error frame by `code`.

The client-side twin of these assertions is `scripts/check-error-honesty.js`
in the app repo, which EXECUTES `classifyClose` and `turnErrorCopy` over the
same codes.

Harness reuse: `AsgiWebSocket`, `build_rig` and `configure` come from
`tests/test_ws_proxy_warmup_harness.py` — the same in-process ASGI socket the
warm-up matrix uses, so the proxy under test is the shipped handler.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. python -m pytest -q \
        tests/test_ws_proxy_fault_honesty.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from app.api import _fault_codes as fc
from app.api import ws_chat_proxy as proxy

from tests.test_ws_proxy_warmup_harness import (  # noqa: E402
    AsgiWebSocket,
    build_rig,
    configure,
)


# ── The agent that breaks ─────────────────────────────────────────────
class FaultyAgent:
    """A tenant agent that fails at a chosen point in the turn.

    `fail_at` names the moment, and the failure MODE reproduces what the
    deployed stack actually does:

      * 'before_ack'  — the incident. The agent raises before reading the
        frame; uvicorn closes the transport with no close frame, so the
        proxy's `websockets` client raises `ConnectionClosedError` with
        `close_code = None` → the proxy reads 1006/None off the socket.
      * 'after_ack'   — the agent acknowledges receipt, then dies.
      * 'mid_document'/'before_output' — frames flow, then the socket dies
        mid-turn: the class where the run continues server-side and the
        client must recover from history rather than re-send.
      * 'after_artifact' — an attachment frame was delivered and then the
        socket died before `done`.
      * 'clean' — the control: a normal turn, which must be untouched.
    """

    def __init__(self, fail_at: str, upstream_code: Optional[int] = None) -> None:
        self.fail_at = fail_at
        self._upstream_code = upstream_code
        # What the proxy reads off the socket afterwards. `websockets` reports
        # 1006 when the peer closed the TCP connection without a close frame —
        # which is exactly what uvicorn does when the ASGI app raises, and
        # therefore exactly what the incident put on this attribute.
        self.close_code: Optional[int] = None
        self.received: List[str] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._dead = False

    # The proxy calls `.send(raw)` for browser→agent frames.
    async def send(self, raw: str) -> None:
        self.received.append(raw)
        if self.fail_at == "before_ack":
            # Nothing is ever emitted; the read side dies below.
            await self.inbox.put(_Die(self._upstream_code))
            return
        await self.inbox.put(json.dumps({
            "type": "status", "stage": "received", "mission_id": "chatturn:1",
        }))
        if self.fail_at == "after_ack":
            await self.inbox.put(_Die(self._upstream_code))
            return
        if self.fail_at in ("mid_document", "before_output"):
            await self.inbox.put(json.dumps({"type": "tool_start", "tool": "read_file"}))
            if self.fail_at == "mid_document":
                await self.inbox.put(_Die(self._upstream_code))
                return
            await self.inbox.put(json.dumps({"type": "text_chunk", "text": "here is"}))
            await self.inbox.put(_Die(self._upstream_code))
            return
        if self.fail_at == "after_artifact":
            await self.inbox.put(json.dumps({
                "type": "attachment", "message_id": "m1", "attachment_id": "a1",
                "filename": "report.pdf", "mime_type": "application/pdf",
            }))
            await self.inbox.put(_Die(self._upstream_code))
            return
        await self.inbox.put(json.dumps({"type": "done", "message_id": "m1"}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, _Die):
            self._dead = True
            self.close_code = item.code if item.code is not None else 1006
            raise _closed_error(item.code)
        return item

    async def close(self) -> None:
        await self.inbox.put(None)


class _Die:
    def __init__(self, code: Optional[int]) -> None:
        self.code = code


def _closed_error(code: Optional[int]) -> Exception:
    """What `websockets` raises for each shape of upstream death.

    `code is None` is the incident: a bare transport close, no close frame.
    The proxy reads `agent_ws.close_code` afterwards, which is None there and
    the integer otherwise."""
    try:
        from websockets.exceptions import ConnectionClosedError  # type: ignore

        return ConnectionClosedError(None, None)
    except Exception:  # websockets is stubbed out in this harness
        return ConnectionError("no close frame received or sent")


# ── Driver ────────────────────────────────────────────────────────────
class Outcome:
    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []
        self.close_code: Optional[int] = None

    @property
    def fault_frames(self) -> List[Dict[str, Any]]:
        return [f for f in self.frames if f.get("type") == "error" and f.get("code")]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Outcome(close={self.close_code}, frames={self.frames})"


async def _drive(mp, fail_at: str, upstream_code: Optional[int] = None) -> Outcome:
    rig = build_rig(mp, 0)
    agent = FaultyAgent(fail_at, upstream_code)
    # build_rig wires its own FakeAgent; swap in the faulty one.
    import sys
    import types as _types

    async def fake_connect(url, **kwargs):
        return agent

    mp.setitem(sys.modules, "websockets", _types.SimpleNamespace(connect=fake_connect))
    configure(mp, hold_virtual_ms=0, retry_virtual_ms=None)

    out = Outcome()
    ws = AsgiWebSocket(rig.app)
    await ws.open()
    ws.send_text(json.dumps({
        "type": "message", "text": "hi", "client_msg_id": "cmid-1", "channel": "mobile",
    }))
    try:
        while True:
            msg = await ws.recv(30_000)
            if msg["type"] == "websocket.close":
                out.close_code = msg.get("code", 1000)
                break
            if msg["type"] == "websocket.send":
                try:
                    out.frames.append(json.loads(msg.get("text") or "{}"))
                except (ValueError, TypeError):
                    pass
    except asyncio.TimeoutError:
        pass
    finally:
        await ws.dispose()
    return out


def run(fail_at: str, upstream_code: Optional[int] = None) -> Outcome:
    async def wrapper():
        mp = pytest.MonkeyPatch()
        try:
            return await _drive(mp, fail_at, upstream_code)
        finally:
            mp.undo()
    return asyncio.run(wrapper())


# ══════════════════════════════════════════════════════════════════════
# 1. The transformation that caused the incident
# ══════════════════════════════════════════════════════════════════════

def test_safe_close_code_no_longer_launders_an_abnormal_close_to_normal():
    """1006 → 1000 was the single most misleading step on the path."""
    assert proxy._safe_close_code(1006) == fc.FAULTS[fc.PLATFORM_UPSTREAM_LOST]["close_code"]
    assert proxy._safe_close_code(1005) == fc.FAULTS[fc.PLATFORM_UPSTREAM_LOST]["close_code"]
    # A future uvicorn that sends a real 1011 must be laundered the same way:
    # 1011 is in the reserved-but-sendable range, so it used to pass through
    # unlaundered and the client would have classed it as an unknown close.
    assert proxy._safe_close_code(1011) == fc.FAULTS[fc.PLATFORM_UPSTREAM_LOST]["close_code"]


def test_safe_close_code_leaves_the_existing_vocabulary_alone():
    assert proxy._safe_close_code(4001) == 4503        # identity repair already tried
    assert proxy._safe_close_code(4503) == 4503
    assert proxy._safe_close_code(4505) == 4505        # the agent's own fault code
    assert proxy._safe_close_code(1000) == 1000
    assert proxy._safe_close_code(1001) == 1001
    assert proxy._safe_close_code(None) == 1000
    assert proxy._safe_close_code("nonsense") == 1000


def test_reserved_codes_are_never_put_on_the_wire():
    """1005/1006 are RESERVED: sending them is a protocol violation, which is
    why the original laundering existed. The fix must not reintroduce that."""
    for code in (1005, 1006, None, 1002, 1011, 4001, 4505):
        assert proxy._safe_close_code(code) not in (1005, 1006)


# ══════════════════════════════════════════════════════════════════════
# 2. The matrix: where the turn broke ⇒ what the client is told
# ══════════════════════════════════════════════════════════════════════

def test_before_ack_is_the_incident_and_is_now_coded():
    """The incident row. Nothing was relayed, so the client cannot know the
    agent ever existed — it must be TOLD, and the close CODE is how."""
    out = run("before_ack")
    assert out.close_code == 4506, out


def test_an_unexplained_upstream_end_sends_NO_frame():
    """`platform_upstream_lost` means "the upstream ended and we do not know
    why" — a container swap mid-dial during a fleet rollout produces exactly
    this (1006, nothing relayed). In builds 123-126 an `{type:'error'}` frame
    settles the turn, and a settled turn makes `onclose`'s `if (settled)
    return` skip the bounded SILENT resend that is what eventually got
    incident 2's turn through. A code those builds ignore costs nothing; a
    frame they obey costs the retry."""
    out = run("before_ack")
    assert out.fault_frames == [], out
    assert out.close_code == 4506, out


def test_an_agent_authored_fault_DOES_send_its_frame():
    """The other half of the rule. 4505/4507/4508 are verdicts the agent
    reached and named — an immediate, honest error is the intent — so the
    class travels as a frame too, for a client that does not know the number.
    (In practice the agent sends its own frame first, which the proxy relays;
    this covers the path where it closed with the code and nothing else.)"""
    out = run("before_ack", upstream_code=fc.FAULTS[fc.AGENT_DB_UNAVAILABLE]["close_code"])
    assert out.close_code == 4505, out
    assert out.fault_frames, out
    frame = out.fault_frames[0]
    assert frame["code"] == fc.AGENT_DB_UNAVAILABLE
    # The frame is a classification, not a sentence the app shows verbatim,
    # but it must never itself blame the user's connection.
    assert "internet" not in frame["message"].lower()


def test_a_warm_up_close_sends_NO_frame_only_the_code():
    """D13's exception. 4404/4502/4503/4504 mean "not yet", and the client
    rides them on its bounded warm-up resend. A fault FRAME here would settle
    the turn as an error and delete that resend — the agent normally sends its
    own `agent_starting` frame first, so this covers the bare-close path."""
    for code in sorted(proxy._WARMUP_UPSTREAM_CODES):
        out = run("before_ack", upstream_code=code)
        assert out.fault_frames == [], (code, out)
        assert out.close_code == code, (code, out)


def test_after_ack_still_gets_a_coded_close():
    """An ack is not content: the agent said "received" and then died, so the
    turn never ran and the client may re-send with the same id."""
    out = run("after_ack")
    assert out.close_code == 4506, out
    assert [f.get("type") for f in out.frames][:1] == ["status"]


def test_mid_document_death_does_not_get_a_fault_frame():
    """Frames were already delivered. The run continues server-side and the
    client recovers from history (`onDropped`), so relabelling this as a
    failure would produce a duplicate send AND a false alarm — the exact
    reason the frame is gated on `sent_to_client == 0`."""
    out = run("mid_document")
    assert out.close_code == 4506, out
    assert out.fault_frames == [], out
    assert any(f.get("type") == "tool_start" for f in out.frames)


def test_before_final_output_death_does_not_get_a_fault_frame():
    out = run("before_output")
    assert out.close_code == 4506, out
    assert out.fault_frames == [], out
    assert any(f.get("type") == "text_chunk" for f in out.frames)


def test_artifact_created_then_death_keeps_the_attachment_frame():
    """The attachment was delivered; only the `done` was lost. The client's
    recovery is history, not a re-send — so again, no fault frame, and the
    attachment must still have reached it."""
    out = run("after_artifact")
    assert out.close_code == 4506, out
    assert out.fault_frames == [], out
    assert any(f.get("type") == "attachment" for f in out.frames)


def test_clean_turn_is_untouched():
    """The control. Every assertion above is worthless if the ordinary path
    changed shape."""
    out = run("clean")
    assert any(f.get("type") == "done" for f in out.frames), out
    assert out.fault_frames == [], out
    assert out.close_code in (None, 1000), out


# ══════════════════════════════════════════════════════════════════════
# 3. The breadcrumb that recorded ten failures as successes
# ══════════════════════════════════════════════════════════════════════

def test_relay_end_breadcrumb_exists_and_dial_ok_replaced_relay_ok():
    """`relay_ok` fired the instant the DIAL succeeded, before a byte moved,
    so the bridge journal recorded all ten of the incident's failed relays as
    successes. The rename is load-bearing for anyone grepping it."""
    src = _proxy_source()
    assert '_diag("dial_ok"' in src
    assert '_diag("relay_ok"' not in src
    assert '_diag(\n            "relay_end"' in src or '"relay_end"' in src


def test_relay_end_carries_no_user_content():
    """Telemetry rule: stage, counts, durations and codes only."""
    src = _proxy_source()
    start = src.index('"relay_end"')
    window = src[start:start + 400]
    for banned in ("text", "message", "client_msg_id", "raw"):
        assert f"{banned}=" not in window, f"relay_end leaks {banned}"


def test_turntrace_records_the_hop_without_the_id_or_the_text(caplog):
    """C10. The proxy's contribution to the one query that can answer "what
    happened to this message" — the reconstruction of incident 3 had to count
    WebSocket accepts instead."""
    import logging

    with caplog.at_level(logging.INFO, logger="app.api._turn_trace"):
        out = run("clean")
    assert any(f.get("type") == "done" for f in out.frames), out
    lines = [r.getMessage() for r in caplog.records if "[TURNTRACE]" in r.getMessage()]
    assert any('"stage":"platform_receipt"' in ln for ln in lines), lines
    blob = "\n".join(lines)
    assert "cmid-1" not in blob, blob          # the raw id never appears
    assert "hi" not in blob.replace("this", ""), blob   # nor the message text


def _proxy_source() -> str:
    import inspect

    return inspect.getsource(proxy)


# ══════════════════════════════════════════════════════════════════════
# 4. The vocabulary itself
# ══════════════════════════════════════════════════════════════════════

def test_fault_frame_shape_is_the_contract():
    frame = fc.fault_frame(fc.AGENT_DB_UNAVAILABLE)
    assert frame["type"] == "error"
    assert frame["code"] == fc.AGENT_DB_UNAVAILABLE
    assert frame["retryable"] is True
    assert isinstance(frame["retry_after_ms"], int)
    assert isinstance(frame["message"], str) and frame["message"]


def test_fault_frame_detail_never_carries_content():
    """`detail` is machine-oriented. It is truncated hard so a caller that
    passes an exception string cannot turn it into a content channel."""
    frame = fc.fault_frame(fc.AGENT_INTERNAL, detail="x" * 5000)
    assert len(frame["detail"]) <= 200


def test_unknown_code_is_internal_never_normal():
    assert fc.close_code_for("something_new") == fc.FAULTS[fc.AGENT_INTERNAL]["close_code"]
    assert fc.fault_frame("something_new")["code"] == fc.AGENT_INTERNAL


def test_close_codes_are_stable_numbers():
    """The app's `src/shared/wsFaults.ts` hard-codes these. A silent change
    here makes every classed sentence over there unreachable in production."""
    assert fc.FAULTS[fc.AGENT_DB_UNAVAILABLE]["close_code"] == 4505
    assert fc.FAULTS[fc.PLATFORM_UPSTREAM_LOST]["close_code"] == 4506
    assert fc.FAULTS[fc.AGENT_INTERNAL]["close_code"] == 4507
    assert fc.FAULTS[fc.AGENT_NOT_READY]["close_code"] == 4503
    assert fc.FAULTS[fc.ATTACHMENT_REJECTED]["close_code"] == 4508
    # Mirrors src/shared/wsFaults.ts INFRA_CLOSE_CODES exactly (4508 included).
    assert fc.INFRA_CLOSE_CODES == frozenset({4505, 4506, 4507, 4508})


def test_infra_error_classifier_separates_down_from_wrong():
    """"I could not check" must never be reported as "you are not who you say
    you are" — that conflation is what turned the outage into 4001 → 4503 →
    fifteen silent re-sends."""
    assert fc.is_infra_error(ConnectionRefusedError(111, "Connection refused"))
    assert fc.is_infra_error(ConnectionResetError("peer reset"))
    assert fc.is_infra_error(fc.WsInfraUnavailable())
    assert not fc.is_infra_error(ValueError("bad token"))
    assert not fc.is_infra_error(KeyError("sub"))
    # A bare OSError is NOT infrastructure. `FileNotFoundError` is one, and a
    # missing file is a deterministic bug, not an outage to retry against.
    assert not fc.is_infra_error(FileNotFoundError("/tmp/nope"))
    assert not fc.is_infra_error(OSError("something else"))


def test_the_two_infra_predicates_can_never_drift_again():
    """Round 46 shipped two of these — `_fault_codes.is_infra_error` on the WS
    fault path and `_infra_errors.is_infrastructure_error` on the swept REST
    reads — and they disagreed on ProgrammingError / DataError / IntegrityError.
    The WS side called them infrastructure, so a tenant with schema drift (or
    an over-long client-supplied `client_msg_id`) closed 4505 `retryable:true`
    and the app re-sent the same id forever against a PERMANENT failure, while
    the REST twin reported the same exception honestly. The WS predicate now
    delegates; this pins that it still does."""
    from sqlalchemy.exc import (
        DataError,
        IntegrityError,
        InterfaceError,
        OperationalError,
        ProgrammingError,
    )

    from app.api._infra_errors import is_infrastructure_error

    def _mk(cls):
        return cls("SELECT 1", {}, Exception("boom"))

    for cls in (ProgrammingError, DataError, IntegrityError, OperationalError, InterfaceError):
        exc = _mk(cls)
        assert fc.is_infra_error(exc) == is_infrastructure_error(exc), cls.__name__

    for exc in (FileNotFoundError(), ConnectionRefusedError(), ValueError(), OSError()):
        assert fc.is_infra_error(exc) == is_infrastructure_error(exc), type(exc).__name__

    # …and the deterministic SQL families are on the NOT-infrastructure side,
    # i.e. `agent_internal` (4507, retryable: False), not a retry loop.
    assert not fc.is_infra_error(_mk(ProgrammingError))


def test_a_wrapped_connection_error_is_still_infrastructure():
    """`is_infrastructure_error` walks `__cause__`; the old local predicate did
    not, so a driver-level refusal wrapped in a generic DBAPIError was read by
    its wrapper alone."""
    from sqlalchemy.exc import DBAPIError

    wrapped = DBAPIError("SELECT 1", {}, ConnectionRefusedError(111, "refused"))
    wrapped.__cause__ = ConnectionRefusedError(111, "refused")
    assert fc.is_infra_error(wrapped)


# ── the fault REPORTER may never be the thing that raises ─────────────

def test_safe_send_fault_survives_a_socket_that_is_already_gone():
    """`safe_send_fault_ws` is the only handler on ws_chat's outer safety net.
    It caught `(WebSocketDisconnect, RuntimeError)` — and uvicorn turns a send
    on a half-dead socket into `uvicorn.protocols.utils.ClientDisconnected`,
    which subclasses OSError, while the pinned 0.27.0 lets the raw
    `websockets.ConnectionClosed` through. Neither is either of those two
    classes, so both escaped the helper, escaped the handler, and came out of
    uvicorn as a bare `transport.close()` — close code 1006 at the peer, which
    is the incident-3 path this round exists to remove."""
    import asyncio

    from app.api._ws_auth_helpers import safe_send_close_ws, safe_send_fault_ws

    class _ClientDisconnected(OSError):
        pass

    class _Dead:
        def __init__(self, exc):
            self._exc = exc
            self.closed = False

        async def send_json(self, _frame):
            raise self._exc()

        async def close(self, code=1000, reason=""):
            self.closed = True
            raise self._exc()

    for exc in (_ClientDisconnected, ConnectionResetError, BrokenPipeError, ValueError):
        sock = _Dead(exc)
        asyncio.run(safe_send_fault_ws(sock, fc.AGENT_DB_UNAVAILABLE))
        assert sock.closed, exc.__name__
        sock2 = _Dead(exc)
        asyncio.run(safe_send_close_ws(sock2, code=4001, message="nope"))
        assert sock2.closed, exc.__name__
