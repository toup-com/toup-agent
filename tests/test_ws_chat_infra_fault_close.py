"""An infrastructure fault must be ANSWERED, never escape as a bare close.

Why this file exists
--------------------
2026-09-15, incident 3. PgBouncer died on the VPS at 15:08:13Z. Every DB access
inside container `toup-agent-pool-38` raised `ConnectionRefusedError`. The
agent's `ws_chat` handler accepted the socket, ran its auth ladder — and the
ladder's first DB access sat ABOVE the handler's own `try:`. The exception
escaped into uvicorn, which logged "Exception in ASGI application" and closed
the transport with no close frame:

    15:08:47.854  WebSocket /api/ws/chat [accepted]
    15:08:47.866  connection open
    15:08:47.887  Exception in ASGI application
    15:08:47.890  connection closed

36 milliseconds, ten times over three minutes. The peer saw 1006, the platform
proxy laundered that to 1000, and the phone told the user to check her
internet.

The other half of the same defect is subtler and worse: `_authenticate_ws`
caught everything and returned `None`, so on the JWT path the SAME dead
database was reported as close 4001 "Authentication required" — which the
proxy maps to 4503 "agent starting", which the app answers by silently
re-sending, up to fifteen times, into an outage that lasted eleven minutes.
"I could not check" is not "you are not who you say you are".

These tests drive the REAL handler over an in-process ASGI socket with a
session maker that refuses connections, and assert what the client receives.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. python -m pytest -q \
        tests/test_ws_chat_infra_fault_close.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI

from app.api import _fault_codes as fc
from app.api import ws_chat
from app.config import settings


USER_ID = "00000000-0000-4000-8000-000000000001"


# ── A session maker that is down ──────────────────────────────────────
class _DeadSessionMaker:
    """`async with async_session_maker() as db:` where the pooler is dead.

    The real failure is raised by asyncpg while CONNECTING, i.e. inside the
    context manager's `__aenter__`, not at call time — which is why the
    exception lands in the middle of the auth ladder rather than at its top."""

    def __init__(self, exc_factory=None) -> None:
        self._exc = exc_factory or (lambda: ConnectionRefusedError(111, "Connection refused"))

    def __call__(self):
        return self

    async def __aenter__(self):
        raise self._exc()

    async def __aexit__(self, *exc):
        return False


# ── Minimal in-process ASGI websocket ─────────────────────────────────
class AsgiWs:
    """One client socket, spoken straight into the ASGI app.

    Deliberately not TestClient: what is under test is what arrives on the
    wire when the handler misbehaves, including the case where it raises —
    and TestClient hides an ASGI exception behind its portal thread."""

    def __init__(self, app: Any, headers: Optional[List] = None, query: str = "") -> None:
        self._app = app
        self._headers = headers or [(b"host", b"testserver")]
        self._query = query
        self._to_app: asyncio.Queue = asyncio.Queue()
        self._from_app: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None

    async def open(self) -> Dict[str, Any]:
        scope = {
            "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "scheme": "ws", "path": "/api/ws/chat",
            "raw_path": b"/api/ws/chat", "query_string": self._query.encode(),
            "root_path": "", "headers": self._headers,
            "client": ("127.0.0.1", 51234), "server": ("testserver", 80),
            "subprotocols": [], "state": {},
        }
        self._to_app.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.create_task(self._app(scope, self._to_app.get, self._from_app.put))
        return await asyncio.wait_for(self._from_app.get(), timeout=5.0)

    def send_text(self, text: str) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": text})

    async def drain(self, timeout: float = 5.0) -> "Transcript":
        """Collect frames until the handler closes (or stops speaking)."""
        t = Transcript()
        try:
            while True:
                msg = await asyncio.wait_for(self._from_app.get(), timeout=timeout)
                if msg["type"] == "websocket.close":
                    t.close_code = msg.get("code", 1000)
                    break
                if msg["type"] == "websocket.send":
                    try:
                        t.frames.append(json.loads(msg.get("text") or "{}"))
                    except (ValueError, TypeError):
                        pass
        except asyncio.TimeoutError:
            t.timed_out = True
        # The load-bearing assertion of this whole file: the handler must have
        # RETURNED, not raised. An exception here is uvicorn's bare transport
        # close in the making.
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=5.0)
                t.raised = self.task.exception()
            except asyncio.TimeoutError:
                t.raised = None
            except Exception as exc:  # noqa: BLE001 — recorded, then asserted on
                t.raised = exc
        return t

    async def dispose(self) -> None:
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": 1000})
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: B014
                pass


class Transcript:
    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []
        self.close_code: Optional[int] = None
        self.raised: Optional[BaseException] = None
        self.timed_out = False

    @property
    def codes(self) -> List[str]:
        return [f.get("code") for f in self.frames if f.get("type") == "error"]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Transcript(close={self.close_code}, frames={self.frames}, raised={self.raised!r})"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(ws_chat.router, prefix="/api")
    return app


def _run(coro):
    return asyncio.run(coro)


async def _agent_key_connect(mp, *, db_down: bool = True, runner: bool = True) -> Transcript:
    """The platform-proxy auth path: X-Agent-Key, stub user lazily created."""
    import app.db.database as db_mod

    if db_down:
        mp.setattr(db_mod, "async_session_maker", _DeadSessionMaker())
    mp.setattr(settings, "agent_api_key", "test-agent-key")
    mp.setattr(settings, "user_id", USER_ID)
    mp.setattr(ws_chat, "_agent_runner", object() if runner else None, raising=False)

    ws = AsgiWs(_app(), headers=[(b"host", b"testserver"), (b"x-agent-key", b"test-agent-key")])
    accepted = await ws.open()
    assert accepted["type"] == "websocket.accept", accepted
    try:
        return await ws.drain()
    finally:
        await ws.dispose()


def run_case(**kwargs) -> Transcript:
    async def wrapper():
        mp = pytest.MonkeyPatch()
        try:
            return await _agent_key_connect(mp, **kwargs)
        finally:
            mp.undo()
    return _run(wrapper())


# ══════════════════════════════════════════════════════════════════════
# 1. The classifier
# ══════════════════════════════════════════════════════════════════════

def test_authenticate_ws_raises_infra_unavailable_instead_of_returning_none():
    """The 4001 half of the defect. `return None` here becomes "Authentication
    required" three lines later, and 4001 becomes 4503 at the proxy."""
    async def go():
        mp = pytest.MonkeyPatch()
        try:
            import app.db.database as db_mod
            import app.services as services

            mp.setattr(db_mod, "async_session_maker", _DeadSessionMaker())
            mp.setattr(services, "decode_access_token", lambda t: USER_ID, raising=False)
            with pytest.raises(fc.WsInfraUnavailable):
                await ws_chat._authenticate_ws("any-token")
        finally:
            mp.undo()
    _run(go())


def test_authenticate_ws_still_returns_none_for_a_genuinely_bad_token():
    """The narrowing must not swallow real rejections: a bad token is still a
    rejection, and it must still be 4001."""
    async def go():
        assert await ws_chat._authenticate_ws("not-a-jwt") is None
    _run(go())


def test_ensure_stub_user_reports_infra_down_rather_than_raising_blind():
    async def go():
        mp = pytest.MonkeyPatch()
        try:
            import app.db.database as db_mod

            mp.setattr(db_mod, "async_session_maker", _DeadSessionMaker())
            assert await ws_chat._ensure_stub_user(USER_ID) == "infra_down"
        finally:
            mp.undo()
    _run(go())


def test_ensure_stub_user_is_one_function_not_two_copies():
    """It replaced two byte-identical inline blocks (the agent-key path and
    the session-token path). Only one of them was ever inside the handler's
    try, which is the whole reason the incident escaped."""
    import inspect

    src = inspect.getsource(ws_chat)
    assert src.count("email=f\"{user_id[:8]}@agent.local\"") == 1, (
        "the stub-user create has been inlined again"
    )


# ══════════════════════════════════════════════════════════════════════
# 2. End to end: what the client receives when the database is down
# ══════════════════════════════════════════════════════════════════════

def test_db_down_closes_4505_with_a_coded_frame():
    t = run_case(db_down=True)
    assert t.raised is None, f"the handler raised instead of answering: {t.raised!r}"
    assert t.close_code == fc.FAULTS[fc.AGENT_DB_UNAVAILABLE]["close_code"] == 4505, t
    assert fc.AGENT_DB_UNAVAILABLE in t.codes, t
    frame = [f for f in t.frames if f.get("type") == "error"][0]
    assert frame["retryable"] is True
    assert isinstance(frame.get("retry_after_ms"), int)


def test_db_down_is_never_reported_as_an_authentication_failure():
    """4001 → (proxy) 4503 → the app re-sends silently. Eleven minutes of it."""
    t = run_case(db_down=True)
    assert t.close_code != 4001, t
    assert "Authentication required" not in json.dumps(t.frames), t


def test_db_down_never_produces_a_bare_transport_close():
    """The shape of the incident: no frame, no close code, just a dead socket.
    `timed_out` would mean the handler neither answered nor closed."""
    t = run_case(db_down=True)
    assert t.timed_out is False, t
    assert t.close_code is not None, t
    assert t.frames, "the client got no frame at all"


def test_the_fault_frame_carries_no_user_content_and_no_raw_ids():
    """Round 46 privacy rule: reason codes, counts and hashes only. `detail`
    is where an exception string would otherwise be smuggled onto the wire —
    the old outer handler sent `str(e)`, and a SQLAlchemy error carries the
    statement and its bound parameters, i.e. the user's own message."""
    t = run_case(db_down=True)
    blob = json.dumps(t.frames)
    assert USER_ID not in blob, blob
    assert "Connection refused" not in blob, blob
    assert "asyncpg" not in blob, blob


def test_an_unhandled_error_after_auth_closes_4507_not_4500():
    """The safety net. Anything that escapes the handler body is an internal
    error with a code, never uvicorn's bare close."""
    async def go():
        mp = pytest.MonkeyPatch()
        try:
            import app.db.database as db_mod
            import app.services.drain_state as drain_state

            # DB is healthy; the crash is elsewhere.
            mp.setattr(db_mod, "async_session_maker", _DeadSessionMaker(
                exc_factory=lambda: RuntimeError("not an infra error")), raising=False)
            mp.setattr(settings, "agent_api_key", "test-agent-key")
            mp.setattr(settings, "user_id", USER_ID)
            mp.setattr(ws_chat, "_agent_runner", object(), raising=False)

            def boom() -> None:
                raise ValueError("something structural broke after auth")

            mp.setattr(drain_state, "increment_active", boom)

            ws = AsgiWs(_app(), headers=[(b"host", b"testserver"),
                                         (b"x-agent-key", b"test-agent-key")])
            await ws.open()
            try:
                t = await ws.drain()
            finally:
                await ws.dispose()
            assert t.raised is None, f"handler raised: {t.raised!r}"
            assert t.close_code == fc.FAULTS[fc.AGENT_INTERNAL]["close_code"] == 4507, t
            assert fc.AGENT_INTERNAL in t.codes, t
            # The exception's own text must not be on the wire.
            assert "something structural broke" not in json.dumps(t.frames)
        finally:
            mp.undo()
    _run(go())


def test_a_healthy_connect_is_untouched():
    """The control. Every assertion above is worthless if the ordinary path
    changed: a healthy agent-key connect must authenticate and stay open."""
    async def go():
        mp = pytest.MonkeyPatch()
        try:
            import app.db.database as db_mod

            class _OkSession:
                def __call__(self):
                    return self

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *exc):
                    return False

            async def _get_user(db, uid):
                return object()   # the row already exists ⇒ no INSERT path

            import app.services.auth_service as auth_service

            mp.setattr(db_mod, "async_session_maker", _OkSession())
            mp.setattr(auth_service, "get_user_by_id", _get_user)
            mp.setattr(settings, "agent_api_key", "test-agent-key")
            mp.setattr(settings, "user_id", USER_ID)
            mp.setattr(ws_chat, "_agent_runner", object(), raising=False)

            ws = AsgiWs(_app(), headers=[(b"host", b"testserver"),
                                         (b"x-agent-key", b"test-agent-key")])
            accepted = await ws.open()
            assert accepted["type"] == "websocket.accept"
            # Nothing should close the socket: give it a moment, then leave.
            await asyncio.sleep(0.05)
            assert ws.task is not None and not ws.task.done(), "healthy connect was closed"
            await ws.dispose()
        finally:
            mp.undo()
    _run(go())


# ══════════════════════════════════════════════════════════════════════
# 3. The privacy half of the same handler
# ══════════════════════════════════════════════════════════════════════

def test_fast_media_log_no_longer_carries_the_users_message():
    """`[FAST-MEDIA] Detected play request: %r → query=%r` wrote the user's
    whole chat message to a log shipped to Loki, on the path that fires for
    every ordinary "play …"."""
    import inspect

    src = inspect.getsource(ws_chat)
    assert "Detected play request: text_len=%d query_len=%d" in src
    assert "Detected play request: %r" not in src
    assert 'No video found for: %s' not in src


# ══════════════════════════════════════════════════════════════════════
# 4. resume → turn_receipt, on the wire (C2)
# ══════════════════════════════════════════════════════════════════════
#
# `_turn_receipt` itself is well covered by test_ws_turn_identity.py, but the
# DISPATCH branch that receives the frame and sends the receipt back was
# checked only by a substring grep over a 1400-character window — a branch
# that computed the receipt and then `continue`d without sending it passed
# every assertion. This is the fix for incident 2's un-endable turn, so it is
# asserted as JSON arriving on a socket.


async def _next_frame(ws: AsgiWs, timeout: float = 3.0) -> Dict[str, Any]:
    while True:
        msg = await asyncio.wait_for(ws._from_app.get(), timeout=timeout)
        if msg["type"] == "websocket.send":
            return json.loads(msg.get("text") or "{}")
        if msg["type"] == "websocket.close":
            raise AssertionError(f"the socket closed instead of answering: {msg}")


class _OkSession:
    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def _authenticated_socket(mp) -> AsgiWs:
    import app.db.database as db_mod
    import app.services.auth_service as auth_service

    async def _get_user(db, uid):
        return object()

    mp.setattr(db_mod, "async_session_maker", _OkSession())
    mp.setattr(auth_service, "get_user_by_id", _get_user)
    mp.setattr(settings, "agent_api_key", "test-agent-key")
    mp.setattr(settings, "user_id", USER_ID)
    mp.setattr(ws_chat, "_agent_runner", object(), raising=False)
    ws = AsgiWs(_app(), headers=[(b"host", b"testserver"),
                                 (b"x-agent-key", b"test-agent-key")])
    accepted = await ws.open()
    assert accepted["type"] == "websocket.accept", accepted
    return ws


def test_a_resume_for_a_running_turn_gets_its_receipt_on_the_socket():
    async def go():
        mp = pytest.MonkeyPatch()
        ws_sock = None
        try:
            ws_chat._active_turns.pop(USER_ID, None)
            ws_sock = await _authenticated_socket(mp)
            ws_chat._set_active_turn(
                USER_ID, mission_id="chatturn:live", stage="thinking",
                started_at=__import__("time").time(), client_msg_id="cm-live",
            )
            ws_sock.send_text(json.dumps({"type": "resume", "client_msg_id": "cm-live"}))
            frame = await _next_frame(ws_sock)
            assert frame["type"] == "turn_receipt", frame
            assert frame["client_msg_id"] == "cm-live"
            assert frame["status"] == "running"
            assert frame["mission_id"] == "chatturn:live"
        finally:
            ws_chat._active_turns.pop(USER_ID, None)
            if ws_sock is not None:
                await ws_sock.dispose()
            mp.undo()
    _run(go())


def test_a_resume_with_no_id_is_answered_and_the_socket_stays_open():
    """Silence here is the incident: the client waits on a frame that never
    comes and falls back to a wall clock."""
    async def go():
        mp = pytest.MonkeyPatch()
        ws_sock = None
        try:
            ws_chat._active_turns.pop(USER_ID, None)
            ws_sock = await _authenticated_socket(mp)
            ws_sock.send_text(json.dumps({"type": "resume"}))
            frame = await _next_frame(ws_sock)
            assert frame["type"] == "turn_receipt", frame
            assert frame["status"] == "unknown"
            assert frame["code"] == "no_client_msg_id"
            # Still alive: a resume is a question, never a protocol error.
            ws_sock.send_text(json.dumps({"type": "ping"}))
            assert (await _next_frame(ws_sock))["type"] == "pong"
        finally:
            if ws_sock is not None:
                await ws_sock.dispose()
            mp.undo()
    _run(go())


def test_an_over_long_client_msg_id_is_treated_as_absent_not_as_a_query():
    """`messages.client_msg_id` is VARCHAR(100) and this value is fully
    client-controlled. Bounded at the branch, the same way the `message`
    branch bounds it — never surfaced as a DataError or a giant SELECT."""
    async def go():
        mp = pytest.MonkeyPatch()
        ws_sock = None
        try:
            ws_chat._active_turns.pop(USER_ID, None)
            ws_sock = await _authenticated_socket(mp)
            ws_sock.send_text(json.dumps({"type": "resume", "client_msg_id": "x" * 400}))
            frame = await _next_frame(ws_sock)
            assert frame["type"] == "turn_receipt"
            assert frame["status"] == "unknown"
            assert frame["code"] == "no_client_msg_id"
            assert frame["client_msg_id"] is None
        finally:
            if ws_sock is not None:
                await ws_sock.dispose()
            mp.undo()
    _run(go())
