"""The request-timing middleware must never hard-cancel the request handler.

2026-07-25 outage: the Supavisor transaction pooler on :6543 stopped handing
out server connections while Postgres itself was perfectly healthy. Cause: 16
sessions stuck `idle in transaction` whose query text was literally `BEGIN;`,
the oldest 27.9 hours old, each pinning a pooler server slot forever.

`BEGIN;` with no isolation suffix is emitted by exactly one line in the whole
dependency tree — SQLAlchemy's `_async_ping` (dialects/postgresql/asyncpg.py),
which `pool_pre_ping=True` runs on nearly every checkout:

    tr = self._connection.transaction()   # no isolation arg -> bare "BEGIN;"
    await tr.start()                      # <- NOT inside the try below
    try:
        await self._connection.fetchrow(";")
    finally:
        await tr.rollback()

A cancellation delivered during `tr.start()` therefore skips the ROLLBACK
entirely: the BEGIN has already reached Postgres, nothing ever ends the
transaction, and the pooler pins that server connection. `ping()` only catches
`Exception`, so `CancelledError` (a `BaseException`) escapes.

The cancellations came from `@app.middleware("http")` — i.e.
`BaseHTTPMiddleware`, which runs the entire downstream app as a child of an
anyio task group and cancels it when the response cannot be delivered (client
vanished mid-request). Rewriting it as pure ASGI removes the cancel scope, so
a disconnect no longer cancels a handler that is mid-DB-round-trip.

These tests pin that: the shipped middleware must not cancel its handler, and
`BaseHTTPMiddleware` demonstrably does (so nobody "simplifies" it back).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from starlette.middleware.base import BaseHTTPMiddleware

from platform_main import _RequestTimingMiddleware

_SRC = (Path(__file__).resolve().parent.parent / "platform_main.py").read_text()


def _scope(path="/api/x", method="GET"):
    return {"type": "http", "path": path, "method": method, "headers": []}


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _make_handler(events, *, fail_send_at=None, work_s=0.15):
    """An app that streams a response and then keeps working (standing in for
    the DB round-trip / cleanup that SQLAlchemy's pre-ping lives inside)."""

    async def handler(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        try:
            await send({"type": "http.response.body", "body": b"x"})
        except Exception:
            events.append("send_raised")
        try:
            await asyncio.sleep(work_s)
            events.append("work_completed")
        except asyncio.CancelledError:
            events.append("work_cancelled")
            raise

    return handler


def _send_that_dies(fail_on="http.response.body"):
    """A client that vanishes: the transport refuses the write."""

    async def send(message):
        if message["type"] == fail_on:
            raise OSError("[Errno 32] Broken pipe")

    return send


# ── The invariant ───────────────────────────────────────────────────────

def test_client_disconnect_does_not_cancel_the_handler():
    """GATE: with the shipped pure-ASGI middleware, a dead client must not
    cancel work already in flight. This is what keeps a cancellation out of
    SQLAlchemy's pre-ping, where it would strand a `BEGIN;` forever."""
    events: list[str] = []
    app = _RequestTimingMiddleware(_make_handler(events))

    asyncio.run(app(_scope(), _receive, _send_that_dies()))

    assert "send_raised" in events, "the broken pipe should surface to the handler"
    assert "work_completed" in events, "handler must run to completion"
    assert "work_cancelled" not in events, (
        "handler was cancelled — this is the outage mechanism: a cancel landing "
        "inside SQLAlchemy's pre-ping skips its ROLLBACK and pins a pooler "
        "connection `idle in transaction` permanently"
    )


def test_basehttpmiddleware_does_cancel_the_handler():
    """The regression this replaced is real, not theoretical: the same handler
    under `@app.middleware("http")` IS cancelled. If Starlette ever stops doing
    this, this test fails and the comment in platform_main can be revisited."""
    events: list[str] = []

    async def _passthrough(request, call_next):
        return await call_next(request)

    app = BaseHTTPMiddleware(_make_handler(events), dispatch=_passthrough)

    with pytest.raises(BaseException):
        asyncio.run(app(_scope(), _receive, _send_that_dies()))

    assert "work_cancelled" in events, (
        "expected BaseHTTPMiddleware's anyio task group to cancel the handler"
    )
    assert "work_completed" not in events


def test_platform_main_does_not_reintroduce_basehttpmiddleware():
    """`@app.middleware("http")` is BaseHTTPMiddleware. Adding one back
    reintroduces the cancel scope for EVERY request on the service."""
    # Match only real usage, not the prose in the class docstring explaining
    # why we don't use it.
    used = [ln for ln in _SRC.splitlines() if ln.lstrip().startswith("@app.middleware(")]
    assert not used, f"BaseHTTPMiddleware reintroduced: {used}"
    assert "app.add_middleware(_RequestTimingMiddleware)" in _SRC


def test_no_basehttpmiddleware_anywhere_in_the_live_stack():
    """Assert on the REAL app object, not the source text.

    The grep above only catches the `@app.middleware("http")` decorator. The
    likelier regression is `app.add_middleware(SomethingMiddleware)` where that
    class happens to subclass BaseHTTPMiddleware — a third-party auth/tracing
    middleware, say — which reinstates the anyio cancel scope for every request
    while the text guard stays green. Walk the live stack instead."""
    import platform_main as pm

    offenders = [
        m.cls.__name__ for m in pm.app.user_middleware
        if isinstance(m.cls, type) and issubclass(m.cls, BaseHTTPMiddleware)
    ]
    assert not offenders, (
        f"BaseHTTPMiddleware subclass in the live stack: {offenders}. It runs the "
        "downstream app as an anyio task-group child and hard-cancels it on "
        "client disconnect, which strands SQLAlchemy's pre-ping BEGIN;"
    )


def test_timing_middleware_is_registered_and_outermost():
    """Ordering matters: it must still wrap CORS the way the decorator did."""
    import platform_main as pm

    names = [m.cls.__name__ for m in pm.app.user_middleware]
    assert "_RequestTimingMiddleware" in names, names
    assert names.index("_RequestTimingMiddleware") < names.index("CORSMiddleware"), names


# ── Behaviour preserved from the decorator version ──────────────────────

def test_slow_request_is_logged_with_status_and_duration(caplog):
    events: list[str] = []
    app = _RequestTimingMiddleware(_make_handler(events, work_s=0))

    async def send(message):
        return None

    async def slow(scope, receive, send):
        await asyncio.sleep(0.6)
        await send({"type": "http.response.start", "status": 503, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = _RequestTimingMiddleware(slow)
    with caplog.at_level(logging.WARNING, logger="req_timing"):
        asyncio.run(app(_scope(path="/api/slow", method="POST"), _receive, send))

    msgs = [r.getMessage() for r in caplog.records]
    assert any("[req-timing] POST /api/slow -> 503" in m for m in msgs), msgs


def test_fast_request_is_not_logged(caplog):
    async def send(message):
        return None

    async def fast(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = _RequestTimingMiddleware(fast)
    with caplog.at_level(logging.WARNING, logger="req_timing"):
        asyncio.run(app(_scope(path="/api/fast"), _receive, send))

    assert not [r for r in caplog.records if "[req-timing]" in r.getMessage()]


def test_auth_paths_are_always_logged_even_when_fast(caplog):
    """Signup-latency diagnostics: /api/auth/* logs regardless of duration."""
    async def send(message):
        return None

    async def fast(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = _RequestTimingMiddleware(fast)
    with caplog.at_level(logging.WARNING, logger="req_timing"):
        asyncio.run(app(_scope(path="/api/auth/login", method="POST"), _receive, send))

    assert any("[req-timing] POST /api/auth/login -> 200" in r.getMessage()
               for r in caplog.records)


def test_streaming_route_with_fast_headers_is_not_logged(caplog):
    """GATE against a regression the pure-ASGI rewrite briefly introduced.

    BaseHTTPMiddleware's `call_next` returned at http.response.start, so this
    diagnostic has always meant TIME TO HEADERS. Timing the whole ASGI call
    instead makes every SSE endpoint log a WARNING on every request — and
    /api/mcp/mcp alone is ~94% of this service's traffic. Headers here are
    instant; only the body is slow, so nothing should be logged."""
    async def send(message):
        return None

    async def sse(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for _ in range(3):
            await asyncio.sleep(0.25)   # a long-lived stream
            await send({"type": "http.response.body", "body": b"x", "more_body": True})
        await send({"type": "http.response.body", "body": b""})

    app = _RequestTimingMiddleware(sse)
    with caplog.at_level(logging.WARNING, logger="req_timing"):
        asyncio.run(app(_scope(path="/api/mcp/mcp", method="POST"), _receive, send))

    logged = [r.getMessage() for r in caplog.records if "[req-timing]" in r.getMessage()]
    assert not logged, f"streamed body must not be timed as latency: {logged}"


def test_non_http_scopes_pass_through_untouched():
    """WebSockets (chat, realtime voice, agent tunnel) must not be wrapped."""
    seen = {}

    async def ws_app(scope, receive, send):
        seen["type"] = scope["type"]

    app = _RequestTimingMiddleware(ws_app)
    asyncio.run(app({"type": "websocket", "path": "/api/ws/chat"}, _receive, None))
    assert seen["type"] == "websocket"
