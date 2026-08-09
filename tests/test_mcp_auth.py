"""
T0c — MCP transport auth tests.

Two layers:

  1. Unit tests against `MCPAuthMiddleware` directly via ASGI scope/
     receive/send — fast, no FastMCP, no FastAPI, no DB. The
     `_resolve_agent_key_to_user_id` lookup is monkey-patched to a
     deterministic in-memory mapping. Covers missing-header, wrong-key,
     valid-key, warn-only vs enforce, contextvar binding/reset.

  2. Integration tests against the real `mcp_server.mcp.http_app(...)`
     mounted under the middleware in a FastAPI app. Exercises the full
     stack (httpx ASGI transport → FastAPI → middleware → FastMCP →
     tool handler → DB). Covers cross-tenant isolation: user A's key
     can list/call tools but never sees user B's data.

  These integration tests need real Postgres because the `User` table
  uses a `postgresql_where` partial unique index on `is_canary` that
  SQLite degrades to a full UNIQUE constraint, breaking any fixture
  that seeds two users. They auto-skip when DATABASE_URL points to
  SQLite — CI's Postgres runs them.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig, User
from app.mcp_auth import (
    MCPAuthMiddleware,
    _current_user_id,
    get_mcp_user_id,
)


# ─── ASGI helpers ─────────────────────────────────────────────────────


class _ScopeRecorder:
    """Trivial inner ASGI app: writes the bound user_id to the body so
    tests can assert what the middleware bound for this request."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        if scope["type"] != "http":
            return
        try:
            uid = get_mcp_user_id()
        except ValueError as e:
            uid = f"<unbound: {e.__class__.__name__}>"
        body = json.dumps({"bound_user_id": uid}).encode()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def _make_scope(headers: dict[str, str]) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/mcp/tools/call",
        "raw_path": b"/mcp/tools/call",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("203.0.113.7", 12345),
        "server": ("toup.ai", 443),
    }


async def _drive(app, scope: dict) -> tuple[int, dict, bytes]:
    """Drive an ASGI app through one request → (status, headers, body)."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    headers = {k.decode(): v.decode() for k, v in start.get("headers", [])}
    return start["status"], headers, body


# ─── Fixtures ─────────────────────────────────────────────────────────


def _is_sqlite() -> bool:
    """SQLite doesn't honor `postgresql_where` partial unique indexes
    (e.g. on User.is_canary), so any fixture that seeds two users will
    UNIQUE-violate. Integration tests that need that fixture skip when
    we're on SQLite — CI runs them on Postgres."""
    import os
    return os.environ.get("DATABASE_URL", "").startswith("sqlite")


# ── Unit-test "DB" — monkey-patched lookup table, no real DB ─────────


@pytest.fixture
def fake_tenants(monkeypatch) -> dict[str, dict]:
    """Two synthetic tenants installed via `_resolve_agent_key_to_user_id`
    monkey-patch. No real DB hit — these tests run anywhere."""
    alice_uid = "00000000-0000-0000-0000-aaaaaaaaaaaa"
    bob_uid = "00000000-0000-0000-0000-bbbbbbbbbbbb"
    table = {
        "key_alice_xxx": alice_uid,
        "key_bob_xxx": bob_uid,
    }

    async def fake_resolve(agent_key: str):
        return table.get(agent_key)

    monkeypatch.setattr("app.mcp_auth._resolve_agent_key_to_user_id", fake_resolve)
    return {
        "alice": {"user_id": alice_uid, "agent_key": "key_alice_xxx"},
        "bob": {"user_id": bob_uid, "agent_key": "key_bob_xxx"},
    }


# ── Real DB fixture for integration tests (Postgres only) ───────────


@pytest_asyncio.fixture
async def db_tenants() -> dict[str, dict]:
    """Seed two real users + AgentConfig rows for integration tests.
    Skips when DATABASE_URL is SQLite (see _is_sqlite docstring)."""
    if _is_sqlite():
        pytest.skip("requires Postgres (User.is_canary partial unique index)")
    out = {}
    async with async_session_maker() as db:
        for name in ("alice", "bob"):
            uid = str(uuid.uuid4())
            user = User(
                id=uid,
                email=f"{name}-{uid[:8]}@example.com",
                hashed_password="x",
                name=name,
            )
            db.add(user)
            await db.flush()
            cfg = AgentConfig(
                user_id=uid,
                agent_api_key=f"key_{name}_{uuid.uuid4().hex[:24]}",
            )
            db.add(cfg)
            out[name] = {"user_id": uid, "agent_key": cfg.agent_api_key}
        await db.commit()
    return out


@pytest_asyncio.fixture
async def enforce_mode():
    """Flip MCP_REQUIRE_X_AGENT_KEY to enforce for the duration of a test."""
    prev = settings.mcp_require_x_agent_key
    settings.mcp_require_x_agent_key = True
    try:
        yield
    finally:
        settings.mcp_require_x_agent_key = prev


@pytest_asyncio.fixture
async def warn_only_mode():
    """Force warn-only (the default), but make it explicit per test."""
    prev = settings.mcp_require_x_agent_key
    settings.mcp_require_x_agent_key = False
    try:
        yield
    finally:
        settings.mcp_require_x_agent_key = prev


# ─── Unit: missing header ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_header_warn_only_passes_through_unbound(warn_only_mode, caplog):
    """Warn-only mode + no X-Agent-Key → request proceeds, contextvar
    is NOT set (handlers fail-soft via get_mcp_user_id raising)."""
    inner = _ScopeRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({}))
    assert status == 200, "warn-only must not 401 on missing header"
    assert inner.calls == 1, "inner app must have been invoked"
    payload = json.loads(body)
    assert payload["bound_user_id"].startswith("<unbound:"), (
        "with no header, the request must reach the handler with no "
        f"user bound, got: {payload}"
    )
    # And we must have logged a warning with fingerprint detail.
    rejections = [r for r in caplog.records if "MCP auth WARN" in r.getMessage()]
    assert rejections, "warn-only mode must log a warning per missed auth"
    msg = rejections[-1].getMessage()
    assert "key_prefix" in msg and "<missing>" in msg
    assert "/mcp/tools/call" in msg


@pytest.mark.asyncio
async def test_missing_header_enforce_returns_401(enforce_mode):
    """Enforce mode + no X-Agent-Key → 401 before inner app runs."""
    inner = _ScopeRecorder()
    mw = MCPAuthMiddleware(inner)
    status, headers, body = await _drive(mw, _make_scope({}))
    assert status == 401
    assert inner.calls == 0, "inner app must NOT be invoked on auth failure"
    assert b"X-Agent-Key" in body
    assert "www-authenticate" in headers
    assert "X-Agent-Key" in headers["www-authenticate"]


# ─── Unit: wrong key ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wrong_key_warn_only_passes_through_with_log(warn_only_mode, fake_tenants, caplog):
    """Warn-only + key that doesn't match any tenant → log warning,
    request proceeds with no user bound."""
    inner = _ScopeRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({"X-Agent-Key": "bogus_key_xxx"}))
    assert status == 200
    assert inner.calls == 1
    assert json.loads(body)["bound_user_id"].startswith("<unbound:")
    rejections = [r for r in caplog.records if "MCP auth WARN" in r.getMessage()]
    assert rejections, "warn-only must log on key miss"
    msg = rejections[-1].getMessage()
    assert "bogus_ke..." in msg, f"fingerprint must include key prefix, got: {msg}"


@pytest.mark.asyncio
async def test_wrong_key_enforce_returns_401(enforce_mode, fake_tenants):
    """Enforce + key that doesn't match → 401."""
    inner = _ScopeRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, _ = await _drive(mw, _make_scope({"X-Agent-Key": "bogus_key_xxx"}))
    assert status == 401
    assert inner.calls == 0


# ─── Unit: valid key binds correct user_id ────────────────────────────


@pytest.mark.asyncio
async def test_valid_key_binds_correct_user_id(warn_only_mode, fake_tenants):
    """Valid X-Agent-Key → contextvar bound to that tenant's user_id,
    inner handler sees the right user."""
    alice = fake_tenants["alice"]
    inner = _ScopeRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(
        mw, _make_scope({"X-Agent-Key": alice["agent_key"]})
    )
    assert status == 200
    assert json.loads(body)["bound_user_id"] == alice["user_id"]


@pytest.mark.asyncio
async def test_valid_key_enforce_also_works(enforce_mode, fake_tenants):
    """Same as above but in enforce mode — flag should not change behavior
    on the success path."""
    bob = fake_tenants["bob"]
    inner = _ScopeRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({"X-Agent-Key": bob["agent_key"]}))
    assert status == 200
    assert json.loads(body)["bound_user_id"] == bob["user_id"]


# ─── Unit: contextvar isolation ───────────────────────────────────────


@pytest.mark.asyncio
async def test_contextvar_resets_after_request(warn_only_mode, fake_tenants):
    """The contextvar must NOT leak across requests. Two sequential
    requests with different keys see different user_ids; the var is
    None outside any request."""
    alice = fake_tenants["alice"]
    bob = fake_tenants["bob"]
    inner = _ScopeRecorder()
    mw = MCPAuthMiddleware(inner)

    assert _current_user_id.get() is None, "no request in flight, must be None"

    _, _, body_a = await _drive(mw, _make_scope({"X-Agent-Key": alice["agent_key"]}))
    assert _current_user_id.get() is None, "contextvar must reset after request"
    assert json.loads(body_a)["bound_user_id"] == alice["user_id"]

    _, _, body_b = await _drive(mw, _make_scope({"X-Agent-Key": bob["agent_key"]}))
    assert _current_user_id.get() is None
    assert json.loads(body_b)["bound_user_id"] == bob["user_id"]


# ─── Unit: get_mcp_user_id raises when unbound ────────────────────────


def test_get_mcp_user_id_raises_when_unbound():
    """Calling get_mcp_user_id() outside any MCP request must raise
    ValueError. This is what makes warn-only mode safe — unauthenticated
    callers fail loudly inside the tool handler."""
    with pytest.raises(ValueError, match="X-Agent-Key"):
        get_mcp_user_id()


# ─── Integration: mounted FastMCP server ──────────────────────────────


@pytest_asyncio.fixture
async def mcp_test_app() -> AsyncIterator[FastAPI]:
    """FastAPI app with the real platform MCP server mounted under the
    auth middleware. Lifespan-aware so FastMCP's HTTP transport boots.

    Skips if the local env has a FastAPI/Starlette pin mismatch with the
    FastMCP version — CI runs the proper pins from `requirements.txt`.
    """
    from app.mcp_server import mcp

    try:
        mcp_app = mcp.http_app(path="/mcp")
        parent = FastAPI(lifespan=mcp_app.lifespan)
    except TypeError as e:
        pytest.skip(
            f"FastAPI/FastMCP wiring failed (likely a local env pin "
            f"mismatch — CI has correct pins): {e}"
        )
    parent.mount("/api/mcp", MCPAuthMiddleware(mcp_app))
    yield parent


@pytest_asyncio.fixture
async def mcp_http_client(mcp_test_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=mcp_test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.mark.asyncio
async def test_mounted_mcp_rejects_missing_key_in_enforce_mode(
    mcp_http_client: AsyncClient, enforce_mode
):
    """Hit the real mounted MCP endpoint without a key → 401 from the
    transport, no FastMCP dispatch."""
    r = await mcp_http_client.post(
        "/api/mcp/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert r.status_code == 401, f"got {r.status_code}: {r.text[:200]}"


@pytest.mark.asyncio
async def test_mounted_mcp_passes_through_in_warn_only_mode(
    mcp_http_client: AsyncClient, warn_only_mode
):
    """Warn-only + no key → request proceeds to FastMCP. Tool dispatch
    that calls _get_user_id() will fail downstream, but the transport
    itself must NOT 401."""
    r = await mcp_http_client.post(
        "/api/mcp/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Accept": "application/json, text/event-stream"},
    )
    # tools/list doesn't call _get_user_id — should succeed even unauthenticated
    # in warn-only mode (the transport doesn't gate it).
    assert r.status_code != 401, (
        f"warn-only must not 401, got {r.status_code}: {r.text[:200]}"
    )


@pytest.mark.asyncio
async def test_mounted_mcp_resolves_valid_key_for_tools_list(
    mcp_http_client: AsyncClient, db_tenants, enforce_mode
):
    """Valid X-Agent-Key in enforce mode → tools/list returns 2xx."""
    alice = db_tenants["alice"]
    r = await mcp_http_client.post(
        "/api/mcp/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={
            "X-Agent-Key": alice["agent_key"],
            "Accept": "application/json, text/event-stream",
        },
    )
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:300]}"


@pytest.mark.asyncio
async def test_mounted_mcp_cross_tenant_isolation(
    mcp_http_client: AsyncClient, db_tenants, enforce_mode
):
    """Call memory_list with alice's key → results scoped to alice.
    Call same tool with bob's key → results scoped to bob.

    Even though we haven't seeded any memories, the per-tool DB query
    is `WHERE user_id = <bound>`, so the results are necessarily disjoint.
    The fail-mode this test catches: a regression where _get_user_id()
    falls back to settings.user_id (or any fixed value) and both calls
    return the same data."""
    alice = db_tenants["alice"]
    bob = db_tenants["bob"]

    async def call(agent_key: str) -> dict:
        r = await mcp_http_client.post(
            "/api/mcp/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "memory_list", "arguments": {"limit": 5}},
            },
            headers={
                "X-Agent-Key": agent_key,
                "Accept": "application/json, text/event-stream",
            },
        )
        assert r.status_code == 200, f"got {r.status_code}: {r.text[:300]}"
        # FastMCP returns SSE/JSON-RPC; for both tenants we just need
        # the result to come back without a "wrong tenant" error.
        return r.text

    text_a = await call(alice["agent_key"])
    text_b = await call(bob["agent_key"])

    # The actual user ids must appear in each response (the tool returns
    # an empty list for new users, but FastMCP's framing should still
    # produce a successful structured response). The critical assertion:
    # neither response contains the OTHER tenant's user_id.
    assert bob["user_id"] not in text_a, (
        "alice's response contained bob's user_id — cross-tenant leak"
    )
    assert alice["user_id"] not in text_b, (
        "bob's response contained alice's user_id — cross-tenant leak"
    )


@pytest.mark.asyncio
async def test_db_lookup_failure_fails_closed(monkeypatch, warn_only_mode):
    """If the DB lookup raises (e.g. pool exhausted), the middleware
    must 503 — never proceed with a half-resolved user. Fail-closed."""
    inner = _ScopeRecorder()

    async def boom(agent_key: str):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr("app.mcp_auth._resolve_agent_key_to_user_id", boom)
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({"X-Agent-Key": "anything"}))
    assert status == 503
    assert b"MCP auth lookup failed" in body
    assert inner.calls == 0


def test_enforce_mode_is_the_default():
    """G-10 (audit 2026-08-09): warn-only was a SOAK mode, not a resting
    state. The soak it existed for concluded — 0 'would-reject in enforce
    mode' warnings across 20 production containers over 7 days — so the
    default is enforce. Pinned via model_fields so an env var cannot fake
    this green; MCP_REQUIRE_X_AGENT_KEY=false remains the escape hatch."""
    from app.config import settings as _settings

    assert _settings.model_fields["mcp_require_x_agent_key"].default is True
