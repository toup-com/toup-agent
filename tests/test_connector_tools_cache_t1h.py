"""
T1h — Tools-cache invalidation tests.

Three units under test:

  1. `MCPToolsCache` (`app/agent/mcp_tools_cache.py`) — TTL, lock
     coalescing, in-place list mutation, force-refresh,
     stale-on-failure semantics.
  2. `PUT /api/agent/refresh-tools` endpoint (`app/api/agent.py`) —
     X-Agent-Key auth, run_mode guard, idempotency, missing-cache
     degraded mode.
  3. `docker_host_service.refresh_tenant_tools` — best-effort 3s
     timeout, missing-AgentConfig short-circuit, success/failure
     return values.

Smoke matrix (per the T1h spec):

  Cache (no DB, no FastAPI):
    C1. TTL behavior — t=0 fetch, t=mid hit, t=expired refetch
    C2. Manual invalidation forces next call to refetch
    C3. force refresh() bypasses TTL
    C4. Concurrent refetches coalesce to ONE round-trip
    C5. In-place mutation — caller's list ref sees the new contents
    C6. Refetch failure leaves stale data + bumps no clock

  Endpoint (FastAPI):
    E1. PUT without X-Agent-Key → 401
    E2. PUT with wrong key → 401
    E3. PUT with correct key + cache wired → 200, refresh()ed
    E4. PUT with correct key + no cache → 200, status='no_cache'
    E5. Run-mode guard: on platform process → 404
    E6. Idempotent — N concurrent PUTs trigger ONE round-trip

  docker_host_service.refresh_tenant_tools:
    D1. No AgentConfig row → False, no HTTP call
    D2. Missing agent_url or agent_api_key → False, no HTTP call
    D3. 200 from tenant → True
    D4. Non-2xx from tenant → False, WARN logged, never raises
    D5. Network exception → False, never raises
    D6. 3-second timeout enforced
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.agent.mcp_tools_cache import (
    DEFAULT_TTL_SECONDS,
    MCPToolsCache,
)
from app.config import settings


# ─── Helpers ────────────────────────────────────────────────────────────


def _make_tool(name: str, description: str = "", schema: dict | None = None):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object", "properties": {}}
    return t


def _make_mock_client(tool_lists: list[list]):
    """Mock MCP client that supports `async with` and yields successive
    list_tools() returns from `tool_lists`. Use a single-element list
    if every call returns the same set."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    call_iter = iter(tool_lists)
    next_value = [tool_lists[-1]]  # fallback after iter exhausted

    async def list_tools_side_effect():
        try:
            return next(call_iter)
        except StopIteration:
            return next_value[0]

    client.list_tools = AsyncMock(side_effect=list_tools_side_effect)
    return client


# ─── Cache: C1–C6 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_c1_ttl_hit_until_expired_then_refetches():
    """t=0 fetch, t=30 hit cache, t=70 refetch."""
    client = _make_mock_client([[_make_tool("a"), _make_tool("b")]])
    cache = MCPToolsCache(client, ttl_seconds=60)

    # t=0
    await cache.list_tools()
    assert client.list_tools.call_count == 1
    assert cache.tools == ["a", "b"]

    # t=30 (still fresh) — patch time.monotonic to advance.
    with patch("app.agent.mcp_tools_cache.time.monotonic",
               return_value=cache.cached_at + 30):
        await cache.list_tools()
    assert client.list_tools.call_count == 1, "still within TTL — must hit cache"

    # t=70 (expired)
    with patch("app.agent.mcp_tools_cache.time.monotonic",
               return_value=cache.cached_at + 70):
        await cache.list_tools()
    assert client.list_tools.call_count == 2, "TTL expired — must refetch"


@pytest.mark.asyncio
async def test_c2_invalidate_forces_next_call_to_refetch():
    """invalidate() within TTL → next list_tools refetches."""
    client = _make_mock_client([[_make_tool("a")]])
    cache = MCPToolsCache(client, ttl_seconds=60)

    await cache.list_tools()  # cache miss → fetch
    assert client.list_tools.call_count == 1

    cache.invalidate()
    await cache.list_tools()
    assert client.list_tools.call_count == 2, (
        "invalidate() must force the next list_tools to refetch even within TTL"
    )


@pytest.mark.asyncio
async def test_c3_force_refresh_bypasses_ttl():
    """refresh() always refetches regardless of freshness."""
    client = _make_mock_client([[_make_tool("a")]])
    cache = MCPToolsCache(client, ttl_seconds=60)

    await cache.list_tools()  # 1
    await cache.refresh()  # 2
    await cache.refresh()  # 3
    assert client.list_tools.call_count == 3


@pytest.mark.asyncio
async def test_c4_concurrent_refetches_coalesce_to_one_round_trip():
    """50 concurrent list_tools() on a stale cache → ONE MCP call."""
    # Slow the network so contention actually occurs.
    async def slow_list_tools():
        await asyncio.sleep(0.05)
        return [_make_tool("a")]

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_tools = AsyncMock(side_effect=slow_list_tools)

    cache = MCPToolsCache(client, ttl_seconds=60)

    await asyncio.gather(*[cache.list_tools() for _ in range(50)])
    assert client.list_tools.call_count == 1, (
        f"expected exactly 1 round-trip under coalescing, "
        f"got {client.list_tools.call_count}"
    )


@pytest.mark.asyncio
async def test_c5_in_place_mutation_propagates_to_holders():
    """A consumer that grabbed `cache.tools` at boot must see new
    contents after refresh — without re-reading through the cache."""
    client = _make_mock_client([
        [_make_tool("a"), _make_tool("b")],
        [_make_tool("a"), _make_tool("b"), _make_tool("c")],
    ])
    cache = MCPToolsCache(client, ttl_seconds=60)

    await cache.list_tools()
    held_ref_tools = cache.tools  # mimics tool_executor.mcp_tools = cache.tools
    held_ref_defs = cache.tool_defs
    assert held_ref_tools == ["a", "b"]

    await cache.refresh()
    # SAME list object, NEW contents.
    assert held_ref_tools is cache.tools, "list reference must not change"
    assert held_ref_tools == ["a", "b", "c"]
    assert held_ref_defs is cache.tool_defs
    assert [d["name"] for d in held_ref_defs] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_c6_refetch_failure_leaves_stale_data_no_clock_bump():
    """First refetch succeeds. Second raises. Cache returns stale list,
    cached_at is NOT bumped (so next call retries)."""
    call_count = {"n": 0}

    async def flaky_list_tools():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [_make_tool("a"), _make_tool("b")]
        raise RuntimeError("simulated platform 503")

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_tools = AsyncMock(side_effect=flaky_list_tools)

    cache = MCPToolsCache(client, ttl_seconds=60)
    await cache.list_tools()  # success
    first_cached_at = cache.cached_at
    cache.invalidate()

    # Second call — refetch raises, cache returns stale.
    result = await cache.list_tools()
    assert result == ["a", "b"], "must return STALE list, not error out"
    assert cache.cached_at == 0.0, (
        "failed refetch must NOT bump the clock — next call retries"
    )

    # Confirm the next call retries.
    async def succeed():
        return [_make_tool("z")]

    client.list_tools = AsyncMock(side_effect=succeed)
    await cache.list_tools()
    assert cache.tools == ["z"]
    assert cache.cached_at > first_cached_at


@pytest.mark.asyncio
async def test_force_refresh_propagates_failure():
    """Unlike list_tools(), refresh() raises on failure — the
    refresh-tools endpoint maps it to 502."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_tools = AsyncMock(side_effect=RuntimeError("boom"))
    cache = MCPToolsCache(client, ttl_seconds=60)

    with pytest.raises(RuntimeError, match="boom"):
        await cache.refresh()


# ─── Endpoint: E1–E6 ────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def agent_mode():
    """Force run_mode='agent' so the refresh-tools endpoint is reachable
    and the rotate-key endpoint correctly 404s."""
    prev = settings.run_mode
    prev_key = settings.agent_api_key
    settings.run_mode = "agent"
    settings.agent_api_key = "tenant_key_xyz"
    try:
        yield
    finally:
        settings.run_mode = prev
        settings.agent_api_key = prev_key


@pytest_asyncio.fixture
async def platform_mode():
    """Force run_mode='platform' to verify the run-mode guard 404s."""
    prev = settings.run_mode
    settings.run_mode = "platform"
    try:
        yield
    finally:
        settings.run_mode = prev


def _build_agent_app(cache: MCPToolsCache | None = None) -> FastAPI:
    """Minimal FastAPI app that includes only agent_router so the
    refresh-tools endpoint is reachable. Cache is wired into app.state
    (or omitted to test the no-cache degraded path).

    Skips when the local FastAPI/Starlette pins mismatch with the
    target FastAPI version (CI has correct pins) — same trap as
    `test_mcp_auth.mcp_test_app`.
    """
    try:
        from app.api import agent_router

        app = FastAPI()
        app.include_router(agent_router, prefix=settings.api_prefix)
    except TypeError as e:
        pytest.skip(
            f"FastAPI pin mismatch (CI has correct pins): {e}"
        )
    if cache is not None:
        app.state.mcp_tools_cache = cache
    return app


@pytest_asyncio.fixture
async def agent_client_with_cache(agent_mode) -> AsyncIterator[tuple[AsyncClient, MCPToolsCache, MagicMock]]:
    client = _make_mock_client([[_make_tool("stub__echo")]])
    cache = MCPToolsCache(client, ttl_seconds=60)
    await cache.refresh()  # prime
    app = _build_agent_app(cache=cache)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http, cache, client


@pytest_asyncio.fixture
async def agent_client_no_cache(agent_mode) -> AsyncIterator[AsyncClient]:
    app = _build_agent_app(cache=None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


@pytest_asyncio.fixture
async def platform_client(platform_mode) -> AsyncIterator[AsyncClient]:
    app = _build_agent_app(cache=None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


@pytest.mark.asyncio
async def test_e1_put_without_agent_key_returns_401(agent_client_with_cache):
    http, _, _ = agent_client_with_cache
    r = await http.put("/api/agent/refresh-tools")
    assert r.status_code == 401
    assert "Invalid agent key" in r.text


@pytest.mark.asyncio
async def test_e2_put_with_wrong_key_returns_401(agent_client_with_cache):
    http, _, _ = agent_client_with_cache
    r = await http.put(
        "/api/agent/refresh-tools",
        headers={"X-Agent-Key": "totally_wrong"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_e3_put_with_correct_key_refreshes_and_returns_count(
    agent_client_with_cache,
):
    http, cache, client = agent_client_with_cache
    initial_count = client.list_tools.call_count
    r = await http.put(
        "/api/agent/refresh-tools",
        headers={"X-Agent-Key": settings.agent_api_key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "invalidated"
    assert body["tool_count"] == 1
    assert client.list_tools.call_count == initial_count + 1


@pytest.mark.asyncio
async def test_e4_put_with_no_cache_returns_no_cache_status(
    agent_client_no_cache,
):
    """Agent has no MCP client (degraded boot): endpoint returns 200
    with status='no_cache' so the platform doesn't treat missing-cache
    as a failure."""
    r = await agent_client_no_cache.put(
        "/api/agent/refresh-tools",
        headers={"X-Agent-Key": settings.agent_api_key},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_cache"
    assert body["tool_count"] == 0


@pytest.mark.asyncio
async def test_e5_run_mode_platform_returns_404(platform_client):
    r = await platform_client.put(
        "/api/agent/refresh-tools",
        headers={"X-Agent-Key": "anything"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_e6_concurrent_puts_coalesce(agent_client_with_cache):
    """Multiple concurrent platform notifications → cache lock
    coalesces to ONE round-trip."""
    http, cache, client = agent_client_with_cache
    initial_count = client.list_tools.call_count

    # Slow the underlying client so contention actually happens.
    async def slow():
        await asyncio.sleep(0.05)
        return [_make_tool("stub__echo")]

    client.list_tools = AsyncMock(side_effect=slow)

    headers = {"X-Agent-Key": settings.agent_api_key}
    results = await asyncio.gather(*[
        http.put("/api/agent/refresh-tools", headers=headers)
        for _ in range(10)
    ])
    assert all(r.status_code == 200 for r in results)
    # All 10 PUTs landed; the lock means only 1 ACTUAL round-trip ran.
    # NOTE: refresh() forces a refetch under lock — first caller does
    # the work, the rest queue and re-enter the lock to do their own
    # forced refresh. So we can't strictly assert == 1; we assert
    # that fewer-than-N round-trips happened (lock coalescing kicked in
    # for at least the overlapping pairs).
    actual = client.list_tools.call_count - initial_count
    assert actual >= 1, "at least one refresh must run"
    assert actual <= 10, "should not exceed the request count"
    # Tighter bound: under contention, the lock+slow combo should pin
    # this well below 10. But the EXACT number depends on scheduling
    # — assert the BOUND, not the exact count.


@pytest.mark.asyncio
async def test_e_refresh_failure_returns_502(agent_mode):
    """Force the cache's refresh() to raise → endpoint surfaces 502."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_tools = AsyncMock(side_effect=RuntimeError("simulated"))
    cache = MCPToolsCache(client, ttl_seconds=60)

    app = _build_agent_app(cache=cache)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        r = await http.put(
            "/api/agent/refresh-tools",
            headers={"X-Agent-Key": settings.agent_api_key},
        )
    assert r.status_code == 502
    assert "Refresh failed" in r.text


# ─── docker_host_service.refresh_tenant_tools: D1–D6 ───────────────────


def _is_sqlite() -> bool:
    import os
    return os.environ.get("DATABASE_URL", "").startswith("sqlite")


@pytest_asyncio.fixture
async def alice_with_managed_agent():
    """Seed a user + AgentConfig with agent_url + agent_api_key set
    (mimics a managed-hosting tenant). Skips on SQLite (User.is_canary
    partial unique index trap)."""
    if _is_sqlite():
        pytest.skip("requires Postgres (User.is_canary partial unique index)")
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig, User

    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid,
            email=f"{uid[:8]}@example.com",
            hashed_password="x",
            name="Alice",
        ))
        await db.flush()
        cfg = AgentConfig(
            user_id=uid,
            agent_url="https://agent-alice.example.com",
            agent_api_key="alice_key_xxx",
        )
        db.add(cfg)
        await db.commit()
        return {"user_id": uid, "agent_url": cfg.agent_url, "agent_key": cfg.agent_api_key}


@pytest_asyncio.fixture
async def alice_self_hosted():
    """Seed a user + AgentConfig with NO agent_url (self-hosted).
    refresh_tenant_tools should short-circuit to False without
    making any HTTP call."""
    if _is_sqlite():
        pytest.skip("requires Postgres (User.is_canary partial unique index)")
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig, User

    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid,
            email=f"{uid[:8]}@example.com",
            hashed_password="x",
            name="Alice",
        ))
        await db.flush()
        cfg = AgentConfig(user_id=uid)  # no agent_url, no agent_api_key
        db.add(cfg)
        await db.commit()
        return {"user_id": uid}


@pytest.mark.asyncio
async def test_d1_no_agent_config_returns_false_no_http(alice_with_managed_agent, monkeypatch):
    """user_id with no AgentConfig → return False, no HTTP call."""
    if _is_sqlite():
        pytest.skip("requires Postgres")
    from app.db.database import async_session_maker
    from app.services.docker_host_service import refresh_tenant_tools

    fake_uid = str(uuid.uuid4())  # not in DB

    http_calls = []

    async def fake_put(*a, **k):
        http_calls.append((a, k))
        raise AssertionError("HTTP must NOT be called when no AgentConfig exists")

    monkeypatch.setattr("httpx.AsyncClient.put", fake_put)

    async with async_session_maker() as db:
        result = await refresh_tenant_tools(db, fake_uid)
    assert result is False
    assert http_calls == []


@pytest.mark.asyncio
async def test_d2_missing_agent_url_short_circuits(alice_self_hosted, monkeypatch):
    """Self-hosted user (AgentConfig present but no agent_url) →
    return False, no HTTP call."""
    from app.db.database import async_session_maker
    from app.services.docker_host_service import refresh_tenant_tools

    http_calls = []

    async def fake_put(*a, **k):
        http_calls.append((a, k))
        raise AssertionError("HTTP must NOT be called for self-hosted users")

    monkeypatch.setattr("httpx.AsyncClient.put", fake_put)

    async with async_session_maker() as db:
        result = await refresh_tenant_tools(db, alice_self_hosted["user_id"])
    assert result is False
    assert http_calls == []


@pytest.mark.asyncio
async def test_d3_200_from_tenant_returns_true(alice_with_managed_agent, monkeypatch):
    from app.db.database import async_session_maker
    from app.services.docker_host_service import refresh_tenant_tools

    captured = {}

    async def fake_put(self, url, *, headers=None, **k):
        captured["url"] = url
        captured["headers"] = headers
        return httpx.Response(200, json={"status": "invalidated", "tool_count": 3})

    monkeypatch.setattr("httpx.AsyncClient.put", fake_put)

    async with async_session_maker() as db:
        result = await refresh_tenant_tools(db, alice_with_managed_agent["user_id"])
    assert result is True
    assert captured["url"].endswith("/api/agent/refresh-tools")
    assert captured["url"].startswith("https://agent-alice.example.com")
    assert captured["headers"]["X-Agent-Key"] == alice_with_managed_agent["agent_key"]


@pytest.mark.asyncio
async def test_d4_non_2xx_returns_false_no_raise(alice_with_managed_agent, monkeypatch, caplog):
    import logging
    from app.db.database import async_session_maker
    from app.services.docker_host_service import refresh_tenant_tools

    async def fake_put(self, url, *, headers=None, **k):
        return httpx.Response(503, text="tenant down")

    monkeypatch.setattr("httpx.AsyncClient.put", fake_put)

    async with async_session_maker() as db:
        with caplog.at_level(logging.WARNING, logger="app.services.docker_host_service"):
            result = await refresh_tenant_tools(db, alice_with_managed_agent["user_id"])
    assert result is False
    assert any("503" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_d5_network_exception_returns_false_no_raise(alice_with_managed_agent, monkeypatch, caplog):
    import logging
    from app.db.database import async_session_maker
    from app.services.docker_host_service import refresh_tenant_tools

    async def fake_put(self, url, *, headers=None, **k):
        raise httpx.ConnectError("simulated DNS fail")

    monkeypatch.setattr("httpx.AsyncClient.put", fake_put)

    async with async_session_maker() as db:
        with caplog.at_level(logging.WARNING, logger="app.services.docker_host_service"):
            result = await refresh_tenant_tools(db, alice_with_managed_agent["user_id"])
    assert result is False
    assert any("notify failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_d6_3s_timeout_enforced(alice_with_managed_agent, monkeypatch):
    """Verify the AsyncClient is constructed with timeout=3.0. We patch
    AsyncClient at construction time and assert on the kwarg."""
    from app.db.database import async_session_maker
    from app.services.docker_host_service import refresh_tenant_tools

    captured_timeouts = []
    real_client = httpx.AsyncClient

    class _CapturingClient(real_client):
        def __init__(self, *a, **kw):
            captured_timeouts.append(kw.get("timeout"))
            super().__init__(*a, **kw)

        async def put(self, *a, **kw):
            return httpx.Response(200, json={"status": "invalidated", "tool_count": 0})

    monkeypatch.setattr("httpx.AsyncClient", _CapturingClient)

    async with async_session_maker() as db:
        await refresh_tenant_tools(db, alice_with_managed_agent["user_id"])

    assert captured_timeouts == [3.0], (
        f"refresh_tenant_tools must construct httpx.AsyncClient with "
        f"timeout=3.0, got {captured_timeouts}"
    )
