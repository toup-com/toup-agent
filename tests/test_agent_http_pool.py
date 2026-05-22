"""Regression tests for TKT-LAT-007 — shared platform→agent httpx pool.

The shared agent_http client must:
  * Build lazily on first call (singleton).
  * Return the same instance on subsequent calls.
  * Tear down cleanly on close + rebuild after reset.
  * Migrate the hot proxy sites (messages_recover, day_chats, sessions,
    dashboard_proxy, stats) off ad-hoc `async with httpx.AsyncClient(...)`.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


def _load_agent_http_directly():
    """Import the agent_http module by file path, bypassing
    app.services.__init__ which eagerly imports services that pull
    in Settings (and the local .env may have stale Stripe keys that
    fail pydantic-settings validation, unrelated to this module)."""
    backend_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "agent_http_under_test",
        backend_dir / "app" / "services" / "agent_http.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_singleton_reuse():
    """Two calls to get_agent_http_client() must return the same
    object — that's the entire point of the shared pool."""
    agent_http = _load_agent_http_directly()
    agent_http.reset_for_test()
    c1 = agent_http.get_agent_http_client()
    c2 = agent_http.get_agent_http_client()
    assert c1 is c2
    asyncio.run(agent_http.close_agent_http_client())


def test_reset_for_test_drops_singleton():
    """reset_for_test() must drop the singleton so the next call
    rebuilds."""
    agent_http = _load_agent_http_directly()
    agent_http.reset_for_test()
    c1 = agent_http.get_agent_http_client()
    agent_http.reset_for_test()
    c2 = agent_http.get_agent_http_client()
    assert c1 is not c2
    asyncio.run(agent_http.close_agent_http_client())


def test_close_drops_singleton_and_rebuilds():
    """close_agent_http_client() must aclose() the client and clear
    the singleton so the next call builds a fresh one."""
    agent_http = _load_agent_http_directly()
    agent_http.reset_for_test()
    c1 = agent_http.get_agent_http_client()
    asyncio.run(agent_http.close_agent_http_client())
    c2 = agent_http.get_agent_http_client()
    assert c1 is not c2
    asyncio.run(agent_http.close_agent_http_client())


def test_pool_limits_are_set():
    """The shared client must configure keepalive limits, not default
    httpx behavior (which can starve the pool under burst)."""
    agent_http = _load_agent_http_directly()
    assert agent_http._LIMITS.max_keepalive_connections == 20
    assert agent_http._LIMITS.max_connections == 100


def test_hot_proxies_use_shared_client():
    """Source-level invariant: the 5 hot proxy sites we migrated
    must no longer construct ad-hoc AsyncClient. Catches regressions
    where a future edit reverts to `async with httpx.AsyncClient(...)`."""
    backend_dir = Path(__file__).resolve().parents[1]
    migrated = [
        backend_dir / "app" / "api" / "messages_recover.py",
        backend_dir / "app" / "api" / "day_chats.py",
        backend_dir / "app" / "api" / "sessions.py",
        backend_dir / "app" / "api" / "dashboard_proxy.py",
        backend_dir / "app" / "api" / "stats.py",
    ]
    for path in migrated:
        src = path.read_text()
        # Every migrated file must import the shared helper.
        assert "from app.services.agent_http import get_agent_http_client" in src, (
            f"{path.name} doesn't import the shared client"
        )
        # And the proxy function body must not contain ad-hoc construction.
        # We allow the import line for httpx (kept for typing/etc) but
        # the proxy fn itself must not call `httpx.AsyncClient(`.
        proxy_idx = src.find("async def _proxy")
        if proxy_idx < 0:
            continue
        # Walk to the next top-level def to bound the search.
        end_idx = src.find("\n\n\nasync def ", proxy_idx + 10)
        if end_idx < 0:
            end_idx = src.find("\n\n\ndef ", proxy_idx + 10)
        if end_idx < 0:
            end_idx = len(src)
        proxy_body = src[proxy_idx:end_idx]
        assert "httpx.AsyncClient(" not in proxy_body, (
            f"{path.name} proxy still constructs an ad-hoc AsyncClient"
        )


def test_close_wired_into_platform_main_shutdown():
    """The shared client must be torn down in platform_main's
    lifespan shutdown so file descriptors don't leak across deploys."""
    backend_dir = Path(__file__).resolve().parents[1]
    src = (backend_dir / "platform_main.py").read_text()
    assert "from app.services.agent_http import close_agent_http_client" in src
    assert "await close_agent_http_client()" in src
