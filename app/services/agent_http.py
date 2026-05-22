"""
Shared httpx.AsyncClient for platform → tenant-agent HTTP calls (TKT-LAT-007).

Every ad-hoc `httpx.AsyncClient(...)` constructed in `api/*.py` pays a
TLS handshake + connection-pool teardown for what's effectively the
same hot path against the same per-tenant agent host. A module-level
client reuses keepalive connections across requests and across call
sites, eliminating ~50–150ms of overhead per hop on cold-pool turns.

This helper is **distinct from** the bridge client in
`docker_host_service.py` — that one talks to the platform's mTLS
bridge daemon; this one talks to per-tenant agents with plain
X-Agent-Key auth.

Usage:
    from app.services.agent_http import get_agent_http_client

    client = get_agent_http_client()
    resp = await client.get(url, headers={"X-Agent-Key": key},
                            timeout=10.0)

Per-call overrides (timeout, follow_redirects, etc.) are passed
directly to the request method, not to the client constructor —
that's the only way to share one client across call sites with
different latency budgets.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Conservative default. Per-call overrides win when callers pass
# `timeout=` to the request method.
_DEFAULT_TIMEOUT_S = 30.0

# Pool sizing: 20 keepalive slots is enough for the platform's typical
# concurrency (one open ws_chat upstream per active user + handful of
# REST hops per page render). max_connections=100 caps under burst.
_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=100)

_client: Optional[httpx.AsyncClient] = None


def get_agent_http_client() -> httpx.AsyncClient:
    """Return the module-level shared AsyncClient, building it on
    first call. Safe to call from any sync or async context."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT_S,
            limits=_LIMITS,
        )
        logger.info(
            "[PERF] agent_http_client=built timeout_s=%s "
            "max_keepalive=%s max_connections=%s",
            _DEFAULT_TIMEOUT_S,
            _LIMITS.max_keepalive_connections,
            _LIMITS.max_connections,
        )
    return _client


async def close_agent_http_client() -> None:
    """Tear down the shared client. Called during process shutdown
    so we don't leak file descriptors. After close, the next call to
    get_agent_http_client() rebuilds a fresh client — useful in
    tests where each test wants an isolated client."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as e:
            logger.warning("agent_http_client close failed: %s", e)
        finally:
            _client = None


def reset_for_test() -> None:
    """Drop the module-level singleton without closing it.
    Test-only helper for monkeypatching."""
    global _client
    _client = None

