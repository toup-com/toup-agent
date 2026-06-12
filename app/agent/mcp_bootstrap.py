"""
Shared MCP client + tools-cache bootstrap.

This block historically lived inline in agent_main's lifespan, gated on
`settings.platform_api_url and settings.agent_api_key`. That gate is
right for dedicated containers (env carries the key at boot), but a
pool-lobby container boots WITHOUT `agent_api_key` — the key only
arrives later via POST /admin/bind — so the lifespan skipped the whole
block and the container lived out its life with zero connector tools
(gmail__*, calendar__*, …) while the platform showed the user's OAuth
identity as Connected. The model, offered no gmail__ tools, improvised
with the browser-extension tools and told users to pair the desktop
Chrome extension. The post-OAuth refresh-tools push couldn't repair it
either: it answered `{"status": "no_cache"}` with HTTP 200, which the
platform logged as success.

`ensure_mcp_initialized()` is the single construction site now:

  * agent_main lifespan calls it at boot (dedicated containers —
    unchanged behavior),
  * POST /admin/bind calls it right after apply_to_settings (pool
    claim — the moment agent_api_key first exists),
  * PUT /api/agent/refresh-tools calls it as a self-heal before
    reporting no_cache (covers containers bound before this code
    deployed, without waiting for a re-bind).

Idempotent and concurrency-safe: the app.state check plus a module
lock make double-init (bind racing a refresh-tools PUT) a no-op.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_init_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    # Created lazily so importing this module never requires a running
    # event loop (Python <3.10 Lock() bound to the current loop).
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


async def ensure_mcp_initialized(
    app: Any, *, defer_initial_refresh: bool = False
) -> str:
    """Construct and wire the MCP client + tools cache if absent.

    Returns a status string for the caller's log line:
      "already"        — app.state.mcp_tools_cache exists; no-op
      "initialized"    — client + cache constructed and wired
      "not_configured" — platform_api_url / agent_api_key still missing
      "unavailable"    — fastmcp not installed
      "error"          — construction failed (logged with traceback)

    With `defer_initial_refresh=True` the first list_tools round-trip
    runs as a background task so the caller (boot under
    agent_defer_boot_init, or /admin/bind which must return fast)
    doesn't wait on the platform; the 60s periodic loop is the retry
    net either way. The cache is wired before its first successful
    refresh on purpose — consumers read `cache.tools` (mutated in
    place by refresh) at call time, so they simply see an empty list
    until the fetch lands.
    """
    from app.config import settings

    if getattr(app.state, "mcp_tools_cache", None) is not None:
        _warn_if_key_drifted(app, settings)
        return "already"
    if not (settings.platform_api_url and settings.agent_api_key):
        return "not_configured"

    async with _get_lock():
        if getattr(app.state, "mcp_tools_cache", None) is not None:
            _warn_if_key_drifted(app, settings)
            return "already"
        try:
            from fastmcp import Client as MCPClient

            from app.agent.mcp_client_auth import AgentMCPAuth
            from app.agent.mcp_tools_cache import (
                MCPToolsCache,
                periodic_refresh_loop,
            )
        except ImportError:
            print("⚠️ fastmcp not installed — MCP client disabled")
            return "unavailable"
        except Exception as e:
            # A dependency-drift TypeError/RuntimeError inside fastmcp's
            # module-level code must degrade the agent (no connector
            # tools), never abort lifespan — this image ships fleet-wide.
            logger.exception("[mcp_bootstrap] MCP dependency import failed")
            print(f"⚠️ MCP client error: {e}")
            return "error"

        try:
            # Platform mounts FastMCP at /api/mcp with inner path=/mcp,
            # so the streamable-HTTP endpoint is /api/mcp/mcp. No
            # trailing slash: in production it triggers FastAPI's
            # redirect_slashes → 307 built as `http://` (X-Forwarded-
            # Proto not honored) → Cloudflare/Caddy 301 back to https →
            # POST downgrades to GET → MCP rejects the GET with 400.
            mcp_url = f"{settings.platform_api_url.rstrip('/')}/mcp/mcp"
            mcp_client = MCPClient(
                mcp_url,
                auth=AgentMCPAuth(settings.agent_api_key),
            )
            mcp_tools_cache = MCPToolsCache(mcp_client)

            # First discovery — primes the cache so the next turn
            # doesn't pay a TTL miss. Errors are swallowed with a full
            # traceback (a persistent failure here silently strips
            # every connector tool from the agent — we lost a day of
            # "no Gmail tool" once because only str(e) was printed);
            # the periodic refresh task retries every 60s.
            async def _initial_refresh() -> None:
                _t0 = time.monotonic()
                try:
                    await mcp_tools_cache.refresh()
                    _ms = int((time.monotonic() - _t0) * 1000)
                    print(
                        f"🔗 MCP connected ({len(mcp_tools_cache.tools)} tools) "
                        f"at {mcp_url}: {mcp_tools_cache.tools} "
                        f"[PERF] boot_mcp_refresh_ms={_ms}"
                    )
                except Exception as e:
                    import traceback

                    print(
                        f"⚠️ MCP tool discovery failed at {mcp_url} "
                        f"({type(e).__name__}: {e}) — connector tools will be "
                        f"unavailable until next refresh succeeds:\n"
                        f"{traceback.format_exc()}"
                    )

            # Wire the cache into the executor BEFORE starting the
            # periodic loop — the loop refreshes the cache, which the
            # executor reads via these same attributes. The list
            # objects are mutated in place by refresh(), so a one-time
            # reference hand-off is enough. Wiring before the first
            # fetch is safe for the same reason: consumers see an
            # empty list until it lands.
            tool_executor = getattr(app.state, "tool_executor", None)
            if tool_executor:
                tool_executor.mcp_client = mcp_client
                tool_executor.mcp_tools_cache = mcp_tools_cache
                tool_executor.mcp_tools = mcp_tools_cache.tools
                tool_executor.mcp_tool_defs = mcp_tools_cache.tool_defs

            # Late-bind the client into the routine/trigger runners —
            # they were constructed before MCP existed, so handlers got
            # `_mcp_client=None` at first. The earliest a routine can
            # fire is the user's tz-local wake time, far after this.
            routine_runner = getattr(app.state, "routine_runner", None)
            if routine_runner is not None:
                routine_runner.set_mcp_client(mcp_client)
                # Also wire the AgentRunner so generic `agent_task`
                # routines can use the agent's tool-using turn pipeline.
                agent_runner = getattr(app.state, "agent_runner", None)
                if agent_runner is not None:
                    routine_runner.set_agent_runner(agent_runner)
            trigger_runner = getattr(app.state, "trigger_runner", None)
            if trigger_runner is not None:
                trigger_runner.set_mcp_client(mcp_client)

            # Periodic refresh task — keeps the sync snapshot fresh
            # without every turn paying the cache-miss cost. Stored on
            # app.state so a graceful shutdown can cancel it.
            app.state.mcp_refresh_task = asyncio.create_task(
                periodic_refresh_loop(mcp_tools_cache),
                name="mcp_tools_periodic_refresh",
            )
            # The key the client's AgentMCPAuth captured — lets the
            # "already" path detect a re-bind that changed the key
            # (the client would silently keep signing with the old
            # one; only a container recreate swaps it today).
            app.state.mcp_bound_key = settings.agent_api_key
            # Setting app.state.mcp_tools_cache is the "initialized"
            # marker the idempotency check above keys on — last, so a
            # failure anywhere in this block leaves the app retryable.
            app.state.mcp_tools_cache = mcp_tools_cache
        except Exception as e:
            logger.exception("[mcp_bootstrap] MCP client init failed")
            print(f"⚠️ MCP client error: {e}")
            return "error"

    # First fetch runs OUTSIDE the init lock — it's a platform network
    # round-trip, and holding the lock across it would park a
    # concurrent /admin/bind behind a slow/hung fetch. The cache's own
    # per-instance lock coalesces this with the periodic loop's first
    # tick, so at worst one extra list_tools fires.
    if defer_initial_refresh:
        # Keep a strong reference — asyncio holds only weak refs to
        # tasks, so an unreferenced fire-and-forget task can be GC'd
        # mid-flight.
        app.state.mcp_initial_refresh_task = asyncio.create_task(
            _initial_refresh(), name="lat017-mcp-refresh"
        )
        print("[PERF] boot_deferred=mcp_refresh")
    else:
        await _initial_refresh()
    return "initialized"


def _warn_if_key_drifted(app: Any, settings: Any) -> None:
    bound = getattr(app.state, "mcp_bound_key", None)
    if bound and settings.agent_api_key and bound != settings.agent_api_key:
        logger.warning(
            "[mcp_bootstrap] agent_api_key changed since MCP init — the "
            "MCP client still signs with the OLD key (connector tools "
            "will 401); recreate the container to swap it"
        )
