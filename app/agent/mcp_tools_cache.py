"""
T1h — Agent-side MCP tools cache.

The agent fetches the per-user tool list from the platform via
`mcp_client.list_tools()`. Without a cache, every tool_defs read in
`agent_runner` (every turn) would round-trip to the platform — wasteful,
and adds a sync-point on the per-user filter middleware (T1f).

This module owns:

  - `MCPToolsCache`: TTL-bound cache (60s) with both async (`list_tools`,
    enforces TTL) and sync (`tools`, `tool_defs` attributes, last
    refreshed snapshot) read shapes. The dual shape exists because
    `agent_runner.tool_defs` is a sync property — it must read without
    awaiting — while T1g's boot-time discovery and T1h's invalidation
    callsite both await fresh data.

  - `invalidate()` clears the freshness timestamp so the next
    `list_tools()` refetches; `refresh()` is the explicit "fetch now,
    update sync attrs" entry point used by the refresh-tools endpoint.

What this module deliberately does NOT do:

  - HTTP / FastAPI surface — that's the endpoint in `app/api/agent.py`.
  - Pub/sub fanout for cross-process invalidation — per `02-rollout.md`
    T1h Q2 sign-off, hybrid TTL + targeted PUT is good enough for v1.
  - Per-connector filter on invalidation — drop-everything is the
    contract.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Match the spec's 60-second TTL exactly. Bound on staleness when the
# platform's connect/disconnect notifier (T1h step 3) fails — even with
# the WARN-and-keep-going semantics of `refresh_tenant_tools`, the cache
# self-heals on this clock.
DEFAULT_TTL_SECONDS = 60


class MCPToolsCache:
    """TTL-bounded cache of `mcp_client.list_tools()` results.

    Concurrency: a per-instance `asyncio.Lock` coalesces concurrent
    refreshes — 50 turns racing on the same expired cache produce ONE
    `list_tools()` round-trip. Same pattern as the dispatcher's
    per-identity refresh lock (T1e).
    """

    def __init__(self, client: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._client = client
        self._ttl_s = ttl_seconds
        # Sync snapshot — these attributes are read by `agent_runner`
        # and `tool_executor` without awaiting. They reflect the LAST
        # successful refetch; a failing refetch leaves stale-but-
        # well-shaped data so the agent doesn't lose all connector
        # tools on a transient platform error.
        self.tools: list[str] = []
        self.tool_defs: list[dict] = []
        # Monotonic timestamp of the last successful refetch. Zero
        # means "never fetched" or "explicitly invalidated".
        self._cached_at: float = 0.0
        self._lock = asyncio.Lock()

    # ── State predicates ──────────────────────────────────────

    def is_fresh(self) -> bool:
        if self._cached_at <= 0:
            return False
        return (time.monotonic() - self._cached_at) < self._ttl_s

    @property
    def cached_at(self) -> float:
        """Read-only view of the last refetch timestamp. Tests assert
        on this to verify the TTL clock advanced as expected."""
        return self._cached_at

    # ── Cache operations ──────────────────────────────────────

    async def list_tools(self) -> list[str]:
        """Return tool names, refetching if stale.

        TTL contract:
          - cache fresh → return cached, no MCP round-trip
          - cache expired or invalidated → refetch under lock,
            update sync attrs, return new list
          - refetch raises → log WARN, return STALE list (don't crash
            callers; the periodic background task will retry)
        """
        if self.is_fresh():
            return self.tools

        async with self._lock:
            # Re-check inside the lock — another waiter may have
            # refreshed while we queued. This is the coalescing
            # point.
            if self.is_fresh():
                return self.tools
            try:
                await self._refetch_locked()
            except Exception as e:
                logger.warning(
                    "[mcp_tools_cache] refetch failed (returning stale): %s",
                    e,
                )
                # Don't update _cached_at — next call retries.
                # Don't bubble — caller gets stale list (well-shaped).
            return self.tools

    async def refresh(self) -> list[str]:
        """Force a refetch regardless of TTL. Used by the
        `/api/agent/refresh-tools` endpoint after platform-side
        connect/disconnect to push the new tool list to the runner's
        sync attrs WITHOUT waiting for the next TTL tick."""
        async with self._lock:
            try:
                await self._refetch_locked()
            except Exception as e:
                logger.warning(
                    "[mcp_tools_cache] forced refresh failed: %s", e,
                )
                raise
            return self.tools

    def invalidate(self) -> None:
        """Mark the cache stale without refetching. The next
        `list_tools()` call will refetch. Used when an inexpensive
        signal says "tools may have changed" but a synchronous
        refetch isn't required (e.g. periodic safety-net)."""
        self._cached_at = 0.0

    # ── Internal ──────────────────────────────────────────────

    async def _refetch_locked(self) -> None:
        """The actual MCP round-trip. Caller MUST hold `self._lock`.

        Updates `self.tools`, `self.tool_defs`, and `self._cached_at`.
        Lists are mutated IN PLACE (clear + extend) rather than
        reassigned so that any consumer holding a reference to the
        cache's list (e.g. `tool_executor.mcp_tools = cache.tools` at
        boot) sees the new contents automatically. The alternative —
        reassignment — would require every consumer to re-read through
        the cache on every access, which we'd then have to plumb
        through `tool_executor`'s sync property pattern.
        """
        # The same `async with` lifecycle T1g uses on call_tool.
        async with self._client:
            tools = await self._client.list_tools()
        new_names = [t.name for t in tools]
        new_defs = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": (t.inputSchema or {"type": "object"}),
            }
            for t in tools
        ]
        # In-place mutation — preserves references held by callers.
        self.tools.clear()
        self.tools.extend(new_names)
        self.tool_defs.clear()
        self.tool_defs.extend(new_defs)
        self._cached_at = time.monotonic()
        logger.info(
            "[mcp_tools_cache] refetched: %d tool(s)", len(self.tools),
        )


# ─── Background refresh task ─────────────────────────────────────────


async def periodic_refresh_loop(
    cache: MCPToolsCache,
    interval_s: int = DEFAULT_TTL_SECONDS,
    *,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Coroutine that wakes every `interval_s` to call
    `await cache.list_tools()`. Keeps the sync attrs warm without
    every turn paying the TTL miss.

    The agent's lifespan launches this once at boot. `stop_event` is
    optional — if provided, the loop exits when it's set; tests use
    this to terminate cleanly.

    Errors inside `list_tools()` are already swallowed to WARN;
    nothing here can fail loudly.
    """
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            await cache.list_tools()
        except Exception as e:
            # Defense in depth — list_tools shouldn't raise, but log
            # if it ever does so we don't have a silent dead loop.
            logger.warning(
                "[mcp_tools_cache] periodic refresh raised: %s", e,
            )
        try:
            if stop_event is not None:
                # Wait for either the interval OR the stop signal.
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
                # If we got here without TimeoutError, the event was
                # set during the wait — exit cleanly.
                return
            else:
                await asyncio.sleep(interval_s)
        except asyncio.TimeoutError:
            # Normal interval tick — continue.
            continue
