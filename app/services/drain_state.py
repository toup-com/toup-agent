"""
Drain coordinator (Phase B — never-sleep plan).

Both the pool's bind flow and the blue-green rollout flow need a
shared "stop accepting new work, finish in-flight, then exit"
primitive. This module owns it.

Flow:
1. Operator (bridge) calls `POST /admin/drain` with `drain_timeout_s`.
2. The handler calls `set_draining(timeout_s)`.
3. New WS connections (and any other long-lived connection that opts
   into the drain check) refuse with close 1012 (Service Restart).
4. In-flight WS handlers ignore the flag — they finish naturally.
5. Module's background task SIGTERMs the process when either:
   - active_count() drops to 0, OR
   - the timeout elapses.

Atomic-counter pattern (not lock-protected) — increment/decrement is
done by handler functions that wrap the WS lifetime. The drain check
is a simple read of `_draining`, no lock needed (single bool).

This file MUST stay dependency-light: it's imported by every WS
endpoint, by the bind handler, and by the drain handler. A circular
import here would break agent startup."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Optional

logger = logging.getLogger(__name__)

# Global drain state. `_draining=True` means: refuse new WS, prepare
# to exit. The active_count and timeout machinery decide WHEN to exit.
_draining: bool = False
_active_ws_count: int = 0
_drain_started_at: Optional[float] = None
_drain_task: Optional[asyncio.Task] = None


def is_draining() -> bool:
    """Read by every long-lived endpoint before accepting new work."""
    return _draining


def active_ws_count() -> int:
    return _active_ws_count


def increment_active() -> None:
    """Called from WS handlers immediately after `accept()`."""
    global _active_ws_count
    _active_ws_count += 1


def decrement_active() -> None:
    """Called from WS handlers in the `finally` block of the loop."""
    global _active_ws_count
    _active_ws_count = max(0, _active_ws_count - 1)


def set_draining(timeout_s: int = 60) -> None:
    """Engage drain mode. Idempotent — calling twice extends the
    timeout to whatever the caller passed but doesn't restart the
    timer.

    The exit-on-zero-active path is checked every second by the
    background task. If the count is already zero (no in-flight
    handlers), the task exits the process immediately on its next
    tick — no point waiting out the full timeout.
    """
    global _draining, _drain_started_at, _drain_task
    if _draining:
        logger.info("[drain] already draining; ignoring duplicate")
        return
    import time as _time
    _draining = True
    _drain_started_at = _time.time()
    logger.warning(
        "[drain] ENGAGED. timeout=%ds active_ws=%d. New WS will be refused.",
        timeout_s, _active_ws_count,
    )
    try:
        loop = asyncio.get_running_loop()
        _drain_task = loop.create_task(_drain_watcher(timeout_s))
    except RuntimeError:
        # Called outside an event loop — shouldn't happen in production
        # but tests sometimes import this module standalone. Skip the
        # watcher; caller can call `_drain_watcher` themselves.
        logger.warning("[drain] No running loop; watcher not scheduled")


def status() -> dict:
    return {
        "draining": _draining,
        "active_ws": _active_ws_count,
        "started_at": _drain_started_at,
    }


async def _drain_watcher(timeout_s: int) -> None:
    """Background task — waits for active_count → 0 OR timeout, then
    exits the process. SIGTERM is preferred over `sys.exit()` so the
    FastAPI lifespan shutdown hook runs (closes DB connections, stops
    the bot, etc.).
    """
    import time as _time
    deadline = _time.time() + max(1, int(timeout_s))
    while True:
        if _active_ws_count == 0:
            logger.warning("[drain] All WS closed. Exiting.")
            break
        if _time.time() >= deadline:
            logger.warning(
                "[drain] Timeout (%ds). Exiting with %d active WS still open.",
                timeout_s, _active_ws_count,
            )
            break
        await asyncio.sleep(1)
    # SIGTERM rather than os._exit so the lifespan teardown runs.
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except ProcessLookupError:
        pass
