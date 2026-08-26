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

# R31-42 / ND-26. In-flight AUTOMATION RUNS, by run id.
#
# The counter above has exactly one incrementer and one decrementer,
# both inside the chat WebSocket handler. Nothing about a run, a routine
# fire, a trigger dispatch or an outbox flush touches it — and the ASGI
# drain gate deliberately lets HTTP through, which is how an inbound
# push STARTS a run during a drain. So with no chat client attached, a
# drain SIGTERMs within about a second no matter how many runs are
# mid-flight, and the lifespan hook then marks every running `BuildJob`
# failed BEFORE the runners get their 30 s.
#
# R31-D measured what that costs: two of the founder's 26 August runs
# ran 362 s and 413 s against a 180 s cap and ended
# `error_class: "interrupted"` — a cap CANNOT fire at 2× its own
# value, so those runs were killed, not capped, and the same work took
# 58 s when nothing interrupted it.
#
# A set of ids rather than an int, deliberately: a counter that leaks
# one increment wedges a deploy forever, and there is no way to find
# out which run did it. The ids are logged when the drain waits, so a
# stuck deploy names the run holding it.
_active_runs: set = set()
# Runs may not hold a drain open past the cap they are already bounded
# by — the bridge hard-stops the slot at `drain_timeout_s + 5` anyway,
# so waiting longer than the run can legally take buys nothing and
# hides the bridge's kill behind our own.
RUN_DRAIN_MAX_S: int = 180


def is_draining() -> bool:
    """Read by every long-lived endpoint before accepting new work."""
    return _draining


def active_ws_count() -> int:
    return _active_ws_count


def run_started(run_id: str) -> None:
    """An automation run began. Called at run mint (R31-42)."""
    if run_id:
        _active_runs.add(str(run_id))


def run_finished(run_id: str) -> None:
    """An automation run reached a terminal. Idempotent."""
    _active_runs.discard(str(run_id))


def active_runs() -> set:
    return set(_active_runs)


def should_refuse_new_run() -> bool:
    """May a NEW run start right now?

    "Never starts a run it will kill" (§4.8). A run minted during a
    drain has, at best, `drain_timeout_s` to do three minutes of work,
    and at worst is killed before its first step — and a killed run
    does not fail quietly: it is reaped as `failed/lost` and the user
    is told their automation broke.
    """
    return _draining


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
        "active_runs": sorted(_active_runs),
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
    # A run may hold the drain, but never past its own cap.
    run_deadline = _time.time() + min(
        max(1, int(timeout_s)), RUN_DRAIN_MAX_S,
    )
    while True:
        runs = _active_runs if _time.time() < run_deadline else set()
        if _active_ws_count == 0 and not runs:
            logger.warning("[drain] All WS closed, no runs in flight. "
                           "Exiting.")
            break
        if _time.time() >= deadline:
            logger.warning(
                "[drain] Timeout (%ds). Exiting with %d active WS and "
                "%d run(s) still in flight: %s",
                timeout_s, _active_ws_count, len(_active_runs),
                sorted(_active_runs),
            )
            break
        if runs:
            # Named, so a deploy held open by a wedged run says which.
            logger.info("[drain] waiting on %d run(s): %s",
                        len(runs), sorted(runs))
        await asyncio.sleep(1)
    # SIGTERM rather than os._exit so the lifespan teardown runs.
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except ProcessLookupError:
        pass
