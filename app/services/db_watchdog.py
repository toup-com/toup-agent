"""DB-layer watchdog — detects and cures a poisoned engine in-process.

Born from the 2026-07-04 incident (tenant 3134fece): the agent's
SQLAlchemy state was stuck in endless PendingRollbackError /
BEGIN-ROLLBACK churn — likely since a blue-green cutover interrupted a
transaction days earlier. Every DB touch failed ("Connection lost" on
every chat turn) while /agent/health stayed green, and the only cure
was a manual `docker restart` three days later.

The watchdog closes that class:

- Every WATCHDOG_INTERVAL_S it opens a FRESH session and runs SELECT 1.
- Two consecutive failures → `database.recover_engine()` (rebuild the
  engine on the same URL + swap the holder — the DB-layer effect of a
  container restart, without one) → immediate re-probe.
- State is exported via `watchdog_state()` for /agent/health (`db_ok`,
  `db_recoveries`) so a broken-DB agent is VISIBLE, never silently
  green (law 2: healthy must mean a user can actually chat). The field
  is diagnostic only — nothing gates on it (law 1).

Runs in both the agent (its tenant DB) and the platform (Supabase
pooler); the probe is one round-trip per minute per process.
"""

import asyncio
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

WATCHDOG_INTERVAL_S = 60
PROBE_TIMEOUT_S = 15
FAILURES_BEFORE_RECOVERY = 2

_state: Dict[str, Any] = {
    "db_ok": True,           # optimistic until the first probe says otherwise
    "consecutive_failures": 0,
    "recoveries": 0,
    "last_ok_at": None,      # epoch seconds
    "last_error": None,
}


def watchdog_state() -> Dict[str, Any]:
    """Snapshot for /agent/health (and tests)."""
    return dict(_state)


async def _probe_once() -> None:
    from sqlalchemy import text
    from app.db.database import async_session_maker

    async def _run() -> None:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))

    await asyncio.wait_for(_run(), timeout=PROBE_TIMEOUT_S)


async def check_and_heal() -> bool:
    """One watchdog step: probe; on repeated failure, recover the engine.

    Returns the post-step db_ok. Factored out of the loop so tests (and
    any caller that just observed a suspicious DB error) can drive it
    directly.
    """
    try:
        await _probe_once()
        _state["db_ok"] = True
        _state["consecutive_failures"] = 0
        _state["last_ok_at"] = time.time()
        _state["last_error"] = None
        return True
    except Exception as e:
        _state["consecutive_failures"] += 1
        _state["db_ok"] = False
        _state["last_error"] = f"{type(e).__name__}: {e}"[:300]
        logger.warning(
            "[db-watchdog] probe failed (%d consecutive): %s",
            _state["consecutive_failures"], _state["last_error"],
        )

    if _state["consecutive_failures"] < FAILURES_BEFORE_RECOVERY:
        return False

    # Recovery: rebuild the engine on the same URL. recover_engine
    # smoke-tests the new engine BEFORE swapping, so a genuinely-down
    # database keeps the old engine and we simply retry next tick.
    try:
        from app.db.database import recover_engine
        await recover_engine()
        _state["recoveries"] += 1
        logger.warning(
            "[db-watchdog] engine recovered (recovery #%d) — re-probing",
            _state["recoveries"],
        )
    except Exception as e:
        logger.warning("[db-watchdog] recovery failed (DB down?): %s", e)
        return False

    try:
        await _probe_once()
        _state["db_ok"] = True
        _state["consecutive_failures"] = 0
        _state["last_ok_at"] = time.time()
        _state["last_error"] = None
        return True
    except Exception as e:
        _state["last_error"] = f"{type(e).__name__}: {e}"[:300]
        logger.warning("[db-watchdog] still failing after recovery: %s", e)
        return False


async def db_watchdog_loop() -> None:
    """Forever loop; start via asyncio.create_task in the lifespan."""
    logger.info("[db-watchdog] started (interval=%ss)", WATCHDOG_INTERVAL_S)
    while True:
        try:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            await check_and_heal()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The loop itself must never die — it's the safety net.
            logger.exception("[db-watchdog] tick crashed; continuing")
