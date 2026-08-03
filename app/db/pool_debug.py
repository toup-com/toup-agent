"""Name the code path that abandons a DB connection.

WHY THIS EXISTS

`SAWarning: The garbage collector is trying to clean up non-checked-in
connection` says a connection was collected while still checked out — a
session that was never closed. It does NOT say where. The warning is emitted
by the garbage collector, so **the surrounding log lines are where the GC
ran, not where the leak happened**, and reading them has now sent three
separate fixes at the wrong target (#407, #408, #418 each removed real
defects and none of them drove the rate to zero).

Guessing has been tried. This records the truth instead: the stack at the
moment the connection was checked OUT, replayed if that connection is ever
collected without being checked back in.

HOW

  * pool `checkout`  -> store the caller's stack, keyed by the connection
  * pool `checkin`   -> drop it; this connection came home
  * connection GC'd  -> if the key is still there it was abandoned, so log
                        the stored checkout stack at ERROR

A `weakref.finalize` is used rather than parsing the SAWarning text: the
warning renders the inner asyncpg object while the pool hands us the adapter
wrapper, and matching them by address would be fragile in exactly the
situation where accuracy matters.

COST, and why it is flag-gated OFF

`traceback.format_stack()` on every checkout is real work on a hot path.
This is a diagnostic to switch on for one tenant while reproducing, not
something to leave running. `POOL_LEAK_DEBUG=true`.
"""

from __future__ import annotations

import logging
import threading
import traceback
import weakref
from typing import Dict

logger = logging.getLogger(__name__)

# How many frames of the checkout stack to keep. Deep enough to cross the
# session-factory plumbing and reach real application code.
_STACK_DEPTH = 25

# Ceiling on tracked checkouts, so a runaway cannot grow this without bound.
_MAX_TRACKED = 500

_lock = threading.Lock()
_origins: Dict[int, str] = {}
_stats = {"checkouts": 0, "checkins": 0, "abandoned": 0, "untracked_gc": 0}


def _format_origin() -> str:
    """The checkout stack, INCLUDING the async caller.

    `traceback.format_stack()` alone is not enough. SQLAlchemy's async layer
    runs the synchronous pool work inside a greenlet, so a stack taken here
    begins at `Session.execute` and the coroutine that actually opened the
    session — the thing we are trying to name — is not on it. The application
    frames live on the PARENT greenlet, so walk that chain too.
    """
    # Drop the last two frames: this function and the event handler.
    parts = traceback.format_stack(limit=_STACK_DEPTH)[:-2]

    try:
        import greenlet

        g = greenlet.getcurrent()
        depth = 0
        while g is not None and depth < 5:
            g = getattr(g, "parent", None)
            frame = getattr(g, "gr_frame", None) if g is not None else None
            if frame is None:
                depth += 1
                continue
            caller = traceback.format_list(
                traceback.extract_stack(frame, limit=_STACK_DEPTH)
            )
            if caller:
                parts = caller + ["  --- greenlet boundary ---\n"] + parts
            depth += 1
    except Exception:
        # The greenlet walk is a bonus; never let it cost us the base stack.
        pass

    return "".join(parts)


def _on_abandoned(key: int) -> None:
    """Fires when a connection object is collected."""
    with _lock:
        origin = _origins.pop(key, None)
        if origin is None:
            # Checked in normally, then collected. The healthy path.
            _stats["untracked_gc"] += 1
            return
        _stats["abandoned"] += 1
        n = _stats["abandoned"]
    logger.error(
        "[pool-leak] connection #%d was garbage-collected WITHOUT being "
        "checked in. It was checked out here:\n%s",
        n, origin,
    )


SENTINEL = "/app/workspace/.pool_leak_debug"


def should_enable(settings_flag: bool) -> bool:
    """True when the instrument should be armed for this container.

    `POOL_LEAK_DEBUG=true` works, but tenant environments are built by the
    bridge and are append-only across a blue-green upgrade, so arming ONE
    tenant by env is a deploy rather than a switch. A workspace sentinel makes
    it a `docker exec touch` + restart — which is what you want for a
    diagnostic you turn on to reproduce and off again straight after.

    Same pattern the blue-green promote marker already uses
    (`/app/workspace/.toup_bg_promoted`).
    """
    if settings_flag:
        return True
    try:
        import os
        return os.path.exists(SENTINEL)
    except Exception:
        return False


def install(engine) -> bool:
    """Attach the listeners. Returns False if it could not be installed."""
    try:
        from sqlalchemy import event

        pool = engine.sync_engine.pool

        @event.listens_for(pool, "checkout")
        def _checkout(dbapi_con, con_record, con_proxy):  # noqa: ANN001
            # Key on the connection RECORD, not the DBAPI connection: the
            # driver adapters use __slots__ without __weakref__
            # (`TypeError: cannot create weak reference to
            # 'AsyncAdapt_aiosqlite_connection'`), and the record is the one
            # object present in both the checkout and checkin events.
            key = id(con_record)
            with _lock:
                _stats["checkouts"] += 1
                if len(_origins) >= _MAX_TRACKED:
                    return
                _origins[key] = _format_origin()
            try:
                # Must not close over con_record, or the finalizer would keep
                # it alive and the abandonment could never be observed.
                weakref.finalize(con_record, _on_abandoned, key)
            except TypeError:
                with _lock:
                    _origins.pop(key, None)
                    _stats["untrackable"] = _stats.get("untrackable", 0) + 1

        @event.listens_for(pool, "checkin")
        def _checkin(dbapi_con, con_record):  # noqa: ANN001
            with _lock:
                _stats["checkins"] += 1
                _origins.pop(id(con_record), None)

        logger.warning(
            "[pool-leak] origin tracking ENABLED (depth=%d, max_tracked=%d). "
            "This formats a stack on every checkout — diagnostic only.",
            _STACK_DEPTH, _MAX_TRACKED,
        )
        return True
    except Exception as exc:
        logger.warning("[pool-leak] could not install origin tracking: %s", exc)
        return False


def stats() -> dict:
    """Counters for /agent/health or a probe."""
    with _lock:
        return dict(_stats, outstanding=len(_origins))
