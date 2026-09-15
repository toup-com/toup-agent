"""Process-lifetime counters exposed on `/agent/health.health_signals`.

Law 1 of the monitoring contract: NOTHING may gate on these. They are
diagnostic only — `container_monitor` folds them into alerts on the
platform, and an old agent image simply omits the key.

Counts only. The endpoint is unauthenticated by design (see the
`init_error_class` redaction in `agent_main`), so no ids, no phone
numbers, no message text may ever enter this module.

Thread-safe because the agent runs uvicorn workers that touch these from
both the event loop and `asyncio.to_thread` helpers; a lost increment is
harmless but a torn dict is not.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# The full signal set, so `snapshot()` always answers the same SHAPE.
# A reader that has to tell "this image does not report media_persisted"
# from "it reported zero" has no way to do it if absent keys mean both.
KNOWN_SIGNALS: tuple[str, ...] = (
    "future_dated_day_chats",
    "day_chats_5xx",
    "turns_completed",
    "assistant_rows_written",
    "media_expected",
    "media_persisted",
    "channel_events_claimed",
    "channel_events_deduped",
    "channel_events_persisted",
    "wa_sidecar_spawns",
    # The adapter attached to a sidecar it did not spawn (port owned by a
    # stranger). Not restartable from inside; the next config push respawns.
    "wa_sidecar_adopted",
    # The periodic sidecar re-read — not an SSE frame — changed the cached
    # WhatsApp session state. Every increment is one event the sidecar's
    # replay-less SSE stream emitted while no consumer was attached; a
    # tenant whose count climbs is losing `connection_open` frames.
    "wa_status_reconciled",
    "channel_tag_prefixed_replies",
    "rebucket_runs",
    "rebucket_failures",
    # A Message written with day_chat_id=NULL — invisible to the day index
    # and to load_day_context (message_helpers.resolve_day_chat_id_for_now).
    "day_chat_resolve_failures",
    # Monotonic partner of the `future_dated_day_chats` GAUGE: how many
    # impossible rows the day index has ever found before healing them. The
    # gauge reads the CURRENT state (zero again once repaired); this one
    # says a tenant produced such a row at all.
    "future_dated_day_chats_seen",
)

_LOCK = threading.Lock()
_COUNTERS: Dict[str, int] = {}

# name -> {"fn": callable, "ttl": float, "value": int, "at": float}
#
# `at` is the monotonic time of the last measurement; NEVER_MEASURED (-inf)
# marks a gauge that has not run yet, so it is due on the first refresh
# whatever the clock reads. It used to be 0.0, which on a host whose
# monotonic clock is younger than the TTL (a fresh CI runner, a just-booted
# VM) made a brand-new gauge "not due" until the machine had been up for a
# full TTL — measured as `assert 0 == 1` in CI and never locally.
NEVER_MEASURED = float("-inf")
#
# A gauge is a value that must be MEASURED rather than counted
# (`future_dated_day_chats` is a COUNT over the tenant's day_chats). It is
# refreshed lazily behind a TTL so the health endpoint never pays for it and
# a hot route pays for it at most once per TTL.
_GAUGES: Dict[str, dict] = {}


def incr(name: str, n: int = 1) -> None:
    """Add ``n`` to the counter ``name``. Never raises."""
    try:
        with _LOCK:
            _COUNTERS[name] = _COUNTERS.get(name, 0) + int(n)
    except Exception:  # pragma: no cover - defensive; a counter may never break a caller
        pass


def get(name: str) -> int:
    """Current value of ``name`` (counter or gauge), 0 when unknown."""
    with _LOCK:
        if name in _COUNTERS:
            return int(_COUNTERS[name])
        g = _GAUGES.get(name)
        return int(g["value"]) if g else 0


def snapshot() -> Dict[str, int]:
    """Every known signal as ints, gauges folded in. Stable key set."""
    with _LOCK:
        out: Dict[str, int] = {k: 0 for k in KNOWN_SIGNALS}
        out.update({k: int(v) for k, v in _COUNTERS.items()})
        for name, g in _GAUGES.items():
            out[name] = int(g["value"])
        return out


def reset_for_tests() -> None:
    """Zero counters and gauge values (registrations survive). Per-process by design."""
    with _LOCK:
        _COUNTERS.clear()
        # Zero the measurements, keep the registrations: a gauge is registered
        # at import time by the route that owns it, and a test that resets the
        # counters between cases must not silently lose the gauge for the rest
        # of the process.
        for g in _GAUGES.values():
            g["value"] = 0
            g["at"] = NEVER_MEASURED


def register_gauge(
    name: str,
    fn: Callable[[], Awaitable[int]],
    *,
    ttl_s: float = 60.0,
) -> None:
    """Register (or replace) a lazily-refreshed gauge.

    ``fn`` is an async zero-arg callable returning an int; it owns its own
    DB session. Registration alone measures nothing — a caller has to
    ``await refresh_gauges()``.
    """
    with _LOCK:
        prev = _GAUGES.get(name)
        _GAUGES[name] = {
            "fn": fn,
            "ttl": float(ttl_s),
            # Keep the last measured value across a re-registration so a
            # module reload cannot make a real count read as zero.
            "value": int(prev["value"]) if prev else 0,
            "at": float(prev["at"]) if prev else NEVER_MEASURED,
        }


def gauge_due(name: str) -> bool:
    """True when ``name``'s cached value has aged past its TTL."""
    with _LOCK:
        g = _GAUGES.get(name)
        if not g:
            return False
        return (time.monotonic() - g["at"]) >= g["ttl"]


async def refresh_gauges(*, only: Optional[str] = None, force: bool = False) -> None:
    """Re-measure every due gauge (or just ``only``). Never raises.

    A gauge that fails keeps its previous value and its previous timestamp
    is advanced anyway, so a broken measurement cannot turn into a
    per-request retry storm on a hot route. ``force`` ignores the TTL — for
    the caller that has just CHANGED what the gauge measures (a heal) and
    must not leave a stale positive on the wire for up to a TTL.
    """
    with _LOCK:
        names = [only] if only else list(_GAUGES.keys())
        due = []
        now = time.monotonic()
        for n in names:
            g = _GAUGES.get(n)
            if g and (force or (now - g["at"]) >= g["ttl"]):
                due.append((n, g["fn"]))
                g["at"] = now  # claim it before awaiting — one refresh at a time
    for name, fn in due:
        try:
            value = int(await fn())
        except Exception as e:
            logger.debug("[health_signals] gauge %s refresh failed: %r", name, e)
            continue
        with _LOCK:
            g = _GAUGES.get(name)
            if g is not None:
                g["value"] = value
