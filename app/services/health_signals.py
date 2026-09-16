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
    # ── Pushed gauges (set_gauge), owned by the loop/db sampler ──
    # 1 = the last DB probe succeeded, 0 = it FAILED. Seeded to 1 so "never
    # measured" reads as the benign default; the authoritative freshness is
    # `deps.db.checked_at` on /agent/health, not this number. It exists
    # because on 2026-09-15 /agent/health answered 200 for eleven minutes
    # while every DB-backed route 500'd.
    "db_reachable",
    # Max event-loop lag in ms observed over the last 30 s, pushed by the
    # loop sampler. 0 = healthy or not yet measured.
    "loop_lag_ms_max_30s",
    # ── Pull gauge, registered by app/db/database.py ──
    # Connections currently checked out of the agent's BOUNDED pool (0 on
    # NullPool, where the number has no meaning). Saturation of that pool
    # otherwise surfaces only as a `pool_timeout` wait and then a 503
    # `backend_unavailable`, with nothing naming the cause.
    "db_pool_checked_out",
    # R44 — one pair per LLM CALL (not per turn; a tool turn makes several).
    # A "hit" here only means the provider read SOME cached prefix, which in
    # the 2026-09-14 sample was always exactly the tools+instructions head:
    # read them with `cached_beyond_head` on the [PERF] llm_total line, which
    # is the half that says whether the day's history is cached at all.
    "llm_cache_hits",
    "llm_cache_misses",
    # Prompt-cache head warms issued on socket attach (app/agent/cache_warm).
    # A warm that never lands still counts here — read it next to
    # `cached=` on the [PERF] cache_warm line, which says whether it did.
    "llm_cache_warms",
    # The provider refused the `allowed_tools` tool_choice shape (a 400 on
    # `tool_choice.type`) and the runner retried the same call unrestricted.
    # Observed on the 2026-09-15 rollout canary; a non-zero count on a tenant
    # means intent gating is off for it and the retry cost one round-trip.
    "llm_tool_choice_rejected",
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
        for name, g in _GAUGES.items():
            # A pushed gauge whose 0 MEANS something (db_reachable = "the last
            # probe failed") must come back to its seed, not to the alarm.
            g["value"] = _GAUGE_SEEDS.get(name, 0)
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


def set_gauge(name: str, value: int) -> None:
    """Record a value that was MEASURED ELSEWHERE. Never raises.

    For a sampler that already owns the measurement (the event-loop lag
    probe, the cached DB reachability probe) — `register_gauge` would make
    this module pull, and both of those must be pushed on the sampler's own
    cadence, never on a health request. A pushed gauge has no `fn` and an
    infinite TTL, so `refresh_gauges()` never considers it due.
    """
    try:
        with _LOCK:
            prev = _GAUGES.get(name)
            if prev is not None and prev.get("fn") is not None:
                # A pull gauge already owns this name; don't silently convert
                # it into a push gauge and strip its refresher.
                prev["value"] = int(value)
                return
            _GAUGES[name] = {
                "fn": None,
                "ttl": float("inf"),
                "value": int(value),
                "at": time.monotonic(),
            }
    except Exception:  # pragma: no cover - defensive; a gauge may never break a caller
        pass


# `db_reachable` means "the last probe FAILED" only when it is 0, so it is
# seeded truthfully before the sampler's first tick (see KNOWN_SIGNALS), and
# `reset_for_tests()` restores the seed rather than the alarm.
_GAUGE_SEEDS: Dict[str, int] = {"db_reachable": 1}
for _n, _v in _GAUGE_SEEDS.items():
    set_gauge(_n, _v)


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
            if g and g.get("fn") is None:
                continue  # pushed gauge (set_gauge) — nothing to pull, even under force
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
