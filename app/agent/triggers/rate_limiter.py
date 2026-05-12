"""Per-trigger rate limit + coalescing window.

Two concerns one module covers because they share state:

  **Rate limit** — protects users (and our LLM bill) from a
  pathological email burst. If a mailing list mistakenly fans out
  500 messages in a minute, the user's trigger doesn't produce 500
  LLM summaries. Token bucket: max N fires per hour AND max M fires
  per minute. Defaults: 30/hour, 5/minute.

  **Coalescing window** — when an event arrives within
  `coalesce_window_sec` of another that's still in-flight for the
  same trigger, we fold the incoming event into the parent's batch
  rather than running a second handler. The parent picks up siblings
  before finalising. Default 120 s. Makes "5 emails in 2 minutes"
  produce ONE digest, not five.

State is in-memory + DB-derived. On container restart, rate buckets
re-warm from `trigger_events` rows in the recent window (count fires
in the last hour); in-flight coalescing state is lost (any events
mid-flight got swept by the restart sweep and re-queue from scratch
— acceptable, the second handler run is cheap and dedupes via the
DB-side UNIQUE).

Thread-safety: the runner is single-threaded asyncio. All state
manipulation happens on the loop thread. No locks needed.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _now() -> float:
    """Monotonic wall-clock seconds. Indirection so tests can monkey-patch
    a `_FakeClock` without touching stdlib `time.time` globally."""
    return time.time()


# ── Defaults ─────────────────────────────────────────────────────────


DEFAULT_PER_HOUR = 30
DEFAULT_PER_MINUTE = 5
DEFAULT_COALESCE_WINDOW_SEC = 120


def parse_rate_limit_config(raw: Optional[dict]) -> "RateLimitConfig":
    """Read the rate_limit sub-dict from `Trigger.config_json` and fill
    in defaults for missing fields. Negative / silly values fall back
    to defaults — we never want a misconfigured trigger to disable the
    safety net."""
    raw = raw or {}
    per_hour = _coerce_positive_int(raw.get("per_hour"), DEFAULT_PER_HOUR)
    per_min = _coerce_positive_int(raw.get("per_minute"), DEFAULT_PER_MINUTE)
    coalesce = _coerce_positive_int(
        raw.get("coalesce_window_sec"), DEFAULT_COALESCE_WINDOW_SEC,
    )
    return RateLimitConfig(per_hour=per_hour, per_minute=per_min,
                           coalesce_window_sec=coalesce)


def _coerce_positive_int(value, default: int) -> int:
    try:
        i = int(value)
        return i if i > 0 else default
    except (TypeError, ValueError):
        return default


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class RateLimitConfig:
    per_hour: int = DEFAULT_PER_HOUR
    per_minute: int = DEFAULT_PER_MINUTE
    coalesce_window_sec: int = DEFAULT_COALESCE_WINDOW_SEC


@dataclass
class TriggerBucket:
    """Per-trigger rate + coalesce state.

    `fire_timestamps` is a deque of unix-seconds for each handler
    *start* in the rate-limit window. We trim entries older than 1 h
    on every check. Cheap — bounded by per_hour (default 30).

    `in_flight_started_at` is unix-seconds when the currently-running
    handler started, or None when idle. Set by `acquire`; cleared by
    `release`.

    `coalesced_event_ids` accumulates trigger_event.id values that
    arrived while in-flight and were folded into the parent. The
    runner reads + clears this when the parent handler is about to
    finalise so siblings can be marked status='coalesced' atomically.
    """

    fire_timestamps: deque[float] = field(default_factory=deque)
    in_flight_started_at: Optional[float] = None
    coalesced_event_ids: list[str] = field(default_factory=list)


# ── Limiter ──────────────────────────────────────────────────────────


class TriggerRateLimiter:
    """One instance per `TriggerRunner` (= per agent container).

    API is shaped around the runner's event-handling loop:

      gate = limiter.gate(trigger_id, event_id, config)
      if gate.action == "fire":
          ... claim event, run handler ...
          limiter.release(trigger_id, list_of_event_ids_handled)
      elif gate.action == "coalesce":
          ... mark event as coalesced_into=gate.parent_event_id ...
      elif gate.action == "rate_limit":
          ... mark event as skipped_rate_limit ...
    """

    def __init__(self):
        self._buckets: dict[str, TriggerBucket] = {}

    # ── State helpers ────────────────────────────────────────────

    def _get_or_create(self, trigger_id: str) -> TriggerBucket:
        b = self._buckets.get(trigger_id)
        if b is None:
            b = TriggerBucket()
            self._buckets[trigger_id] = b
        return b

    def warmup(self, trigger_id: str, recent_fire_times: list[float]) -> None:
        """Seed the rate bucket from a restart-time SELECT of recent
        `trigger_events.started_at` values. Caller does the query on
        agent boot so a hot rate-limited trigger doesn't reset its
        budget when the container recycles."""
        b = self._get_or_create(trigger_id)
        now = _now()
        # Keep only timestamps within the last hour.
        b.fire_timestamps = deque(
            sorted(t for t in recent_fire_times if (now - t) <= 3600)
        )

    # ── Gate decision ────────────────────────────────────────────

    def gate(
        self,
        trigger_id: str,
        event_id: str,
        config: RateLimitConfig,
    ) -> "GateDecision":
        """Decide what to do with an incoming event.

        Three outcomes:
          - **coalesce**: another event for this trigger is in-flight
            within the coalesce window — fold into it. Returns the
            parent event id; caller marks this event as `coalesced`.
          - **rate_limit**: rate budget exhausted in either window.
            Caller marks the event `skipped_rate_limit`.
          - **fire**: handler should run. Caller claims the event,
            invokes the handler, then calls `release`. The bucket's
            `in_flight_started_at` is set and the fire timestamp is
            appended.

        Coalesce takes precedence over rate-limit so a burst is
        always collapsed into one handler call (which itself counts
        as one fire against the rate limit, not N).
        """
        b = self._get_or_create(trigger_id)
        now = _now()

        # ── Coalesce check ──
        if b.in_flight_started_at is not None:
            elapsed = now - b.in_flight_started_at
            if elapsed <= config.coalesce_window_sec:
                # The parent is whichever event actually owns the
                # in-flight handler — recorded as `_parent_event_id`
                # on the bucket when that event fired. Every coalesced
                # sibling points to THAT id (not at preceding siblings),
                # so the handler's batch is unambiguous.
                parent = getattr(b, "_parent_event_id", None) or "<unknown-parent>"
                b.coalesced_event_ids.append(event_id)
                return GateDecision(
                    action="coalesce",
                    parent_event_id=parent,
                )

        # ── Rate limit check ──
        # Trim fires older than 1 h.
        while b.fire_timestamps and (now - b.fire_timestamps[0]) > 3600:
            b.fire_timestamps.popleft()
        in_last_hour = len(b.fire_timestamps)
        in_last_minute = sum(1 for t in b.fire_timestamps if (now - t) <= 60)
        if in_last_hour >= config.per_hour:
            return GateDecision(
                action="rate_limit",
                reason="per_hour",
                fires_in_window=in_last_hour,
            )
        if in_last_minute >= config.per_minute:
            return GateDecision(
                action="rate_limit",
                reason="per_minute",
                fires_in_window=in_last_minute,
            )

        # ── Fire ──
        b.fire_timestamps.append(now)
        b.in_flight_started_at = now
        # Remember parent for any sibling that arrives during this
        # handler run.
        b._parent_event_id = event_id  # type: ignore[attr-defined]
        return GateDecision(action="fire")

    def acquire(self, trigger_id: str, parent_event_id: str) -> None:
        """Mark a handler as in-flight (called by the runner just
        before invoking `handler.execute`). The `gate()` path already
        does this; `acquire` is exposed for the test-trigger code
        path that bypasses gate()."""
        b = self._get_or_create(trigger_id)
        if b.in_flight_started_at is None:
            b.in_flight_started_at = _now()
            b._parent_event_id = parent_event_id  # type: ignore[attr-defined]

    def drain_coalesced(self, trigger_id: str) -> list[str]:
        """Return + clear the list of event ids that coalesced into
        the in-flight handler. Called by the runner right before
        passing the batch to `handler.execute`."""
        b = self._get_or_create(trigger_id)
        siblings = list(b.coalesced_event_ids)
        b.coalesced_event_ids.clear()
        return siblings

    def release(self, trigger_id: str) -> None:
        """Mark the handler complete. Caller is the runner, after
        `handler.execute` returns (whether success or failure)."""
        b = self._get_or_create(trigger_id)
        b.in_flight_started_at = None
        b._parent_event_id = None  # type: ignore[attr-defined]
        # coalesced_event_ids is drained separately right before
        # the handler runs; if not, we drop here as a safety net.
        b.coalesced_event_ids.clear()


@dataclass
class GateDecision:
    action: str                                 # "fire" | "coalesce" | "rate_limit"
    parent_event_id: Optional[str] = None       # set when action=coalesce
    reason: Optional[str] = None                # "per_hour" | "per_minute"
    fires_in_window: int = 0
