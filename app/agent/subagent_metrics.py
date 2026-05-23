"""Sub-agent metrics — Phase 6 of the sub-agent spawning arc.

Reuses the in-process counter+histogram primitives from
``app/services/connector_metrics.py`` (Prometheus text-format
exposition via the existing ``/metrics`` endpoint). No new
dependency. No new exposition layer.

Counters
--------
- ``subagent_spawn_attempted_total{channel,parent_depth}`` — every
  call to ``_tool_spawn`` (including those that get rejected by
  pre_spawn_checks). ``parent_depth`` is the depth-0 root (the
  user's main turn) or depth-1 (a nested spawn).
- ``subagent_spawn_rejected_total{reason}`` — one label value per
  ``REJECTION_CODES`` entry. Operator dashboards graph the
  per-reason breakdown to spot rate-limit thrash.
- ``subagent_lane_acquired_total`` — paired with
  ``subagent_lane_released_total{outcome}`` so the implicit gauge
  (acquired − released) gives per-process concurrent count.

Histograms
----------
- ``subagent_run_duration_ms{outcome}`` — wall-clock from
  acquire-slot to release-slot. One observation per terminal
  transition.
- ``subagent_run_credit_spent_thousandths{outcome}`` — credit
  spent per run, in 1/1000 of a credit (the histogram buckets are
  integer ms by default; we scale credits → thousandths to land on
  the same bucket grid).

Why not ``user_id`` labels
--------------------------
The spec called for ``{user_id, ...}`` labels on the spawn-attempted
counter. Cardinality-unbounded labels (one per active user) make
the /metrics page explode O(N_users × N_other_dimensions). We log
``user_id`` in the structured-log line for per-user attribution and
keep the metric labels to fixed enumerations. Operators who need
per-user analysis pivot on logs (BigQuery / Loki / etc.), not
metrics.

Runaway detection
-----------------
``record_spawn_attempt(user_id)`` tracks per-user spawn timestamps
in a rolling 5-minute window. When > 10 spawns land in 5 minutes,
emits a dedicated structured-log event ``subagent.runaway_detected``
that an operator can wire an alert against. (The alert rule
itself is out of scope per the spec; we emit the signal.)
"""
from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Optional


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Metric names (constants so a typo lights up here, not in
# production where the dashboard silently shows nothing).
# ──────────────────────────────────────────────────────────────────────


M_SPAWN_ATTEMPTED = "subagent_spawn_attempted_total"
M_SPAWN_REJECTED = "subagent_spawn_rejected_total"
M_LANE_ACQUIRED = "subagent_lane_acquired_total"
M_LANE_RELEASED = "subagent_lane_released_total"
M_RUN_DURATION_MS = "subagent_run_duration_ms"
M_RUN_CREDIT_SPENT_THOUSANDTHS = "subagent_run_credit_spent_thousandths"


# ──────────────────────────────────────────────────────────────────────
# Counter helpers
# ──────────────────────────────────────────────────────────────────────


def record_spawn_attempted(*, channel: str, parent_depth: int) -> None:
    """One spawn attempt — independent of whether it was accepted."""
    from app.services.connector_metrics import inc

    inc(M_SPAWN_ATTEMPTED, labels={
        "channel": channel or "unknown",
        "parent_depth": str(parent_depth),
    })


def record_spawn_rejected(*, reason: str) -> None:
    """One rejection. ``reason`` MUST be one of REJECTION_CODES."""
    from app.services.connector_metrics import inc

    inc(M_SPAWN_REJECTED, labels={"reason": reason or "unknown"})


def record_lane_acquired() -> None:
    from app.services.connector_metrics import inc
    inc(M_LANE_ACQUIRED)


def record_lane_released(*, outcome: str) -> None:
    """Outcome ∈ {success, failed, timeout, cancelled, budget_exhausted}."""
    from app.services.connector_metrics import inc
    inc(M_LANE_RELEASED, labels={"outcome": outcome or "unknown"})


def record_run_duration(*, duration_ms: float, outcome: str) -> None:
    from app.services.connector_metrics import observe_ms

    observe_ms(M_RUN_DURATION_MS, duration_ms, labels={
        "outcome": outcome or "unknown",
    })


def record_credit_spent(*, credit: float, outcome: str) -> None:
    """Histogram on credit spend per run. The connector_metrics
    histogram buckets are millisecond-integer; we encode credits
    as thousandths (0.001 credits = 1 bucket-unit) so 0.001 lands
    in the 1ms bucket, 1.0 credits in the 1000ms bucket, etc."""
    from app.services.connector_metrics import observe_ms

    if credit is None or credit <= 0:
        return
    observe_ms(
        M_RUN_CREDIT_SPENT_THOUSANDTHS,
        credit * 1000.0,
        labels={"outcome": outcome or "unknown"},
    )


# ──────────────────────────────────────────────────────────────────────
# Runaway detection — per-user 5-minute rolling window
# ──────────────────────────────────────────────────────────────────────


_RUNAWAY_WINDOW_SECONDS = 5 * 60
_RUNAWAY_THRESHOLD = 10

_attempts: dict[str, deque] = {}
_attempts_lock = Lock()


def record_spawn_attempt(user_id: str) -> bool:
    """Record a per-user spawn attempt. When the rolling 5-min
    window exceeds ``_RUNAWAY_THRESHOLD``, emits one
    ``subagent.runaway_detected`` log event and returns True.
    Otherwise returns False.

    The window is held in-process — process restart resets it.
    That's fine: a runaway burst that spans a restart is by
    definition no longer a burst.

    Thread-safe via a single lock around the deque mutation.
    """
    now = time.time()
    cutoff = now - _RUNAWAY_WINDOW_SECONDS
    runaway = False

    with _attempts_lock:
        bucket = _attempts.setdefault(user_id, deque())
        # Drop timestamps that have aged out of the window.
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        bucket.append(now)
        count = len(bucket)
        runaway = count > _RUNAWAY_THRESHOLD

    if runaway:
        # One log line — operators wire an alert rule against this
        # exact string. Match shape: structured key=value pairs so
        # ingest pipelines (Loki / Datadog logs / etc.) split cleanly.
        logger.warning(
            "subagent.runaway_detected user_id=%s spawns_in_5min=%d "
            "threshold=%d",
            user_id, count, _RUNAWAY_THRESHOLD,
        )
    return runaway


def reset_runaway_state_for_tests() -> None:
    with _attempts_lock:
        _attempts.clear()


# ──────────────────────────────────────────────────────────────────────
# Structured log helper — used at every state transition so an
# operator can grep one prefix and see the full lifecycle
# ──────────────────────────────────────────────────────────────────────


def log_lifecycle(
    *,
    event: str,
    user_id: str,
    job_id: str,
    parent_job_id: Optional[str] = None,
    depth: Optional[int] = None,
    outcome: Optional[str] = None,
    duration_ms: Optional[float] = None,
    credit_spent: Optional[float] = None,
    reason: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Single structured log line per lifecycle event. Stable
    key=value shape so a log ingest pipeline can split cleanly.

    ``event`` ∈ {attempted, accepted, rejected, started, completed,
                  failed, timeout, cancelled, budget_exhausted,
                  orphan_swept, runaway_detected}.
    """
    parts = [
        f"event={event}",
        f"user_id={user_id}",
        f"job_id={job_id}",
    ]
    if parent_job_id is not None:
        parts.append(f"parent_job_id={parent_job_id}")
    if depth is not None:
        parts.append(f"depth={depth}")
    if outcome is not None:
        parts.append(f"outcome={outcome}")
    if duration_ms is not None:
        parts.append(f"duration_ms={duration_ms:.1f}")
    if credit_spent is not None:
        parts.append(f"credit_spent={credit_spent:.4f}")
    if reason is not None:
        parts.append(f"reason={reason}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")

    logger.info("subagent.%s %s", event, " ".join(parts))
