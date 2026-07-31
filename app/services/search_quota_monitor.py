"""Search-quota monitor — so Brave cannot run out, or fall over, silently.

Why the monthly alarm cannot be built on Brave's headers
────────────────────────────────────────────────────────
Brave returns two rate-limit buckets on every response:

    x-ratelimit-limit:     50, 0
    x-ratelimit-policy:    50;w=1, 0;w=2678400
    x-ratelimit-remaining: 49, 0

The FIRST bucket is the per-second account ceiling. It is real, it is
fleet-wide, and ``search_proxy._FleetGuard`` already sheds traffic against it.
The SECOND is the ~31-day window (``w=2678400``) and on this plan it reports
**limit 0, remaining 0** — Brave does not publish remaining monthly credits in
any form we can act on. There is no other endpoint that exposes them either.

So monthly-quota alerting cannot read the answer off the wire. It has to be
*counted on our side*, from ``search_events`` — the row the gateway writes for
every call it makes — against a budget the operator states in config. That is
the whole reason this module exists rather than a header parse in
``_FleetGuard``, and it is why ``brave_monthly_budget`` is an operator input
and not a discovered fact.

What it watches (each its own alert category, so they rate-limit apart)
──────────────────────────────────────────────────────────────────────
* ``search-quota-monthly``  — calls that reached Brave this calendar month vs
  the configured budget, with a month-end projection in the message.
* ``search-throttle-rate``  — sustained share of calls shed to slower tiers.
* ``search-fleet-headroom`` — how often Brave reported us near the shared
  per-second ceiling.
* ``search-blackout``       — searches were attempted and NONE were served.
  This is the "Brave silently disconnected" signature the whole investigation
  started from.
* ``search-unconfigured``   — no platform Brave key at all. Needs its own
  check because the gateway returns before writing a row in that state, so
  ``search_events`` is *empty*, not wrong: no aggregate can ever see it.

Runs as a bare ``asyncio`` loop started in ``platform_main`` lifespan, not on
APScheduler — the scheduler does not run on this deployment (it silently
killed all notification dispatch once; see the note beside
``notification_dispatch_loop`` there).
"""
from __future__ import annotations

import asyncio
import calendar
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import SearchEvent
from app.services.alerting import send_infra_alert

logger = logging.getLogger(__name__)

CAT_MONTHLY = "search-quota-monthly"
CAT_THROTTLE = "search-throttle-rate"
CAT_HEADROOM = "search-fleet-headroom"
CAT_BLACKOUT = "search-blackout"
CAT_UNCONFIGURED = "search-unconfigured"

# platform-api runs more than one replica and alerting.py rate-limits
# per-process, so an alert can arrive once per replica. The intervals below are
# sized for "a few a day, worst case", not for one.
MIN_INTERVAL_MONTHLY = 21600     # 6h
MIN_INTERVAL_THROTTLE = 10800    # 3h
MIN_INTERVAL_HEADROOM = 21600    # 6h
MIN_INTERVAL_BLACKOUT = 3600     # every tick at the default cadence
MIN_INTERVAL_UNCONFIGURED = 21600

# Degrade reasons recorded for calls the gateway shed BEFORE any HTTP request
# left the process. They cost a user latency but cost Brave nothing, so they
# must not count against the monthly budget.
_NEVER_REACHED_BRAVE = ("tenant_rate_limit", "fleet_headroom", "cooldown_after_429")


def _cfg(name: str, default):
    """Read a threshold from settings, tolerating its absence.

    The config keys this module wants are listed in its ticket; until they are
    added the defaults here are the live values, so quota alerting works on a
    deploy that changed nothing else.
    """
    value = getattr(settings, name, None)
    return default if value is None else value


@dataclass
class _Window:
    total: int
    ok: int
    throttled: int
    errored: int
    headroom_samples: int
    low_headroom: int
    min_remaining: Optional[int]
    avg_remaining: Optional[float]


async def _window_stats(db: AsyncSession, since: datetime, floor: int) -> _Window:
    # The status vocabulary is owned by the writer; import it rather than
    # re-spelling it here, so a rename there breaks loudly instead of silently
    # zeroing every rate below. Lazy because app.api.__init__ pulls six routers.
    from app.api.search_proxy import ST_ERROR, ST_OK, ST_THROTTLED

    row = (
        await db.execute(
            select(
                func.count().label("total"),
                func.sum(case((SearchEvent.status == ST_OK, 1), else_=0)).label("ok"),
                func.sum(case((SearchEvent.status == ST_THROTTLED, 1), else_=0)).label("throttled"),
                func.sum(case((SearchEvent.status == ST_ERROR, 1), else_=0)).label("errored"),
                func.count(SearchEvent.brave_remaining).label("headroom_samples"),
                func.sum(case((SearchEvent.brave_remaining <= floor, 1), else_=0)).label("low_headroom"),
                func.min(SearchEvent.brave_remaining).label("min_remaining"),
                func.avg(SearchEvent.brave_remaining).label("avg_remaining"),
            ).where(SearchEvent.created_at >= since)
        )
    ).one()

    return _Window(
        total=int(row.total or 0),
        ok=int(row.ok or 0),
        throttled=int(row.throttled or 0),
        errored=int(row.errored or 0),
        headroom_samples=int(row.headroom_samples or 0),
        low_headroom=int(row.low_headroom or 0),
        min_remaining=int(row.min_remaining) if row.min_remaining is not None else None,
        avg_remaining=float(row.avg_remaining) if row.avg_remaining is not None else None,
    )


async def _brave_calls_since(db: AsyncSession, since: datetime) -> int:
    """Count calls that actually consumed Brave quota."""
    # NOT IN drops NULLs in SQL, and a served call has degraded_reason NULL —
    # which is the majority of the table. The IS NULL arm is load-bearing.
    result = await db.execute(
        select(func.count()).where(
            SearchEvent.created_at >= since,
            or_(
                SearchEvent.degraded_reason.is_(None),
                SearchEvent.degraded_reason.notin_(_NEVER_REACHED_BRAVE),
            ),
        )
    )
    return int(result.scalar() or 0)


async def _top_reasons(db: AsyncSession, since: datetime, limit: int = 3) -> str:
    rows = (
        await db.execute(
            select(SearchEvent.degraded_reason, func.count().label("n"))
            .where(
                SearchEvent.created_at >= since,
                SearchEvent.degraded_reason.isnot(None),
            )
            .group_by(SearchEvent.degraded_reason)
            .order_by(func.count().desc())
            .limit(limit)
        )
    ).all()
    return ", ".join(f"{reason}={n}" for reason, n in rows) or "none"


async def check_search_quota() -> dict:
    """One monitor pass. Returns the snapshot, with the categories it fired.

    Factored out of the loop so a test — or an operator in a shell — can drive
    it directly.
    """
    from app.db.database import async_session_maker

    fired: list[str] = []

    # An empty key never reaches the aggregate: search_proxy returns
    # `unconfigured` before it records anything, on purpose (a platform
    # misconfig is not tenant usage). Reading the setting is the only way to
    # tell "nobody searched" from "nobody could".
    if not (getattr(settings, "brave_api_key", "") or "").strip():
        await send_infra_alert(
            CAT_UNCONFIGURED, "critical",
            "Search gateway has no Brave key configured (brave_api_key is empty). "
            "Every tenant is falling back to slower local tiers, and the gateway "
            "writes no telemetry row in this state, so no dashboard can show it.",
            min_interval_s=MIN_INTERVAL_UNCONFIGURED,
        )
        fired.append(CAT_UNCONFIGURED)

    window_h = int(_cfg("search_alert_window_h", 3))
    min_samples = int(_cfg("search_alert_min_samples", 20))
    floor = int(_cfg("brave_fleet_floor", 5))

    # search_events.created_at is written with datetime.utcnow() (naive UTC) by
    # the gateway; the boundaries must be built the same way or every window
    # silently shifts by the host offset.
    now = datetime.utcnow()
    since = now - timedelta(hours=window_h)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    async with async_session_maker() as db:
        window = await _window_stats(db, since, floor)
        month_calls = await _brave_calls_since(db, month_start)
        reasons = await _top_reasons(db, since) if window.total else "none"

    logger.info(
        "[search-quota] window=%dh total=%d ok=%d throttled=%d error=%d "
        "headroom(min=%s avg=%s low=%d/%d) month_calls=%d",
        window_h, window.total, window.ok, window.throttled, window.errored,
        window.min_remaining, window.avg_remaining,
        window.low_headroom, window.headroom_samples, month_calls,
    )

    # (a) Monthly budget. Our count, not Brave's — see the module docstring.
    budget = int(_cfg("brave_monthly_budget", 2000))
    used_pct = (100.0 * month_calls / budget) if budget > 0 else 0.0
    if budget > 0:
        elapsed_s = max(1.0, (now - month_start).total_seconds())
        month_s = calendar.monthrange(now.year, now.month)[1] * 86400.0
        projected = int(month_calls * month_s / elapsed_s)
        level = None
        if used_pct >= float(_cfg("brave_monthly_critical_pct", 95.0)):
            level = "critical"
        elif used_pct >= float(_cfg("brave_monthly_warn_pct", 80.0)):
            level = "warning"
        if level:
            await send_infra_alert(
                CAT_MONTHLY, level,
                f"Brave monthly usage {month_calls}/{budget} calls "
                f"({used_pct:.0f}% of budget) since {month_start:%Y-%m-%d}, "
                f"projecting {projected} by month end. Brave does not report "
                f"remaining monthly credits, so this is our own count of gateway "
                f"calls that reached Brave — raise brave_monthly_budget if the plan changed.",
                min_interval_s=MIN_INTERVAL_MONTHLY,
            )
            fired.append(CAT_MONTHLY)

    # (d) Blackout, checked before the rate alarms: if nothing was served, the
    # throttle rate is a consequence and not the story.
    if window.total >= min_samples and window.ok == 0:
        await send_infra_alert(
            CAT_BLACKOUT, "critical",
            f"Search gateway served ZERO results in the last {window_h}h across "
            f"{window.total} attempts ({window.throttled} throttled, "
            f"{window.errored} errored). Reasons: {reasons}. This is the "
            f"'Brave silently disconnected' signature — check GET /api/search/health.",
            min_interval_s=MIN_INTERVAL_BLACKOUT,
        )
        fired.append(CAT_BLACKOUT)

    # (b) Sustained throttling. Below min_samples a rate is noise, not a signal.
    if window.total >= min_samples:
        throttle_pct = 100.0 * window.throttled / window.total
        level = None
        if throttle_pct >= float(_cfg("search_throttle_critical_pct", 50.0)):
            level = "critical"
        elif throttle_pct >= float(_cfg("search_throttle_warn_pct", 25.0)):
            level = "warning"
        if level:
            await send_infra_alert(
                CAT_THROTTLE, level,
                f"Search throttling at {throttle_pct:.0f}% over {window_h}h "
                f"({window.throttled}/{window.total} calls shed to slower tiers). "
                f"Reasons: {reasons}. Users still get answers, from tiers 2/3, "
                f"but slower and without Brave's snippets.",
                min_interval_s=MIN_INTERVAL_THROTTLE,
            )
            fired.append(CAT_THROTTLE)

    # (c) Fleet headroom. Denominator is calls that carried a reading, not all
    # calls — a shed call records the last value seen, an errored one records
    # nothing.
    if window.headroom_samples >= min_samples:
        low_pct = 100.0 * window.low_headroom / window.headroom_samples
        if low_pct >= float(_cfg("search_headroom_warn_pct", 10.0)):
            avg = f"{window.avg_remaining:.1f}" if window.avg_remaining is not None else "n/a"
            await send_infra_alert(
                CAT_HEADROOM, "warning",
                f"Brave fleet headroom low: {low_pct:.0f}% of the last "
                f"{window.headroom_samples} calls saw x-ratelimit-remaining at or "
                f"below the shed floor ({floor}); min {window.min_remaining}, "
                f"avg {avg}. The per-second ceiling is shared by every tenant, so "
                f"this is contention on the account, not on one user.",
                min_interval_s=MIN_INTERVAL_HEADROOM,
            )
            fired.append(CAT_HEADROOM)

    return {
        "window_h": window_h,
        "total": window.total,
        "ok": window.ok,
        "throttled": window.throttled,
        "errored": window.errored,
        "headroom_samples": window.headroom_samples,
        "low_headroom": window.low_headroom,
        "min_remaining": window.min_remaining,
        "avg_remaining": window.avg_remaining,
        "month_calls": month_calls,
        "month_budget": budget,
        "month_used_pct": round(used_pct, 1),
        "alerts": fired,
    }


async def search_quota_monitor_loop() -> None:
    """Forever loop; start via asyncio.create_task in the lifespan."""
    interval = max(300, int(_cfg("search_quota_check_interval_s", 3600)))
    logger.info("[search-quota] monitor started (interval=%ss)", interval)
    while True:
        try:
            # Sleep first: at boot the window holds whatever this replica's
            # predecessor left behind, and a deploy is the worst moment to page.
            await asyncio.sleep(interval)
            await check_search_quota()
        except asyncio.CancelledError:
            raise
        except Exception:
            # One bad query must not take the monitor down for the process
            # lifetime — but it is never swallowed quietly either.
            logger.exception("[search-quota] check failed; continuing")
