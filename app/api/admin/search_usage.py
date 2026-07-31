"""Admin search usage — the read side of the search gateway's telemetry.

Answers the five questions ``search_events`` was shaped for:

  1. searches per user per day       → ``/admin/search/daily``
  2. how slow                        → p50/p95 on both rollups
  3. which tier answered             → ``tiers`` on the daily rollup
  4. what was throttled or fell back → ``/admin/search/events``
  5. what did it cost whom           → ``credits`` on ``/admin/search/users``

Each is a range scan on an index that already exists
(``ix_search_events_created``, ``ix_search_events_user_created``,
``ix_search_events_status_created``) rather than the JSONB ledger scan this
replaced — see ``SearchEvent``'s docstring for why the table exists at all.

Query text never reaches this module. ``search_events.query_sha256`` is a
16-hex truncation of a sha256; it is surfaced only on the raw event feed and
only as ``query_fingerprint``, so nothing downstream can mistake it for a
query. It exists to make one query repeating in a loop visible.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional, Sequence

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.db.database import get_db, get_engine
from app.db.models import SearchEvent, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/search", tags=["admin-search"])

# Mirrors search_proxy.ST_OK / ST_THROTTLED / ST_ERROR. Literals rather than an
# import so this read path does not pull the gateway's httpx client and Brave
# key handling into the admin router's import graph.
ST_OK = "ok"
ST_THROTTLED = "throttled"
ST_ERROR = "error"

_P50 = 0.5
_P95 = 0.95


def _window_start(days: int) -> datetime:
    """UTC midnight `days` days ago, inclusive.

    Same convention as llm_proxy's /admin/llm/cache-daily so the two rollups
    line up day for day when an operator reads them side by side.
    """
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=days - 1)


def _f(value: Optional[Decimal | float]) -> float:
    return round(float(value), 4) if value is not None else 0.0


def _n_status(status: str):
    return func.sum(case((SearchEvent.status == status, 1), else_=0))


def _n_fallback():
    return func.sum(case((SearchEvent.was_fallback.is_(True), 1), else_=0))


def _sum_charged_credits():
    """Credits that actually moved. `credits` is the quote and is written on
    every served search; `charged` is False for the whole
    ``web_tool_metering_charge`` dry-run window, so summing `credits` alone
    would report revenue that was never billed."""
    return func.coalesce(
        func.sum(case((SearchEvent.charged.is_(True), SearchEvent.credits), else_=0)), 0
    )


async def _percentiles(
    db: AsyncSession,
    where: Sequence,
    group_col,
    fractions: tuple[float, ...],
) -> dict[str, tuple[Optional[int], ...]]:
    """latency_ms percentiles per group, keyed by ``str(group value)``.

    Takes the WHERE clause rather than a built statement so it provably scans
    the same index range as the counts query beside it.

    Postgres computes the ordered-set aggregate inside that scan. SQLite has no
    ``percentile_cont``, so it falls back to sorting the window's latencies in
    Python — a row-per-search fetch, tolerable only because SQLite is local
    platform dev: ``search_events`` is PLATFORM_ONLY, so no agent DB carries
    this table and prod is always Postgres.
    """
    if get_engine().dialect.name == "postgresql":
        stmt = (
            select(
                group_col.label("grp"),
                *[
                    func.percentile_cont(f)
                    .within_group(SearchEvent.latency_ms.asc())
                    .label(f"p{int(f * 100)}")
                    for f in fractions
                ],
            )
            .where(*where)
            .group_by(group_col)
        )
        return {
            str(r.grp): tuple(int(v) if v is not None else None for v in r[1:])
            for r in await db.execute(stmt)
        }

    stmt = select(group_col.label("grp"), SearchEvent.latency_ms).where(*where)
    buckets: dict[str, list[int]] = {}
    for row in await db.execute(stmt):
        buckets.setdefault(str(row.grp), []).append(int(row.latency_ms or 0))
    out: dict[str, tuple[Optional[int], ...]] = {}
    for key, values in buckets.items():
        values.sort()
        out[key] = tuple(
            values[min(int(f * (len(values) - 1) + 0.5), len(values) - 1)]
            for f in fractions
        )
    return out


def _p(lat: dict[str, tuple[Optional[int], ...]], key: Any, idx: int = 0) -> Optional[int]:
    hit = lat.get(str(key))
    return hit[idx] if hit else None


# ── /admin/search/daily ──────────────────────────────────────────────

class SearchDailyRow(BaseModel):
    day: str
    searches: int
    served: int
    throttled: int
    errors: int
    fallbacks: int
    p50_latency_ms: Optional[int] = None
    p95_latency_ms: Optional[int] = None
    credits: float
    charged_credits: float
    cost_cents: float
    tiers: dict[str, int]
    brave_remaining_min: Optional[int] = None


class SearchDailyResponse(BaseModel):
    days: int
    user_id: Optional[str] = None
    rows: list[SearchDailyRow]


@router.get("/daily", response_model=SearchDailyResponse)
async def get_search_daily(
    days: int = Query(7, ge=1, le=90),
    user_id: Optional[str] = Query(None),
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> SearchDailyResponse:
    """Founder asks 1, 2 and 3: how many searches per day (per user when
    `user_id` is given), how slow they were, and which tier answered.

    `cost_cents` reads 0 on every Brave row — the gateway's ``_charge`` writes
    only ``credits``, because Brave is a flat monthly plan with no per-call
    upstream price. The column is carried for an engine that has one.
    """
    since = _window_start(days)
    where = [SearchEvent.created_at >= since]
    if user_id:
        where.append(SearchEvent.user_id == user_id)

    day_col = func.date(SearchEvent.created_at)
    counts = (
        select(
            day_col.label("day"),
            func.count().label("searches"),
            _n_status(ST_OK).label("served"),
            _n_status(ST_THROTTLED).label("throttled"),
            _n_status(ST_ERROR).label("errors"),
            _n_fallback().label("fallbacks"),
            func.coalesce(func.sum(SearchEvent.credits), 0).label("credits"),
            _sum_charged_credits().label("charged_credits"),
            func.coalesce(func.sum(SearchEvent.cost_cents), 0).label("cost_cents"),
            func.min(SearchEvent.brave_remaining).label("brave_remaining_min"),
        )
        .where(*where)
        .group_by(day_col)
        .order_by(day_col.desc())
    )
    tiers_stmt = (
        select(day_col.label("day"), SearchEvent.tier, func.count().label("n"))
        .where(*where)
        .group_by(day_col, SearchEvent.tier)
    )

    count_rows = (await db.execute(counts)).all()
    tier_map: dict[str, dict[str, int]] = {}
    for r in await db.execute(tiers_stmt):
        tier_map.setdefault(str(r.day), {})[r.tier] = int(r.n or 0)
    lat = await _percentiles(db, where, day_col, (_P50, _P95))

    rows = [
        SearchDailyRow(
            day=str(r.day),
            searches=int(r.searches or 0),
            served=int(r.served or 0),
            throttled=int(r.throttled or 0),
            errors=int(r.errors or 0),
            fallbacks=int(r.fallbacks or 0),
            p50_latency_ms=_p(lat, r.day, 0),
            p95_latency_ms=_p(lat, r.day, 1),
            credits=_f(r.credits),
            charged_credits=_f(r.charged_credits),
            cost_cents=_f(r.cost_cents),
            tiers=tier_map.get(str(r.day), {}),
            brave_remaining_min=(
                int(r.brave_remaining_min) if r.brave_remaining_min is not None else None
            ),
        )
        for r in count_rows
    ]
    return SearchDailyResponse(days=days, user_id=user_id, rows=rows)


# ── /admin/search/users ──────────────────────────────────────────────

class SearchUserRow(BaseModel):
    user_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    searches: int
    served: int
    throttled: int
    errors: int
    fallbacks: int
    p50_latency_ms: Optional[int] = None
    credits: float
    charged_credits: float
    last_search_at: Optional[str] = None


class SearchUsersResponse(BaseModel):
    days: int
    rows: list[SearchUserRow]


@router.get("/users", response_model=SearchUsersResponse)
async def get_search_users(
    days: int = Query(30, ge=1, le=90),
    limit: int = Query(200, ge=1, le=1000),
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> SearchUsersResponse:
    """Founder asks 1, 2, 4 and 5 collapsed into the fleet table: one row per
    user over the window, ordered by volume.

    The users join is outer on purpose — a deleted user's usage still belongs
    in a cost-attribution table, and an inner join would make these totals
    silently stop matching /daily.
    """
    since = _window_start(days)
    where = [SearchEvent.created_at >= since]

    stmt = (
        select(
            SearchEvent.user_id.label("user_id"),
            User.email.label("email"),
            User.name.label("name"),
            func.count().label("searches"),
            _n_status(ST_OK).label("served"),
            _n_status(ST_THROTTLED).label("throttled"),
            _n_status(ST_ERROR).label("errors"),
            _n_fallback().label("fallbacks"),
            func.coalesce(func.sum(SearchEvent.credits), 0).label("credits"),
            _sum_charged_credits().label("charged_credits"),
            func.max(SearchEvent.created_at).label("last_search_at"),
        )
        .select_from(SearchEvent)
        .outerjoin(User, User.id == SearchEvent.user_id)
        .where(*where)
        .group_by(SearchEvent.user_id, User.email, User.name)
        .order_by(func.count().desc())
        .limit(limit)
    )

    result = (await db.execute(stmt)).all()
    lat = await _percentiles(db, where, SearchEvent.user_id, (_P50,))

    rows = [
        SearchUserRow(
            user_id=r.user_id,
            email=r.email,
            name=r.name,
            searches=int(r.searches or 0),
            served=int(r.served or 0),
            throttled=int(r.throttled or 0),
            errors=int(r.errors or 0),
            fallbacks=int(r.fallbacks or 0),
            p50_latency_ms=_p(lat, r.user_id),
            credits=_f(r.credits),
            charged_credits=_f(r.charged_credits),
            last_search_at=r.last_search_at.isoformat() if r.last_search_at else None,
        )
        for r in result
    ]
    return SearchUsersResponse(days=days, rows=rows)


# ── /admin/search/events ─────────────────────────────────────────────

class SearchEventRow(BaseModel):
    id: str
    created_at: str
    user_id: str
    email: Optional[str] = None
    tier: str
    engine: Optional[str] = None
    status: str
    degraded_reason: Optional[str] = None
    was_fallback: bool
    latency_ms: int
    result_count: int
    channel: Optional[str] = None
    credits: Optional[float] = None
    charged: bool
    brave_remaining: Optional[int] = None
    # 16 hex chars of sha256(normalized query). NOT the query and not
    # reversible to it — present only so a runaway loop repeating one search
    # shows up as one repeated fingerprint.
    query_fingerprint: Optional[str] = None


class SearchEventsResponse(BaseModel):
    days: int
    status: str
    rows: list[SearchEventRow]


@router.get("/events", response_model=SearchEventsResponse)
async def get_search_events(
    days: int = Query(7, ge=1, le=90),
    status: str = Query(
        "problems", pattern=r"^(problems|all|ok|throttled|error|degraded|denied)$"
    ),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> SearchEventsResponse:
    """Founder ask 4: the raw throttle / fallback feed, newest first.

    Before ``search_events`` there was no row anywhere for a shed or degraded
    search — the container decided locally and dropped to a lower tier in
    silence, which is how a disconnected Brave key stayed invisible for months.

    `status` defaults to `problems`: anything not `ok`, plus anything flagged
    ``was_fallback``. An exact status seeks ``ix_search_events_status_created``;
    `problems` and `all` range-scan ``ix_search_events_created`` instead, which
    is why this endpoint always bounds by both `days` and `limit`.
    """
    since = _window_start(days)
    stmt = (
        select(SearchEvent, User.email)
        .select_from(SearchEvent)
        .outerjoin(User, User.id == SearchEvent.user_id)
        .where(SearchEvent.created_at >= since)
        .order_by(SearchEvent.created_at.desc())
        .limit(limit)
    )
    if status == "problems":
        stmt = stmt.where(
            (SearchEvent.status != ST_OK) | (SearchEvent.was_fallback.is_(True))
        )
    elif status != "all":
        stmt = stmt.where(SearchEvent.status == status)
    if user_id:
        stmt = stmt.where(SearchEvent.user_id == user_id)

    rows = [
        SearchEventRow(
            id=ev.id,
            created_at=ev.created_at.isoformat() if ev.created_at else "",
            user_id=ev.user_id,
            email=email,
            tier=ev.tier,
            engine=ev.engine,
            status=ev.status,
            degraded_reason=ev.degraded_reason,
            was_fallback=bool(ev.was_fallback),
            latency_ms=int(ev.latency_ms or 0),
            result_count=int(ev.result_count or 0),
            channel=ev.channel,
            credits=_f(ev.credits) if ev.credits is not None else None,
            charged=bool(ev.charged),
            brave_remaining=ev.brave_remaining,
            query_fingerprint=ev.query_sha256,
        )
        for ev, email in await db.execute(stmt)
    ]
    return SearchEventsResponse(days=days, status=status, rows=rows)
