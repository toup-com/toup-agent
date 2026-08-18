"""Platform-side search gateway — the only thing that talks to Brave.

Topology, and why it changed
────────────────────────────
Until this module, the shared platform Brave key was pushed into every tenant
container as ``BRAVE_API_KEY`` and each container called
``api.search.brave.com`` directly. That put one fleet-wide secret in 42 places,
made rotation a 42-container recreate, and left metering and rate limiting as
things the container *reported* rather than things we *enforced*.

This is the same topology the LLM path already rejected. There, the container
holds ``TOUP_TOKEN`` — a Toup-issued opaque credential with no provider
meaning — and ``llm_proxy.py`` exchanges it for the real provider key
server-side. The container never sees an OpenAI key; rotation is one Railway
variable and zero container restarts.

Search now works the same way, and reuses the *same* credential: every tenant
already has ``TOUP_TOKEN`` in its env and ``llm_token_hash`` on its
``AgentConfig`` row (verified 42/42 before this was written). So there is no
new secret to mint, no new column, no container env change, and **no fleet
backfill** — the backfill was a symptom of shipping the key, and it disappears
with the key.

What lives here because it can only be correct here
───────────────────────────────────────────────────
* **Fleet-wide rate limiting.** Brave's ceiling (50 rps) is one *account*
  limit shared by every key on the plan, so 42 independent in-container token
  buckets could never add up to it. See ``_FleetGuard``.
* **Metering.** The gateway holds the DB session and knows who called, so a
  row is written from observation rather than from a container's self-report.
* **The circuit breaker.** A 429 seen by one tenant is information about the
  whole fleet.
* **The key.** One process, one variable, rotatable without touching a
  container.

What deliberately stays in the container: the in-process result cache (tier 0
— free, zero-latency, never metered) and tiers 2/3 (httpx race, headless
Chromium), which have their own egress and no shared secret. See
``tool_executor._tool_web_search``.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_db
from app.db.models import AgentConfig, SearchEvent
from app.services.credit_service import CreditService, FLAT_FEES
from app.websearch import freshness as _fresh
from app.websearch.render import brave_news_to_dicts, brave_web_to_dicts

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search-gateway"])

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"

# F7 counters — process-local, surfaced on /search/health. Not a metric
# system; enough to answer "is the freshness ladder ever widening" and "how
# often does the stale filter bite" without a schema change.
_COUNTERS: dict[str, int] = {
    "recent_queries": 0,
    "evergreen_queries": 0,
    "freshness_widened": 0,       # pm → py (or → none) because the page was thin
    "stale_dropped": 0,           # results removed by the 18-month filter
    "news_blended": 0,            # requests that merged news results
    "site_discovery": 0,          # site:-anchored recency queries that also ran neutral
    "brave_calls": 0,             # upstream HTTP calls, all endpoints
}

# Tier names are a shared vocabulary with the agent's ladder
# (tool_executor._tool_web_search) and with the admin panel. Do not rename one
# side alone.
TIER_BRAVE = "brave_api"

# Statuses written to search_events.status.
ST_OK = "ok"
ST_THROTTLED = "throttled"   # we shed it; the agent falls to a lower tier
ST_ERROR = "error"           # upstream failed; the agent falls to a lower tier

# Per-user daily Brave ceiling.
RSN_USER_DAILY_CAP = "user_daily_cap"
# Recency query whose every dated result was older than the stale cutoff. The
# agent must NOT fall through to undated tiers on this one — see search_web.
RSN_ALL_STALE = "all_stale"

# degraded_reason values that mean the request never reached Brave and so
# consumed none of its quota. This module PRODUCES every one of them, so it
# owns the vocabulary; search_quota_monitor imports it rather than keeping a
# second copy (they had drifted apart the moment a new reason was added).
# A capped attempt must appear here or the ceiling becomes self-reinforcing:
# capped rows would count as usage and hold the user over the cap forever.
NEVER_REACHED_BRAVE = (
    "tenant_rate_limit", "fleet_headroom", "cooldown_after_429",
    RSN_USER_DAILY_CAP,
)


# ── Token auth ───────────────────────────────────────────────────────
#
# Same credential and same hash as llm_proxy._auth_agent. Two differences,
# both deliberate:
#   * Only `Authorization: Bearer` is accepted. The dual-header dance in
#     llm_proxy exists for OpenAI-vs-Anthropic SDK compatibility; this is a
#     private contract with our own client, so it gets one shape.
#   * The token→AgentConfig lookup is cached. llm_proxy does an uncached
#     SELECT per request, which is fine at LLM-turn cadence but not here:
#     AgentRunner.PARALLEL_SAFE_TOOLS fans several web_search calls out of a
#     single turn, and `llm_token_hash` carries no index.

_TOKEN_CACHE: dict[str, tuple[float, str]] = {}   # hash -> (expires_at, user_id)
_TOKEN_CACHE_TTL = 300.0
_TOKEN_CACHE_MAX = 512


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _cache_get_user(token_hash: str) -> Optional[str]:
    hit = _TOKEN_CACHE.get(token_hash)
    if not hit:
        return None
    if hit[0] < time.monotonic():
        _TOKEN_CACHE.pop(token_hash, None)
        return None
    return hit[1]


def _cache_put_user(token_hash: str, user_id: str) -> None:
    if len(_TOKEN_CACHE) >= _TOKEN_CACHE_MAX:
        # Cheap eviction: this only ever holds one entry per tenant, so the
        # cap is a runaway guard, not a working eviction policy.
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[token_hash] = (time.monotonic() + _TOKEN_CACHE_TTL, user_id)


async def auth_agent_user_id(request: Request, db: AsyncSession) -> str:
    """Resolve TOUP_TOKEN → user_id, or raise 401.

    Returns only the user_id: the gateway needs attribution, not the whole
    AgentConfig, and not holding the row keeps the cache trivially safe
    against a stale bundle_status.
    """
    auth_header = request.headers.get("authorization", "")
    token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    if not token:
        raise HTTPException(401, "Missing token. Provide 'Authorization: Bearer <TOUP_TOKEN>'.")

    token_hash = _hash_token(token)
    cached = _cache_get_user(token_hash)
    if cached:
        return cached

    row = await db.execute(
        select(AgentConfig.user_id).where(AgentConfig.llm_token_hash == token_hash)
    )
    user_id = row.scalar_one_or_none()
    if not user_id:
        # 401, not 404 — never confirm whether a token shape exists.
        raise HTTPException(401, "Invalid token")

    _cache_put_user(token_hash, user_id)
    return user_id


# ── Rate limiting ────────────────────────────────────────────────────


class _TokenBucket:
    """Per-tenant share of the fleet ceiling. Bounds one runaway loop.

    Held per PROCESS, so with N replicas a tenant gets N buckets. Measured on
    production 2026-07-31: a 10-request burst inside 762 ms was served in full
    against a configured burst of 5, and Brave's own remaining counter fell
    49 -> 40, confirming all ten reached upstream. ``_effective_rate`` divides
    the configured numbers by ``platform_replicas`` so the config value means
    what it says fleet-wide. That division is an approximation — it assumes
    even load balancing — which is why it is the *secondary* bound. The
    primary one is ``_FleetGuard``, which needs no such assumption because it
    reads Brave's own fleet-wide number.
    """

    __slots__ = ("_rate", "_burst", "_tokens", "_last")

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self._rate = max(0.01, float(rate_per_sec))
        self._burst = max(1, int(burst))
        self._tokens = float(self._burst)
        self._last = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self._tokens = min(self._burst, self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class _FleetGuard:
    """Fleet-wide headroom against Brave's shared account ceiling.

    There is no Redis on this platform and platform-api runs more than one
    replica, so a counter held in this process cannot see the fleet. It does
    not have to: **Brave reports the fleet state on every response.**

        x-ratelimit-limit:     50, 0
        x-ratelimit-policy:    50;w=1, 0;w=2678400
        x-ratelimit-remaining: 49, 0

    The first bucket is the per-second account ceiling, computed by Brave
    across every key on the plan — which is exactly the number no coordinator
    of ours could reconstruct. We read it off each response and shed new
    requests while it sits below ``brave_fleet_floor``. The signal is at most
    one request stale, which at a 1-second window is the correct resolution.

    A 429 trips the breaker outright for ``brave_cooldown_s``: shedding to a
    lower tier costs the user latency, but hammering a rate-limited shared
    account costs every tenant their search.
    """

    __slots__ = ("_remaining", "_seen_at", "_cooldown_until")

    def __init__(self) -> None:
        self._remaining: Optional[int] = None
        self._seen_at = 0.0
        self._cooldown_until = 0.0

    @property
    def remaining(self) -> Optional[int]:
        return self._remaining

    def observe(self, headers) -> Optional[int]:
        """Record fleet headroom from a Brave response. Returns it, or None."""
        raw = headers.get("x-ratelimit-remaining") if headers else None
        if not raw:
            return None
        try:
            first = raw.split(",")[0].strip()
            value = int(first)
        except (ValueError, AttributeError):
            return None
        self._remaining = value
        self._seen_at = time.monotonic()
        return value

    def trip(self, seconds: float) -> None:
        self._cooldown_until = time.monotonic() + max(1.0, seconds)

    def allowed(self) -> tuple[bool, Optional[str]]:
        now = time.monotonic()
        if now < self._cooldown_until:
            return False, "cooldown_after_429"
        # A reading older than 5s tells us nothing about the current second.
        if self._remaining is not None and (now - self._seen_at) < 5.0:
            if self._remaining <= int(getattr(settings, "brave_fleet_floor", 5)):
                return False, "fleet_headroom"
        return True, None


_buckets: dict[str, _TokenBucket] = {}
_fleet = _FleetGuard()


def _effective_rate() -> tuple[float, int]:
    """Per-process share of the configured per-tenant SUSTAINED rate.

    Buckets are per process, so the sustained rate is divided by the replica
    count to mean what it says fleet-wide.

    BURST IS NOT DIVIDED. Rate is about sustained load, where dividing is the
    right approximation. Burst is about how many searches ONE turn may have in
    flight, and dividing it there makes a turn's outcome depend on load-balancer
    luck rather than on the configured number.

    Both production observations fit that reading:

      2026-07-31  a 10-request burst inside 762ms was served IN FULL against a
                  configured burst of 5 — the requests spread over two replicas,
                  two buckets, and Brave's own counter fell 49 -> 40.
      2026-08-01  a voice research turn issuing three concurrent searches had
                  exactly one shed, degraded_reason='tenant_rate_limit', with a
                  repeating 2-ok/1-throttled signature at 16:56:58 and again at
                  18:36:45 — three requests landing on ONE replica whose burst
                  had become max(1, 5 // 2) = 2.

    So the divisor does not bound a turn; it decides whether a routine turn is
    shed based on how the balancer happened to fan three simultaneous requests.
    The shed search still answered through the slower in-container tier, so the
    cost was pure latency on the path voice is most sensitive to, and it never
    surfaced as an error.

    Removing the divisor is safe on the number that actually matters: burst 5 on
    each of 2 replicas is at most 10 instantaneous requests against Brave's
    50 rps account ceiling — precisely the 2026-07-31 case, which was fine. And
    ``_FleetGuard`` reads Brave's own fleet-wide counter, needs no even-load
    assumption, and remains the primary bound. This was always the secondary.
    """
    replicas = max(1, int(getattr(settings, "platform_replicas", 1)))
    rate = float(getattr(settings, "brave_rate_per_sec", 2.0)) / replicas
    burst = max(1, int(getattr(settings, "brave_burst", 5)))
    return rate, burst


def _tenant_allowed(user_id: str) -> tuple[bool, Optional[str]]:
    bucket = _buckets.get(user_id)
    if bucket is None:
        bucket = _TokenBucket(*_effective_rate())
        _buckets[user_id] = bucket
    if not bucket.take():
        return False, "tenant_rate_limit"
    return True, None


# ── Request / response ───────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=400)
    count: int = Field(8, ge=1, le=20)
    country: Optional[str] = Field(None, max_length=2)
    # Which surface asked. Telemetry only — never trusted for authorization.
    channel: Optional[str] = Field(None, max_length=20)
    # "recent" | "evergreen" — the agent's own verdict from the shared
    # classifier. Optional and advisory: absent (older agent images) or
    # unrecognised → the gateway classifies the query itself, so every tenant
    # gets freshness-aware search from the platform deploy alone.
    freshness_class: Optional[str] = Field(None, max_length=12)


class SearchResult(BaseModel):
    title: str
    url: str
    description: str = ""
    # Up to 5 extra passages Brave pulls from each result page. Kept on the
    # wire because they are why the agent can often answer WITHOUT a slow
    # web_fetch — dropping them here would quietly raise the fetch count.
    extra_snippets: list[str] = Field(default_factory=list)
    # Brave's page date, both shapes, verbatim: `page_age` is ISO
    # ("2026-07-24T00:00:00"), `age` is human ("3 weeks ago", "July 9, 2026").
    # These were dropped on the wire before the 2026-08-18 incident, which is
    # how a 2023 blog post and a last-week announcement looked identical to
    # the model. Optional so older agents ignore them.
    age: Optional[str] = None
    page_age: Optional[str] = None
    # "web" | "news" — which Brave index produced it.
    source: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    tier: str
    engine: Optional[str] = None
    # When served=False the agent must fall through to its local tiers. The
    # gateway degrades, it does not deny — except on `denied`, below.
    served: bool
    degraded_reason: Optional[str] = None
    latency_ms: int = 0
    # Freshness policy that was applied — echoed so the agent can log it,
    # render it in the result header and pick the cache TTL. All optional.
    freshness_class: Optional[str] = None
    freshness_applied: Optional[str] = None     # "pm" | "py" | None
    dropped_stale: int = 0
    news_count: int = 0
    brave_calls: int = 0


def _query_hash(query: str) -> str:
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


async def _user_brave_calls_24h(db: AsyncSession, user_id: str) -> Optional[int]:
    """How many Brave calls this user has actually consumed in the last 24h.

    Mirrors ``search_quota_monitor._brave_calls_since`` — the NOT IN arm drops
    NULLs in SQL and a served call has ``degraded_reason IS NULL``, so the
    IS NULL arm is load-bearing, not defensive.

    Returns None if the count could not be taken; callers must fail OPEN. A
    broken counter must never take search down — but it is logged, because a
    silently-open ceiling is indistinguishable from no ceiling at all.
    """
    since = datetime.utcnow() - timedelta(hours=24)
    try:
        result = await db.execute(
            select(func.count()).where(
                SearchEvent.user_id == user_id,
                SearchEvent.created_at >= since,
                SearchEvent.tier == TIER_BRAVE,
                or_(
                    SearchEvent.degraded_reason.is_(None),
                    SearchEvent.degraded_reason.notin_(NEVER_REACHED_BRAVE),
                ),
            )
        )
        return int(result.scalar() or 0)
    except Exception:
        logger.exception("[search-gw] daily-cap count FAILED user=%s (failing open)", user_id)
        return None


async def _record(
    db: AsyncSession,
    *,
    user_id: str,
    tier: str,
    engine: Optional[str],
    status: str,
    degraded_reason: Optional[str],
    latency_ms: int,
    result_count: int,
    channel: Optional[str],
    query_sha256: str,
    brave_remaining: Optional[int],
) -> Optional[str]:
    """Write the telemetry row. Returns its id, or None if the write failed.

    Metering must never take a user's search away from them, so this is
    fail-open — but it is fail-open *loudly*. A silent swallow here is exactly
    the class of defect that let Brave sit disconnected for months.
    """
    event = SearchEvent(
        user_id=user_id,
        tier=tier,
        engine=engine,
        status=status,
        degraded_reason=degraded_reason,
        was_fallback=(status != ST_OK),
        latency_ms=latency_ms,
        result_count=result_count,
        channel=channel,
        query_sha256=query_sha256,
        brave_remaining=brave_remaining,
        created_at=datetime.utcnow(),
    )
    try:
        db.add(event)
        await db.flush()
        return event.id
    except Exception:
        logger.exception(
            "[search-gw] telemetry write FAILED user=%s tier=%s — the search "
            "was served but is unaccounted for", user_id, tier,
        )
        return None


async def _charge(
    db: AsyncSession, *, user_id: str, event_id: Optional[str], event: Optional[SearchEvent],
) -> None:
    """Bill the search into the credit ledger, in whatever mode is configured.

    ``web_tool_metering_charge`` is the same dry-run switch the in-container
    metering used: False writes a zero-amount ledger row carrying
    ``credits_quoted`` so the usage series and the would-be denial rate both
    exist before any real money moves.
    """
    if not getattr(settings, "web_tool_metering_enabled", True):
        return

    fee = FLAT_FEES.get("web_search")
    if not fee:
        return
    charge_for_real = bool(getattr(settings, "web_tool_metering_charge", False))

    try:
        result = await CreditService().try_charge(
            db, user_id, "tool_call", fee["bucket"], fee["credits"],
            idempotency_key=event_id,
            event_id=event_id,
            provider="brave",
            metadata={"tool": "web_search", "tier": TIER_BRAVE, "engine": "brave", "via": "gateway"},
            meter_only=not charge_for_real,
        )
        if event is not None:
            event.credits = Decimal(fee["credits"])
            event.charged = bool(charge_for_real and getattr(result, "success", False))
    except Exception:
        logger.exception("[search-gw] credit write FAILED user=%s event=%s", user_id, event_id)


@router.post("/web", response_model=SearchResponse)
async def search_web(
    body: SearchRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Serve one web search from the platform's Brave key.

    Contract with the agent: a non-2xx is a real error, but ``served=False``
    with 200 is a *degrade* — the caller should fall through to its own lower
    tiers rather than surface a failure. Throttling a user into a slower
    answer is acceptable; throttling them into no answer is not.

    There is deliberately NO deny path. Credit exhaustion is handled by the
    ledger in meter-only mode, so nothing here can refuse a search; if a real
    deny is ever added it belongs at this contract's edge, with its own status,
    and the agent has to learn to distinguish it from a degrade.
    """
    user_id = await auth_agent_user_id(request, db)
    qhash = _query_hash(body.query)
    key = (getattr(settings, "brave_api_key", "") or "").strip()

    def degraded(reason: str) -> SearchResponse:
        # The wire shape is the same for every shed reason; the distinction
        # that matters (throttled vs error) is carried by the search_events
        # row, not by the response the agent sees. It falls to a lower tier
        # either way.
        return SearchResponse(
            results=[], tier=TIER_BRAVE, engine="brave",
            served=False, degraded_reason=reason, latency_ms=0,
        )

    if not key:
        # Nothing to record against a tenant — this is a platform misconfig,
        # not tenant usage.
        logger.error("[search-gw] no platform Brave key configured; every tenant is degraded")
        return degraded("unconfigured")

    # Fleet guard first: it has no side effect, so a fleet-wide cooldown must
    # not also burn this tenant's bucket tokens.
    ok, reason = _fleet.allowed()
    if ok:
        ok, reason = _tenant_allowed(user_id)
    if not ok:
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave",
            status=ST_THROTTLED, degraded_reason=reason, latency_ms=0,
            result_count=0, channel=body.channel, query_sha256=qhash,
            brave_remaining=_fleet.remaining,
        )
        await db.commit()
        logger.warning("[search-gw] degraded user=%s reason=%s", user_id, reason)
        return degraded(reason or "throttled")

    # Per-user daily VOLUME ceiling. Every other guard above bounds RATE and
    # sheds to a lower tier, so a runaway loop was reshaped, never bounded —
    # and metering cannot bound it either while web_tool_metering_charge is
    # False (meter_only rows charge 0.00 and `_charge` below never denies).
    # This is the only thing that caps what one user can cost us at Brave.
    # Deliberately a DEGRADE, not a 4xx: the agent still answers from its free
    # lower tiers, which is the correct trade when the cost being protected is
    # Brave quota specifically. A hard error here would break the turn for a
    # limit the user cannot see.
    cap = int(getattr(settings, "search_daily_cap_per_user", 0) or 0)
    if getattr(settings, "search_daily_cap_enabled", False) and cap > 0:
        used = await _user_brave_calls_24h(db, user_id)
        if used is not None and used >= cap:
            await _record(
                db, user_id=user_id, tier=TIER_BRAVE, engine="brave",
                status=ST_THROTTLED, degraded_reason=RSN_USER_DAILY_CAP,
                latency_ms=0, result_count=0, channel=body.channel,
                query_sha256=qhash, brave_remaining=_fleet.remaining,
            )
            await db.commit()
            logger.warning(
                "[search-gw] user daily Brave cap reached user=%s used=%d cap=%d",
                user_id, used, cap,
            )
            return degraded(RSN_USER_DAILY_CAP)

    started = time.monotonic()
    plan = await _run_brave_plan(body, key)
    latency = int((time.monotonic() - started) * 1000)
    remaining = plan.remaining

    if plan.outcome == "upstream_error":
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_ERROR,
            degraded_reason="upstream_error", latency_ms=latency, result_count=0,
            channel=body.channel, query_sha256=qhash, brave_remaining=remaining,
        )
        await db.commit()
        logger.warning("[search-gw] upstream error user=%s: %s", user_id, plan.error)
        return degraded("upstream_error")

    if plan.outcome == "http_429":
        _fleet.trip(getattr(settings, "brave_cooldown_s", 30.0))
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_THROTTLED,
            degraded_reason="http_429", latency_ms=latency, result_count=0,
            channel=body.channel, query_sha256=qhash, brave_remaining=remaining,
        )
        await db.commit()
        logger.warning("[search-gw] Brave 429 — breaker tripped for the whole fleet")
        return degraded("http_429")

    results = [
        SearchResult(
            title=r["title"],
            url=r["url"],
            description=r["description"],
            extra_snippets=[x for x in (r.get("extra_snippets") or [])[:4] if x],
            age=(str(r["age"]) if r.get("age") else None),
            page_age=(str(r["page_age"]) if r.get("page_age") else None),
            source=r.get("source"),
        )
        for r in plan.results
    ]

    if not results:
        # An empty page is not an error, but it is not a served search either —
        # the agent should try a lower tier rather than tell the user nothing
        # exists. Recorded so the rate of it is visible.
        #
        # EXCEPT when the page was non-empty and the stale filter removed all
        # of it: then "nothing fresh exists" IS the answer, and falling to the
        # undated scrape/browser tiers would re-serve the very pages just
        # withheld. `all_stale` tells the agent to say so instead of guessing.
        _reason = RSN_ALL_STALE if plan.dropped_stale else "empty_result"
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_ERROR,
            degraded_reason=_reason, latency_ms=latency, result_count=0,
            channel=body.channel, query_sha256=qhash, brave_remaining=remaining,
        )
        await db.commit()
        if _reason == RSN_ALL_STALE:
            logger.info(
                "[search-gw] all_stale user=%s class=%s dropped=%d — nothing newer than %d days",
                user_id[:8], plan.freshness_class, plan.dropped_stale,
                int(getattr(settings, "search_stale_max_days", 548) or 548),
            )
            resp = degraded(RSN_ALL_STALE)
            resp.freshness_class = plan.freshness_class
            resp.freshness_applied = plan.freshness_applied
            resp.dropped_stale = plan.dropped_stale
            resp.brave_calls = plan.brave_calls
            return resp
        return degraded("empty_result")

    event = SearchEvent(
        user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_OK,
        degraded_reason=None, was_fallback=False, latency_ms=latency,
        result_count=len(results), channel=body.channel, query_sha256=qhash,
        brave_remaining=remaining, created_at=datetime.utcnow(),
    )
    event_id: Optional[str] = None
    try:
        db.add(event)
        await db.flush()
        event_id = event.id
    except Exception:
        logger.exception("[search-gw] telemetry write FAILED user=%s (search still served)", user_id)

    await _charge(db, user_id=user_id, event_id=event_id, event=event if event_id else None)
    try:
        await db.commit()
    except Exception:
        logger.exception("[search-gw] commit FAILED user=%s — search served, usage lost", user_id)
        await db.rollback()

    # F7: one structured line per served search — what Brave was asked, what
    # the model will be shown, and how old it is. This is the line to grep
    # when an answer is stale.
    _oldest, _newest, _undated = _fresh.date_span(plan.results)
    logger.info(
        "[search-gw] served user=%s class=%s freshness=%s ladder=%s brave_calls=%d "
        "web=%d news_found=%d news_merged=%d discovery=%d dropped_stale=%d n=%d "
        "oldest=%s newest=%s undated=%d latency_ms=%d remaining=%s reasons=%s",
        user_id[:8], plan.freshness_class, plan.freshness_applied,
        ">".join(str(a) for a in plan.attempts), plan.brave_calls,
        plan.web_count, plan.news_found, plan.news_count, plan.discovery_count, plan.dropped_stale,
        len(results), _oldest, _newest, _undated, latency, remaining,
        ",".join(plan.reasons),
    )

    return SearchResponse(
        results=results, tier=TIER_BRAVE, engine="brave",
        served=True, degraded_reason=None, latency_ms=latency,
        freshness_class=plan.freshness_class,
        freshness_applied=plan.freshness_applied,
        dropped_stale=plan.dropped_stale,
        news_count=plan.news_count,
        brave_calls=plan.brave_calls,
    )


# ── Upstream plan ────────────────────────────────────────────────────


class _Plan:
    """Outcome of one logical search against Brave (possibly several calls)."""

    __slots__ = (
        "outcome", "error", "results", "remaining", "freshness_class",
        "freshness_applied", "attempts", "reasons", "brave_calls",
        "web_count", "news_count", "discovery_count", "dropped_stale",
        "news_found",
    )

    def __init__(self) -> None:
        self.outcome = "ok"            # ok | upstream_error | http_429
        self.error: Optional[str] = None
        self.results: list[dict] = []
        self.remaining: Optional[int] = None
        self.freshness_class: str = _fresh.EVERGREEN
        self.freshness_applied: Optional[str] = None
        self.attempts: list = []
        self.reasons: list[str] = []
        self.brave_calls = 0
        self.web_count = 0
        self.news_count = 0          # news results in the MERGED page (on the wire)
        self.news_found = 0          # news results Brave returned (log only)
        self.discovery_count = 0
        self.dropped_stale = 0


class _BraveHTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"brave http {status}")
        self.status = status


async def _brave_get(client: httpx.AsyncClient, key: str, endpoint: str, params: dict, plan: _Plan) -> dict:
    """One upstream call. Records fleet headroom from the response headers,
    raises ``_BraveHTTPError`` on any 4xx/5xx (429 included) so the caller can
    map it, and returns the decoded JSON (``{}`` on a malformed body)."""
    plan.brave_calls += 1
    _COUNTERS["brave_calls"] += 1
    resp = await client.get(
        endpoint, params=params,
        headers={"Accept": "application/json", "X-Subscription-Token": key},
    )
    seen = _fleet.observe(resp.headers)
    if seen is not None:
        plan.remaining = seen
    if resp.status_code >= 400:
        raise _BraveHTTPError(resp.status_code)
    try:
        return resp.json()
    except ValueError:
        return {}


async def _run_brave_plan(body: SearchRequest, key: str) -> _Plan:
    """Decide the freshness policy for ``body.query`` and execute it.

    Evergreen: exactly the pre-incident single call (no ``freshness``), plus
    the page dates that were always in the payload and never forwarded.

    Recency: web call with ``freshness`` from the ladder (pm first), and —
    concurrently, so wall-clock is one round trip — the News index and, for a
    ``site:``-anchored query, the neutral discovery form. If the freshness-
    filtered web page is thin (< ``search_freshness_min_results``) the window
    widens (py, then none) sequentially. Then the 18-month stale filter and a
    round-robin merge so each index's top hits survive the cap.

    Failure semantics are unchanged from the single-call version: the PRIMARY
    web call decides — a 429 trips the breaker, any other error degrades.
    Auxiliary calls (news / discovery / a widening attempt) that fail are
    dropped silently and logged; they can only add results, never take the
    primary away.
    """
    plan = _Plan()
    fresh_on = bool(getattr(settings, "search_freshness_enabled", True))
    verdict = _fresh.classify(body.query)
    supplied = _fresh.normalize_class(body.freshness_class)
    klass = supplied or verdict.freshness_class
    if not fresh_on:
        klass = _fresh.EVERGREEN
    plan.freshness_class = klass
    plan.reasons = list(verdict.reasons) if not supplied else [f"agent:{supplied}"] + list(verdict.reasons)
    is_recent = klass == _fresh.RECENT
    _COUNTERS["recent_queries" if is_recent else "evergreen_queries"] += 1

    ladder = list(verdict.ladder) if (is_recent and fresh_on) else [None]
    if is_recent and supplied and not verdict.is_recent:
        # The agent said recent, our patterns did not fire: honour the agent
        # with the default ladder rather than no ladder.
        ladder = list(_fresh.LADDER_DEFAULT)
    plan.attempts = []

    query = body.query
    if is_recent and getattr(settings, "search_recency_append_year", False):
        query = _fresh.with_year(query)

    neutral_query: Optional[str] = None
    if is_recent and getattr(settings, "search_site_discovery_enabled", True):
        neutral, op = _fresh.split_site_operator(query)
        if op and neutral and neutral.lower() != query.lower():
            neutral_query = neutral

    min_results = max(1, int(getattr(settings, "search_freshness_min_results", 3)))
    news_on = is_recent and bool(getattr(settings, "search_news_blend_enabled", True))
    news_count = max(1, int(getattr(settings, "search_news_count", 5)))
    country = body.country

    web_results: list[dict] = []
    news_results: list[dict] = []
    disc_results: list[dict] = []

    async with httpx.AsyncClient(timeout=12.0) as client:
        first = ladder[0]
        plan.attempts.append(first)

        async def _web(freshness):
            return await _brave_get(
                client, key, BRAVE_ENDPOINT,
                _fresh.brave_params(query, body.count, freshness, country=country), plan,
            )

        async def _news(freshness):
            return await _brave_get(
                client, key, BRAVE_NEWS_ENDPOINT,
                _fresh.brave_params(query, news_count, freshness or _fresh.FRESH_MONTH, extra_snippets=False, country=country),
                plan,
            )

        async def _disc(freshness):
            return await _brave_get(
                client, key, BRAVE_ENDPOINT,
                _fresh.brave_params(neutral_query, body.count, freshness, country=country), plan,
            )

        tasks = [_web(first)]
        if news_on:
            tasks.append(_news(first))
        if neutral_query:
            tasks.append(_disc(first))
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        primary = gathered[0]
        if isinstance(primary, _BraveHTTPError):
            plan.outcome = "http_429" if primary.status == 429 else "upstream_error"
            plan.error = str(primary)
            return plan
        if isinstance(primary, BaseException):
            plan.outcome = "upstream_error"
            plan.error = f"{type(primary).__name__}: {primary}"
            return plan
        web_results = brave_web_to_dicts(primary)
        plan.freshness_applied = first

        idx = 1
        if news_on:
            aux = gathered[idx]; idx += 1
            if isinstance(aux, BaseException):
                logger.info("[search-gw] news blend failed (ignored): %s", aux)
            else:
                news_results = brave_news_to_dicts(aux)
        if neutral_query:
            aux = gathered[idx]; idx += 1
            if isinstance(aux, BaseException):
                logger.info("[search-gw] site discovery failed (ignored): %s", aux)
            else:
                disc_results = brave_web_to_dicts(aux)
                _COUNTERS["site_discovery"] += 1

        # Widen the window while the filtered web page is thin. Only the web
        # call is retried; the auxiliaries already ran at the narrow window.
        for nxt in ladder[1:]:
            if len(web_results) >= min_results:
                break
            plan.attempts.append(nxt)
            _COUNTERS["freshness_widened"] += 1
            try:
                wider = brave_web_to_dicts(await _web(nxt))
            except _BraveHTTPError as exc:
                if exc.status == 429:
                    # Do not throw away a served (if thin) page over a widening
                    # attempt — but do trip the breaker so the fleet backs off.
                    _fleet.trip(getattr(settings, "brave_cooldown_s", 30.0))
                logger.info("[search-gw] freshness widen %s failed (ignored): %s", nxt, exc)
                break
            except Exception as exc:
                logger.info("[search-gw] freshness widen %s failed (ignored): %s", nxt, exc)
                break
            if len(wider) > len(web_results):
                web_results = wider
                plan.freshness_applied = nxt

    # Stale filter — recency only. Undated pages survive, labelled by the
    # renderer; a page dated older than the cutoff is not evidence for
    # "newest".
    if is_recent and getattr(settings, "search_stale_filter_enabled", True):
        max_days = int(getattr(settings, "search_stale_max_days", _fresh.DEFAULT_STALE_DAYS) or _fresh.DEFAULT_STALE_DAYS)
        web_results, d1 = _fresh.filter_stale(web_results, max_age_days=max_days)
        news_results, d2 = _fresh.filter_stale(news_results, max_age_days=max_days)
        disc_results, d3 = _fresh.filter_stale(disc_results, max_age_days=max_days)
        plan.dropped_stale = d1 + d2 + d3
        _COUNTERS["stale_dropped"] += plan.dropped_stale

    plan.web_count = len(web_results)
    plan.news_found = len(news_results)
    plan.discovery_count = len(disc_results)

    if is_recent and (news_results or disc_results):
        plan.results = _fresh.merge_results(
            web_results, disc_results, news_results, limit=body.count, interleave=True,
        )
    else:
        plan.results = _fresh.merge_results(web_results, limit=body.count, interleave=False)
    plan.news_count = sum(1 for r in plan.results if r.get("source") == "news")
    if plan.news_count:
        _COUNTERS["news_blended"] += 1
    return plan


@router.get("/health")
async def search_health() -> dict:
    """Operator-facing: is the gateway configured, and what is fleet headroom?

    No tenant data, so no tenant auth — but also no secret: only whether a key
    is present, never any part of it.

    Reports the per-user daily cap because there was otherwise NO way to tell
    from outside whether it is armed. A ceiling that only exists in an env var
    is one nobody can confirm is on, and this subsystem's whole failure mode
    is looking healthy while enforcing nothing — the gateway shipped with a
    comment reading "there is deliberately NO deny path" and stayed that way
    for months because no surface contradicted it.
    """
    allowed, reason = _fleet.allowed()
    _cap = int(getattr(settings, "search_daily_cap_per_user", 0) or 0)
    _cap_on = bool(getattr(settings, "search_daily_cap_enabled", False)) and _cap > 0
    return {
        "configured": bool((getattr(settings, "brave_api_key", "") or "").strip()),
        "fleet_allowed": allowed,
        "fleet_block_reason": reason,
        "brave_remaining": _fleet.remaining,
        "tenants_seen": len(_buckets),
        # `enforcing` is the answer to "will a user actually be refused", which
        # is NOT the same as the flag: a cap of 0 with the flag on enforces
        # nothing. Report the conjunction, not the switch.
        "user_daily_cap_enforcing": _cap_on,
        "user_daily_cap": _cap if _cap_on else None,
        # Freshness policy posture + process-local counters (F7). The flags
        # are reported so "is freshness on in prod" has an answer that is
        # not an env-var reading.
        "freshness_enabled": bool(getattr(settings, "search_freshness_enabled", True)),
        "news_blend_enabled": bool(getattr(settings, "search_news_blend_enabled", True)),
        "stale_filter_days": (
            int(getattr(settings, "search_stale_max_days", 548))
            if getattr(settings, "search_stale_filter_enabled", True) else None
        ),
        "counters": dict(_COUNTERS),
    }
