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
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_db
from app.db.models import AgentConfig, SearchEvent
from app.services.credit_service import CreditService, FLAT_FEES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search-gateway"])

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Tier names are a shared vocabulary with the agent's ladder
# (tool_executor._tool_web_search) and with the admin panel. Do not rename one
# side alone.
TIER_BRAVE = "brave_api"

# Statuses written to search_events.status.
ST_OK = "ok"
ST_THROTTLED = "throttled"   # we shed it; the agent falls to a lower tier
ST_ERROR = "error"           # upstream failed; the agent falls to a lower tier


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
    """Per-process share of the configured per-tenant rate.

    See ``_TokenBucket``: buckets are per process, so the configured value has
    to be divided by the replica count to mean anything fleet-wide.
    """
    replicas = max(1, int(getattr(settings, "platform_replicas", 1)))
    rate = float(getattr(settings, "brave_rate_per_sec", 2.0)) / replicas
    burst = max(1, int(getattr(settings, "brave_burst", 5)) // replicas)
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


class SearchResult(BaseModel):
    title: str
    url: str
    description: str = ""
    # Up to 5 extra passages Brave pulls from each result page. Kept on the
    # wire because they are why the agent can often answer WITHOUT a slow
    # web_fetch — dropping them here would quietly raise the fetch count.
    extra_snippets: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchResult]
    tier: str
    engine: Optional[str] = None
    # When served=False the agent must fall through to its local tiers. The
    # gateway degrades, it does not deny — except on `denied`, below.
    served: bool
    degraded_reason: Optional[str] = None
    latency_ms: int = 0


def _query_hash(query: str) -> str:
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


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

    started = time.monotonic()
    params = {"q": body.query, "count": body.count, "extra_snippets": "true"}
    if body.country:
        params["country"] = body.country

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                BRAVE_ENDPOINT,
                params=params,
                headers={"Accept": "application/json", "X-Subscription-Token": key},
            )
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_ERROR,
            degraded_reason="upstream_error", latency_ms=latency, result_count=0,
            channel=body.channel, query_sha256=qhash, brave_remaining=None,
        )
        await db.commit()
        logger.warning("[search-gw] upstream error user=%s: %s", user_id, exc)
        return degraded("upstream_error")

    latency = int((time.monotonic() - started) * 1000)
    remaining = _fleet.observe(resp.headers)

    if resp.status_code == 429:
        _fleet.trip(getattr(settings, "brave_cooldown_s", 30.0))
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_THROTTLED,
            degraded_reason="http_429", latency_ms=latency, result_count=0,
            channel=body.channel, query_sha256=qhash, brave_remaining=remaining,
        )
        await db.commit()
        logger.warning("[search-gw] Brave 429 — breaker tripped for the whole fleet")
        return degraded("http_429")

    if resp.status_code >= 400:
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_ERROR,
            degraded_reason="upstream_error", latency_ms=latency, result_count=0,
            channel=body.channel, query_sha256=qhash, brave_remaining=remaining,
        )
        await db.commit()
        logger.warning("[search-gw] Brave HTTP %d user=%s", resp.status_code, user_id)
        return degraded("upstream_error")

    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    web = (payload.get("web") or {}).get("results") or []

    results = [
        SearchResult(
            title=(r.get("title") or "").strip(),
            url=(r.get("url") or "").strip(),
            description=(r.get("description") or "").strip(),
            extra_snippets=[x for x in (r.get("extra_snippets") or [])[:4] if x],
        )
        for r in web[: body.count]
        if r.get("url")
    ]

    if not results:
        # An empty page is not an error, but it is not a served search either —
        # the agent should try a lower tier rather than tell the user nothing
        # exists. Recorded so the rate of it is visible.
        await _record(
            db, user_id=user_id, tier=TIER_BRAVE, engine="brave", status=ST_ERROR,
            degraded_reason="empty_result", latency_ms=latency, result_count=0,
            channel=body.channel, query_sha256=qhash, brave_remaining=remaining,
        )
        await db.commit()
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

    return SearchResponse(
        results=results, tier=TIER_BRAVE, engine="brave",
        served=True, degraded_reason=None, latency_ms=latency,
    )


@router.get("/health")
async def search_health() -> dict:
    """Operator-facing: is the gateway configured, and what is fleet headroom?

    No tenant data, so no tenant auth — but also no secret: only whether a key
    is present, never any part of it.
    """
    allowed, reason = _fleet.allowed()
    return {
        "configured": bool((getattr(settings, "brave_api_key", "") or "").strip()),
        "fleet_allowed": allowed,
        "fleet_block_reason": reason,
        "brave_remaining": _fleet.remaining,
        "tenants_seen": len(_buckets),
    }
