"""
LLM Proxy — routes agent LLM calls through the platform with budget enforcement.

POST /api/llm/chat       — Anthropic Messages API-compatible (SSE streaming)
POST /api/llm/embeddings — OpenAI Embeddings API-compatible
GET  /api/llm/usage      — current-period spend + remaining budget
GET  /api/admin/llm/stats — admin-only aggregate stats

Auth: per-agent TOUP_LLM_TOKEN in Authorization: Bearer header.
All budget enforcement happens here. Agents in bundle mode never talk
to providers directly.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.config import settings
from app.db import get_db, AgentConfig, LLMProxyEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM Proxy"])
admin_router = APIRouter(prefix="/admin/llm", tags=["Admin LLM"])

# ── Budget cache (in-memory, TTL-based) ──────────────────────────────

_budget_cache: dict[str, tuple[float, Decimal]] = {}  # key → (expiry_ts, cost_cents)
_CACHE_TTL = 30  # seconds


def _cache_key(user_id: str, provider: str, scope: str) -> str:
    return f"{user_id}:{provider}:{scope}"


def _get_cached_spend(key: str) -> Optional[Decimal]:
    entry = _budget_cache.get(key)
    if entry and entry[0] > time.time():
        return entry[1]
    return None


def _set_cached_spend(key: str, cents: Decimal):
    _budget_cache[key] = (time.time() + _CACHE_TTL, cents)


def _invalidate_cache(user_id: str):
    keys_to_remove = [k for k in _budget_cache if k.startswith(user_id)]
    for k in keys_to_remove:
        _budget_cache.pop(k, None)


# ── Per-tenant request-rate limit (G-20) ─────────────────────────────
#
# The budget caps bound SPEND per month (with a 30s cache the burst can
# overshoot), but nothing bounded REQUESTS: a leaked or runaway
# TOUP_LLM_TOKEN could fire unlimited RPS until the monthly cents cap
# tripped — and for an admin-role tenant _check_budget returns None, so a
# leaked admin token had no ceiling at all.
#
# THIS WINDOW IS PER PROCESS, AND platform-api RUNS TWO REPLICAS
# (railway.json: "numReplicas": 2). So the effective ceiling is up to 2x
# the configured value, and which replica a request lands on decides
# whether it counts — Retry-After is computed from one replica's view of
# the world. An earlier version of this comment claimed "platform-api runs
# as one process", which was simply false.
#
# That is tolerable ONLY because of what this control is for. It is a
# backstop against a leaked or runaway token, which does not look like 2x
# normal traffic; it looks like orders of magnitude more, and it trips on
# either replica. It is NOT a fair-share quota, and it must not be sized
# as though it were — see llm_proxy_rate_limit_per_min for the measured
# distribution behind the number. A precise cap needs shared state
# (Redis/Postgres) and should not pretend to exist until it does.
#
# In-memory is still the right shape for the backstop: the key (the
# tenant's user_id) only exists after _auth_agent resolves the token
# inside the handler, so a middleware cannot key this. Admins are
# deliberately NOT exempt.
_rate_windows: dict[str, list[float]] = {}
_RATE_WINDOW_S = 60.0


def _check_rate_limit(user_id: str) -> Optional[int]:
    """None = allowed (and the call is recorded). Otherwise the number of
    seconds after which the oldest recorded call leaves the window —
    served as Retry-After on the 429."""
    limit = settings.llm_proxy_rate_limit_per_min
    if limit <= 0:
        return None
    now = time.time()
    window = [t for t in _rate_windows.get(user_id, ()) if t > now - _RATE_WINDOW_S]
    if len(window) >= limit:
        _rate_windows[user_id] = window
        return max(1, int(window[0] + _RATE_WINDOW_S - now) + 1)
    window.append(now)
    _rate_windows[user_id] = window
    return None


def _enforce_rate_limit(config) -> None:
    """Called by every handler that can SPEND on the tenant's behalf.

    That is chat, responses, embeddings, and the five image routes
    (`/openai/v1/images/generations`, `/openai/v1/images/edits`,
    `/kie/image`, `/kie/image/start`, `/kie/image/poll`). The SDK-shim
    aliases delegate into `proxy_chat`, so they are covered by it.

    An earlier version of this docstring claimed three call sites covered
    "every provider path". They did not: the image routes are independent
    handlers, not shims, and they are the most expensive calls on the
    proxy per request. Their only other ceiling is
    `reserve_free_image_slot`, which is a FREE-TIER monthly cap — so paid
    and admin tenants had none at all, which is exactly the leaked-token
    threat this limiter exists for.

    `get_proxy_usage` (GET /usage) is deliberately NOT limited: it reports
    spend, it does not cause any.
    """
    retry_after = _check_rate_limit(str(config.user_id))
    if retry_after is None:
        return
    logger.warning(
        "[RATELIMIT] 429 user=%s retry_after=%ss limit=%s/min",
        str(config.user_id)[:8], retry_after,
        settings.llm_proxy_rate_limit_per_min,
    )
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded for this tenant token.",
        headers={"Retry-After": str(retry_after)},
    )


# ── Token auth ───────────────────────────────────────────────────────


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def _auth_agent(request: Request, db: AsyncSession) -> AgentConfig:
    """Validate TOUP_LLM_TOKEN and return the AgentConfig.

    Accepts the token from EITHER `Authorization: Bearer <token>` OR
    `x-api-key: <token>`. Both conventions are valid because:
      - OpenAI Python SDK sends `Authorization: Bearer <key>`
      - Anthropic Python SDK sends `x-api-key: <key>` (their canonical header)
    The agent's llm_client_factory configures both SDKs with `api_key=
    settings.toup_token`; whichever header the SDK chooses, we accept it.
    Without this, every Anthropic-routed call returned 401 → the agent's
    friendly-error handler converted to "Your API key is invalid" — the
    2026-04-27 latent bug uncovered by matin's smoke test.
    """
    # Prefer Authorization: Bearer (OpenAI SDK), fall back to x-api-key
    # (Anthropic SDK). Both are equivalent for our auth model.
    token: str = ""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.headers.get("x-api-key", "").strip()
    if not token:
        raise HTTPException(
            401,
            "Missing token. Provide either 'Authorization: Bearer <TOUP_TOKEN>' "
            "or 'x-api-key: <TOUP_TOKEN>'.",
        )

    token_hash = _hash_token(token)
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.llm_token_hash == token_hash)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(401, "Invalid token")

    if config.bundle_status != "active" and config.bundle_status != "cancelling":
        raise HTTPException(403, "Bundle subscription is not active")

    return config


# ── Budget checks ────────────────────────────────────────────────────


async def _get_spend(
    db: AsyncSession,
    user_id: str,
    provider: str,
    since: datetime,
    cache_scope: str,
) -> Decimal:
    """Get user-attributable spend in cents for a user+provider since a given time.

    Excludes events tagged with operation_type starting with "system." — those are
    platform-side operations (e.g. end-of-day archival) tracked for cost dashboards
    but exempt from user budget caps.
    """
    cache_key = _cache_key(user_id, provider, cache_scope)
    cached = _get_cached_spend(cache_key)
    if cached is not None:
        return cached

    # Budget counts only user-attributable events. System ops (operation_type LIKE 'system.%')
    # are logged for cost tracking but don't consume the user's cap.
    result = await db.execute(
        select(func.coalesce(func.sum(LLMProxyEvent.cost_cents), 0)).where(
            LLMProxyEvent.user_id == user_id,
            LLMProxyEvent.provider == provider,
            LLMProxyEvent.created_at >= since,
            (LLMProxyEvent.operation_type.is_(None))
            | (~LLMProxyEvent.operation_type.startswith("system.")),
        )
    )
    # Numeric column → Decimal sum. int() here would truncate fractional
    # spend just before the budget comparison; keep the fraction.
    cents = Decimal(str(result.scalar() or 0))
    _set_cached_spend(cache_key, cents)
    return cents


def _today_utc_start() -> datetime:
    now = datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def _check_budget(config: AgentConfig, provider: str, db: AsyncSession) -> Optional[str]:
    """
    Check budget. Returns None if OK, or an error reason string.
    Also returns the HTTP status code to use.
    """
    # Admins are unlimited — same policy as the credit system (admins are
    # never gated or deducted). Without this, an admin/founder/canary account
    # still hit the per-tenant monthly OpenAI budget cap and got the
    # misleading "Rate limit reached — too many requests" chat error once the
    # $10 default was exhausted (2026-07-05: the chat canary, running every
    # 5 min, tripped it and started false-alarming). One cheap role lookup.
    try:
        from sqlalchemy import select as _select
        from app.db.models import User as _User
        _role = (await db.execute(
            _select(_User.role).where(_User.id == config.user_id)
        )).scalar_one_or_none()
        if _role == "admin":
            return None
    except Exception:
        pass  # role lookup best-effort; fall through to normal budget checks

    period_start = config.bundle_period_start or config.bundle_started_at
    if not period_start:
        return None  # No period tracking yet, allow

    if provider == "anthropic":
        # Monthly check
        monthly_spend = await _get_spend(
            db, config.user_id, "anthropic", period_start, "monthly"
        )
        if monthly_spend >= config.bundle_anthropic_budget_cents:
            return "monthly_exceeded"

        # Daily soft cap
        daily_spend = await _get_spend(
            db, config.user_id, "anthropic", _today_utc_start(), "daily"
        )
        if daily_spend >= config.bundle_anthropic_daily_cap_cents:
            return "daily_exceeded"

    elif provider == "openai":
        monthly_spend = await _get_spend(
            db, config.user_id, "openai", period_start, "monthly"
        )
        if monthly_spend >= config.bundle_openai_budget_cents:
            return "monthly_exceeded"

    return None


# ── Cost calculation ─────────────────────────────────────────────────


# Floor a per-image charge at 0.1 credit so try_charge (which rejects amount<=0)
# never receives a zero even if pricing is misconfigured to 0 cents.
_MIN_IMAGE_CREDITS = Decimal("0.1")

# Models already reported as missing a `cached_input` rate, so the warning in
# _calc_cost_cents fires once per model per process instead of once per call.
_MISSING_CACHED_RATE_WARNED: set[str] = set()


def _calc_cost_cents(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal:
    """Calculate cost in cents from token counts using the pricing table.

    G1 prep (docs/audits/2026-07-g1-model-gate.md): cache-aware billing is
    active ONLY for models whose pricing entry carries the optional
    `cached_input` / `cache_write` columns (the gpt-5.6 family — reads at
    the cached rate, writes at 1.25x input). For every model without those
    columns the extra args are ignored and the math is byte-identical to
    the pre-G1 pure input/output form (A9-2 stays out of scope for the
    live fleet).
    """
    pricing = settings.pricing_per_1k.get(model)
    if not pricing:
        # Fallback: use a conservative estimate
        pricing = {"input": 0.003, "output": 0.015}
    cached_rate = pricing.get("cached_input")
    write_rate = pricing.get("cache_write")
    # Detection-only guard. A model whose pricing entry has no `cached_input`
    # column silently bills every cached token at the FULL input rate — there
    # is no error, no warning, just a wrong number that every downstream cost
    # figure inherits. That is exactly how gpt-5.5 and gpt-4o-mini went
    # mispriced for months while the provider was discounting 57% and 43% of
    # their input respectively (found 2026-08-07 only by reading OpenAI's own
    # billing, see docs/audits/2026-08-g1-cost-and-latency.md).
    #
    # So: if the provider reports cached tokens for a model we have no cached
    # rate for, say so. Once per model per process — this is a hot path, and a
    # per-call log would bury it. Deliberately does NOT change the arithmetic:
    # guessing a discount we have not measured would replace a known-wrong
    # number with an unknown-wrong one.
    if cached_rate is None and int(cached_tokens or 0) > 0:
        if model not in _MISSING_CACHED_RATE_WARNED:
            _MISSING_CACHED_RATE_WARNED.add(model)
            logger.warning(
                "[pricing] %s reports cached_tokens but has no cached_input "
                "rate — cached reads are being billed at the full input rate. "
                "Measure it against organization billing and add the column.",
                model,
            )
    # Cached-read and cache-write tokens are disjoint subsets of
    # prompt_tokens; clamp defensively so bogus provider usage can never
    # produce a negative base.
    cached = min(max(int(cached_tokens or 0), 0), input_tokens) if cached_rate is not None else 0
    written = min(max(int(cache_write_tokens or 0), 0), input_tokens - cached) if write_rate is not None else 0
    base_input = input_tokens - cached - written
    cost_usd = (
        (base_input * pricing["input"] / 1000)
        + (cached * cached_rate / 1000 if cached else 0.0)
        + (written * write_rate / 1000 if written else 0.0)
        + (output_tokens * pricing["output"] / 1000)
    )
    return _never_higher_cents(cost_usd)


def _never_higher_cents(cost_usd: float) -> Decimal:
    """Recorded cost in cents: exact to 4 decimal places, capped at the
    legacy ``max(1, int(cents))`` value.

    The old floor recorded a 0.03¢ embedding as 1¢ (55 calls / 630 tokens
    were recorded as 55¢), and the bundle budget gate SUMS this column —
    fake spend consumed real user budget. The R-3 authorization allows
    recorded costs to go DOWN or stay equal, never up, so the exact value
    is capped at the legacy one: sub-cent calls become accurate, and the
    int() truncation on multi-cent calls (3.7¢ recorded as 3¢) stays until
    a change that may raise recorded costs is separately approved.
    """
    exact = (Decimal(str(cost_usd)) * 100).quantize(Decimal("0.0001"))
    if exact <= 0:
        return Decimal("0")
    legacy = Decimal(max(1, int(cost_usd * 100)))
    return min(exact, legacy)


def _embedding_cost_cents(total_tokens: int) -> Decimal:
    """text-embedding-3-small at $0.00002/token, unfloored (see
    ``_never_higher_cents``). The price constant is unchanged from the
    route's original inline math — R-3's scope is the floor, not pricing."""
    return _never_higher_cents(total_tokens * 0.00002)


# ── Usage event logging ──────────────────────────────────────────────


#: Header the agent uses to report which surface a turn came from.
CHANNEL_HEADER = "x-toup-channel"


#: Hard ceiling matching llm_proxy_events.channel VARCHAR(20). A value longer
#: than the column would raise on INSERT — inside the metering write that runs
#: after a successful LLM call — so it is truncated here, never rejected.
_CHANNEL_MAX = 20


def _sanitize_channel(raw: Optional[str]) -> Optional[str]:
    """Normalise a reported channel to something safe to store, or None.

    This deliberately does NOT validate against `channel_util.KNOWN_CHANNELS`,
    for two reasons.

    1. An allowlist here would SILENTLY DROP a newly-added channel. The
       telemetry would then show a surface's traffic vanishing rather than a
       new label appearing, which is the worse failure and exactly the
       "vocabulary that drifts out of sync" problem this codebase has been
       bitten by before (see app/memory_taxonomy.py's four-vocabulary note).
    2. `llm_proxy.py` runs in the PLATFORM image and has no `app.agent`
       dependency today. Importing one to reach a constant would put agent
       code on the platform's import path — the drift that
       requirements.platform.txt exists to prevent.

    So the contract is narrow and local: lowercase, strip, keep only the
    characters a channel name can contain, truncate to the column width, and
    return None for anything empty. It cannot raise, and it cannot produce a
    value the column will reject — which matters because this runs inside the
    metering write AFTER the user's LLM call already succeeded. A logging
    failure there would turn a served request into a 500.
    """
    if not raw or not isinstance(raw, str):
        return None
    cleaned = "".join(ch for ch in raw.strip().lower() if ch.isalnum() or ch in "_-")
    return cleaned[:_CHANNEL_MAX] or None


async def _log_event(
    db: AsyncSession,
    user_id: str,
    provider: str,
    model: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    cost_cents: Decimal | int,
    latency_ms: int,
    was_fallback: bool = False,
    status: str = "ok",
    operation_type: Optional[str] = None,
    cached_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    channel: Optional[str] = None,
):
    """Log an LLM usage event.

    `channel` (alembic 082): the surface the turn came from, sanitized by
    `_sanitize_channel` — see there for why this is not validated against
    KNOWN_CHANNELS. NULL for every caller that does not report one.

    `cached_tokens` (F-7 / A9-1): prompt-cache read hits reported by the
    provider (OpenAI usage.prompt_tokens_details.cached_tokens, Anthropic
    cache_read_input_tokens). Telemetry-only for the live fleet — it does
    NOT participate in cost_cents or credit math for any model without the
    optional cached_input/cache_write pricing columns (A9-2 stays out of
    scope for those). None means the call site had no usage to inspect
    (error paths).

    `cache_write_tokens` (G1 prep; persisted since alembic 083): prompt-
    cache WRITE tokens — billed at a premium on part of the gpt-5.6
    family. Threaded into the credit charge (tokens_to_credits) alongside
    cached_tokens; both are no-ops for models whose pricing entry lacks
    the cache columns, so legacy billing is byte-identical. Before 083 it
    was priced and then dropped — recoverable only from [CACHE] log lines.

    `operation_type` semantics (CRITICAL — do not change without updating _get_spend):
      - None or "user.*" → user-attributable, counts toward the user's cap.
      - "system.*" → platform-side, EXEMPT from user cap, still shown in cost
        dashboards. Must only be set for genuine platform operations (archival
        summaries, etc.), never for user-facing chat.

    Defensive invariant: the HTTP /chat and /embeddings proxy endpoints NEVER
    pass operation_type — they leave it None so user cap logic applies. System
    operations route via app/services/internal_llm.py which enforces the
    "system." prefix with a ValueError.

    Credit deduction (F-credit): when status=="ok" and the call is
    user-attributable (operation_type is None or starts with "user."), we
    convert token counts to credits and atomically deduct from the user's
    message-credits bucket. The LLMProxyEvent.id doubles as the credit-ledger
    idempotency key so SDK retries / proxy replays don't double-charge.
    Shadow-mode (credit_enforcement_enabled=False) still writes the ledger
    row but never denies. System ops (operation_type startswith "system.")
    are platform overhead and are NOT charged to the user.
    """
    event = LLMProxyEvent(
        id=str(uuid.uuid4()),
        user_id=user_id,
        provider=provider,
        model=model,
        endpoint=endpoint,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_cents=cost_cents,
        was_fallback=was_fallback,
        latency_ms=latency_ms,
        status=status,
        operation_type=operation_type,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        channel=_sanitize_channel(channel),
    )
    db.add(event)

    is_system_op = bool(operation_type and operation_type.startswith("system."))
    if status == "ok" and not is_system_op and (input_tokens > 0 or output_tokens > 0):
        try:
            from app.services.credit_service import (
                credit_service, tokens_to_credits, BUCKET_MESSAGE,
            )
            from app.db.models import LEDGER_CHAT_MESSAGE
            credits = tokens_to_credits(
                model, input_tokens, output_tokens,
                cached_tokens=cached_tokens or 0,
                cache_write_tokens=cache_write_tokens or 0,
            )
            result = await credit_service.try_charge(
                db, user_id, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, credits,
                idempotency_key=event.id, event_id=event.id, model=model,
                provider=provider, input_tokens=input_tokens, output_tokens=output_tokens,
                underlying_cost_cents=cost_cents,
                metadata={"endpoint": endpoint, "operation_type": operation_type or "user"},
                # We are downstream of the provider call: the tokens are spent
                # and the user already has the answer. Denying here cannot
                # un-spend them, it only hides the cost — which is exactly what
                # produced 274 free calls / $17.17 of provider spend carrying
                # reason="daily_cap_exceeded". The charge lands; the resulting
                # over-cap used_today is what makes the NEXT pre-flight (:1017)
                # return 402 and stop the loop.
                already_incurred=True,
            )
            if not result.success:
                logger.warning(
                    "[credits] charge DENIED but response already served "
                    "user=%s model=%s reason=%s credits=%s cost_cents=%s",
                    user_id[:8], model, result.reason, credits, cost_cents,
                )
            logger.info(
                "[credits] deducted user=%s model=%s tokens=%d/%d credits=%s "
                "balance_after=%s idempotent=%s success=%s",
                user_id[:8], model, input_tokens, output_tokens, credits,
                result.balance_after, result.idempotent_hit, result.success,
            )
        except Exception:
            # Full stack trace so we can debug silent deduction failures.
            # Earlier warning-only log made schema mismatches invisible.
            logger.exception(
                "[credits] try_charge failed user=%s event=%s model=%s tokens=%d/%d",
                user_id[:8], event.id[:8], model, input_tokens, output_tokens,
            )

    await db.commit()
    # Only invalidate the user-budget cache when a user-attributable event landed.
    # System operations don't affect user caps so they can leave the cache intact.
    if not is_system_op:
        _invalidate_cache(user_id)

    logger.info(
        "llm_proxy user=%s provider=%s model=%s tokens_in=%d tokens_out=%d "
        "cached=%d cost_cents=%d latency=%dms fallback=%s status=%s op=%s",
        user_id[:8], provider, model, input_tokens, output_tokens,
        cached_tokens or 0, cost_cents, latency_ms, was_fallback, status,
        operation_type or "user",
    )


# ── Provider forwarding ─────────────────────────────────────────────
# LLMBackend interface with Anthropic and OpenAI implementations.
# The routing decision lives in _route_chat() below.


class LLMBackend:
    """Abstract base for LLM provider backends."""
    name: str = "base"

    async def chat(self, body: dict, api_key: str) -> httpx.Response:
        raise NotImplementedError

    async def chat_stream(self, body: dict, api_key: str):
        raise NotImplementedError

    async def embeddings(self, body: dict, api_key: str) -> httpx.Response:
        raise NotImplementedError


class UpstreamProviderError(Exception):
    """Raised by chat_stream when the provider returns a non-2xx status
    BEFORE any SSE bytes were forwarded to the client. proxy_chat catches
    this and converts it to a clean HTTPException so the agent's SDK
    surfaces a useful error instead of a half-streamed empty response.

    Carries `body` (truncated upstream response) so logs and error pages
    can show what the provider actually said (model-not-found, rate-limit,
    plan restriction, etc.).
    """
    def __init__(self, status: int, body: bytes, provider: str):
        self.status = status
        self.body = body[:500]
        self.provider = provider
        super().__init__(f"{provider} returned {status}: {self.body[:200]!r}")


class AnthropicBackend(LLMBackend):
    name = "anthropic"
    BASE_URL = "https://api.anthropic.com"

    async def chat(self, body: dict, api_key: str) -> httpx.Response:
        body["stream"] = False
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                f"{self.BASE_URL}/v1/messages",
                json=body,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )

    async def chat_stream(self, body: dict, api_key: str):
        body["stream"] = True
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/v1/messages",
                json=body,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            ) as resp:
                # Detect 4xx/5xx BEFORE forwarding bytes. Once we yield a
                # single chunk the StreamingResponse commits its 200 OK
                # headers and the agent SDK starts parsing SSE — too late
                # to surface a clean error. (matin incident, 2026-04-27 —
                # cryptic "Your request was blocked." instead of the real
                # "model not found" body from Anthropic.)
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    raise UpstreamProviderError(resp.status_code, body_bytes, "anthropic")
                async for chunk in resp.aiter_bytes():
                    yield chunk


class OpenAIBackend(LLMBackend):
    name = "openai"
    BASE_URL = "https://api.openai.com"

    async def chat(self, body: dict, api_key: str) -> httpx.Response:
        body["stream"] = False
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                f"{self.BASE_URL}/v1/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
            )

    async def chat_stream(self, body: dict, api_key: str):
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/v1/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
            ) as resp:
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    raise UpstreamProviderError(resp.status_code, body_bytes, "openai")
                # W0.2b: the httpx response never leaves this generator, so
                # the upstream cache/routing headers are only visible here.
                _debug_log_upstream_cache_headers(resp.headers, body.get("model", ""))
                async for chunk in resp.aiter_bytes():
                    yield chunk

    async def responses(self, body: dict, api_key: str) -> httpx.Response:
        """Non-streaming /v1/responses twin of chat()."""
        body["stream"] = False
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                f"{self.BASE_URL}/v1/responses",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
            )

    async def responses_stream(self, body: dict, api_key: str):
        """Streaming /v1/responses passthrough. Unlike chat_stream we do
        NOT inject stream_options — Responses streams always carry usage in
        the response.completed event (their stream_options only controls
        obfuscation); any client-sent value is forwarded untouched."""
        body["stream"] = True
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/v1/responses",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
            ) as resp:
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    raise UpstreamProviderError(resp.status_code, body_bytes, "openai")
                _debug_log_upstream_cache_headers(resp.headers, body.get("model", ""))
                async for chunk in resp.aiter_bytes():
                    yield chunk

    async def embeddings(self, body: dict, api_key: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30) as client:
            return await client.post(
                f"{self.BASE_URL}/v1/embeddings",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
            )

    async def images(self, body: dict, api_key: str) -> httpx.Response:
        # gpt-image-1 can take tens of seconds; use the configured image timeout.
        timeout = getattr(settings, "image_gen_timeout_s", 180.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                f"{self.BASE_URL}/v1/images/generations",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
            )

    async def images_edit(self, data: dict, files: list, api_key: str) -> httpx.Response:
        # Multipart image edit (gpt-image-1 /images/edits). httpx derives the
        # multipart boundary from `files`; do NOT set content-type ourselves or
        # the upstream boundary breaks.
        timeout = getattr(settings, "image_gen_timeout_s", 180.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                f"{self.BASE_URL}/v1/images/edits",
                data=data,
                files=files,
                headers={"Authorization": f"Bearer {api_key}"},
            )


class ToupModelBackend(LLMBackend):
    """
    TODO: Future fine-tuned Toup model backend.

    When we host our own models, this class will forward requests to
    our inference server (vLLM, TGI, etc.) instead of external providers.

    The routing decision in _route_chat() is the single place to change:
    add a condition like `if model.startswith("toup-")` to route here.

    For percentage-based rollout, add a random check:
        if model == "toup-1" or (model == "claude-..." and random() < rollout_pct):
            return ToupModelBackend()
    """
    name = "toup"

    async def chat(self, body: dict, api_key: str) -> httpx.Response:
        raise HTTPException(501, "Toup model backend not yet available")

    async def chat_stream(self, body: dict, api_key: str):
        raise HTTPException(501, "Toup model backend not yet available")


# Singletons
_anthropic = AnthropicBackend()
_openai = OpenAIBackend()
_toup = ToupModelBackend()


# Toup-internal model aliases → real provider model identifiers.
# The agent's model_router emits stable Toup-internal names (e.g.
# "claude-opus-4-7") so we can swap the underlying upstream model
# without redeploying every tenant container. Add an entry here if a
# Toup-internal name needs to resolve to a *different* upstream model
# (e.g. degraded tier on a master key without entitlement). Empty by
# default — bare names pass through to Anthropic / OpenAI unchanged.
#
# History: during the matin incident (2026-04-27) we mapped 4-7→4-1 and
# 4-6→4-5 as a temporary downgrade because the platform master key
# returned 404 on the bare 4-7/4-6 names. Verified 2026-04-28 that the
# key now has full 4-7/4-6 entitlement, so the downgrade was removed.
MODEL_ALIASES: dict[str, str] = {}


def _resolve_model_alias(model: str) -> str:
    """Translate a Toup-internal model name to a real provider model id.
    Pass-through for names that aren't aliases (allows callers to send
    real provider names directly when they want)."""
    return MODEL_ALIASES.get(model, model)


def _route_chat(model: str, config: AgentConfig) -> tuple[LLMBackend, str]:
    """
    Pick the backend + API key for a chat request.
    This is the ONE function to change when adding new providers or our own model.
    Returns (backend, api_key).

    For OpenAI: prefer the per-user `bundle_openai_api_key` (auto-provisioned
    via OpenAI Admin API on bundle activation, billed per project for
    granular usage attribution). Fall back to the platform master key if the
    user's project hasn't been provisioned yet (e.g. activated before Phase 2
    deployed, or transient OpenAI Admin API outage during webhook). This is
    the β architecture's defense-in-depth: agent never sees the OpenAI key,
    proxy always authenticates the agent and forwards with the right outbound.

    For Anthropic: always use the platform master key (no Admin API for
    per-user Anthropic key auto-provisioning; tracked as future work).
    """
    m = (model or "").lower()

    # Anthropic models
    if m.startswith("claude"):
        # Hard backstop for the platform-wide Anthropic deactivation
        # (settings.anthropic_enabled=False). The router + preferred_provider
        # data fix mean no well-behaved agent should send a Claude model, so
        # this only catches stragglers (e.g. a stale tenant container whose
        # auto-router still defaults to Claude). We can't transparently serve
        # OpenAI here — the caller used the Anthropic SDK and expects Anthropic
        # response framing, and there is no OpenAI→Anthropic response converter
        # — so we reject cleanly instead of hitting the unfunded shared Claude
        # account. The agent surfaces this as a "temporarily unavailable" 5xx.
        if not getattr(settings, "anthropic_enabled", True):
            logger.warning(
                "[LLM-PROXY] anthropic disabled — rejecting Claude request "
                "model=%s. Agent should be on an OpenAI model (check "
                "agent_configs.preferred_provider).",
                model,
            )
            raise HTTPException(
                503,
                "Anthropic is temporarily disabled on this platform; "
                "this request targeted a Claude model. Your agent should "
                "use an OpenAI model — try again or pick GPT in Settings.",
            )
        key = settings.platform_anthropic_api_key
        if not key:
            raise HTTPException(500, "Platform Anthropic key not configured")
        return _anthropic, key

    # OpenAI models (GPT, o1, o3, o4, etc.)
    if m.startswith(("gpt", "o1", "o3", "o4")):
        key = config.bundle_openai_api_key or settings.platform_openai_api_key
        if not key:
            raise HTTPException(500, "Platform OpenAI key not configured")
        return _openai, key

    # TODO: Route toup-* models to ToupModelBackend
    # if m.startswith("toup"):
    #     return _toup, ""

    # Default to Anthropic
    key = settings.platform_anthropic_api_key
    if not key:
        raise HTTPException(500, "Platform Anthropic key not configured")
    return _anthropic, key


# ── Streaming SSE helpers ────────────────────────────────────────────


def _extract_anthropic_usage(raw_bytes: bytes) -> tuple[int, int, int]:
    """Extract input, output and cached tokens from Anthropic SSE stream bytes."""
    import json
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    for line in raw_bytes.decode("utf-8", errors="replace").split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            if obj.get("type") == "message_start" and "message" in obj:
                usage = obj["message"].get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                cached_tokens = usage.get("cache_read_input_tokens", 0) or 0
            elif obj.get("type") == "message_delta":
                usage = obj.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    return input_tokens, output_tokens, cached_tokens


def _extract_openai_cached_tokens(usage: dict) -> int:
    """Read prompt-cache hits from an OpenAI usage dict (0 when absent).

    F-7 / A9-1: OpenAI nests the count under prompt_tokens_details;
    older API versions may omit the field (or return null) entirely.
    Shared by the streamed-SSE and non-stream JSON extraction paths so
    both shapes stay in lockstep.
    """
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        return 0
    # Review pr5-#2: a truthy non-numeric value here would raise out of
    # the stream path's narrow except and abort the log write in
    # stream_and_log's finally — telemetry must never take down logging.
    try:
        return int(details.get("cached_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _extract_openai_cache_write_tokens(usage, details_key: str = "prompt_tokens_details") -> int:
    """Read prompt-cache WRITE tokens from an OpenAI usage payload (0 when absent).

    G1 prep: the gpt-5.6 explicit-caching regime bills cache writes at
    1.25x input, so the write count must reach _calc_cost_cents. The
    field nests under prompt_tokens_details like cached_tokens, but the
    exact name is unverified until the 5.6 canary (SDK docs list
    `cache_write_tokens`; Anthropic-style `cache_creation_input_tokens`
    is accepted as a fallback spelling). Coded defensively per the gate:
    handles dict AND SDK-object shapes via getattr, returns 0 on any
    miss/garbage — models without a cache_write price ignore it anyway.

    `details_key` (Responses wire): the Responses usage shape nests its
    details under `input_tokens_details` instead — same 3 candidate write
    spellings probed either way. The default keeps every existing chat
    call site byte-identical.
    """
    if usage is None:
        return 0
    if isinstance(usage, dict):
        details = usage.get(details_key) or {}
    else:
        details = getattr(usage, details_key, None)
    for field in ("cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"):
        if isinstance(details, dict):
            value = details.get(field)
        else:
            value = getattr(details, field, None)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    return 0


def _extract_openai_cache_write_from_sse(raw_bytes: bytes) -> int:
    """SSE twin of _extract_openai_cache_write_tokens (same skeleton as
    _extract_openai_usage, kept separate so the pinned 3-tuple contract
    of that extractor stays byte-stable)."""
    import json
    write_tokens = 0
    for line in raw_bytes.decode("utf-8", errors="replace").split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            usage = obj.get("usage")
            if usage:
                write_tokens = _extract_openai_cache_write_tokens(usage)
        except (json.JSONDecodeError, KeyError):
            pass
    return write_tokens


def _extract_responses_cached_tokens(usage) -> int:
    """Responses twin of _extract_openai_cached_tokens: cached-read hits
    nest under usage.input_tokens_details.cached_tokens. Dict AND object
    shapes, garbage-safe int, 0 on any miss."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        details = usage.get("input_tokens_details") or {}
    else:
        details = getattr(usage, "input_tokens_details", None)
    if isinstance(details, dict):
        value = details.get("cached_tokens", 0)
    else:
        value = getattr(details, "cached_tokens", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_responses_usage(raw_bytes: bytes) -> tuple[int, int, int, int]:
    """Extract (input, output, cached, cache_write) tokens from Responses
    SSE stream bytes.

    Responses streams carry usage inside terminal `response.*` events
    (`response.completed`, and usage-bearing `response.incomplete`) as
    obj["response"]["usage"] with the input_tokens/output_tokens/
    input_tokens_details.cached_tokens shape — NOT the chat-completions
    prompt_tokens shape, which is why routing a Responses stream through
    _extract_openai_usage would silently meter zeros. Last usage wins;
    all-garbage input returns zeros.
    """
    import json
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    write_tokens = 0
    for line in raw_bytes.decode("utf-8", errors="ignore").split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            usage = obj.get("response", {}).get("usage")
            if usage:
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
                cached_tokens = _extract_responses_cached_tokens(usage)
                write_tokens = _extract_openai_cache_write_tokens(
                    usage, details_key="input_tokens_details"
                )
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass
    return input_tokens, output_tokens, cached_tokens, write_tokens


def _extract_openai_usage(raw_bytes: bytes) -> tuple[int, int, int]:
    """Extract usage (input, output, cached) from OpenAI SSE stream bytes."""
    import json
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    for line in raw_bytes.decode("utf-8", errors="replace").split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            usage = obj.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cached_tokens = _extract_openai_cached_tokens(usage)
        except (json.JSONDecodeError, KeyError):
            pass
    return input_tokens, output_tokens, cached_tokens


# ── Cache observability (W0.2b) ──────────────────────────────────────
# Read-only [CACHE] log lines that make OpenAI prompt-cache behavior
# auditable per call: retention="24h" is verified sent end-to-end yet
# prod measured 0 cache hits, so the proxy must produce the evidence
# series (request-side key/retention + usage-side prompt/cached tokens)
# a single platform-log grep for '[CACHE]' can aggregate per tenant
# per day. No behavior change, no DB change.


def _cache_log_fields(body: dict) -> tuple[bool, str, str]:
    """Extract loggable cache fields from an outbound OpenAI chat body.

    Returns (has_cache_key, key_hash_8, retention). The prompt_cache_key
    itself is NEVER logged — only a stable 8-char sha256 prefix so calls
    that should share a cache entry can be correlated across log lines.
    """
    key = body.get("prompt_cache_key")
    has_key = isinstance(key, str) and bool(key)
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:8] if has_key else "none"
    retention = body.get("prompt_cache_retention") or "none"
    return has_key, key_hash, str(retention)


def _debug_log_upstream_cache_headers(headers, model: str) -> None:
    """Debug-level dump of cache/routing-related upstream response headers.

    Evidence for the OpenAI escalation if retention proves dead
    server-side: x-request-id lets OpenAI trace the exact request, and
    any cache-* header shows what their edge reported. Auth headers can
    never match the filter, so no key material is ever logged.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    interesting = {
        k: v for k, v in headers.items()
        if k.lower() in ("x-request-id", "cf-ray") or "cache" in k.lower()
    }
    if interesting:
        logger.debug("[CACHE] upstream model=%s headers=%s", model, interesting)


# ── Tool-name dedup (shared by proxy_chat + proxy_responses) ─────────


def _dedup_tool_names(tools: list) -> tuple[list, list]:
    """First-wins dedup of tool definitions by their top-level `name`.

    Defensive: Anthropic 400s the whole turn with "tools: Tool names must
    be unique." on a name collision, and the agent assembles tools from
    core + skills + (optional) MCP without dedup. First-wins is safer than
    last-write-wins because core tools come first in the assembly order.
    Anthropic chat tools and Responses flattened function tools both carry
    a top-level `name`, so one helper serves both endpoints (tools without
    one — e.g. nested chat-completions format — pass through untouched).
    Returns (deduped, duplicate_names).
    """
    seen: set[str] = set()
    deduped: list = []
    dups: list[str] = []
    for t in tools:
        name = t.get("name") if isinstance(t, dict) else None
        if not isinstance(name, str) or not name:
            deduped.append(t)
            continue
        if name in seen:
            dups.append(name)
            continue
        seen.add(name)
        deduped.append(t)
    return deduped, dups


# OpenAI rejects any request whose `tools` array exceeds this, with
# `array_above_max_length` — a 400 on the WHOLE turn, so the user sees
# "There was an issue with the request. Please try rephrasing your
# message." no matter what they typed. Anthropic has no equivalent hard
# cap, so this is applied on the OpenAI path only; capping elsewhere
# would drop tools for no reason.
_OPENAI_MAX_TOOLS = 128


def _cap_tools(tools: list, limit: int = _OPENAI_MAX_TOOLS) -> tuple[list, list]:
    """Trim an over-long tools array from the TAIL. Returns (kept, dropped_names).

    This is a cliff, not a slope: at `limit` everything works and at
    `limit + 1` every single turn fails, for every prompt, with an error
    that blames the user's phrasing. One tenant crossed it on 2026-08-08
    by connecting a 5-tool connector (124 → 129) and lost chat entirely.

    Tail-first because the agent assembles core tools before skills
    before MCP/connectors (the same ordering `_dedup_tool_names` relies
    on for first-wins). So the agent keeps memory, files and messaging —
    the tools it is broken without — and degrades at the margin instead.
    That is the least-bad truncation, not a good one: the real fix is for
    the agent not to offer 129 tools, and the WARN below is what makes
    that visible rather than silent.
    """
    if len(tools) <= limit:
        return tools, []
    kept = tools[:limit]
    dropped = [
        t.get("name") or (t.get("function") or {}).get("name") or "<unnamed>"
        for t in tools[limit:]
        if isinstance(t, dict)
    ]
    return kept, dropped


# ── Endpoints ────────────────────────────────────────────────────────


class UsageResponse(BaseModel):
    # float since R-3/alembic 084: these come from SUM(cost_cents) over a
    # Numeric column — a fractional Decimal into an int field is a
    # ValidationError, i.e. a 500 on /usage.
    anthropic_monthly_cents: float
    anthropic_daily_cents: float
    openai_monthly_cents: float
    anthropic_budget_cents: int
    anthropic_daily_cap_cents: int
    openai_budget_cents: int
    period_start: Optional[str] = None
    period_end: Optional[str] = None


@router.post("/chat")
async def proxy_chat(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Proxy a chat completion request. Accepts Anthropic Messages API format.
    Streams SSE responses without buffering.
    """
    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    # Captured once, then passed explicitly to every _log_event below —
    # including the ones inside the streaming generator. A local (closed over
    # by the generator) rather than a ContextVar, because _log_event runs
    # inside async generators whose context is whoever drives __anext__, and
    # telemetry that silently records NULL would be worse than none.
    req_channel = _sanitize_channel(request.headers.get(CHANNEL_HEADER))
    body = await request.json()
    requested_model = body.get("model", "claude-sonnet-4-6")
    model = _resolve_model_alias(requested_model)
    body["model"] = model  # rewrite so the upstream call uses the real id
    is_stream = body.get("stream", False)

    # Defensive dedup of tool names. Anthropic 400s the whole turn with
    # "tools: Tool names must be unique." if the agent's tools array has
    # a name collision. The agent assembles tools from core + skills +
    # (optional) MCP without dedup, so a skill that registers a name
    # already in core, or any MCP collision, kills the turn end-to-end
    # (user sees "Please try rephrasing your message"). The root cause
    # still wants fixing in the agent's tool_defs, but dedup here keeps
    # tenants unblocked + WARN-logs the offending name so we can patch
    # the offender. Last-write-wins is arbitrary; first-wins is safer
    # because core tools come first in the agent's assembly order.
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        deduped, dups = _dedup_tool_names(tools)
        if dups:
            logger.warning(
                "[LLM-PROXY] dedup'd %d duplicate tool name(s) for user=%s model=%s: %s",
                len(dups), config.user_id[:8], model, sorted(set(dups)),
            )
            body["tools"] = deduped

    # Surface the resolved upstream model id back to the agent so the UI can
    # show the real provider model (e.g. "claude-opus-4-1-20250805") instead
    # of the Toup-internal alias (e.g. "claude-opus-4-7"). Read by the agent's
    # response handler from the response headers.
    # NOTE: this header currently has no client consumer (dead metadata) and
    # is a model-identity leak source; when security_leak_filter is on we drop
    # it (docs/security/audit-2026.md MI-4). Default off = unchanged.
    resolved_model_header = {} if settings.security_leak_filter else {"x-toup-resolved-model": model}

    backend, api_key = _route_chat(model, config)

    # Cap the tools array on the OpenAI path — see `_cap_tools`. Must sit
    # AFTER routing, because the limit belongs to the provider and not to
    # the requested model id: an alias can resolve across backends, and
    # trimming an Anthropic turn would drop tools it would have accepted.
    if backend.name == "openai":
        _tools = body.get("tools")
        if isinstance(_tools, list) and len(_tools) > _OPENAI_MAX_TOOLS:
            _kept, _dropped = _cap_tools(_tools)
            body["tools"] = _kept
            logger.warning(
                "[LLM-PROXY] tools array over OpenAI's cap for user=%s model=%s: "
                "%d > %d, dropped %d from the tail: %s",
                config.user_id[:8], model, len(_tools), _OPENAI_MAX_TOOLS,
                len(_dropped), _dropped,
            )

    # Credit pre-flight: zero-balance gate. Only enforces when
    # credit_enforcement_enabled=True; in shadow mode this is a no-op.
    # Returns 402 with a structured body the agent / chat client can
    # decode to render the "out of credits / upgrade" UI.
    try:
        from app.services.credit_service import (
            credit_service, BUCKET_MESSAGE,
            REASON_INSUFFICIENT_MESSAGE, REASON_DAILY_CAP_EXCEEDED,
            REASON_EMAIL_NOT_VERIFIED,
        )
        if getattr(settings, "credit_enforcement_enabled", False):
            preflight = await credit_service.check_balance(
                db, config.user_id, BUCKET_MESSAGE, Decimal("0.1"),
            )
            if not preflight.success:
                raise HTTPException(
                    402,
                    detail={
                        "error": "out_of_credits",
                        "reason": preflight.reason or REASON_INSUFFICIENT_MESSAGE,
                        "bucket": "message",
                        "balance_after": str(preflight.balance_after),
                    },
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[credits] pre-flight check failed for user=%s: %s",
                       config.user_id[:8], e)

    # Budget check
    budget_result = await _check_budget(config, backend.name, db)
    if budget_result == "monthly_exceeded":
        raise HTTPException(429, f"Monthly {backend.name} budget exceeded")
    if budget_result == "daily_exceeded":
        # Anthropic daily cap hit — try OpenAI fallback. Prefer the user's
        # auto-provisioned per-project key here too, fall back to master.
        fallback_key = config.bundle_openai_api_key or settings.platform_openai_api_key
        if backend.name == "anthropic" and fallback_key:
            logger.info("Daily Anthropic cap hit for user %s, falling back to OpenAI", config.user_id[:8])
            backend = _openai
            api_key = fallback_key
            # Convert Anthropic request to OpenAI format
            body = _anthropic_to_openai_request(body)
            model = body.get("model", "gpt-4o-mini")
            is_fallback = True
        else:
            raise HTTPException(402, "Daily Anthropic cap exceeded and no fallback available")
    else:
        is_fallback = False

    # W0.2b: one request-side [CACHE] line per OpenAI chat call (after
    # fallback resolution so it reflects the body actually sent upstream).
    # Pairs with the usage-side [CACHE] line below for hit-ratio series.
    has_cache_key, cache_key_hash, cache_retention = _cache_log_fields(body)
    if backend.name == "openai":
        logger.info(
            "[CACHE] user=%s model=%s has_cache_key=%s cache_key_hash=%s retention=%s",
            config.user_id[:8], model, has_cache_key, cache_key_hash, cache_retention,
        )

    start_ts = time.time()

    if is_stream:
        # Streaming: collect bytes for usage extraction, forward in real-time.
        # Pre-flight the upstream by pulling the first chunk INSIDE a try
        # block — chat_stream raises UpstreamProviderError before yielding
        # if the upstream returned non-2xx, so we can convert to a clean
        # HTTPException BEFORE committing the StreamingResponse headers.
        gen = backend.chat_stream(body, api_key)
        try:
            first_chunk = await gen.__anext__()
        except UpstreamProviderError as e:
            await _log_event(
                db, config.user_id, backend.name, model, "chat",
                0, 0, 0, int((time.time() - start_ts) * 1000), is_fallback, "error",
                channel=req_channel,
            )
            logger.warning(
                "[LLM-PROXY] %s upstream %d for user=%s model=%s body=%r",
                e.provider, e.status, config.user_id[:8], model, e.body,
            )
            # Re-raise the upstream status so the agent SDK surfaces a useful
            # error (NotFound for invalid model, RateLimit for 429, etc.)
            # rather than the previous opaque "Your request was blocked".
            try:
                detail = e.body.decode("utf-8", errors="replace")
            except Exception:
                detail = str(e.body)
            if settings.security_leak_filter:
                from app.services.model_alias import scrub_provider_names
                detail = scrub_provider_names(detail)
            raise HTTPException(e.status, detail=detail)
        except StopAsyncIteration:
            first_chunk = b""

        collected_bytes = bytearray(first_chunk)

        async def stream_and_log():
            try:
                if first_chunk:
                    yield first_chunk
                async for chunk in gen:
                    collected_bytes.extend(chunk)
                    yield chunk
            finally:
                # Log usage after stream completes
                latency = int((time.time() - start_ts) * 1000)
                cache_write = 0
                if backend.name == "anthropic":
                    inp, out, cached = _extract_anthropic_usage(bytes(collected_bytes))
                else:
                    inp, out, cached = _extract_openai_usage(bytes(collected_bytes))
                    cache_write = _extract_openai_cache_write_from_sse(bytes(collected_bytes))
                    # W0.2b usage-side [CACHE] line: prompt+cached tokens
                    # together so one grep yields the per-tenant ratio series.
                    logger.info(
                        "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
                        "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s",
                        config.user_id[:8], model, has_cache_key, cache_retention,
                        inp, cached, cache_write,
                    )
                cost = _calc_cost_cents(
                    model, inp, out,
                    cached_tokens=cached, cache_write_tokens=cache_write,
                )
                # Do NOT reuse `db` here. It comes from Depends(get_db), and
                # since FastAPI 0.106 the dependency AsyncExitStack is exited
                # BEFORE the response body streams — so by the time this
                # `finally` runs that session is already closed and owned by
                # nobody. Writing through it silently re-checks out a
                # connection that no `async with` will ever return, and if
                # anything raises after checkout the `except` below swallows it
                # and the connection is stranded. Own a fresh session, and
                # shield it: this runs in the generator's finally, which is
                # exactly where a client disconnect delivers cancellation.
                # Reached by /chat AND the SDK aliases (/v1/messages,
                # /openai/v1/chat/completions), which all delegate here.
                async def _log_usage() -> None:
                    from app.db.database import async_session_maker
                    async with async_session_maker() as _log_db:
                        await _log_event(
                            _log_db, config.user_id, backend.name, model, "chat",
                            inp, out, cost, latency, is_fallback,
                            cached_tokens=cached,
                            cache_write_tokens=cache_write,
                            channel=req_channel,
                        )

                try:
                    await asyncio.shield(asyncio.create_task(_log_usage()))
                except Exception as e:
                    logger.warning("Failed to log usage event: %s", e)

        return StreamingResponse(
            stream_and_log(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                **resolved_model_header,
            },
        )
    else:
        # Non-streaming
        try:
            resp = await backend.chat(body, api_key)
        except Exception as e:
            latency = int((time.time() - start_ts) * 1000)
            await _log_event(
                db, config.user_id, backend.name, model, "chat",
                0, 0, 0, latency, is_fallback, "error",
                channel=req_channel,
            )
            raise HTTPException(502, f"Provider error: {e}")
        # Surface clean upstream errors (model-not-found, rate-limit, etc.)
        if resp.status_code >= 400:
            body_bytes = resp.content
            await _log_event(
                db, config.user_id, backend.name, model, "chat",
                0, 0, 0, int((time.time() - start_ts) * 1000), is_fallback, "error",
                channel=req_channel,
            )
            try:
                detail = body_bytes.decode("utf-8", errors="replace")
            except Exception:
                detail = str(body_bytes)
            if settings.security_leak_filter:
                from app.services.model_alias import scrub_provider_names
                detail = scrub_provider_names(detail)
            raise HTTPException(resp.status_code, detail=detail)

        latency = int((time.time() - start_ts) * 1000)

        # Extract usage from response
        resp_data = resp.json()
        cache_write = 0
        if backend.name == "anthropic":
            usage = resp_data.get("usage", {})
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            cached = usage.get("cache_read_input_tokens", 0) or 0
        else:
            usage = resp_data.get("usage", {})
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
            cached = _extract_openai_cached_tokens(usage)
            cache_write = _extract_openai_cache_write_tokens(usage)
            # W0.2b usage-side [CACHE] line (non-stream twin of the SSE path).
            logger.info(
                "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
                "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s",
                config.user_id[:8], model, has_cache_key, cache_retention,
                inp, cached, cache_write,
            )
            _debug_log_upstream_cache_headers(resp.headers, model)

        cost = _calc_cost_cents(
            model, inp, out,
            cached_tokens=cached, cache_write_tokens=cache_write,
        )
        await _log_event(
            db, config.user_id, backend.name, model, "chat",
            inp, out, cost, latency, is_fallback,
            cached_tokens=cached,
            cache_write_tokens=cache_write,
            channel=req_channel,
        )

        # Use JSONResponse so we can attach the resolved-model header.
        from fastapi.responses import JSONResponse
        return JSONResponse(content=resp_data, headers=resolved_model_header)


# ── SDK-compatible path aliases ──────────────────────────────────────
#
# Drop-in compatibility for the official Anthropic & OpenAI Python SDKs
# when they're configured with our proxy as `base_url`. Each SDK appends
# its own canonical path on every call, so we alias those paths to the
# main `/chat` handler. Without these, the agent's bundle-mode client
# constructed in anthropic_service / openai_agent_service hits 405 (no
# route) because /llm/chat alone isn't enough.
#   - Anthropic SDK → POST {base_url}/v1/messages
#   - OpenAI SDK    → POST {base_url}/v1/chat/completions
# We mount /openai/v1/... so the agent can pick base_url=".../llm/openai/v1"
# and keep the Anthropic path at /v1/messages from the same proxy root.


@router.post("/v1/messages")
async def proxy_anthropic_messages(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Anthropic SDK compatibility shim — body is already in Anthropic format."""
    return await proxy_chat(request, db)


@router.post("/openai/v1/chat/completions")
async def proxy_openai_chat_completions(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """OpenAI SDK compatibility shim — body is already in OpenAI format."""
    return await proxy_chat(request, db)


@router.post("/openai/v1/responses")
async def proxy_openai_responses(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """OpenAI SDK compatibility shim — client.responses.create() POSTs
    {base_url}/responses; body is already in Responses format. Serves the
    agent's openai_wire_api="responses" path (G1: gpt-5.6-* requires
    /v1/responses for function tools)."""
    return await proxy_responses(request, db)


async def proxy_responses(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Proxy an OpenAI Responses API request (SSE streaming or JSON) with the
    same auth/budget/metering scaffolding as proxy_chat. A separate full
    handler — NOT a shim into proxy_chat — because both the request body
    (input/instructions vs messages) and the usage shape (input_tokens/
    output_tokens under response.completed vs prompt_tokens on a usage
    chunk) differ; routing a Responses stream through _extract_openai_usage
    would silently meter zeros.

    Metering parity with /chat: one llm_proxy_events row per request
    (endpoint="responses", operation_type None → user-attributable), the
    same _calc_cost_cents math, credit deduction keyed on the event id
    inside _log_event, and _get_spend sums by provider so these rows count
    toward the openai monthly budget automatically.
    """
    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    req_channel = _sanitize_channel(request.headers.get(CHANNEL_HEADER))
    body = await request.json()
    requested_model = body.get("model")
    # No claude-* default here — this endpoint is OpenAI-only, and letting
    # _route_chat's unknown-prefix→Anthropic default apply would route a
    # Responses body to the Anthropic Messages API.
    if not requested_model:
        raise HTTPException(422, "model is required")
    model = _resolve_model_alias(requested_model)
    body["model"] = model  # rewrite so the upstream call uses the real id
    is_stream = body.get("stream", False)

    if not str(model).lower().startswith(("gpt", "o1", "o3", "o4")):
        raise HTTPException(400, "/responses is OpenAI-only")

    # Defensive dedup of tool names (same rationale as proxy_chat).
    # Responses flattened function tools carry a top-level `name`, so the
    # shared first-wins helper applies verbatim.
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        deduped, dups = _dedup_tool_names(tools)
        if dups:
            logger.warning(
                "[LLM-PROXY] dedup'd %d duplicate tool name(s) for user=%s model=%s: %s",
                len(dups), config.user_id[:8], model, sorted(set(dups)),
            )
            body["tools"] = deduped

    resolved_model_header = {} if settings.security_leak_filter else {"x-toup-resolved-model": model}

    backend, api_key = _route_chat(model, config)
    if backend.name != "openai":
        # Unreachable given the prefix gate above; kept as defense in depth
        # so a routing change can never send a Responses body elsewhere.
        raise HTTPException(400, "/responses is OpenAI-only")

    # Same cap as proxy_chat. This endpoint is OpenAI-only by construction,
    # so no backend test is needed — but it is the same 400 and the same
    # unreadable "try rephrasing" for the user, and it would have been easy
    # to fix one path and leave this one live.
    _tools = body.get("tools")
    if isinstance(_tools, list) and len(_tools) > _OPENAI_MAX_TOOLS:
        _kept, _dropped = _cap_tools(_tools)
        body["tools"] = _kept
        logger.warning(
            "[LLM-PROXY] tools array over OpenAI's cap for user=%s model=%s: "
            "%d > %d, dropped %d from the tail: %s",
            config.user_id[:8], model, len(_tools), _OPENAI_MAX_TOOLS,
            len(_dropped), _dropped,
        )

    # Credit pre-flight: zero-balance gate (same contract as proxy_chat).
    try:
        from app.services.credit_service import (
            credit_service, BUCKET_MESSAGE,
            REASON_INSUFFICIENT_MESSAGE,
        )
        if getattr(settings, "credit_enforcement_enabled", False):
            preflight = await credit_service.check_balance(
                db, config.user_id, BUCKET_MESSAGE, Decimal("0.1"),
            )
            if not preflight.success:
                raise HTTPException(
                    402,
                    detail={
                        "error": "out_of_credits",
                        "reason": preflight.reason or REASON_INSUFFICIENT_MESSAGE,
                        "bucket": "message",
                        "balance_after": str(preflight.balance_after),
                    },
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[credits] pre-flight check failed for user=%s: %s",
                       config.user_id[:8], e)

    # Budget check (openai only — no Anthropic fallback on this endpoint)
    budget_result = await _check_budget(config, "openai", db)
    if budget_result == "monthly_exceeded":
        raise HTTPException(429, "Monthly openai budget exceeded")

    # W0.2b request-side [CACHE] line — Responses bodies use the same
    # top-level prompt_cache_key/prompt_cache_retention names, so the
    # shared field extractor works unchanged.
    has_cache_key, cache_key_hash, cache_retention = _cache_log_fields(body)
    logger.info(
        "[CACHE] user=%s model=%s has_cache_key=%s cache_key_hash=%s retention=%s",
        config.user_id[:8], model, has_cache_key, cache_key_hash, cache_retention,
    )

    start_ts = time.time()

    if is_stream:
        # Streaming: pre-pull the first chunk INSIDE try so an upstream
        # non-2xx converts to a clean HTTPException BEFORE the
        # StreamingResponse commits its 200 headers (same as proxy_chat).
        gen = backend.responses_stream(body, api_key)
        try:
            first_chunk = await gen.__anext__()
        except UpstreamProviderError as e:
            await _log_event(
                db, config.user_id, "openai", model, "responses",
                0, 0, 0, int((time.time() - start_ts) * 1000), False, "error",
                channel=req_channel,
            )
            logger.warning(
                "[LLM-PROXY] %s upstream %d for user=%s model=%s body=%r",
                e.provider, e.status, config.user_id[:8], model, e.body,
            )
            try:
                detail = e.body.decode("utf-8", errors="replace")
            except Exception:
                detail = str(e.body)
            if settings.security_leak_filter:
                from app.services.model_alias import scrub_provider_names
                detail = scrub_provider_names(detail)
            raise HTTPException(e.status, detail=detail)
        except StopAsyncIteration:
            first_chunk = b""

        collected_bytes = bytearray(first_chunk)

        async def stream_and_log():
            try:
                if first_chunk:
                    yield first_chunk
                async for chunk in gen:
                    collected_bytes.extend(chunk)
                    yield chunk
            finally:
                latency = int((time.time() - start_ts) * 1000)
                inp, out, cached, cache_write = _extract_responses_usage(
                    bytes(collected_bytes)
                )
                # W0.2b usage-side [CACHE] line (same format as /chat so the
                # per-tenant hit-ratio grep spans both wires).
                logger.info(
                    "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
                    "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s",
                    config.user_id[:8], model, has_cache_key, cache_retention,
                    inp, cached, cache_write,
                )
                cost = _calc_cost_cents(
                    model, inp, out,
                    cached_tokens=cached, cache_write_tokens=cache_write,
                )
                # Do NOT reuse `db` here — the Depends(get_db) session is
                # closed before the response body streams (see the
                # proxy_chat comment). Own a fresh session, and shield it:
                # this finally is where a client disconnect delivers
                # cancellation.
                async def _log_usage() -> None:
                    from app.db.database import async_session_maker
                    async with async_session_maker() as _log_db:
                        await _log_event(
                            _log_db, config.user_id, "openai", model, "responses",
                            inp, out, cost, latency, False,
                            cached_tokens=cached,
                            cache_write_tokens=cache_write,
                            channel=req_channel,
                        )

                try:
                    await asyncio.shield(asyncio.create_task(_log_usage()))
                except Exception as e:
                    logger.warning("Failed to log usage event: %s", e)

        return StreamingResponse(
            stream_and_log(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                **resolved_model_header,
            },
        )
    else:
        # Non-streaming
        try:
            resp = await backend.responses(body, api_key)
        except Exception as e:
            latency = int((time.time() - start_ts) * 1000)
            await _log_event(
                db, config.user_id, "openai", model, "responses",
                0, 0, 0, latency, False, "error",
                channel=req_channel,
            )
            raise HTTPException(502, f"Provider error: {e}")
        if resp.status_code >= 400:
            body_bytes = resp.content
            await _log_event(
                db, config.user_id, "openai", model, "responses",
                0, 0, 0, int((time.time() - start_ts) * 1000), False, "error",
                channel=req_channel,
            )
            try:
                detail = body_bytes.decode("utf-8", errors="replace")
            except Exception:
                detail = str(body_bytes)
            if settings.security_leak_filter:
                from app.services.model_alias import scrub_provider_names
                detail = scrub_provider_names(detail)
            raise HTTPException(resp.status_code, detail=detail)

        latency = int((time.time() - start_ts) * 1000)

        resp_data = resp.json()
        usage = resp_data.get("usage", {})
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cached = _extract_responses_cached_tokens(usage)
        cache_write = _extract_openai_cache_write_tokens(
            usage, details_key="input_tokens_details"
        )
        # W0.2b usage-side [CACHE] line (non-stream twin of the SSE path).
        logger.info(
            "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
            "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s",
            config.user_id[:8], model, has_cache_key, cache_retention,
            inp, cached, cache_write,
        )
        _debug_log_upstream_cache_headers(resp.headers, model)

        cost = _calc_cost_cents(
            model, inp, out,
            cached_tokens=cached, cache_write_tokens=cache_write,
        )
        await _log_event(
            db, config.user_id, "openai", model, "responses",
            inp, out, cost, latency, False,
            cached_tokens=cached,
            cache_write_tokens=cache_write,
            channel=req_channel,
        )

        from fastapi.responses import JSONResponse
        return JSONResponse(content=resp_data, headers=resolved_model_header)


@router.post("/openai/v1/embeddings")
async def proxy_openai_embeddings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """OpenAI SDK compatibility shim for embeddings — lets the agent's
    embedding service set base_url=.../llm/openai/v1 and auth with TOUP_TOKEN,
    so NO OpenAI key needs to live in the container (hardening-runbook Step 1).
    Inert until settings.embeddings_via_proxy is enabled."""
    return await proxy_embeddings(request, db)


@router.post("/embeddings")
async def proxy_embeddings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Proxy an embeddings request to OpenAI."""
    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    body = await request.json()
    model = body.get("model", "text-embedding-3-small")

    # Prefer the user's auto-provisioned per-project key (β architecture);
    # fall back to platform master if not yet provisioned. Same pattern as
    # _route_chat for OpenAI; embeddings stays OpenAI-only for now.
    api_key = config.bundle_openai_api_key or settings.platform_openai_api_key
    if not api_key:
        raise HTTPException(500, "Platform OpenAI key not configured")

    # Budget check
    budget_result = await _check_budget(config, "openai", db)
    if budget_result == "monthly_exceeded":
        raise HTTPException(429, "Monthly OpenAI budget exceeded")

    start_ts = time.time()
    try:
        resp = await _openai.embeddings(body, api_key)
    except Exception as e:
        latency = int((time.time() - start_ts) * 1000)
        await _log_event(db, config.user_id, "openai", model, "embeddings", 0, 0, 0, latency, status="error")
        raise HTTPException(502, f"OpenAI error: {e}")

    latency = int((time.time() - start_ts) * 1000)
    resp_data = resp.json()

    # OpenAI embeddings don't return token counts in the same way — estimate
    usage = resp_data.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    cost = _embedding_cost_cents(total_tokens)

    await _log_event(db, config.user_id, "openai", model, "embeddings", total_tokens, 0, cost, latency)

    return resp_data


@router.post("/openai/v1/images/generations")
async def proxy_openai_images(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Proxy an OpenAI image-generation request (gpt-image-1 / dall-e) for a
    BUNDLE-mode agent, and charge the user per image inline.

    This is the bundle counterpart of the manual-mode path: a manual/BYO agent
    calls OpenAI directly and self-reports via /credits/agent-charge, whereas a
    bundle agent's client is pointed at .../llm/openai/v1 (no OpenAI key of its
    own), so image generation MUST be proxied here — and this is the one place
    with DB access to deduct credits. The per-tenant OpenAI project key is
    applied outbound exactly like _route_chat / proxy_embeddings.

    Billing: image generation is priced PER IMAGE by (size, quality), not by
    tokens, so we bypass _log_event's token→credit conversion and call
    try_charge directly with the per-image cost; _log_event is still invoked
    (0 tokens) so the LLMProxyEvent cost shows in usage/budget dashboards.
    """
    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    # Free-tier monthly image cap (audit-2026 re-audit round 7): the same hard
    # product limit the Kie route enforces. Without it here, a free-tier user
    # bypasses the cap by routing image generation through the OpenAI proxy.
    from app.services.credit_service import (
        reserve_free_image_slot, release_free_image_slot, settle_free_image_slot,
    )
    _exceeded, _used, _limit, _img_slot = await reserve_free_image_slot(db, config.user_id)
    if _exceeded:
        raise HTTPException(status_code=429, detail={
            "code": "image_quota_exceeded", "used": _used, "limit": _limit,
            "message": (f"Your free plan includes {_limit} images per month, and "
                        f"you've used them all. Upgrade for unlimited images."),
        })
    body = await request.json()
    model = body.get("model") or getattr(settings, "image_gen_model", "gpt-image-1")
    size = body.get("size") or getattr(settings, "image_gen_default_size", "1024x1024")
    quality = body.get("quality") or getattr(settings, "image_gen_default_quality", "high")
    n = int(body.get("n", 1) or 1)

    api_key = config.bundle_openai_api_key or settings.platform_openai_api_key
    if not api_key:
        raise HTTPException(500, "Platform OpenAI key not configured")

    # Budget check — image cost lands on the tenant's OpenAI allocation.
    budget_result = await _check_budget(config, "openai", db)
    if budget_result == "monthly_exceeded":
        raise HTTPException(429, "Monthly OpenAI budget exceeded")

    start_ts = time.time()
    try:
        resp = await _openai.images(body, api_key)
    except Exception as e:
        latency = int((time.time() - start_ts) * 1000)
        await _log_event(db, config.user_id, "openai", model, "images", 0, 0, 0, latency, status="error")
        await release_free_image_slot(db, _img_slot)   # generation failed → free the slot
        raise HTTPException(502, f"OpenAI image error: {e}")

    latency = int((time.time() - start_ts) * 1000)
    if resp.status_code >= 400:
        # Surface OpenAI's error verbatim (e.g. org-not-verified for gpt-image-1)
        # so the agent can retry with the fallback model.
        await _log_event(db, config.user_id, "openai", model, "images", 0, 0, 0, latency, status="error")
        await release_free_image_slot(db, _img_slot)   # no image produced → free the slot
        try:
            detail = resp.json()
        except Exception:
            detail = {"error": resp.text[:500]}
        from fastapi.responses import JSONResponse
        return JSONResponse(content=detail, status_code=resp.status_code)

    resp_data = resp.json()

    # Per-image cost × n. Charge BEFORE returning so a bundle user can't get a
    # free image if the response body write races. Idempotency key is a fresh
    # UUID per request (the agent does not retry the proxy call).
    from app.services.credit_service import (
        credit_service, image_generation_cost_cents, underlying_cost_to_credits,
    )
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE
    cents_each = image_generation_cost_cents(size, quality, model)
    total_cents = cents_each * n
    credits = max(_MIN_IMAGE_CREDITS, underlying_cost_to_credits(total_cents))
    charge_id = str(uuid.uuid4())
    try:
        await credit_service.try_charge(
            db, config.user_id, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, credits,
            idempotency_key=charge_id, event_id=charge_id,
            model=model, provider="openai",
            underlying_cost_cents=total_cents,
            metadata={"endpoint": "images", "size": size, "quality": quality, "n": n},
            already_incurred=True,   # image exists; the cap gates admission, not settlement
        )
    except Exception:
        logger.exception(
            "[credits] image try_charge failed user=%s model=%s size=%s q=%s",
            config.user_id[:8], model, size, quality,
        )
    await settle_free_image_slot(db, _img_slot)   # slot consumed → stop double-counting
    # Record the usage event (0 tokens → its internal charge is a no-op) and
    # commit both the charge ledger row and the event together.
    await _log_event(
        db, config.user_id, "openai", model, "images",
        0, 0, int(float(total_cents)), latency,
    )

    return resp_data


@router.post("/openai/v1/images/edits")
async def proxy_openai_image_edits(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Proxy an OpenAI image-EDIT (gpt-image-1 /images/edits) for a BUNDLE-mode
    agent, charging per image inline. Multipart counterpart of
    proxy_openai_images: a bundle agent's `client.images.edit()` POSTs
    multipart/form-data here (its base_url is .../llm/openai/v1, no OpenAI key
    of its own). Manual/BYO agents hit OpenAI directly and self-report via
    /credits/agent-charge, so they never reach this route.
    """
    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    # Free-tier monthly image cap — mirror the generate route so the edit path
    # can't bypass the cap either. RESERVE before generating (round 12 TOCTOU).
    from app.services.credit_service import (
        reserve_free_image_slot, release_free_image_slot, settle_free_image_slot,
    )
    _exceeded, _used, _limit, _img_slot = await reserve_free_image_slot(db, config.user_id)
    if _exceeded:
        raise HTTPException(status_code=429, detail={
            "code": "image_quota_exceeded", "used": _used, "limit": _limit,
            "message": (f"Your free plan includes {_limit} images per month, and "
                        f"you've used them all. Upgrade for unlimited images."),
        })
    form = await request.form()

    def _sval(key: str, default: str) -> str:
        v = form.get(key)
        return v if isinstance(v, str) and v else default

    model = _sval("model", getattr(settings, "image_gen_model", "gpt-image-1"))
    size = _sval("size", getattr(settings, "image_gen_default_size", "1024x1024"))
    quality = _sval("quality", getattr(settings, "image_gen_default_quality", "high"))
    prompt = _sval("prompt", "")
    try:
        n = int(_sval("n", "1"))
    except (TypeError, ValueError):
        n = 1

    # Collect the uploaded source image(s) + optional mask. The SDK sends the
    # field as "image" (gpt-image-1 also accepts "image[]" for multi-image
    # compositing). UploadFiles expose .read()/.filename/.content_type.
    files: list = []
    for _key in ("image", "image[]"):
        for _uf in form.getlist(_key):
            if hasattr(_uf, "read"):
                files.append(("image", (
                    getattr(_uf, "filename", None) or "image.png",
                    await _uf.read(),
                    getattr(_uf, "content_type", None) or "image/png",
                )))
    _mask = form.get("mask")
    if _mask is not None and hasattr(_mask, "read"):
        files.append(("mask", (
            getattr(_mask, "filename", None) or "mask.png",
            await _mask.read(),
            getattr(_mask, "content_type", None) or "image/png",
        )))
    if not files:
        raise HTTPException(400, "images/edits requires an 'image' file")

    data = {"model": model, "prompt": prompt, "size": size, "quality": quality, "n": n}

    api_key = config.bundle_openai_api_key or settings.platform_openai_api_key
    if not api_key:
        raise HTTPException(500, "Platform OpenAI key not configured")

    budget_result = await _check_budget(config, "openai", db)
    if budget_result == "monthly_exceeded":
        raise HTTPException(429, "Monthly OpenAI budget exceeded")

    start_ts = time.time()
    try:
        resp = await _openai.images_edit(data, files, api_key)
    except Exception as e:
        latency = int((time.time() - start_ts) * 1000)
        await _log_event(db, config.user_id, "openai", model, "images", 0, 0, 0, latency, status="error")
        await release_free_image_slot(db, _img_slot)   # generation failed → free the slot
        raise HTTPException(502, f"OpenAI image edit error: {e}")

    latency = int((time.time() - start_ts) * 1000)
    if resp.status_code >= 400:
        # Surface OpenAI's error verbatim (moderation, bad image, etc.).
        await _log_event(db, config.user_id, "openai", model, "images", 0, 0, 0, latency, status="error")
        await release_free_image_slot(db, _img_slot)   # no image produced → free the slot
        try:
            detail = resp.json()
        except Exception:
            detail = {"error": resp.text[:500]}
        from fastapi.responses import JSONResponse
        return JSONResponse(content=detail, status_code=resp.status_code)

    resp_data = resp.json()

    # Per-image cost × n, charged inline before returning (same pricing table as
    # generation). Fresh-UUID idempotency; the agent does not retry the proxy.
    from app.services.credit_service import (
        credit_service, image_generation_cost_cents, underlying_cost_to_credits,
    )
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE
    cents_each = image_generation_cost_cents(size, quality, model)
    total_cents = cents_each * n
    credits = max(_MIN_IMAGE_CREDITS, underlying_cost_to_credits(total_cents))
    charge_id = str(uuid.uuid4())
    try:
        await credit_service.try_charge(
            db, config.user_id, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, credits,
            idempotency_key=charge_id, event_id=charge_id,
            model=model, provider="openai",
            underlying_cost_cents=total_cents,
            metadata={"endpoint": "images", "op": "edit", "size": size, "quality": quality, "n": n},
            already_incurred=True,   # image exists; the cap gates admission, not settlement
        )
    except Exception:
        logger.exception(
            "[credits] image-edit try_charge failed user=%s model=%s size=%s q=%s",
            config.user_id[:8], model, size, quality,
        )
    await settle_free_image_slot(db, _img_slot)   # slot consumed → stop double-counting
    await _log_event(
        db, config.user_id, "openai", model, "images",
        0, 0, int(float(total_cents)), latency,
    )

    return resp_data


@router.post("/kie/image")
async def proxy_kie_image(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Generate or edit an image via Kie.ai (Nano Banana Pro) for ANY agent.

    The ONE shared platform Kie key lives here (never in agents, like the bundle
    OpenAI key). The agent's generate_image/edit_image tools POST
    {mode, prompt, size, image_b64} here; we enforce the free-tier monthly image
    cap, run Kie's async job (create → poll → fetch), charge the user per image,
    and return the result as base64. Failure semantics the agent relies on:
      • 429 {code:image_quota_exceeded} → free cap hit → show upgrade, NO fallback
      • 502 {code:kie_failed}          → Kie error → agent falls back to gpt-image-2
      • 200 {b64,...}                  → deliver the image
    """
    import base64 as _b64
    from app.services import kie_client
    from app.services.credit_service import (
        credit_service, underlying_cost_to_credits,
        reserve_free_image_slot, release_free_image_slot, settle_free_image_slot,
    )
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE

    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    body = await request.json()
    mode = (body.get("mode") or "generate").strip().lower()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")

    # Free-tier monthly image cap — a hard product limit. RESERVE the slot BEFORE
    # spending any Kie credits (round 12): the reservation is TOCTOU-safe, so
    # concurrent requests can't all pass a stale count and blow past the cap.
    # (This also fixes the prior instance-attr call that raised AttributeError.)
    exceeded, used, limit, _img_slot = await reserve_free_image_slot(db, config.user_id)
    if exceeded:
        raise HTTPException(status_code=429, detail={
            "code": "image_quota_exceeded", "used": used, "limit": limit,
            "message": (f"Your free plan includes {limit} images per month, and "
                        f"you've used them all. Upgrade for unlimited images."),
        })

    start_ts = time.time()
    try:
        if mode == "edit":
            image_b64 = body.get("image_b64")
            if not image_b64:
                raise HTTPException(400, "edit mode requires image_b64")
            try:
                src = _b64.b64decode(image_b64)
            except Exception:
                raise HTTPException(400, "image_b64 is not valid base64")
            result = await kie_client.edit(prompt, src, body.get("image_mime") or "image/png")
        else:
            result = await kie_client.generate(prompt, body.get("size"))
    except kie_client.KieError as e:
        latency = int((time.time() - start_ts) * 1000)
        await _log_event(db, config.user_id, "kie", settings.kie_image_model, "images",
                         0, 0, 0, latency, status="error")
        await release_free_image_slot(db, _img_slot)   # generation failed → free the slot
        raise HTTPException(status_code=502, detail={
            "code": "kie_failed", "moderation": bool(e.moderation), "message": str(e)[:300],
        })

    latency = int((time.time() - start_ts) * 1000)

    # Charge: Kie credits × kie_credit_cents → our credits (1¢ = 1 credit).
    cents = (float(result.credits_consumed) * float(settings.kie_credit_cents)
             if result.credits_consumed else float(settings.kie_fallback_cents))
    cents_d = Decimal(str(round(cents, 4)))
    credits = max(_MIN_IMAGE_CREDITS, underlying_cost_to_credits(cents_d))
    charge_id = str(uuid.uuid4())
    try:
        await credit_service.try_charge(
            db, config.user_id, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, credits,
            idempotency_key=charge_id, event_id=charge_id,
            model=result.model, provider="kie",
            underlying_cost_cents=cents_d,
            metadata={"endpoint": "kie_image", "mode": mode,
                      "kie_credits": result.credits_consumed},
            already_incurred=True,   # image exists; the cap gates admission, not settlement
        )
    except Exception:
        logger.exception("[credits] kie image try_charge failed user=%s", config.user_id[:8])
    await settle_free_image_slot(db, _img_slot)   # slot consumed → stop double-counting
    await _log_event(db, config.user_id, "kie", result.model, "images", 0, 0, int(cents), latency)

    return {
        "b64": _b64.b64encode(result.image_bytes).decode(),
        "mime": result.mime, "model": result.model, "credits": float(credits),
    }


# ── Async image jobs (2026-07-23) ─────────────────────────────────────────
# The synchronous route above holds one HTTP request open for the whole render.
# Kie's latency is wildly variable (measured 25/36/39/74s and a 399s success),
# so a fixed budget either abandons a task the user ALREADY PAID 18 credits for
# — the founder's 20:08 edit was still `running` on Kie, charged, and never
# delivered — or exceeds what an HTTP hop will tolerate. start+poll keeps every
# request short and lets a 400s render finish and still be delivered.

@router.post("/kie/image/start")
async def proxy_kie_image_start(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a Kie image job. Returns fast — does NOT wait for the render.

    Returns {task_id, reservation_id}. The caller polls /kie/image/poll. The
    free-tier slot is reserved here (TOCTOU-safe) and is settled or released by
    the poll endpoint, so the reservation travels with the job.
    """
    from app.services import kie_client
    from app.services.credit_service import reserve_free_image_slot, release_free_image_slot

    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    body = await request.json()
    mode = (body.get("mode") or "generate").strip().lower()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")

    exceeded, used, limit, _img_slot = await reserve_free_image_slot(db, config.user_id)
    if exceeded:
        raise HTTPException(status_code=429, detail={
            "code": "image_quota_exceeded", "used": used, "limit": limit,
            "message": (f"Your free plan includes {limit} images per month, and "
                        f"you've used them all. Upgrade for unlimited images."),
        })

    src: Optional[bytes] = None
    if mode == "edit":
        image_b64 = body.get("image_b64")
        if not image_b64:
            await release_free_image_slot(db, _img_slot)
            raise HTTPException(400, "edit mode requires image_b64")
        try:
            import base64 as _b64m
            src = _b64m.b64decode(image_b64)
        except Exception:
            await release_free_image_slot(db, _img_slot)
            raise HTTPException(400, "image_b64 is not valid base64")

    try:
        task_id = await kie_client.start_task(
            mode, prompt, size=body.get("size"), image_bytes=src,
            mime=body.get("image_mime") or "image/png")
    except kie_client.KieError as e:
        await release_free_image_slot(db, _img_slot)   # nothing started → free it
        raise HTTPException(status_code=502, detail={
            "code": "kie_failed", "moderation": bool(e.moderation),
            "message": str(e)[:300],
        })

    # Hold the estimated cost NOW that a real (billable) render is running.
    # The charge used to live only in /poll, so a job that was started and never
    # polled to completion — client crash, agent rollout, deadline overrun —
    # burned real Kie spend that was never billed. The hold is keyed on the
    # task so /poll can find and settle it without the client carrying the id;
    # if the job is abandoned the hold simply stands (no expiry sweeper), which
    # is the correct outcome: the render happened, so it stays paid for.
    # settle() clamps the final charge to this estimate and refunds the rest.
    from app.services.credit_service import credit_service, underlying_cost_to_credits
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE
    _est_cents = Decimal(str(round(float(settings.kie_fallback_cents), 4)))
    _est_credits = max(_MIN_IMAGE_CREDITS, underlying_cost_to_credits(_est_cents))
    try:
        await credit_service.reserve(
            db, config.user_id, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, _est_credits,
            ttl_seconds=int(float(getattr(settings, "kie_job_timeout_s", 420.0))) + 600,
            idempotency_key=f"kie_task:{task_id}",
            event_id=f"kie_task:{task_id}",
            metadata={"endpoint": "kie_image_start", "mode": mode, "task_id": task_id},
        )
        await db.commit()
    except Exception:
        # Never fail a started render on the accounting hop; /poll still charges
        # directly when no hold is found.
        logger.exception("[credits] kie start reserve failed user=%s task=%s",
                         config.user_id[:8], task_id)

    return {"task_id": task_id, "reservation_id": _img_slot, "status": "pending"}


@router.post("/kie/image/poll")
async def proxy_kie_image_poll(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Probe a Kie job once. Short request regardless of render time.

    pending → {status:"pending"}; fail → 502 (moderation flagged when it was a
    content refusal); success → charge once, settle the slot, return the image.
    """
    import base64 as _b64
    from app.services import kie_client
    from app.services.credit_service import (
        credit_service, underlying_cost_to_credits,
        release_free_image_slot, settle_free_image_slot,
        find_open_reservation_by_key,
    )
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE

    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    body = await request.json()
    task_id = (body.get("task_id") or "").strip()
    if not task_id:
        raise HTTPException(400, "task_id is required")
    reservation_id = body.get("reservation_id")

    try:
        st = await kie_client.poll_task(task_id)
    except kie_client.KieError as e:
        raise HTTPException(status_code=502, detail={
            "code": "kie_failed", "moderation": bool(e.moderation), "message": str(e)[:300],
        })

    if st.get("state") == "pending":
        return {"status": "pending"}

    if st.get("state") == "fail":
        await release_free_image_slot(db, reservation_id)
        # No image was produced → give back the hold /start took.
        _hold_id = await find_open_reservation_by_key(
            db, config.user_id, f"kie_task:{task_id}")
        if _hold_id:
            await credit_service.refund(db, _hold_id, reason="kie_render_failed")
            await db.commit()
        await _log_event(db, config.user_id, "kie", settings.kie_image_model, "images",
                         0, 0, 0, 0, status="error")
        raise HTTPException(status_code=502, detail={
            "code": "kie_failed", "moderation": bool(st.get("moderation")),
            "message": str(st.get("message"))[:300],
        })

    # success — download, charge ONCE (task_id is the idempotency key so repeat
    # polls after a dropped response can never double-bill), settle the slot.
    try:
        img, mime = await kie_client.fetch_result(st["result_url"])
    except kie_client.KieError as e:
        raise HTTPException(status_code=502, detail={
            "code": "kie_failed", "moderation": False, "message": str(e)[:300],
        })

    kie_credits = float(st.get("credits") or 0.0)
    cents = (kie_credits * float(settings.kie_credit_cents)
             if kie_credits else float(settings.kie_fallback_cents))
    cents_d = Decimal(str(round(cents, 4)))
    credits = max(_MIN_IMAGE_CREDITS, underlying_cost_to_credits(cents_d))
    charge_key = f"kie_task:{task_id}"
    try:
        # Settle the hold /start took (clamped to the estimate; the difference
        # is refunded). Falls back to a direct charge only for a job started by
        # a build that predates the hold, so neither path can double-bill.
        _hold_id = await find_open_reservation_by_key(db, config.user_id, charge_key)
        if _hold_id:
            await credit_service.settle(
                db, _hold_id, credits,
                metadata={"endpoint": "kie_image_poll", "kie_credits": kie_credits,
                          "task_id": task_id},
            )
        else:
            await credit_service.try_charge(
                db, config.user_id, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, credits,
                idempotency_key=charge_key, event_id=charge_key,
                model=settings.kie_image_model, provider="kie",
                underlying_cost_cents=cents_d,
                metadata={"endpoint": "kie_image_poll", "kie_credits": kie_credits,
                          "task_id": task_id},
                already_incurred=True,   # image exists; the cap gates admission, not settlement
            )
    except Exception:
        logger.exception("[credits] kie image charge/settle failed user=%s", config.user_id[:8])
    await settle_free_image_slot(db, reservation_id)
    await _log_event(db, config.user_id, "kie", settings.kie_image_model, "images",
                     0, 0, int(cents), 0)

    return {
        "status": "success", "b64": _b64.b64encode(img).decode(),
        "mime": mime, "model": settings.kie_image_model, "credits": float(credits),
    }


@router.get("/usage", response_model=UsageResponse)
async def get_proxy_usage(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Return the calling agent's current usage and budget."""
    config = await _auth_agent(request, db)
    period_start = config.bundle_period_start or config.bundle_started_at

    if not period_start:
        return UsageResponse(
            anthropic_monthly_cents=0,
            anthropic_daily_cents=0,
            openai_monthly_cents=0,
            anthropic_budget_cents=config.bundle_anthropic_budget_cents,
            anthropic_daily_cap_cents=config.bundle_anthropic_daily_cap_cents,
            openai_budget_cents=config.bundle_openai_budget_cents,
        )

    anthropic_monthly = await _get_spend(db, config.user_id, "anthropic", period_start, "monthly")
    anthropic_daily = await _get_spend(db, config.user_id, "anthropic", _today_utc_start(), "daily")
    openai_monthly = await _get_spend(db, config.user_id, "openai", period_start, "monthly")

    return UsageResponse(
        anthropic_monthly_cents=anthropic_monthly,
        anthropic_daily_cents=anthropic_daily,
        openai_monthly_cents=openai_monthly,
        anthropic_budget_cents=config.bundle_anthropic_budget_cents,
        anthropic_daily_cap_cents=config.bundle_anthropic_daily_cap_cents,
        openai_budget_cents=config.bundle_openai_budget_cents,
        period_start=period_start.isoformat() if period_start else None,
        period_end=config.bundle_period_end.isoformat() if config.bundle_period_end else None,
    )


# ── Format adapter: Anthropic ↔ OpenAI ──────────────────────────────


def _anthropic_to_openai_request(body: dict) -> dict:
    """
    Convert an Anthropic Messages API request to OpenAI Chat Completions format.
    Handles system prompts, message roles, and basic content types.
    """
    messages = []

    # System prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system blocks
            text_parts = [b["text"] for b in system if b.get("type") == "text"]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # Messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Anthropic content can be a list of blocks or a string
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    text_parts.append(f"[Tool result: {block.get('content', '')}]")
                elif block.get("type") == "tool_use":
                    # Skip tool_use blocks in conversion — they're assistant-generated
                    pass
            content = "\n".join(text_parts)

        messages.append({"role": role, "content": content})

    # Pick a reasonable OpenAI model as fallback
    model = "gpt-4o-mini"
    max_tokens = body.get("max_tokens", 4096)

    result: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("stream"):
        result["stream"] = True
        result["stream_options"] = {"include_usage": True}

    return result


# ── Admin stats ──────────────────────────────────────────────────────


class AdminStatsResponse(BaseModel):
    total_requests_today: int
    # float, not int: cost_cents is Numeric(12,4) since R-3 (alembic 084) —
    # a fractional Decimal into an int field is a ValidationError, i.e. a
    # 500 on the admin dashboard the first sub-cent call after the deploy.
    total_cost_cents_today: float
    anthropic_cost_cents_today: float
    openai_cost_cents_today: float
    fallback_count_today: int
    error_count_today: int
    top_users: list[dict]


@admin_router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: aggregate LLM proxy stats for today.

    require_admin, same as /cache-daily. The previous hand-rolled check
    called get_current_user with two positional args — binding the SESSION
    to the ``credentials`` parameter — so it AttributeError'd before any
    auth strategy ran and the blanket except 403'd every caller, admins
    included, since the route shipped.
    """

    today = _today_utc_start()

    # Total requests + cost
    result = await db.execute(
        select(
            func.count().label("cnt"),
            func.coalesce(func.sum(LLMProxyEvent.cost_cents), 0).label("cost"),
        ).where(LLMProxyEvent.created_at >= today)
    )
    row = result.first()
    total_requests = row.cnt if row else 0
    total_cost = row.cost if row else 0

    # By provider
    by_provider = await db.execute(
        select(
            LLMProxyEvent.provider,
            func.coalesce(func.sum(LLMProxyEvent.cost_cents), 0).label("cost"),
        ).where(LLMProxyEvent.created_at >= today).group_by(LLMProxyEvent.provider)
    )
    provider_costs = {r.provider: round(float(r.cost), 4) for r in by_provider}

    # Fallback + error counts
    fallback_result = await db.execute(
        select(func.count()).where(
            LLMProxyEvent.created_at >= today,
            LLMProxyEvent.was_fallback == True,
        )
    )
    fallback_count = fallback_result.scalar() or 0

    error_result = await db.execute(
        select(func.count()).where(
            LLMProxyEvent.created_at >= today,
            LLMProxyEvent.status == "error",
        )
    )
    error_count = error_result.scalar() or 0

    # Top 10 users by spend
    top_users_result = await db.execute(
        select(
            LLMProxyEvent.user_id,
            func.coalesce(func.sum(LLMProxyEvent.cost_cents), 0).label("cost"),
            func.count().label("cnt"),
        )
        .where(LLMProxyEvent.created_at >= today)
        .group_by(LLMProxyEvent.user_id)
        .order_by(func.sum(LLMProxyEvent.cost_cents).desc())
        .limit(10)
    )
    top_users = [
        {"user_id": r.user_id[:8], "cost_cents": round(float(r.cost), 4), "requests": r.cnt}
        for r in top_users_result
    ]

    return AdminStatsResponse(
        total_requests_today=total_requests,
        total_cost_cents_today=total_cost,
        anthropic_cost_cents_today=provider_costs.get("anthropic", 0),
        openai_cost_cents_today=provider_costs.get("openai", 0),
        fallback_count_today=fallback_count,
        error_count_today=error_count,
        top_users=top_users,
    )


class CacheDailyRow(BaseModel):
    day: str
    prompt_tokens: int
    cached_tokens: int
    # Prompt-cache WRITE volume (alembic 083). 0 for days recorded before
    # 083 — NULLs aggregate as 0 here, same convention as cached_tokens.
    cache_write_tokens: int
    cache_hit_ratio: float
    calls: int


class CacheDailyResponse(BaseModel):
    days: int
    user_id: Optional[str] = None
    rows: list[CacheDailyRow]


@admin_router.get("/cache-daily", response_model=CacheDailyResponse)
async def get_admin_cache_daily(
    days: int = Query(7, ge=1, le=90),
    user_id: Optional[str] = Query(None),
    endpoint: Optional[str] = Query(None, pattern=r"^[a-z_]{1,32}$|^all$"),
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: per-day prompt-cache hit telemetry (F-7 / A9-1).

    Aggregates llm_proxy_events over the last `days` UTC days, optionally
    filtered to one user. cache_hit_ratio = sum(cached_tokens) /
    sum(input_tokens); rows recorded before migration 075 have NULL
    cached_tokens and count as 0 hits, so early ratios understate.

    Review pr5-#1: filters to successful calls on ONE endpoint —
    voice/embeddings/image and error rows would dilute prompt_tokens and
    inflate calls, understating the hit ratio this exists to watch. Pass
    endpoint=all for the unfiltered view.

    The default is DERIVED, not the literal "chat" it used to be. That
    literal was correct when it was written and silently wrong from the
    moment the fleet moved to gpt-5.6-terra on the Responses wire (#507):
    agent turns began writing endpoint="responses", so the default view
    stopped containing a single one of them. Measured 2026-08-08 over 14
    days — chat 1,967 calls / 9.0M input tokens / 49.0% cached, responses
    783 calls / 11.2M input tokens / 18.3% cached. The endpoint this view
    defaulted to was the healthy half; the half carrying 55% of all input
    tokens at a third of the hit rate was the one it excluded, and the
    dashboard read green throughout.

    Deriving it from the fleet's own model resolution means the next wire
    migration moves this view with it instead of quietly emptying it —
    the same reason `wire_api_for` derives the wire rather than reading a
    flag that can disagree with the model.
    """
    if endpoint is None:
        from app.services.model_resolver import default_model, wire_api_for

        endpoint = wire_api_for(default_model())
    since = _today_utc_start() - timedelta(days=days - 1)
    day_col = func.date(LLMProxyEvent.created_at)
    stmt = (
        select(
            day_col.label("day"),
            func.coalesce(func.sum(LLMProxyEvent.input_tokens), 0).label("prompt_tokens"),
            func.coalesce(func.sum(LLMProxyEvent.cached_tokens), 0).label("cached_tokens"),
            func.coalesce(func.sum(LLMProxyEvent.cache_write_tokens), 0).label("cache_write_tokens"),
            func.count().label("calls"),
        )
        .where(LLMProxyEvent.created_at >= since)
        .where(LLMProxyEvent.status == "ok")
        .group_by(day_col)
        .order_by(day_col.desc())
    )
    if endpoint != "all":
        stmt = stmt.where(LLMProxyEvent.endpoint == endpoint)
    if user_id:
        stmt = stmt.where(LLMProxyEvent.user_id == user_id)
    result = await db.execute(stmt)

    rows = []
    for r in result:
        prompt = int(r.prompt_tokens or 0)
        cached = int(r.cached_tokens or 0)
        rows.append(CacheDailyRow(
            day=str(r.day),
            prompt_tokens=prompt,
            cached_tokens=cached,
            cache_write_tokens=int(r.cache_write_tokens or 0),
            cache_hit_ratio=round(cached / prompt, 4) if prompt > 0 else 0.0,
            calls=int(r.calls or 0),
        ))

    return CacheDailyResponse(days=days, user_id=user_id, rows=rows)
