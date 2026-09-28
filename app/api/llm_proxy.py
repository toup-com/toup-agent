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
import os
import re
import secrets
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
    from app.services.model_resolver import long_context_price_multipliers
    input_mult, output_mult = long_context_price_multipliers(model, input_tokens)
    cost_usd = (
        (base_input * pricing["input"] / 1000)
        + (cached * cached_rate / 1000 if cached else 0.0)
        + (written * write_rate / 1000 if written else 0.0)
    ) * input_mult + (
        output_tokens * pricing["output"] / 1000
    ) * output_mult
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


#: text-embedding-3-small: $0.02 per MILLION tokens = $0.00002 per 1,000.
#: Same per-1k convention as ``settings.pricing_per_1k``.
#:
#: MODEL-BLIND, knowingly: the embeddings route prices every request at
#: this one rate. The only production caller (embedding_service) pins
#: text-embedding-3-small; a hypothetical 3-large request ($0.13/1M)
#: would be under-recorded 6.5x — an undercharge, which the R-3
#: never-higher rule permits. If a second embedding model ever ships,
#: this constant must become a lookup keyed on the request's model.
_EMBEDDING_USD_PER_1K = 0.00002


def _embedding_cost_cents(total_tokens: int) -> Decimal:
    """Cost of an embeddings call in cents, unfloored.

    The rate is **per 1,000 tokens**, not per token. The route's original
    inline math was ``total_tokens * 0.00002``, which applied the per-1k
    rate once per token and overstated every embedding by 1000x — the
    sibling ``_calc_cost_cents`` divides by 1000 for exactly this reason,
    because the table it reads is named ``pricing_per_1k``.

    The 1¢ floor hid it: nearly every embeddings call clamped to 1¢
    whatever the arithmetic said. Taking the floor off made the pricing
    error load-bearing instead of cosmetic, so it is fixed here with it.
    Cross-check: ``docs/credits/coverage.md`` and
    ``credit_health_monitor.py`` both state that a 15-token call "truly
    costs about $0.0000003" — which is 15/1000 × $0.00002, and 1000x
    below what this function used to return.
    """
    return _never_higher_cents(total_tokens / 1000 * _EMBEDDING_USD_PER_1K)


# ── Usage event logging ──────────────────────────────────────────────


#: Header the agent uses to report which surface a turn came from.
CHANNEL_HEADER = "x-toup-channel"

#: R44 — the agent tags platform overhead (a prompt-cache warm) so it is not
#: billed to the user. `_log_event` already exempts any operation_type
#: starting with "system." from BOTH the credit charge and `_get_spend`, so
#: this header reaches straight into the money path and is the one place a
#: caller could try to exempt its own traffic.
OPERATION_TYPE_HEADER = "x-toup-operation-type"

#: An ALLOWLIST, not a "system.*" prefix test. The prefix alone would let any
#: holder of an agent key invent `system.whatever` and stop paying for chat;
#: a closed set means adding an exemption is a platform change, reviewed here.
_ALLOWED_SYSTEM_OPERATIONS = frozenset({"system.cache_warm"})


def _system_operation_for(raw: Optional[str], body: dict) -> Optional[str]:
    """The operation_type to record, or None (→ user-attributable).

    Two gates, because the header is client-supplied and lands on the billing
    decision. The value must be one we issue, AND the request must still LOOK
    like the operation it claims to be: a cache warm asks for at most a
    handful of output tokens, forbids tool calls and sends a single input
    item. A chat turn dressed in this header fails the shape test and is
    billed normally, so the worst a leaked agent key buys is a free 16-token
    completion — not free chat.
    """
    value = (raw or "").strip().lower()
    if not value:
        return None
    if value not in _ALLOWED_SYSTEM_OPERATIONS:
        logger.warning(
            "[LLM-PROXY] rejected unknown operation_type header %r — billing "
            "as a user request", value[:64],
        )
        return None
    if value == "system.cache_warm":
        try:
            max_out = int(body.get("max_output_tokens") or 0)
            items = body.get("input")
            shape_ok = (
                0 < max_out <= 32
                and body.get("tool_choice") == "none"
                and isinstance(items, list)
                and len(items) == 1
            )
        except Exception:  # noqa: BLE001 — a malformed body is not a warm
            shape_ok = False
        if not shape_ok:
            logger.warning(
                "[LLM-PROXY] operation_type=%s claimed on a request that is "
                "not shaped like a warm — billing as a user request", value,
            )
            return None
    return value


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
    wire_suffix: str = "",
):
    """Log an LLM usage event.

    `wire_suffix` (R48): a pre-rendered ` rid=… trace=…` fragment appended
    to the `llm_proxy` summary line so that line joins the two [CACHE]
    lines of the same request. Built by `_wire_join_suffix`, which returns
    `""` unless wire observability is on for the user — so the default and
    every caller that does not pass it leave the summary line
    byte-identical to R47. It is a string rather than the ids themselves
    because the gate belongs to one function: a second `if flag` here
    could drift out of step with the [CACHE] lines and produce a summary
    line nothing can join to.

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
                # reason="daily_cap_exceeded".
                #
                # What stops the loop, per dimension:
                #   * BALANCE — try_charge debits what the wallet holds even on
                #     a refusal and drives the MESSAGE bucket to zero, so the
                #     pre-flight below (PREFLIGHT_QUOTE_CREDITS, asked before
                #     the provider call) refuses the next turn.
                #   * DAILY CAP — an incurred charge lands past the cap and the
                #     over-cap used_today refuses the next pre-flight. That
                #     half needs `credit_cap_admission_control` AND a non-NULL
                #     cap; every cap went NULL on 2026-08-29, so it is inert
                #     today and `already_incurred` reaches nothing but
                #     `ignore_daily_cap`.
                # An earlier version of this comment claimed the cap clause was
                # the whole stop and pointed at a pre-flight at :1017 that has
                # not been there for releases. Between them, 21 turns were
                # served unbilled to one account in 6h07m on 2026-09-15.
                already_incurred=True,
            )
            if not result.success:
                logger.warning(
                    "[credits] charge DENIED but response already served "
                    "user=%s model=%s reason=%s credits=%s settled=%s "
                    "shortfall=%s cost_cents=%s",
                    user_id[:8], model, result.reason, credits,
                    result.charged, result.shortfall, cost_cents,
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
        # cost_cents is a Decimal since R-3 and is usually SUB-CENT; %d
        # truncates toward zero, so every fractional call logged as 0 —
        # the exact class of call the floor removal exists to record.
        "cached=%d cost_cents=%s latency=%dms fallback=%s status=%s op=%s%s",
        user_id[:8], provider, model, input_tokens, output_tokens,
        cached_tokens or 0, cost_cents, latency_ms, was_fallback, status,
        operation_type or "user", wire_suffix,
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

    async def responses_stream(self, body: dict, api_key: str,
                               meta: Optional[dict] = None):
        """Streaming /v1/responses passthrough. Unlike chat_stream we do
        NOT inject stream_options — Responses streams always carry usage in
        the response.completed event (their stream_options only controls
        obfuscation); any client-sent value is forwarded untouched.

        R48: `meta`, when given, receives the upstream's opaque
        `x-request-id` under "request_id" the moment the response headers
        land. The httpx response object never leaves this generator — which
        is precisely why that id had no reader outside a DEBUG dump nobody
        enables, and why no slow turn could be joined to OpenAI's own trace.
        An out-parameter rather than a return value because this is an async
        generator; the handler pre-pulls the first chunk before it returns
        its StreamingResponse, so by the time anything reads `meta` the
        generator has already run past this line. A caller that passes
        nothing gets byte-identical behaviour.

        The write sits ABOVE the status check on purpose (R48 review, N6):
        a 4xx/5xx is precisely the response an OpenAI escalation needs the
        id for, and raising first left the handler's UpstreamProviderError
        WARNING with nothing to quote. It is also above `aiter_bytes()` for
        a second reason — on a client disconnect the body loop never
        completes, and a `meta` written after it would be empty on exactly
        the turns worth investigating (`test_meta_is_written_before_any_body_byte`).
        """
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
                if meta is not None:
                    meta["request_id"] = resp.headers.get("x-request-id") or ""
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


# ── Log-atom allowlist ───────────────────────────────────────────────
# One shared shape for every value this module prints that it did not
# author: a single run of printable, non-space ASCII. It is the same
# discipline as `_wire_trace_header` and `_wire_req_id` below, hoisted
# here because `_cache_log_fields` needs it and runs before them.
#
# What it is for, exactly: a `\n` inside a logged value ends the real
# line and starts a line of the SENDER's composition in a shared,
# two-replica platform log stream. That is a log-forgery primitive, and
# the [CACHE] lines are the substrate of gates G7/G8 — a forged line
# makes either of them count a request that never happened.
#
# Deliberately WIDE (any printable non-space ASCII) rather than a closed
# vocabulary: these fields report values that legitimately change without
# our involvement (a new retention spelling, a new provider id format),
# and a value we have never seen should still be REPORTED rather than
# silently read as absent. What it rejects is the log-structure set —
# newline, carriage return, tab, space, NUL, every other control
# character, and anything longer than the caller's bound.
#
# Rejection renders the vocabulary word `invalid` and echoes NOTHING of
# the rejected value: not a prefix, not its length, not a `%r`. Quoting
# an attacker-supplied string into the line is the bug the validation
# exists to prevent.
_LOG_ATOM_RE = re.compile(r"\A[\x21-\x7e]+\Z")


def _log_atom(value, *, max_len: int, fallback: str = "invalid") -> str:
    """`value` if it is one printable ASCII token within `max_len`, else `fallback`.

    The length guard runs BEFORE the regex so a megabyte of body field
    costs a `len()` and not a scan (the same ordering, and the same
    reason, as `_wire_trace_header`'s 12-char guard).
    """
    if not isinstance(value, str) or not value or len(value) > max_len:
        return fallback
    return value if _LOG_ATOM_RE.match(value) else fallback


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

    `retention` reports whichever cache-lifetime control the body actually
    carries, because the two spellings belong to different model
    generations and the fleet will cross between them:

      * `prompt_cache_retention` (pre-GPT-5.6) → reported verbatim, e.g. 24h
      * `prompt_cache_options.ttl` (GPT-5.6+)  → reported as `opt:30m`

    R48: this function read ONLY the legacy key, so the day the agent starts
    sending `prompt_cache_options` this line would have printed
    `retention=none` for every migrated request — the instrument that has to
    verify the migration going dark exactly on the migration, and reading as
    "the fleet stopped asking for retention" rather than "the fleet moved".
    The legacy field wins when both are present: that is the one the request
    is actually sending to a pre-5.6 model, and a body carrying both is a
    bug we would want to see as the legacy value it will be billed under.

    `retention` IS A CLIENT-SUPPLIED BODY STRING on both routes, so it goes
    out through `_log_atom` (FINAL review round 2, BLOCKING-1). Before that
    guard, `prompt_cache_options.ttl` — the route this patch ADDED — was
    rendered with an f-string and no check of any kind, so
    `{"ttl": "5m\\n[CACHE] user=deadbeef …"}` put a well-formed forged
    `[CACHE]` line into the platform stream, and `{"ttl": {"a": "b"}}`
    produced `opt:{'a': 'b'}` — the exact wholesale-`str()` shape that made
    `_tool_choice_kind` a blocking finding one round earlier. The legacy
    `prompt_cache_retention` route had the same hole at `4f0e9fe1` and is
    closed by the same call. Reproduced at unit and handler level before
    the fix; see the mutation set in the NOTES.

    A rejected value renders `retention=invalid`, which is also how an
    operator learns a caller is sending something that is not a TTL —
    `none` would have hidden it. Every real spelling (`24h`, `opt:30m`,
    `opt:1h`) is a printable ASCII token well under 32 characters and is
    reported verbatim, so no line the fleet produces today changes.

    NOT closed by this, and the [CACHE] line's remaining caller-written
    field: `model=`. It is rendered raw at 21 `model=%s` sites in this
    module — all four [CACHE] lines, the `llm_proxy` summary, three
    `[credits]` lines, both `[LLM-PROXY]` upstream WARNINGs, the
    dedup/cap/prune warnings and the DEBUG header dump — all PRE-EXISTING
    at `4f0e9fe1`, and closing only the [CACHE] four would leave the class
    open through the other 17 while looking closed. Four of the 21 are
    inside `_log_event`, which embeddings, images, kie and internal_llm
    also call, so a full fix is a behaviour delta outside a log-only
    patch's envelope. `/responses` only requires
    `str(model).lower().startswith(("gpt","o1","o3","o4"))`, which
    `"gpt-5.6\\n[CACHE] …"` satisfies. Consequence, stated where it
    matters rather than only in a document: **G7/G8 rows are
    caller-influenced** — by the holder of that tenant's own agent token,
    so this is a self-corruption risk for one tenant's own series, not a
    cross-tenant one. `test_the_model_field_is_still_caller_written_pinned`
    executes that residual and fails the day someone closes it.
    """
    key = body.get("prompt_cache_key")
    has_key = isinstance(key, str) and bool(key)
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:8] if has_key else "none"
    retention = body.get("prompt_cache_retention")
    if not retention:
        opts = body.get("prompt_cache_options")
        ttl = opts.get("ttl") if isinstance(opts, dict) else None
        retention = f"opt:{ttl}" if ttl else None
    return has_key, key_hash, _log_atom(str(retention or "none"), max_len=32)


# ── R48 wire observability (log-only) ────────────────────────────────
# What the PROVIDER received, as opposed to what the agent believes it
# sent. Everything below is read-only over the already-assembled outbound
# body: no request or response byte changes, no llm_proxy_events column.
#
# WHAT THIS CAN AND CANNOT JOIN — read before building anything on it.
#   * `rid` joins the THREE PLATFORM LINES of one request to each other
#     (request-side [CACHE], usage-side [CACHE], `llm_proxy` summary).
#     It is platform-local: it is generated here and is NOT returned to
#     the agent in any header or body field, so it does not join the
#     agent's `[PERF]` lines to these.
#   * `trace` joins an agent line to these IF AND ONLY IF the agent sends
#     the `x-toup-trace` header. Nothing in the deployed agent
#     (166b835e) sends it — that half is a separate patch
#     (`g-llm-trace-header`) and until it ships every line here reads
#     `trace=-` and there is NO agent↔platform join.
#   * `req_id` is the provider's own opaque id. It joins a platform line
#     to an OpenAI support escalation; it is not a token the agent or
#     the browser ever sees.
# There is therefore no single value spanning agent → platform → OpenAI
# today. Anything that needs one needs BOTH this patch and the agent half.

# `x-toup-trace`: the agent's own turn marker, in the one shape the agent
# already computes — `<cmid_h>.<iteration>`, where `cmid_h` is
# `app/api/_turn_trace.py::cmid_hash` (FNV-1a/32, always exactly 8
# lowercase hex) and the iteration is the turn's tool-loop index, bounded
# well under 1000 by `agent_max_tool_iterations` (40 at 166b835e). So the
# pattern is the contract, not a guess at one. Note the consequence: if
# that ceiling is ever raised above 999 the header silently becomes `-`
# here. Validated, never parsed for meaning — this side never splits it,
# never hashes it and never reads an iteration out of it.
#
# NOTHING IN THE DEPLOYED AGENT SENDS THIS HEADER (grepped at 166b835e).
# The producer is a separate patch (`g-llm-trace-header`); until it ships
# every line reads `trace=-`.
#
# `\A…\Z`, not `^…$`: in Python `$` also matches just before a trailing
# newline, so `^…$` would accept "a1b2c3d4.7\n" and that value is a log
# FORGERY primitive — it would end the [CACHE] line and start a line of
# the sender's own composition in the platform's log stream. A bare `$`
# here is the whole vulnerability. (Starlette/h11 reject a header value
# containing a newline before it reaches us today, so this is defence in
# depth behind a wall someone else owns — an ASGI server swap, or a
# future non-HTTP caller, and this validator is the only one left. The
# cheapest place to be right is the validator itself.)
_TRACE_RE = re.compile(r"\A[0-9a-f]{8}\.[0-9]{1,3}\Z")
_TRACE_MAX_LEN = 12  # 8 + 1 + 3, the longest string the pattern accepts

# `req_id` is the PROVIDER's opaque `x-request-id`, not ours, so the same
# rule applies to it as to anything else we did not author: a value that
# could carry a line break must not reach a shared log stream unchecked
# (R48 FINAL review, N5 — the same class as B2). httpx/h11 reject a
# control character in a response header today, so this is defence in
# depth behind someone else's wall; the shape is deliberately WIDE (any
# run of printable, non-space ASCII up to 128 chars) so a provider that
# changes its id format keeps being reported rather than silently reading
# `-`, while every log-structure primitive — newline, carriage return,
# tab, space, NUL — is rejected.
_REQ_ID_RE = re.compile(r"\A[\x21-\x7e]{1,128}\Z")


def _wire_req_id(raw) -> str:
    """The provider's `x-request-id`, or `-` if it is absent or unsafe."""
    if not isinstance(raw, str) or not raw:
        return "-"
    return raw if _REQ_ID_RE.match(raw) else "-"


def _wire_trace_header(headers) -> str:
    """The validated inbound `x-toup-trace`, or `-`.

    ALLOWLIST, not sanitisation: a value that does not match the exact
    shape is discarded here and never reaches a log line, a metric, an
    exception message or the upstream request. Nothing about the
    rejected value is echoed — not a prefix, not its length, not a
    "malformed: %r" — because an attacker-supplied string quoted into a
    log line is the log-forging bug the validation exists to prevent, and
    because the header is un-authenticated by construction (the agent
    token authenticates the CALLER, not this field's contents).

    `-` therefore means "absent OR rejected" and those two are
    deliberately indistinguishable in the log.
    """
    try:
        raw = headers.get("x-toup-trace")
    except Exception:
        return "-"
    if not isinstance(raw, str) or not raw or len(raw) > _TRACE_MAX_LEN:
        return "-"
    return raw if _TRACE_RE.match(raw) else "-"


def _new_request_rid() -> str:
    """A per-request opaque join id: 8 hex characters, 4 random bytes.

    Carries no user data, no time, no counter — it is a random label whose
    only job is to make the three platform log lines of ONE request
    joinable. `cache_key_hash` + `tools_sha` is a COHORT key, not a unique
    one: two concurrent requests from the same user with the same body
    (an ordinary event — `subagent.py:129` starts child runs with
    `asyncio.create_task`, `cache_warm` is a third producer, and
    platform-api runs two replicas into one log stream) produce four lines
    carrying identical cohort keys, and pairing them by adjacency is then
    a coin flip that reads like a measurement.

    32 bits is not a uniqueness guarantee, and is not meant to be: the
    join is only ever performed inside one short log window for one user,
    where a collision needs two of the handful of concurrent requests to
    draw the same 4 bytes. Two lines that disagree about `tools_n` under
    one `rid` are a visible collision; a silent adjacency mis-join is not.

    `secrets` rather than `random`: the value is printed in a shared log
    stream, and a predictable per-request label in a multi-tenant log is
    free information nobody asked to publish. The cost is one
    `os.urandom(4)` — LOCAL 0.95 µs, 200k iterations, unloaded laptop.
    """
    return secrets.token_hex(4)


def _wire_observability_on(user_id: Optional[str]) -> bool:
    """Whether R48's wire fields ride this request's [CACHE] lines.

    Global flag OR a comma-separated canary allowlist of full user ids —
    the same shape `stable_prefix_enabled` uses on the agent side, and for
    the same reason: a fleet-wide flag is the wrong first move for a field
    that canonically re-serialises the forwarded tools array on the event
    loop of a shared, CPU-capped replica (0.57 ms p50 for 128 tools /
    112 KB on an unloaded laptop; the container's factor is unmeasured).

    Parsed per call, and deliberately NOT memoised on this function or on
    the module. `/admin/bind` can re-point a process at another tenant, and
    a memo that outlives the bind is a tenant-isolation bug; a set
    comprehension over a handful of ids is not worth that class of risk.
    """
    if getattr(settings, "llm_proxy_wire_observability", False):
        return True
    raw = getattr(settings, "llm_proxy_wire_observability_canary_user_ids", "") or ""
    if not raw or not user_id:
        return False
    return user_id in {u.strip() for u in raw.split(",") if u.strip()}


def _tools_digest(tools) -> str:
    """8-char sha256 of the FORWARDED tools array, canonically encoded.

    NOT a digest of the literal outbound bytes, and must never be read as
    one: httpx re-encodes `json=body` with its own separators, its own
    `ensure_ascii` and the dict's insertion order. Canonicalising
    (`sort_keys`) is the point — it makes two structurally identical arrays
    hash equal, which is the only question this field exists to answer:
    *did two requests offer the provider the same tools?*

    One-way over bytes we already hold. No name, no description, no schema
    fragment is recoverable from 8 hex characters, so this is loggable where
    the arrays themselves are not.

    The `encode` is INSIDE the try, and that placement is the whole point of
    the try (R48 FINAL review, B1). `ensure_ascii=False` leaves any lone
    surrogate in the string, `json.dumps` accepts one happily, and
    `str.encode("utf-8")` then raises `UnicodeEncodeError` — a `ValueError`
    subclass, so the clause below catches it. A lone surrogate reaches a
    tools array by an ordinary route (`json.loads('"\\ud800"')` succeeds, so
    a third-party MCP server mis-serialising UTF-16, or half an emoji pair,
    produces one), and with the encode outside the try that raise left the
    handler BEFORE the upstream call and lost the turn outright — a 500
    caused by a telemetry field, in exactly the canary state this patch asks
    to enter. `import json` is function-local to match this file's existing
    convention (four other call sites do the same).
    """
    import json
    if not isinstance(tools, list):
        return "none"
    try:
        blob = json.dumps(tools, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
    except TypeError:
        # A value `json.dumps` cannot represent at all (a set, bytes, a
        # custom object). A tools array like this would 400 upstream
        # anyway; a telemetry field must not be the thing that raises.
        return "unserialisable"
    except ValueError:
        # `json.dumps` accepted it and `encode` refused: a lone surrogate
        # (the ordinary case — see above), or a circular reference, which
        # raises ValueError from `dumps` itself. Split from the TypeError
        # branch because the two have different owners: a surrogate is a
        # third party's UTF-16 mis-serialisation, a TypeError is our own
        # tool-definition assembly (FINAL review round 2, non-blocking 3).
        return "unencodable"
    return digest


# The complete vocabulary `tc=` may print. Anything a client sends that is
# not one of these renders as "other" — see `_tool_choice_kind`.
_TC_WORDS = ("none", "auto", "required")


def _tool_choice_kind(body: dict) -> str:
    """`tc=` — the SHAPE of the forwarded tool_choice, never its names.

    ALLOWLIST, the same discipline as `_wire_trace_header`, and for the same
    reason (R48 FINAL review, B2). `tool_choice` is a CLIENT-SUPPLIED JSON
    body field: before this was an allowlist, a string form was returned
    VERBATIM and a dict's `type` was `str()`-ed wholesale, so an arbitrary
    length, arbitrary bytes, NEWLINE-BEARING value went straight onto a
    `[CACHE]` line in a shared multi-tenant log stream. One `\\n` ends the
    real line and starts a well-formed forged one under any `user=` prefix
    the sender likes, which corrupts exactly the two gates this patch exists
    to feed (the `tools_sha`-per-lineage count and the request→usage `rid`
    join). The h11 wall that rejects a newline in a HEADER value does not
    apply to a JSON body field, so this validator is the only guard there is.

    The complete output vocabulary, and nothing else can ever be printed:

      `none`         tool_choice absent, or the literal string "none".
                     (Those two are deliberately not distinguished — `tc=`
                     reports the shape the cap sees, and both give the cap
                     an empty protected set.)
      `auto`         the literal string "auto".
      `required`     the literal string "required".
      `allowed:<n>`  an allowlist, printed as its SIZE — the number that
                     moves between a turn's iterations. `<n>` is a `len()`,
                     so it is an integer by construction.
      `function`     a forced single call.
      `other`        EVERYTHING else: an unknown string, a non-str `type`,
                     a dict or list `type`, a number, a hostile payload.

    Tool names themselves stay where they already are: the cap's WARN and
    the prune's ERROR.

    Shape matters here because `_requested_tool_names` derives the cap's
    protected set from exactly this field, so it is the input that makes
    `tools_sha` vary between two otherwise identical requests.
    """
    tc = body.get("tool_choice")
    if tc is None:
        return "none"
    if isinstance(tc, str):
        return tc if tc in _TC_WORDS else "other"
    if not isinstance(tc, dict):
        return "other"
    kind = tc.get("type")
    if kind == "allowed_tools":
        container = tc["allowed_tools"] if isinstance(
            tc.get("allowed_tools"), dict) else tc
        entries = container.get("tools")
        return "allowed:%d" % (len(entries) if isinstance(entries, list)
                               else 0)
    if kind == "function":
        return "function"
    return "other"


def _wire_join_suffix(user_id: Optional[str], *, rid: str, trace: str) -> str:
    """` rid=… trace=…`, or `""` when observability is off for this user.

    The join prefix shared by every R48 line: the request-side [CACHE]
    line, the usage-side [CACHE] line, the `llm_proxy` summary line, and
    the upstream-error WARNING. Emitted FIRST inside every suffix so that
    `grep rid=<x>` returns one request's whole trail regardless of which
    line shape it lands on.

    `trace` is the agent's marker when the agent sent one and `-`
    otherwise — which is every request today, because no producer of
    `x-toup-trace` exists in the deployed agent (166b835e). See the
    section header above for what each id can and cannot join.
    """
    if not _wire_observability_on(user_id):
        return ""
    return " rid=%s trace=%s" % (rid or "-", trace or "-")


def _wire_tools_suffix(body: dict, tools_sent: int,
                       user_id: Optional[str], *,
                       rid: str, trace: str) -> tuple[str, str]:
    """`(suffix, tools_sha)` for a request-side [CACHE] line, or `("", "")`.

    Call it with the body as it will go upstream — after dedup, after
    `_cap_tools`, after `_prune_tool_choice` — because the pre-cap array is
    the one the agent already fingerprints, and a stable PRE-cap hash says
    nothing either way about what the wire carried.

    What the cap does, stated exactly: `_cap_tools` derives its protected
    set from THIS request's `tool_choice` (`_requested_tool_names`), so the
    forwarded tools array is a function of `tool_choice` as well as of the
    array the agent sent. PRODUCTION (supervisor E2, test account, three
    gpt-5.6-terra requests on 2026-09-19/20): identical agent-side
    fingerprints, identical `cache_key_hash=3c275b30`, all `174 > 128,
    dropped 46 (namespace-fair)`, three DIFFERENT dropped lists, and
    `cached=0` on all three — one of them 10 s after a 45,347-token cache
    write. That is an observed coincidence of a wire-array fork with three
    misses on one account, not a demonstration that caching cannot work
    over 128 tools and not a claim that every over-cap turn must miss: a
    partial-prefix hit up to the first differing tool, an unchanged
    protected set across turns, and early eviction are all still open, and
    the documented 30-minute retention is a MINIMUM lifetime rather than a
    ceiling. `tools_sha` is what turns the open question into a count.

    The digest comes back to the caller as well as onto the line so the
    usage-side suffix can print the SAME value without re-serialising 112 KB
    (R48 review, B1). Note what that pair is and is not: `cache_key_hash` +
    `tools_sha` is a COHORT key — every request in a cohort shares it — so
    it answers "were these two requests offering the same tools?" and NOT
    "are these two lines the same request". `rid` is the unique-per-request
    join (F6); the cohort key rides both lines so a cohort can still be
    aggregated when one of its lines is missing.

    `dropped_n` is the RAW `tools_sent - tools_n` and may be negative: it
    counts dedup casualties as well as capped ones, and a converter that
    ADDS tools would be an anomaly worth seeing rather than clamping to 0.
    Two shapes to read correctly before blaming the cap:
      * `tools_n=0 tools_sent=N dropped_n=N tools_sha=none` on `/chat` is the
        Anthropic→OpenAI daily-cap fallback, not the cap:
        `_anthropic_to_openai_request` builds its result with no `tools` key
        at all while `_cap_tools` ran earlier against the PRE-fallback
        backend. Literally correct for "the body going upstream", and a real
        defect of that fallback (it strips every tool while the system prompt
        keeps advertising them) — but it is not a cap casualty.
      * `dropped_n` > 0 with `tools_sent` <= 128 is dedup alone.

    The empty string when observability is off is what keeps the line
    byte-identical to R47.
    """
    if not _wire_observability_on(user_id):
        return "", ""
    tools = body.get("tools")
    tools_n = len(tools) if isinstance(tools, list) else 0
    sha = _tools_digest(tools)
    return (
        _wire_join_suffix(user_id, rid=rid, trace=trace) +
        " tools_n=%d tools_sha=%s tools_sent=%d dropped_n=%d tc=%s" % (
            tools_n, sha, tools_sent,
            tools_sent - tools_n, _tool_choice_kind(body),
        ),
        sha,
    )


def _wire_timing_suffix(user_id: Optional[str], *, rid: str, trace: str,
                        req_id: str,
                        cache_key_hash: str, tools_sha: str,
                        body_ms: int, pre_ms: int, ttfb_ms: int,
                        total_ms: int) -> str:
    """The R48 suffix for the usage-side [CACHE] line, or "".

    `req_id` is the provider's opaque `x-request-id`, passed through
    `_wire_req_id` (printable non-space ASCII, <=128 chars, else `-`) for
    the same reason `x-toup-trace` is allowlisted: it is a string we did
    not author reaching a shared log stream. It joins THIS line to an
    OpenAI support escalation about the same call. It carries no prompt
    content, no model identity and no key material. `-` means the upstream
    sent none, we never reached it, or what it sent was not a safe token.
    It does NOT join anything to the agent: the agent never sees it.

    `rid` is the platform-local per-request id and is the ONLY unique join
    between this line and its own request-side line (F6). `cache_key_hash`
    and `tools_sha` are repeats of the request line's values and are a
    COHORT key: two concurrent requests from one user with the same body
    share them exactly, so pairing on the cohort key — or worse, on log
    adjacency — silently mis-joins, and a mis-joined answer to "does the
    cap fork cost the cache?" is indistinguishable from a real one.
    Concurrency here is ordinary, not hypothetical: `subagent.py:129`
    starts child runs with `asyncio.create_task`, `cache_warm` is a third
    producer, and platform-api runs two replicas into a single Railway
    stream.

    The split exists because `latency` has never been decomposable:
    `start_ts` is stamped after auth, dedup, the tool cap, the credit
    pre-flight and the budget check, so the proxy's own pre-work has always
    been invisible, and everything from connect through the last token was
    one scalar.

      `body_ms`  the `await request.json()` call alone — the socket RECEIVE
                 of the agent's request body (112,088 canonical bytes of
                 tools alone on the sampled turn) plus its parse, over
                 Contabo → Cloudflare → Railway. This is UPLOAD, not proxy
                 work, and it is a sub-term of `pre_ms`; subtract it before
                 attributing anything to the proxy. Without it a 250 ms
                 `pre_ms` from a congested hop reads as platform-DB
                 pre-work and sends the next round somewhere else entirely.
      `pre_ms`   handler entry → `start_ts`, i.e. `body_ms` + auth SELECT +
                 rate-limit check + dedup + cap + credit pre-flight +
                 budget SELECT.
      `ttfb_ms`  `start_ts` → the first upstream chunk (on the streaming path
                 the same boundary `[req-timing]` reports, because the
                 handler pre-pulls that chunk before returning the
                 StreamingResponse). `-1` means "no first-chunk boundary
                 exists" — the non-streaming path — and must never be
                 averaged into the series.
      `total_ms` the existing `latency`.

    BLIND AREAS — the three things this split provably cannot see. Each of
    them lands in the residual against `[req-timing]`, so a reconciliation
    that does not subtract them is a false dichotomy rather than a check.

      1. Everything before the handler BODY. `entry_mono` is the first
         statement of `proxy_responses`, so the ASGI middleware chain,
         CORS, FastAPI routing and `db: AsyncSession = Depends(get_db)` —
         the platform session/connection acquire, resolved by FastAPI
         before the first statement runs, on an engine that sets no
         `pool_timeout` (SQLAlchemy's silent 30 s default) — appear here
         as nothing at all. That is the exact term workstream A is
         chasing. There is no `dep_ms` field because the timing
         middleware's start stamp is NOT reachable from a handler:
         `_t0` is a local of `_RequestTimingMiddleware.__call__`
         (platform_main.py:1018) and is never written to `scope`,
         `scope["state"]` or `request.state`. Publishing it would mean an
         unconditional `scope` mutation on every HTTP request of the
         shared platform process, which is outside this patch's
         log-only/flag-gated envelope — and it would still not be "ASGI
         entry", because `AttachmentBodyLimitMiddleware` (added later, so
         mounted outside it) and Starlette's own `ServerErrorMiddleware`
         both run before that stamp.
      2. The `/openai/v1/chat/completions` wire has NO timing suffix at
         all — only the request-side tools fields and the `rid`/`trace`
         join. `gpt-5.6-*` is forced onto `/responses` by the resolver, so
         the investigated turns are not on that wire; a turn that IS on it
         has `total_ms` from the summary line and nothing finer.
      3. `[req-timing]`, the only external clock to reconcile against, is
         emitted only when its gate passes, and that gate has THREE
         disjuncts, quoted from SOURCE at 4f0e9fe1
         (platform_main.py:1035): `_dur_ms > 500 or
         _p.startswith("/api/auth/") or "/agent-setup/config" in _p`.
         `/api/llm/openai/v1/responses` matches none of the three, so for
         THIS route the only live disjunct is the 500 ms one and the
         reconciliation can be run on the slow tail alone — never on the
         fast requests a p50 needs. (The third disjunct was omitted from an
         earlier draft of this block; it changes nothing for `/api/llm/`,
         and it is quoted in full here because a partial quote of a gate
         reads as the whole gate to the next person.)

    Log-only by construction: `llm_proxy_events` is the highest-insert-rate
    table in the platform DB and R48 is not the round that migrates it.
    """
    if not _wire_observability_on(user_id):
        return ""
    return (
        _wire_join_suffix(user_id, rid=rid, trace=trace) +
        " req_id=%s cache_key_hash=%s tools_sha=%s"
        " body_ms=%d pre_ms=%d ttfb_ms=%d total_ms=%d" % (
            req_id or "-", cache_key_hash or "none", tools_sha or "none",
            body_ms, pre_ms, ttfb_ms, total_ms,
        )
    )


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


def _requested_tool_names(body: dict) -> set:
    """Every tool name THIS request explicitly names.

    R31-41: these must survive the cap. `_prune_tool_choice` repairs the
    allowlist after the fact, which keeps the request valid but silently
    removes a capability the caller had asked for — and for a forced
    `{"type":"function"}` it cannot repair anything at all, so the
    request is a guaranteed 400. Protecting them at the cap makes both
    cases unreachable instead of handled.
    """
    out: set = set()
    tc = body.get("tool_choice")
    if not isinstance(tc, dict):
        return out
    if tc.get("type") == "function":
        fn = tc.get("function")
        name = (fn or {}).get("name") if isinstance(fn, dict) else tc.get("name")
        if name:
            out.add(str(name))
        return out
    if tc.get("type") != "allowed_tools":
        return out
    container = tc["allowed_tools"] if isinstance(
        tc.get("allowed_tools"), dict) else tc
    for e in container.get("tools") or []:
        if not isinstance(e, dict):
            continue
        name = e.get("name") or (
            (e.get("function") or {}).get("name")
            if isinstance(e.get("function"), dict) else None
        )
        if name:
            out.add(str(name))
    return out


def _prune_tool_choice(body: dict, dropped_names: list, *,
                       model_name: str = "?", original_len: int = 0) -> list:
    """Drop capped-away tools from `tool_choice` so the request stays
    internally consistent. Returns the names actually pruned.

    `_cap_tools` trims the tools array from the tail and left `tool_choice`
    untouched — so a request could name a tool in its allowlist that the
    same request no longer offered, and OpenAI answers 400 "Tool choice
    'X' not found in 'tools' parameter". The tail is exactly where this
    bites: the agent assembles core → skills → CONNECTOR tools last, so
    the dropped names are connector tools, and the allowlist is built from
    the uncapped array.

    Observed live 2026-08-26 on the founder's VOICE turn: 141 tools sent,
    13 over the cap, `slack__list_channels` capped away and still named in
    allowed_tools. Three 400s, then a silent fallback to a weaker model —
    20.7s for 2 output tokens and no tool calls. The user saw a slow,
    empty answer and no error.

    Handles both wire shapes: chat `{"type":"allowed_tools",
    "allowed_tools":{"tools":[{"function":{"name":n}}]}}` and Responses
    `{"type":"allowed_tools","tools":[{"name":n}]}`. A forced single
    `{"type":"function", ...}` naming a dropped tool cannot be repaired by
    pruning — the request has no valid form — so it is left alone and
    logged loudly rather than silently rewritten into something the caller
    did not ask for.
    """
    tc = body.get("tool_choice")
    if not isinstance(tc, dict) or not dropped_names:
        return []
    gone = set(dropped_names)

    if tc.get("type") == "function":
        named = ""
        if isinstance(tc.get("function"), dict):
            named = tc["function"].get("name", "")
        else:
            named = tc.get("name", "") or ""
        if named in gone:
            logger.error(
                "[LLM-PROXY] tool_choice FORCES '%s' but the cap dropped it "
                "— this request cannot succeed; the tools array is too "
                "large for a forced choice", named,
            )
        return []

    if tc.get("type") != "allowed_tools":
        return []

    # Chat shape nests under "allowed_tools"; Responses shape is flat.
    container = tc["allowed_tools"] if isinstance(
        tc.get("allowed_tools"), dict) else tc
    entries = container.get("tools")
    if not isinstance(entries, list):
        return []

    pruned: list = []
    kept_entries: list = []
    for e in entries:
        if not isinstance(e, dict):
            kept_entries.append(e)
            continue
        name = e.get("name") or (
            e.get("function", {}) or {}).get("name", "")
        if name in gone:
            pruned.append(name)
        else:
            kept_entries.append(e)
    if not pruned:
        return []

    # Its own line, at ERROR, not a clause on the cap's WARNING.
    #
    # Pruning is a different and worse event than a long array: the
    # request asked for a capability and was made valid by REMOVING it.
    # And it is now the ONLY signal. Before this function existed the
    # mismatch announced itself as a 400 — loud, three retries, a model
    # downgrade. Stripping the name makes the request succeed, so the
    # model quietly takes whatever survived the cut instead of erroring.
    # Measured on the founder 2026-08-26: asked which Slack channels it
    # could see with every `slack__*` tool capped away, the agent
    # answered correctly through `automations__list_targets` — a skill
    # tool upstream of the cut. Reads have a survivor; writes
    # (`slack__send_message`, `teams__send_chat_message`,
    # `outlook__send_message`) have none. So the agent looks capable when
    # asked to LOOK and is silently incapable when asked to ACT, which is
    # worse than a uniformly broken connector because it is not legible.
    #
    # This line is what replaces the 400 as the thing a human can find.
    logger.error(
        "[LLM-PROXY] CAPABILITY REMOVED to make the request valid: "
        "pruned %s from tool_choice for user=%s model=%s because the "
        "%d-tool array was capped to %d. The model will now silently "
        "choose from what survived instead of failing.",
        pruned, (body.get("_toup_user") or "?"), model_name, original_len,
        _OPENAI_MAX_TOOLS,
    )

    if kept_entries:
        container["tools"] = kept_entries
    else:
        # An allowlist emptied by the cap would forbid every tool. Dropping
        # the restriction entirely is the honest degradation: the model may
        # choose from what it was actually offered.
        body.pop("tool_choice", None)
        logger.error(
            "[LLM-PROXY] the cap emptied tool_choice's allowlist — dropping "
            "the restriction so the turn can proceed",
        )
    return pruned


def _tool_name_of(t) -> str:
    if not isinstance(t, dict):
        return ""
    return t.get("name") or (t.get("function") or {}).get("name") or ""


def _namespace_of(name: str) -> str:
    """`slack__send_message` → `slack`. Core tools share one namespace."""
    return name.split("__", 1)[0] if "__" in name else ""


#: Tools the overflow path may not reach until nothing else is left.
#:
#: `protected` is whatever the REQUEST named, which is the right thing to
#: preserve and is also entirely under the caller's control — so when the
#: allow-list itself overflows (see the R44 note in `_cap_tools`) the trim
#: runs over exactly the names the caller cares about, namespace-fair and
#: tail-first, with no idea which of them the PRODUCT cannot work without.
#: Measured on the founder's tenant 2026-09-20: `play_media` was in the drop
#: list on every unprotected round, so its survival on any given turn was
#: positional luck. This floor is the small set for which that is not
#: acceptable: music, the two web tools, day recall, and the memory surface
#: the system prompt advertises by name on every turn.
#:
#: A floor entry that is not in the array costs nothing. Override with
#: LLM_PROXY_PROTECTED_CORE_TOOLS (comma-separated) to widen or empty it
#: without a deploy of new code.
#:
#: Exactly the five names the brief specifies, and no more. Every extra entry
#: is one more name the floor can hold on the wire at the expense of a tool
#: the REQUEST named — the trade `_cap_tools` makes below — so widening this
#: set is not free and is not a judgement call to make while writing it.
_PROTECTED_CORE_DEFAULT = (
    "play_media", "web_search", "web_fetch", "recall_day", "memory_search",
)


def _protected_core_tools() -> frozenset:
    raw = os.environ.get("LLM_PROXY_PROTECTED_CORE_TOOLS")
    if raw is None:
        return frozenset(_PROTECTED_CORE_DEFAULT)
    return frozenset(n.strip() for n in raw.split(",") if n.strip())


#: Read once at import: the value has to be a constant for the process, or the
#: kept ORDER stops being a pure function of the wire array and the provider's
#: prefix cache re-forks on whatever changed the env.
PROTECTED_CORE_TOOLS = _protected_core_tools()


def _capped_hash(kept: list) -> str:
    """Fingerprint of the array that actually goes upstream.

    A3-6: `_cap_tools` reads `protected` from `tool_choice`, and the agent
    builds `tool_choice` on iteration 0 only — so iteration 0 protected ~14
    names and dropped 46, iteration 1 protected nothing and dropped a
    DIFFERENT 46. Tools serialize ahead of system and history, so the whole
    ~45k-token prefix was invalidated between round 1 and round 2 of every
    over-128 turn (measured: the run whose array changed cached nothing on its
    first two rounds; the run whose array did not, cached on all three).
    Two rounds of one turn are comparable by eye only if the line carries a
    digest — reconstructing the kept array from two drop lists is not
    something anyone does at 2am.

    V-5: the `kept_hash=` field this feeds is NOT comparable to the latency
    programme's `tools_sha=`. Both digest a tools array and they canonicalise
    it differently, so two equal arrays can print two different values and an
    eyeball comparison across the two log lines will report a prefix re-fork
    that did not happen. Compare `kept_hash=` only with `kept_hash=`."""
    try:
        # V-3: this file documents "no app.agent dependency by design", and
        # this is the one exception. It stays inside the function (so importing
        # llm_proxy never pulls app.agent), the target is stdlib-only, the
        # whole thing is wrapped, and a failure degrades to "?" — a log line
        # may never fail a request. Anything more than a pure hash here breaks
        # that contract.
        from app.agent.prefix_stability import tools_wire_hash
        return tools_wire_hash(kept)[:12]
    except Exception:  # noqa: BLE001 — a log line may never fail a request
        return "?"


def _cap_tools(tools: list, limit: int = _OPENAI_MAX_TOOLS,
               protected: Optional[set] = None,
               floor: Optional[frozenset] = None) -> tuple[list, list]:
    """Fit an over-long tools array under `limit`. Returns (kept, dropped).

    This is a cliff, not a slope: at `limit` everything works and at
    `limit + 1` every single turn fails, for every prompt, with an error
    that blames the user's phrasing. One tenant crossed it on 2026-08-08
    by connecting a 5-tool connector (124 → 129) and lost chat entirely.

    **R31-41 changes WHICH tools go.** It used to be `tools[:limit]` — a
    head slice — and the tail is not arbitrary: the agent assembles core
    → skills → MCP/connectors, and `mcp_tools_cache` sorts the MCP block
    BY NAME. So the casualties were always the same alphabetical tail:
    `outlook__send_message`, `session_create`, `session_list`, then every
    `slack__*` and every `teams__*`. Reads and writes alike, and entire
    connectors at a time.

    That is the shape of the R30 blocker. An automation whose steps read
    Slack and post to Slack loses BOTH on a ten-connector account, and
    the only signal is a log line — so the run fails with "it did not
    answer" and the user is told to reconnect a connector that is
    perfectly healthy. Alphabetical order is not importance order, and
    losing all five of a connector's tools is categorically worse than
    losing one of five.

    So the trim is now:

      1. **`protected` is dropped LAST, and only to stay valid.** These
         are the tools the request itself names (`tool_choice` /
         `allowed_tools`) — dropping one costs a capability, which is
         the ND-22 path. But an allow-list that alone exceeds the cap
         cannot be honoured at all (see below), so "never" became
         "last".
      2. **Every namespace keeps at least one tool.** A connector that
         is present at all stays reachable; the model can discover the
         rest is missing, but it cannot discover a connector that has
         vanished.
      3. **Beyond that, drop from the LARGEST namespace first**, tail
         within it. Degradation spreads across the connectors that can
         best afford it instead of landing entirely on whoever sorts
         last.

    **R44: rule 1 used to be absolute, and that shipped an invalid
    request.** (This block and the three-tier `_pick` below are the
    change written on 2026-09-15 in the `toup-r44-fix` worktree,
    adopted here verbatim except for the floor tier.) Measured on the
    founder's tenant 2026-09-15: 173 tools wire, of which the voice
    channel's intent filter allow-listed 130 (`[PERF] stable_tools:
    wire=173 allowed=130 intent=code`). The loop dropped all 43
    unprotected tools, reached `n_kept=130 > 128`, found no droppable
    victim and `break`'d — leaving the array two OVER the cap.
    `_prune_tool_choice` then pruned nothing, because no allow-listed
    name had been dropped. So the request went upstream with 130 tools
    AND a 130-entry allow-list, and OpenAI's Responses API answered
    `400 invalid_request_error param=tool_choice.type "Invalid value:
    'allowed_tools'"` — the union validator fails the allowed_tools
    variant and reports the discriminator, which is why the error blames
    `type` and not the array length. Three identical 400s per turn, then
    the silent fallback to gpt-4o.

    A too-long array is a guaranteed 400 for the whole turn; a trimmed
    allow-list is a lost capability on one turn, announced twice (the
    ERROR below and `_prune_tool_choice`'s CAPABILITY REMOVED). So
    when the protected set alone overflows, the same namespace-fair
    tail-first policy keeps running with protected names as candidates
    until the array actually fits, and the trimmed names come back in
    `dropped` so `_prune_tool_choice` takes them out of the allow-list
    too. Deterministic for identical input — the prefix-cache lineage
    depends on the kept order being a pure function of the wire array.

    **R48 adds a third tier under that.** `floor` (see
    `PROTECTED_CORE_TOOLS`) is the LAST thing the overflow path may
    touch: allow-listed names are trimmed first, the floor only if the
    floor alone still does not fit. Without it the R44 policy would have
    trimmed `play_media` off the founder's voice turn, namespace-fair
    and tail-first, which is the capability the whole round is about.

    The floor applies in full only when the request names NOTHING — the
    unrestricted retry, which is the round measured on 2026-09-20 and
    the only one the floor was written for. When the request DOES carry
    an allow-list the floor is intersected with it, because a floored
    name the allow-list omits is not callable on that request: holding
    it on the wire would evict an allow-listed tool that is, turning a
    valid fully-honoured request into a CAPABILITY REMOVED. This
    function runs for every user and every channel, so that trade has to
    be strictly a gain.

    Still the least-bad truncation, not a good one. The real fix is per
    step tool selection at the agent, and the WARN below is what keeps
    that visible rather than silent.
    """
    if len(tools) <= limit:
        return tools, []

    floor = floor if floor is not None else PROTECTED_CORE_TOOLS
    n_named = len(protected or ())
    protected = set(protected or ())

    # A request that RESTRICTS the callable set can only be helped by flooring
    # names it can actually call. Under `allowed_tools`, a tool that is in the
    # array but not in the allow-list is dead weight — the model may not call
    # it — so holding it on the wire costs an allow-listed tool its place for
    # nothing. Measured on this policy (wire 175, allow-list built to exclude
    # the floor): allow-list 121 → 1 allow-listed tool lost, 124 → 4, 128 → 8,
    # each one then announced by `_prune_tool_choice` as CAPABILITY REMOVED on
    # a request that was previously valid and fully honoured. Reachable with
    # the real gate: every non-`full` intent omits 1–3 floor names, so a `code`
    # turn on a ~98-connector tenant forces a trim every time. The case the
    # floor was actually written for — the unrestricted retry, which names
    # nothing and where the cap trims purely by position — is untouched.
    if protected:
        floor = frozenset(floor) & protected
    # What the trim may not touch at all, for the ERROR line below. These three
    # are reported separately rather than summed: `floor` is a SUBSET of
    # `protected` on a restricted request and disjoint from it on an
    # unrestricted one, so one number cannot describe both.
    n_floored = len(floor)
    n_untrimmable = len(protected | set(floor))

    # NOTE: the floor is NOT merged into `protected`. `_eligible`'s tier-0
    # test is the single authoritative guard — it has to be, because a merged
    # floor name would pass tier 1 (`if tier >= 1: return True`) and become
    # trimmable one tier too early. Read the two together before editing
    # either: a guard whose precondition something above it destroys is
    # invisible to every check in this repo.
    keep_flags = [True] * len(tools)
    names = [_tool_name_of(t) for t in tools]
    spaces = [_namespace_of(n) for n in names]

    # How many of each namespace are still in.
    #
    # R48 CORRECTION. The comment that stood here claimed un-namespaced
    # tools can never be the largest namespace, and that rule 2 protects
    # them anyway. Both halves are false, and together they invert rule 3.
    #
    # `_namespace_of` returns "" for every name without `__`, so ALL of
    # those tools share ONE namespace here. Its membership is not just the
    # static core definitions: it is the static core PLUS the un-namespaced
    # first-party MCP tools the agent also offers. Counted 2026-09-20 by
    # importing the shipped definitions at the deployed agent revision
    # 166b835e, the static subset alone is 63 (get_agent_tools 31 +
    # get_extended_tools 22 + get_doc_generation_tools 9 +
    # get_navigation_tools 1, none of them namespaced), and the production
    # WARN lines below add at least nine more names that are in none of
    # those four functions (entity_search, graph_traverse, memory_remember,
    # identity_get, …, all defined in the agent's app/mcp_server.py) — so
    # the "" namespace is >= 72, not 63. Against a typical connector's ~10
    # it is the LARGEST namespace by a wide margin on every account, so
    # "drop from the largest namespace first" reads as "drop un-namespaced
    # first" — and rule 2 does not protect them, it only guarantees the
    # namespace keeps ONE tool. What holds SPECIFIC un-namespaced names is
    # (rule 1) whatever THIS request's tool_choice names — the allow-list
    # the agent sends on its first iteration, or a forced function — and,
    # under that, the Voice programme's (PR 756) `floor` above
    # (`PROTECTED_CORE_TOOLS`, five names by default). On a request that
    # names nothing, those five are never trimmed while any other candidate
    # remains. On a request that names anything (allow-list or forced
    # function), the floor is intersected with what it names, so it holds
    # nothing rule 1 does not already hold and only decides that those
    # names are trimmed last if the named set alone overflows; every other
    # floor name is an ordinary candidate.
    #
    # Measured in production on three of the founder's gpt-5.6-terra turns
    # (2026-09-19 21:00, 21:00, 2026-09-20 04:07; each `174 > 128, dropped
    # 46 (namespace-fair)`) — before the floor existed; it is not in
    # 4f0e9fe1, and `recall_day`, one victim below, is now a floor name:
    # every dropped name was UN-NAMESPACED, and not
    # one namespaced connector tool was dropped on any of them. What "core"
    # alone can claim is narrower and is the claim this comment makes: of
    # the 51 distinct names dropped across the three, 42 are static core
    # definitions (analyze_image, generate_image, edit_image, process, tts
    # on one; recall_day, start_mission, create_job, update_job,
    # navigate_to on the next) and the remaining 9 are first-party MCP
    # tools that live in the same "" namespace. The system prompt meanwhile
    # keeps advertising what the cap removed, which is the constrained
    # decode failure class documented in query_intent.py.
    #
    # This correction deliberately changes NO selection logic. The two
    # selection changes since it was written are the Voice programme's
    # (PR 756), not this comment's: the R44 overflow tiers (named names
    # trimmable last, only when they alone overflow) and the floor. Neither
    # reorders namespaces, so for every name outside the floor "largest
    # namespace first" still reads as "un-namespaced first". Which tools a
    # turn should carry beyond that is a product decision with a known
    # blocker (a naive "spare core" inversion gutted skills and connectors
    # when it was simulated), and the first thing that decision needs is the
    # `tools_sha` series this round adds — until then nobody can count how
    # often, or for whom, the forwarded array actually differs.
    from collections import Counter
    remaining = Counter(spaces)
    n_kept = len(tools)

    # Drop candidates, tail-first within a namespace, largest namespace
    # first — recomputed each step so the biggest never runs away.
    order = sorted(
        range(len(tools)),
        key=lambda i: -i,          # tail first
    )

    def _eligible(i: int, tier: int) -> bool:
        """Tier 0: never named. Tier 1: named by the request. Tier 2: the
        product floor. Each tier only WIDENS the candidate pool; the policy
        (rule 2 first, then the flat tail) is identical in all three, so the
        overflow path degrades the same way the ordinary one does."""
        if tier >= 2:
            return True
        if names[i] in floor:
            return False
        if tier >= 1:
            return True
        return names[i] not in protected

    def _pick(tier: int) -> Optional[int]:
        victim = None
        best_size = 0
        for i in order:
            if not keep_flags[i] or not _eligible(i, tier):
                continue
            ns = spaces[i]
            if remaining[ns] <= 1:
                continue           # rule 2: never empty a namespace
            if remaining[ns] > best_size:
                best_size, victim = remaining[ns], i
        if victim is not None:
            return victim
        # Every candidate is its namespace's last. Rule 2 yields — a
        # 400 for everyone is worse than a degraded turn.
        for i in order:
            if keep_flags[i] and _eligible(i, tier):
                return i
        return None

    trimmed_protected: list = []
    trimmed_floor: list = []
    while n_kept > limit:
        victim = _pick(0)
        if victim is None:
            # Nothing unnamed left and still over the cap: the allow-list
            # alone exceeds `limit`. Keep trimming, allow-listed names first.
            victim = _pick(1)
            if victim is not None:
                trimmed_protected.append(names[victim] or "<unnamed>")
            else:
                victim = _pick(2)
                if victim is not None:
                    trimmed_floor.append(names[victim] or "<unnamed>")
        if victim is None:
            break                  # unreachable while n_kept > limit >= 0
        keep_flags[victim] = False
        remaining[spaces[victim]] -= 1
        n_kept -= 1

    if trimmed_protected or trimmed_floor:
        # Its own ERROR line, separate from `_prune_tool_choice`'s.
        # That one says "a capability the caller asked for was removed";
        # this one says WHY it was unavoidable — the request named more
        # tools than the provider will accept in one array, which is a
        # defect in whatever built the allow-list, not in the cap.
        # The counts are reported, never asserted: the old line said
        # "(%d named > %d)" with n_named == limit, which is a falsehood the
        # moment the floor is what pushed the untrimmable set over.
        logger.error(
            "[LLM-PROXY] the untrimmable set does not fit under the provider "
            "cap: %d untrimmable (%d named by tool_choice, %d on the protected "
            "core floor) vs limit %d — trimmed %d allow-listed tool(s) so the "
            "request is valid at all: %s%s",
            n_untrimmable, n_named, n_floored, limit,
            len(trimmed_protected), trimmed_protected,
            (f" | AND {len(trimmed_floor)} from the protected core floor, "
             f"i.e. the floor itself no longer fits and has been widened "
             f"past what the provider accepts: {trimmed_floor}"
             if trimmed_floor else ""),
        )

    kept = [t for t, k in zip(tools, keep_flags) if k]
    dropped = [n or "<unnamed>" for n, k in zip(names, keep_flags) if not k]
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
    # R48 join ids. Minted at handler entry, before anything can fail, so
    # every line this request produces carries the same pair. Both are
    # computed unconditionally (4 random bytes + one header lookup +
    # one anchored regex on a ≤12-char string); only the PRINTING is
    # gated, so the gate cannot leave one line joinable and another not.
    wire_rid = _new_request_rid()
    wire_trace = _wire_trace_header(request.headers)
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
    # R48: captured before dedup, for `dropped_n` on the [CACHE] line below.
    tools_sent = len(tools) if isinstance(tools, list) else 0
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
            _kept, _dropped = _cap_tools(
                _tools, protected=_requested_tool_names(body))
            body["tools"] = _kept
            _pruned = _prune_tool_choice(
                body, _dropped, model_name=str(model),
                original_len=len(_tools))
            logger.warning(
                "[LLM-PROXY] tools array over OpenAI's cap for user=%s model=%s "
                "channel=%s: %d > %d, dropped %d (namespace-fair) kept_hash=%s: "
                "%s%s",
                config.user_id[:8], model, req_channel or "-",
                len(_tools), _OPENAI_MAX_TOOLS,
                len(_dropped), _capped_hash(_kept), _dropped,
                (f" | pruned from tool_choice: {_pruned}" if _pruned else ""),
            )

    # Credit pre-flight: zero-balance gate. Only enforces when
    # credit_enforcement_enabled=True; in shadow mode this is a no-op.
    # Returns 402 with a structured body the agent / chat client can
    # decode to render the "out of credits / upgrade" UI.
    #
    # PREFLIGHT_QUOTE_CREDITS, not a literal: this probe is nominal, the
    # turn's real cost is settled afterwards, and the shadow-admission
    # verdict has to ask the same question the live gate asks or its numbers
    # describe a gate that never shipped.
    #
    # One constant, four readers — here, `proxy_responses`, the shadow verdict
    # in `credit_service._shadow_observe_message_charge`, and
    # `/credits/agent-deduct`'s admission probe. `ws_realtime`'s voice
    # pre-flight is a FIFTH site, still spelling the number itself (that file
    # is untouched this round), so it shares the quote by convention rather
    # than by reference. The value is pinned in
    # tests/test_credit_settlement_shortfall.py, because since 2026-09-18 this
    # is the live 402 gate rather than a shadow-mode mirror: changing it moves
    # production admission on two endpoints.
    try:
        from app.credit_shadow import PREFLIGHT_QUOTE_CREDITS
        from app.services.credit_service import (
            credit_service, BUCKET_MESSAGE,
            REASON_INSUFFICIENT_MESSAGE, REASON_DAILY_CAP_EXCEEDED,
            REASON_EMAIL_NOT_VERIFIED,
        )
        if getattr(settings, "credit_enforcement_enabled", False):
            preflight = await credit_service.check_balance(
                db, config.user_id, BUCKET_MESSAGE, PREFLIGHT_QUOTE_CREDITS,
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
        # R48 suffix (""-by-default; see _wire_tools_suffix). On a daily-cap
        # fallback `body` has already been rewritten into OpenAI shape and
        # carries NO tools at all, so this line reads tools_n=0 dropped_n=N —
        # correct for "the body going upstream", and documented in
        # _wire_tools_suffix so it is not read as a cap casualty.
        # No timing suffix on this wire (BLIND AREA 2 in
        # _wire_timing_suffix): gpt-5.6-* is forced onto /responses by the
        # resolver, so the investigated turns are not here. The rid/trace
        # join IS here, on all three of this wire's lines.
        #
        # `model=` on this line is CALLER-WRITTEN and unvalidated — PRE-
        # EXISTING at 4f0e9fe1, not introduced or widened here, and not
        # closed here either (it is rendered from `_log_event`, the
        # `[credits]` lines and the upstream WARNINGs too, none of which
        # a log-only patch owns). `retention=` next to it IS allowlisted
        # as of the FINAL round-2 pass. Read the consequence in
        # `_cache_log_fields`: G7/G8 rows are caller-influenced.
        _tools_suffix, _ = _wire_tools_suffix(
            body, tools_sent, config.user_id, rid=wire_rid, trace=wire_trace)
        logger.info(
            "[CACHE] user=%s model=%s has_cache_key=%s cache_key_hash=%s retention=%s%s",
            config.user_id[:8], model, has_cache_key, cache_key_hash, cache_retention,
            _tools_suffix,
        )
    # Rendered once for every line below, including the ones inside the
    # streaming generator (which closes over it). `""` unless the user is on
    # the canary, so the untouched lines stay byte-identical to R47.
    _join_suffix = _wire_join_suffix(
        config.user_id, rid=wire_rid, trace=wire_trace)

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
            # R48 (FINAL review round 2, non-blocking 2): the FAILED-turn
            # summary line carries the join too. Without it a failed turn
            # emitted a `[CACHE]` request line with a `rid` and an
            # `llm_proxy … status=error` line WITHOUT one, so the two lines
            # that exist for the turn nobody got an answer from — the turns
            # G7/G8 most need to exclude — were the only pair that could not
            # be joined. Gated like every other render, so off-canary this
            # line stays byte-identical to R47.
            await _log_event(
                db, config.user_id, backend.name, model, "chat",
                0, 0, 0, int((time.time() - start_ts) * 1000), is_fallback, "error",
                channel=req_channel,
                wire_suffix=_join_suffix,
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
                    # R48 appends the rid/trace join and nothing else — this
                    # wire carries no timing split (BLIND AREA 2).
                    logger.info(
                        "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
                        "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s%s",
                        config.user_id[:8], model, has_cache_key, cache_retention,
                        inp, cached, cache_write, _join_suffix,
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
                            wire_suffix=_join_suffix,
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
                wire_suffix=_join_suffix,
            )
            raise HTTPException(502, f"Provider error: {e}")
        # Surface clean upstream errors (model-not-found, rate-limit, etc.)
        if resp.status_code >= 400:
            body_bytes = resp.content
            await _log_event(
                db, config.user_id, backend.name, model, "chat",
                0, 0, 0, int((time.time() - start_ts) * 1000), is_fallback, "error",
                channel=req_channel,
                wire_suffix=_join_suffix,
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
            # R48: rid/trace join only — no timing split on this wire.
            logger.info(
                "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
                "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s%s",
                config.user_id[:8], model, has_cache_key, cache_retention,
                inp, cached, cache_write, _join_suffix,
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
            wire_suffix=_join_suffix,
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
    # R48: `start_ts` below is stamped after auth, dedup, the tool cap, the
    # credit pre-flight and the budget check, so `latency` has never
    # contained one millisecond of the proxy's own pre-work. This stamp is
    # what makes that gap a number instead of an assumption.
    #
    # Read the boundary honestly (R48 review, B2): this is the first
    # statement of the handler BODY, so it is already too late for the
    # middleware chain, FastAPI routing and `Depends(get_db)` — the platform
    # session acquire resolves before it, and never appears in the split. And
    # it is early enough to include `await request.json()` below, i.e. the
    # socket receive of the agent's ~200 KB body over Contabo → Cloudflare →
    # Railway, which is upload and not proxy work. `body_ms` exists to hold
    # that term separately so `pre_ms` can be read net of it.
    #
    # The full list of what this split cannot see — including why there is
    # no `dep_ms` for the session acquire — is BLIND AREAS in
    # `_wire_timing_suffix`. Read it before reconciling anything against
    # `[req-timing]`.
    entry_mono = time.monotonic()
    # R48 join ids, minted at handler entry (see proxy_chat for why both
    # are unconditional while only the printing is gated). They sit after
    # `entry_mono` so that stamp stays the first statement and the ~1 µs
    # they cost is inside the window it measures rather than before it.
    wire_rid = _new_request_rid()
    wire_trace = _wire_trace_header(request.headers)
    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    req_channel = _sanitize_channel(request.headers.get(CHANNEL_HEADER))
    _body_mono = time.monotonic()
    body = await request.json()
    body_ms = int((time.monotonic() - _body_mono) * 1000)
    # R44: platform overhead (a prompt-cache warm) is logged for cost
    # tracking but never charged and never counted against the user's cap —
    # `_log_event` and `_get_spend` both key that off "system.".
    req_operation = _system_operation_for(
        request.headers.get(OPERATION_TYPE_HEADER), body,
    )
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
    # Captured BEFORE dedup so `dropped_n` on the [CACHE] line counts every
    # tool the proxy removed, by whatever route.
    tools_sent = len(tools) if isinstance(tools, list) else 0
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
        _kept, _dropped = _cap_tools(
            _tools, protected=_requested_tool_names(body))
        body["tools"] = _kept
        _pruned = _prune_tool_choice(
            body, _dropped, model_name=str(model),
            original_len=len(_tools))
        logger.warning(
            "[LLM-PROXY] tools array over OpenAI's cap for user=%s model=%s "
            "channel=%s: %d > %d, dropped %d (namespace-fair) kept_hash=%s: "
            "%s%s",
            config.user_id[:8], model, req_channel or "-",
            len(_tools), _OPENAI_MAX_TOOLS,
            len(_dropped), _capped_hash(_kept), _dropped,
            (f" | pruned from tool_choice: {_pruned}" if _pruned else ""),
        )

    # Credit pre-flight: zero-balance gate (same contract as proxy_chat,
    # including the shared nominal quote).
    try:
        from app.credit_shadow import PREFLIGHT_QUOTE_CREDITS
        from app.services.credit_service import (
            credit_service, BUCKET_MESSAGE,
            REASON_INSUFFICIENT_MESSAGE,
        )
        if getattr(settings, "credit_enforcement_enabled", False):
            preflight = await credit_service.check_balance(
                db, config.user_id, BUCKET_MESSAGE, PREFLIGHT_QUOTE_CREDITS,
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

    # W0.2b request-side [CACHE] line — the shared extractor reads both
    # prompt_cache_retention and GPT-6's prompt_cache_options.ttl.
    has_cache_key, cache_key_hash, cache_retention = _cache_log_fields(body)
    # R48 suffix — renders as "" unless this user is on the wire-observability
    # canary, which keeps the line byte-identical to R47 by default. It sits
    # here, after dedup + _cap_tools + _prune_tool_choice, because the array
    # the agent fingerprints is the PRE-cap one and that instrument is blind
    # to the only array the provider ever caches. `wire_tools_sha` is carried
    # to the usage line so the pair can be joined without relying on log
    # adjacency, which concurrent turns destroy.
    _tools_suffix, wire_tools_sha = _wire_tools_suffix(
        body, tools_sent, config.user_id, rid=wire_rid, trace=wire_trace)
    logger.info(
        "[CACHE] user=%s model=%s has_cache_key=%s cache_key_hash=%s retention=%s%s",
        config.user_id[:8], model, has_cache_key, cache_key_hash, cache_retention,
        _tools_suffix,
    )
    # Rendered once and reused by the usage line's builder, the summary line
    # and the upstream-error WARNING — `""` off the canary.
    _join_suffix = _wire_join_suffix(
        config.user_id, rid=wire_rid, trace=wire_trace)

    start_ts = time.time()
    start_mono = time.monotonic()
    pre_ms = int((start_mono - entry_mono) * 1000)

    if is_stream:
        # Streaming: pre-pull the first chunk INSIDE try so an upstream
        # non-2xx converts to a clean HTTPException BEFORE the
        # StreamingResponse commits its 200 headers (same as proxy_chat).
        # R48: the only channel for the upstream's x-request-id — the httpx
        # response is confined to the generator. Passed unconditionally; the
        # canary decides whether it is LOGGED, not whether it is captured,
        # so the two halves cannot drift apart.
        upstream_meta: dict = {}
        gen = backend.responses_stream(body, api_key, meta=upstream_meta)
        try:
            first_chunk = await gen.__anext__()
        except UpstreamProviderError as e:
            # R48 (FINAL review round 2, non-blocking 2): the join rides the
            # FAILED-turn summary line too — see the /chat twin above for
            # why the error path is the one that most needs it. Gated.
            await _log_event(
                db, config.user_id, "openai", model, "responses",
                0, 0, 0, int((time.time() - start_ts) * 1000), False, "error",
                channel=req_channel,
                operation_type=req_operation,
                wire_suffix=_join_suffix,
            )
            # R48 review N6: the id is written into `meta` before the status
            # check inside the generator, so an upstream 4xx/5xx — the one
            # response an OpenAI escalation is actually about — can quote it.
            # Unconditional: it is the provider's opaque id, the same value
            # the DEBUG header dump has always been allowed to print, and no
            # usage-side [CACHE] line is emitted on this path at all.
            # The rid/trace join rides it too, gated. A failed turn emits no
            # usage-side [CACHE] line, so its trail is exactly three lines —
            # the request-side [CACHE], this WARNING, and the `llm_proxy …
            # status=error` summary. All three now carry `rid`; until the
            # round-2 FINAL pass the summary did not, which left the turns
            # G7/G8 most need to EXCLUDE as the only ones that could not be
            # identified as failures.
            logger.warning(
                "[LLM-PROXY] %s upstream %d for user=%s model=%s req_id=%s%s body=%r",
                e.provider, e.status, config.user_id[:8], model,
                _wire_req_id(upstream_meta.get("request_id")),
                _join_suffix, e.body,
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

        # R48: the pre-pull above is exactly the upstream's time-to-first-byte
        # — the same boundary `[req-timing]` reports for this route, because
        # the StreamingResponse is not returned until it completes.
        ttfb_ms = int((time.monotonic() - start_mono) * 1000)

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
                # per-tenant hit-ratio grep spans both wires). R48 appends the
                # provider request id and the timing split; "" when off.
                logger.info(
                    "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
                    "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s%s",
                    config.user_id[:8], model, has_cache_key, cache_retention,
                    inp, cached, cache_write,
                    _wire_timing_suffix(
                        config.user_id,
                        rid=wire_rid, trace=wire_trace,
                        req_id=_wire_req_id(upstream_meta.get("request_id")),
                        cache_key_hash=cache_key_hash,
                        tools_sha=wire_tools_sha,
                        body_ms=body_ms, pre_ms=pre_ms,
                        ttfb_ms=ttfb_ms, total_ms=latency,
                    ),
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
                            operation_type=req_operation,
                            wire_suffix=_join_suffix,
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
                operation_type=req_operation,
                wire_suffix=_join_suffix,
            )
            raise HTTPException(502, f"Provider error: {e}")
        if resp.status_code >= 400:
            body_bytes = resp.content
            await _log_event(
                db, config.user_id, "openai", model, "responses",
                0, 0, 0, int((time.time() - start_ts) * 1000), False, "error",
                channel=req_channel,
                operation_type=req_operation,
                wire_suffix=_join_suffix,
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
        # R48: ttfb_ms=-1 — this path has no first-chunk boundary at all, and
        # a number that silently meant "the whole call" would poison the p50
        # of a series whose whole purpose is separating the two.
        logger.info(
            "[CACHE] user=%s model=%s has_cache_key=%s retention=%s "
            "prompt_tokens=%s cached_tokens=%s cache_write_tokens=%s%s",
            config.user_id[:8], model, has_cache_key, cache_retention,
            inp, cached, cache_write,
            _wire_timing_suffix(
                config.user_id,
                rid=wire_rid, trace=wire_trace,
                req_id=_wire_req_id(resp.headers.get("x-request-id")),
                cache_key_hash=cache_key_hash,
                tools_sha=wire_tools_sha,
                body_ms=body_ms, pre_ms=pre_ms, ttfb_ms=-1, total_ms=latency,
            ),
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
            operation_type=req_operation,
            wire_suffix=_join_suffix,
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


def _decode_image_sources(body: dict) -> list[tuple[bytes, str]]:
    """The edit sources on a Kie request: `images_b64` first, else `image_b64`.

    ``images_b64`` is ``[{b64, mime}, …]`` in render order — entry 0 is the
    BASE, the rest are references. An agent that predates references sends only
    ``image_b64``, and an agent that has them sends BOTH (the scalar for the
    base) so either side of a mixed fleet renders the same picture. Raises 400
    on bad base64 or an overflow, before any billable work.
    """
    import base64 as _b64m
    from app.services.kie_client import KIE_MAX_IMAGE_INPUT

    raw = body.get("images_b64")
    out: list[tuple[bytes, str]] = []
    if isinstance(raw, list) and raw:
        if len(raw) > KIE_MAX_IMAGE_INPUT:
            raise HTTPException(400, (
                f"images_b64 accepts at most {KIE_MAX_IMAGE_INPUT} images "
                f"(1 base + {KIE_MAX_IMAGE_INPUT - 1} references); got {len(raw)}"
            ))
        for i, item in enumerate(raw):
            if isinstance(item, str):
                b64, mime = item, "image/png"
            elif isinstance(item, dict):
                b64 = item.get("b64") or item.get("image_b64") or ""
                mime = item.get("mime") or item.get("image_mime") or "image/png"
            else:
                raise HTTPException(400, f"images_b64[{i}] must be an object or a base64 string")
            if not b64:
                raise HTTPException(400, f"images_b64[{i}] carries no base64 data")
            try:
                out.append((_b64m.b64decode(b64), mime))
            except Exception:
                raise HTTPException(400, f"images_b64[{i}] is not valid base64")
        return out
    image_b64 = body.get("image_b64")
    if not image_b64:
        return []
    try:
        return [(_b64m.b64decode(image_b64), body.get("image_mime") or "image/png")]
    except Exception:
        raise HTTPException(400, "image_b64 is not valid base64")


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
            try:
                srcs = _decode_image_sources(body)
            except HTTPException:
                await release_free_image_slot(db, _img_slot)
                raise
            if not srcs:
                await release_free_image_slot(db, _img_slot)
                raise HTTPException(400, "edit mode requires image_b64")
            result = await kie_client.edit(prompt, sources=srcs)
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
    # The charge is per RENDER, and the render on this route is not idempotent:
    # it holds one request open for the whole job, so a retry is a second Kie
    # job by construction — already paid for, in real money. Letting the CALLER
    # choose the ledger key made every render after the first free
    # (`try_charge` short-circuits on a prior (user_id, idempotency_key) row and
    # returns idempotent_hit with no deduction), i.e. unlimited billed-to-Toup
    # renders for one charge, reachable by anything holding the tenant's token.
    # The start+poll pair below is the path with real render idempotency — its
    # reservation marker is keyed on the caller's key BEFORE the render — and it
    # is the one the agent uses.
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
    from app.db.models import BUCKET_MESSAGE, CreditReservation, RESERVATION_OPEN

    config = await _auth_agent(request, db)
    _enforce_rate_limit(config)
    body = await request.json()
    mode = (body.get("mode") or "generate").strip().lower()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")

    # ── Render idempotency ─────────────────────────────────────────────
    # A started Kie job is a BILLED job (the hold below exists so an abandoned
    # render still stays paid for). So a retry of THIS route — a dropped
    # response, an agent restart mid-call — must return the job that already
    # exists rather than start a second one and charge the user twice for one
    # request. The marker is a `credit_reservations` row under a key of the
    # caller's choosing: its UNIQUE (user_id, idempotency_key) index is the
    # atomic claim, and `event_type` is deliberately NOT an image ledger type
    # so it is invisible to the free-tier cap's count of open image holds.
    # The lookup FAILS OPEN: a marker we cannot read means we start the render,
    # which is today's behaviour, never a refusal.
    _idem = str(body.get("idempotency_key") or "").strip()[:100]
    _idem_key = f"kie_idem:{_idem}" if _idem else ""
    if _idem_key:
        try:
            _prior = (await db.execute(
                select(CreditReservation).where(
                    CreditReservation.user_id == config.user_id,
                    CreditReservation.idempotency_key == _idem_key,
                )
            )).scalar_one_or_none()
        except Exception:
            logger.exception("[kie] idempotency lookup failed user=%s", config.user_id[:8])
            _prior = None
        if _prior is not None and (_prior.metadata_json or {}).get("task_id"):
            _meta = _prior.metadata_json or {}
            logger.info("[kie] start deduped user=%s task=%s",
                        config.user_id[:8], str(_meta.get("task_id"))[:12])
            return {"task_id": _meta.get("task_id"),
                    "reservation_id": _meta.get("free_slot"),
                    "status": "pending", "deduped": True}

    exceeded, used, limit, _img_slot = await reserve_free_image_slot(db, config.user_id)
    if exceeded:
        raise HTTPException(status_code=429, detail={
            "code": "image_quota_exceeded", "used": used, "limit": limit,
            "message": (f"Your free plan includes {limit} images per month, and "
                        f"you've used them all. Upgrade for unlimited images."),
        })

    srcs: list[tuple[bytes, str]] = []
    if mode == "edit":
        try:
            srcs = _decode_image_sources(body)
        except HTTPException:
            await release_free_image_slot(db, _img_slot)
            raise
        if not srcs:
            await release_free_image_slot(db, _img_slot)
            raise HTTPException(400, "edit mode requires image_b64")

    try:
        task_id = await kie_client.start_task(
            mode, prompt, size=body.get("size"), sources=srcs or None)
    except kie_client.KieError as e:
        await release_free_image_slot(db, _img_slot)   # nothing started → free it
        raise HTTPException(status_code=502, detail={
            "code": "kie_failed", "moderation": bool(e.moderation),
            "message": str(e)[:300],
        })

    if _idem_key:
        # Written AFTER createTask so the marker can only ever name a job that
        # really exists. A crash in the window between the two loses the marker
        # and a retry starts a second render — exactly today's behaviour, and
        # the reason this is a narrowing of the window rather than a proof.
        try:
            db.add(CreditReservation(
                user_id=config.user_id, event_type="kie_idem",
                bucket=BUCKET_MESSAGE, estimated_amount=Decimal("0"),
                status=RESERVATION_OPEN, idempotency_key=_idem_key,
                event_id=f"kie_task:{task_id}",
                metadata_json={"task_id": task_id, "free_slot": _img_slot,
                               "mode": mode, "sources": len(srcs)},
                expires_at=datetime.utcnow() + timedelta(
                    seconds=int(float(getattr(settings, "kie_job_timeout_s", 420.0))) + 600),
            ))
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("[kie] idempotency marker write failed user=%s",
                             config.user_id[:8])

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
