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

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db, AgentConfig, LLMProxyEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM Proxy"])
admin_router = APIRouter(prefix="/admin/llm", tags=["Admin LLM"])

# ── Budget cache (in-memory, TTL-based) ──────────────────────────────

_budget_cache: dict[str, tuple[float, int]] = {}  # key → (expiry_ts, cost_cents)
_CACHE_TTL = 30  # seconds


def _cache_key(user_id: str, provider: str, scope: str) -> str:
    return f"{user_id}:{provider}:{scope}"


def _get_cached_spend(key: str) -> Optional[int]:
    entry = _budget_cache.get(key)
    if entry and entry[0] > time.time():
        return entry[1]
    return None


def _set_cached_spend(key: str, cents: int):
    _budget_cache[key] = (time.time() + _CACHE_TTL, cents)


def _invalidate_cache(user_id: str):
    keys_to_remove = [k for k in _budget_cache if k.startswith(user_id)]
    for k in keys_to_remove:
        _budget_cache.pop(k, None)


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
            "or 'x-api-key: <TOUP_TOKEN>' (Anthropic SDK convention).",
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
) -> int:
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
    cents = int(result.scalar())
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


def _calc_cost_cents(model: str, input_tokens: int, output_tokens: int) -> int:
    """Calculate cost in cents from token counts using the pricing table."""
    pricing = settings.pricing_per_1k.get(model)
    if not pricing:
        # Fallback: use a conservative estimate
        pricing = {"input": 0.003, "output": 0.015}
    cost_usd = (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)
    return max(1, int(cost_usd * 100))  # At least 1 cent per request


# ── Usage event logging ──────────────────────────────────────────────


async def _log_event(
    db: AsyncSession,
    user_id: str,
    provider: str,
    model: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    cost_cents: int,
    latency_ms: int,
    was_fallback: bool = False,
    status: str = "ok",
    operation_type: Optional[str] = None,
):
    """Log an LLM usage event.

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
    )
    db.add(event)

    is_system_op = bool(operation_type and operation_type.startswith("system."))
    if status == "ok" and not is_system_op and (input_tokens > 0 or output_tokens > 0):
        try:
            from app.services.credit_service import (
                credit_service, tokens_to_credits, BUCKET_MESSAGE,
            )
            from app.db.models import LEDGER_CHAT_MESSAGE
            credits = tokens_to_credits(model, input_tokens, output_tokens)
            await credit_service.try_charge(
                db, user_id, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, credits,
                idempotency_key=event.id, event_id=event.id, model=model,
                provider=provider, input_tokens=input_tokens, output_tokens=output_tokens,
                underlying_cost_cents=cost_cents,
                metadata={"endpoint": endpoint, "operation_type": operation_type or "user"},
            )
        except Exception as e:
            logger.warning("[credits] try_charge failed for user=%s event=%s: %s",
                           user_id[:8], event.id[:8], e)

    await db.commit()
    # Only invalidate the user-budget cache when a user-attributable event landed.
    # System operations don't affect user caps so they can leave the cache intact.
    if not is_system_op:
        _invalidate_cache(user_id)

    logger.info(
        "llm_proxy user=%s provider=%s model=%s tokens_in=%d tokens_out=%d "
        "cost_cents=%d latency=%dms fallback=%s status=%s op=%s",
        user_id[:8], provider, model, input_tokens, output_tokens,
        cost_cents, latency_ms, was_fallback, status, operation_type or "user",
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


def _extract_anthropic_usage(raw_bytes: bytes) -> tuple[int, int]:
    """Extract input_tokens and output_tokens from Anthropic SSE stream bytes."""
    import json
    input_tokens = 0
    output_tokens = 0
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
            elif obj.get("type") == "message_delta":
                usage = obj.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    return input_tokens, output_tokens


def _extract_openai_usage(raw_bytes: bytes) -> tuple[int, int]:
    """Extract usage from OpenAI SSE stream bytes."""
    import json
    input_tokens = 0
    output_tokens = 0
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
        except (json.JSONDecodeError, KeyError):
            pass
    return input_tokens, output_tokens


# ── Endpoints ────────────────────────────────────────────────────────


class UsageResponse(BaseModel):
    anthropic_monthly_cents: int
    anthropic_daily_cents: int
    openai_monthly_cents: int
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
    body = await request.json()
    requested_model = body.get("model", "claude-sonnet-4-6")
    model = _resolve_model_alias(requested_model)
    body["model"] = model  # rewrite so the upstream call uses the real id
    is_stream = body.get("stream", False)

    # Surface the resolved upstream model id back to the agent so the UI can
    # show the real provider model (e.g. "claude-opus-4-1-20250805") instead
    # of the Toup-internal alias (e.g. "claude-opus-4-7"). Read by the agent's
    # response handler from the response headers.
    resolved_model_header = {"x-toup-resolved-model": model}

    backend, api_key = _route_chat(model, config)

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
                if backend.name == "anthropic":
                    inp, out = _extract_anthropic_usage(bytes(collected_bytes))
                else:
                    inp, out = _extract_openai_usage(bytes(collected_bytes))
                cost = _calc_cost_cents(model, inp, out)
                try:
                    await _log_event(
                        db, config.user_id, backend.name, model, "chat",
                        inp, out, cost, latency, is_fallback,
                    )
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
            )
            raise HTTPException(502, f"Provider error: {e}")
        # Surface clean upstream errors (model-not-found, rate-limit, etc.)
        if resp.status_code >= 400:
            body_bytes = resp.content
            await _log_event(
                db, config.user_id, backend.name, model, "chat",
                0, 0, 0, int((time.time() - start_ts) * 1000), is_fallback, "error",
            )
            try:
                detail = body_bytes.decode("utf-8", errors="replace")
            except Exception:
                detail = str(body_bytes)
            raise HTTPException(resp.status_code, detail=detail)

        latency = int((time.time() - start_ts) * 1000)

        # Extract usage from response
        resp_data = resp.json()
        if backend.name == "anthropic":
            usage = resp_data.get("usage", {})
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
        else:
            usage = resp_data.get("usage", {})
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)

        cost = _calc_cost_cents(model, inp, out)
        await _log_event(
            db, config.user_id, backend.name, model, "chat",
            inp, out, cost, latency, is_fallback,
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


@router.post("/embeddings")
async def proxy_embeddings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Proxy an embeddings request to OpenAI."""
    config = await _auth_agent(request, db)
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
    cost = max(1, int(total_tokens * 0.00002 * 100))  # text-embedding-3-small pricing

    await _log_event(db, config.user_id, "openai", model, "embeddings", total_tokens, 0, cost, latency)

    return resp_data


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
    total_cost_cents_today: int
    anthropic_cost_cents_today: int
    openai_cost_cents_today: int
    fallback_count_today: int
    error_count_today: int
    top_users: list[dict]


@admin_router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Admin-only: aggregate LLM proxy stats for today."""
    # Simple admin check via platform auth
    from app.api.auth import get_current_user
    try:
        user = await get_current_user(request, db)
        if user.role != "admin":
            raise HTTPException(403, "Admin only")
    except Exception:
        raise HTTPException(403, "Admin only")

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
    provider_costs = {r.provider: int(r.cost) for r in by_provider}

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
        {"user_id": r.user_id[:8], "cost_cents": int(r.cost), "requests": r.cnt}
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
