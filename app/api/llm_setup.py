"""
LLM Setup API — key validation, allocations, usage.

POST /api/llm-setup/validate-key       — validate a provider API key
GET  /api/llm-setup/bundle/allocations — get current allocations
PUT  /api/llm-setup/bundle/allocations — update allocations (must sum to 4000 cents)
GET  /api/llm-setup/bundle/usage       — usage summary for current period
POST /api/llm-setup/report-usage       — agent reports token usage (agent auth)

Note: subscription purchase goes through /api/billing/create-subscription
(embedded Stripe Payment Element), not through this module.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, update, func, case
from sqlalchemy.ext.asyncio import AsyncSession

import httpx

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, AgentConfig, LLMBundleAllocation, LLMUsageRecord, Message, Conversation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-setup", tags=["LLM Setup"])

BUNDLE_TOTAL_CENTS = 4000  # $40.00

VALID_PROVIDERS = {"openai", "anthropic", "google", "mistral", "groq", "xai", "deepseek"}


# ── Schemas ───────────────────────────────────────────────────────────────

class ValidateKeyRequest(BaseModel):
    provider: str
    api_key: str


class ValidateKeyResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


class BundleCheckoutRequest(BaseModel):
    # Still used as the payload type for PUT /bundle/allocations below.
    allocations: dict[str, int]  # provider → cents, must sum to 4000


class AllocationOut(BaseModel):
    provider: str
    allocation_cents: int
    used_cents: int


class UsageOut(BaseModel):
    allocations: list[AllocationOut]
    period_end: Optional[str] = None
    total_used_cents: int


class ModelUsageOut(BaseModel):
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    request_count: int
    source: Optional[str] = None  # "chat" | "auto_builder" — null for unaggregated entries


class ProviderUsageOut(BaseModel):
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    request_count: int
    models: list[ModelUsageOut]


class UsageSummaryOut(BaseModel):
    period: str  # today, 7d, 30d, all
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_requests: int
    providers: list[ProviderUsageOut]


class ReportUsageRequest(BaseModel):
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/validate-key", response_model=ValidateKeyResponse)
async def validate_key(body: ValidateKeyRequest, current_user=Depends(get_current_user)):
    """Validate an LLM provider API key via real API call."""
    from app.services.llm_key_validator import validate_key as _validate
    result = await _validate(body.provider, body.api_key)
    return ValidateKeyResponse(**result)


@router.get("/bundle/allocations", response_model=list[AllocationOut])
async def get_allocations(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current bundle allocations for the user."""
    result = await db.execute(
        select(LLMBundleAllocation).where(LLMBundleAllocation.user_id == current_user.id)
    )
    allocs = result.scalars().all()
    return [
        AllocationOut(
            provider=a.provider,
            allocation_cents=a.allocation_cents,
            used_cents=a.used_cents,
        )
        for a in allocs
    ]


@router.put("/bundle/allocations", response_model=list[AllocationOut])
async def update_allocations(
    body: BundleCheckoutRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update bundle allocations (must sum to 4000 cents)."""
    total = sum(body.allocations.values())
    if total != BUNDLE_TOTAL_CENTS:
        raise HTTPException(400, f"Allocations must sum to {BUNDLE_TOTAL_CENTS} cents, got {total}")

    for provider, cents in body.allocations.items():
        if provider not in VALID_PROVIDERS:
            raise HTTPException(400, f"Unknown provider: {provider}")
        result = await db.execute(
            select(LLMBundleAllocation).where(
                LLMBundleAllocation.user_id == current_user.id,
                LLMBundleAllocation.provider == provider,
            )
        )
        alloc = result.scalar_one_or_none()
        if alloc:
            alloc.allocation_cents = cents
            alloc.updated_at = datetime.utcnow()
        else:
            alloc = LLMBundleAllocation(
                id=str(uuid.uuid4()),
                user_id=current_user.id,
                provider=provider,
                allocation_cents=cents,
            )
            db.add(alloc)
    await db.commit()

    return await get_allocations(current_user=current_user, db=db)


@router.get("/bundle/usage", response_model=UsageOut)
async def get_usage(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get usage summary for the current billing period."""
    # Get allocations
    result = await db.execute(
        select(LLMBundleAllocation).where(LLMBundleAllocation.user_id == current_user.id)
    )
    allocs = result.scalars().all()

    # Get config for period end
    cfg_result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == current_user.id)
    )
    config = cfg_result.scalar_one_or_none()

    period_end = None
    if config and config.bundle_current_period_end:
        period_end = config.bundle_current_period_end.isoformat()

    total_used = sum(a.used_cents for a in allocs)

    return UsageOut(
        allocations=[
            AllocationOut(
                provider=a.provider,
                allocation_cents=a.allocation_cents,
                used_cents=a.used_cents,
            )
            for a in allocs
        ],
        period_end=period_end,
        total_used_cents=total_used,
    )


PROVIDER_PREFIXES = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
    "mistral": "mistral",
    "llama": "groq",
    "mixtral": "groq",
    "grok": "xai",
    "deepseek": "deepseek",
}


def model_to_provider(model: str) -> str:
    """Derive provider from model name."""
    m = (model or "").lower()
    for prefix, provider in PROVIDER_PREFIXES.items():
        if m.startswith(prefix):
            return provider
    return "other"


async def _merge_build_usage(
    user_id: str, since: datetime, provider_map: dict, db: AsyncSession,
) -> None:
    """Merge auto-builder LLM usage (BuildUsage table) into an in-progress provider_map.

    Runs in a fresh, isolated session so a missing table (build_usage doesn't exist
    on platform-only DBs) can't poison the caller's transaction. Model entries are
    tagged `source=\"auto_builder\"` so the UI can tell them apart from chat."""
    try:
        from app.db.models import BuildUsage
        from app.db.database import async_session_maker
    except Exception:
        return
    rows = []
    try:
        async with async_session_maker() as iso:
            stmt = (
                select(
                    BuildUsage.provider,
                    BuildUsage.model,
                    func.coalesce(func.sum(BuildUsage.input_tokens), 0).label("inp"),
                    func.coalesce(func.sum(BuildUsage.output_tokens), 0).label("out"),
                    func.coalesce(func.sum(BuildUsage.cost_usd), 0.0).label("cost"),
                    func.count().label("cnt"),
                )
                .where(BuildUsage.user_id == user_id, BuildUsage.created_at >= since)
                .group_by(BuildUsage.provider, BuildUsage.model)
            )
            rows = (await iso.execute(stmt)).all()
    except Exception as _e:
        logger.warning("BuildUsage aggregation skipped: %s", _e)
        return
    for provider, model, inp, out, cost, cnt in rows:
        inp_i = int(inp); out_i = int(out); cnt_i = int(cnt); cost_f = float(cost)
        if provider not in provider_map:
            provider_map[provider] = {
                "provider": provider, "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "cost_usd": 0.0, "request_count": 0, "models": [],
            }
        prov = provider_map[provider]
        prov["input_tokens"] += inp_i
        prov["output_tokens"] += out_i
        prov["total_tokens"] += inp_i + out_i
        prov["cost_usd"] = round(prov["cost_usd"] + cost_f, 4)
        prov["request_count"] += cnt_i
        prov["models"].append({
            "model": model or "unknown",
            "provider": provider,
            "input_tokens": inp_i,
            "output_tokens": out_i,
            "total_tokens": inp_i + out_i,
            "cost_usd": round(cost_f, 4),
            "request_count": cnt_i,
            "source": "auto_builder",
        })


@router.get("/usage/summary", response_model=list[UsageSummaryOut])
async def get_usage_summary(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed usage summary broken down by provider and model for multiple time periods."""
    from datetime import timedelta

    # Cache primitive user_id BEFORE any session manipulation. Once we
    # rollback the session below, current_user becomes expired and any
    # later attribute access (e.g. current_user.id) triggers a lazy
    # SELECT that can fail in surprising ways. Working with a plain str
    # avoids the entire ORM-state-machine class of bug.
    current_user_id = current_user.id

    # Try proxying to remote agent first (platform has no local messages).
    # On agent containers `agent_configs` doesn't exist — that throws, leaves
    # the asyncpg session in InFailedSQLTransaction, then the local-DB
    # fallback below also fails. Rollback after the probe so the fallback
    # can reuse the same session.
    try:
        result = await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key)
            .where(AgentConfig.user_id == current_user_id, AgentConfig.deploy_status == "active")
        )
        row = result.first()
        if row and row.agent_url and row.agent_api_key:
            url = f"{row.agent_url}/api/llm-setup/usage/summary"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers={"X-Agent-Key": row.agent_api_key})
                if resp.status_code == 200:
                    return resp.json()
            logger.warning("Agent usage proxy failed, falling back to local DB")
    except Exception as e:
        logger.warning("Agent usage proxy error: %s, falling back to local DB", e)
        try:
            await db.rollback()
        except Exception:
            pass

    now = datetime.utcnow()
    periods = {
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "7d": now - timedelta(days=7),
        "30d": now - timedelta(days=30),
        "all": datetime(2000, 1, 1),
    }

    # Get user's conversations
    conv_result = await db.execute(
        select(Conversation.id).where(Conversation.user_id == current_user_id)
    )
    conv_ids = [r[0] for r in conv_result.all()]

    # Query messages grouped by model for each period, then merge BuildUsage (auto builder).
    results = []
    pricing = settings.pricing_per_1k

    for period_name, since in periods.items():
        provider_map: dict[str, dict] = {}

        if conv_ids:
            stmt = (
                select(
                    Message.model_used,
                    func.coalesce(func.sum(func.coalesce(Message.tokens_prompt, 0)), 0).label("inp"),
                    func.coalesce(func.sum(func.coalesce(Message.tokens_completion, 0)), 0).label("out"),
                    func.count().label("cnt"),
                )
                .where(
                    Message.conversation_id.in_(conv_ids),
                    Message.role == "assistant",
                    Message.model_used.isnot(None),
                    Message.created_at >= since,
                )
                .group_by(Message.model_used)
            )
            rows = (await db.execute(stmt)).all()

            for model_used, inp, out, cnt in rows:
                provider = model_to_provider(model_used or "")
                p_cost = pricing.get(model_used, {"input": 0.003, "output": 0.012})
                cost = (inp * p_cost["input"] / 1000) + (out * p_cost["output"] / 1000)

                if provider not in provider_map:
                    provider_map[provider] = {
                        "provider": provider, "input_tokens": 0, "output_tokens": 0,
                        "total_tokens": 0, "cost_usd": 0.0, "request_count": 0, "models": [],
                    }
                prov = provider_map[provider]
                prov["input_tokens"] += int(inp)
                prov["output_tokens"] += int(out)
                prov["total_tokens"] += int(inp + out)
                prov["cost_usd"] = round(prov["cost_usd"] + cost, 4)
                prov["request_count"] += int(cnt)
                prov["models"].append({
                    "model": model_used or "unknown",
                    "provider": provider,
                    "input_tokens": int(inp),
                    "output_tokens": int(out),
                    "total_tokens": int(inp + out),
                    "cost_usd": round(cost, 4),
                    "request_count": int(cnt),
                    "source": "chat",
                })

        await _merge_build_usage(current_user_id, since, provider_map, db)

        providers_sorted = sorted(provider_map.values(), key=lambda p: p["total_tokens"], reverse=True)
        total_inp = sum(p["input_tokens"] for p in providers_sorted)
        total_out = sum(p["output_tokens"] for p in providers_sorted)

        results.append(UsageSummaryOut(
            period=period_name,
            total_input_tokens=total_inp,
            total_output_tokens=total_out,
            total_tokens=total_inp + total_out,
            total_cost_usd=round(sum(p["cost_usd"] for p in providers_sorted), 4),
            total_requests=sum(p["request_count"] for p in providers_sorted),
            providers=[ProviderUsageOut(**p) for p in providers_sorted],
        ))

    return results


@router.post("/report-usage")
async def report_usage(
    body: ReportUsageRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Agent reports LLM token usage. Authenticated via agent API key."""
    # Authenticate via agent API key (from header)
    api_key = request.headers.get("X-Toup-Token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        raise HTTPException(401, "Missing API key")

    result = await db.execute(
        select(AgentConfig).where(AgentConfig.agent_api_key == api_key)
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(401, "Invalid API key")

    # Allow active, cancelling (paid through period end), and past_due (Stripe retrying)
    _allowed = {"active", "cancelling", "past_due"}
    if config.llm_mode != "bundle" or config.bundle_status not in _allowed:
        raise HTTPException(400, "Bundle not active")

    # If cancelling, check if period has actually ended
    if config.bundle_status == "cancelling" and config.bundle_current_period_end:
        if datetime.utcnow() > config.bundle_current_period_end:
            raise HTTPException(400, "Bundle subscription has expired")

    # Calculate cost
    from app.config import settings as s
    pricing = s.pricing_per_1k.get(body.model, {"input": 0.003, "output": 0.012})
    cost_usd = (body.input_tokens * pricing["input"] / 1000) + (body.output_tokens * pricing["output"] / 1000)
    cost_cents = int(cost_usd * 100)

    # Record usage
    record = LLMUsageRecord(
        id=str(uuid.uuid4()),
        user_id=config.user_id,
        provider=body.provider,
        model=body.model,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        cost_usd=cost_usd,
    )
    db.add(record)

    # Update allocation used_cents
    alloc_result = await db.execute(
        select(LLMBundleAllocation).where(
            LLMBundleAllocation.user_id == config.user_id,
            LLMBundleAllocation.provider == body.provider,
        )
    )
    alloc = alloc_result.scalar_one_or_none()
    if alloc:
        alloc.used_cents += cost_cents
        alloc.updated_at = datetime.utcnow()

    await db.commit()
    return {"recorded": True, "cost_usd": round(cost_usd, 6)}
