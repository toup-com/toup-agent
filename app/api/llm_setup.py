"""
LLM Setup API — key validation, bundle checkout, allocations, usage.

POST /api/llm-setup/validate-key       — validate a provider API key
POST /api/llm-setup/bundle/checkout    — create Stripe checkout for $40/mo bundle
GET  /api/llm-setup/bundle/allocations — get current allocations
PUT  /api/llm-setup/bundle/allocations — update allocations (must sum to 4000 cents)
GET  /api/llm-setup/bundle/usage       — usage summary for current period
POST /api/llm-setup/report-usage       — agent reports token usage (agent auth)
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, AgentConfig, LLMBundleAllocation, LLMUsageRecord

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
    allocations: dict[str, int]  # provider → cents, must sum to 4000


class BundleCheckoutResponse(BaseModel):
    checkout_url: str


class AllocationOut(BaseModel):
    provider: str
    allocation_cents: int
    used_cents: int


class UsageOut(BaseModel):
    allocations: list[AllocationOut]
    period_end: Optional[str] = None
    total_used_cents: int


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


@router.post("/bundle/checkout", response_model=BundleCheckoutResponse)
async def bundle_checkout(
    body: BundleCheckoutRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a Stripe checkout session for the $40/mo LLM bundle."""
    # Validate allocations sum
    total = sum(body.allocations.values())
    if total != BUNDLE_TOTAL_CENTS:
        raise HTTPException(400, f"Allocations must sum to {BUNDLE_TOTAL_CENTS} cents, got {total}")

    for provider in body.allocations:
        if provider not in VALID_PROVIDERS:
            raise HTTPException(400, f"Unknown provider: {provider}")
        if body.allocations[provider] < 0:
            raise HTTPException(400, f"Negative allocation for {provider}")

    # Save allocations
    for provider, cents in body.allocations.items():
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

    # Create Stripe checkout
    from app.services.stripe_service import create_bundle_checkout_session
    try:
        session = create_bundle_checkout_session(
            user_id=current_user.id,
            user_email=current_user.email,
            success_url=f"{settings.platform_api_url.rstrip('/api')}/agent?bundle_success=1",
            cancel_url=f"{settings.platform_api_url.rstrip('/api')}/agent?bundle_cancelled=1",
        )
    except Exception as e:
        logger.exception("Stripe bundle checkout error: %s", e)
        raise HTTPException(500, f"Stripe error: {e}")

    return BundleCheckoutResponse(checkout_url=session["url"])


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

    if config.llm_mode != "bundle" or config.bundle_status != "active":
        raise HTTPException(400, "Bundle not active")

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
