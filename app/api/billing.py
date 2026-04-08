"""
Billing API — Stripe customer portal, subscription status, and pricing.

POST /api/billing/portal    — create Stripe Customer Portal session
GET  /api/billing/status    — current subscription status for the user
GET  /api/billing/prices    — live pricing from backend config
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, AgentConfig, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["Billing"])


# ── Schemas ──────────────────────────────────────────────────────────


class PortalRequest(BaseModel):
    return_url: Optional[str] = None


class PortalResponse(BaseModel):
    url: str


class BillingStatusResponse(BaseModel):
    bundle_status: str  # none | active | cancelling | past_due | cancelled
    bundle_current_period_end: Optional[str] = None
    bundle_started_at: Optional[str] = None
    has_stripe_customer: bool
    stripe_customer_id: Optional[str] = None


class PriceItem(BaseModel):
    id: str
    name: str
    amount_cents: int
    interval: str


class PricesResponse(BaseModel):
    llm_bundle: Optional[PriceItem] = None
    vps_plans: list[PriceItem] = []


# ── Endpoints ────────────────────────────────────────────────────────


@router.post("/portal", response_model=PortalResponse)
async def create_portal(
    body: PortalRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a Stripe Customer Portal session for self-serve billing management."""
    from app.services.stripe_service import (
        create_billing_portal_session,
        get_or_create_customer,
    )

    # Ensure user has a Stripe customer
    user_result = await db.execute(select(User).where(User.id == current_user.id))
    user = user_result.scalar_one()

    if not user.stripe_customer_id:
        cust_id = get_or_create_customer(
            user_id=user.id,
            email=user.email,
            name=user.name,
        )
        user.stripe_customer_id = cust_id
        await db.commit()

    return_url = body.return_url or "https://toup.ai/setup"
    url = create_billing_portal_session(user.stripe_customer_id, return_url)
    return PortalResponse(url=url)


@router.get("/status", response_model=BillingStatusResponse)
async def get_billing_status(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's subscription status."""
    cfg_result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == current_user.id)
    )
    config = cfg_result.scalar_one_or_none()

    user_result = await db.execute(select(User).where(User.id == current_user.id))
    user = user_result.scalar_one()

    if not config:
        return BillingStatusResponse(
            bundle_status="none",
            has_stripe_customer=bool(user.stripe_customer_id),
            stripe_customer_id=user.stripe_customer_id,
        )

    return BillingStatusResponse(
        bundle_status=config.bundle_status or "none",
        bundle_current_period_end=(
            config.bundle_current_period_end.isoformat()
            if config.bundle_current_period_end else None
        ),
        bundle_started_at=(
            config.bundle_started_at.isoformat()
            if config.bundle_started_at else None
        ),
        has_stripe_customer=bool(user.stripe_customer_id),
        stripe_customer_id=user.stripe_customer_id,
    )


@router.get("/prices", response_model=PricesResponse)
async def get_prices(db: AsyncSession = Depends(get_db)):
    """
    Return configured pricing. The LLM bundle price is read from the
    Stripe Price object (cached at startup) so the frontend never
    hardcodes dollar amounts.
    """
    from app.services.stripe_service import get_price
    from app.db import VPSPlan

    result = PricesResponse()

    # LLM Bundle price
    if settings.stripe_llm_bundle_price_id:
        price_info = get_price(settings.stripe_llm_bundle_price_id)
        if price_info:
            result.llm_bundle = PriceItem(
                id="llm_bundle",
                name=price_info["product_name"] or "Toup LLM Bundle",
                amount_cents=price_info["unit_amount"],
                interval=price_info["interval"] or "month",
            )

    # VPS plans from DB
    plans_result = await db.execute(select(VPSPlan).order_by(VPSPlan.price_cents))
    for plan in plans_result.scalars().all():
        result.vps_plans.append(PriceItem(
            id=plan.id,
            name=plan.name,
            amount_cents=plan.price_cents,
            interval="month",
        ))

    return result
