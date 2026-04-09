"""
Billing API — Stripe subscriptions, customer portal, status, pricing, config.

GET  /api/billing/config                 — Stripe publishable key for frontend
POST /api/billing/create-subscription    — create subscription + return client_secret for Elements
POST /api/billing/portal                 — create Stripe Customer Portal session
GET  /api/billing/status                 — current subscription status for the user
GET  /api/billing/prices                 — live pricing from backend config
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


class BillingConfigResponse(BaseModel):
    publishable_key: str


class CreateSubscriptionResponse(BaseModel):
    subscription_id: str
    client_secret: Optional[str] = None
    status: str  # incomplete | active | already_active


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


# ── Helpers ──────────────────────────────────────────────────────────


async def _ensure_stripe_customer(user, db: AsyncSession) -> str:
    """Return user's stripe_customer_id, creating one if needed."""
    from app.services.stripe_service import get_or_create_customer

    if user.stripe_customer_id:
        return user.stripe_customer_id

    cust_id = get_or_create_customer(
        user_id=user.id,
        email=user.email,
        name=user.name,
    )
    user.stripe_customer_id = cust_id
    await db.commit()
    return cust_id


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/config", response_model=BillingConfigResponse)
async def get_billing_config():
    """Return the Stripe publishable key for the frontend to initialize Elements."""
    if not settings.stripe_publishable_key:
        raise HTTPException(500, "Stripe publishable key not configured")
    return BillingConfigResponse(publishable_key=settings.stripe_publishable_key)


@router.post("/create-subscription", response_model=CreateSubscriptionResponse)
async def create_subscription(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Stripe Subscription with payment_behavior='default_incomplete'.
    Returns the client_secret for the frontend PaymentElement.

    Idempotent: if the user already has an active bundle subscription,
    returns it without creating a new one.
    """
    from app.services.stripe_service import (
        create_subscription_with_intent,
        get_active_subscription_for_customer,
    )

    price_id = settings.stripe_llm_bundle_price_id
    if not price_id:
        raise HTTPException(400, "LLM Bundle pricing is not configured.")

    # Get user + ensure Stripe customer
    user_result = await db.execute(select(User).where(User.id == current_user.id))
    user = user_result.scalar_one()
    customer_id = await _ensure_stripe_customer(user, db)

    # Check AgentConfig — already active? Don't create another subscription.
    cfg_result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == current_user.id)
    )
    config = cfg_result.scalar_one_or_none()
    if config and config.bundle_status == "active":
        return CreateSubscriptionResponse(
            subscription_id=config.bundle_stripe_subscription_id or "",
            status="already_active",
        )

    # Check Stripe — existing active or incomplete subscription?
    existing = get_active_subscription_for_customer(customer_id, price_id)
    if existing:
        if existing["status"] == "active":
            return CreateSubscriptionResponse(
                subscription_id=existing["subscription_id"],
                status="already_active",
            )
        # Incomplete — return the existing client_secret so user can retry payment
        if existing.get("client_secret"):
            return CreateSubscriptionResponse(
                subscription_id=existing["subscription_id"],
                client_secret=existing["client_secret"],
                status="incomplete",
            )

    # Create new subscription
    result = create_subscription_with_intent(
        customer_id=customer_id,
        price_id=price_id,
        user_id=current_user.id,
    )

    # Save subscription ID to AgentConfig immediately (status still incomplete)
    if config:
        config.bundle_stripe_subscription_id = result["subscription_id"]
        await db.commit()

    return CreateSubscriptionResponse(**result)


@router.post("/portal", response_model=PortalResponse)
async def create_portal(
    body: PortalRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a Stripe Customer Portal session for self-serve billing management."""
    from app.services.stripe_service import create_billing_portal_session

    user_result = await db.execute(select(User).where(User.id == current_user.id))
    user = user_result.scalar_one()
    customer_id = await _ensure_stripe_customer(user, db)

    return_url = body.return_url or "https://toup.ai/setup"
    url = create_billing_portal_session(customer_id, return_url)
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
    """Return configured pricing from Stripe."""
    from app.services.stripe_service import get_price
    from app.db import VPSPlan

    result = PricesResponse()

    if settings.stripe_llm_bundle_price_id:
        price_info = get_price(settings.stripe_llm_bundle_price_id)
        if price_info:
            result.llm_bundle = PriceItem(
                id="llm_bundle",
                name=price_info["product_name"] or "Toup LLM Bundle",
                amount_cents=price_info["unit_amount"],
                interval=price_info["interval"] or "month",
            )

    plans_result = await db.execute(select(VPSPlan).order_by(VPSPlan.price_cents))
    for plan in plans_result.scalars().all():
        result.vps_plans.append(PriceItem(
            id=plan.id,
            name=plan.name,
            amount_cents=plan.price_cents,
            interval="month",
        ))

    return result
