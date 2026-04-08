"""
Combined checkout endpoint.

POST /api/checkout — single Stripe Checkout Session for VPS + LLM Bundle + Managed Supabase
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, VPSPlan, VPSInstance, AgentConfig, User
from app.services.stripe_service import (
    get_stripe_price_for_plan,
    create_combined_checkout_session,
    get_or_create_customer,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Checkout"])


class CombinedCheckoutRequest(BaseModel):
    vps_plan_id: Optional[str] = None       # "starter" | "standard" | "pro" | None
    llm_bundle: bool = False                 # Include $40/mo LLM bundle?
    managed_supabase: bool = False           # Include $5/mo managed Supabase?


class CombinedCheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


async def _get_or_create_config(
    user_id: str, db: AsyncSession,
) -> AgentConfig:
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    config = result.scalar_one_or_none()
    if not config:
        config = AgentConfig(id=str(uuid.uuid4()), user_id=user_id)
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


@router.post("/checkout", response_model=CombinedCheckoutResponse)
async def combined_checkout(
    body: CombinedCheckoutRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Single checkout for all paid items: VPS + LLM Bundle + Managed Supabase.
    Creates one Stripe Checkout Session with multiple line items.
    """
    line_items: list[dict] = []
    metadata: dict[str, str] = {"user_id": current_user.id}

    # ── VPS plan ──────────────────────────────────────────────
    if body.vps_plan_id:
        if not settings.vps_provisioning_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="VPS provisioning is not yet enabled.",
            )

        result = await db.execute(
            select(VPSPlan).where(VPSPlan.id == body.vps_plan_id)
        )
        plan = result.scalar_one_or_none()
        if not plan:
            raise HTTPException(400, f"Unknown VPS plan: {body.vps_plan_id}")

        # Check if user already has an active/pending VPS
        existing = await db.execute(
            select(VPSInstance)
            .where(VPSInstance.user_id == current_user.id)
            .where(VPSInstance.status.in_(["pending", "provisioning", "active"]))
        )
        if existing.scalar_one_or_none():
            raise HTTPException(409, "You already have an active or pending VPS instance.")

        # Create pending VPSInstance
        vps_id = str(uuid.uuid4())
        provider = plan.provider or "aws"
        vps = VPSInstance(
            id=vps_id,
            user_id=current_user.id,
            plan_id=plan.id,
            status="pending",
            provider=provider,
            aws_region=settings.aws_region if provider == "aws" else "",
            ami_id=settings.aws_ami_id if provider == "aws" else "",
        )
        db.add(vps)
        await db.flush()

        price_id = get_stripe_price_for_plan(plan.id)
        line_items.append({"price": price_id, "quantity": 1})
        metadata["vps_instance_id"] = vps.id
        metadata["vps_plan_id"] = plan.id

    # ── LLM Bundle ────────────────────────────────────────────
    if body.llm_bundle:
        price_id = settings.stripe_llm_bundle_price_id
        if not price_id:
            raise HTTPException(400, "LLM Bundle pricing is not configured.")
        line_items.append({"price": price_id, "quantity": 1})
        metadata["llm_bundle"] = "true"

    # ── Managed Supabase ──────────────────────────────────────
    if body.managed_supabase:
        price_id = settings.stripe_supabase_price_id
        if not price_id:
            raise HTTPException(400, "Managed Supabase pricing is not configured.")
        line_items.append({"price": price_id, "quantity": 1})
        metadata["managed_supabase"] = "true"

    if not line_items:
        raise HTTPException(400, "No paid items selected.")

    await db.commit()

    # Resolve or create Stripe customer
    user_result = await db.execute(select(User).where(User.id == current_user.id))
    user_obj = user_result.scalar_one()
    stripe_customer_id = user_obj.stripe_customer_id
    if not stripe_customer_id:
        stripe_customer_id = get_or_create_customer(
            user_id=current_user.id,
            email=current_user.email,
            name=getattr(current_user, "name", None),
        )
        user_obj.stripe_customer_id = stripe_customer_id
        await db.commit()

    # Build URLs
    base_url = "https://toup.ai"
    success_url = f"{base_url}/setup?step=6&checkout=success"
    cancel_url = f"{base_url}/setup?step=6&checkout=cancelled"

    try:
        session = create_combined_checkout_session(
            user_id=current_user.id,
            user_email=current_user.email,
            line_items=line_items,
            metadata=metadata,
            success_url=success_url,
            cancel_url=cancel_url,
            stripe_customer_id=stripe_customer_id,
        )
    except Exception as exc:
        # Clean up the pending VPS record if we created one
        if body.vps_plan_id and "vps_instance_id" in metadata:
            result = await db.execute(
                select(VPSInstance).where(VPSInstance.id == metadata["vps_instance_id"])
            )
            pending_vps = result.scalar_one_or_none()
            if pending_vps:
                await db.delete(pending_vps)
                await db.commit()
        logger.exception("Stripe combined checkout failed: %s", exc)
        raise HTTPException(502, "Payment service unavailable")

    # Store Stripe session ID on VPS instance (for webhook lookup)
    if body.vps_plan_id and "vps_instance_id" in metadata:
        result = await db.execute(
            select(VPSInstance).where(VPSInstance.id == metadata["vps_instance_id"])
        )
        pending_vps = result.scalar_one_or_none()
        if pending_vps:
            pending_vps.stripe_session_id = session["id"]
            await db.commit()

    return CombinedCheckoutResponse(
        checkout_url=session["url"],
        session_id=session["id"],
    )
