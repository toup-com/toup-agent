"""Credits API — workspace credit panel + ledger surface.

GET  /api/credits/status   — current balance, plan, period, daily counter
GET  /api/credits/ledger   — recent deductions for run-history display
GET  /api/billing/plans    — public plan catalog (no auth)
POST /api/admin/credits/plans/stripe-price-id  — backfill a tier's Stripe Price ID
POST /api/admin/credits/flat-fees              — tune a flat-fee tool cost
GET  /api/admin/credits/flat-fees              — read current overrides
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import (
    CreditLedger, PlatformSetting, SubscriptionPlan, User,
)
from app.services.credit_service import FLAT_FEES, credit_service


router = APIRouter(prefix="/credits", tags=["Credits"])
billing_router = APIRouter(prefix="/billing", tags=["Billing"])
admin_router = APIRouter(prefix="/admin/credits", tags=["Admin Credits"])


async def sync_subscription_plan_stripe_ids(db_session_maker) -> dict[str, str]:
    """Sync env-configured Stripe price IDs into subscription_plans.

    Called on platform-api boot. Idempotent. If the operator clears an
    env var, the prior value is preserved (we don't NULL out price IDs
    accidentally on restart). Returns the {tier:price_id} map applied
    for boot-log visibility.
    """
    from app.config import settings as _s
    pairs = {
        "starter": _s.stripe_price_id_starter,
        "builder": _s.stripe_price_id_builder,
        "pro":     _s.stripe_price_id_pro,
        "elite":   _s.stripe_price_id_elite,
    }
    applied: dict[str, str] = {}
    async with db_session_maker() as db:
        for tier_id, price_id in pairs.items():
            price_id = (price_id or "").strip()
            if not price_id:
                continue
            plan = await db.get(SubscriptionPlan, tier_id)
            if plan is None:
                continue
            if plan.stripe_price_id != price_id:
                plan.stripe_price_id = price_id
                applied[tier_id] = price_id
        if applied:
            await db.commit()
    return applied


# ── Schemas ─────────────────────────────────────────────────────────


class BucketStatus(BaseModel):
    remaining: float
    monthly: float
    used_today: Optional[float] = None
    daily_cap: Optional[float] = None


class CreditStatusResponse(BaseModel):
    plan_id: str
    plan_display_name: str
    message: BucketStatus
    integration: BucketStatus
    period_start: datetime
    period_end: datetime
    enforcement_enabled: bool


class LedgerRow(BaseModel):
    id: str
    event_type: str
    bucket: str
    amount: float
    balance_after: float
    event_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    created_at: datetime


class LedgerResponse(BaseModel):
    rows: list[LedgerRow]


class PlanRow(BaseModel):
    id: str
    display_name: str
    price_cents: int
    message_credits_monthly: float
    integration_credits_monthly: float
    message_credits_daily_cap: Optional[float] = None
    rollover_message_credits: bool
    rollover_integration_credits: bool
    rollover_max_pct: Optional[float] = None
    stripe_price_id: Optional[str] = None
    sort_order: int


class PlansResponse(BaseModel):
    plans: list[PlanRow]


# ── User-facing endpoints ──────────────────────────────────────────


@router.get("/status", response_model=CreditStatusResponse)
async def get_credit_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CreditStatusResponse:
    """Workspace credit panel data — what the sidebar drawer renders."""
    view = await credit_service.get_balance_view(db, current_user.id)
    return CreditStatusResponse(
        plan_id=view.plan_id,
        plan_display_name=view.plan_display_name,
        message=BucketStatus(
            remaining=float(view.message_credits_remaining),
            monthly=float(view.message_credits_monthly),
            used_today=float(view.message_credits_used_today),
            daily_cap=(float(view.message_credits_daily_cap)
                       if view.message_credits_daily_cap is not None else None),
        ),
        integration=BucketStatus(
            remaining=float(view.integration_credits_remaining),
            monthly=float(view.integration_credits_monthly),
        ),
        period_start=view.period_start,
        period_end=view.period_end,
        enforcement_enabled=view.enforcement_enabled,
    )


@router.get("/ledger", response_model=LedgerResponse)
async def get_credit_ledger(
    limit: int = Query(50, le=200),
    event_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LedgerResponse:
    """Recent ledger rows. Optional event_id filter for per-routine/per-task display."""
    stmt = (
        select(CreditLedger).where(CreditLedger.user_id == current_user.id)
        .order_by(CreditLedger.created_at.desc()).limit(limit)
    )
    if event_id:
        stmt = stmt.where(CreditLedger.event_id == event_id)
    rows = (await db.execute(stmt)).scalars().all()
    return LedgerResponse(rows=[
        LedgerRow(
            id=r.id, event_type=r.event_type, bucket=r.bucket,
            amount=float(r.amount), balance_after=float(r.balance_after),
            event_id=r.event_id, model=r.model, provider=r.provider,
            created_at=r.created_at,
        ) for r in rows
    ])


@billing_router.get("/plans", response_model=PlansResponse)
async def list_plans(db: AsyncSession = Depends(get_db)) -> PlansResponse:
    """Public plan catalog for the pricing page. No auth."""
    rows = (await db.execute(
        select(SubscriptionPlan).where(SubscriptionPlan.active.is_(True))
        .order_by(SubscriptionPlan.sort_order)
    )).scalars().all()
    return PlansResponse(plans=[
        PlanRow(
            id=p.id, display_name=p.display_name, price_cents=p.price_cents,
            message_credits_monthly=float(p.message_credits_monthly),
            integration_credits_monthly=float(p.integration_credits_monthly),
            message_credits_daily_cap=(float(p.message_credits_daily_cap)
                                       if p.message_credits_daily_cap is not None else None),
            rollover_message_credits=p.rollover_message_credits,
            rollover_integration_credits=p.rollover_integration_credits,
            rollover_max_pct=(float(p.rollover_max_pct)
                              if p.rollover_max_pct is not None else None),
            stripe_price_id=p.stripe_price_id,
            sort_order=p.sort_order,
        ) for p in rows
    ])


# ── Stripe Checkout for credit-tier upgrade ───────────────────────


class CreditCheckoutResponse(BaseModel):
    url: str


@billing_router.post("/credit-checkout/{plan_id}", response_model=CreditCheckoutResponse)
async def create_credit_checkout(
    plan_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CreditCheckoutResponse:
    """Create a Stripe Checkout Session for a credit-tier subscription.

    Distinct from the legacy /billing/create-subscription (which is for
    the old LLM Bundle): this endpoint mints the checkout for the four
    new credit tiers (starter/builder/pro/elite). The Stripe price id
    comes from `subscription_plans.stripe_price_id` which the boot-
    time `sync_subscription_plan_stripe_ids` populates from env vars
    `STRIPE_PRICE_ID_{TIER}`.

    Returns 503 with a clear, actionable error when the tier's price
    id isn't configured — operators see "set STRIPE_PRICE_ID_X in
    Railway" instead of a cryptic Stripe error.
    """
    from app.config import settings as _settings
    from app.api.billing import _ensure_stripe_customer
    from app.services.stripe_service import create_credit_checkout_session

    if plan_id == "free":
        raise HTTPException(400, "The free tier doesn't require checkout.")

    plan = await db.get(SubscriptionPlan, plan_id)
    if plan is None:
        raise HTTPException(404, f"Unknown plan: {plan_id!r}")
    if not plan.stripe_price_id:
        raise HTTPException(
            503,
            f"{plan.display_name} isn't available for checkout yet — "
            f"Stripe price not configured. Set "
            f"STRIPE_PRICE_ID_{plan_id.upper()} in Railway env "
            f"and redeploy.",
        )
    if not _settings.stripe_secret_key:
        raise HTTPException(503, "Stripe secret key not configured on platform.")

    user_result = await db.execute(select(User).where(User.id == current_user.id))
    user = user_result.scalar_one()
    customer_id = await _ensure_stripe_customer(user, db)

    base_url = (_settings.app_public_base_url or "https://toup.ai").rstrip("/")
    try:
        session = create_credit_checkout_session(
            customer_id=customer_id,
            price_id=plan.stripe_price_id,
            plan_id=plan_id,
            user_id=current_user.id,
            success_url=f"{base_url}/account?upgrade=success&plan={plan_id}",
            cancel_url=f"{base_url}/pricing?upgrade=cancelled",
        )
    except Exception as e:
        raise HTTPException(502, f"Stripe checkout creation failed: {e}")
    return CreditCheckoutResponse(url=session["url"])


# ── Admin endpoints ─────────────────────────────────────────────────


class StripePriceIdUpdate(BaseModel):
    plan_id: str
    stripe_price_id: str


@admin_router.post("/plans/stripe-price-id")
async def admin_set_plan_stripe_price_id(
    payload: StripePriceIdUpdate,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Admin: backfill a Stripe Price ID for one tier without a redeploy."""
    plan = await db.get(SubscriptionPlan, payload.plan_id)
    if plan is None:
        raise HTTPException(404, f"plan {payload.plan_id!r} not found")
    plan.stripe_price_id = payload.stripe_price_id.strip()
    await db.commit()
    return {"plan_id": plan.id, "stripe_price_id": plan.stripe_price_id}


class StripePriceIdBulkUpdate(BaseModel):
    """Bulk update — set all four credit-tier Stripe Price IDs at once.

    Useful for operators who want to wire up Stripe via a single POST
    without flipping Railway env vars and waiting for a redeploy.
    Empty strings are ignored (preserves existing value).
    """
    starter: Optional[str] = None
    builder: Optional[str] = None
    pro: Optional[str] = None
    elite: Optional[str] = None


@admin_router.post("/plans/stripe-price-ids/bulk")
async def admin_set_plan_stripe_price_ids_bulk(
    payload: StripePriceIdBulkUpdate,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Set all credit-tier price IDs in one call. Idempotent; empty
    values preserved. Returns the resulting {tier_id: price_id} map."""
    updates = {
        "starter": (payload.starter or "").strip(),
        "builder": (payload.builder or "").strip(),
        "pro":     (payload.pro or "").strip(),
        "elite":   (payload.elite or "").strip(),
    }
    applied: dict[str, str] = {}
    for tier_id, price_id in updates.items():
        if not price_id:
            continue
        plan = await db.get(SubscriptionPlan, tier_id)
        if plan is None:
            continue
        plan.stripe_price_id = price_id
        applied[tier_id] = price_id
    if applied:
        await db.commit()
    # Return the FULL set so caller sees what's now in the DB.
    rows = (await db.execute(
        select(SubscriptionPlan).where(
            SubscriptionPlan.id.in_(["starter", "builder", "pro", "elite"])
        )
    )).scalars().all()
    return {
        "applied": applied,
        "current": {r.id: r.stripe_price_id for r in rows},
    }


class CreditBalanceRehydrate(BaseModel):
    user_id: str


@admin_router.post("/balance/rehydrate")
async def admin_rehydrate_balance(
    payload: CreditBalanceRehydrate,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Force-create a CreditBalance row for a user.

    Useful for diagnosing "credits aren't deducting" — users created
    before mig 053 don't have a balance row, and the
    `credit_service.get_or_create_balance` is called lazily on first
    deduction. If the deduction hook silently fails (try/except swallow),
    the row never gets created and subsequent reads show zero usage.
    This endpoint forces creation so the operator can verify writes.
    """
    target = await db.get(User, payload.user_id)
    if target is None:
        raise HTTPException(404, f"user {payload.user_id!r} not found")
    balance = await credit_service.get_or_create_balance(db, payload.user_id)
    await db.commit()
    return {
        "user_id": target.id,
        "plan_id": balance.plan_id,
        "message_credits_remaining": float(balance.message_credits_remaining),
        "integration_credits_remaining": float(balance.integration_credits_remaining),
    }


class FlatFeeUpdate(BaseModel):
    fee_key: str
    credits: float
    bucket: Optional[str] = None


@admin_router.post("/flat-fees")
async def admin_set_flat_fee(
    payload: FlatFeeUpdate,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Admin: tune a flat-fee tool cost without a redeploy.

    Updates in-process FLAT_FEES AND persists to platform_settings so
    the change survives restart (rehydrated by
    ``credit_service.load_flat_fee_overrides``).
    """
    from decimal import Decimal as _Dec
    if payload.fee_key not in FLAT_FEES:
        raise HTTPException(404, f"unknown fee_key {payload.fee_key!r}")
    new_bucket = payload.bucket or FLAT_FEES[payload.fee_key]["bucket"]
    FLAT_FEES[payload.fee_key] = {"bucket": new_bucket, "credits": _Dec(str(payload.credits))}
    setting_key = f"credit.flat_fee.{payload.fee_key}"
    existing = await db.get(PlatformSetting, setting_key)
    value = json.dumps({"bucket": new_bucket, "credits": str(payload.credits)})
    if existing is None:
        db.add(PlatformSetting(key=setting_key, value=value))
    else:
        existing.value = value
    await db.commit()
    return {"fee_key": payload.fee_key, "applied": {
        "bucket": new_bucket, "credits": float(FLAT_FEES[payload.fee_key]["credits"]),
    }}


@admin_router.get("/flat-fees")
async def admin_list_flat_fees(_admin: User = Depends(require_admin)) -> dict[str, dict[str, Any]]:
    return {k: {"bucket": v["bucket"], "credits": float(v["credits"])} for k, v in FLAT_FEES.items()}
