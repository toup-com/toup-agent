"""
VPS provisioning endpoints.

GET  /api/vps/plans         — list available plans (public)
POST /api/vps/checkout      — create Stripe checkout session (authenticated)
GET  /api/vps/status        — get current user's VPS status (authenticated)
POST /api/vps/webhook/stripe — Stripe webhook (no auth, verified by signature)
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, VPSPlan, VPSInstance
from app.services.stripe_service import create_checkout_session, verify_webhook
from app.services.aws_service import provision_instance as aws_provision_instance
from app.services.hostinger_service import provision_instance as hostinger_provision_instance
from app.services.hetzner_service import provision_instance as hetzner_provision_instance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vps", tags=["VPS"])

# ── Response schemas ──────────────────────────────────────────────────────────

class VPSPlanOut(BaseModel):
    id: str
    name: str
    instance_type: str
    vcpu: int
    ram_gb: int
    storage_gb: int
    price_cents: int
    price_display: str  # e.g. "$10/mo"
    provider: str = "aws"  # "aws" | "hostinger"

    class Config:
        from_attributes = True


class VPSCheckoutRequest(BaseModel):
    plan_id: str


class VPSCheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


class VPSStatusResponse(BaseModel):
    status: str  # pending | provisioning | active | error | terminated | none
    plan_id: str | None = None
    plan_name: str | None = None
    provider: str | None = None  # "aws" | "hostinger"
    public_ip: str | None = None
    public_dns: str | None = None
    ssh_password: str | None = None  # Only returned when status first becomes active
    error_message: str | None = None
    provisioned_at: str | None = None


class AgentURLResponse(BaseModel):
    agent_url: str | None = None  # e.g. "http://1.2.3.4:8001"
    agent_ws_url: str | None = None  # e.g. "ws://1.2.3.4:8001/api/ws/chat"
    agent_api_key: str | None = None
    status: str  # active | none | provisioning | ...


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/plans", response_model=list[VPSPlanOut])
async def list_plans(db: AsyncSession = Depends(get_db)):
    """Return all available VPS plans. No authentication required."""
    result = await db.execute(select(VPSPlan).order_by(VPSPlan.price_cents))
    plans = result.scalars().all()

    return [
        VPSPlanOut(
            id=p.id,
            name=p.name,
            instance_type=p.instance_type,
            vcpu=p.vcpu,
            ram_gb=p.ram_gb,
            storage_gb=p.storage_gb,
            price_cents=p.price_cents,
            price_display=f"${p.price_cents // 100}/mo",
            provider=p.provider or "aws",
        )
        for p in plans
    ]


@router.post("/checkout", response_model=VPSCheckoutResponse)
async def create_vps_checkout(
    body: VPSCheckoutRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Stripe Checkout Session for the selected plan.
    A pending VPSInstance record is written immediately so the webhook
    can locate it via stripe_session_id.
    """
    if not settings.vps_provisioning_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VPS provisioning is not yet enabled. Contact support.",
        )

    # Validate plan
    result = await db.execute(select(VPSPlan).where(VPSPlan.id == body.plan_id))
    plan: VPSPlan | None = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {body.plan_id}")

    # Check if user already has an active/pending VPS
    existing = await db.execute(
        select(VPSInstance)
        .where(VPSInstance.user_id == current_user.id)
        .where(VPSInstance.status.in_(["pending", "provisioning", "active"]))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="You already have an active or pending VPS instance.",
        )

    # Create pending VPSInstance (stripe_session_id filled after checkout creation)
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
    await db.commit()

    # Build URLs — Stripe appends {CHECKOUT_SESSION_ID} automatically
    base_url = "https://toup.ai"
    success_url = f"{base_url}/signup?step=4&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{base_url}/signup?step=2"

    try:
        session = create_checkout_session(
            user_id=current_user.id,
            user_email=current_user.email,
            plan_id=plan.id,
            vps_instance_id=vps_id,
            success_url=success_url,
            cancel_url=cancel_url,
        )
    except Exception as exc:
        # Roll back the pending VPS record so user can retry
        await db.delete(vps)
        await db.commit()
        logger.exception("Stripe checkout creation failed: %s", exc)
        raise HTTPException(status_code=502, detail="Payment service unavailable")

    # Persist the session ID so the webhook can look up this VPSInstance
    vps.stripe_session_id = session["id"]
    await db.commit()

    return VPSCheckoutResponse(checkout_url=session["url"], session_id=session["id"])


@router.get("/status", response_model=VPSStatusResponse)
async def get_vps_status(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's most recent VPS instance status."""
    result = await db.execute(
        select(VPSInstance)
        .where(VPSInstance.user_id == current_user.id)
        .order_by(VPSInstance.created_at.desc())
    )
    vps: VPSInstance | None = result.scalars().first()

    if not vps:
        return VPSStatusResponse(status="none")

    plan_result = await db.execute(select(VPSPlan).where(VPSPlan.id == vps.plan_id))
    plan = plan_result.scalar_one_or_none()

    return VPSStatusResponse(
        status=vps.status,
        plan_id=vps.plan_id,
        plan_name=plan.name if plan else None,
        provider=vps.provider or "aws",
        public_ip=vps.public_ip,
        public_dns=vps.public_dns,
        # Only expose SSH password once status is active
        ssh_password=vps.ssh_password if vps.status == "active" else None,
        error_message=vps.error_message,
        provisioned_at=vps.provisioned_at.isoformat() if vps.provisioned_at else None,
    )


@router.get("/agent-url", response_model=AgentURLResponse)
async def get_agent_url(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the user's Agent connection URLs.

    Used by the chat frontend to establish WebSocket connections
    directly to the user's Agent service.

    Checks VPSInstance first (cloud VPS), then falls back to
    AgentConfig.agent_url (local/self-hosted agents).
    """
    # 1. Check VPSInstance (cloud VPS)
    result = await db.execute(
        select(VPSInstance)
        .where(VPSInstance.user_id == current_user.id)
        .where(VPSInstance.status == "active")
        .order_by(VPSInstance.created_at.desc())
    )
    vps: VPSInstance | None = result.scalars().first()

    if vps and vps.public_ip:
        base = f"http://{vps.public_ip}:8001"
        return AgentURLResponse(
            agent_url=base,
            agent_ws_url=f"ws://{vps.public_ip}:8001{settings.api_prefix}/ws/chat",
            agent_api_key=vps.agent_api_key,
            status="active",
        )

    # 2. Fallback: check AgentConfig.agent_url (local/self-hosted)
    from app.db import AgentConfig
    cfg_result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == current_user.id)
    )
    cfg = cfg_result.scalar_one_or_none()

    if cfg and cfg.agent_url:
        host = cfg.agent_url.rstrip("/")
        ws_host = host.replace("http://", "ws://").replace("https://", "wss://")
        return AgentURLResponse(
            agent_url=host,
            agent_ws_url=f"{ws_host}{settings.api_prefix}/ws/chat",
            agent_api_key=cfg.agent_api_key,
            status="active",
        )

    return AgentURLResponse(status="none")


@router.post("/webhook/stripe", status_code=200)
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Stripe webhook endpoint — handles the full subscription lifecycle.

    Events handled:
        checkout.session.completed      — initial purchase (VPS, bundle, supabase)
        customer.subscription.updated   — plan change, cancellation scheduled, status change
        customer.subscription.deleted   — subscription ended
        invoice.payment_succeeded       — renewal payment
        invoice.payment_failed          — payment failure (card declined, etc.)
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = verify_webhook(payload, sig_header)
    except ValueError as exc:
        logger.error("Webhook config error: %s", exc)
        raise HTTPException(status_code=500, detail="Webhook verification not configured")
    if event is None:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event.get("type", "")
    event_id = event.get("id", "")
    obj = event.get("data", {}).get("object", {})

    logger.info(
        "stripe_webhook event_type=%s event_id=%s object_id=%s",
        event_type, event_id, obj.get("id", ""),
    )

    # ── checkout.session.completed ────────────────────────────────
    if event_type == "checkout.session.completed":
        await _handle_checkout_completed(obj, background_tasks, db)

    # ── customer.subscription.updated ─────────────────────────────
    elif event_type == "customer.subscription.updated":
        await _handle_subscription_updated(obj, db)

    # ── customer.subscription.deleted ─────────────────────────────
    elif event_type == "customer.subscription.deleted":
        await _handle_subscription_deleted(obj, db)

    # ── invoice.payment_succeeded ─────────────────────────────────
    elif event_type == "invoice.payment_succeeded":
        await _handle_invoice_succeeded(obj, db)

    # ── invoice.payment_failed ────────────────────────────────────
    elif event_type == "invoice.payment_failed":
        await _handle_invoice_failed(obj, db)

    else:
        logger.debug("Unhandled Stripe event type: %s", event_type)

    return {"received": True}


# ── Webhook handler helpers ──────────────────────────────────────────


async def _handle_checkout_completed(
    session_data: dict,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
):
    """Process a completed checkout session — activate subscriptions, provision VPS."""
    from app.db import AgentConfig, User
    from app.services.stripe_service import get_subscription

    stripe_session_id = session_data.get("id")
    subscription_id = session_data.get("subscription")
    metadata = session_data.get("metadata", {})
    user_id = metadata.get("user_id", "")
    vps_instance_id = metadata.get("vps_instance_id")
    stripe_customer_id = session_data.get("customer")

    # ── Backfill stripe_customer_id on User ───────────────────
    if user_id and stripe_customer_id:
        user_result = await db.execute(select(User).where(User.id == user_id))
        user_obj = user_result.scalar_one_or_none()
        if user_obj and not user_obj.stripe_customer_id:
            user_obj.stripe_customer_id = stripe_customer_id
            await db.commit()

    # ── Fetch subscription details for period_end ─────────────
    period_end = None
    if subscription_id:
        sub_info = get_subscription(subscription_id)
        if sub_info:
            period_end = sub_info["current_period_end"]

    # ── 1. Handle VPS provisioning ────────────────────────────
    if vps_instance_id and stripe_session_id:
        result = await db.execute(
            select(VPSInstance).where(VPSInstance.stripe_session_id == stripe_session_id)
        )
        vps_obj: VPSInstance | None = result.scalar_one_or_none()

        if vps_obj and vps_obj.status == "pending":
            vps_obj.status = "provisioning"
            if subscription_id:
                vps_obj.stripe_subscription_id = subscription_id
            await db.commit()

            from app.db.database import async_session_maker
            provider = vps_obj.provider or "aws"
            provisioners = {
                "aws": aws_provision_instance,
                "hostinger": hostinger_provision_instance,
                "hetzner": hetzner_provision_instance,
            }
            provisioner = provisioners.get(provider, aws_provision_instance)
            background_tasks.add_task(provisioner, vps_obj.id, async_session_maker)
            logger.info(
                "Provisioning queued for VPS %s (provider=%s, user=%s)",
                vps_obj.id, provider, vps_obj.user_id,
            )

    # ── 2. Handle LLM Bundle activation ──────────────────────
    if metadata.get("llm_bundle") == "true" and user_id:
        cfg_result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        agent_config = cfg_result.scalar_one_or_none()
        if agent_config:
            agent_config.bundle_status = "active"
            agent_config.bundle_stripe_subscription_id = subscription_id or ""
            agent_config.bundle_started_at = datetime.utcnow()
            agent_config.bundle_current_period_end = period_end
            agent_config.llm_mode = "bundle"
            await db.commit()
            logger.info("LLM Bundle activated for user %s (period_end=%s)", user_id, period_end)

    # ── 3. Handle Managed Supabase activation ─────────────────
    if metadata.get("managed_supabase") == "true" and user_id:
        cfg_result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        agent_config = cfg_result.scalar_one_or_none()
        if agent_config:
            agent_config.db_mode = "managed"
            await db.commit()
            logger.info("Managed Supabase activated for user %s", user_id)


async def _handle_subscription_updated(sub_data: dict, db: AsyncSession):
    """
    Sync subscription status changes — covers plan changes,
    cancellation scheduling (cancel_at_period_end), and status transitions.
    """
    from app.db import AgentConfig

    sub_id = sub_data.get("id", "")
    sub_status = sub_data.get("status", "")  # active, past_due, canceled, etc.
    cancel_at_period_end = sub_data.get("cancel_at_period_end", False)
    metadata = sub_data.get("metadata", {})
    user_id = metadata.get("user_id", "")

    period_end_ts = sub_data.get("current_period_end")
    period_end = (
        datetime.utcfromtimestamp(period_end_ts) if period_end_ts else None
    )

    # ── LLM Bundle ────────────────────────────────────────────
    if metadata.get("type") == "llm_bundle" and user_id:
        cfg_result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        config = cfg_result.scalar_one_or_none()
        if config and config.bundle_stripe_subscription_id == sub_id:
            if sub_status == "active" and not cancel_at_period_end:
                config.bundle_status = "active"
            elif sub_status == "active" and cancel_at_period_end:
                # User scheduled cancellation — keep active until period end
                config.bundle_status = "cancelling"
            elif sub_status == "past_due":
                config.bundle_status = "past_due"
            elif sub_status in ("canceled", "unpaid"):
                config.bundle_status = "cancelled"
            if period_end:
                config.bundle_current_period_end = period_end
            await db.commit()
            logger.info(
                "Subscription updated for user %s: status=%s cancel_at_period_end=%s",
                user_id, sub_status, cancel_at_period_end,
            )
        return

    # ── VPS subscription ──────────────────────────────────────
    vps_result = await db.execute(
        select(VPSInstance).where(VPSInstance.stripe_subscription_id == sub_id)
    )
    vps_obj = vps_result.scalar_one_or_none()
    if vps_obj:
        if sub_status in ("canceled", "unpaid"):
            vps_obj.status = "terminated"
            vps_obj.terminated_at = datetime.utcnow()
        logger.info(
            "VPS subscription updated for instance %s: status=%s",
            vps_obj.id, sub_status,
        )
        await db.commit()


async def _handle_subscription_deleted(sub_data: dict, db: AsyncSession):
    """
    Subscription fully ended — revoke access.
    Stripe sends this when the subscription period ends after cancellation,
    or on immediate cancellation.
    """
    from app.db import AgentConfig

    sub_id = sub_data.get("id", "")
    metadata = sub_data.get("metadata", {})
    user_id = metadata.get("user_id", "")

    # ── LLM Bundle ────────────────────────────────────────────
    if metadata.get("type") == "llm_bundle" and user_id:
        cfg_result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        config = cfg_result.scalar_one_or_none()
        if config and config.bundle_stripe_subscription_id == sub_id:
            config.bundle_status = "cancelled"
            config.llm_mode = "manual"
            await db.commit()
            logger.info("LLM Bundle cancelled for user %s (subscription deleted)", user_id)
        return

    # ── VPS subscription ──────────────────────────────────────
    vps_result = await db.execute(
        select(VPSInstance).where(VPSInstance.stripe_subscription_id == sub_id)
    )
    vps_obj = vps_result.scalar_one_or_none()
    if vps_obj and vps_obj.status != "terminated":
        vps_obj.status = "terminated"
        vps_obj.terminated_at = datetime.utcnow()
        await db.commit()
        logger.info("VPS %s terminated (subscription %s deleted)", vps_obj.id, sub_id)


async def _handle_invoice_succeeded(invoice_data: dict, db: AsyncSession):
    """
    Renewal payment succeeded — extend the billing period.
    Only acts on subscription invoices (not one-time).
    """
    from app.db import AgentConfig
    from app.services.stripe_service import get_subscription

    sub_id = invoice_data.get("subscription")
    if not sub_id:
        return  # Not a subscription invoice

    # Look up subscription to get metadata and period_end
    sub_info = get_subscription(sub_id)
    if not sub_info:
        return

    period_end = sub_info["current_period_end"]

    # ── Find AgentConfig by subscription ID ───────────────────
    cfg_result = await db.execute(
        select(AgentConfig).where(AgentConfig.bundle_stripe_subscription_id == sub_id)
    )
    config = cfg_result.scalar_one_or_none()
    if config:
        config.bundle_status = "active"
        config.bundle_current_period_end = period_end
        await db.commit()
        logger.info(
            "Bundle renewed for user %s — period_end=%s",
            config.user_id, period_end,
        )
        return

    # ── VPS — just log, no period tracking needed ─────────────
    vps_result = await db.execute(
        select(VPSInstance).where(VPSInstance.stripe_subscription_id == sub_id)
    )
    vps_obj = vps_result.scalar_one_or_none()
    if vps_obj:
        logger.info("VPS %s renewal payment succeeded", vps_obj.id)


async def _handle_invoice_failed(invoice_data: dict, db: AsyncSession):
    """
    Payment failed — mark subscription as past_due.
    Stripe will retry based on your Smart Retries settings.
    """
    from app.db import AgentConfig

    sub_id = invoice_data.get("subscription")
    if not sub_id:
        return

    # ── LLM Bundle ────────────────────────────────────────────
    cfg_result = await db.execute(
        select(AgentConfig).where(AgentConfig.bundle_stripe_subscription_id == sub_id)
    )
    config = cfg_result.scalar_one_or_none()
    if config:
        config.bundle_status = "past_due"
        await db.commit()
        logger.warning(
            "Payment failed for user %s — bundle marked past_due (subscription %s)",
            config.user_id, sub_id,
        )
        return

    # ── VPS ───────────────────────────────────────────────────
    vps_result = await db.execute(
        select(VPSInstance).where(VPSInstance.stripe_subscription_id == sub_id)
    )
    vps_obj = vps_result.scalar_one_or_none()
    if vps_obj:
        logger.warning(
            "Payment failed for VPS %s (subscription %s)",
            vps_obj.id, sub_id,
        )
