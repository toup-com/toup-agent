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
    Stripe webhook endpoint.
    Verifies signature and triggers EC2 provisioning on checkout.session.completed.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    event = verify_webhook(payload, sig_header)
    if event is None:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    if event.get("type") == "checkout.session.completed":
        session_data = event.get("data", {}).get("object", {})
        stripe_session_id = session_data.get("id")
        subscription_id = session_data.get("subscription")
        metadata = session_data.get("metadata", {})
        user_id = metadata.get("user_id", "")
        vps_instance_id = metadata.get("vps_instance_id")

        # ── 1. Handle VPS provisioning ─────────────────────────
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
                logger.info("Provisioning queued for VPS %s (provider=%s, user=%s)", vps_obj.id, provider, vps_obj.user_id)

        # ── 2. Handle LLM Bundle activation ────────────────────
        if metadata.get("llm_bundle") == "true" and user_id:
            from app.db import AgentConfig
            cfg_result = await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )
            agent_config = cfg_result.scalar_one_or_none()
            if agent_config:
                agent_config.bundle_status = "active"
                agent_config.bundle_stripe_subscription_id = subscription_id or ""
                agent_config.bundle_started_at = datetime.utcnow()
                agent_config.llm_mode = "bundle"
                await db.commit()
                logger.info("LLM Bundle activated for user %s", user_id)

        # ── 3. Handle Managed Supabase activation ──────────────
        if metadata.get("managed_supabase") == "true" and user_id:
            from app.db import AgentConfig
            cfg_result = await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )
            agent_config = cfg_result.scalar_one_or_none()
            if agent_config:
                agent_config.db_mode = "managed"
                await db.commit()
                logger.info("Managed Supabase activated for user %s", user_id)

    return {"received": True}
