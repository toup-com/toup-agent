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

from fastapi import APIRouter, Depends, HTTPException, Request
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
    # ISO 4217 currency code (lowercase, e.g. "usd", "cad"). Sourced
    # from the Stripe Price object so the frontend can render the right
    # symbol instead of guessing from the user's browser locale.
    currency: str = "usd"


class PricesResponse(BaseModel):
    llm_bundle: Optional[PriceItem] = None
    vps_plans: list[PriceItem] = []


# ── Helpers ──────────────────────────────────────────────────────────


def _country_from_request(request: Request) -> str:
    """Resolve the request origin's ISO 3166-1 alpha-2 country code.

    Source of truth (in order):
      1. ``X-Country`` header — explicit override. Cloudflare doesn't
         strip this so it works for staging / curl smoke tests, and
         normal browsers don't send it so prod traffic ignores it.
      2. Cloudflare's ``CF-IPCountry`` header — set on every request
         when toup.ai is proxied through Cloudflare. Authoritative
         for real users.
      3. Empty string (caller treats as "default" → USD).

    Returned uppercase. Returns "" rather than None so callers can use
    cheap string comparisons. Safety: the price difference is roughly
    currency-equivalent (USD \$30 ≈ CAD \$40), so even if someone
    spoofed X-Country they'd land on a comparably-priced sub.
    """
    override = (request.headers.get("X-Country") or "").strip().upper()
    if override:
        return override
    cf = (request.headers.get("CF-IPCountry") or "").strip().upper()
    if cf and cf not in ("XX", "T1"):  # XX/T1 = Tor / unknown — don't trust
        return cf
    return ""


def _bundle_price_id_for_country(country: str) -> str:
    """Pick the LLM bundle Price ID that matches the user's country.

    Today we have a CAD price for Canadian users; everyone else gets
    the USD default. Adding more regions is a one-line change here +
    a new env var.
    """
    if country == "CA" and settings.stripe_llm_bundle_price_id_cad:
        return settings.stripe_llm_bundle_price_id_cad
    return settings.stripe_llm_bundle_price_id


def _resolve_user_country(user: User, request: Optional[Request]) -> str:
    """Two-tier country lookup for an authenticated user.

    1. Stripe Customer's billing address country, when set on Stripe's
       side. Most authoritative — the user told us where they are when
       they entered payment details.
    2. The request's CF-IPCountry header. Fallback for users who
       haven't paid yet (still in onboarding LLM step).

    Returns "" if neither is known. The price resolver treats "" as
    "use the USD default."
    """
    if user.stripe_customer_id:
        try:
            from app.services.stripe_service import _get_stripe_client
            client = _get_stripe_client()
            cust = client.customers.retrieve(user.stripe_customer_id)
            addr = getattr(cust, "address", None)
            if addr and getattr(addr, "country", None):
                return str(addr.country).upper()
        except Exception as exc:
            logger.warning(
                "Stripe customer country lookup failed for %s: %s",
                user.id, exc,
            )
    if request is not None:
        return _country_from_request(request)
    return ""


async def _sync_active_subscription_to_db(
    db: AsyncSession, config: AgentConfig, subscription_id: str
) -> None:
    """Mirror what `_handle_invoice_succeeded` does on first-time activation,
    for the case where `create-subscription` discovered an existing active
    Stripe subscription via customer-by-email match (e.g., a re-signup with
    a previously-paying email, or any path where the webhook didn't fire
    against this AgentConfig row).

    Sets bundle_status='active', bundle_stripe_subscription_id, llm_mode,
    bundle_started_at, period_{start,end}, and mints llm_token_hash if
    none exists yet. Idempotent: if bundle_status is already 'active', does
    nothing (caller's webhook is the source of truth for ongoing renewals).
    """
    from app.services.stripe_service import get_subscription
    import secrets, hashlib

    if config.bundle_status == "active":
        return  # Webhook already synced; don't clobber period_end with stale data.

    sub_info = get_subscription(subscription_id)
    period_end = sub_info["current_period_end"] if sub_info else None
    now = datetime.utcnow()

    config.bundle_stripe_subscription_id = subscription_id
    config.bundle_status = "active"
    config.bundle_cancelled_at = None  # already-active sync also clears the reaper marker
    config.llm_mode = "bundle"
    config.bundle_current_period_end = period_end
    if not config.bundle_started_at:
        config.bundle_started_at = now
    config.bundle_period_start = now
    if period_end:
        config.bundle_period_end = period_end
    # Hash the agent's connect_token (= TOUP_TOKEN env in the container) so
    # the LLM proxy can authenticate the agent's incoming Bearer header.
    # Generating a fresh random secret here would never reach the agent.
    if not config.connect_token:
        config.connect_token = f"toup_ct_{secrets.token_urlsafe(32)}"
    # ALWAYS recompute the hash — sha256(token) is idempotent and cheap.
    # The previous `if not config.llm_token_hash` guard let stale hashes
    # from earlier BYOK setups survive bundle activation, causing 401s
    # at the proxy because hash(connect_token) != stored hash.
    # Root cause of the Nariman incident (2026-04-28).
    new_hash = hashlib.sha256(config.connect_token.encode()).hexdigest()
    if config.llm_token_hash != new_hash:
        config.llm_token_hash = new_hash
        logger.info(
            "Refreshed llm_token_hash from connect_token for user %s via already_active sync",
            config.user_id,
        )
    # Per-user OpenAI project auto-provisioning (β architecture). Idempotent:
    # only fires if no project_id is set yet. Failures are logged + non-fatal
    # so a transient OpenAI Admin API outage doesn't block bundle activation.
    _provision_openai_project_if_needed(config)
    await db.commit()
    logger.info(
        "Synced already-active Stripe sub %s to platform DB for user %s",
        subscription_id, config.user_id,
    )
    # Refresh the running tenant container's env so the bundle activation
    # (llm_mode + connect_token + hash) actually reaches the agent. The
    # container was provisioned during onboarding before Stripe finished,
    # so its env still has llm_mode=manual + empty TOUP_TOKEN. Without this,
    # the agent stays in BYOK fallback and every chat returns "API key
    # invalid". (Arshia incident, 2026-04-28.) Failures are non-fatal —
    # the DB-level activation already succeeded.
    try:
        from app.services.docker_host_service import update_container_env
        refreshed = await update_container_env(db, str(config.user_id), config)
        if refreshed:
            logger.info(
                "Refreshed tenant container env after already-active sync for user %s",
                config.user_id,
            )
    except Exception as _e:
        logger.warning(
            "Container env refresh failed after already-active sync for user %s: %s",
            config.user_id, _e,
        )


def _provision_openai_project_if_needed(
    config: AgentConfig,
    user_name: Optional[str] = None,
) -> None:
    """Idempotent: if config has no bundle_openai_project_id yet, call the
    OpenAI Admin API to create a per-user project + service-account key, and
    write both onto the AgentConfig (caller commits).

    `user_name`, when passed, is interpolated into the OpenAI project name
    so the operator's dashboard shows `toup-<NAME>-<prefix>` instead of an
    opaque `toup-tenant-<prefix>`. Callers that have the User row handy
    should pass it; callers that don't (or are running before the User
    row is fully hydrated) can omit it and accept the legacy naming.

    Failures are logged + swallowed — bundle activation must NOT fail because
    of a transient OpenAI outage. The proxy's outbound logic falls back to
    the platform master OpenAI key in that case (defense-in-depth).
    """
    if config.bundle_openai_project_id and config.bundle_openai_api_key:
        return  # already provisioned
    try:
        from app.services.openai_admin_service import (
            provision_tenant, OpenAIAdminUnavailable,
        )
        prefix = (config.user_id or "")[:8] or "unknown"
        project_id, api_key = provision_tenant(prefix, user_name=user_name)
        config.bundle_openai_project_id = project_id
        config.bundle_openai_api_key = api_key
        logger.info(
            "Provisioned OpenAI project %s for user %s",
            project_id, config.user_id,
        )
    except OpenAIAdminUnavailable as e:
        logger.warning(
            "OpenAI admin not configured; bundle for user %s falls back to "
            "master key: %s",
            config.user_id, e,
        )
    except Exception as e:
        logger.warning(
            "OpenAI auto-provisioning failed for user %s — proxy will fall "
            "back to master key. Error: %s",
            config.user_id, e,
        )


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
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Stripe Subscription with payment_behavior='default_incomplete'.
    Returns the client_secret for the frontend PaymentElement.

    Idempotent: if the user already has an active bundle subscription,
    returns it without creating a new one.

    Country-aware pricing: looks up the user's country (Stripe billing
    address > CF-IPCountry header) and picks the matching Price ID.
    Canadian users charge in CAD; everyone else USD.
    """
    from app.services.stripe_service import (
        create_subscription_with_intent,
        get_active_subscription_for_customer,
    )

    # Get user + ensure Stripe customer
    user_result = await db.execute(select(User).where(User.id == current_user.id))
    user = user_result.scalar_one()
    customer_id = await _ensure_stripe_customer(user, db)

    country = _resolve_user_country(user, request)
    price_id = _bundle_price_id_for_country(country)
    if not price_id:
        raise HTTPException(400, "LLM Bundle pricing is not configured.")

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
            # Stripe says active; sync platform DB to match BEFORE returning
            # already_active. Without this, the frontend's PaymentForm fires
            # onSuccess() → triggers /managed-agent/provision, but the LLM
            # proxy's bundle_status gate denies the user (they have an agent
            # but it can't reach the LLM). The 2026-04-27 incident: fresh
            # signup re-uses orphan Stripe customer's still-active sub via
            # email match → no payment fires → no webhook fires → bundle_
            # status stays 'none' forever. Mirror what _handle_invoice_
            # succeeded does on first activation.
            if config:
                await _sync_active_subscription_to_db(
                    db, config, existing["subscription_id"]
                )
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
async def get_prices(request: Request, db: AsyncSession = Depends(get_db)):
    """Return configured pricing from Stripe.

    Picks the right LLM bundle price for the request's country —
    Canadian users get the CAD price (~$40 CAD), everyone else gets
    USD ($30). Country comes from Cloudflare's CF-IPCountry header
    (toup.ai is proxied through CF). The currency in the response
    matches the chosen Price's actual Stripe currency, so the frontend
    renders the right symbol without guessing.
    """
    from app.services.stripe_service import get_price
    from app.db import VPSPlan

    result = PricesResponse()

    country = _country_from_request(request)
    bundle_price_id = _bundle_price_id_for_country(country)
    if bundle_price_id:
        price_info = get_price(bundle_price_id)
        if price_info:
            result.llm_bundle = PriceItem(
                id="llm_bundle",
                name=price_info["product_name"] or "Toup LLM Bundle",
                amount_cents=price_info["unit_amount"],
                interval=price_info["interval"] or "month",
                currency=(price_info.get("currency") or "usd").lower(),
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
