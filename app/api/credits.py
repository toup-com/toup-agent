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

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import (
    LEDGER_TOOL_CALL,
    AgentConfig, CreditLedger, PlatformSetting, SubscriptionPlan, User,
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
    # Null means "this bucket has NO daily dimension", not "we failed to
    # compute it". Only the MESSAGE bucket has per-day counters
    # (credit_balances.message_credits_used_today / _daily_cap); there is no
    # integration equivalent in the schema, so integration reports null here
    # deliberately. `has_daily_limit` lets a client tell the two apart instead
    # of inferring from a null it cannot interpret.
    used_today: Optional[float] = None
    daily_cap: Optional[float] = None
    has_daily_limit: bool = False
    # UTC instant the daily counter next rolls; null when the bucket has no
    # daily dimension.
    #
    # This field is why the mobile Usage screen's "Daily limit" meter has never
    # rendered for anybody. It has consumed `daily_reset_at` since it was
    # written and gated the whole row on the value being present — and no
    # deploy has ever sent one, so the row was not degraded, it was absent.
    # Free is the ONLY plan carrying a daily cap, so the single tier the meter
    # exists for is exactly the one that could never see it, which is also why
    # the gap survived every test on a paid account.
    #
    # It must come from the server: the counter rolls at the boundary of
    # User.timezone, NULL for most accounts, in which case the backend rolls at
    # UTC midnight — 8pm in Toronto. A device using its own midnight is wrong by
    # the whole UTC offset, in the direction that promises capacity the user
    # does not have yet.
    daily_reset_at: Optional[datetime] = None
    # Credits consumed this PERIOD = monthly - remaining, computed server-side.
    # The endpoint used to return no "used" figure at all, so every client
    # re-derived it — and the two mobile screens derived it differently (one
    # subtracted the purchased wallet first, one did not) and then rounded it
    # differently. One number, one place.
    used_this_period: float = 0.0


class CreditStatusResponse(BaseModel):
    plan_id: str
    plan_display_name: str
    message: BucketStatus
    integration: BucketStatus
    period_start: datetime
    period_end: datetime
    enforcement_enabled: bool
    # Non-expiring credits bought via StoreKit IAP. message.remaining already
    # INCLUDES this (it's the spendable total); this field is the purchased
    # slice on its own so the client can render "X bought" separately.
    purchased_credits_remaining: float = 0.0
    # Admins have no usage limits — never charged, never gated (see
    # credit_service._is_unlimited_user). The frontend uses this to suppress
    # the low-balance pill / exhausted card and render "Unlimited".
    unlimited: bool = False
    # Subscription source for the mobile billing UI:
    #   'iap'  → Apple auto-renewable sub (manage in iOS Settings)
    #   'web'  → Stripe sub or legacy paid (manage at toup.ai/account)
    #   'free' → no paid plan
    # All additive/optional so older clients ignore them.
    plan_source: Optional[str] = None
    subscription_product_id: Optional[str] = None
    subscription_renews_at: Optional[datetime] = None


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
    from app.services.credit_service import _is_unlimited_user
    return CreditStatusResponse(
        unlimited=_is_unlimited_user(current_user),
        plan_id=view.plan_id,
        plan_display_name=view.plan_display_name,
        message=BucketStatus(
            remaining=float(view.message_credits_remaining),
            monthly=float(view.message_credits_monthly),
            used_today=float(view.message_credits_used_today),
            daily_cap=(float(view.message_credits_daily_cap)
                       if view.message_credits_daily_cap is not None else None),
            has_daily_limit=view.message_credits_daily_cap is not None,
            daily_reset_at=view.daily_reset_at,
            # `remaining` is plan + purchased, so subtracting it from the plan's
            # monthly allotment would under-report usage for anyone holding IAP
            # credits. Bought credits are not part of the period allowance.
            used_this_period=max(0.0, float(
                view.message_credits_monthly
                - (view.message_credits_remaining - view.purchased_credits_remaining)
            )),
        ),
        integration=BucketStatus(
            remaining=float(view.integration_credits_remaining),
            monthly=float(view.integration_credits_monthly),
            # No per-day integration counter exists in the schema — null here
            # is the schema's answer, not a missing computation.
            has_daily_limit=False,
            used_this_period=max(0.0, float(
                view.integration_credits_monthly - view.integration_credits_remaining
            )),
        ),
        period_start=view.period_start,
        period_end=view.period_end,
        enforcement_enabled=view.enforcement_enabled,
        purchased_credits_remaining=float(view.purchased_credits_remaining),
        plan_source=view.plan_source,
        subscription_product_id=view.subscription_product_id,
        subscription_renews_at=view.subscription_renews_at,
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

    # White-label: alias the real model id + provider name before they leave the
    # API — the ledger feeds the mobile Credits screen (docs/security/audit-2026.md
    # MI-2, re-audit). Flag-gated; token/amount fields are untouched.
    from app.config import settings as _settings
    _leak = _settings.security_leak_filter
    if _leak:
        from app.services.model_alias import public_model_label, public_provider_label

    def _m(v):
        return public_model_label(v) if (_leak and v) else v

    def _p(v):
        return public_provider_label(v) if (_leak and v) else v

    return LedgerResponse(rows=[
        LedgerRow(
            id=r.id, event_type=r.event_type, bucket=r.bucket,
            amount=float(r.amount), balance_after=float(r.balance_after),
            event_id=r.event_id, model=_m(r.model), provider=_p(r.provider),
            created_at=r.created_at,
        ) for r in rows
    ])


# ── Agent → platform credit deduction (manual mode metering) ──────
#
# Bundle-mode LLM calls flow through /api/llm/chat where the
# _log_event hook already deducts. Manual-mode users (own API key)
# bypass the proxy entirely — their agent talks direct to
# Anthropic/OpenAI and the platform never learns of the call.
#
# This endpoint closes that gap: the agent's anthropic_service /
# openai_agent_service POSTs a deduction record here AFTER every
# successful LLM call. The platform validates the agent_api_key,
# looks up the user, and routes through credit_service.try_charge
# with the same idempotency + enforcement semantics as the proxy
# hook. Manual-mode users now get the same metering + enforcement
# as bundle-mode users.


class AgentDeductRequest(BaseModel):
    """Posted by the tenant agent after every direct LLM call.

    `idempotency_key` should be the agent's own request_id (e.g.
    Anthropic message.id or OpenAI completion.id) so a retry from
    inside the agent doesn't double-charge.
    """
    user_id: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    underlying_cost_cents: Optional[float] = None
    operation_type: Optional[str] = None  # "user.*" or "system.*"
    idempotency_key: Optional[str] = None
    event_id: Optional[str] = None
    # Prompt-cache read hits (F-7 / A9-1). Telemetry only — stored in the
    # ledger row's metadata_json, never enters the charge amount. Optional
    # so older agent builds that omit the field stay compatible.
    cached_tokens: Optional[int] = None


class AgentDeductResponse(BaseModel):
    success: bool
    bucket: str
    amount_charged: float
    balance_after: float
    enforcement_enabled: bool
    reason: Optional[str] = None
    idempotent_hit: bool = False
    # Plan + period metadata — lets the agent's CreditReporter stamp
    # last_known_plan_id / last_known_period_end on every deduct cycle
    # so build_exhausted_response shows the real renewal date instead
    # of the "now + 30 days" fallback. Optional for backward compat
    # with older agent builds that don't read these fields.
    plan_id: Optional[str] = None
    plan_display_name: Optional[str] = None
    period_end: Optional[datetime] = None


@router.post("/agent-deduct", response_model=AgentDeductResponse)
async def agent_deduct(
    body: AgentDeductRequest,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
    db: AsyncSession = Depends(get_db),
) -> AgentDeductResponse:
    """Authenticated by X-Agent-Key. The agent's
    `agent_api_key` (stored in AgentConfig on the platform side and
    in the agent container's env) is the shared secret.

    Returns success=False when enforcement is on AND the user is
    out of credits — the agent SHOULD honor this by short-circuiting
    the next LLM call client-side. Until then, this endpoint is a
    metering surface; the deduction still gets recorded so admin
    /credits/ledger sees the true cost.
    """
    from app.services.credit_service import (
        tokens_to_credits as _tokens_to_credits,
        BUCKET_MESSAGE as _BUCKET_MESSAGE,
        REASON_INSUFFICIENT_MESSAGE as _REASON_INSUFFICIENT_MESSAGE,
    )
    from app.db.models import LEDGER_CHAT_MESSAGE as _LEDGER_CHAT_MESSAGE

    if not x_agent_key:
        raise HTTPException(401, "X-Agent-Key required")

    # Authenticate: agent_api_key must match the AgentConfig of the
    # claimed user_id. Stops one tenant's agent from charging another
    # tenant's credit balance even if it discovers their X-Agent-Key
    # (which it shouldn't — keys are per-tenant secrets).
    cfg_result = await db.execute(
        select(AgentConfig).where(
            AgentConfig.user_id == body.user_id,
            AgentConfig.agent_api_key == x_agent_key,
        )
    )
    cfg = cfg_result.scalar_one_or_none()
    if cfg is None:
        raise HTTPException(403, "agent key mismatch")

    # System ops aren't charged to the user. Match the proxy hook's
    # rule: "system.*" → platform overhead, exempt.
    op = body.operation_type or "user"
    if op.startswith("system."):
        return AgentDeductResponse(
            success=True, bucket=_BUCKET_MESSAGE, amount_charged=0.0,
            balance_after=0.0, enforcement_enabled=False,
            reason="system_op_exempt",
        )

    # Bundle-mode guard — DO NOT double-charge.
    #
    # This endpoint exists for MANUAL-mode agents that talk direct to
    # OpenAI/Anthropic and bypass the proxy (see the section header
    # above). A BUNDLE-mode agent routes every LLM call through
    # /api/llm/chat, where the proxy's _log_event hook already deducts
    # via credit_service.try_charge (idempotency_key=event.id). But the
    # agent's openai_agent_service/anthropic_service ALSO POST here after
    # every completion, with idempotency_key=prompt_cache_key — a
    # per-SESSION constant ("{user_id}:{session_id}"). That key never
    # collides with the proxy's per-event UUID, so the FIRST call of every
    # bundle session was charged TWICE (later calls idempotent-hit the
    # per-session key and were free). Measured: a 13–36% overcharge on
    # every bundle user; it drained a fresh free demo account in one
    # session. cfg.llm_mode is the platform's authoritative bundle signal
    # (free_tier_activation/billing set it). Skip the charge but return the
    # real balance/enforcement so the agent's CreditState stays accurate
    # for its pre-flight short-circuit — the deduction already happened at
    # the proxy. Idempotent-hit=True tells the agent "already counted".
    if (cfg.llm_mode or "").strip().lower() == "bundle":
        view = await credit_service.get_balance_view(db, body.user_id)
        # Commit any lazy balance creation get_balance_view may have done
        # (brand-new user with no row yet). No charge is ever written here.
        await db.commit()
        remaining = float(view.message_credits_remaining)
        enforcement = bool(view.enforcement_enabled)
        sufficient = (not enforcement) or remaining > 0
        return AgentDeductResponse(
            success=sufficient,
            bucket=_BUCKET_MESSAGE,
            amount_charged=0.0,
            balance_after=remaining,
            enforcement_enabled=enforcement,
            reason=None if sufficient else _REASON_INSUFFICIENT_MESSAGE,
            idempotent_hit=True,
            plan_id=view.plan_id,
            plan_display_name=view.plan_display_name,
            period_end=view.period_end,
        )

    credits = _tokens_to_credits(body.model, body.input_tokens, body.output_tokens)
    # cached_tokens rides in metadata_json only (F-7 / A9-1) — the charge
    # amount above is computed from input/output alone, unchanged.
    _deduct_meta: dict = {"surface": "agent_direct", "operation_type": op}
    if body.cached_tokens is not None:
        _deduct_meta["cached_tokens"] = int(body.cached_tokens)
    result = await credit_service.try_charge(
        db, body.user_id, _LEDGER_CHAT_MESSAGE, _BUCKET_MESSAGE, credits,
        idempotency_key=body.idempotency_key,
        event_id=body.event_id,
        model=body.model,
        provider=body.provider,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        underlying_cost_cents=body.underlying_cost_cents,
        metadata=_deduct_meta,
        # This endpoint is a REPORT, not a request: the agent already talked to
        # the provider on its own key and we are recording what that cost. The
        # daily cap must gate ADMISSION (check_balance), never zero a bill we
        # have already paid — that is the 2026-08-03 defect, and llm_proxy is
        # only ONE of the two deduct paths. Manual mode has no proxy in front
        # of it, so without this the same over-cap turn is served free here.
        already_incurred=True,
    )
    # Read the balance view AFTER try_charge committed so plan +
    # period_end reflect any lazy monthly-renewal that just landed.
    view = await credit_service.get_balance_view(db, body.user_id)

    # `success` is an ADMISSION verdict, not a receipt.
    #
    # The agent's only gate on this path is `credit_reporter.raise_if_exhausted`,
    # which blocks the NEXT call iff the last deduct came back success=False —
    # manual mode has no proxy in front of it and never calls check_balance. So
    # the two things this response has to do are in tension under the old code:
    # deny (to stop the loop) and charge (because the provider was already
    # paid). It denied, which stopped the loop and gave the turn away free.
    #
    # Now the charge always lands (already_incurred above) and admission is
    # asked separately, exactly as the proxy asks it. The agent keeps blocking
    # after the cap, and we keep the money. `amount_charged` reports what really
    # happened; it is deliberately no longer tied to `success`.
    from decimal import Decimal as _Dec
    admission = await credit_service.check_balance(
        db, body.user_id, _BUCKET_MESSAGE, _Dec("0.1"),
    )
    await db.commit()

    from app.config import settings as _settings
    return AgentDeductResponse(
        success=result.success and admission.success,
        bucket=_BUCKET_MESSAGE,
        amount_charged=float(credits) if result.success else 0.0,
        balance_after=float(result.balance_after),
        enforcement_enabled=bool(getattr(_settings, "credit_enforcement_enabled", False)),
        reason=result.reason or admission.reason,
        idempotent_hit=result.idempotent_hit,
        plan_id=view.plan_id,
        plan_display_name=view.plan_display_name,
        period_end=view.period_end,
    )


class AgentChargeRequest(BaseModel):
    """Posted by the tenant agent to charge an EXPLICIT per-event credit
    amount that is NOT derived from token counts.

    Used for tools priced per-event rather than per-token — image generation
    (gpt-image-1, priced per image by size/quality) and the web tools
    (web_search / web_fetch, flat-fee per call). ``credits`` is the
    already-converted amount to deduct; ``underlying_cost_cents`` is stored
    for audit. ``idempotency_key`` should uniquely identify the event (a
    fresh UUID per event is fine).

    ``meter_only=True`` records the ledger row WITHOUT touching the balance
    and always returns success — the landing mode for a newly-priced event.
    See ``CreditService.try_charge`` for the full contract.
    """
    user_id: str
    event_type: str = "image_generation"
    credits: float
    bucket: str = "message"           # "message" | "integration"
    provider: Optional[str] = None
    model: Optional[str] = None
    underlying_cost_cents: Optional[float] = None
    idempotency_key: Optional[str] = None
    metadata: Optional[dict] = None
    meter_only: bool = False


@router.post("/agent-charge", response_model=AgentDeductResponse)
async def agent_charge(
    body: AgentChargeRequest,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
    db: AsyncSession = Depends(get_db),
) -> AgentDeductResponse:
    """Charge an explicit per-event credit amount (image generation, etc.).

    Same auth + guards as /agent-deduct: X-Agent-Key must match the claimed
    user's AgentConfig; bundle-mode agents are NOT charged here (their cost is
    already deducted inline by the platform LLM proxy — see
    llm_proxy.proxy_openai_images), which prevents a double charge; admins are
    unlimited; shadow mode records the ledger row without denying.
    """
    from decimal import Decimal as _Decimal
    from app.services.credit_service import (
        BUCKET_MESSAGE as _BUCKET_MESSAGE,
        BUCKET_INTEGRATION as _BUCKET_INTEGRATION,
        REASON_INSUFFICIENT_MESSAGE as _REASON_INSUFFICIENT_MESSAGE,
    )

    if not x_agent_key:
        raise HTTPException(401, "X-Agent-Key required")

    cfg = (await db.execute(
        select(AgentConfig).where(
            AgentConfig.user_id == body.user_id,
            AgentConfig.agent_api_key == x_agent_key,
        )
    )).scalar_one_or_none()
    if cfg is None:
        raise HTTPException(403, "agent key mismatch")

    bucket = _BUCKET_INTEGRATION if body.bucket == "integration" else _BUCKET_MESSAGE

    # Bundle-mode guard — the proxy already charged this event inline, so skip
    # the agent-side charge but return the real balance/enforcement so the
    # agent's CreditState stays accurate. Mirrors the /agent-deduct guard.
    #
    # meter_only is EXEMPT, and the exemption is load-bearing. A metered row
    # carries amount=0 and never touches the balance, so it cannot produce the
    # double charge this guard exists to prevent. Without the exemption the
    # guard returns before writing anything, and web_search / web_fetch usage
    # is invisible for every bundle tenant — currently 100% of the fleet (42
    # of 42). Those are direct outbound calls from the agent to Brave; the LLM
    # proxy never sees them, so they were never charged inline in the first
    # place and there is nothing to double-count.
    #
    # But meter_only alone was the WRONG key. It is True today only because
    # `web_tool_metering_charge` is False; flipping that flag — the documented
    # way to start billing web tools — makes the agent send meter_only=False,
    # re-engages this guard, and the usage series DISAPPEARS for the whole
    # fleet instead of starting to bill. The flag advertised as "start
    # charging" would in fact have blinded the system.
    #
    # So key on the real question: did the LLM proxy already bill this event
    # inline? Only events that traverse the proxy can double-charge.
    #   tool_call        — agent → Brave directly, proxy never sees it. EXEMPT.
    #   image_generation — the proxy charges it at llm_proxy.py:1675/1804/
    #                      1899/1992/2087. NOT exempt; suppression is correct.
    # An allowlist, not a denylist: an unrecognised future event_type falls to
    # the suppressed side, which risks an under-charge rather than billing a
    # user twice.
    _PROXY_NEVER_BILLS = {LEDGER_TOOL_CALL}
    _bundle_exempt = body.meter_only or (body.event_type in _PROXY_NEVER_BILLS)
    if (cfg.llm_mode or "").strip().lower() == "bundle" and not _bundle_exempt:
        view = await credit_service.get_balance_view(db, body.user_id)
        await db.commit()
        remaining = float(view.message_credits_remaining)
        enforcement = bool(view.enforcement_enabled)
        sufficient = (not enforcement) or remaining > 0
        return AgentDeductResponse(
            success=sufficient, bucket=bucket, amount_charged=0.0,
            balance_after=remaining, enforcement_enabled=enforcement,
            reason=None if sufficient else _REASON_INSUFFICIENT_MESSAGE,
            idempotent_hit=True,
            plan_id=view.plan_id, plan_display_name=view.plan_display_name,
            period_end=view.period_end,
        )

    # Floor at 0.1 credit so try_charge (rejects amount<=0) never gets a zero.
    credits = max(_Decimal("0.1"), _Decimal(str(body.credits)))
    result = await credit_service.try_charge(
        db, body.user_id, body.event_type or "image_generation", bucket, credits,
        idempotency_key=body.idempotency_key,
        model=body.model,
        provider=body.provider,
        underlying_cost_cents=body.underlying_cost_cents,
        metadata={"surface": "agent_charge", **(body.metadata or {})},
        meter_only=body.meter_only,
    )
    view = await credit_service.get_balance_view(db, body.user_id)
    await db.commit()

    from app.config import settings as _settings
    return AgentDeductResponse(
        success=result.success,
        bucket=bucket,
        # meter_only never moves the balance, so it never "charged" anything —
        # reporting a non-zero amount here would double-count in any caller
        # that sums amount_charged.
        amount_charged=(
            0.0 if body.meter_only
            else (float(credits) if result.success else 0.0)
        ),
        balance_after=float(result.balance_after),
        enforcement_enabled=bool(getattr(_settings, "credit_enforcement_enabled", False)),
        reason=result.reason,
        idempotent_hit=result.idempotent_hit,
        plan_id=view.plan_id,
        plan_display_name=view.plan_display_name,
        period_end=view.period_end,
    )


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


def _resolve_checkout_redirect(
    base_url: str, *, candidate: Optional[str], default_path: str,
) -> str:
    """Pick a Stripe success/cancel URL, defending against open redirects.

    The frontend may pass a same-origin path (e.g.
    `/onboarding/install?upgraded=1&plan=builder`) — we accept any path
    that starts with `/`, prefix it with the platform's public base
    URL, and reject anything else by falling back to `default_path`.
    """
    if candidate and candidate.startswith("/"):
        return f"{base_url}{candidate}"
    return f"{base_url}{default_path}"


@billing_router.post("/credit-checkout/{plan_id}", response_model=CreditCheckoutResponse)
async def create_credit_checkout(
    plan_id: str,
    return_url: Optional[str] = None,
    cancel_url: Optional[str] = None,
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

    Optional `return_url` / `cancel_url` accept a same-origin PATH
    (e.g. `/onboarding/install?upgraded=1&plan=builder`) used by the
    onboarding-v2 LLM step. Non-path candidates are ignored — Stripe
    only ever sees a URL on the platform's public host.
    """
    from app.config import settings as _settings
    from app.api.billing import _ensure_stripe_customer
    from app.services.stripe_service import create_credit_checkout_session

    if plan_id == "free":
        raise HTTPException(400, "The free tier doesn't require checkout.")

    # Reconciliation (§3b): refuse a web subscription when the user already has
    # an active Apple sub. Gated at SESSION CREATION — before any Stripe charge,
    # so it's strictly safe (no money moved). Symmetric to the Apple-side 409.
    if await credit_service.active_paid_source(db, current_user.id) == "apple":
        raise HTTPException(409, detail={
            "code": "subscription_exists_other_platform",
            "message": "You already have a subscription via the iOS app. "
                       "Manage it in Settings → Subscriptions.",
        })

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
    success = _resolve_checkout_redirect(
        base_url, candidate=return_url,
        default_path=f"/account?upgrade=success&plan={plan_id}",
    )
    cancel = _resolve_checkout_redirect(
        base_url, candidate=cancel_url,
        default_path="/pricing?upgrade=cancelled",
    )
    try:
        session = create_credit_checkout_session(
            customer_id=customer_id,
            price_id=plan.stripe_price_id,
            plan_id=plan_id,
            user_id=current_user.id,
            success_url=success,
            cancel_url=cancel,
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


@admin_router.post("/backfill-openai-projects")
async def admin_backfill_openai_projects(
    limit: int = Query(200, ge=1, le=1000),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Provision a per-tenant OpenAI project for every active-bundle user
    that doesn't have one yet.

    On-demand twin of the platform-boot reconciler. Run this after an
    OpenAI Admin API outage to cure any users who were left on the shared
    master key, so every user ends up on their OWN isolated per-tenant key.
    Idempotent + safe to re-run.
    """
    from app.api.billing import backfill_missing_openai_projects
    return await backfill_missing_openai_projects(db, limit=limit)


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


@admin_router.get("/audit/{user_id}")
async def admin_credit_audit(
    user_id: str,
    limit: int = Query(50, le=500),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Per-user credit audit — answers "why isn't this user's balance changing?"

    Returns the user's:
      * Current balance + plan + period window
      * AgentConfig snapshot (agent_api_key present? when last seen?)
      * Last N ledger rows with timestamps + denial reasons surfaced
      * Last N LLMProxyEvent rows (proxy bundle-mode metering trail)
      * Computed gaps: ledger rows last 24h vs proxy events last 24h
        (mismatch → the proxy hook isn't firing OR the bundle/manual
        routing has drifted)

    This is the FIRST endpoint to hit when the operator sees a user
    showing 100/100 credits despite active chat usage. The output
    distinguishes the four common failure modes:

      A. AgentConfig.agent_api_key is empty → tenant agent boot env
         drift; the agent can't authenticate against /agent-deduct.
      B. ledger has 0 deductions but llm_proxy_events shows charges →
         the bundle-mode hook is broken OR enforcement_enabled was off
         when those events landed (shadow mode + early flag flip).
      C. ledger has 0 rows AND llm_proxy_events has 0 user.* rows →
         manual mode + agent code on this tenant predates credit_reporter
         wiring. Operator should restart the agent on a current image.
      D. Recent ledger rows show `metadata.denied=true` → enforcement IS
         working; the user really is out of credits, and the operator
         should look at the user-side UX (did the upgrade card render?).
    """
    from datetime import timedelta
    from sqlalchemy import func

    target = await db.get(User, user_id)
    if target is None:
        raise HTTPException(404, f"user {user_id!r} not found")

    view = await credit_service.get_balance_view(db, user_id)

    cfg_q = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    cfg = cfg_q.scalar_one_or_none()

    ledger_q = await db.execute(
        select(CreditLedger)
        .where(CreditLedger.user_id == user_id)
        .order_by(CreditLedger.created_at.desc())
        .limit(limit)
    )
    ledger_rows = ledger_q.scalars().all()

    # Cross-reference with the bundle-mode proxy log so we can spot the
    # bundle-vs-manual + enforcement-was-off failure modes.
    from app.db.models import LLMProxyEvent
    proxy_q = await db.execute(
        select(LLMProxyEvent)
        .where(LLMProxyEvent.user_id == user_id)
        .order_by(LLMProxyEvent.created_at.desc())
        .limit(limit)
    )
    proxy_rows = proxy_q.scalars().all()

    now = datetime.utcnow()
    day_ago = now - timedelta(hours=24)
    ledger_24h_count = sum(1 for r in ledger_rows if r.created_at >= day_ago)
    proxy_24h_count = sum(1 for r in proxy_rows if r.created_at >= day_ago)
    proxy_24h_user_count = sum(
        1 for r in proxy_rows
        if r.created_at >= day_ago and (not r.operation_type or not r.operation_type.startswith("system."))
    )
    ledger_24h_denials = sum(
        1 for r in ledger_rows
        if r.created_at >= day_ago
        and r.metadata_json
        and r.metadata_json.get("denied") is True
    )

    # Heuristic diagnosis — what's the most likely root cause?
    diagnosis: list[str] = []
    if cfg is None:
        diagnosis.append(
            "AgentConfig row missing — user never finished onboarding "
            "or the row was deleted; manual-mode metering can't authenticate."
        )
    elif not (cfg.agent_api_key or "").strip():
        diagnosis.append(
            "AgentConfig.agent_api_key is empty — the tenant agent can't "
            "authenticate against /credits/agent-deduct. Fix by re-running "
            "agent provisioning or rotating the key."
        )
    if ledger_24h_count == 0 and proxy_24h_user_count > 0:
        diagnosis.append(
            "0 ledger rows in last 24h but proxy_events shows user.* "
            "calls — the bundle-mode hook isn't writing to ledger. "
            "Check llm_proxy._log_event for the try_charge branch."
        )
    if ledger_24h_count == 0 and proxy_24h_user_count == 0 and proxy_24h_count > 0:
        diagnosis.append(
            "All proxy_events in last 24h are system.* (exempt). The user "
            "is on direct-API manual mode; deductions should arrive via "
            "/credits/agent-deduct. If the ledger is still empty, the agent "
            "container is running pre-credit_reporter code — redeploy."
        )
    if ledger_24h_count == 0 and proxy_24h_count == 0:
        diagnosis.append(
            "No LLM activity recorded for this user in 24h via any path. "
            "Either the user isn't chatting, or their agent's outbound "
            "calls are landing on a different platform_api_url (Railway "
            "env drift)."
        )
    if ledger_24h_denials > 0:
        diagnosis.append(
            f"{ledger_24h_denials} denial(s) in last 24h — enforcement IS "
            "working server-side. Verify the user is seeing the "
            "credit_exhausted card on the client (check ChatPage console)."
        )

    from app.config import settings as _settings
    return {
        "user": {
            "id": target.id,
            "email": target.email,
            "created_at": target.created_at.isoformat() if target.created_at else None,
            "email_verified_at": (
                target.email_verified_at.isoformat()
                if target.email_verified_at else None
            ),
        },
        "balance": {
            "plan_id": view.plan_id,
            "plan_display_name": view.plan_display_name,
            "message_credits_remaining": float(view.message_credits_remaining),
            "message_credits_monthly": float(view.message_credits_monthly),
            "message_credits_used_today": float(view.message_credits_used_today),
            "message_credits_daily_cap": (
                float(view.message_credits_daily_cap)
                if view.message_credits_daily_cap is not None else None
            ),
            "integration_credits_remaining": float(view.integration_credits_remaining),
            "integration_credits_monthly": float(view.integration_credits_monthly),
            "period_start": view.period_start.isoformat(),
            "period_end": view.period_end.isoformat(),
            "enforcement_enabled": view.enforcement_enabled,
        },
        "agent_config": {
            "exists": cfg is not None,
            "agent_api_key_set": bool(cfg and (cfg.agent_api_key or "").strip()),
            "agent_url": (cfg.agent_url if cfg else None),
        },
        "activity_24h": {
            "ledger_rows": ledger_24h_count,
            "ledger_denials": ledger_24h_denials,
            "proxy_events_total": proxy_24h_count,
            "proxy_events_user_attributable": proxy_24h_user_count,
        },
        "platform_state": {
            "credit_enforcement_enabled": bool(
                getattr(_settings, "credit_enforcement_enabled", False)
            ),
            "require_email_verification_for_credits": bool(
                getattr(_settings, "require_email_verification_for_credits", False)
            ),
        },
        "diagnosis": diagnosis or [
            "No anomaly detected — balance state appears coherent with activity."
        ],
        "recent_ledger": [
            {
                "id": r.id,
                "event_type": r.event_type,
                "bucket": r.bucket,
                "amount": float(r.amount),
                "balance_after": float(r.balance_after),
                "denied": bool(r.metadata_json and r.metadata_json.get("denied")),
                "reason": (r.metadata_json or {}).get("reason"),
                "model": r.model,
                "provider": r.provider,
                "created_at": r.created_at.isoformat(),
            } for r in ledger_rows
        ],
        "recent_proxy_events": [
            {
                "id": r.id,
                "provider": r.provider,
                "model": r.model,
                "operation_type": r.operation_type,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_cents": r.cost_cents,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            } for r in proxy_rows
        ],
    }


# ── Lightweight preflight endpoint for the agent ──────────────────


class PreflightResponse(BaseModel):
    enforcement_enabled: bool
    sufficient: bool
    bucket: str
    remaining: float
    reason: Optional[str] = None
    plan_id: str
    plan_display_name: str
    period_end: datetime


@router.get("/preflight", response_model=PreflightResponse)
async def credit_preflight(
    bucket: str = Query("message", regex="^(message|integration)$"),
    required: float = Query(0.5, ge=0),
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
    x_agent_user_id: Optional[str] = Header(None, alias="X-Agent-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> PreflightResponse:
    """Cheap preflight read for tenant agents.

    The agent can call this to refresh its in-process CreditState
    without burning a deduct — useful at agent boot (so the first
    chat call doesn't fail-open through CreditState's cold start)
    and before scheduled routines fire (to skip cleanly).

    Authenticated via X-Agent-Key matched against AgentConfig.
    """
    if not x_agent_key or not x_agent_user_id:
        raise HTTPException(401, "X-Agent-Key + X-Agent-User-Id required")

    cfg_q = await db.execute(
        select(AgentConfig).where(
            AgentConfig.user_id == x_agent_user_id,
            AgentConfig.agent_api_key == x_agent_key,
        )
    )
    if cfg_q.scalar_one_or_none() is None:
        raise HTTPException(403, "agent key mismatch")

    from decimal import Decimal as _Dec
    from app.services.credit_service import (
        BUCKET_MESSAGE, BUCKET_INTEGRATION,
    )
    bucket_const = BUCKET_MESSAGE if bucket == "message" else BUCKET_INTEGRATION

    view = await credit_service.get_balance_view(db, x_agent_user_id)
    check = await credit_service.check_balance(
        db, x_agent_user_id, bucket_const, _Dec(str(required)),
    )
    await db.commit()

    remaining = (
        view.message_credits_remaining if bucket == "message"
        else view.integration_credits_remaining
    )
    return PreflightResponse(
        enforcement_enabled=view.enforcement_enabled,
        sufficient=check.success,
        bucket=bucket,
        remaining=float(remaining),
        reason=check.reason,
        plan_id=view.plan_id,
        plan_display_name=view.plan_display_name,
        period_end=view.period_end,
    )
