"""
Onboarding-v2 telemetry endpoints.

Three surfaces, mounted on `platform_main:app` under `/api`:

  GET  /api/system/feature-flags         (public; per-caller flag readout)
  POST /api/onboarding/events            (auth; frontend emits step events)
  PUT  /api/admin/feature-flags/onboarding-v2   (admin; flip rollout %)

Why a server-side event endpoint instead of letting the SPA log to its
own console: we want one source of truth for the rollout funnel that the
admin Loki dashboard can graph without correlating frontend + backend
clocks, and `user_id` shouldn't be trusted from the client side anyway.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import User, get_db
from app.api.auth import get_current_user
from app.api.admin.deps import require_admin
from app.services import feature_flags
from app.services.onboarding_events import (
    emit_step_completed, emit_plan_selected,
)

logger = logging.getLogger(__name__)


# ── Public: per-caller flag readout ──────────────────────────────────

public_router = APIRouter(prefix="/system", tags=["System"])


@public_router.get("/feature-flags")
async def get_feature_flags(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    # No auth required — the OnboardingShell mounts before login on the
    # signup path and still needs to know which LLM step to render. We
    # bucket anonymous callers by source IP (deterministic) so the same
    # browser keeps the same answer across the session; for logged-in
    # callers a Bearer token bumps the seed to the user_id so they
    # transition predictably across rollout-pct bumps.
    seed: Optional[str] = None
    auth_header = request.headers.get("authorization") or ""
    if auth_header.startswith("Bearer "):
        from app.services.auth_service import decode_access_token
        try:
            user_id = decode_access_token(auth_header.removeprefix("Bearer ").strip())
            if user_id:
                seed = user_id
        except Exception:
            seed = None
    if not seed and request.client:
        seed = request.client.host or None

    # EVERY registered flag, not just onboarding_v2. A client cannot tell a
    # missing key from a disabled one — `mobileShell.ts` reads an absent
    # `web_mobile_shell` as off — so a readout that names only one flag is a
    # readout that silently pins every other flag to false no matter what the
    # admin toggle says. Adding a flag to `feature_flags.FLAGS` is all it takes
    # to appear here.
    return await feature_flags.all_flags_for(db, seed=seed)


# ── Auth: frontend event emission ────────────────────────────────────

events_router = APIRouter(prefix="/onboarding", tags=["Onboarding"])


class StepCompletedPayload(BaseModel):
    step: str = Field(..., min_length=1, max_length=64)
    duration_ms: int = Field(..., ge=0, le=24 * 60 * 60 * 1000)


class PlanSelectedPayload(BaseModel):
    plan: str = Field(..., min_length=1, max_length=64)
    was_default: bool = False


@events_router.post(
    "/events/step-completed",
    status_code=204,
    response_class=Response,
)
async def post_step_completed(
    body: StepCompletedPayload,
    user: User = Depends(get_current_user),
):
    emit_step_completed(
        step=body.step, user_id=str(user.id), duration_ms=body.duration_ms,
    )
    return Response(status_code=204)


@events_router.post(
    "/events/plan-selected",
    status_code=204,
    response_class=Response,
)
async def post_plan_selected(
    body: PlanSelectedPayload,
    user: User = Depends(get_current_user),
):
    emit_plan_selected(
        plan=body.plan, user_id=str(user.id), was_default=body.was_default,
    )
    return Response(status_code=204)


# ── Free-tier activation (hotfix for PR 2 BYO removal) ──────────────
#
# The legacy LLM Bundle flow paid for chat via Stripe; the
# `invoice.payment_succeeded` webhook then minted `llm_token_hash` and
# flipped `bundle_status='active'`. The LLM proxy still gates on
# `bundle_status in ('active', 'cancelling')`, so Free users (who
# never go through Stripe) land in chat with `bundle_status='none'`
# and a NULL `llm_token_hash` — every LLM call 403s and the chat
# surfaces "Error: Something went wrong."
#
# This endpoint closes the gap. Called by the new LlmRoute when the
# user clicks "Continue on Free", and (idempotent) on Install mount
# as a defensive backstop. Mirrors the first-activation half of
# `vps.py::_handle_invoice_succeeded` minus the Stripe / period
# bookkeeping — Free has no period to track; the credit-system
# `subscription_plans.free` row is the quota source of truth.


class FreeTierActivateResponse(BaseModel):
    activated: bool
    already_active: bool
    bundle_status: str
    env_pushed: bool
    env_error: Optional[str] = None


# Path carries the `/events/` segment to match the sibling routes above
# (/events/step-completed, /events/plan-selected) and EVERY client: mobile
# (api.ts activateFreeTier), web LlmRoute/InstallRoute. Without it the full path
# was /api/onboarding/activate-free-tier while clients POST
# /api/onboarding/events/activate-free-tier → 405 Method Not Allowed. Fixed
# backend-side (not the clients) because already-shipped mobile binaries hardcode
# the /events/ path and can't be patched in flight.
@events_router.post("/events/activate-free-tier", response_model=FreeTierActivateResponse)
async def post_activate_free_tier(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FreeTierActivateResponse:
    # Delegates to the shared `activate_free_tier` service so the same
    # logic runs from BOTH onboarding (LlmRoute / InstallRoute) and the
    # critical-path "Wake your agent up" provision call. Idempotent.
    from app.services.free_tier_activation import activate_free_tier
    result = await activate_free_tier(db, str(user.id), force_env_push=True)
    return FreeTierActivateResponse(
        activated=result.activated,
        already_active=result.already_active,
        bundle_status=result.bundle_status,
        env_pushed=result.env_pushed,
        env_error=result.env_error,
    )


# ── Admin: rollout toggle ────────────────────────────────────────────

admin_router = APIRouter(
    prefix="/admin/feature-flags", tags=["Admin — Feature flags"],
)


class OnboardingV2Snapshot(BaseModel):
    rollout_pct: int
    env_default_pct: int


class OnboardingV2Update(BaseModel):
    rollout_pct: int = Field(..., ge=0, le=100)


@admin_router.get("/onboarding-v2", response_model=OnboardingV2Snapshot)
async def get_onboarding_v2(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> OnboardingV2Snapshot:
    from app.config import settings as _s
    return OnboardingV2Snapshot(
        rollout_pct=await feature_flags.get_onboarding_v2_rollout_pct(db),
        env_default_pct=_s.onboarding_v2_rollout_pct,
    )


@admin_router.put("/onboarding-v2", response_model=OnboardingV2Snapshot)
async def put_onboarding_v2(
    body: OnboardingV2Update,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> OnboardingV2Snapshot:
    from app.config import settings as _s
    pct = await feature_flags.set_onboarding_v2_rollout_pct(db, body.rollout_pct)
    logger.info(
        "admin.feature_flag.onboarding_v2 changed_pct=%d by_admin=%s",
        pct, admin.id,
    )
    return OnboardingV2Snapshot(
        rollout_pct=pct, env_default_pct=_s.onboarding_v2_rollout_pct,
    )


# ── Admin: any registered flag ───────────────────────────────────────
#
# The two routes above are `onboarding-v2` in the PATH, so every new flag needed
# its own pair. These take the flag NAME instead. The hyphenated routes stay:
# the admin panel calls them and the shapes are identical, so they cost nothing
# to keep and breaking a deployed console to save a handler is a bad trade.
#
# Declared AFTER the static ones — FastAPI matches in declaration order, so a
# `/{flag}` above them would swallow `/onboarding-v2` and answer it generically.


class FlagSnapshot(BaseModel):
    flag: str
    rollout_pct: int
    env_default_pct: int


class FlagUpdate(BaseModel):
    rollout_pct: int = Field(..., ge=0, le=100)


def _known_flag(flag: str) -> str:
    if flag not in feature_flags.FLAGS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown feature flag {flag!r}; known: {sorted(feature_flags.FLAGS)}",
        )
    return flag


@admin_router.get("", response_model=list[FlagSnapshot])
async def list_feature_flags(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> list[FlagSnapshot]:
    from app.config import settings as _s
    return [
        FlagSnapshot(
            flag=name,
            rollout_pct=await feature_flags.get_rollout_pct(db, name),
            env_default_pct=int(getattr(_s, spec.env_attr, 0) or 0),
        )
        for name, spec in feature_flags.FLAGS.items()
    ]


@admin_router.get("/flag/{flag}", response_model=FlagSnapshot)
async def get_feature_flag(
    flag: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> FlagSnapshot:
    from app.config import settings as _s
    _known_flag(flag)
    return FlagSnapshot(
        flag=flag,
        rollout_pct=await feature_flags.get_rollout_pct(db, flag),
        env_default_pct=int(getattr(_s, feature_flags.FLAGS[flag].env_attr, 0) or 0),
    )


@admin_router.put("/flag/{flag}", response_model=FlagSnapshot)
async def put_feature_flag(
    flag: str,
    body: FlagUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> FlagSnapshot:
    from app.config import settings as _s
    _known_flag(flag)
    pct = await feature_flags.set_rollout_pct(db, flag, body.rollout_pct)
    logger.info(
        "admin.feature_flag.%s changed_pct=%d by_admin=%s", flag, pct, admin.id,
    )
    return FlagSnapshot(
        flag=flag,
        rollout_pct=pct,
        env_default_pct=int(getattr(_s, feature_flags.FLAGS[flag].env_attr, 0) or 0),
    )
