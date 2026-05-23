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

    onboarding_v2 = await feature_flags.is_onboarding_v2_enabled(db, seed=seed)
    return {"onboarding_v2": bool(onboarding_v2)}


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
