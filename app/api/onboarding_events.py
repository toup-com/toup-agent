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


@events_router.post("/activate-free-tier", response_model=FreeTierActivateResponse)
async def post_activate_free_tier(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FreeTierActivateResponse:
    import hashlib
    import secrets as _secrets
    from sqlalchemy import select
    from datetime import datetime
    from app.db.models import AgentConfig

    cfg = (await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user.id)
    )).scalar_one_or_none()
    if cfg is None:
        # Fresh signup that hasn't hit /agent-setup yet — the legacy
        # `_get_or_create_config` path is the proper way to materialise
        # the row. Use it so we don't drift from defaults.
        from app.api.agent_setup import _get_or_create_config
        cfg = await _get_or_create_config(str(user.id), db)

    was_already_active = cfg.bundle_status in ("active", "cancelling")

    # llm_token: agent reads this from its env (TOUP_LLM_TOKEN); proxy
    # SHA-256-hashes the incoming bearer and compares to llm_token_hash.
    # We reuse connect_token as the canonical source so a future Stripe
    # checkout doesn't mint a second one.
    if not cfg.connect_token:
        cfg.connect_token = f"toup_ct_{_secrets.token_urlsafe(32)}"
    _expected_hash = hashlib.sha256(cfg.connect_token.encode()).hexdigest()
    if cfg.llm_token_hash != _expected_hash:
        cfg.llm_token_hash = _expected_hash
        logger.info(
            "onboarding.free_activate: refreshed llm_token_hash user=%s",
            str(user.id),
        )

    if not was_already_active:
        cfg.bundle_status = "active"
        cfg.bundle_started_at = cfg.bundle_started_at or datetime.utcnow()
        cfg.llm_mode = "bundle"
        logger.info(
            "onboarding.free_activate: bundle_status -> active user=%s",
            str(user.id),
        )

    await db.commit()
    await db.refresh(cfg)

    # Best-effort: push the new env to the running tenant container so
    # the agent picks up the TOUP_LLM_TOKEN without a manual restart.
    # The bridge call is idempotent and returns quickly when the env
    # is already up-to-date.
    try:
        from app.services.docker_host_service import update_container_env
        await update_container_env(db, str(user.id), cfg)
    except Exception as e:
        logger.warning(
            "onboarding.free_activate: container env push failed user=%s err=%s",
            str(user.id), e,
        )
        # Non-fatal — the next /agent-setup/config save will re-push.

    return FreeTierActivateResponse(
        activated=not was_already_active,
        already_active=was_already_active,
        bundle_status=cfg.bundle_status or "none",
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
