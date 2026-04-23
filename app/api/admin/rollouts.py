"""
Rollout API — endpoints for CI + admin rollout orchestration.

Endpoints:
  POST   /api/admin/rollout/start     CI webhook (X-Rollout-Secret)
  POST   /api/admin/rollout/manual    Admin-authenticated manual trigger
  GET    /api/admin/rollout/latest    Active rollout (or most recent)
  GET    /api/admin/rollout/{id}      Rollout with full attempt breakdown
  POST   /api/admin/rollout/{id}/cancel   Mark in-flight rollout cancelled

See docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md §5-§6.
"""

from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.api.admin.deps import require_admin
from app.config import settings
from app.db.database import get_db
from app.db.models import Rollout, RolloutAttempt, User
from app.services.rollout_service import (
    start_rollout, cancel_rollout, active_rollout,
    RolloutInProgress,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/rollout", tags=["admin-rollout"])


# ─── Request / response schemas ───────────────────────────────────


class StartRolloutReq(BaseModel):
    image_tag: str = Field(..., description="ghcr.io/toup-com/toup-agent:<sha>")
    canary_wait_minutes: Optional[int] = Field(None, ge=1, le=60)
    trigger: str = Field(default="ci", description="ci | manual | rollback")
    notes: Optional[str] = None

    @field_validator("image_tag")
    @classmethod
    def _prefix_check(cls, v: str) -> str:
        # Belt + suspenders: rollout_service re-validates, but rejecting
        # bad payloads at the API boundary returns a cleaner 400.
        if not v.startswith("ghcr.io/toup-com/toup-agent:"):
            raise ValueError(
                "image_tag must start with 'ghcr.io/toup-com/toup-agent:'"
            )
        return v


class RolloutSummaryResp(BaseModel):
    id: str
    image_tag: str
    status: str
    trigger: str
    triggered_by: Optional[str]
    canary_prefix: Optional[str]
    canary_wait_minutes: int
    started_at: str
    completed_at: Optional[str]
    notes: Optional[str]
    # Aggregates
    total_attempts: int
    ok_count: int
    failed_count: int
    rolled_back_count: int
    rollback_failed_count: int


class RolloutAttemptResp(BaseModel):
    id: str
    container_user_prefix: str   # 8-char prefix derived from container's owner
    prior_tag: str
    new_tag: str
    status: str
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]
    health_checks_passed: int
    duration_ms: Optional[int]


class RolloutDetailResp(RolloutSummaryResp):
    attempts: list[RolloutAttemptResp]


# ─── Helpers ──────────────────────────────────────────────────────


def _verify_rollout_secret(header_value: Optional[str]) -> None:
    """Constant-time comparison of X-Rollout-Secret header vs settings.
    Raises 401 on mismatch.
    """
    expected = settings.rollout_secret or ""
    if not expected:
        # Misconfigured — block rather than allow everything
        raise HTTPException(503, "ROLLOUT_SECRET not configured on platform")
    if not header_value or not hmac.compare_digest(header_value, expected):
        raise HTTPException(401, "invalid X-Rollout-Secret")


async def _build_summary(db: AsyncSession, rollout: Rollout) -> RolloutSummaryResp:
    """Aggregate attempt counts for a rollout. Uses SQL count rather than
    loading all attempt rows — constant time regardless of tenant count.
    """
    from sqlalchemy import func
    result = await db.execute(
        select(
            RolloutAttempt.status,
            func.count(RolloutAttempt.id),
        )
        .where(RolloutAttempt.rollout_id == rollout.id)
        .group_by(RolloutAttempt.status)
    )
    counts = dict(result.all())
    total = sum(counts.values())
    return RolloutSummaryResp(
        id=rollout.id,
        image_tag=rollout.image_tag,
        status=rollout.status,
        trigger=rollout.trigger,
        triggered_by=rollout.triggered_by,
        canary_prefix=rollout.canary_prefix,
        canary_wait_minutes=rollout.canary_wait_minutes,
        started_at=rollout.started_at.isoformat() if rollout.started_at else "",
        completed_at=rollout.completed_at.isoformat() if rollout.completed_at else None,
        notes=rollout.notes,
        total_attempts=total,
        ok_count=counts.get("ok", 0),
        failed_count=counts.get("failed", 0),
        rolled_back_count=counts.get("rolled_back", 0),
        rollback_failed_count=counts.get("rollback_failed", 0),
    )


# ─── Endpoints ────────────────────────────────────────────────────


@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def ci_start_rollout(
    req: StartRolloutReq,
    x_rollout_secret: Optional[str] = Header(None, alias="X-Rollout-Secret"),
    db: AsyncSession = Depends(get_db),
):
    """CI webhook. Validates the shared secret, not a user session."""
    _verify_rollout_secret(x_rollout_secret)
    try:
        rollout = await start_rollout(
            db,
            image_tag=req.image_tag,
            trigger=req.trigger,
            canary_wait_minutes=req.canary_wait_minutes,
            notes=req.notes,
        )
    except RolloutInProgress as e:
        # 409 per W9 scenario 10 decision
        raise HTTPException(
            status_code=409,
            detail={
                "error": "rollout_in_progress",
                "active_rollout_id": e.active_id,
                "active_image_tag": e.active_tag,
            },
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"rollout_id": rollout.id, "status": rollout.status}


@router.post("/manual", status_code=status.HTTP_202_ACCEPTED)
async def admin_manual_rollout(
    req: StartRolloutReq,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Manual trigger for admins. Use for rollback (pass a prior SHA as
    image_tag with trigger='rollback') or to retry after a failure.
    """
    try:
        rollout = await start_rollout(
            db,
            image_tag=req.image_tag,
            trigger=req.trigger or "manual",
            triggered_by=current_user.id,
            canary_wait_minutes=req.canary_wait_minutes,
            notes=req.notes,
        )
    except RolloutInProgress as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "rollout_in_progress",
                "active_rollout_id": e.active_id,
                "active_image_tag": e.active_tag,
            },
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"rollout_id": rollout.id, "status": rollout.status}


@router.get("/latest", response_model=RolloutSummaryResp)
async def get_latest(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Current in-flight rollout if any; otherwise the most recent one.
    Used by the admin UI banner.
    """
    active = await active_rollout(db)
    if active:
        return await _build_summary(db, active)
    result = await db.execute(
        select(Rollout).order_by(Rollout.started_at.desc()).limit(1)
    )
    latest = result.scalar_one_or_none()
    if not latest:
        raise HTTPException(404, "no rollouts yet")
    return await _build_summary(db, latest)


@router.get("/history")
async def get_history(
    limit: int = 30,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> list[RolloutSummaryResp]:
    """Last N rollouts, newest first. Used by the admin table view."""
    limit = max(1, min(100, limit))
    result = await db.execute(
        select(Rollout).order_by(Rollout.started_at.desc()).limit(limit)
    )
    rollouts = list(result.scalars().all())
    return [await _build_summary(db, r) for r in rollouts]


@router.get("/{rollout_id}", response_model=RolloutDetailResp)
async def get_rollout(
    rollout_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Full rollout detail — summary + per-attempt breakdown."""
    result = await db.execute(
        select(Rollout)
        .where(Rollout.id == rollout_id)
        .options(joinedload(Rollout.attempts))
    )
    rollout = result.unique().scalar_one_or_none()
    if not rollout:
        raise HTTPException(404, "rollout not found")

    # Resolve container → user_id prefix for each attempt
    from app.db.models import ManagedContainer
    container_ids = [a.container_id for a in rollout.attempts]
    prefix_map: dict[str, str] = {}
    if container_ids:
        cresult = await db.execute(
            select(ManagedContainer.id, ManagedContainer.user_id)
            .where(ManagedContainer.id.in_(container_ids))
        )
        for cid, uid in cresult.all():
            prefix_map[cid] = (uid or "")[:8]

    summary = await _build_summary(db, rollout)
    return RolloutDetailResp(
        **summary.model_dump(),
        attempts=[
            RolloutAttemptResp(
                id=a.id,
                container_user_prefix=prefix_map.get(a.container_id, ""),
                prior_tag=a.prior_tag,
                new_tag=a.new_tag,
                status=a.status,
                started_at=a.started_at.isoformat() if a.started_at else "",
                completed_at=a.completed_at.isoformat() if a.completed_at else None,
                error=a.error,
                health_checks_passed=a.health_checks_passed,
                duration_ms=a.duration_ms,
            )
            for a in rollout.attempts
        ],
    )


@router.post("/{rollout_id}/cancel")
async def cancel(
    rollout_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Mark an in-flight rollout as cancelled. Does NOT revert already-upgraded
    tenants — 'cancelled' means 'stop after current batch'.
    """
    rollout = await cancel_rollout(db, rollout_id)
    if not rollout:
        raise HTTPException(404, "rollout not found")
    return await _build_summary(db, rollout)
