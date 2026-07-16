"""Admin — Live Activity delivery debugging.

One read-only snapshot per user answering "why didn't the card reach
this phone?": registered ActivityKit devices, live activity rows, and
the last notification_queue rows WITH the dispatcher's per-channel
verdicts (channels_json). Built 2026-07-16 while debugging silent
non-delivery to the founder's test account — the platform DB is not
reachable from dev machines without a Railway session, so this is the
durable inspection path.

Tokens are surfaced as short prefixes only — enough to correlate with
device logs, never enough to push to.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.db import get_db
from app.db.models import (
    KNOWN_APNS_ENVIRONMENTS, LiveActivity, LiveActivityDevice,
    NotificationQueue,
)

router = APIRouter(prefix="/admin/live-activity", tags=["Admin — Live Activity"])


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


class RestoreDeviceRequest(BaseModel):
    device_id: Optional[str] = None
    environment: Optional[str] = None


@router.post("/{user_id}/restore-device")
async def restore_live_activity_device(
    user_id: str,
    body: RestoreDeviceRequest,
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Un-revoke a user's Live Activity device (newest first, or a
    specific device_id), optionally correcting its APNs environment.
    Ops lever for the environment-mismatch revocation class — the
    dispatcher's env self-heal fixes future sends, but a device
    already revoked is invisible to the lane until restored."""
    if body.environment and body.environment not in KNOWN_APNS_ENVIRONMENTS:
        raise HTTPException(422, f"environment must be one of {sorted(KNOWN_APNS_ENVIRONMENTS)}")
    q = select(LiveActivityDevice).where(LiveActivityDevice.user_id == user_id)
    if body.device_id:
        q = q.where(LiveActivityDevice.id == body.device_id)
    q = q.order_by(LiveActivityDevice.last_seen_at.desc())
    device = (await db.execute(q)).scalars().first()
    if device is None:
        raise HTTPException(404, "no device found")
    device.revoked_at = None
    if body.environment:
        device.apns_environment = body.environment
    resp = {
        "id": device.id,
        "token_prefix": (device.push_to_start_token or "")[:12],
        "environment": device.apns_environment,
        "revoked_at": None,
    }
    await db.commit()
    return resp


@router.get("/{user_id}")
async def live_activity_debug(
    user_id: str,
    limit: int = 20,
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    limit = max(1, min(limit, 100))

    devices = (
        (
            await db.execute(
                select(LiveActivityDevice)
                .where(LiveActivityDevice.user_id == user_id)
                .order_by(LiveActivityDevice.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    activities = (
        (
            await db.execute(
                select(LiveActivity)
                .where(LiveActivity.user_id == user_id)
                .order_by(LiveActivity.updated_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    notifications = (
        (
            await db.execute(
                select(NotificationQueue)
                .where(NotificationQueue.user_id == user_id)
                .order_by(NotificationQueue.created_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )

    return {
        "user_id": user_id,
        "devices": [
            {
                "id": d.id,
                "token_prefix": (d.push_to_start_token or "")[:12],
                "environment": d.apns_environment,
                "device_name": d.device_name,
                "revoked_at": _iso(d.revoked_at),
                "created_at": _iso(d.created_at),
                "last_seen_at": _iso(d.last_seen_at),
            }
            for d in devices
        ],
        "activities": [
            {
                "mission_id": a.mission_id,
                "device_id": a.device_id,
                "status": a.status,
                "last_progress": a.last_progress,
                "has_activity_token": bool(a.activity_push_token),
                "updated_at": _iso(a.updated_at),
            }
            for a in activities
        ],
        "notifications": [
            {
                "id": n.id,
                "event_kind": n.event_kind,
                "title": n.title,
                "priority": n.priority,
                "status": n.status,
                "attempts": n.attempts,
                "last_error": n.last_error,
                "scheduled_for": _iso(n.scheduled_for),
                "sent_at": _iso(n.sent_at),
                "channels": n.channels_json,
                "created_at": _iso(n.created_at),
            }
            for n in notifications
        ],
    }
