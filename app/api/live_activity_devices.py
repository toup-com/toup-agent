"""Live Activity device registry — ActivityKit push-token registration.

Platform-only router (mounted in platform_main.py — NOT agent_main):
APNs tokens are per-device credentials the platform owns; agent
containers never see them (base.py isolation stance, same as
push_devices).

- POST /devices/live-activity                 — register/refresh the
      device's push-to-start token (upsert; app calls on every launch
      and whenever Apple rotates the token)
- GET  /devices/live-activity                 — list the caller's
      active registrations
- POST /devices/live-activity/activity-token  — the app observed a
      per-activity update token for a mission activity; store it so
      updates prefer it over the input-push-token path
- POST /devices/live-activity/unregister      — revoke by token
      (sign-out flow)

Token shape: APNs tokens are hex — reject anything else at the door
(an Expo token pasted here would otherwise 400 at Apple weeks later).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import (
    KNOWN_APNS_ENVIRONMENTS, LA_STARTED, LiveActivity, LiveActivityDevice,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devices/live-activity", tags=["Live Activity Devices"])


def _is_hex_token(v: str) -> bool:
    try:
        bytes.fromhex(v)
        return True
    except ValueError:
        return False


class LiveActivityDeviceRegister(BaseModel):
    token: str = Field(min_length=32, max_length=200)
    environment: str = "development"
    device_name: Optional[str] = Field(default=None, max_length=200)
    app_version: Optional[str] = Field(default=None, max_length=40)

    @field_validator("token")
    @classmethod
    def _token_shape(cls, v: str) -> str:
        v = v.strip().lower()
        if not _is_hex_token(v):
            raise ValueError("not a hex APNs token")
        return v

    @field_validator("environment")
    @classmethod
    def _environment_known(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in KNOWN_APNS_ENVIRONMENTS:
            raise ValueError(
                f"environment must be one of {sorted(KNOWN_APNS_ENVIRONMENTS)}"
            )
        return v


class LiveActivityDeviceOut(BaseModel):
    id: str
    environment: str
    device_name: Optional[str]
    app_version: Optional[str]
    created_at: str
    last_seen_at: Optional[str]


class ActivityTokenReport(BaseModel):
    """From the app's onTokenReceived listener. ``mission_id`` is the
    ActivityAttributes ``name`` we set in the start payload (local,
    non-mission activities report name='ExpoLiveActivity' — the row
    lookup simply misses and we no-op)."""

    mission_id: str = Field(min_length=1, max_length=64)
    activity_push_token: str = Field(min_length=32, max_length=200)

    @field_validator("activity_push_token")
    @classmethod
    def _token_shape(cls, v: str) -> str:
        v = v.strip().lower()
        if not _is_hex_token(v):
            raise ValueError("not a hex APNs token")
        return v


class LiveActivityDeviceUnregister(BaseModel):
    token: str = Field(min_length=32, max_length=200)


def _to_out(d: LiveActivityDevice) -> LiveActivityDeviceOut:
    return LiveActivityDeviceOut(
        id=d.id,
        environment=d.apns_environment,
        device_name=d.device_name,
        app_version=d.app_version,
        created_at=d.created_at.isoformat() if d.created_at else "",
        last_seen_at=d.last_seen_at.isoformat() if d.last_seen_at else None,
    )


@router.post("", response_model=LiveActivityDeviceOut)
@router.post("/", response_model=LiveActivityDeviceOut, include_in_schema=False)
async def register_live_activity_device(
    body: LiveActivityDeviceRegister,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LiveActivityDeviceOut:
    # Upsert with a bounded retry: the app can fire several parallel
    # registrations of the same token (its retry loop, dev fast-refresh
    # stacking listeners), and the losers of the unique-constraint race
    # may re-read before the winner has committed — so on conflict we
    # back off briefly and re-run the whole select-or-insert.
    #
    # current_user shares this request's session (FastAPI caches the
    # get_db dependency), so our rollback expires it — read the id into
    # a plain string BEFORE any rollback can happen.
    user_id = current_user.id
    now = datetime.utcnow()
    for attempt in range(4):
        result = await db.execute(
            select(LiveActivityDevice).where(
                LiveActivityDevice.push_to_start_token == body.token
            )
        )
        device = result.scalar_one_or_none()
        if device is not None:
            if device.user_id != user_id:
                # Device changed accounts: last sign-in wins.
                logger.info(
                    "live-activity device %s re-bound %s → %s",
                    device.id, device.user_id[:8], user_id[:8],
                )
                device.user_id = user_id
            device.apns_environment = body.environment
            device.device_name = body.device_name or device.device_name
            device.app_version = body.app_version or device.app_version
            device.last_seen_at = now
            device.revoked_at = None
        else:
            # Explicit id + created_at so the response is built BEFORE
            # commit — pgbouncer txn-mode rule.
            device = LiveActivityDevice(
                id=str(uuid.uuid4()),
                user_id=user_id,
                push_to_start_token=body.token,
                apns_environment=body.environment,
                device_name=body.device_name,
                app_version=body.app_version,
                created_at=now,
                last_seen_at=now,
            )
            db.add(device)
        out = _to_out(device)
        try:
            await db.commit()
            return out
        except IntegrityError:
            await db.rollback()
            await asyncio.sleep(0.05 * (attempt + 1))
    raise HTTPException(503, "registration raced repeatedly — retry")


@router.get("", response_model=list[LiveActivityDeviceOut])
@router.get("/", response_model=list[LiveActivityDeviceOut], include_in_schema=False)
async def list_live_activity_devices(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[LiveActivityDeviceOut]:
    result = await db.execute(
        select(LiveActivityDevice)
        .where(
            LiveActivityDevice.user_id == current_user.id,
            LiveActivityDevice.revoked_at.is_(None),
        )
        .order_by(LiveActivityDevice.created_at.desc())
    )
    return [_to_out(d) for d in result.scalars().all()]


@router.post("/activity-token")
async def report_activity_token(
    body: ActivityTokenReport,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Best-effort upgrade path: once the app reports the real
    per-activity token, updates for that activity stop relying on the
    input-push-token fallback."""
    result = await db.execute(
        select(LiveActivity).where(
            LiveActivity.user_id == current_user.id,
            LiveActivity.mission_id == body.mission_id,
            LiveActivity.status == LA_STARTED,
        )
    )
    rows = result.scalars().all()
    for row in rows:
        row.activity_push_token = body.activity_push_token
        row.updated_at = datetime.utcnow()
    updated = len(rows)
    if updated:
        await db.commit()
    return {"ok": True, "updated": updated}


@router.post("/unregister")
async def unregister_live_activity_device(
    body: LiveActivityDeviceUnregister,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Sign-out flow. Revoking someone else's token is a silent no-op
    (don't leak whether a token exists)."""
    result = await db.execute(
        select(LiveActivityDevice).where(
            LiveActivityDevice.push_to_start_token == body.token.strip().lower(),
            LiveActivityDevice.user_id == current_user.id,
        )
    )
    device = result.scalar_one_or_none()
    if device is not None and device.revoked_at is None:
        device.revoked_at = datetime.utcnow()
        await db.commit()
        return {"ok": True, "revoked": True}
    return {"ok": True, "revoked": False}
