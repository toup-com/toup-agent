"""Push-device registry — Expo push-token registration (Autopilot PR2).

Platform-only router (mounted in platform_main.py — NOT agent_main):
push tokens are per-device credentials the platform owns; agent
containers never see them (base.py isolation stance, same as connector
tokens).

Endpoint shapes copy the extension-device registry
(app/api/extension.py: _record_pairing / _list_devices / revoke):

- POST   /devices/push            — register/refresh a token (upsert)
- GET    /devices/push            — list the caller's active devices
- POST   /devices/push/unregister — revoke by token (sign-out flow,
                                    device id unknown client-side)
- DELETE /devices/push/{device_id} — revoke one device (settings UI)

Upsert semantics: ``expo_push_token`` is globally unique and stable per
app-install. Re-registration bumps ``last_seen_at`` and clears
``revoked_at``; a token that shows up under a different account is
re-bound to the new user (last sign-in on that device wins — the
previous owner signed out or the device changed hands). The dispatcher
(PR3) additionally revokes tokens whose Expo receipts return
``DeviceNotRegistered``.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import PushDevice

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devices/push", tags=["Push Devices"])


_KNOWN_PLATFORMS = {"ios", "android"}


class PushDeviceRegister(BaseModel):
    token: str = Field(min_length=10, max_length=200)
    platform: str
    device_name: Optional[str] = Field(default=None, max_length=200)
    app_version: Optional[str] = Field(default=None, max_length=40)

    @field_validator("token")
    @classmethod
    def _token_shape(cls, v: str) -> str:
        v = v.strip()
        # Expo tokens look like ExponentPushToken[xxxx] (legacy:
        # ExpoPushToken[xxxx]). Reject anything else at the door so a
        # raw APNs/FCM token pasted by mistake fails loudly here, not
        # as a silent Expo send error weeks later.
        if not (v.startswith("ExponentPushToken[") or v.startswith("ExpoPushToken[")):
            raise ValueError("not an Expo push token")
        return v

    @field_validator("platform")
    @classmethod
    def _platform_known(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in _KNOWN_PLATFORMS:
            raise ValueError(f"platform must be one of {sorted(_KNOWN_PLATFORMS)}")
        return v


class PushDeviceOut(BaseModel):
    id: str
    platform: str
    device_name: Optional[str]
    app_version: Optional[str]
    created_at: str
    last_seen_at: Optional[str]


class PushDeviceUnregister(BaseModel):
    token: str = Field(min_length=10, max_length=200)


def _to_out(d: PushDevice) -> PushDeviceOut:
    return PushDeviceOut(
        id=d.id,
        platform=d.platform,
        device_name=d.device_name,
        app_version=d.app_version,
        created_at=d.created_at.isoformat() if d.created_at else "",
        last_seen_at=d.last_seen_at.isoformat() if d.last_seen_at else None,
    )


@router.post("", response_model=PushDeviceOut)
@router.post("/", response_model=PushDeviceOut, include_in_schema=False)
async def register_push_device(
    body: PushDeviceRegister,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PushDeviceOut:
    """Register (or refresh) this device's Expo push token.

    Called by the app after login and on every cold start — the
    repeat calls are what keep ``last_seen_at`` honest, which the
    dispatcher uses to prefer the most-recently-alive device."""
    now = datetime.utcnow()
    result = await db.execute(
        select(PushDevice).where(PushDevice.expo_push_token == body.token)
    )
    device = result.scalar_one_or_none()
    if device is not None:
        if device.user_id != current_user.id:
            # Device changed accounts: last sign-in wins.
            logger.info(
                "push device %s re-bound %s → %s",
                device.id, device.user_id[:8], current_user.id[:8],
            )
            device.user_id = current_user.id
        device.platform = body.platform
        device.device_name = body.device_name or device.device_name
        device.app_version = body.app_version or device.app_version
        device.last_seen_at = now
        device.revoked_at = None
    else:
        # Explicit id + created_at (column defaults fire at flush) so
        # the response is built BEFORE commit — pgbouncer txn-mode rule.
        device = PushDevice(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            expo_push_token=body.token,
            platform=body.platform,
            device_name=body.device_name,
            app_version=body.app_version,
            created_at=now,
            last_seen_at=now,
        )
        db.add(device)
    out = _to_out(device)
    await db.commit()
    return out


@router.get("", response_model=list[PushDeviceOut])
@router.get("/", response_model=list[PushDeviceOut], include_in_schema=False)
async def list_push_devices(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[PushDeviceOut]:
    result = await db.execute(
        select(PushDevice)
        .where(
            PushDevice.user_id == current_user.id,
            PushDevice.revoked_at.is_(None),
        )
        .order_by(PushDevice.created_at.desc())
    )
    return [_to_out(d) for d in result.scalars().all()]


@router.post("/unregister")
async def unregister_push_device(
    body: PushDeviceUnregister,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Sign-out flow: the app knows its token but not our device id.
    Revoking someone else's token is a silent no-op (don't leak
    whether a token exists)."""
    result = await db.execute(
        select(PushDevice).where(
            PushDevice.expo_push_token == body.token.strip(),
            PushDevice.user_id == current_user.id,
        )
    )
    device = result.scalar_one_or_none()
    if device is not None and device.revoked_at is None:
        device.revoked_at = datetime.utcnow()
        await db.commit()
        return {"ok": True, "revoked": True}
    return {"ok": True, "revoked": False}


@router.delete("/{device_id}")
async def revoke_push_device(
    device_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    device = await db.get(PushDevice, device_id)
    if device is None or device.user_id != current_user.id:
        raise HTTPException(404, "Device not found")
    if device.revoked_at is None:
        device.revoked_at = datetime.utcnow()
        await db.commit()
    return {"ok": True}
