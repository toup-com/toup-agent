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
from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import (
    KNOWN_APNS_ENVIRONMENTS, LA_ENDED, LA_STARTED,
    LiveActivity, LiveActivityDevice, NotificationQueue,
    NOTIFY_KIND_MISSION_COMPLETED, NOTIFY_KIND_MISSION_FAILED,
    NOTIFY_KIND_PROGRESS, NQ_QUEUED, NQ_SENDING, NQ_SUPPRESSED,
)
from app.services import apns_push

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
    # Stable per-install UUID minted by the app on first launch.
    # Reinstalls rotate the push-to-start token; matching on
    # (user_id, install_id) lets us update the row in place instead
    # of accreting a stale sibling (APNs 200s dead tokens forever).
    install_id: Optional[str] = Field(default=None, max_length=64)

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
    # Optional context from the app: the ActivityKit activity id, and
    # how the activity came to exist ("local_start" = the app started
    # the card itself, e.g. a chat-turn card armed while foregrounded).
    activity_id: Optional[str] = Field(default=None, max_length=64)
    source: Optional[str] = Field(default=None, max_length=32)

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
        # Row selection order: token first, then install_id.
        # push_to_start_token is the hard UNIQUE key — when a row
        # already owns this token it MUST be the row we update
        # (pointing a different row at the same token would violate
        # the constraint). The install_id lookup covers the reinstall
        # case where the token ROTATED: same install, new token →
        # update that row in place (un-revoke, adopt the new token)
        # instead of accreting a stale sibling whose dead token APNs
        # keeps accepting with 200 — pushes into the void forever.
        result = await db.execute(
            select(LiveActivityDevice).where(
                LiveActivityDevice.push_to_start_token == body.token
            )
        )
        device = result.scalar_one_or_none()
        if device is None and body.install_id:
            result = await db.execute(
                select(LiveActivityDevice)
                .where(
                    LiveActivityDevice.user_id == user_id,
                    LiveActivityDevice.install_id == body.install_id,
                )
                .order_by(LiveActivityDevice.last_seen_at.desc())
            )
            device = result.scalars().first()
        if device is not None:
            if device.user_id != user_id:
                # Device changed accounts: last sign-in wins.
                logger.info(
                    "live-activity device %s re-bound %s → %s",
                    device.id, device.user_id[:8], user_id[:8],
                )
                device.user_id = user_id
            device.push_to_start_token = body.token
            device.apns_environment = body.environment
            device.device_name = body.device_name or device.device_name
            device.app_version = body.app_version or device.app_version
            if body.install_id:
                device.install_id = body.install_id
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
                install_id=body.install_id,
                created_at=now,
                last_seen_at=now,
            )
            db.add(device)

        # Reinstall hygiene: revoke stale siblings — any OTHER live row
        # of this user whose install_id matches the new one (older row
        # of the same install) OR is NULL (row from an app build that
        # predates install_id) and whose token differs. A wrongly-swept
        # second real device self-heals on its next launch: its own
        # registration un-revokes its row above.
        if body.install_id:
            swept = await db.execute(
                update(LiveActivityDevice)
                .where(
                    LiveActivityDevice.user_id == user_id,
                    LiveActivityDevice.id != device.id,
                    LiveActivityDevice.revoked_at.is_(None),
                    LiveActivityDevice.push_to_start_token != body.token,
                    or_(
                        LiveActivityDevice.install_id == body.install_id,
                        LiveActivityDevice.install_id.is_(None),
                    ),
                )
                .values(revoked_at=now)
            )
            if swept.rowcount:
                logger.info(
                    "live-activity register: revoked %d superseded sibling row(s) "
                    "for user %s (install %s)",
                    swept.rowcount, user_id[:8], body.install_id[:12],
                )

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
    input-push-token fallback.

    ADOPTION (chat turns only): the app can start a turn card LOCALLY
    while foregrounded — the platform never pushed a start, so no
    LiveActivity row exists and every later platform update/end for
    that turn would terminate ``no_active_activity``. When no started
    row matches and the mission is a chat turn (``chatturn:`` prefix),
    create the row here so the platform can drive the locally-started
    card. Other mission kinds do NOT adopt — their cards are always
    platform-started."""
    user_id = current_user.id
    now = datetime.utcnow()
    result = await db.execute(
        select(LiveActivity).where(
            LiveActivity.user_id == user_id,
            LiveActivity.mission_id == body.mission_id,
            LiveActivity.status == LA_STARTED,
        )
    )
    rows = result.scalars().all()
    for row in rows:
        row.activity_push_token = body.activity_push_token
        row.updated_at = now
    updated = len(rows)
    if updated:
        await db.commit()
        return {"ok": True, "updated": updated}

    if not body.mission_id.startswith("chatturn:"):
        return {"ok": True, "updated": 0}

    # Adopt: bind the locally-started turn card to the caller's most
    # recently seen live device (the app that just observed this token
    # is by definition the freshest registration).
    result = await db.execute(
        select(LiveActivityDevice)
        .where(
            LiveActivityDevice.user_id == user_id,
            LiveActivityDevice.revoked_at.is_(None),
        )
        .order_by(LiveActivityDevice.last_seen_at.desc())
    )
    device = result.scalars().first()
    if device is None:
        return {"ok": True, "updated": 0}

    # ONE-ACTIVITY-PER-DEVICE bookkeeping (and the partial unique
    # index): the locally-started card is now THE card on this device —
    # force-end any other started rows in DB. No pushes: a preempt end
    # here would race the card the user is looking at; stale platform
    # cards self-heal on their next progress tick.
    await db.execute(
        update(LiveActivity)
        .where(
            LiveActivity.device_id == device.id,
            LiveActivity.status == LA_STARTED,
        )
        .values(status=LA_ENDED, ended_at=now, updated_at=now)
    )

    # UNIQUE (device_id, mission_id): revive an earlier ended row for
    # this turn if one exists, else insert fresh.
    existing = (await db.execute(
        select(LiveActivity).where(
            LiveActivity.device_id == device.id,
            LiveActivity.mission_id == body.mission_id,
        )
    )).scalar_one_or_none()
    if existing is not None and existing.status == LA_ENDED:
        # The turn ALREADY completed — the platform end beat the token
        # report. Store the token (a later ack/cleanup push can still
        # reach the card) but do NOT revive: nothing ever ends a
        # revived row for a finished turn, and it squats the
        # one-started-per-device slot until the GC (2026-07-18: 8h
        # wedge that every reminder-fire attempt fought).
        existing.activity_push_token = body.activity_push_token
        existing.updated_at = now
        await db.commit()
        return {"ok": True, "updated": 0, "already_ended": True}
    if existing is not None:
        existing.status = LA_STARTED
        existing.activity_push_token = body.activity_push_token
        existing.apns_environment = device.apns_environment
        existing.started_at = now
        existing.updated_at = now
        existing.ended_at = None
    else:
        db.add(LiveActivity(
            id=str(uuid.uuid4()),
            user_id=user_id,
            mission_id=body.mission_id,
            device_id=device.id,
            activity_push_token=body.activity_push_token,
            apns_environment=device.apns_environment,
            status=LA_STARTED,
            started_at=now,
            updated_at=now,
        ))
    try:
        await db.commit()
    except IntegrityError:
        # Raced a concurrent platform start for the same turn — that
        # row exists now; the app can simply re-report the token.
        await db.rollback()
        return {"ok": True, "updated": 0}
    logger.info(
        "live-activity adopt: %s bound to device %s (source=%s)",
        body.mission_id, device.id, body.source or "unspecified",
    )
    return {"ok": True, "updated": 0, "adopted": True}


class LiveActivityAck(BaseModel):
    mission_id: str = Field(..., min_length=1, max_length=64)


@router.post("/ack")
async def ack_live_activity(
    body: LiveActivityAck,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Tap acknowledgment: the user tapped this mission's card (deep
    links carry ``?mission=<id>``), which proves the alert was SEEN.
    End the mission's cards on every device (immediate dismissal) and
    suppress its still-pending completed/failed/progress queue rows so
    an already-seen mission never re-alerts. needs_input/needs_approval
    rows are deliberately NOT suppressed — seen is not answered."""
    user_id = current_user.id
    now = datetime.utcnow()

    rows = (
        await db.execute(
            select(LiveActivity, LiveActivityDevice)
            .join(LiveActivityDevice, LiveActivity.device_id == LiveActivityDevice.id)
            .where(
                LiveActivity.user_id == user_id,
                LiveActivity.mission_id == body.mission_id,
                LiveActivity.status == LA_STARTED,
            )
        )
    ).all()
    ended = 0
    for la, device in rows:
        token = la.activity_push_token or device.push_to_start_token
        if token and apns_push.apns_configured():
            payload = apns_push.build_end_payload(
                title="Done",
                dismissal_date=int(now.timestamp()) - 1,
                timestamp=int(now.timestamp()),
            )
            # Best-effort: the DB end below is the invariant; the push
            # just clears the on-screen card faster than staleness.
            await apns_push.send_live_activity(
                token, payload,
                environment=device.apns_environment or "development",
                priority=10,
            )
        la.status = LA_ENDED
        la.ended_at = now
        la.updated_at = now
        ended += 1

    pending = (
        await db.execute(
            select(NotificationQueue).where(
                NotificationQueue.user_id == user_id,
                NotificationQueue.status.in_([NQ_QUEUED, NQ_SENDING]),
                NotificationQueue.event_kind.in_([
                    NOTIFY_KIND_MISSION_COMPLETED,
                    NOTIFY_KIND_MISSION_FAILED,
                    NOTIFY_KIND_PROGRESS,
                ]),
            )
        )
    ).scalars().all()
    suppressed = 0
    for row in pending:
        if (row.data_json or {}).get("mission_id") != body.mission_id:
            continue
        row.status = NQ_SUPPRESSED
        row.claimed_at = None
        row.channels_json = {**(row.channels_json or {}),
                             "policy": {"suppressed": "acked"}}
        suppressed += 1

    await db.commit()
    if ended or suppressed:
        logger.info(
            "live-activity ack: mission %s — ended %d cards, suppressed %d rows",
            body.mission_id, ended, suppressed,
        )
    return {"ok": True, "ended": ended, "suppressed": suppressed}


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
