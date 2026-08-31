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
from datetime import datetime, timedelta
from typing import List, Optional

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
    NOTIFY_KIND_MISSION_STARTED, NOTIFY_KIND_PROGRESS,
    NQ_QUEUED, NQ_SENDING, NQ_SUPPRESSED,
)
from app.services import apns_push

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devices/live-activity", tags=["Live Activity Devices"])


def _looks_like_job_id(mission_id: str) -> bool:
    """A raw uuid mission name is a JOB card (the widget's own rule:
    ``JobFace.isWorkCard`` — anything without a reserved prefix). Every
    reserved family carries a ``prefix:``; a bare uuid never does."""
    try:
        return str(uuid.UUID(mission_id)) == mission_id.lower()
    except (ValueError, AttributeError, TypeError):
        return False


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
    # AlarmKit observability (2026-07-22 silent-T-0 incident: nothing
    # anywhere could say 'this phone has no device alarm armed').
    # 'authorized' | 'denied' | 'notDetermined' | 'unavailable'.
    alarm_auth: Optional[str] = Field(default=None, max_length=16)
    alarms_armed: Optional[int] = Field(default=None, ge=0, le=1000)

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
            if body.alarm_auth is not None:
                device.alarm_auth = body.alarm_auth
            if body.alarms_armed is not None:
                device.alarms_armed = body.alarms_armed
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
                alarm_auth=body.alarm_auth,
                alarms_armed=body.alarms_armed,
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

    ADOPTION (chat turns and voice calls): the app can start a card
    LOCALLY while foregrounded — the platform never pushed a start, so
    no LiveActivity row exists and every later platform update/end for
    that card would terminate ``no_active_activity``. When no started
    row matches and the mission is a chat turn (``chatturn:`` prefix)
    or a live voice call (``voice:`` prefix), create the row here so
    the platform can drive the locally-started card. The voice case is
    what lets ws_realtime END the island card when the client dies —
    the 2026-08-16 force-quit left "Listening…" on the Lock Screen
    indefinitely precisely because this endpoint dropped the voice
    card's token on the floor. Other mission kinds do NOT adopt —
    their cards are always platform-started."""
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

    # Round 8: JOB cards adopt too. The app starts a job's card locally when
    # it backgrounds mid-job, named after the raw job id ("the platform's
    # mission id for this job" — liveActivityLocal.ios.ts) or, per the
    # updated contract, ``chatjob:<chat_id>``. The platform keys a chat
    # job's pushes ``chatjob:<chat_id>`` and, since Round 8, also delivers
    # them to a card registered under the job id (live_activity_service
    # ``_job_id_of``). Without adoption the local card was invisible here,
    # so it froze at whatever state the app last painted while the platform
    # went on updating a card of its own — the "Live Activity says 1/3, the
    # app says 2/3" disagreement (2026-08-19).
    if not (body.mission_id.startswith("chatturn:")
            or body.mission_id.startswith("voice:")
            or body.mission_id.startswith("chatjob:")
            or _looks_like_job_id(body.mission_id)):
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

    # ONE-ACTIVITY-PER-DEVICE bookkeeping — for CHAT TURNS. A voice call is
    # not a card in that rotation: it coexists with whatever card the device
    # is showing (ActivityKit runs several activities side by side, and the
    # only DB constraint is UNIQUE(device_id, mission_id)), so adopting one
    # must not force-end a live reminder countdown's row (review finding,
    # 2026-08-16). Chat-turn adoption keeps the old semantics: the
    # locally-started card is now THE card on this device. No pushes either
    # way: a preempt end here would race the card the user is looking at;
    # stale platform cards self-heal on their next progress tick.
    if not body.mission_id.startswith("voice:"):
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


class LiveActivityPresence(BaseModel):
    """Is this device SHOWING THE APP right now?

    ``token`` (the device's push-to-start token) scopes the report to
    one device. Without it the report lands on the user's most recently
    seen device, which is the single-phone reality and the same
    heuristic adoption uses; with two devices it matters, because a
    phone in the user's hand must not silence a card on their iPad —
    the iPad IS out of app."""

    foreground: bool
    token: Optional[str] = Field(default=None, min_length=32, max_length=200)


# How long one `foreground: true` report is believed. Refreshed on every
# foreground transition, so the only way to reach the end of it is for the
# app to have gone away without being able to say so (suspended, crashed,
# force-quit). Expiry fails OPEN — cards allowed — which is the direction
# that can only ever cost a card the user did not need, never a reminder
# that never arrived.
PRESENCE_TTL = timedelta(minutes=5)


@router.post("/presence")
async def report_presence(
    body: LiveActivityPresence,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The client-side half of the foreground gate.

    A Live Activity is the agent reaching OUT of the app, and the app is
    the only actor that can say whether it is on screen. Until this
    existed the platform pushed an out-of-app card at a phone whose owner
    was watching the answer stream into the thread — the card that
    appears "while still in the app", and then lingers for its dismiss
    window after they leave.

    Both edges are reported: `false` on the way out, so the gate lifts
    the moment the user leaves rather than after the TTL lapses."""
    user_id = current_user.id
    now = datetime.utcnow()
    q = select(LiveActivityDevice).where(
        LiveActivityDevice.user_id == user_id,
        LiveActivityDevice.revoked_at.is_(None),
    )
    if body.token:
        q = q.where(
            LiveActivityDevice.push_to_start_token == body.token.strip().lower()
        )
    else:
        q = q.order_by(LiveActivityDevice.last_seen_at.desc())
    device = (await db.execute(q)).scalars().first()
    if device is None:
        return {"ok": True, "updated": 0}
    device.foreground_until = (now + PRESENCE_TTL) if body.foreground else None
    device.last_seen_at = now
    await db.commit()
    return {"ok": True, "updated": 1,
            "foreground_until": device.foreground_until.isoformat()
            if device.foreground_until else None}


class LiveActivitiesCleared(BaseModel):
    """The device ended these missions' cards locally.

    ``token`` scopes the report to the reporting DEVICE, and it matters:
    a card is a per-device object, so one phone's sweep must not end the
    row of a card that is still on screen on the user's iPad."""

    mission_ids: List[str] = Field(..., min_length=1, max_length=32)
    token: Optional[str] = Field(default=None, min_length=32, max_length=200)


@router.post("/cleared")
async def report_cleared(
    body: LiveActivitiesCleared,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The app ended these cards itself — a foreground sweep, or a launch
    sweep after a force-quit — and the platform still has them STARTED.

    That is not cosmetic bookkeeping. A started row holds the device's
    one-activity slot (``_preempt_device``, the partial unique index) and
    it is what makes the start-if-missing lane DECLINE to render the
    mission's next alert: the lane only restarts a card it believes is
    gone. So a reminder whose countdown card a force-quit swept would
    reach its fire with the platform still convinced a card was on
    screen, and push an update into an activity that no longer exists.

    Deliberately NOT the tap ack. ``/ack`` and ``/seen`` also SUPPRESS
    the mission's pending completed/failed rows, which is right for "the
    user read it" and exactly wrong here — nobody has seen anything; a
    card was removed from a screen. This ends rows and stops there.

    DB-only: there is no card left to push an end to."""
    user_id = current_user.id
    now = datetime.utcnow()
    missions = [m.strip()[:64] for m in body.mission_ids if m and m.strip()]
    if not missions:
        return {"ok": True, "ended": 0}
    q = select(LiveActivity).where(
        LiveActivity.user_id == user_id,
        LiveActivity.mission_id.in_(missions),
        LiveActivity.status == LA_STARTED,
    )
    if body.token:
        # Device-scoped: a card is a per-device object. Without this, one
        # phone's foreground sweep ends the row of a card still on screen on
        # the user's other device — and the platform then stops updating it.
        device = (
            await db.execute(
                select(LiveActivityDevice).where(
                    LiveActivityDevice.user_id == user_id,
                    LiveActivityDevice.push_to_start_token
                    == body.token.strip().lower(),
                )
            )
        ).scalars().first()
        if device is None:
            return {"ok": True, "ended": 0, "reason": "unknown_device"}
        q = q.where(LiveActivity.device_id == device.id)
    rows = (await db.execute(q)).scalars().all()
    for la in rows:
        la.status = LA_ENDED
        la.ended_at = now
        la.updated_at = now
        # `alarm_owned_at` is deliberately LEFT ALONE. The ack clears it
        # because a post-fire tap consumes the ownership cycle; nothing is
        # consumed here — an AlarmKit alarm that owns this reminder's fire is
        # still armed, and clearing the flag would let the platform's loud
        # restart ring on top of it (the double-alert this device reported
        # ownership to prevent).
    await db.commit()
    if rows:
        logger.info(
            "live-activity cleared: %d row(s) for %d mission(s) (user %s)",
            len(rows), len(missions), user_id[:8],
        )
    return {"ok": True, "ended": len(rows)}


@router.get("/active-missions")
async def list_active_live_activity_missions(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Missions the platform believes have a live (started) card for
    this user. The app's foreground reconcile ends any on-device
    activity whose mission is absent here — the only authoritative
    cleanup for a card whose end push APNs accepted but the device
    never applied (shared-token routing)."""
    missions = (
        await db.execute(
            select(LiveActivity.mission_id)
            .where(
                LiveActivity.user_id == current_user.id,
                LiveActivity.status == LA_STARTED,
            )
            .distinct()
        )
    ).scalars().all()
    return {"missions": list(missions)}


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
    rows are deliberately NOT suppressed — seen is not answered.

    PRE-FIRE EXEMPTION (2026-07-22 founder repro): a reminder's
    countdown card and its fired card carry the SAME deep link, so a
    tap on a still-counting card used to be treated as an alert ack —
    it wiped the pending countdown and the user lost their alert
    surface. When the mission's newest countdown arm
    (mission_started row with ``timer_end_ms``) is still in the
    future, the tap is just navigation: keep the card, suppress
    nothing. Post-fire taps keep full ack semantics (end everywhere +
    suppress queued rings)."""
    user_id = current_user.id
    now = datetime.utcnow()

    if body.mission_id.startswith("reminder:"):
        # Newest countdown arm for THIS mission decides the phase —
        # filtered in SQL (JSON ->> / as_string, portable across the
        # JSONB prod column and the sqlite test variant) so a chatty
        # day's flood of other mission_started rows can never push the
        # arm out of scope.
        newest_arm = (
            await db.execute(
                select(NotificationQueue)
                .where(
                    NotificationQueue.user_id == user_id,
                    NotificationQueue.event_kind == NOTIFY_KIND_MISSION_STARTED,
                    NotificationQueue.data_json["mission_id"].as_string()
                    == body.mission_id,
                )
                .order_by(NotificationQueue.created_at.desc())
                .limit(1)
            )
        ).scalars().first()
        if newest_arm is not None:
            end_ms = (newest_arm.data_json or {}).get("timer_end_ms")
            if (
                isinstance(end_ms, (int, float))
                and end_ms / 1000.0 > now.timestamp()
            ):
                # Still counting down — navigation only.
                return {"ok": True, "ended": 0, "suppressed": 0,
                        "pre_fire": True}

    ended, suppressed = await _end_mission_seen(
        db, user_id, body.mission_id, now, reason="acked",
    )
    await db.commit()
    if ended or suppressed:
        logger.info(
            "live-activity ack: mission %s — ended %d cards, suppressed %d rows",
            body.mission_id, ended, suppressed,
        )
    return {"ok": True, "ended": ended, "suppressed": suppressed}


async def _end_mission_seen(
    db: AsyncSession, user_id: str, mission_id: str, now: datetime, *,
    reason: str,
) -> tuple[int, int]:
    """The SEEN core shared by the tap ack and the in-app seen signal:
    end this mission's started cards on every device (immediate
    dismissal, best-effort push, DB end is the invariant) and suppress
    its still-pending completed/failed/progress rows so an already-seen
    mission never re-alerts. Does NOT commit. Returns (ended,
    suppressed)."""
    rows = (
        await db.execute(
            select(LiveActivity, LiveActivityDevice)
            .join(LiveActivityDevice, LiveActivity.device_id == LiveActivityDevice.id)
            .where(
                LiveActivity.user_id == user_id,
                LiveActivity.mission_id == mission_id,
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
        # A post-fire ack consumes the alarm-ownership cycle too — a
        # stale flag on this row must never mute a future fire of the
        # same mission (the app re-reports on every fresh arming).
        la.alarm_owned_at = None
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
        data = row.data_json or {}
        if data.get("mission_id") != mission_id:
            continue
        # Suppress only rows that are part of an in-progress alert:
        # already due, chained alarm rings (deliberately future-
        # scheduled 20s out — the whole point of the tap ack is
        # stopping them), or a RETRYING row (attempts >= 1: its
        # future scheduled_for is dispatcher backoff, not a schedule —
        # without this, a ring-1 fire row that failed once would
        # survive the ack and loud-restart minutes later). A fresh
        # FUTURE-scheduled ordinary row (attempts 0 — e.g. the next
        # fire of a daily reminder queued by an edit race) survives a
        # tap that only meant 'I saw this one'. Round 3's deferred end
        # rows carry la_only and so count as rings: seen ends the card
        # now, the booked end has nothing left to do.
        is_ring = bool(data.get("la_only")) or (
            isinstance(data.get("realert_seq"), int)
            and data["realert_seq"] >= 2
        )
        due = row.scheduled_for is None or row.scheduled_for <= now
        retrying = (row.attempts or 0) >= 1
        if not (is_ring or due or retrying):
            continue
        row.status = NQ_SUPPRESSED
        row.claimed_at = None
        row.channels_json = {**(row.channels_json or {}),
                             "policy": {"suppressed": reason}}
        suppressed += 1
    return ended, suppressed


class LiveActivitySeen(BaseModel):
    """The app reports the user VIEWED a response in-app (Round 3,
    item 5). ``chat_id`` names the conversation whose job card should
    end (``chatjob:<chat_id>``); ``message_id`` is the response they saw
    (recorded for the trail, not used to gate — seeing the thread is
    seeing the answer); ``mission_id`` optionally names one more card to
    end (the turn's ``chatturn:`` card, whose id the app got in its
    status frames). Only chat-turn cards may be ended this way — never a
    reminder countdown, which has its own ack semantics."""
    chat_id: str = Field(..., min_length=1, max_length=64)
    message_id: Optional[str] = Field(None, max_length=64)
    mission_id: Optional[str] = Field(None, max_length=64)


_SEEN_MISSION_PREFIXES = ("chatjob:", "chatturn:")


@router.post("/seen")
async def seen_live_activity(
    body: LiveActivitySeen,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Dismiss signal (Round 3, item 5): the user viewed the response
    in-app, so the conversation's Live Activity — the ``chatjob:`` card
    every job of that chat shares — ends everywhere (immediate
    dismissal) and its pending completed/progress/deferred-end rows are
    suppressed. Idempotent; a chat with no card returns ended=0."""
    user_id = current_user.id
    now = datetime.utcnow()
    missions = [f"chatjob:{body.chat_id.strip()}"[:64]]
    if body.mission_id:
        mid = body.mission_id.strip()
        if mid.startswith(_SEEN_MISSION_PREFIXES) and mid not in missions:
            missions.append(mid[:64])
    ended = suppressed = 0
    for mid in missions:
        e, s = await _end_mission_seen(db, user_id, mid, now, reason="seen")
        ended += e
        suppressed += s
    await db.commit()
    if ended or suppressed:
        logger.info(
            "live-activity seen: chat %s (message %s) — ended %d cards, "
            "suppressed %d rows",
            body.chat_id[:12], (body.message_id or "-")[:12], ended, suppressed,
        )
    return {"ok": True, "ended": ended, "suppressed": suppressed,
            "missions": missions}


class AlarmOwnedReport(BaseModel):
    """The app armed (or disarmed) a DEVICE alarm — AlarmKit, iOS 26 —
    for this mission and retired the platform countdown card. ``token``
    (the device's push-to-start token) scopes the report to that
    device's rows; without it every row of the mission is marked
    (single-phone reality, and safe: the flag only quiets the fire
    lane's own restart+rings)."""

    mission_id: str = Field(..., min_length=1, max_length=64)
    owned: bool = True
    token: Optional[str] = Field(default=None, min_length=32, max_length=200)


@router.post("/alarm-owned")
async def report_alarm_owned(
    body: AlarmOwnedReport,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Record device-alarm ownership of a mission's fire moment so the
    fire lane skips its loud restart + ring chain on that device
    (AlarmKit already rings through silent/Focus — a platform ring on
    top double-alerts), and — token-scoped, PRE-start — plant a
    STARTED MARKER row so the platform never pushes its own countdown
    card at all (the app arms the AlarmKit card ~1-2s after the
    reminder tool runs; the platform push would arrive 15-40s later
    as a brief duplicate). The marker triples as the REMINDER-WINS
    anchor (chat-turn cards yield to started reminder rows) and is
    consumed by the fire lane's quiet alarm_owned end.

    Rules (2026-07-23 design review):
    - Marker inserts/revivals happen ONLY with a device token. A
      tokenless report stays flag-only forever — external-channel
      pocket reminders depend on the platform card as their WAKE
      vector, and a blind marker would silence it.
    - The device's one-started-row slot is never fought over: if some
      OTHER mission's card is live, the report stays flag-only and the
      normal pipeline (platform start preempts, countdown-armed lane
      retires + re-reports) converges one dispatcher tick later.
    - owned=false ends a STARTED row (no push — no on-device card
      exists in any owned=false scenario): a phantom marker would
      otherwise squat the unique slot and keep chat turns yielding to
      nothing."""
    user_id = current_user.id
    now = datetime.utcnow()

    device: Optional[LiveActivityDevice] = None
    if body.token:
        device = (
            await db.execute(
                select(LiveActivityDevice).where(
                    LiveActivityDevice.user_id == user_id,
                    LiveActivityDevice.push_to_start_token
                    == body.token.strip().lower(),
                )
            )
        ).scalar_one_or_none()
        if device is None:
            return {"ok": True, "updated": 0}

    stmt = select(LiveActivity).where(
        LiveActivity.user_id == user_id,
        LiveActivity.mission_id == body.mission_id,
    )
    if device is not None:
        stmt = stmt.where(LiveActivity.device_id == device.id)
    rows = (await db.execute(stmt)).scalars().all()

    updated = 0
    if not body.owned:
        for la in rows:
            la.alarm_owned_at = None
            if la.status == LA_STARTED:
                la.status = LA_ENDED
                la.ended_at = now
            la.updated_at = now
            updated += 1
        if updated:
            await db.commit()
        return {"ok": True, "updated": updated}

    for la in rows:
        la.alarm_owned_at = now
        la.updated_at = now
        updated += 1

    marker = None
    if device is not None:
        other_started = (
            await db.execute(
                select(LiveActivity.id).where(
                    LiveActivity.device_id == device.id,
                    LiveActivity.status == LA_STARTED,
                    LiveActivity.mission_id != body.mission_id,
                ).limit(1)
            )
        ).scalar_one_or_none() is not None
        mine = rows[0] if rows else None
        if mine is not None and mine.status == LA_ENDED and not other_started:
            # Revive as the marker (reuse-row pattern of _send_start):
            # the countdown is device-owned again for a fresh cycle.
            mine.status = LA_STARTED
            mine.started_at = now
            mine.ended_at = None
            mine.activity_push_token = None
            marker = "revived"
        elif mine is None and not other_started:
            db.add(LiveActivity(
                id=str(uuid.uuid4()),
                user_id=user_id,
                mission_id=body.mission_id,
                device_id=device.id,
                apns_environment=device.apns_environment,
                status=LA_STARTED,
                alarm_owned_at=now,
                started_at=now,
                updated_at=now,
            ))
            marker = "inserted"
            updated += 1
        elif other_started:
            # A live card owns the slot (e.g. a chatturn working card).
            # Flag-only: the platform's own start will preempt it, the
            # countdown-armed lane retires + re-reports, and the flag
            # lands on the real row — one brief card flash, accepted.
            marker = "skipped"

    try:
        await db.commit()
    except IntegrityError:
        # uq_live_activities_device_started: a concurrent start won the
        # slot between our check and the commit. Flag-only is the safe
        # outcome; the countdown-armed retirement lane converges it.
        await db.rollback()
        return {"ok": True, "updated": 0, "marker": "raced"}
    if updated or marker:
        logger.info(
            "live-activity alarm-owned: mission %s owned=%s — %d row(s), marker=%s",
            body.mission_id, body.owned, updated, marker,
        )
    out: dict = {"ok": True, "updated": updated}
    if marker:
        out["marker"] = marker
    return out


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
