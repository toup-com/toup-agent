"""Live Activity lane of the notification dispatcher.

Maps background-task notification rows (Autopilot missions AND quick
spawned jobs — anything carrying ``data.mission_id``) onto APNs Live
Activity pushes:

    mission_started   → push-to-start on every registered device
                        (banner alert rides in the start payload)
    progress          → content-state update, priority 5 (unbudgeted);
                        if the card is gone (preempted / device reboot)
                        it is SILENTLY re-started — self-healing
    needs_input /
    needs_approval    → content-state update + alert, priority 10
                        (activity stays alive — the task is waiting)
    mission_completed /
    mission_failed    → alerting UPDATE (banner + sound on Apple's
                        documented alert surface) then a bannerless
                        event=end with the final content-state; card
                        lingers on the lock screen (default 4h, or
                        ``data.dismiss_after_s`` when the producer
                        wants a shorter linger, e.g. quick jobs).
                        When the card had to be restarted first, the
                        start itself carries the alert (loud restart)
                        and both follow-ups go bannerless — exactly
                        one audible alert per terminal row.
                        ALARM-CLASS rows (``data.sound``/``data.alarm``
                        — reminder fires) ring up to _ALARM_MAX_RINGS
                        times: each ring but the last leaves the card
                        open and chains an LA-only follow-up row one
                        gap out; the tap ack ends the card and
                        suppresses queued rings; the last ring closes
                        the card. Alert sounds are ALWAYS the system
                        default tone — iOS has never honored a named
                        sound on a Live Activity push alert (silence,
                        not fallback).

ONE ACTIVITY PER DEVICE — the load-bearing correctness rule. Apple
leaves the behavior of multiple concurrent activities sharing one
push-to-start token UNDEFINED (the ``input-push-token`` alias is
"initially" only; updates/ends to a shared token may hit the wrong
card or silently no-op while APNs returns 200 — see the 2026-07-10
investigation). So: starting a new activity first ENDS any other
platform-driven activity on that device (immediate dismissal), and
when a tokenless send finds more than one LA_STARTED row for a
device the lane RESOLVES the ambiguity by preempting every other
row first (end push + DB force-end) instead of skipping — a skip
would repeat forever (rows are never mutated) and the alert would
retry to failure (2026-07-16 founder incident: a replica race left
two started rows and every terminal send for that device was lost).
Preempted missions self-heal: the next progress heartbeat silently
restarts their card. Per-activity tokens reported by the app are
always preferred when present.

Progress bars: discrete ``data.progress`` (0-100) for missions; for
bounded quick jobs the producer sends ``data.timer_end_ms`` instead
and the widget animates the bar on-device with zero pushes.

Every function returns a channels_json fragment — the dispatcher
records it verbatim, so a production incident can be read straight
off the notification row.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED,
    AgentConfig, LiveActivity, LiveActivityDevice, NotificationQueue,
    NOTIFY_KIND_MISSION_COMPLETED, NOTIFY_KIND_MISSION_FAILED,
    NOTIFY_KIND_MISSION_STARTED, NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_NEEDS_INPUT, NOTIFY_KIND_PROGRESS,
)
from app.services import apns_push

logger = logging.getLogger(__name__)

# Kinds this lane knows how to render on the activity.
LIVE_ACTIVITY_KINDS = {
    NOTIFY_KIND_MISSION_STARTED,
    NOTIFY_KIND_PROGRESS,
    NOTIFY_KIND_NEEDS_INPUT,
    NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_MISSION_COMPLETED,
    NOTIFY_KIND_MISSION_FAILED,
}


# Alert kinds that restart a missing card before delivering (mirrors
# the progress self-heal): without this, a terminal/needs-* row for a
# never-started or preempted card terminated 'no_active_activity' and
# the alert was silently lost — exactly when it mattered most (LA is
# the ONLY iOS push surface while expo-notifications isn't installed).
_START_IF_MISSING_KINDS = {
    NOTIFY_KIND_NEEDS_INPUT,
    NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_MISSION_COMPLETED,
    NOTIFY_KIND_MISSION_FAILED,
}


def live_activity_ready() -> bool:
    return settings.live_activity_enabled and apns_push.apns_configured()


# Apple hard-caps Live Activities at 8 hours — a started row older
# than that is guaranteed dead on-device, so ending it is bookkeeping
# only (DB-only, zero pushes). Without this sweep a stale row keeps
# counting against the one-activity-per-device invariant forever.
_STALE_ACTIVITY_MAX_AGE = timedelta(hours=8)

# Chat turns are minutes-long; their cards carry a 10-minute local
# staleDate. A chatturn row still started after this is a wedge.
_TURN_ACTIVITY_MAX_AGE = timedelta(minutes=30)


async def sweep_stale_activities(
    db, now: datetime, user_id: Optional[str] = None,
) -> int:
    """End every LA_STARTED row older than Apple's 8h hard cap.

    DB-only — the on-device card is already gone, a push would be
    wasted (and on a rotated token, misdirected). Deliberately does
    NOT filter on activity_push_token: tokenful rows die at 8h too.
    Returns the number of rows swept."""
    stmt = (
        update(LiveActivity)
        .where(
            LiveActivity.status == LA_STARTED,
            LiveActivity.started_at < now - _STALE_ACTIVITY_MAX_AGE,
        )
        .values(status=LA_ENDED, ended_at=now, updated_at=now)
    )
    if user_id is not None:
        stmt = stmt.where(LiveActivity.user_id == user_id)
    result = await db.execute(stmt)
    swept = result.rowcount or 0
    # Chat-turn rows age out far sooner: a turn lives minutes and its
    # local card carries a 10-min staleDate — a chatturn row still
    # 'started' after 30 min is a wedge (2026-07-18: an adopted row
    # for an already-answered turn squatted the one-per-device slot
    # for 8h and every reminder fire fought it).
    turn_stmt = (
        update(LiveActivity)
        .where(
            LiveActivity.status == LA_STARTED,
            LiveActivity.mission_id.like("chatturn:%"),
            LiveActivity.started_at < now - _TURN_ACTIVITY_MAX_AGE,
        )
        .values(status=LA_ENDED, ended_at=now, updated_at=now)
    )
    if user_id is not None:
        turn_stmt = turn_stmt.where(LiveActivity.user_id == user_id)
    turn_result = await db.execute(turn_stmt)
    swept += turn_result.rowcount or 0
    if swept:
        await db.commit()
        logger.info(
            "live-activity GC: ended %d stale started rows%s",
            swept, f" for user {user_id[:8]}" if user_id else "",
        )
    return swept


# ── Row field extraction ──────────────────────────────────────────


def _mission_id(row: NotificationQueue) -> Optional[str]:
    mid = (row.data_json or {}).get("mission_id")
    if isinstance(mid, str) and mid:
        return mid[:64]
    return None


def _progress_fraction(row: NotificationQueue) -> Optional[float]:
    raw = (row.data_json or {}).get("progress")
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw) / 100.0))
    return None


def _timer_end_ms(row: NotificationQueue) -> Optional[int]:
    raw = (row.data_json or {}).get("timer_end_ms")
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw)
    return None


def _dismissal_date(row: NotificationQueue, now: datetime) -> Optional[int]:
    raw = (row.data_json or {}).get("dismiss_after_s")
    if isinstance(raw, (int, float)) and raw >= 0:
        return int(now.timestamp()) + int(raw)
    return None


# ── Alarm-class rows ──────────────────────────────────────────────
# data.sound (reminder fires send 'toup_alarm.caf') marks a row
# ALARM-CLASS; the named file itself is NEVER put on the wire. iOS
# has never honored a custom named sound on an ActivityKit push
# alert — the result is total silence, not a fallback to default
# (verified on the founder's device 2026-07-20/21 with the file in
# both the app and widget-extension bundles; Apple forums thread
# 718659 reports exactly this since Oct 2022, unanswered, and every
# production payload ever verified audible uses "default"). So the
# alert always plays the system default tone, and alarm-class rows
# compensate by RE-RINGING: each ring but the last leaves the card
# open and chains an LA-only follow-up row one gap out; the tap ack
# (live_activity_devices /ack) ends the card and suppresses the
# queued rings, and the last ring closes the card itself. True alarm
# audio that breaks the silent switch/Focus is out of reach for Live
# Activities BY DESIGN — that tier is AlarmKit (iOS 26), a separate
# app-side project.
_ALARM_MAX_RINGS = 3
_ALARM_RING_GAP = timedelta(seconds=20)


def _is_alarm_row(row: NotificationQueue) -> bool:
    data = row.data_json or {}
    return bool(data.get("sound") or data.get("alarm"))


def _realert_seq(row: NotificationQueue) -> int:
    """1 on the original terminal row; 2.. on chained ring rows."""
    raw = (row.data_json or {}).get("realert_seq")
    if isinstance(raw, int) and raw >= 1:
        return raw
    return 1


def _alarm_rings(row: NotificationQueue) -> int:
    """Total rings for this row's mission: alarm-class rows ring
    _ALARM_MAX_RINGS times, everything else exactly once."""
    return _ALARM_MAX_RINGS if _is_alarm_row(row) else 1


async def _enqueue_next_ring(
    db, row: NotificationQueue, mission_id: str, next_seq: int, now: datetime,
) -> bool:
    """Chain the next alarm ring as its own queue row: LA-only,
    scheduled one gap out, idempotent per (mission, seq) so a partial
    -failure retry of this row can never double-book a ring. The
    caller's end-of-lane commit persists it."""
    idem = f"alarm-ring:{mission_id}:{next_seq}"
    existing = await db.execute(
        select(NotificationQueue.id).where(
            NotificationQueue.user_id == row.user_id,
            NotificationQueue.idempotency_key == idem,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return False
    data = dict(row.data_json or {})
    data.update({
        "realert_seq": next_seq,
        # Re-rings only re-sound the card — the first ring already
        # carried the Expo copy and the agent's own channel fan-out.
        "la_only": True,
        "no_agent_fallback": True,
    })
    db.add(NotificationQueue(
        id=str(uuid.uuid4()),
        user_id=row.user_id,
        source="platform",
        event_kind=row.event_kind,
        title=row.title,
        body=row.body,
        data_json=data,
        priority=row.priority,
        scheduled_for=now + _ALARM_RING_GAP,
        idempotency_key=idem,
    ))
    return True


def _mission_title(row: NotificationQueue) -> str:
    title = (row.data_json or {}).get("mission_title")
    if isinstance(title, str) and title.strip():
        return title.strip()[:80]
    return (row.title or "Background task")[:80]


# ── DB helpers ────────────────────────────────────────────────────


async def _active_devices(db, user_id: str) -> List[LiveActivityDevice]:
    result = await db.execute(
        select(LiveActivityDevice)
        .where(
            LiveActivityDevice.user_id == user_id,
            LiveActivityDevice.revoked_at.is_(None),
        )
        .order_by(LiveActivityDevice.last_seen_at.desc())
    )
    return list(result.scalars().all())


async def _revoke_device(db, device: LiveActivityDevice, now: datetime, reason: str) -> None:
    if device.revoked_at is None:
        device.revoked_at = now
        logger.info("live-activity device %s revoked (%s)", device.id, reason)


async def _started_rows_for_device(db, device_id: str) -> List[LiveActivity]:
    result = await db.execute(
        select(LiveActivity).where(
            LiveActivity.device_id == device_id,
            LiveActivity.status == LA_STARTED,
        )
    )
    return list(result.scalars().all())


async def _activities_for_mission(
    db, user_id: str, mission_id: str,
) -> List[Tuple[LiveActivity, LiveActivityDevice]]:
    result = await db.execute(
        select(LiveActivity, LiveActivityDevice)
        .join(LiveActivityDevice, LiveActivity.device_id == LiveActivityDevice.id)
        .where(
            LiveActivity.user_id == user_id,
            LiveActivity.mission_id == mission_id,
            LiveActivity.status == LA_STARTED,
        )
    )
    return [(la, dev) for la, dev in result.all()]


async def _row_for(db, device_id: str, mission_id: str) -> Optional[LiveActivity]:
    result = await db.execute(
        select(LiveActivity).where(
            LiveActivity.device_id == device_id,
            LiveActivity.mission_id == mission_id,
        )
    )
    return result.scalar_one_or_none()


# ── APNs send primitives ──────────────────────────────────────────


async def _send_with_env_selfheal(
    db, device: LiveActivityDevice, token: str, payload: Dict[str, Any],
    *, priority: int, environment: Optional[str] = None,
) -> Tuple[int, str, str]:
    """Send; on a dead-token verdict retry ONCE on the flipped APNs
    environment before letting the caller declare the token dead.

    APNs returns BadDeviceToken both for genuinely dead tokens AND for
    environment mismatches — indistinguishable from here. The mismatch
    is real and easy to hit: a dev-provisioned build whose JS runs in
    Release mode registers itself as 'production' while its
    aps-environment entitlement is sandbox (seen live 2026-07-16 — the
    founder's device got revoked and every card went silent). Success
    on the flipped host self-heals the DEVICE row so subsequent sends
    go straight to the right environment.

    Returns (status, reason, environment_used).
    """
    env = environment or device.apns_environment
    status, reason = await apns_push.send_live_activity(
        token, payload, environment=env, priority=priority,
    )
    if not apns_push.is_token_dead(status, reason):
        return status, reason, env
    flipped = "production" if env == "development" else "development"
    status2, reason2 = await apns_push.send_live_activity(
        token, payload, environment=flipped, priority=priority,
    )
    if status2 == 200:
        logger.info(
            "live-activity env self-heal: device %s %s → %s",
            device.id, env, flipped,
        )
        device.apns_environment = flipped
        return status2, reason2, flipped
    # Dead on both hosts — the original verdict stands.
    return status, reason, env


async def _preempt_device(
    db, device: LiveActivityDevice, keep_mission_id: str, now: datetime,
) -> int:
    """ONE-ACTIVITY-PER-DEVICE enforcement: end (immediate dismissal,
    no alert) every other started activity on this device before a new
    one starts (or before an unambiguous shared-token send). The DB
    force-end is unconditional — even when APNs rejects the end push
    the row must stop claiming the device. Ends ride the env-self-heal
    path so a stale-environment row's end still lands on the flipped
    host. Returns the number of rows preempted."""
    preempted = 0
    for la in await _started_rows_for_device(db, device.id):
        if la.mission_id == keep_mission_id:
            continue
        token = la.activity_push_token or device.push_to_start_token
        payload = apns_push.build_end_payload(
            title="Superseded",
            dismissal_date=int(now.timestamp()) - 1,
            timestamp=int(now.timestamp()),
        )
        status, reason, _env = await _send_with_env_selfheal(
            db, device, token, payload,
            priority=10, environment=la.apns_environment,
        )
        la.status = LA_ENDED
        la.ended_at = now
        la.updated_at = now
        preempted += 1
        if status != 200:
            logger.info(
                "live-activity preempt end for %s/%s: %s %s",
                device.id, la.mission_id, status, reason,
            )
    if preempted:
        # The session runs with autoflush off (pgbouncer setup): flush
        # the force-ends NOW so (a) a follow-up started-rows SELECT in
        # the same unit of work can't see the preempted rows and
        # preempt them twice, and (b) a subsequent INSERT of a started
        # row can't race the partial unique index ahead of these
        # UPDATEs (SQLAlchemy flushes INSERTs before UPDATEs).
        await db.flush()
    return preempted


async def _user_orb_color(db, user_id: str) -> Optional[str]:
    """The user's live agent color (agent_configs.agent_color) — the
    same source of truth the in-app orb renders, NOT the stale
    bind-time runtime.json snapshot. None when unset (widget falls
    back to the brand default)."""
    return (
        await db.execute(
            select(AgentConfig.agent_color).where(AgentConfig.user_id == user_id)
        )
    ).scalar_one_or_none()


async def _send_start(
    db, device: LiveActivityDevice, row: NotificationQueue,
    mission_id: str, now: datetime, *, silent: bool = False,
) -> Dict[str, Any]:
    await _preempt_device(db, device, mission_id, now)

    progress = _progress_fraction(row)
    data = row.data_json or {}
    # Card-tap target: chat turns land in the conversation where the
    # answer lives; everything else keeps Mission Control. The mission
    # id rides as a query param so the app can ACK the tap (end the
    # card everywhere + stop re-alerts for this mission).
    base_link = "toup://chat" if data.get("route") == "chat" else "toup://mission-control"
    deep_link = f"{base_link}?mission={mission_id}"
    # Stale backstop: a countdown card goes visually stale 2 minutes
    # after its timer fires; a non-timer card after 30 minutes with no
    # update. Progress updates refresh the horizon — only a card whose
    # pushes are LOST can ever dim.
    timer_ms = _timer_end_ms(row)
    stale_ts = (
        int(timer_ms / 1000) + 120 if timer_ms
        else int(now.timestamp()) + 1800
    )
    # Producer subtitle override (e.g. reminder countdowns carry the
    # reminder text); non-silent starts keep the legacy 'Starting…'.
    subtitle_override = data.get("subtitle") if isinstance(data.get("subtitle"), str) else None
    timer_type = data.get("timer_type")
    payload = apns_push.build_start_payload(
        mission_id=mission_id,
        title=_mission_title(row),
        subtitle=(row.body or "Working…")[:120] if silent
                 else (subtitle_override or "Starting…")[:120],
        progress=progress if progress is not None else 0.0,
        timer_end_ms=timer_ms,
        alert_title=None if silent else row.title,
        alert_body=None if silent else row.body,
        timestamp=int(now.timestamp()),
        deep_link=deep_link,
        timer_type=timer_type if timer_type in ("circular", "digital") else None,
        orb_color=await _user_orb_color(db, row.user_id),
        stale_date=stale_ts,
    )
    status, reason, _env = await _send_with_env_selfheal(
        db, device, device.push_to_start_token, payload, priority=10,
    )
    if status == 200:
        existing = await _row_for(db, device.id, mission_id)
        if existing is not None:
            # Re-start after preemption/reboot: reuse the row (UNIQUE
            # (device_id, mission_id)) — the old activity is gone, so
            # its reported token is stale too.
            existing.status = LA_STARTED
            existing.activity_push_token = None
            existing.started_at = now
            existing.updated_at = now
            existing.ended_at = None
            if progress is not None:
                existing.last_progress = int(progress * 100)
        else:
            db.add(LiveActivity(
                id=str(uuid.uuid4()),
                user_id=row.user_id,
                mission_id=mission_id,
                device_id=device.id,
                apns_environment=device.apns_environment,
                status=LA_STARTED,
                last_progress=int((progress or 0.0) * 100),
                started_at=now,
            ))
        return {"status": "ok"}
    if apns_push.is_token_dead(status, reason):
        await _revoke_device(db, device, now, reason or f"http_{status}")
    return {"status": "error", "http": status, "reason": reason}


async def _send_to_activity(
    db, la: LiveActivity, device: LiveActivityDevice,
    payload: Dict[str, Any], *, priority: int, now: datetime,
    end: bool = False,
) -> Dict[str, Any]:
    token = la.activity_push_token
    preempted = 0
    if not token:
        # Shared-token fallback is only safe while it is unambiguous —
        # Apple's routing with 2+ activities on one token is undefined.
        # RESOLVE the ambiguity instead of skipping (a skip repeats
        # forever: nothing mutates the rows, so the queue row retries
        # to failure and the alert is lost — 2026-07-16 incident).
        # _preempt_device ends every OTHER started row on this device
        # (end push + unconditional DB force-end), after which the
        # shared push-to-start token is unambiguous again.
        started = await _started_rows_for_device(db, device.id)
        if len(started) > 1:
            preempted = await _preempt_device(db, device, la.mission_id, now)
        token = device.push_to_start_token

    status, reason, env_used = await _send_with_env_selfheal(
        db, device, token, payload,
        priority=priority, environment=la.apns_environment,
    )
    if status == 200:
        if env_used != la.apns_environment:
            la.apns_environment = env_used
        la.updated_at = now
        if end:
            la.status = LA_ENDED
            la.ended_at = now
        out: Dict[str, Any] = {"status": "ok"}
        if preempted:
            out["preempted"] = preempted  # ladebug: ambiguity was resolved here
        return out
    if apns_push.is_token_dead(status, reason):
        if la.activity_push_token:
            # The reported per-activity token went stale — drop it so
            # the next update falls back to the push-to-start token.
            la.activity_push_token = None
        else:
            await _revoke_device(db, device, now, reason or f"http_{status}")
        if end:
            # Terminal transition on a dead token: close the row out —
            # there is nothing left to update.
            la.status = LA_ENDED
            la.ended_at = now
    err: Dict[str, Any] = {"status": "error", "http": status, "reason": reason}
    if preempted:
        err["preempted"] = preempted
    return err


# ── Dispatcher entry point ────────────────────────────────────────


async def handle_notification_row(
    db, row: NotificationQueue, now: datetime,
) -> Optional[Dict[str, Any]]:
    """→ channels_json fragment, or None when the lane doesn't apply
    (non-mission row). ``delivered`` in the fragment is True when APNs
    accepted the push for at least one device."""
    if row.event_kind not in LIVE_ACTIVITY_KINDS:
        return None
    mission_id = _mission_id(row)
    if not mission_id:
        return None
    if not live_activity_ready():
        return {"status": "skipped", "reason": "apns_not_configured"}

    # Staleness GC at lane entry: rows past Apple's 8h hard cap are
    # dead on-device — end them (DB-only) before they can masquerade
    # as a second started activity and trip the preemption path.
    await sweep_stale_activities(db, now, user_id=row.user_id)

    per_device: Dict[str, Any] = {}
    delivered = False
    errored = False

    if row.event_kind == NOTIFY_KIND_MISSION_STARTED:
        # data.silent → card appears without a banner (reminder
        # countdown arms: the user just got chat confirmation — the
        # card IS the feedback).
        start_silent = bool((row.data_json or {}).get("silent"))
        devices = await _active_devices(db, row.user_id)
        if not devices:
            return {"status": "skipped", "reason": "no_live_activity_devices"}
        is_chat_turn = (row.data_json or {}).get("kind") == "chat_turn"
        for device in devices:
            started = await _started_rows_for_device(db, device.id)
            if any(la.mission_id == mission_id for la in started):
                # at-least-once retry after a partial failure — a second
                # start push would spawn a duplicate card on screen.
                per_device[device.id] = {"status": "skipped", "reason": "already_started"}
                continue
            if is_chat_turn and any(
                la.mission_id.startswith("reminder:") for la in started
            ):
                # REMINDER WINS (founder rule 2026-07-20): a working
                # chat-turn card must never preempt a live countdown —
                # the preempt end rides the shared token, routinely
                # no-ops on-device, and the fire's restart then stacks
                # a duplicate card. The answer still arrives as its
                # own banner; only the ambient working card yields.
                per_device[device.id] = {"status": "skipped", "reason": "yields_to_reminder"}
                continue
            result = await _send_start(db, device, row, mission_id, now, silent=start_silent)
            per_device[device.id] = result
            delivered = delivered or result["status"] == "ok"
            errored = errored or result["status"] == "error"
        try:
            await db.commit()
        except IntegrityError:
            # uq_live_activities_device_started: the OTHER replica won
            # the start race for this device. Roll back and report an
            # error — the queue row retries on backoff and next time
            # sees the winner's row (already_started dedup).
            await db.rollback()
            logger.info(
                "live-activity start lost replica race for mission %s", mission_id,
            )
            return {"status": "error", "delivered": False,
                    "reason": "device_started_race", "devices": per_device}
        status = "ok" if delivered else ("error" if errored else "skipped")
        return {"status": status, "delivered": delivered, "devices": per_device}

    activities = await _activities_for_mission(db, row.user_id, mission_id)
    restarted = False
    restart_loud = False

    if not activities and (
        (
            row.event_kind == NOTIFY_KIND_PROGRESS
            # update_only progress (turn/job status beacons) refreshes
            # an existing card and must NEVER start one — that is what
            # lets producers emit on every turn without growing cards
            # on ordinary foreground turns.
            and not (row.data_json or {}).get("update_only")
        )
        or (
            row.event_kind in _START_IF_MISSING_KINDS
            and not (row.data_json or {}).get("silent")
            # A re-ring never resurrects a card: ring 1 restarts (a
            # fire must always alert), but if the card is gone by
            # ring 2+ the user closed or acked it — silence is the
            # correct outcome, not a fresh card.
            and _realert_seq(row) == 1
        )
    ):
        # Self-healing: the card was preempted by a newer task (or the
        # device rebooted / was never started because the producer only
        # emits terminal kinds — foregrounded chat turns, reminder
        # fires whose countdown never armed). Progress rows restart
        # silently. Alert kinds restart LOUD: the start alert (with
        # sound) is the one Live Activity surface iOS 26 provably
        # renders — it mandates an alert config on every start — while
        # an alert riding the follow-up end travels Apple's least-
        # documented surface (end-event alert over the shared
        # push-to-start token) and reached the founder's phone silently
        # or not at all (2026-07-18: every backgrounded fire). The
        # follow-up update/end then goes bannerless — one alert, on
        # the surface that works. Silent terminal rows (reminder
        # cancel ends) skip the restart: there is nothing to alert,
        # an end for a card that isn't there is a no-op.
        restart_loud = row.event_kind != NOTIFY_KIND_PROGRESS
        devices = await _active_devices(db, row.user_id)
        if not devices:
            return {"status": "skipped", "reason": "no_live_activity_devices"}
        for device in devices:
            result = await _send_start(
                db, device, row, mission_id, now, silent=not restart_loud,
            )
            per_device[device.id] = result
            delivered = delivered or result["status"] == "ok"
            errored = errored or result["status"] == "error"
        try:
            await db.commit()
        except IntegrityError:
            # Same replica race as the mission_started branch — the
            # silent restart lost; retry via the normal backoff path.
            await db.rollback()
            logger.info(
                "live-activity restart lost replica race for mission %s", mission_id,
            )
            return {"status": "error", "delivered": False,
                    "reason": "device_started_race", "devices": per_device}
        if row.event_kind == NOTIFY_KIND_PROGRESS:
            status = "ok" if delivered else ("error" if errored else "skipped")
            return {"status": status, "delivered": delivered,
                    "restarted": True, "devices": per_device}
        # Alert kinds fall through to the normal update/end loop on the
        # freshly started card.
        restarted = True
        delivered = False
        activities = await _activities_for_mission(db, row.user_id, mission_id)

    if not activities:
        return {"status": "skipped", "reason": "no_active_activity"}

    progress = _progress_fraction(row)
    title = _mission_title(row)

    for la, device in activities:
        # Never move the on-screen bar backwards on reordered rows.
        effective = progress
        if effective is not None and la.last_progress is not None:
            effective = max(effective, la.last_progress / 100.0)

        if row.event_kind == NOTIFY_KIND_PROGRESS:
            headline = (row.body or row.title or "Working…")[:120]
            payload = apns_push.build_update_payload(
                title=title, subtitle=headline, progress=effective,
                timer_end_ms=_timer_end_ms(row),
                stale_date=int(now.timestamp()) + 1800,
                timestamp=int(now.timestamp()),
            )
            result = await _send_to_activity(
                db, la, device, payload, priority=5, now=now,
            )
        elif row.event_kind in (NOTIFY_KIND_NEEDS_INPUT, NOTIFY_KIND_NEEDS_APPROVAL):
            payload = apns_push.build_update_payload(
                title=title, subtitle="Needs your answer",
                progress=effective,
                # After a loud restart the start already delivered the
                # banner+sound — a second alert here would double-bang.
                alert_title=None if restart_loud else row.title,
                alert_body=None if restart_loud else row.body,
                timestamp=int(now.timestamp()),
            )
            result = await _send_to_activity(
                db, la, device, payload, priority=10, now=now,
            )
        else:  # mission_completed | mission_failed
            done = row.event_kind == NOTIFY_KIND_MISSION_COMPLETED
            _data = row.data_json or {}
            # data.silent → silent end (reminder cancel/delete: the card
            # just vanishes, no banner). data.subtitle → producer text
            # on the final card (reminder fires show the reminder text
            # instead of 'Completed ✓').
            _silent_end = bool(_data.get("silent"))
            _subtitle = _data.get("subtitle") if isinstance(_data.get("subtitle"), str) else None
            final_subtitle = _subtitle or ("Completed ✓" if done else "Stopped — needs attention")
            final_progress = 1.0 if done else effective
            # Alarm-class rows ring more than once: every ring except
            # the last is an alerting UPDATE that leaves the card open
            # (already showing the fired state) for the next chained
            # ring row — or for the tap ack, which ends the card and
            # suppresses the remaining rings. Only the last ring
            # closes the card.
            _last_ring = _silent_end or _realert_seq(row) >= _alarm_rings(row)
            # The audible banner rides an alerting UPDATE (Apple's
            # documented alert surface — the end-event alert is
            # undocumented and proved unreliable on iOS 26), then a
            # bannerless end closes the card. Skip the update when the
            # loud restart already alerted, or the producer asked for
            # silence.
            alerted = restart_loud
            upd_result: Optional[Dict[str, Any]] = None
            if not _silent_end and not alerted:
                upd = apns_push.build_update_payload(
                    title=title, subtitle=final_subtitle,
                    progress=final_progress,
                    alert_title=row.title, alert_body=row.body,
                    timestamp=int(now.timestamp()),
                )
                upd_result = await _send_to_activity(
                    db, la, device, upd, priority=10, now=now,
                )
                alerted = upd_result["status"] == "ok"
            if not _last_ring:
                # Loud restart counts as this ring's bang: the card is
                # freshly started in fired state, nothing else to send.
                result = upd_result or {"status": "ok", "reason": "restart_alerted"}
            else:
                payload = apns_push.build_end_payload(
                    title=title,
                    subtitle=final_subtitle,
                    progress=final_progress,
                    # Fallback only: if neither the loud start nor the
                    # alerting update got through, the end keeps the alert
                    # rather than closing the card silently.
                    alert_title=None if (_silent_end or alerted) else row.title,
                    alert_body=None if (_silent_end or alerted) else row.body,
                    # Producer override, else 1h — a finished card must
                    # never linger the full system default 4h.
                    dismissal_date=_dismissal_date(row, now)
                    or int(now.timestamp()) + 3600,
                    timestamp=int(now.timestamp()),
                )
                result = await _send_to_activity(
                    db, la, device, payload, priority=10, now=now, end=True,
                )
                if upd_result and upd_result.get("preempted"):
                    # Ambiguity was resolved during the alerting update —
                    # keep the marker on the recorded fragment (ladebug).
                    result.setdefault("preempted", upd_result["preempted"])
                if result["status"] != "ok" and alerted:
                    # The banner reached the device even though the close
                    # didn't — count the row delivered (the alert is the
                    # job; the card close is hygiene), and close the DB row
                    # so it can't squat the one-activity-per-device slot.
                    la.status = LA_ENDED
                    la.ended_at = now
                    la.updated_at = now
                    result = {**result, "alerted": True}
                    delivered = True

        if result["status"] == "ok" and effective is not None:
            la.last_progress = int(effective * 100)
        per_device[device.id] = result
        delivered = delivered or result["status"] == "ok"

    # Alarm chain: once THIS ring reached a device, book the next one.
    # Gated on delivered so a failed ring retries via the row's own
    # backoff instead of forking the chain; the idempotency key makes
    # the booking safe to repeat on partial-failure retries.
    if (
        delivered
        and row.event_kind in (NOTIFY_KIND_MISSION_COMPLETED,
                               NOTIFY_KIND_MISSION_FAILED)
        and _is_alarm_row(row)
        and not bool((row.data_json or {}).get("silent"))
        and _realert_seq(row) < _alarm_rings(row)
    ):
        await _enqueue_next_ring(db, row, mission_id, _realert_seq(row) + 1, now)

    try:
        await db.commit()
    except IntegrityError:
        # Same replica race as the start/restart commit sites — roll
        # back and let the queue row retry on the normal backoff. A
        # bare raise here strands the row in 'sending' (2026-07-18).
        await db.rollback()
        logger.info(
            "live-activity update/end lost replica race for mission %s", mission_id,
        )
        return {"status": "error", "delivered": False,
                "reason": "device_started_race", "devices": per_device}
    frag: Dict[str, Any] = {"status": "ok" if delivered else "error",
                            "delivered": delivered, "devices": per_device}
    if restarted:
        frag["restarted"] = True
    return frag
