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
    mission_failed    → event=end + final content-state + alert;
                        card lingers on the lock screen (default 4h,
                        or ``data.dismiss_after_s`` when the producer
                        wants a shorter linger, e.g. quick jobs)

ONE ACTIVITY PER DEVICE — the load-bearing correctness rule. Apple
leaves the behavior of multiple concurrent activities sharing one
push-to-start token UNDEFINED (the ``input-push-token`` alias is
"initially" only; updates/ends to a shared token may hit the wrong
card or silently no-op while APNs returns 200 — see the 2026-07-10
investigation). So: starting a new activity first ENDS any other
platform-driven activity on that device (immediate dismissal), and
the shared-token fallback is refused outright whenever more than one
LA_STARTED row exists for a device. Preempted missions self-heal: the
next progress heartbeat silently restarts their card. Per-activity
tokens reported by the app are always preferred when present.

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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED,
    LiveActivity, LiveActivityDevice, NotificationQueue,
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


def live_activity_ready() -> bool:
    return settings.live_activity_enabled and apns_push.apns_configured()


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
) -> None:
    """ONE-ACTIVITY-PER-DEVICE enforcement: end (immediate dismissal,
    no alert) every other started activity on this device before a new
    one starts. At this point at most one other exists (the invariant
    this function maintains), so a shared-token end is unambiguous."""
    for la in await _started_rows_for_device(db, device.id):
        if la.mission_id == keep_mission_id:
            continue
        token = la.activity_push_token or device.push_to_start_token
        payload = apns_push.build_end_payload(
            title="Superseded",
            dismissal_date=int(now.timestamp()) - 1,
            timestamp=int(now.timestamp()),
        )
        status, reason = await apns_push.send_live_activity(
            token, payload, environment=la.apns_environment, priority=10,
        )
        la.status = LA_ENDED
        la.ended_at = now
        la.updated_at = now
        if status != 200:
            logger.info(
                "live-activity preempt end for %s/%s: %s %s",
                device.id, la.mission_id, status, reason,
            )


async def _send_start(
    db, device: LiveActivityDevice, row: NotificationQueue,
    mission_id: str, now: datetime, *, silent: bool = False,
) -> Dict[str, Any]:
    await _preempt_device(db, device, mission_id, now)

    progress = _progress_fraction(row)
    payload = apns_push.build_start_payload(
        mission_id=mission_id,
        title=_mission_title(row),
        subtitle=(row.body or "Working…")[:120] if silent else "Starting…",
        progress=progress if progress is not None else 0.0,
        timer_end_ms=_timer_end_ms(row),
        alert_title=None if silent else row.title,
        alert_body=None if silent else row.body,
        timestamp=int(now.timestamp()),
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
    if not token:
        # Shared-token fallback is only safe while it is unambiguous —
        # Apple's routing with 2+ activities on one token is undefined.
        started = await _started_rows_for_device(db, device.id)
        if len(started) > 1:
            return {"status": "skipped", "reason": "ambiguous_shared_token"}
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
        return {"status": "ok"}
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
    return {"status": "error", "http": status, "reason": reason}


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

    per_device: Dict[str, Any] = {}
    delivered = False
    errored = False

    if row.event_kind == NOTIFY_KIND_MISSION_STARTED:
        devices = await _active_devices(db, row.user_id)
        if not devices:
            return {"status": "skipped", "reason": "no_live_activity_devices"}
        for device in devices:
            started = await _started_rows_for_device(db, device.id)
            if any(la.mission_id == mission_id for la in started):
                # at-least-once retry after a partial failure — a second
                # start push would spawn a duplicate card on screen.
                per_device[device.id] = {"status": "skipped", "reason": "already_started"}
                continue
            result = await _send_start(db, device, row, mission_id, now)
            per_device[device.id] = result
            delivered = delivered or result["status"] == "ok"
            errored = errored or result["status"] == "error"
        await db.commit()
        status = "ok" if delivered else ("error" if errored else "skipped")
        return {"status": status, "delivered": delivered, "devices": per_device}

    activities = await _activities_for_mission(db, row.user_id, mission_id)

    if not activities and row.event_kind == NOTIFY_KIND_PROGRESS:
        # Self-healing: the card was preempted by a newer task (or the
        # device rebooted). Bring it back silently — no banner — with
        # the freshest progress.
        devices = await _active_devices(db, row.user_id)
        if not devices:
            return {"status": "skipped", "reason": "no_live_activity_devices"}
        for device in devices:
            result = await _send_start(db, device, row, mission_id, now, silent=True)
            per_device[device.id] = result
            delivered = delivered or result["status"] == "ok"
            errored = errored or result["status"] == "error"
        await db.commit()
        status = "ok" if delivered else ("error" if errored else "skipped")
        return {"status": status, "delivered": delivered,
                "restarted": True, "devices": per_device}

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
                timestamp=int(now.timestamp()),
            )
            result = await _send_to_activity(
                db, la, device, payload, priority=5, now=now,
            )
        elif row.event_kind in (NOTIFY_KIND_NEEDS_INPUT, NOTIFY_KIND_NEEDS_APPROVAL):
            payload = apns_push.build_update_payload(
                title=title, subtitle="Needs your answer",
                progress=effective,
                alert_title=row.title, alert_body=row.body,
                timestamp=int(now.timestamp()),
            )
            result = await _send_to_activity(
                db, la, device, payload, priority=10, now=now,
            )
        else:  # mission_completed | mission_failed
            done = row.event_kind == NOTIFY_KIND_MISSION_COMPLETED
            payload = apns_push.build_end_payload(
                title=title,
                subtitle="Completed ✓" if done else "Stopped — needs attention",
                progress=1.0 if done else effective,
                alert_title=row.title, alert_body=row.body,
                dismissal_date=_dismissal_date(row, now),
                timestamp=int(now.timestamp()),
            )
            result = await _send_to_activity(
                db, la, device, payload, priority=10, now=now, end=True,
            )

        if result["status"] == "ok" and effective is not None:
            la.last_progress = int(effective * 100)
        per_device[device.id] = result
        delivered = delivered or result["status"] == "ok"

    await db.commit()
    return {"status": "ok" if delivered else "error",
            "delivered": delivered, "devices": per_device}
