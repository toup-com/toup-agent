"""Notification dispatcher (Autopilot PR3).

Claims queued ``notification_queue`` rows, evaluates delivery policy
(prefs, dedup, daily cap, quiet hours), sends Expo push, requests
agent-side Telegram/WhatsApp fallback, and reconciles Expo receipts.

REPLICA SAFETY — the load-bearing design constraint: platform-api runs
2 Railway replicas and the APScheduler in scheduled_tasks runs on BOTH
(no leader election, see platform_main.py:857-861). Every transition
here is therefore a per-row compare-and-swap::

    UPDATE notification_queue SET status='sending', ...
    WHERE id=:id AND status='queued'

and the loser of the race sees rowcount=0 and skips the row. Same
pattern as run_end_of_day_archival's archival_summary_status CAS
(scheduled_tasks.py). Portable to the sqlite test harness (no
FOR UPDATE SKIP LOCKED — no precedent in this repo and sqlite lacks
it).

POLICY IS EVALUATED AT CLAIM TIME, not enqueue time — prefs can change
while a row sits queued (ingest is deliberately dumb, agent_notify.py).

Row lifecycle::

    queued ──claim──▶ sending ──▶ sent_pending_receipt ──▶ sent
       ▲                 │              │(receipts ok / gave up)
       │                 ├──▶ suppressed (prefs/dedup/cap — terminal)
       │(retry backoff)  ├──▶ failed    (attempts exhausted)
       └─────────────────┘
    queued(scheduled_for=…) ◀── quiet-hours deferral re-queue

A row stuck in ``sending``/``checking_receipt`` longer than
``_STUCK_CLAIM_MAX_AGE`` (dispatcher crashed mid-send) is re-queued —
at-least-once by design; the device dedups by content.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, or_, select, update

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    AgentConfig,
    NotificationQueue,
    PushDevice,
    User,
    NOTIFY_KIND_MISSION_STARTED,
    NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_NEEDS_INPUT,
    NOTIFY_KIND_PROGRESS,
    NQ_FAILED,
    NQ_QUEUED,
    NQ_SENDING,
    NQ_SENT,
    NQ_SUPPRESSED,
)
from app.services import expo_push, live_activity_service

logger = logging.getLogger(__name__)

# Two extra in-flight statuses (String column — no migration needed).
NQ_SENT_PENDING_RECEIPT = "sent_pending_receipt"
NQ_CHECKING_RECEIPT = "checking_receipt"

_CLAIM_BATCH = 20
_STUCK_CLAIM_MAX_AGE = timedelta(minutes=10)
_RECEIPT_GIVE_UP_AGE = timedelta(hours=24)

# Kinds that block a mission bypass the daily cap — the user asked to
# be interrupted for exactly these.
_CAP_BYPASS_KINDS = {NOTIFY_KIND_NEEDS_INPUT, NOTIFY_KIND_NEEDS_APPROVAL}

# Kinds worth escalating out-of-band when no push device can be
# reached. progress/digest never escalate.
_FALLBACK_KINDS = {
    NOTIFY_KIND_NEEDS_INPUT,
    NOTIFY_KIND_NEEDS_APPROVAL,
    "mission_completed",
    "mission_failed",
}

# Dedup suppression windows by priority tier (docs/autopilot/PLAN.md
# D5: severity-sized windows; high-priority items get the narrowest).
_DEDUP_WINDOW_BY_PRIORITY = {
    "high": timedelta(minutes=5),
    "default": timedelta(minutes=30),
    "low": timedelta(hours=2),
}

_RETRY_BACKOFF_BASE = timedelta(minutes=1)
_RETRY_BACKOFF_CAP = timedelta(minutes=30)


def _utcnow() -> datetime:
    """Naive UTC — matches the DB column convention repo-wide."""
    return datetime.utcnow()


# ── Policy (pure — unit-testable with injected now/prefs/tz) ──────


def _local_now(now_utc: datetime, tz_name: Optional[str]) -> Optional[datetime]:
    if not tz_name:
        return None
    try:
        return now_utc.replace(tzinfo=dt_timezone.utc).astimezone(ZoneInfo(tz_name))
    except Exception:  # noqa: BLE001 — bad tz string on the row
        return None


def _parse_hhmm(v: str) -> Optional[Tuple[int, int]]:
    try:
        h, m = v.split(":")
        h, m = int(h), int(m)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except (ValueError, AttributeError):
        pass
    return None


def quiet_hours_defer_until(
    prefs: Dict[str, Any], tz_name: Optional[str], now_utc: datetime,
) -> Optional[datetime]:
    """When *now* falls inside the user's quiet-hours window, return
    the UTC (naive) instant the window ends; else None. Missing/broken
    timezone disables quiet hours (we can't know the user's night —
    User.timezone is two unsynced copies; this reads the PLATFORM
    copy, see docs/autopilot/INVESTIGATION.md tz follow-up)."""
    qh = prefs.get("quiet_hours") or {}
    if not qh.get("enabled"):
        return None
    local = _local_now(now_utc, tz_name)
    if local is None:
        return None
    start = _parse_hhmm(qh.get("start", ""))
    end = _parse_hhmm(qh.get("end", ""))
    if not start or not end:
        return None

    start_today = local.replace(hour=start[0], minute=start[1], second=0, microsecond=0)
    end_today = local.replace(hour=end[0], minute=end[1], second=0, microsecond=0)

    if start_today <= end_today:
        # Same-day window (e.g. 01:00-06:00).
        inside = start_today <= local < end_today
        window_end = end_today
    else:
        # Crosses midnight (e.g. 22:00-08:00).
        inside = local >= start_today or local < end_today
        window_end = end_today if local < end_today else end_today + timedelta(days=1)

    if not inside:
        return None
    return window_end.astimezone(dt_timezone.utc).replace(tzinfo=None)


def evaluate_policy(
    row: NotificationQueue,
    prefs: Dict[str, Any],
    tz_name: Optional[str],
    now_utc: datetime,
    sent_today_count: int,
    recent_dedup_hit: bool,
) -> Tuple[str, Optional[Any]]:
    """→ ("send", None) | ("suppress", reason) | ("defer", until_utc).

    Order matters: cheap terminal suppressions first, deferral last so
    a suppressed row never bounces around the queue until morning."""
    data = row.data_json or {}

    if row.event_kind == NOTIFY_KIND_PROGRESS:
        # Progress is an in-app surface (WS frames / Mission Control),
        # never an OS interruption.
        return ("suppress", "progress_in_app_only")

    # The autopilot_push pref governs autopilot MISSIONS only. Rows from
    # the other producer families declare themselves via data.kind
    # (chat_turn / reminder / routine / job) and are never suppressed by
    # it — pre-2026-07-17 one toggle silently killed chat answers and
    # reminders too.
    if (
        row.event_kind != "generic"
        and data.get("kind") in (None, "mission")
        and not prefs.get("autopilot_push", True)
    ):
        return ("suppress", "autopilot_push_disabled")

    if recent_dedup_hit:
        return ("suppress", "dedup_window")

    urgent = bool(data.get("urgent"))

    # Cap bypass: needs_input/needs_approval (the user asked to be
    # interrupted for exactly these), urgent rows (user-scheduled
    # instants — a 07:00 reminder must not be eaten by cap slot #11),
    # and producer-declared cap_exempt rows (chat answers).
    cap = prefs.get("daily_push_cap", 10)
    if (
        cap
        and row.event_kind not in _CAP_BYPASS_KINDS
        and not urgent
        and not data.get("cap_exempt")
        and sent_today_count >= cap
    ):
        return ("suppress", "daily_cap")

    if not urgent:
        until = quiet_hours_defer_until(prefs, tz_name, now_utc)
        if until is not None:
            return ("defer", until)

    return ("send", None)


# ── Claim machinery (CAS — replica-safe) ──────────────────────────


async def _cas_status(
    db, row_id: str, expected: str, new: str, now: datetime,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """Single-row status CAS. False = another replica won the race."""
    values: Dict[str, Any] = {"status": new, **(extra or {})}
    result = await db.execute(
        update(NotificationQueue)
        .where(NotificationQueue.id == row_id, NotificationQueue.status == expected)
        .values(**values)
    )
    await db.commit()
    return result.rowcount > 0


async def _requeue_stuck(db, now: datetime) -> int:
    """Rows claimed by a dispatcher that died mid-send. Re-queue —
    at-least-once by contract, but NEVER past the attempts cap: a row
    whose every attempt dies in an exception completes no dispatch, so
    the in-dispatch cap checks never run — pre-fix, such a row cycled
    sending→stuck→queued every 10 minutes forever (founder incident
    2026-07-18: a reminder fire hit attempts=48 over 8h and delivered
    its alarm at 01:40 in the morning)."""
    capped = await db.execute(
        update(NotificationQueue)
        .where(
            NotificationQueue.status.in_([NQ_QUEUED, NQ_SENDING, NQ_CHECKING_RECEIPT]),
            NotificationQueue.attempts >= settings.notification_max_attempts,
        )
        .values(status=NQ_FAILED, claimed_at=None,
                last_error="stuck_requeue_exhausted")
    )
    cutoff = now - _STUCK_CLAIM_MAX_AGE
    result = await db.execute(
        update(NotificationQueue)
        .where(
            NotificationQueue.status.in_([NQ_SENDING, NQ_CHECKING_RECEIPT]),
            NotificationQueue.claimed_at.isnot(None),
            NotificationQueue.claimed_at < cutoff,
        )
        .values(status=NQ_QUEUED, claimed_at=None)
    )
    await db.commit()
    if capped.rowcount:
        logger.warning(
            "notification dispatch: failed %d rows at the attempts cap",
            capped.rowcount,
        )
    if result.rowcount:
        logger.warning("notification dispatch: re-queued %d stuck rows", result.rowcount)
    # Live Activity staleness GC for ALL users, mirrored from the
    # per-user sweep at the LA lane entry: idle devices (no queued
    # rows) must converge too, or a >8h-dead started row blocks the
    # one-activity-per-device invariant until the next mission.
    try:
        await live_activity_service.sweep_stale_activities(db, now)
    except Exception:  # noqa: BLE001 — GC must never stall the tick
        logger.exception("live-activity stale sweep failed")
        await db.rollback()
    return result.rowcount or 0


async def _claim_batch(db, now: datetime, limit: int = _CLAIM_BATCH) -> List[str]:
    """Select candidates, then CAS-claim each. Returns claimed ids."""
    result = await db.execute(
        select(NotificationQueue.id)
        .where(
            NotificationQueue.status == NQ_QUEUED,
            or_(
                NotificationQueue.scheduled_for.is_(None),
                NotificationQueue.scheduled_for <= now,
            ),
        )
        .order_by(NotificationQueue.created_at)
        .limit(limit)
    )
    claimed: List[str] = []
    for (row_id,) in result.all():
        ok = await _cas_status(
            db, row_id, NQ_QUEUED, NQ_SENDING, now,
            extra={"claimed_at": now, "attempts": NotificationQueue.attempts + 1},
        )
        if ok:
            claimed.append(row_id)
    return claimed


# ── Per-row context loads ─────────────────────────────────────────


async def _load_prefs_and_tz(db, user_id: str) -> Tuple[Dict[str, Any], Optional[str]]:
    # _merged_prefs/_prefs_dict live next to the account API's Pydantic
    # contract on purpose — import the functions, don't re-implement
    # the type-checked merge.
    from app.api.account import _merged_prefs, _prefs_dict

    user = await db.get(User, user_id)
    raw = _prefs_dict(getattr(user, "notification_preferences", None)) if user else None
    tz_name = getattr(user, "timezone", None) if user else None
    return _merged_prefs(raw), tz_name


async def _sent_today_count(db, user_id: str, tz_name: Optional[str], now: datetime) -> int:
    """Pushes sent in the user's local day (UTC day when tz missing)."""
    local = _local_now(now, tz_name)
    if local is not None:
        day_start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = day_start_local.astimezone(dt_timezone.utc).replace(tzinfo=None)
    else:
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (
        await db.scalar(
            select(func.count()).select_from(NotificationQueue).where(
                NotificationQueue.user_id == user_id,
                NotificationQueue.sent_at.isnot(None),
                NotificationQueue.sent_at >= day_start,
            )
        )
    ) or 0


async def _recent_dedup_hit(db, row: NotificationQueue, now: datetime) -> bool:
    if not row.dedup_key:
        return False
    window = _DEDUP_WINDOW_BY_PRIORITY.get(row.priority, timedelta(minutes=30))
    return (
        await db.scalar(
            select(func.count()).select_from(NotificationQueue).where(
                NotificationQueue.user_id == row.user_id,
                NotificationQueue.dedup_key == row.dedup_key,
                NotificationQueue.id != row.id,
                NotificationQueue.status.in_(
                    [NQ_SENT, NQ_SENT_PENDING_RECEIPT, NQ_CHECKING_RECEIPT]
                ),
                NotificationQueue.created_at >= now - window,
            )
        )
    ) > 0


async def _active_devices(db, user_id: str) -> List[PushDevice]:
    result = await db.execute(
        select(PushDevice).where(
            PushDevice.user_id == user_id,
            PushDevice.revoked_at.is_(None),
        )
        .order_by(PushDevice.last_seen_at.desc())
    )
    return list(result.scalars().all())


async def _revoke_device(db, device_id: str, now: datetime, reason: str) -> None:
    await db.execute(
        update(PushDevice)
        .where(PushDevice.id == device_id, PushDevice.revoked_at.is_(None))
        .values(revoked_at=now)
    )
    await db.commit()
    logger.info("push device %s revoked (%s)", device_id, reason)


# ── Agent-side channel fallback ───────────────────────────────────


async def _request_agent_channel_delivery(
    db, row: NotificationQueue,
) -> Dict[str, Any]:
    """Ask the tenant agent to deliver via Telegram/WhatsApp — channel
    connections live in the agent container, never here. The agent
    endpoint (POST /api/notify/deliver) lands in PR4; until a tenant's
    container has it, this records agent_endpoint_missing and the row
    still resolves on the push outcome."""
    import httpx

    result = await db.execute(
        select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
            AgentConfig.user_id == row.user_id,
            AgentConfig.deploy_status == "active",
        )
    )
    cfg = result.first()
    if not cfg or not cfg.agent_url or not cfg.agent_api_key:
        return {"status": "skipped", "reason": "no_active_agent"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{cfg.agent_url.rstrip('/')}/api/notify/deliver",
                headers={"X-Agent-Key": cfg.agent_api_key},
                json={
                    "event_kind": row.event_kind,
                    "title": row.title,
                    "body": row.body,
                    "data": row.data_json or {},
                },
            )
        if resp.status_code == 404:
            return {"status": "skipped", "reason": "agent_endpoint_missing"}
        if resp.status_code >= 400:
            return {"status": "error", "reason": f"http_{resp.status_code}"}
        detail = resp.json() or {}
        # A 200 with delivered=false means "agent reachable, but no
        # channel had a recipient" (e.g. user never DM'd the Telegram
        # bot) — that is NOT a delivery.
        if detail.get("delivered") is True:
            return {"status": "ok", "detail": detail}
        return {
            "status": "skipped",
            "reason": "no_channel_recipient",
            "detail": detail,
        }
    except Exception as e:  # noqa: BLE001 — agent may be recycling
        return {"status": "error", "reason": str(e)[:200]}


# ── Send one claimed row ──────────────────────────────────────────


async def _finalize(
    db, row_id: str, new_status: str, now: datetime,
    channels: Optional[Dict[str, Any]] = None,
    last_error: Optional[str] = None,
    sent: bool = False,
    scheduled_for: Optional[datetime] = None,
) -> None:
    extra: Dict[str, Any] = {"channels_json": channels, "last_error": last_error}
    if sent:
        extra["sent_at"] = now
    if scheduled_for is not None:
        extra["scheduled_for"] = scheduled_for
        extra["claimed_at"] = None
    await _cas_status(db, row_id, NQ_SENDING, new_status, now, extra=extra)


async def _dispatch_row(db, row_id: str, now: datetime) -> str:
    """Returns the terminal decision for logging/tests."""
    row = await db.get(NotificationQueue, row_id)
    if row is None:  # deleted user cascade mid-flight
        return "gone"

    # Live Activity lane, progress rows: the activity update IS the
    # delivery — progress never becomes an OS alert push (policy
    # invariant), so the row terminates suppressed exactly as before,
    # with the APNs outcome recorded alongside for audit.
    if row.event_kind == NOTIFY_KIND_PROGRESS:
        la = await live_activity_service.handle_notification_row(db, row, now)
        channels: Dict[str, Any] = {"policy": {"suppressed": "progress_in_app_only"}}
        if la is not None:
            channels["live_activity"] = la
        await _finalize(db, row_id, NQ_SUPPRESSED, now, channels=channels)
        return "suppressed:progress_in_app_only"

    prefs, tz_name = await _load_prefs_and_tz(db, row.user_id)
    decision, arg = evaluate_policy(
        row, prefs, tz_name, now,
        sent_today_count=await _sent_today_count(db, row.user_id, tz_name, now),
        recent_dedup_hit=await _recent_dedup_hit(db, row, now),
    )

    if decision == "suppress":
        await _finalize(
            db, row_id, NQ_SUPPRESSED, now,
            channels={"policy": {"suppressed": arg}},
        )
        return f"suppressed:{arg}"

    if decision == "defer":
        # Back to queued with the deferral timestamp; the claim query
        # skips it until then. attempts was bumped at claim — undo so
        # deferral never counts against the retry budget.
        await _cas_status(
            db, row_id, NQ_SENDING, NQ_QUEUED, now,
            extra={
                "scheduled_for": arg,
                "claimed_at": None,
                "attempts": NotificationQueue.attempts - 1,
            },
        )
        return "deferred"

    # Live Activity lane, mission_started rows: their ONLY surface is
    # the push-to-start (the banner alert rides inside it) — an Expo
    # alert or Telegram message saying "mission started" would be
    # noise on top of the agent's own chat confirmation. Transient
    # APNs errors follow the normal retry backoff; a user with no
    # registered device terminates quietly instead of retry-looping.
    if row.event_kind == NOTIFY_KIND_MISSION_STARTED:
        la = await live_activity_service.handle_notification_row(db, row, now)
        la = la or {"status": "skipped", "reason": "no_mission_id"}
        if la.get("delivered"):
            await _finalize(
                db, row_id, NQ_SENT, now,
                channels={"live_activity": la}, sent=True,
            )
            return "sent"
        if la.get("status") == "error" and row.attempts < settings.notification_max_attempts:
            backoff = min(
                _RETRY_BACKOFF_BASE * (4 ** max(0, row.attempts - 1)),
                _RETRY_BACKOFF_CAP,
            )
            await _finalize(
                db, row_id, NQ_QUEUED, now,
                channels={"live_activity": la},
                last_error="live_activity_send_failed",
                scheduled_for=now + backoff,
            )
            return "retry_scheduled"
        await _finalize(
            db, row_id, NQ_SUPPRESSED, now,
            channels={
                "live_activity": la,
                "policy": {"suppressed": "live_activity_unavailable"},
            },
        )
        return "suppressed:live_activity_unavailable"

    channels: Dict[str, Any] = {}
    delivered = False
    has_tickets = False

    # Live Activity lane, alert kinds (needs_input/needs_approval/
    # completed/failed): update or end the on-screen activity IN
    # ADDITION to the normal push/fallback flow. APNs acceptance
    # counts as a delivery — the lock screen changed — which stops
    # the retry loop from re-alerting the activity, but deliberately
    # does NOT short-circuit the Telegram/WhatsApp fallback below
    # (the chat message carries the full text and is answerable).
    la_delivered = False
    la = await live_activity_service.handle_notification_row(db, row, now)
    if la is not None:
        channels["live_activity"] = la
        la_delivered = bool(la.get("delivered"))

    devices = await _active_devices(db, row.user_id)
    if devices:
        messages = [
            expo_push.build_message(
                d.expo_push_token, row.title, row.body, {
                    **(row.data_json or {}),
                    "event_kind": row.event_kind,
                    "notification_id": row.id,
                },
                row.priority,
            )
            for d in devices
        ]
        try:
            tickets = await expo_push.send_push_messages(messages)
            ticket_map: Dict[str, Any] = {}
            for device, ticket in zip(devices, tickets):
                if expo_push.is_device_not_registered(ticket):
                    await _revoke_device(db, device.id, now, "DeviceNotRegistered ticket")
                    ticket_map[device.id] = {"status": "device_not_registered"}
                elif ticket.get("status") == "ok":
                    delivered = True
                    has_tickets = True
                    ticket_map[device.id] = {"status": "ok", "ticket": ticket.get("id")}
                else:
                    ticket_map[device.id] = {
                        "status": "error",
                        "error": str(ticket.get("message", ""))[:200],
                    }
            channels["expo"] = ticket_map
        except expo_push.ExpoPushError as e:
            channels["expo"] = {"status": "batch_error", "error": str(e)[:200]}
    else:
        channels["expo"] = {"status": "skipped", "reason": "no_active_devices"}

    # Out-of-band fallback only when push couldn't reach anyone.
    # Producers that already fanned the same text out via the agent's
    # channel_dispatcher (reminder/routine fires) set no_agent_fallback
    # — the fallback would land the identical message on Telegram/
    # WhatsApp twice.
    if (
        not delivered
        and row.event_kind in _FALLBACK_KINDS
        and prefs.get("autopilot_channel_fallback", True)
        and not (row.data_json or {}).get("no_agent_fallback")
    ):
        fallback = await _request_agent_channel_delivery(db, row)
        channels["agent_fallback"] = fallback
        if fallback.get("status") == "ok":
            delivered = True

    if delivered or la_delivered:
        new_status = NQ_SENT_PENDING_RECEIPT if has_tickets else NQ_SENT
        await _finalize(db, row_id, new_status, now, channels=channels, sent=True)
        return "sent"

    # Nothing reached the user — retry with backoff or give up.
    attempts = row.attempts  # already incremented at claim
    if attempts >= settings.notification_max_attempts:
        await _finalize(
            db, row_id, NQ_FAILED, now,
            channels=channels, last_error="no_channel_delivered",
        )
        return "failed"
    backoff = min(
        _RETRY_BACKOFF_BASE * (4 ** max(0, attempts - 1)), _RETRY_BACKOFF_CAP,
    )
    await _finalize(
        db, row_id, NQ_QUEUED, now,
        channels=channels, last_error="no_channel_delivered",
        scheduled_for=now + backoff,
    )
    return "retry_scheduled"


# ── Scheduled entrypoints ─────────────────────────────────────────


async def notification_dispatch_loop() -> None:
    """Self-contained dispatch loop — started from the platform_main
    LIFESPAN, deliberately independent of ENABLE_SCHEDULER/APScheduler:
    proactive notifications must flow even on deployments where the
    memory-maintenance scheduler is disabled (the documented
    multi-worker guidance), and a scheduler-start failure must not
    take the pipeline down with it. Replica safety comes from the
    per-row status CAS, not from scheduling. Receipt checks piggyback
    every 10th tick (~5 min at the 30s default)."""
    logger.info(
        "notification dispatch loop started (interval %ss)",
        settings.notification_dispatch_interval_seconds,
    )
    tick = 0
    while True:
        try:
            await run_notification_dispatch()
            tick += 1
            if tick % 10 == 0:
                await run_notification_receipt_check()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the loop never dies
            logger.exception("notification dispatch tick failed")
        await asyncio.sleep(settings.notification_dispatch_interval_seconds)


async def run_notification_dispatch() -> Dict[str, int]:
    """Dispatcher tick — registered in scheduled_tasks.setup_scheduler.
    Safe to run on every replica concurrently (per-row CAS)."""
    if not settings.notification_dispatch_enabled:
        return {"claimed": 0}
    now = _utcnow()
    stats: Dict[str, int] = {"claimed": 0}
    async with async_session_maker() as db:
        await _requeue_stuck(db, now)
        claimed = await _claim_batch(db, now)
        stats["claimed"] = len(claimed)
        for row_id in claimed:
            try:
                outcome = await _dispatch_row(db, row_id, now)
                stats[outcome.split(":")[0]] = stats.get(outcome.split(":")[0], 0) + 1
            except Exception as exc:  # noqa: BLE001 — one bad row must not stall the batch
                logger.exception("notification dispatch failed for row %s", row_id)
                await db.rollback()
                # An exception must not strand the row in 'sending'
                # (pre-fix that meant a 10-min stuck-requeue loop with
                # no cap and no recorded error). Requeue on the normal
                # backoff — or fail at the cap — with the exception on
                # last_error so the incident is readable off the row.
                try:
                    err = f"dispatch_exception: {exc!r}"[:200]
                    row = await db.get(NotificationQueue, row_id)
                    if row is not None and row.status == NQ_SENDING:
                        if row.attempts >= settings.notification_max_attempts:
                            await _cas_status(
                                db, row_id, NQ_SENDING, NQ_FAILED, now,
                                extra={"last_error": err, "claimed_at": None},
                            )
                        else:
                            backoff = min(
                                _RETRY_BACKOFF_BASE * (4 ** max(0, row.attempts - 1)),
                                _RETRY_BACKOFF_CAP,
                            )
                            await _cas_status(
                                db, row_id, NQ_SENDING, NQ_QUEUED, now,
                                extra={"last_error": err, "claimed_at": None,
                                       "scheduled_for": now + backoff},
                            )
                        stats["errored"] = stats.get("errored", 0) + 1
                except Exception:  # noqa: BLE001
                    logger.exception("failed to finalize errored row %s", row_id)
                    await db.rollback()
    if stats["claimed"]:
        logger.info("notification dispatch: %s", stats)
    return stats


async def run_notification_receipt_check() -> Dict[str, int]:
    """Poll Expo receipts ~15 min after send; prune dead tokens.
    Missing receipts → poll again next tick; >24h old → give up
    (best-effort, the push either landed or it didn't)."""
    if not settings.notification_dispatch_enabled:
        return {"checked": 0}
    now = _utcnow()
    stats = {"checked": 0, "pruned": 0}
    cutoff = now - timedelta(minutes=settings.notification_receipt_delay_minutes)
    async with async_session_maker() as db:
        result = await db.execute(
            select(NotificationQueue.id)
            .where(
                NotificationQueue.status == NQ_SENT_PENDING_RECEIPT,
                NotificationQueue.sent_at.isnot(None),
                NotificationQueue.sent_at <= cutoff,
            )
            .limit(50)
        )
        for (row_id,) in result.all():
            claimed = await _cas_status(
                db, row_id, NQ_SENT_PENDING_RECEIPT, NQ_CHECKING_RECEIPT, now,
                extra={"claimed_at": now},
            )
            if not claimed:
                continue
            stats["checked"] += 1
            try:
                stats["pruned"] += await _check_receipts_for_row(db, row_id, now)
            except Exception:  # noqa: BLE001
                logger.exception("receipt check failed for row %s", row_id)
                await db.rollback()
                await _cas_status(db, row_id, NQ_CHECKING_RECEIPT, NQ_SENT_PENDING_RECEIPT, now)
    return stats


async def _check_receipts_for_row(db, row_id: str, now: datetime) -> int:
    row = await db.get(NotificationQueue, row_id)
    if row is None:
        return 0
    expo_channel = (row.channels_json or {}).get("expo") or {}
    ticket_by_device = {
        device_id: entry.get("ticket")
        for device_id, entry in expo_channel.items()
        if isinstance(entry, dict) and entry.get("ticket")
    }
    if not ticket_by_device:
        await _cas_status(db, row_id, NQ_CHECKING_RECEIPT, NQ_SENT, now)
        return 0

    receipts = await expo_push.get_push_receipts(list(ticket_by_device.values()))
    pruned = 0
    all_terminal = True
    channels = dict(row.channels_json or {})
    expo_updated = dict(expo_channel)
    for device_id, ticket_id in ticket_by_device.items():
        receipt = receipts.get(ticket_id)
        if receipt is None:
            all_terminal = False
            continue
        entry = dict(expo_updated.get(device_id) or {})
        entry["receipt"] = receipt.get("status")
        if expo_push.is_device_not_registered(receipt):
            await _revoke_device(db, device_id, now, "DeviceNotRegistered receipt")
            pruned += 1
        expo_updated[device_id] = entry
    channels["expo"] = expo_updated

    gave_up = row.sent_at is not None and (now - row.sent_at) > _RECEIPT_GIVE_UP_AGE
    new_status = NQ_SENT if (all_terminal or gave_up) else NQ_SENT_PENDING_RECEIPT
    await _cas_status(
        db, row_id, NQ_CHECKING_RECEIPT, new_status, now,
        extra={"channels_json": channels, "claimed_at": None},
    )
    return pruned
