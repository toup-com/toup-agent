"""Live Activity lane of the notification dispatcher.

Maps background-task notification rows (Autopilot missions AND quick
spawned jobs — anything carrying ``data.mission_id``) onto APNs Live
Activity pushes:

    mission_started   → push-to-start on every registered device
                        (banner alert rides in the start payload)
    announcement      → push-to-start too (operator → user, admin
                        dispatch): one card, the brand orb color
                        instead of the agent's, and no follow-up —
                        the row IS the whole lifecycle
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
                        not fallback). That holds for an operator's
                        chosen dispatch tone too: this lane resolves
                        `data.tone` to a file name and `apns_push`
                        drops it (`ACTIVITY_KIT_SOUND`).

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
Alarm-class terminal rows send ``content-state.fired`` instead of any
progress — the widget renders a ringing presentation, never a bar.

REMINDER WINS (founder rule): a chat-turn card — or an admin
announcement — must never displace or orphan a live ``reminder:*``
countdown; enforced in the start branch, the restart-if-missing lane,
AND ``_preempt_device`` itself. When the app reports that a device alarm
(AlarmKit) owns a reminder's fire (``alarm_owned_at``), the fire lane
stays quiet on that device: no loud restart, no rings — AlarmKit is
already ringing through silent/Focus.

Every function returns a channels_json fragment — the dispatcher
records it verbatim, so a production incident can be read straight
off the notification row.
"""
from __future__ import annotations

import logging
import urllib.parse
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.db.models import (
    LA_ENDED, LA_STARTED,
    AgentConfig, LiveActivity, LiveActivityDevice, NotificationQueue,
    NOTIFY_KIND_ANNOUNCEMENT,
    NOTIFY_KIND_MISSION_COMPLETED, NOTIFY_KIND_MISSION_FAILED,
    NOTIFY_KIND_MISSION_STARTED, NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_NEEDS_INPUT, NOTIFY_KIND_PROGRESS,
)
from app.services import apns_push, dispatch_tones

logger = logging.getLogger(__name__)

# Kinds this lane knows how to render on the activity.
LIVE_ACTIVITY_KINDS = {
    NOTIFY_KIND_MISSION_STARTED,
    NOTIFY_KIND_PROGRESS,
    NOTIFY_KIND_NEEDS_INPUT,
    NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_MISSION_COMPLETED,
    NOTIFY_KIND_MISSION_FAILED,
    NOTIFY_KIND_ANNOUNCEMENT,
}

# Kinds whose row OPENS a card (push-to-start) instead of updating one.
# An operator announcement has no lifecycle to narrate — its single row
# is the whole card — so it takes the same branch a mission's first row
# does.
_START_EVENT_KINDS = {NOTIFY_KIND_MISSION_STARTED, NOTIFY_KIND_ANNOUNCEMENT}

# Mission-id prefixes whose card must never end a live ``reminder:``
# countdown to make room for itself (see REMINDER WINS above).
# ``chatjob:`` (Round 3, 2026-08-18) is the per-CONVERSATION job card —
# a job the agent creates inside a chat turn — and is a chat-turn card
# in every sense the founder rule cares about.
_NEVER_PREEMPT_REMINDER_PREFIXES = ("chatturn:", "chatjob:", "admin:")

# Mission-id prefixes that are chat-turn cards for the yield rules
# (start branch + restart lane): a working card that must not displace
# a live countdown.
_CHAT_TURN_PREFIXES = ("chatturn:", "chatjob:")


def _is_chat_turn_mission(row: NotificationQueue, mission_id: str) -> bool:
    return (
        (row.data_json or {}).get("kind") == "chat_turn"
        or mission_id.startswith(_CHAT_TURN_PREFIXES)
    )


# Round 8: the widget's phase vocabulary by row kind (mirror of
# subagent_orchestrator.JOB_PHASE_BY_KIND, kept here so the platform can
# derive it for rows an older agent sent without ``data.phase``).
_JOB_PHASE_BY_KIND = {
    NOTIFY_KIND_MISSION_STARTED: "starting",
    NOTIFY_KIND_PROGRESS: "running",
    NOTIFY_KIND_MISSION_COMPLETED: "completed",
    NOTIFY_KIND_MISSION_FAILED: "failed",
    NOTIFY_KIND_NEEDS_INPUT: "needs_you",
    NOTIFY_KIND_NEEDS_APPROVAL: "needs_you",
}


def _job_id_of(row: NotificationQueue) -> Optional[str]:
    """The job a JOB row is about (``data.job_id``), when it differs from
    the mission id it addresses. Round 8: the app starts a job's card
    LOCALLY under the raw job id ("the platform's mission id for this
    job") while the platform keys a chat job's pushes ``chatjob:<chat>``
    — so a job push must also reach a card registered under the job id,
    or the two surfaces drift apart the moment the app backgrounds."""
    data = row.data_json or {}
    if data.get("kind") != "job":
        return None
    jid = data.get("job_id")
    if not isinstance(jid, str) or not jid.strip():
        return None
    jid = jid.strip()[:64]
    return jid if jid != _mission_id(row) else None

# An admin announcement is NOT from the user's agent, so its card must
# not wear the agent's orb color — the Toup brand hue disowns it on the
# lock screen the same way AdminNoticeCard does in chat.
_ADMIN_ORB_COLOR = "#7C6BF5"


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
    # chatjob: rows (per-conversation job cards) age the same way — a
    # create_job job is turn-bound (the runner's finalizer closes it at
    # turn end, the reaper at 30 min), and a refresh for a NEW job on the
    # same card re-stamps started_at, so the clock always measures the
    # latest job.
    turn_stmt = (
        update(LiveActivity)
        .where(
            LiveActivity.status == LA_STARTED,
            or_(
                LiveActivity.mission_id.like("chatturn:%"),
                LiveActivity.mission_id.like("chatjob:%"),
            ),
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


def _timer_start_ms(row: NotificationQueue) -> Optional[int]:
    """The countdown's SET instant (``data.timer_start_ms``) — rides the
    content-state so the widget draws the absolute set→fire span instead of
    restarting its bar at every view rebuild."""
    raw = (row.data_json or {}).get("timer_start_ms")
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
        # Note this re-stamps an agent-authored data_json as 'platform'.
        # It cannot launder a data.deep_link past _send_start's source
        # check, because ring 2+ rows are excluded from the restart lane by
        # `_realert_seq(row) == 1` and so never reach _send_start. If that
        # gate ever moves, this stamp has to move with it.
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


# ── Round 3 (2026-08-18): card content + deep link from row data ──
# Producers (the agent's _notify_job_event, ws_chat's chat-turn pushes)
# put flat scalars on data_json; this lane maps them onto the Swift
# content-state keys (see apns_push._extra_state) and the tap URL.

_DEEP_LINK_ID_MAX = 64


def _safe_id(value: Any) -> Optional[str]:
    """An id we are willing to put on the wire: short, printable, no
    whitespace. Anything else is dropped (never truncated into a
    different id)."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or len(v) > _DEEP_LINK_ID_MAX or any(ch.isspace() for ch in v):
        return None
    return v


def _row_extra_state(row: NotificationQueue) -> Dict[str, Any]:
    """Content-state extras for this row (camelCase, whitelisted by
    apns_push). ``percent`` mirrors ``data.progress`` so the widget's
    n/m · NN% line needs no arithmetic; the lane's never-backwards clamp
    re-syncs it from the fraction it actually sends."""
    data = row.data_json or {}
    out: Dict[str, Any] = {}
    # Round 8: the widget's ContentState v2 keys ride alongside the Round-3
    # ones — ``stepLabel`` = the step line, ``jobKind`` = the icon tag —
    # so a pushed card carries the same fields a locally-started one does.
    for src, dst in (("job_type", "jobType"), ("job_type", "jobKind"),
                     ("step_name", "stepName"), ("step_name", "stepLabel"),
                     ("preview", "preview")):
        v = data.get(src)
        if isinstance(v, str) and v.strip():
            out[dst] = v
    for src, dst in (("steps_done", "stepsDone"), ("steps_total", "stepsTotal")):
        v = data.get(src)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[dst] = int(v)
    prog = data.get("progress")
    if isinstance(prog, (int, float)) and not isinstance(prog, bool):
        out["percent"] = max(0, min(100, int(prog)))
    # ``phase``: the producer's word when it sent one (Round-8 agents stamp
    # every job push), else derived from the row kind — so cards pushed by
    # older agents render the same face. Job/turn rows only: a reminder or
    # voice row is not a work card and must not grow a phase.
    if data.get("kind") in ("job", "chat_turn") or _is_chat_turn_mission(row, _mission_id(row) or ""):
        ph = data.get("phase")
        if not (isinstance(ph, str) and ph in _JOB_PHASE_BY_KIND.values()):
            ph = _JOB_PHASE_BY_KIND.get(row.event_kind)
        if ph:
            out["phase"] = ph
    cid = _safe_id(data.get("chat_id"))
    if cid:
        out["chatId"] = cid
    mid = _safe_id(data.get("message_id"))
    if mid:
        out["messageId"] = mid
    return out


def automation_deep_link(data: Dict[str, Any], mission_id: str) -> Optional[str]:
    """R28: the tap target for an automation-run card — COMPUTED from two
    _safe_id-validated tokens, never a tenant-chosen URL. Colon-free host
    with ids in the query (the app's Android launch path rebuilds
    `toup://<route>` and a colon would parse as a port); `mission` rides
    along for the tap-ACK machinery. None unless the row declares
    `route:"automation"` with both ids intact."""
    if data.get("route") != "automation":
        return None
    auto_id = _safe_id(data.get("automation_id"))
    run_id = _safe_id(data.get("run_id"))
    if not auto_id or not run_id:
        return None
    return (
        "toup://automation?session="
        + urllib.parse.quote(auto_id, safe="")
        + "&run=" + urllib.parse.quote(run_id, safe="")
        + f"&mission={mission_id}"
    )


def _deep_link_params(row: NotificationQueue) -> str:
    """``&chat_id=…&message_id=…`` for the card's tap URL (attributes are
    start-fixed, so this reflects the FIRST job on a conversation card;
    the content-state copies are the live ones). Empty when unknown."""
    data = row.data_json or {}
    parts = []
    cid = _safe_id(data.get("chat_id"))
    if cid:
        parts.append("chat_id=" + urllib.parse.quote(cid, safe=""))
    mid = _safe_id(data.get("message_id"))
    if mid:
        parts.append("message_id=" + urllib.parse.quote(mid, safe=""))
    return ("&" + "&".join(parts)) if parts else ""


def _end_after_s(row: NotificationQueue) -> Optional[int]:
    """Producer-requested linger of the COMPLETED state before the
    ``event=end`` push (Round 3, item 2). Bounded 1..120s."""
    raw = (row.data_json or {}).get("end_after_s")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    v = int(raw)
    if v <= 0:
        return None
    return min(v, 120)


def _deferred_end_idem(mission_id: str, row: NotificationQueue) -> str:
    """Idempotency key of the chained end row for THIS terminal row —
    per job (data.job_id) so two jobs finishing on one conversation card
    book two distinct ends, and a partial-failure retry of the same row
    books none twice."""
    data = row.data_json or {}
    tag = data.get("job_id") or row.dedup_key or row.id
    return f"la-end:{mission_id}:{tag}"[:200]


async def _enqueue_deferred_end(
    db, row: NotificationQueue, mission_id: str, now: datetime, delay_s: int,
) -> bool:
    """Round 3 (item 2): the completed state has just been pushed as an
    alerting update; book the ``event=end`` as its own queue row one
    delay out instead of sending it now — the card keeps its ✓ + preview
    in the Dynamic Island for a beat before retiring to the lock screen.
    Same shape as ``_enqueue_next_ring``: LA-only, no Expo/agent
    fan-out, idempotent per (mission, job), urgent so quiet hours never
    hold a bannerless close, no dedup_key so the dedup window cannot
    swallow it. ``end_only`` makes the lane skip the alerting update on
    the follow-up; ``silent`` keeps the restart lane out of it (a card
    that is already gone stays gone). The caller's end-of-lane commit
    persists it. Effective delay = delay_s + up to one dispatcher tick."""
    idem = _deferred_end_idem(mission_id, row)
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
        "end_only": True,
        "silent": True,
        "la_only": True,
        "no_agent_fallback": True,
        "urgent": True,
        "cap_exempt": True,
    })
    data.pop("end_after_s", None)
    data.pop("refresh_if_started", None)
    db.add(NotificationQueue(
        id=str(uuid.uuid4()),
        user_id=row.user_id,
        source="platform",
        event_kind=row.event_kind,
        title=row.title,
        body=row.body,
        data_json=data,
        priority=row.priority,
        scheduled_for=now + timedelta(seconds=delay_s),
        idempotency_key=idem,
    ))
    return True


async def suppress_pending_ends(db, user_id: str, mission_id: str, now: datetime) -> int:
    """Cancel the not-yet-fired deferred-end rows of a mission (a NEW job
    took over the conversation card, or the user has seen the answer in
    the app). Filters on the idempotency-key prefix — a plain string
    column, portable across the JSONB prod column and sqlite tests."""
    rows = (await db.execute(
        select(NotificationQueue).where(
            NotificationQueue.user_id == user_id,
            NotificationQueue.status.in_(("queued", "sending")),
            NotificationQueue.idempotency_key.like(f"la-end:{mission_id}:%"),
        )
    )).scalars().all()
    n = 0
    for r in rows:
        r.status = "suppressed"
        r.claimed_at = None
        r.channels_json = {**(r.channels_json or {}),
                           "policy": {"suppressed": "end_superseded"}}
        n += 1
    return n


def _row_is_newer_than_card(row: NotificationQueue, la: LiveActivity) -> bool:
    """True when this row was created AFTER the card was (re)started —
    i.e. it is a NEW job on the conversation card, not an at-least-once
    retry of the start that opened the card (whose success stamped
    ``started_at`` at or after that row's ``created_at``). Compared
    against ``started_at``, NOT ``updated_at``: progress rows of the new
    job may have refreshed the card before its (loop-scheduled) start
    row was processed, and they must not turn the start into a skip."""
    created = getattr(row, "created_at", None)
    started = la.started_at
    if created is None or started is None:
        return True
    return created > started


async def _refresh_started_card(
    db, la: LiveActivity, device: LiveActivityDevice, row: NotificationQueue,
    mission_id: str, now: datetime,
) -> Dict[str, Any]:
    """ONE CARD PER CONVERSATION (Round 3, item 4): a ``mission_started``
    row for a mission whose card is already up on this device REFRESHES
    the card with the new job's start state instead of skipping as an
    at-least-once duplicate — same title/subtitle/timer/alert the start
    would have carried, over the live activity's token. Progress resets
    (a new job starts at zero: ``last_progress`` is cleared so the
    never-backwards clamp does not pin it at the previous job's bar) and
    the previous job's not-yet-fired deferred end is cancelled so it
    cannot close the card out from under the new job."""
    data = row.data_json or {}
    timer_ms = _timer_end_ms(row)
    progress = _progress_fraction(row)
    subtitle_override = data.get("subtitle") if isinstance(data.get("subtitle"), str) else None
    payload = apns_push.build_update_payload(
        title=_mission_title(row),
        subtitle=(subtitle_override or "Starting…")[:120],
        progress=(progress if progress is not None else 0.0),
        timer_end_ms=timer_ms,
        timer_start_ms=_timer_start_ms(row),
        alert_title=row.title,
        alert_body=row.body,
        alert_sound=dispatch_tones.apns_sound(data.get("tone")),
        stale_date=(int(timer_ms / 1000) + 120 if timer_ms
                    else int(now.timestamp()) + 1800),
        timestamp=int(now.timestamp()),
        extra=_row_extra_state(row),
    )
    result = await _send_to_activity(db, la, device, payload, priority=10, now=now)
    if result["status"] == "ok":
        # The new job's bar starts where its start row says (usually 0):
        # the never-backwards clamp must not pin it at the previous job's
        # last value.
        la.last_progress = int((progress or 0.0) * 100)
        la.started_at = now
        cancelled = await suppress_pending_ends(db, row.user_id, mission_id, now)
        result = {**result, "refreshed": True}
        if cancelled:
            result["ends_cancelled"] = cancelled
    return result


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


# ── The foreground gate ──────────────────────────────────────────────────
# A Live Activity is the agent reaching the user OUTSIDE the app. Pushing one
# at a device whose owner is looking at the app is the founder's report of
# 2026-08-31: "when a user asks a question and stays in the app waiting for the
# answer, the notification card and the Dynamic Island activate" — and then sit
# there for their dismiss window once the user does leave.
#
# The app reports both edges of its presence (POST .../presence, a TTL on the
# device row). This is the one place that consults it.
#
# EXEMPT, and it is not a softening of the rule but the rule applied to a
# different kind of object: a `reminder:` countdown is not a report of
# something that happened, it is the PENDING ALERT ITSELF. Its card has to be
# armed at the moment the reminder is created — which is, by definition, while
# the user is in the app talking to their agent — and the platform cannot come
# back and arm it later. Everything else can wait until the user leaves,
# because everything else is a message ABOUT something that already happened.
#
# What a suppressed row COSTS, recorded rather than discovered later: the LA
# lane is this app's only OS surface, so a card declined here is never
# retried — `_START_EVENT_KINDS` rows terminate quietly on a non-error skip
# (notification_dispatcher, the `_START_EVENT_KINDS` branch). That is the
# intended trade for every family this gate covers, because each of them
# already lands IN the app the user is looking at: a chat turn is the answer
# in the thread, a job is its card in the run surface, and an operator
# announcement "already lands as its own chat card written by the fan-out"
# (the dispatcher's own words). A reminder is the one that does not, which is
# exactly why it is exempt.
_PRESENCE_EXEMPT_PREFIXES = ("reminder:",)


def _device_is_foreground(device: LiveActivityDevice, now: datetime) -> bool:
    until = getattr(device, "foreground_until", None)
    return until is not None and until > now


def _suppressed_by_presence(
    device: LiveActivityDevice, mission_id: str, now: datetime,
) -> bool:
    """Would starting `mission_id`'s card on `device` put a card on screen
    while its owner is already looking at the app?"""
    if mission_id.startswith(_PRESENCE_EXEMPT_PREFIXES):
        return False
    return _device_is_foreground(device, now)


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
    db, user_id: str, mission_id: str, also_mission_id: Optional[str] = None,
) -> List[Tuple[LiveActivity, LiveActivityDevice]]:
    """Started activities for ``mission_id`` — and, when given, for
    ``also_mission_id`` (Round 8: a job row's raw job id, the name the
    app's locally-started job card registers under). One row per device
    when both names are up on it, and the LOCAL (``also_mission_id``) row
    wins: it is the newer card and always carries a per-activity token
    (adoption requires one), whereas an update to the platform's row over
    the shared push-to-start token would preempt-end the local card to
    disambiguate — the exact duplicate/flap this exists to prevent."""
    names = [mission_id] + ([also_mission_id] if also_mission_id else [])
    result = await db.execute(
        select(LiveActivity, LiveActivityDevice)
        .join(LiveActivityDevice, LiveActivity.device_id == LiveActivityDevice.id)
        .where(
            LiveActivity.user_id == user_id,
            LiveActivity.mission_id.in_(names),
            LiveActivity.status == LA_STARTED,
        )
    )
    rows = [(la, dev) for la, dev in result.all()]
    if not also_mission_id or len(rows) < 2:
        return rows
    by_device: Dict[str, Tuple[LiveActivity, LiveActivityDevice]] = {}
    for la, dev in rows:
        prev = by_device.get(la.device_id)
        if prev is None or (prev[0].mission_id != also_mission_id
                            and la.mission_id == also_mission_id):
            by_device[la.device_id] = (la, dev)
    return list(by_device.values())


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
        if la.mission_id.startswith("voice:"):
            # A LIVE CALL is not a card in the rotation — ActivityKit runs
            # multiple activities side by side, and a mission/reminder start
            # must never end the island presence of a conversation in
            # progress (review finding, 2026-08-16). The call's own enders
            # (relay disconnect, staleDate, app reconcile) retire it.
            continue
        if (
            la.mission_id.startswith("reminder:")
            and keep_mission_id.startswith(_NEVER_PREEMPT_REMINDER_PREFIXES)
        ):
            # REMINDER WINS, enforced at the preempt itself (2026-07-22
            # incident: the Answer-ready restart force-ended the live
            # countdown row here; the end push no-op'd on-device, the
            # platform lost track of the visible card, and the fire had
            # to restart a bare 0% card). A chat-turn card must never
            # kill a user-scheduled countdown, and neither may an admin
            # announcement (same founder rule — an operator message is
            # not more urgent than the alarm the user set) — the
            # callers' yield guards make this path unreachable for both,
            # and if a future caller slips through we keep the countdown
            # row and accept shared-token ambiguity as the lesser evil.
            logger.info(
                "live-activity preempt: refusing to end %s for %s",
                la.mission_id, keep_mission_id,
            )
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
    # ORDER: before _preempt_device. A reminder-countdown START whose fire
    # instant has already passed is a straggler (its inline fast-lane
    # dispatch failed and the 30s loop picked it up after the FIRE row —
    # which rides the fast lane since 2026-08-19). Arming a countdown over
    # the ring — or preempting the ringing card to do it — resurrects a
    # timer for a reminder that already fired.
    if (
        mission_id.startswith("reminder:")
        and row.event_kind == NOTIFY_KIND_MISSION_STARTED
    ):
        _end = _timer_end_ms(row)
        if _end and _end < int(now.timestamp() * 1000) - 5000:
            return {"status": "skipped", "reason": "countdown_expired"}
    await _preempt_device(db, device, mission_id, now)

    progress = _progress_fraction(row)
    data = row.data_json or {}
    # Card-tap target. A producer-supplied data.deep_link is honored ONLY
    # from a PLATFORM row carrying a toup:// URL — the admin dispatch
    # fan-out is the one producer, and its links already carry their own
    # ?mission=. Both halves are load-bearing: POST /api/agent/notify
    # writes source='agent' and copies any scalar data key verbatim from a
    # caller holding a TENANT X-Agent-Key, so without this a compromised
    # container could aim the user's own lock-screen card at any URL iOS
    # will open. Every other case falls through to the platform-computed
    # link, which is also the fail-closed direction: the column defaults to
    # 'agent' and an unflushed row reads None. Chat turns land in the
    # conversation where the answer lives, everything else keeps Mission
    # Control; the mission id rides as a query param so the app can ACK the
    # tap (end the card everywhere + stop re-alerts for this mission).
    explicit_link = data.get("deep_link")
    if (
        row.source == "platform"
        and isinstance(explicit_link, str)
        and explicit_link.startswith("toup://")
    ):
        deep_link = explicit_link
    else:
        if explicit_link:
            # Either a platform producer emitted a non-toup:// link (a bug)
            # or a tenant tried to choose the tap target (an attack) — both
            # are worth one line on the trail; neither changes the outcome.
            logger.info(
                "live-activity: ignoring data.deep_link on a %s row (mission %s)",
                row.source, mission_id,
            )
        # R28: an automation run's tap lands on its session run card —
        # see automation_deep_link for the trust posture (computed from
        # validated tokens, never a tenant-chosen URL).
        _auto_link = automation_deep_link(data, mission_id)
        if _auto_link:
            deep_link = _auto_link
        else:
            base_link = "toup://chat" if data.get("route") == "chat" else "toup://mission-control"
            # Round 3 (item 3): the conversation + answer ids ride the tap URL
            # too, so a tap can open the thread and scroll to the reply. The
            # ids come from the same producer as the mission id and are
            # validated as short opaque tokens (_safe_id) — no URL is ever
            # taken from a tenant here.
            deep_link = f"{base_link}?mission={mission_id}{_deep_link_params(row)}"
    timer_ms = _timer_end_ms(row)
    # INSTANT-OPEN seed (2026-07-23): reminder cards carry the reminder
    # text + fire instant on the tap URL so the app can render the
    # message the user is navigating to IMMEDIATELY — before any
    # network round-trip (the fire row persists server-side ~1s after
    # T-0; the old tap showed a stale thread, then the message popped
    # in). rtext clipped to 120 pre-encoding keeps the worst-case
    # percent-encoded URL (~1.1KB) inside the 4KB APNs LA start budget.
    # rat = the fire instant: countdown cards pass their timer end (the
    # app's pre-fire gate refuses to seed before it), fire cards pass
    # now. Old app binaries ignore the extra params.
    _subtitle_for_link = data.get("subtitle") if isinstance(data.get("subtitle"), str) else None
    if not _subtitle_for_link and isinstance(row.body, str) and row.body.strip():
        # Countdown arm rows carry the reminder text as row.body, not
        # data.subtitle (review F3) — without this fallback the most
        # common reminder surface never seeded.
        _subtitle_for_link = row.body
    if data.get("route") == "chat" and mission_id.startswith("reminder:") and _subtitle_for_link:
        _rat = timer_ms or int(now.timestamp() * 1000)
        # Cap the ENCODED length, not the character count (review F5):
        # emoji percent-encode at ~12 bytes/char, and the 4KB APNs
        # budget is shared with body/content-state copies of the text.
        _sub = _subtitle_for_link[:120]
        _q = urllib.parse.quote(_sub)
        while len(_q) > 600 and _sub:
            _sub = _sub[:-8]
            _q = urllib.parse.quote(_sub)
        deep_link = f"{deep_link}&rtext={_q}&rat={_rat}"

    # Stale backstop: a countdown card goes visually stale 2 minutes
    # after its timer fires; a non-timer card after 30 minutes with no
    # update. Progress updates refresh the horizon — only a card whose
    # pushes are LOST can ever dim.
    stale_ts = (
        int(timer_ms / 1000) + 120 if timer_ms
        else int(now.timestamp()) + 1800
    )
    # Producer subtitle override (e.g. reminder countdowns carry the
    # reminder text); non-silent starts keep the legacy 'Starting…'.
    subtitle_override = data.get("subtitle") if isinstance(data.get("subtitle"), str) else None
    if subtitle_override is None and row.event_kind == NOTIFY_KIND_ANNOUNCEMENT:
        # An announcement narrates no work: the operator's message IS
        # the card and it rides on row.body (the admin fan-out builds
        # data_json without a subtitle). 'Starting…' under an operator's
        # title would be a lie.
        subtitle_override = row.body or row.title
    timer_type = data.get("timer_type")
    # An alarm-class terminal row restarting its card IS the fire
    # moment: the fresh card renders the ringing presentation, never a
    # bare 0% progress bar (the fire row carries no timer/progress —
    # 2026-07-22 founder repro: '0% silently, then a stale 100% bar').
    fired = (
        _is_alarm_row(row)
        and row.event_kind in (NOTIFY_KIND_MISSION_COMPLETED,
                               NOTIFY_KIND_MISSION_FAILED)
    )
    payload = apns_push.build_start_payload(
        mission_id=mission_id,
        title=_mission_title(row),
        subtitle=(row.body or "Working…")[:120] if silent
                 else (subtitle_override or "Starting…")[:120],
        # An announcement narrates NO WORK, so it must carry no progress at
        # all — not "no progress yet". The client treats these as bound
        # optionals (`else if let progress = contentState.progress`), so a
        # substituted 0.0 does not read as absent, it reads as zero percent:
        # LiveActivityWidget.swift:407 renders Text("0%") in the Dynamic
        # Island and :430 / LiveActivityView.swift:253 render an empty
        # ProgressView. `nil` renders neither. That is the founder's
        # 2026-08-13 report — an operator's message under a progress bar for
        # a job that does not exist.
        #
        # The substitution stays for every other kind, where a start really
        # is at zero percent, and the `fired` flag below is the same escape
        # already built for alarm rows (see its comment, and the identical
        # 2026-07-22 repro).
        # An announcement narrates NO WORK, so it must carry no progress at
        # all — not "no progress yet". The client treats these as bound
        # optionals (`else if let progress = contentState.progress`), so a
        # substituted 0.0 does not read as absent, it reads as zero percent:
        # LiveActivityWidget.swift:407 renders Text("0%") in the Dynamic
        # Island and :430 / LiveActivityView.swift:253 render an empty
        # ProgressView. `nil` renders neither. That is the founder's
        # 2026-08-13 report — an operator's message under a progress bar for
        # a job that does not exist.
        #
        # The substitution stays for every other kind, where a start really
        # is at zero percent, and the `fired` flag below is the same escape
        # already built for alarm rows (see its comment, and the identical
        # 2026-07-22 repro).
        progress=(
            None if row.event_kind == NOTIFY_KIND_ANNOUNCEMENT
            else (progress if progress is not None else 0.0)
        ),
        timer_end_ms=timer_ms,
        timer_start_ms=_timer_start_ms(row),
        alert_title=None if silent else row.title,
        alert_body=None if silent else row.body,
        # The operator's chosen dispatch tone, resolved to the file name the
        # app bundles. `apns_push` drops the name on every ActivityKit
        # payload (a name there is silence, not a fallback — see the
        # alarm-class note below), so what this actually buys is ONE place
        # that knows how `data.tone` becomes an APNs sound value.
        alert_sound=dispatch_tones.apns_sound(data.get("tone")),
        timestamp=int(now.timestamp()),
        deep_link=deep_link,
        timer_type=timer_type if timer_type in ("circular", "digital") else None,
        orb_color=(
            _ADMIN_ORB_COLOR if row.event_kind == NOTIFY_KIND_ANNOUNCEMENT
            else await _user_orb_color(db, row.user_id)
        ),
        stale_date=stale_ts,
        fired=fired,
        extra=_row_extra_state(row),
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
            # Fresh card = fresh alarm-ownership cycle: the app re-arms
            # its device alarm off this very start (countdown-armed
            # sync) and re-reports ownership if it still holds one.
            existing.alarm_owned_at = None
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
    # Round 8: a job row also addresses the card the app started for the
    # job under its raw id (see _job_id_of).
    job_alias = _job_id_of(row)

    if row.event_kind in _START_EVENT_KINDS:
        # data.silent → card appears without a banner (reminder
        # countdown arms: the user just got chat confirmation — the
        # card IS the feedback).
        start_silent = bool((row.data_json or {}).get("silent"))
        devices = await _active_devices(db, row.user_id)
        if not devices:
            return {"status": "skipped", "reason": "no_live_activity_devices"}
        # Belt-and-braces: key on the mission prefix too, so a future
        # chatturn/admin producer that forgets data.kind still yields
        # (the _preempt_device refusal keys on the prefix — a kind-less
        # chatturn start would otherwise insert-race the partial unique
        # index forever).
        yields_to_reminder = (
            (row.data_json or {}).get("kind") in ("chat_turn", "announcement")
            or mission_id.startswith(_NEVER_PREEMPT_REMINDER_PREFIXES)
        )
        for device in devices:
            started = await _started_rows_for_device(db, device.id)
            _same = next((la for la in started
                          if la.mission_id == mission_id
                          or (job_alias and la.mission_id == job_alias)), None)
            if _same is not None:
                if (
                    (row.data_json or {}).get("refresh_if_started")
                    and _row_is_newer_than_card(row, _same)
                ):
                    # A NEW job on the conversation's card (Round 3, item
                    # 4): update the card the previous job left up. A row
                    # older than the card's last push is the retry of the
                    # start that opened it — that one still dedups.
                    result = await _refresh_started_card(
                        db, _same, device, row, mission_id, now,
                    )
                    per_device[device.id] = result
                    delivered = delivered or result["status"] == "ok"
                    errored = errored or result["status"] == "error"
                    continue
                # at-least-once retry after a partial failure — a second
                # start push would spawn a duplicate card on screen.
                per_device[device.id] = {"status": "skipped", "reason": "already_started"}
                continue
            if yields_to_reminder and any(
                la.mission_id.startswith("reminder:") for la in started
            ):
                # REMINDER WINS (founder rule 2026-07-20): a working
                # chat-turn card must never preempt a live countdown —
                # the preempt end rides the shared token, routinely
                # no-ops on-device, and the fire's restart then stacks
                # a duplicate card. The answer still arrives as its
                # own banner; only the ambient working card yields. An
                # operator announcement yields for the same reason: it
                # is not more urgent than the alarm the user set, and
                # it also lands as a chat card the user keeps.
                per_device[device.id] = {"status": "skipped", "reason": "yields_to_reminder"}
                continue
            if _suppressed_by_presence(device, mission_id, now):
                # The user is IN the app. The thread, the Activity list and
                # the Admin inbox are all better surfaces than a card in the
                # island describing what is already on their screen.
                per_device[device.id] = {"status": "skipped", "reason": "app_in_foreground"}
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

    activities = await _activities_for_mission(db, row.user_id, mission_id, job_alias)
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
        is_chat_turn = _is_chat_turn_mission(row, mission_id)
        for device in devices:
            if is_chat_turn and any(
                la.mission_id.startswith("reminder:")
                for la in await _started_rows_for_device(db, device.id)
            ):
                # REMINDER WINS in the restart lane too (2026-07-22
                # founder repro: the 'Answer ready' terminal row
                # push-to-started a chat-turn card 20s after a countdown
                # armed — most-recently-started owns the Dynamic Island,
                # so the job card squatted the island for its 5-minute
                # linger while the live countdown was demoted to the
                # lock screen, and the start's preempt orphaned the
                # countdown row). The answer already reached the chat;
                # only this ambient card yields. Mirror of the
                # mission_started guard above.
                per_device[device.id] = {
                    "status": "skipped", "reason": "yields_to_reminder",
                }
                continue
            if _is_alarm_row(row):
                prior = await _row_for(db, device.id, mission_id)
                if prior is not None and prior.alarm_owned_at is not None:
                    # This device's AlarmKit alarm owns the fire moment
                    # (the app retired the countdown card when it armed
                    # the alarm). A platform loud restart on top would
                    # double-alert — AlarmKit is already ringing through
                    # silent/Focus, strictly louder than anything this
                    # lane can do.
                    per_device[device.id] = {
                        "status": "skipped", "reason": "alarm_owned",
                    }
                    continue
            if _suppressed_by_presence(device, mission_id, now):
                # THE ONE THE FOUNDER SAW. "Answer ready" is a
                # mission_completed row, mission_completed is in
                # _START_IF_MISSING_KINDS, and no card exists for a turn the
                # user never left — so every answer to a question asked and
                # waited for in the app push-STARTED a loud card, which then
                # sat in the island for its 5-minute dismiss window. The
                # answer still reaches them: it is already in the thread they
                # are looking at, and the push notification is unaffected.
                per_device[device.id] = {"status": "skipped", "reason": "app_in_foreground"}
                continue
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
        activities = await _activities_for_mission(db, row.user_id, mission_id, job_alias)

    if not activities:
        frag_skip: Dict[str, Any] = {
            "status": "skipped", "reason": "no_active_activity",
        }
        if per_device:
            # The restart lane ran and skipped (yields_to_reminder /
            # alarm_owned) — keep the per-device verdicts on the row so
            # ladebug shows WHY nothing was pushed.
            frag_skip["devices"] = per_device
        return frag_skip

    progress = _progress_fraction(row)
    title = _mission_title(row)
    # Round 3: icon tag, step n/m, percent, preview, deep-link ids ride
    # every update/end (see apns_push._extra_state for the key contract).
    extra_state = _row_extra_state(row)

    for la, device in activities:
        # Never move the on-screen bar backwards on reordered rows.
        effective = progress
        if effective is not None and la.last_progress is not None:
            effective = max(effective, la.last_progress / 100.0)

        if row.event_kind == NOTIFY_KIND_PROGRESS:
            # Round 8: a JOB progress row that would move the bar backwards
            # is a reordered/retried OLDER tick — its step line and n/m
            # counts are older too. The content state is replaced whole by
            # every push, so "clamp the bar, apply the rest" put a stale
            # "1/3 · Compare evidence" over a card the app already showed
            # at 2/3. Drop the row for this device instead. Job rows only:
            # their percent is a function of steps done, so lower means
            # older; a mission's self-estimated percent may honestly fall.
            if (
                job_alias is not None or (row.data_json or {}).get("kind") == "job"
            ) and progress is not None and la.last_progress is not None \
                    and int(round(progress * 100)) < la.last_progress:
                per_device[device.id] = {"status": "skipped", "reason": "stale_progress"}
                continue
            headline = (row.body or row.title or "Working…")[:120]
            payload = apns_push.build_update_payload(
                title=title, subtitle=headline, progress=effective,
                timer_end_ms=_timer_end_ms(row),
                timer_start_ms=_timer_start_ms(row),
                stale_date=int(now.timestamp()) + 1800,
                timestamp=int(now.timestamp()),
                extra=extra_state,
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
                extra=extra_state,
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
            _alarm = _is_alarm_row(row)
            _subtitle = _data.get("subtitle") if isinstance(_data.get("subtitle"), str) else None
            final_subtitle = _subtitle or ("Completed ✓" if done else "Stopped — needs attention")
            # Alarm-class terminal rows render the RINGING presentation
            # (content-state.fired), never progress math: 1.0 on a fire
            # card paints the stale full bar the founder reproduced
            # 2026-07-22; the widget shows a bell + 'Ringing' instead.
            final_progress = None if _alarm else (1.0 if done else effective)
            if _alarm and not _silent_end and la.alarm_owned_at is not None:
                # This device's AlarmKit alarm owns the fire — it is
                # already ringing through silent/Focus. Close our card
                # quietly and contribute nothing to the ring chain.
                payload = apns_push.build_end_payload(
                    title=title, subtitle=final_subtitle, fired=True,
                    dismissal_date=_dismissal_date(row, now)
                    or int(now.timestamp()) + 120,
                    timestamp=int(now.timestamp()),
                )
                result = await _send_to_activity(
                    db, la, device, payload, priority=10, now=now, end=True,
                )
                # The fire CONSUMES the ownership cycle: without this
                # clear, a later fire of the same mission that never
                # re-armed (reschedule beyond the countdown window —
                # no card, no re-report) would be muted by the stale
                # flag — recreating the silent-T-0 incident. Marker
                # key kept separate from `reason` so a failed end's
                # APNs verdict stays readable in ladebug.
                la.alarm_owned_at = None
                result = {**result, "alarm_owned": True}
                per_device[device.id] = result
                delivered = delivered or result["status"] == "ok"
                continue
            # Alarm-class rows ring more than once: every ring except
            # the last is an alerting UPDATE that leaves the card open
            # (already showing the fired state) for the next chained
            # ring row — or for the tap ack, which ends the card and
            # suppresses the remaining rings. Only the last ring
            # closes the card.
            _last_ring = _silent_end or _realert_seq(row) >= _alarm_rings(row)
            # Round 3 (item 2): a producer may ask for the completed state
            # to SHOW before the card ends (data.end_after_s) — the end
            # then rides a chained queue row one delay out (see
            # _enqueue_deferred_end); ``end_only`` marks that follow-up,
            # which sends nothing but the end. Alarm-class rows keep
            # their ring chain; silent ends have nothing to show.
            _defer_s = _end_after_s(row)
            _defer = (
                _defer_s is not None and not _alarm and not _silent_end
                and not _data.get("end_only")
            )
            # The audible banner rides an alerting UPDATE (Apple's
            # documented alert surface — the end-event alert is
            # undocumented and proved unreliable on iOS 26), then a
            # bannerless end closes the card. Skip the update when the
            # loud restart already alerted, or the producer asked for
            # silence. With a deferred end after a loud restart the
            # update still goes — bannerless — because the restart
            # opened the card reading "Starting…", and that state would
            # otherwise be what lingers.
            alerted = restart_loud
            upd_result: Optional[Dict[str, Any]] = None
            if not _silent_end and (not alerted or _defer):
                upd = apns_push.build_update_payload(
                    title=title, subtitle=final_subtitle,
                    progress=final_progress,
                    fired=_alarm,
                    alert_title=None if alerted else row.title,
                    alert_body=None if alerted else row.body,
                    timestamp=int(now.timestamp()),
                    extra=extra_state,
                )
                upd_result = await _send_to_activity(
                    db, la, device, upd, priority=10, now=now,
                )
                alerted = alerted or upd_result["status"] == "ok"
            if not _last_ring:
                # Loud restart counts as this ring's bang: the card is
                # freshly started in fired state, nothing else to send.
                result = upd_result or {"status": "ok", "reason": "restart_alerted"}
            elif _defer:
                booked = await _enqueue_deferred_end(db, row, mission_id, now, _defer_s)
                result = upd_result or {"status": "ok", "reason": "restart_alerted"}
                result = {**result, "end_deferred_s": _defer_s, "end_booked": booked}
            else:
                payload = apns_push.build_end_payload(
                    title=title,
                    subtitle=final_subtitle,
                    progress=final_progress,
                    fired=_alarm,
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
                    extra=extra_state,
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

    # Alarm chain: once THIS ring actually RANG a device, book the next
    # one. Gated on a real ring — not mere delivery — so an
    # alarm-owned quiet close (AlarmKit is doing the ringing) never
    # chains platform rings on top of it, while a failed ring retries
    # via the row's own backoff instead of forking the chain; the
    # idempotency key makes the booking safe to repeat on
    # partial-failure retries.
    rang = any(
        isinstance(r, dict)
        and r.get("status") == "ok"
        and not r.get("alarm_owned")
        for r in per_device.values()
    )
    if (
        rang
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


# ── Voice call cards ─────────────────────────────────────────────────────
# The voice Live Activity is LOCALLY started by the app (voiceActivity.ios.ts,
# mission "voice:<uuid>") and its per-activity token reaches us through the
# same report_activity_token adoption path chat turns use. This is the other
# half of that contract: when the realtime session dies — force-quit is the
# one that matters, because no app code runs — ws_realtime's finally calls
# this, and the island/Lock-Screen card ends within seconds instead of
# claiming "Listening…" for up to Apple's 8-hour cap (2026-08-16 recording B).

async def end_voice_activities(user_id: str, mission_id: Optional[str]) -> int:
    """End the started card(s) of ONE voice call: APNs end push (best-effort)
    + the DB end (the invariant). ``mission_id`` is REQUIRED in effect — a
    session that never identified its card (every web voice session, older
    app builds) ends nothing. The earlier fallback swept every ``voice:`` row
    for the user, which meant a web session's close killed the island card of
    a live PHONE call (review finding, 2026-08-16); a card that never got an
    identified end still degrades via its staleDate and the app reconcile.

    Opens its own session: the caller is a websocket finally block, not a
    request with a Depends(db)."""
    from app.db.database import async_session_maker

    if not mission_id or not mission_id.startswith("voice:"):
        return 0
    now = datetime.utcnow()
    ended = 0
    async with async_session_maker() as db:
        q = (
            select(LiveActivity, LiveActivityDevice)
            .join(LiveActivityDevice, LiveActivity.device_id == LiveActivityDevice.id)
            .where(
                LiveActivity.user_id == user_id,
                LiveActivity.mission_id == mission_id,
                LiveActivity.status == LA_STARTED,
            )
        )
        rows = (await db.execute(q)).all()
        for la, device in rows:
            if not la.mission_id.startswith("voice:"):
                continue  # a narrowed mission_id must still be a voice card
            token = la.activity_push_token or device.push_to_start_token
            if token and apns_push.apns_configured():
                payload = apns_push.build_end_payload(
                    title="Call ended",
                    dismissal_date=int(now.timestamp()) - 1,
                    timestamp=int(now.timestamp()),
                )
                # Best-effort, same rule as the ack path: the DB end below is
                # the invariant; the push just clears the screen faster than
                # staleness would.
                try:
                    await apns_push.send_live_activity(
                        token, payload,
                        environment=device.apns_environment or "development",
                        priority=10,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("voice LA end push failed for %s", la.mission_id)
            la.status = LA_ENDED
            la.ended_at = now
            la.updated_at = now
            ended += 1
        if ended:
            await db.commit()
            logger.info("voice LA end: %d card(s) for user %s", ended, user_id[:8])
    return ended
