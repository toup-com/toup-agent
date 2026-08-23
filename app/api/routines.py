"""Routines HTTP surface.

User-facing CRUD + force-run + run-history endpoints. All routes assume
single-user-per-container (agent mode); user_id is read from
`settings.user_id`, NOT resolved from the request. The `X-Agent-Key`
middleware in `agent_main.py` is what gates request access.

Feature gate: `email_briefing` kind requires
`settings.routines_email_briefing_enabled`. Other kinds (none yet) will
add their own flags.

One-per-kind constraint is enforced HERE (`POST` returns 409 if an
enabled routine of the same kind already exists). The DB schema permits
multiple per kind so weekday/weekend variants stay possible later.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

from apscheduler.triggers.cron import CronTrigger
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator


def _utc_iso(v: Optional[datetime]) -> Optional[str]:
    """Emit RFC 3339 UTC with trailing ``Z``. Mirrors
    ``app.api.triggers._utc_iso`` — see that docstring for the
    rationale (2026-05-14 dashboard timezone bug). Duplicated rather
    than imported to keep the two API modules' blast radius
    independent."""
    if v is None:
        return None
    if v.tzinfo is None:
        return v.isoformat() + "Z"
    return v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
from sqlalchemy import delete, desc, select
from sqlalchemy.exc import IntegrityError


router = APIRouter(prefix="/routines", tags=["routines"])
logger = logging.getLogger(__name__)

_runner = None  # set by agent_main.py via set_runner_ref()


def set_runner_ref(runner) -> None:
    """Called by agent_main.py after RoutineRunner.start() succeeds."""
    global _runner
    _runner = runner


# ── Helpers ────────────────────────────────────────────────────────────


def _user_id() -> str:
    """Single-user-per-container. user_id is the container-owner's id,
    injected at boot via `USER_ID` env var → `settings.user_id`."""
    from app.config import settings
    return settings.user_id


def _validate_cron(expr: str) -> None:
    """Reject obviously-broken cron expressions BEFORE persisting. We
    construct an APScheduler `CronTrigger` — same parser the runner
    uses at trigger registration — and catch its exception."""
    parts = (expr or "").split()
    if len(parts) != 5:
        raise HTTPException(
            status_code=400,
            detail=f"schedule_cron_local must be a 5-part cron expression "
                   f"(got {len(parts)} parts: {expr!r})",
        )
    try:
        CronTrigger(
            minute=parts[0], hour=parts[1], day=parts[2],
            month=parts[3], day_of_week=parts[4],
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid cron expression {expr!r}: {e}",
        )


def _validate_kind(kind: str) -> None:
    """Require kind to be a registered handler. Prevents typos and
    future-kind rows from being created before their handler ships."""
    from app.agent.routines.registry import KIND_HANDLERS
    if kind not in KIND_HANDLERS:
        registered = sorted(KIND_HANDLERS.keys()) or ["(none)"]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown routine kind {kind!r}. Registered: {registered}",
        )


def _kind_enabled_or_404(kind: str) -> None:
    """Feature-flag gate, mirrors `RoutineRunner._kind_enabled`. A
    disabled-flag routine should be invisible — 404, not 403 — so
    Mission Control can render an empty state without leaking the
    existence of the surface."""
    from app.config import settings
    if kind == "email_briefing":
        if not getattr(settings, "routines_email_briefing_enabled", False):
            raise HTTPException(status_code=404, detail="Feature not available")


# ── Reminder countdown Live Activity (2026-07-17) ─────────────────────
#
# Near-term one-shot reminders arm an on-device countdown card at
# creation (notify → LA push-to-start with timer_end_ms; the widget
# animates the bar with zero update pushes). The fire-time notify in
# ReminderHandler shares the mission_id and flips the card into the
# urgent "now" alert; cancel/delete/reschedule ends it silently.
# ActivityKit hard-caps an activity at 8h of active life — beyond the
# window there is no countdown (the fire alert still renders via the
# LA lane's restart-if-missing). Recurring reminders get NO countdown:
# a perpetual card would permanently occupy the one-activity-per-device
# slot.

_REMINDER_COUNTDOWN_WINDOW_S = 8 * 3600 - 600  # 8h ActivityKit cap − 10min margin


def _reminder_countdown_eligible(routine) -> bool:
    from app.config import settings
    if not getattr(settings, "reminder_countdown_live_activity_enabled", True):
        return False
    if routine.kind != "reminder" or not routine.enabled:
        return False
    if (getattr(routine, "schedule_kind", None) or "cron") != "at":
        return False
    at = getattr(routine, "schedule_at", None)
    if at is None:
        return False
    delta = (at - datetime.utcnow()).total_seconds()
    return 0 < delta <= _REMINDER_COUNTDOWN_WINDOW_S


async def _reminder_countdown_notify(routine) -> None:
    """Arm the countdown card. Best-effort by contract — the CRUD
    response must never fail on notification plumbing."""
    try:
        from app.services.agent_notify_client import notify

        name = (routine.name or routine.reminder_text or "Reminder").strip()[:80]
        # schedule_at is stored UTC-NAIVE (RoutineCreate strips tzinfo)
        # — attach utc before the epoch-ms conversion or a bare
        # .timestamp() applies the container's local offset.
        end_ms = int(
            routine.schedule_at.replace(tzinfo=timezone.utc).timestamp() * 1000
        )
        # The SET instant: the card's bar draws the absolute set→fire span
        # (same track the in-app card fills) instead of restarting at every
        # widget rebuild. created_at is UTC-naive like schedule_at.
        start_ms = None
        try:
            if routine.created_at:
                start_ms = int(
                    routine.created_at.replace(tzinfo=timezone.utc).timestamp() * 1000
                )
        except Exception:  # noqa: BLE001 — the card works without a start
            start_ms = None
        await notify(
            event_kind="mission_started",
            title=f"⏰ {name}"[:200],
            body=(routine.reminder_text or "")[:400] or None,
            data={
                "mission_id": f"reminder:{routine.id}",
                "mission_title": f"⏰ {name}"[:80],
                "kind": "reminder",
                "route": "chat",
                "timer_end_ms": end_ms,
                **({"timer_start_ms": start_ms} if start_ms else {}),
                "timer_type": "digital",
                # Round 4 (item 5a): dispatch inline on ingest — a short
                # reminder's countdown must be on the phone within a
                # second, not after the next 30 s tick.
                "fast_lane": True,
                # silent: the card appears with no banner — the user
                # just got chat confirmation; the card IS the feedback.
                "silent": True,
                # urgent: a reminder set at 23:00 for 07:00 gets its
                # card immediately, not after quiet hours.
                "urgent": True,
                "progress": 0,
            },
            priority="default",
            # No dedup_key: the LA lane's per-device already_started
            # guard dedups; a window here would swallow the re-arm
            # right after a reschedule.
            dedup_key=None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[routines] countdown notify failed routine_id=%s: %s",
            getattr(routine, "id", "?"), e,
        )


async def _reminder_cancel_notify(routine_id: str) -> None:
    """Silent end + immediate dismissal for a canceled/disabled/
    rescheduled reminder's countdown card. silent suppresses both the
    banner AND the LA lane's restart-if-missing (nothing to alert); if
    no card is live the lane skips 'no_active_activity' harmlessly.
    urgent so the end push isn't deferred by quiet hours — a stale
    countdown sitting on the lock screen all night is worse than a
    silent nighttime end."""
    try:
        from app.services.agent_notify_client import notify

        await notify(
            event_kind="mission_completed",
            title="Reminder canceled",
            data={
                "mission_id": f"reminder:{routine_id}",
                "kind": "reminder",
                "route": "chat",
                "silent": True,
                "urgent": True,
                "cap_exempt": True,
                "dismiss_after_s": 0,
                "no_agent_fallback": True,
            },
            priority="low",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[routines] cancel notify failed routine_id=%s: %s", routine_id, e,
        )


# ── Pydantic models ────────────────────────────────────────────────────


_VALID_DELIVERY_CHANNELS = {"website", "telegram", "whatsapp"}
_VALID_SCHEDULE_KINDS = {"cron", "at", "every"}
_VALID_ROUTINE_KINDS = {"email_briefing", "agent_task", "reminder"}
_MIN_INTERVAL_SECONDS = 60


def _validate_window_str(v: Optional[str], field: str) -> Optional[str]:
    """HH:MM or HH:MM:SS → canonical HH:MM:SS. Used for window bounds."""
    if v is None:
        return None
    parts = v.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{field} must be HH:MM or HH:MM:SS (got {v!r})")
    try:
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) == 3 else 0
        if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
            raise ValueError(f"{field} out of range: {v!r}")
    except ValueError as e:
        raise ValueError(f"{field} parse error: {e}")
    return f"{h:02d}:{m:02d}:{s:02d}"


class RoutineCreate(BaseModel):
    kind: str = Field(
        ...,
        description=(
            "Routine handler kind: 'email_briefing' (Gmail summary), "
            "'agent_task' (generic LLM-driven prompt), or 'reminder' "
            "(text-only delivery, no LLM/MCP)."
        ),
    )
    # Mig 042 — schedule shape selector. Legacy callers passing only
    # `schedule_cron_local` default to 'cron' and keep working.
    schedule_kind: str = Field(
        default="cron",
        description="'cron' (default) | 'at' (one-shot) | 'every' (interval).",
    )
    schedule_cron_local: Optional[str] = Field(
        default=None,
        description="5-part cron in the user's tz. Required for schedule_kind='cron'.",
        examples=["30 6 * * *"],
    )
    schedule_at: Optional[datetime] = Field(
        default=None,
        description="UTC datetime. Required for schedule_kind='at'. Must be future.",
    )
    schedule_interval_seconds: Optional[int] = Field(
        default=None,
        description="Seconds between fires. Required for schedule_kind='every'. Min 60.",
    )
    schedule_window_start_local: Optional[str] = Field(
        default=None,
        description="HH:MM in user's tz. Daily window-start for interval reminders.",
    )
    schedule_window_end_local: Optional[str] = Field(
        default=None,
        description="HH:MM in user's tz. Daily window-end for interval reminders.",
    )
    auto_disable_after_fire: Optional[bool] = Field(
        default=None,
        description="When true, routine disables itself after a successful fire. "
                    "Server-default true for schedule_kind='at'.",
    )
    reminder_text: Optional[str] = Field(
        default=None, max_length=4000,
        description="Literal text to deliver. Required for kind='reminder'.",
    )
    name: Optional[str] = Field(default=None, max_length=100,
        description="User-visible name. Defaults to a kind-based label.")
    prompt_text: Optional[str] = Field(default=None,
        description="Required when kind='agent_task'. The natural-language "
                    "task the agent runs at fire time.")
    enabled: bool = True
    delivery_channels: Optional[list[str]] = Field(
        default=None,
        description="Subset of ['website','telegram','whatsapp']. Website is "
                    "always included; listing telegram/whatsapp fans the "
                    "summary out to those channels too.",
    )
    config: Optional[dict] = None

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, v: str) -> str:
        if v not in _VALID_ROUTINE_KINDS:
            raise ValueError(
                f"Unknown kind {v!r}. Allowed: {sorted(_VALID_ROUTINE_KINDS)}"
            )
        return v

    @field_validator("schedule_kind")
    @classmethod
    def _validate_schedule_kind(cls, v: str) -> str:
        if v not in _VALID_SCHEDULE_KINDS:
            raise ValueError(
                f"Unknown schedule_kind {v!r}. Allowed: {sorted(_VALID_SCHEDULE_KINDS)}"
            )
        return v

    @field_validator("schedule_window_start_local")
    @classmethod
    def _validate_start(cls, v: Optional[str]) -> Optional[str]:
        return _validate_window_str(v, "schedule_window_start_local")

    @field_validator("schedule_window_end_local")
    @classmethod
    def _validate_end(cls, v: Optional[str]) -> Optional[str]:
        return _validate_window_str(v, "schedule_window_end_local")

    @field_validator("delivery_channels")
    @classmethod
    def _validate_delivery_channels(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        bad = [c for c in v if c not in _VALID_DELIVERY_CHANNELS]
        if bad:
            raise ValueError(
                f"Unknown delivery channels: {bad}. "
                f"Allowed: {sorted(_VALID_DELIVERY_CHANNELS)}"
            )
        return v

    @model_validator(mode="after")
    def _validate_shape_invariants(self):
        """Cross-field invariants. Surfaces clean 400s instead of letting
        downstream code blow up on missing fields."""
        sk = self.schedule_kind
        if sk == "cron":
            if not (self.schedule_cron_local or "").strip():
                raise ValueError(
                    "schedule_cron_local is required when schedule_kind='cron'."
                )
        elif sk == "at":
            if self.schedule_at is None:
                raise ValueError(
                    "schedule_at is required when schedule_kind='at'."
                )
            # The routines.schedule_at column is `TIMESTAMP WITHOUT TIME
            # ZONE`. Pydantic parses an ISO string with a Z suffix into a
            # tz-aware datetime, which asyncpg refuses to bind to the
            # naive column. Strip the offset HERE (not just locally for
            # the comparison) so every downstream consumer sees a single
            # canonical naive-UTC value.
            if self.schedule_at.tzinfo is not None:
                from datetime import timezone as _tz
                self.schedule_at = self.schedule_at.astimezone(
                    _tz.utc
                ).replace(tzinfo=None)
            now = datetime.utcnow()
            if self.schedule_at <= now:
                raise ValueError(
                    f"schedule_at must be in the future "
                    f"(got {self.schedule_at.isoformat()} vs now {now.isoformat()})."
                )
        elif sk == "every":
            if self.schedule_interval_seconds is None:
                raise ValueError(
                    "schedule_interval_seconds is required when schedule_kind='every'."
                )
            if self.schedule_interval_seconds < _MIN_INTERVAL_SECONDS:
                raise ValueError(
                    f"schedule_interval_seconds must be >= "
                    f"{_MIN_INTERVAL_SECONDS} (got {self.schedule_interval_seconds})."
                )

        if self.kind == "reminder" and not (self.reminder_text or "").strip():
            raise ValueError("reminder_text is required when kind='reminder'.")

        return self


class RoutineUpdate(BaseModel):
    schedule_kind: Optional[str] = None
    schedule_cron_local: Optional[str] = None
    schedule_at: Optional[datetime] = None
    schedule_interval_seconds: Optional[int] = None
    schedule_window_start_local: Optional[str] = None
    schedule_window_end_local: Optional[str] = None
    auto_disable_after_fire: Optional[bool] = None
    reminder_text: Optional[str] = Field(default=None, max_length=4000)
    enabled: Optional[bool] = None
    config: Optional[dict] = None
    name: Optional[str] = Field(default=None, max_length=100)
    prompt_text: Optional[str] = None
    delivery_channels: Optional[list[str]] = None

    @field_validator("schedule_kind")
    @classmethod
    def _validate_schedule_kind(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in _VALID_SCHEDULE_KINDS:
            raise ValueError(
                f"Unknown schedule_kind {v!r}. Allowed: {sorted(_VALID_SCHEDULE_KINDS)}"
            )
        return v

    @field_validator("schedule_at")
    @classmethod
    def _strip_tz_from_schedule_at(cls, v: Optional[datetime]) -> Optional[datetime]:
        # Match the naive-UTC convention enforced in RoutineCreate so
        # PATCH /routines/{id} doesn't fall over on tz-aware datetimes.
        if v is None or v.tzinfo is None:
            return v
        from datetime import timezone as _tz
        return v.astimezone(_tz.utc).replace(tzinfo=None)

    @field_validator("schedule_window_start_local")
    @classmethod
    def _validate_start(cls, v: Optional[str]) -> Optional[str]:
        return _validate_window_str(v, "schedule_window_start_local")

    @field_validator("schedule_window_end_local")
    @classmethod
    def _validate_end(cls, v: Optional[str]) -> Optional[str]:
        return _validate_window_str(v, "schedule_window_end_local")

    @field_validator("delivery_channels")
    @classmethod
    def _validate_delivery_channels(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        bad = [c for c in v if c not in _VALID_DELIVERY_CHANNELS]
        if bad:
            raise ValueError(
                f"Unknown delivery channels: {bad}. "
                f"Allowed: {sorted(_VALID_DELIVERY_CHANNELS)}"
            )
        return v


class RoutineRunResponse(BaseModel):
    id: str
    scheduled_for_local_date: date
    started_at: datetime
    finished_at: Optional[datetime]
    # Ticket 2.3: APScheduler's scheduled fire time (UTC). Differs from
    # `started_at` for slow handlers; the dashboard renders this as
    # "fired at X" and `finished_at - fire_instant` as handler latency.
    fire_instant: Optional[datetime] = None
    # Ticket 2.1 — finished_at rendered in the user's tz (ISO8601 string).
    finished_local_at: Optional[str] = None
    status: str  # legacy compat
    # Ticket 2.1 — richer outcome state machine:
    # "success" | "success_empty" | "partial" | "tool_error" | "failure".
    outcome: Optional[str] = None
    error_class: Optional[str] = None
    error_detail: Optional[str] = None
    # Structured error blob (superset of error_class + error_detail).
    error_json: Optional[dict] = None
    # Ticket 2.5 — per-channel delivery confirmations:
    # {channel: {"status": "delivered"|"skipped"|"failed", "message_id": ..., ...}}
    channel_results_json: Optional[dict] = None
    # MCP tool names the handler invoked during this run.
    tools_invoked_json: Optional[list] = None
    emails_fetched: int
    attempt: int
    summary_message_id: Optional[str] = None

    @field_serializer("started_at", "finished_at", "fire_instant", when_used="json")
    def _ser_dt(self, v: Optional[datetime]) -> Optional[str]:
        return _utc_iso(v)


class RoutineResponse(BaseModel):
    id: str
    kind: str
    name: Optional[str] = None
    prompt_text: Optional[str] = None
    enabled: bool
    # Mig 042 — schedule shape (NULL → 'cron' for legacy rows).
    schedule_kind: str = "cron"
    schedule_cron_local: str
    schedule_at: Optional[datetime] = None
    schedule_interval_seconds: Optional[int] = None
    schedule_window_start_local: Optional[str] = None
    schedule_window_end_local: Optional[str] = None
    auto_disable_after_fire: bool = False
    reminder_text: Optional[str] = None
    config: Optional[dict]
    last_state: Optional[dict]
    last_run_at: Optional[datetime]
    next_run_at: Optional[datetime]
    last_status: str
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    recent_runs: list[RoutineRunResponse] = []

    # PR 7 of the unified-jobs arc: last 5 mirrored BuildJob rows
    # for this routine (one per fire — the dual-write from PR 4b).
    # The Mission Control card renders these as a "Last fires"
    # list linking through to the Job detail drawer. Sibling of
    # ``TriggerResponse.recent_jobs``.
    recent_jobs: list["RoutineRecentJob"] = []

    @field_serializer(
        "schedule_at", "last_run_at", "next_run_at", "created_at", "updated_at",
        when_used="json",
    )
    def _ser_dt(self, v: Optional[datetime]) -> Optional[str]:
        return _utc_iso(v)


class RoutineRecentJob(BaseModel):
    """Compact projection of a recent ``build_jobs`` row mirrored
    from a RoutineRun (PR 4b). Sibling of ``TriggerRecentJob`` —
    same shape so frontend cards can share the rendering
    component."""

    id: str
    title: Optional[str] = None
    status: Optional[str] = None
    outcome: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


def _run_to_response(j) -> RoutineRunResponse:
    """Project a ``build_jobs`` row into the legacy ``RoutineRunResponse``
    shape so the runs-history endpoints can serve from BuildJob without
    a frontend contract change.

    PR #51 of the unified-jobs cutover arc: the legacy ``routine_runs``
    table is going away in PR #52 (mig 050). Every API consumer that
    used to SELECT RoutineRun now reads BuildJob; this helper keeps the
    wire shape stable so the dashboard / extension don't notice.

    Field mapping (mirrors the runner's terminal-write mapping in
    ``app/agent/routines/runner.py:_finalize_run``):

      BuildJob.id                       → id
      BuildJob.idempotency_key (ISO date string) → scheduled_for_local_date
      BuildJob.created_at               → started_at  (the routine's
                                           runner writes status='running'
                                           in the same transaction that
                                           creates the row, so created_at
                                           ≈ legacy started_at)
      BuildJob.completed_at             → finished_at
      BuildJob.fire_instant             → fire_instant     (mig 051 col)
      BuildJob.finished_local_at        → finished_local_at (mig 052 col)
      BuildJob.status + outcome         → status      (legacy enum)
      BuildJob.outcome                  → outcome
      BuildJob.error_json               → error_json (mig 052 col)
      BuildJob.channel_results_json     → channel_results_json (mig 052)
      BuildJob.tools_invoked_json       → tools_invoked_json (mig 052)
      BuildJob.emails_fetched           → emails_fetched (mig 052 col,
                                           defaults to 0 if NULL)
      BuildJob.attempt                  → attempt (mig 051 col, defaults
                                           to 1 if NULL — first fire was
                                           the only fire)
      BuildJob.summary_message_id       → summary_message_id

    ``error_class`` and ``error_detail`` are derived best-effort from
    ``error_json`` (richer blob) and ``error_message`` (single-column
    text). The legacy two-column split is gone on BuildJob; the
    response keeps the field names so the dashboard's "error class
    chip" still renders something when present."""
    err_json = getattr(j, "error_json", None) or {}
    err_class: Optional[str] = err_json.get("error_class") if isinstance(err_json, dict) else None
    err_detail: Optional[str] = (
        err_json.get("error_detail") if isinstance(err_json, dict) else None
    ) or j.error_message

    return RoutineRunResponse(
        id=j.id,
        scheduled_for_local_date=_parse_local_date_from_idempotency(j.idempotency_key),
        started_at=j.created_at,
        finished_at=j.completed_at,
        fire_instant=getattr(j, "fire_instant", None),
        finished_local_at=getattr(j, "finished_local_at", None),
        status=_job_status_to_legacy_run(j.status, getattr(j, "outcome", None)),
        outcome=getattr(j, "outcome", None),
        error_class=err_class,
        error_detail=err_detail,
        error_json=err_json or None,
        channel_results_json=getattr(j, "channel_results_json", None),
        tools_invoked_json=getattr(j, "tools_invoked_json", None),
        emails_fetched=int(getattr(j, "emails_fetched", 0) or 0),
        attempt=int(getattr(j, "attempt", 1) or 1),
        summary_message_id=j.summary_message_id,
    )


def _parse_local_date_from_idempotency(key: Optional[str]) -> date:
    """``BuildJob.idempotency_key`` for routine fires is the local date
    as ISO format (set by ``RoutineRunner._fire``). Parse it back to a
    ``date`` for the response. Falls back to today (UTC) if the key
    is missing or malformed — defensive against the rare row written
    by a pre-cutover code path with a non-date key."""
    if key:
        try:
            return date.fromisoformat(key)
        except ValueError:
            pass
    return datetime.utcnow().date()


def _job_status_to_legacy_run(status: Optional[str], outcome: Optional[str]) -> str:
    """Collapse BuildJob's (status, outcome) two-column state back to
    the single legacy ``RoutineRun.status`` enum the frontend reads.

    Inverse of the routine runner's write mapping (see
    ``app/agent/routines/runner.py:_finalize_run`` docstring):

      completed + 'success'        → 'success'
      completed + 'success_empty'  → 'success'   (legacy lumped both)
      completed + 'partial'        → 'partial'
      completed + 'skipped_reauth' → 'skipped_reauth'
      failed    + NULL             → 'failed'
      running   + *                → 'running'
      queued    + *                → 'queued'

    Unknown combinations fall back to the raw outcome (if present)
    then the raw status — better than returning an empty string and
    breaking the dashboard's status pill switch."""
    if status == "completed":
        if outcome in ("success_empty",):
            return "success"
        return outcome or "success"
    if status == "failed":
        return "failed"
    return status or "queued"


def _row_to_response(routine, recent_runs=(), recent_jobs=()) -> RoutineResponse:
    # Mig 042 — `getattr(..., default)` for every new column so rows
    # that pre-date the migration (NULL columns) render with sensible
    # defaults. The Optional bits stay None; schedule_kind falls
    # through to "cron".
    return RoutineResponse(
        id=routine.id,
        kind=routine.kind,
        name=getattr(routine, "name", None),
        prompt_text=getattr(routine, "prompt_text", None),
        enabled=bool(routine.enabled),
        schedule_kind=(getattr(routine, "schedule_kind", None) or "cron"),
        schedule_cron_local=routine.schedule_cron_local,
        schedule_at=getattr(routine, "schedule_at", None),
        schedule_interval_seconds=getattr(routine, "schedule_interval_seconds", None),
        schedule_window_start_local=getattr(routine, "schedule_window_start_local", None),
        schedule_window_end_local=getattr(routine, "schedule_window_end_local", None),
        auto_disable_after_fire=bool(getattr(routine, "auto_disable_after_fire", False)),
        reminder_text=getattr(routine, "reminder_text", None),
        config=routine.config_json or None,
        last_state=routine.last_state_json or None,
        last_run_at=routine.last_run_at,
        next_run_at=routine.next_run_at,
        last_status=routine.last_status,
        last_error=routine.last_error,
        created_at=routine.created_at,
        updated_at=routine.updated_at,
        recent_runs=[_run_to_response(r) for r in recent_runs],
        recent_jobs=[
            RoutineRecentJob(
                id=j.id,
                title=j.title,
                status=j.status,
                outcome=getattr(j, "outcome", None),
                created_at=j.created_at,
                completed_at=j.completed_at,
            )
            for j in recent_jobs
        ],
    )


# ── Endpoints ──────────────────────────────────────────────────────────


@router.get("/_runner_status")
async def runner_status():
    """Cheap probe for ops/debug: is the routine scheduler up, how many
    routines are registered, when does the next one fire."""
    if _runner is None:
        return {
            "running": False,
            "routines_registered": 0,
            "next_fire_at": None,
            "reason": "runner_not_started",
        }
    return _runner.status_snapshot()


@router.post("/_reload_all")
async def runner_reload_all():
    """Ticket 2.3(d) — post-deploy reload backfill.

    Drops every registered trigger and rebuilds from DB. Operator-run
    after a deploy that changes scheduler semantics so any
    pre-deploy-stamped `next_run_at` columns get refreshed to the
    post-deploy code's calculation. Logs before/after delta via the
    runner.
    """
    if _runner is None:
        raise HTTPException(status_code=503, detail="Routine runner not started")
    await _runner.reload_all()
    return _runner.status_snapshot()


@router.get("", response_model=list[RoutineResponse])
async def list_routines():
    """All routines belonging to this container's owner, each with the
    most recent 7 runs.

    Returns 200 with `[]` (not 404) if the user has zero routines —
    consumers want an empty list, not an error.

    Per-row failures (Pydantic validation on a row whose mig 042 columns
    haven't been backfilled to a known-good value, etc.) are caught and
    logged so the rest of the list still renders. This avoids the
    dashboard's "Internal Server Error" banner on the whole Routines
    section because one routine row in agent DB has stale state.
    Diagnosed 2026-05-18: a single bad row in one tenant returned 500
    on GET / instead of degrading to "empty plus a server-side log."
    """
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Routine

    user_id = _user_id()
    out: list[RoutineResponse] = []
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(Routine)
                .where(
                    Routine.user_id == user_id,
                    # Autopilot missions are IMPLEMENTED as
                    # self-rescheduling routines, but they are not
                    # user-managed schedules — listing them here leaked
                    # the "every 5 min" heartbeat internal into the
                    # Routines UI (founder bug 2026-07-16). Missions
                    # have their own surface: /api/autopilot/missions.
                    Routine.kind != "autopilot",
                    # Round 26: poll bindings are engine internals for
                    # the same reason (a hidden system routine); the
                    # automation itself is the user-visible unit at
                    # /api/automations. `automation_schedule` stays
                    # listed deliberately — it IS a schedule the user
                    # asked for in words.
                    Routine.kind != "automation_poll",
                )
                .order_by(Routine.created_at)
            )
            routines = list(result.scalars().all())

            for r in routines:
                try:
                    # PR #51 cutover: source the recent-runs list from
                    # ``build_jobs`` instead of the legacy ``routine_runs``
                    # table (which mig 050 drops in PR #52). Each fire
                    # mints one BuildJob row with ``source_kind='routine'``
                    # + ``source_id=routine.id`` — same shape as
                    # ``recent_jobs`` below, just with a higher limit and
                    # the richer terminal column set (mig 051 / 052).
                    runs_result = await db.execute(
                        select(BuildJob).where(
                            BuildJob.source_kind == "routine",
                            BuildJob.source_id == r.id,
                        ).order_by(desc(BuildJob.created_at)).limit(7)
                    )
                    recent = list(runs_result.scalars().all())
                    # PR 7 of the unified-jobs arc: surface the
                    # last 5 mirrored BuildJob rows for this routine
                    # (one per fire — PR 4b dual-write). The
                    # ``recent_runs`` and ``recent_jobs`` arrays are
                    # complementary: recent_runs carries the
                    # routine-specific fields (outcome,
                    # channel_results, tools_invoked), recent_jobs
                    # carries the Job-row projection the dashboard
                    # links to from the unified Jobs surface.
                    jobs_result = await db.execute(
                        select(BuildJob).where(
                            BuildJob.source_kind == "routine",
                            BuildJob.source_id == r.id,
                        ).order_by(desc(BuildJob.created_at)).limit(5)
                    )
                    recent_jobs = list(jobs_result.scalars().all())
                    out.append(_row_to_response(r, recent, recent_jobs))
                except Exception as e:
                    logger.exception(
                        "list_routines: skipping routine_id=%s due to render "
                        "error: %s: %s",
                        getattr(r, "id", "?"), type(e).__name__, str(e)[:200],
                    )
    except Exception as e:
        # Catastrophic: model column drift, DB unreachable, etc. Return
        # an empty list with a server-side log instead of bricking the
        # dashboard Routines panel for everyone. /_runner_status is the
        # operational source of truth for "is the runner alive" anyway.
        logger.exception(
            "list_routines: catastrophic failure (returning []): %s: %s",
            type(e).__name__, str(e)[:200],
        )
    return out


@router.post("", response_model=RoutineResponse, status_code=201)
async def create_routine(req: RoutineCreate):
    """Create one routine. 409 if an enabled routine of the same kind
    already exists for this user — v1 UX limits each kind to one active
    routine. The schema permits more so the constraint can be relaxed
    later without a migration."""
    _validate_kind(req.kind)
    _kind_enabled_or_404(req.kind)
    # Mig 042 — cron is validated only when the routine uses it.
    # 'at' and 'every' kinds carry their own schedule fields (already
    # validated by Pydantic's model_validator).
    if req.schedule_kind == "cron":
        _validate_cron(req.schedule_cron_local)

    # Generic agent_task routines NEED a prompt — that's the whole point.
    # Preset kinds (email_briefing) don't use prompt_text; the handler
    # has its own hard-coded prompt template. Reminders carry their
    # delivery text in `reminder_text` (Pydantic enforces).
    if req.kind == "agent_task" and not (req.prompt_text or "").strip():
        raise HTTPException(
            status_code=400,
            detail="agent_task routines require `prompt_text` (what should "
                   "the agent do?).",
        )

    from app.db.database import async_session_maker
    from app.db.models import Routine, User

    user_id = _user_id()

    # Fail loudly when User.timezone is missing — routines without a tz
    # silently fall back to UTC inside `_resolve_tz`, so a user who said
    # "fire at 1:21 PM" would get a fire at 1:21 PM UTC (= 9:21 AM
    # Toronto during EDT). Recent auth commits (`f62ae3a` silent
    # browser-Intl capture, `2d1e14c` location-share refinement) cover
    # new sign-ins, but legacy users without a captured tz must set
    # one before scheduling work — a UTC misfire is a worse UX than a
    # clear "set your timezone" prompt.
    async with async_session_maker() as db:
        user = await db.get(User, user_id)
    if user is None or not getattr(user, "timezone", None):
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "missing_timezone",
                "message": (
                    "Your timezone isn't set, so we can't schedule "
                    "this routine to fire at the right local time. "
                    "Open the chat once more (we'll capture it from "
                    "your browser) or set it manually in account "
                    "settings, then try again."
                ),
            },
        )
    async with async_session_maker() as db:
        # One-active-per-kind enforcement is preserved for preset kinds
        # (email_briefing → one Gmail briefing per user). For the generic
        # `agent_task` kind we allow MANY active routines — each one is a
        # distinct task ("morning briefing", "noon GitHub check", etc.).
        #
        # Pre-2026-05-14 this used `scalar_one_or_none()` which raises
        # `MultipleResultsFound` when ≥2 enabled routines of the same
        # kind already exist — i.e., the guard CRASHED in exactly the
        # state it was meant to prevent. After the cleanup migration
        # (041) and the partial UNIQUE index that pairs with it, two
        # enabled rows of a preset kind cannot exist concurrently; the
        # Python check below is the friendly 409 path. The DB index is
        # the load-bearing guarantee, the Python check is for the nice
        # error message.
        from app.services.routine_uniqueness import (
            find_conflicting_enabled_routine_ids,
        )
        existing_ids = await find_conflicting_enabled_routine_ids(
            db, user_id=user_id, kind=req.kind,
        )
        if existing_ids:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": (
                        f"An enabled {req.kind!r} routine already exists "
                        f"for this user. Disable or delete it first, "
                        f"or update the existing one instead of "
                        f"creating a new one."
                    ),
                    "existing_routine_ids": existing_ids,
                    "kind": req.kind,
                },
            )

        # Merge delivery_channels into config_json. We keep them on the
        # SAME blob so the handler can read everything at fire time
        # without a separate query — config_json is the per-kind state +
        # routing dict. If the client passed an explicit `config` too,
        # delivery_channels wins on key collision (it's the typed field).
        merged_config: Optional[dict] = dict(req.config) if req.config else None
        if req.delivery_channels is not None:
            merged_config = merged_config or {}
            merged_config["delivery_channels"] = list(req.delivery_channels)

        # Mig 042 — auto-disable defaults to True for one-shot 'at'
        # reminders (user said "remind me at 5pm" → fire ONCE).
        # Explicit True/False from the caller always wins.
        auto_disable = req.auto_disable_after_fire
        if auto_disable is None:
            auto_disable = (req.schedule_kind == "at")

        # `schedule_cron_local` is non-nullable on the schema (legacy
        # column). For non-cron shapes we store a placeholder; the
        # runner's _build_trigger_for_routine ignores it whenever
        # schedule_kind != 'cron'. Avoids a destructive schema change.
        cron_str = req.schedule_cron_local
        if not cron_str:
            cron_str = "* * * * *" if req.schedule_kind != "cron" else ""

        routine = Routine(
            user_id=user_id,
            kind=req.kind,
            name=(req.name or "").strip() or None,
            prompt_text=(req.prompt_text or "").strip() or None,
            schedule_cron_local=cron_str,
            schedule_kind=req.schedule_kind,
            schedule_at=req.schedule_at,
            schedule_interval_seconds=req.schedule_interval_seconds,
            schedule_window_start_local=req.schedule_window_start_local,
            schedule_window_end_local=req.schedule_window_end_local,
            auto_disable_after_fire=bool(auto_disable),
            reminder_text=(req.reminder_text or "").strip() or None,
            enabled=bool(req.enabled),
            config_json=merged_config,
            last_status="never_run",
        )
        db.add(routine)
        try:
            await db.commit()
        except IntegrityError as e:
            # The partial UNIQUE index (migration 041) caught a race
            # the Python guard missed — two concurrent `POST /routines`
            # both passed the in-app check, only one row can win. The
            # loser comes back as 409 (not 500) so the agent / UI can
            # retry-via-update cleanly.
            await db.rollback()
            if "uq_routines_one_per_kind" in str(e).lower() \
                    or "unique" in str(e).lower():
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            f"Another concurrent request created an "
                            f"enabled {req.kind!r} routine first. "
                            f"Refresh and update the existing one."
                        ),
                        "kind": req.kind,
                    },
                )
            raise
        await db.refresh(routine)

    # Register the trigger with the runner so the routine fires at its
    # next scheduled time. Safe to call before the runner has an MCP
    # client — registration just adds an APScheduler job.
    if _runner is not None and routine.enabled:
        try:
            await _runner.reload_routine(routine.id)
        except Exception:
            # Registration failure is non-fatal — operator can call
            # reload_all later. The row is already persisted.
            pass

    # Arm the on-device countdown card for near-term one-shot reminders.
    # Covers both the routines skill (delegates here in-process) and
    # Mission Control HTTP.
    if _reminder_countdown_eligible(routine):
        await _reminder_countdown_notify(routine)

    return _row_to_response(routine)


@router.patch("/{routine_id}", response_model=RoutineResponse)
async def update_routine(routine_id: str, req: RoutineUpdate):
    """Update schedule / enabled / config. Triggers `reload_routine` so
    the next fire honours the new schedule immediately — no restart
    needed."""
    from app.db.database import async_session_maker
    from app.db.models import Routine

    if req.schedule_cron_local is not None:
        _validate_cron(req.schedule_cron_local)

    user_id = _user_id()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
        if routine is None or routine.user_id != user_id:
            raise HTTPException(status_code=404, detail="Routine not found")

        _kind_enabled_or_404(routine.kind)
        if routine.kind == "autopilot":
            # Missions ride the Routine table but are managed through
            # their own lifecycle API — a generic PATCH here can clobber
            # config_json (goal/budget) wholesale or zombie-toggle the
            # engine's enabled flag.
            raise HTTPException(
                status_code=409,
                detail="This is an Autopilot mission — manage it via /api/autopilot/missions",
            )

        # Mig 042 — `schedule_changed` covers ALL shape edits (cron
        # string, schedule_kind, schedule_at, interval). Each invalidates
        # today's failed runs so a new fire can claim a fresh slot.
        schedule_changed = (
            (req.schedule_cron_local is not None
             and req.schedule_cron_local != routine.schedule_cron_local)
            or (req.schedule_kind is not None
                and req.schedule_kind != (getattr(routine, "schedule_kind", None) or "cron"))
            or (req.schedule_at is not None
                and req.schedule_at != getattr(routine, "schedule_at", None))
            or (req.schedule_interval_seconds is not None
                and req.schedule_interval_seconds
                    != getattr(routine, "schedule_interval_seconds", None))
        )

        if req.schedule_cron_local is not None:
            routine.schedule_cron_local = req.schedule_cron_local
        if req.schedule_kind is not None:
            routine.schedule_kind = req.schedule_kind
        if req.schedule_at is not None:
            routine.schedule_at = req.schedule_at
        if req.schedule_interval_seconds is not None:
            routine.schedule_interval_seconds = req.schedule_interval_seconds
        if req.schedule_window_start_local is not None:
            routine.schedule_window_start_local = req.schedule_window_start_local
        if req.schedule_window_end_local is not None:
            routine.schedule_window_end_local = req.schedule_window_end_local
        if req.auto_disable_after_fire is not None:
            routine.auto_disable_after_fire = bool(req.auto_disable_after_fire)
        if req.reminder_text is not None:
            routine.reminder_text = (req.reminder_text or "").strip() or None
        if req.enabled is not None:
            routine.enabled = bool(req.enabled)
        if req.config is not None:
            routine.config_json = req.config
        if req.delivery_channels is not None:
            # Merge into the live config blob — don't clobber other keys
            # like `connector_identity_id` that the email_briefing handler
            # uses.
            cfg = dict(routine.config_json or {})
            cfg["delivery_channels"] = list(req.delivery_channels)
            routine.config_json = cfg
        if req.name is not None:
            routine.name = (req.name or "").strip() or None
        if req.prompt_text is not None:
            routine.prompt_text = (req.prompt_text or "").strip() or None
        routine.updated_at = datetime.utcnow()

        # When the user moves the schedule mid-day (the 2026-05-12 case:
        # 8 AM briefing failed → user moved it to 1:21 PM, then the new
        # fire would have been blocked by the idempotency UNIQUE on
        # (source_id, idempotency_key)), clear any FAILED run rows for
        # today's local date so the new cron tick can claim a fresh row.
        # We do NOT touch successful runs — those represent real work the
        # user already received. Running rows are also left alone
        # (race-safe).
        #
        # PR #51 cutover: targets ``build_jobs`` instead of the legacy
        # ``routine_runs`` table. The idempotency_key for a routine fire
        # is the local-date ISO string (set by RoutineRunner._fire), so
        # the WHERE clause filters by that string equality on the same
        # date the next fire would try to claim. We only delete jobs in
        # terminal ``failed`` status — ``running`` jobs in the queue are
        # left alone (PR #50 retired skipped_reauth-as-a-status; that
        # state now lives in BuildJob.outcome alongside completed).
        if schedule_changed:
            from app.agent.routines.runner import _resolve_tz
            from app.db.models import BuildJob, User as _User
            user = await db.get(_User, user_id)
            tz, _ = _resolve_tz(getattr(user, "timezone", None), user_id)
            local_today = datetime.now(tz).date()
            from sqlalchemy import and_, or_
            await db.execute(
                delete(BuildJob).where(
                    BuildJob.source_kind == "routine",
                    BuildJob.source_id == routine.id,
                    BuildJob.idempotency_key == local_today.isoformat(),
                    # Two terminal states block a re-fire today:
                    #   1. status='failed' (raw runner failure).
                    #   2. status='completed' + outcome='skipped_reauth'
                    #      (OAuth was missing — preserves the legacy
                    #      ``RoutineRun.status='skipped_reauth'`` semantics
                    #      after PR #50's status→outcome split). Successful
                    #      ``completed/success`` and ``completed/partial``
                    #      runs are preserved — the user already received
                    #      that fire's work.
                    or_(
                        BuildJob.status == "failed",
                        and_(
                            BuildJob.status == "completed",
                            BuildJob.outcome == "skipped_reauth",
                        ),
                    ),
                )
            )

        try:
            await db.commit()
        except IntegrityError as e:
            # Enabling a disabled preset-kind routine when another
            # enabled one already exists for this user trips the partial
            # UNIQUE (migration 041). 409 with the existing routine id
            # so the agent can offer to swap them.
            await db.rollback()
            from app.services.routine_uniqueness import (
                find_conflicting_enabled_routine_ids, KINDS_EXEMPT_FROM_ONE_PER_KIND,
            )
            if routine.kind not in KINDS_EXEMPT_FROM_ONE_PER_KIND:
                other_ids = await find_conflicting_enabled_routine_ids(
                    db, user_id=user_id, kind=routine.kind,
                    exclude_routine_id=routine_id,
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            f"Cannot enable: another enabled "
                            f"{routine.kind!r} routine already exists. "
                            f"Disable it first or update this one's "
                            f"settings into that routine."
                        ),
                        "existing_routine_ids": other_ids,
                        "kind": routine.kind,
                    },
                )
            raise
        await db.refresh(routine)

    # Countdown lifecycle on edits: disabling or moving the schedule
    # ends the armed card (silent, immediate dismissal); a still-enabled
    # in-window 'at' reminder re-arms. Cancel-end lands before the fresh
    # start via the dispatcher's created_at ordering.
    if routine.kind == "reminder" and (req.enabled is False or schedule_changed):
        from app.config import settings as _settings
        if getattr(_settings, "reminder_countdown_live_activity_enabled", True):
            await _reminder_cancel_notify(routine.id)
            if _reminder_countdown_eligible(routine):
                await _reminder_countdown_notify(routine)

    if _runner is not None:
        try:
            await _runner.reload_routine(routine.id)
        except Exception as e:
            # Ticket 3 / Bug B: do NOT swallow silently. A failed reload
            # leaves the OLD job registered with the OLD cron — the API
            # response would lie about the update having taken effect.
            # Count it (status_snapshot surfaces this) and re-raise so
            # the client sees a 500 and the operator sees a stack trace.
            try:
                _runner._reload_failures_total += 1  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover — counter is best-effort
                pass
            logger.exception(
                "[routines.update] reload_routine failed routine_id=%s err=%s",
                routine.id, e,
            )
            raise HTTPException(
                status_code=503,
                detail="Schedule saved to DB but scheduler reload failed. "
                       "Retry the update or contact support.",
            )

    # Re-read the routine after reload so the response carries the
    # freshly-synced `next_run_at` (runner._sync_next_run writes it
    # from APScheduler's in-memory next_run_time during reload). Without
    # this re-read, the response shows the pre-update next_run_at and
    # users / the agent's routines__list see "stale scheduler" — the
    # original Ticket 3 / Bug A symptom.
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
        if routine is None:
            raise HTTPException(status_code=404, detail="Routine not found")

    return _row_to_response(routine)


@router.delete("/{routine_id}", status_code=204)
async def delete_routine(routine_id: str):
    """Drop the routine and (via CASCADE) every routine_runs row that
    references it. The trigger is removed from the scheduler too."""
    from app.db.database import async_session_maker
    from app.db.models import Routine

    user_id = _user_id()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
        if routine is None or routine.user_id != user_id:
            raise HTTPException(status_code=404, detail="Routine not found")
        if routine.kind == "autopilot":
            # A generic routines delete must not silently destroy a
            # running mission (its working state lives on this row).
            raise HTTPException(
                status_code=409,
                detail="This is an Autopilot mission — cancel it via /api/autopilot/missions",
            )
        # Captured before delete — the ORM instance is unusable after
        # the commit expunges it.
        was_reminder = routine.kind == "reminder"
        await db.delete(routine)
        await db.commit()

    if was_reminder:
        from app.config import settings as _settings
        if getattr(_settings, "reminder_countdown_live_activity_enabled", True):
            # End any armed countdown card silently.
            await _reminder_cancel_notify(routine_id)

    if _runner is not None:
        try:
            await _runner.reload_routine(routine_id)  # idempotent — removes
        except Exception:
            pass
    return None


@router.post("/{routine_id}/run", response_model=RoutineRunResponse)
async def force_run(routine_id: str):
    """Trigger a fire NOW for today's local-date slot. Idempotent — if
    today's run already exists (scheduled or force-run), returns 409 with
    the existing run's row so the UI can render it inline.

    Behaves identically to a scheduled fire: same handler dispatch,
    same retry loop, same Message-write path. The only difference is
    timing — we run immediately instead of waiting for the cron tick.

    PR #51 cutover: backed by ``build_jobs`` instead of the legacy
    ``routine_runs`` table (which mig 050 drops in PR #52). The
    pre-fire idempotency check + the post-fire re-read both query
    BuildJob with ``source_kind='routine'`` + ``idempotency_key=
    <local_today>``."""
    from app.agent.routines.runner import _resolve_tz
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Routine, User

    user_id = _user_id()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
        if routine is None or routine.user_id != user_id:
            raise HTTPException(status_code=404, detail="Routine not found")
        _kind_enabled_or_404(routine.kind)

        # Free-tier 60s run-now cooldown — prevents a credit-poor user
        # from button-mashing a routine to drain their free 100 message
        # credits in a few seconds. Paid tiers have no cooldown (they
        # paid for the work). last_run_at is updated by the runner on
        # every successful or attempted fire; comparing now()-60s
        # against it gives the right window for an HTTP-paced retry.
        try:
            from app.db.models import CreditBalance
            balance = await db.get(CreditBalance, user_id)
            plan_id = (balance.plan_id if balance else "free") or "free"
            if plan_id == "free" and routine.last_run_at is not None:
                from datetime import datetime as _dt, timedelta as _td
                if (_dt.utcnow() - routine.last_run_at) < _td(seconds=60):
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "message": "Free tier limits run-now to once per 60s per routine.",
                            "cooldown_seconds": 60,
                            "last_run_at": routine.last_run_at.isoformat(),
                        },
                    )
        except HTTPException:
            raise
        except Exception:
            # Gate is best-effort — credit-table outage shouldn't break
            # the force-run path. Fall through to the existing
            # idempotency logic.
            pass

        # Compute today in the user's tz — same path the runner uses,
        # so the idempotency key matches a scheduled fire on this day.
        user = await db.get(User, user_id)
        tz, _ = _resolve_tz(getattr(user, "timezone", None), user_id)
        local_today = datetime.now(tz).date()
        local_today_key = local_today.isoformat()

        existing = await db.execute(
            select(BuildJob).where(
                BuildJob.source_kind == "routine",
                BuildJob.source_id == routine_id,
                BuildJob.idempotency_key == local_today_key,
            )
        )
        prior = existing.scalar_one_or_none()
        if prior is not None:
            # Re-run policy: if today's prior run ended in a recoverable
            # terminal state (failed, or completed+skipped_reauth), the
            # user explicitly asking to re-run is a retry — drop the
            # prior row so the idempotency claim in ``_fire`` succeeds
            # with a fresh row. We KEEP success rows (don't double-
            # deliver to Telegram / WhatsApp) and KEEP running rows
            # (race-safe — another request is already mid-fire).
            prior_outcome = getattr(prior, "outcome", None)
            is_recoverable = (
                prior.status == "failed"
                or (prior.status == "completed"
                    and prior_outcome == "skipped_reauth")
            )
            if is_recoverable:
                await db.execute(
                    delete(BuildJob).where(BuildJob.id == prior.id)
                )
                await db.commit()
            else:
                # Surface the legacy status name (success/partial/
                # running/queued) for the 409 detail so the UI's
                # existing copy still makes sense to the user.
                legacy_status = _job_status_to_legacy_run(
                    prior.status, prior_outcome,
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Today's run already exists for this routine.",
                        "run_id": prior.id,
                        "status": legacy_status,
                    },
                )

    if _runner is None:
        raise HTTPException(status_code=503, detail="Routine runner not started")

    # Dispatch synchronously — _fire handles its own idempotency claim
    # (via JobRunner.create_job + the (source_id, idempotency_key) UNIQUE),
    # which will succeed since we just verified no prior row exists.
    await _runner._fire(routine_id)

    async with async_session_maker() as db:
        # Multi-fire kinds (autopilot) key fires per-instant, not per
        # local day (RoutineRunner._fire_idempotency_key) — the daily
        # key would miss their row. Re-read the newest fire instead;
        # daily kinds still resolve to today's single row.
        result = await db.execute(
            select(BuildJob)
            .where(
                BuildJob.source_kind == "routine",
                BuildJob.source_id == routine_id,
            )
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )
        run = result.scalar_one_or_none()
    if run is None:
        # Should be unreachable — _fire either claims a row or hits the
        # idempotency gate (already covered above). Defensive 500 so
        # the operator notices.
        raise HTTPException(status_code=500, detail="run row missing after _fire")
    return _run_to_response(run)


@router.get("/{routine_id}/runs", response_model=list[RoutineRunResponse])
async def list_runs(
    routine_id: str,
    limit: int = Query(default=30, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Paginated run history. Most-recent first.

    PR #51 cutover: backed by ``build_jobs`` (mig 050 drops the legacy
    ``routine_runs`` table in PR #52). Same composite index on
    ``(source_kind, source_id, created_at desc)`` from migration 046
    keeps this cheap up to N=200 / offset=10k."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Routine

    user_id = _user_id()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
        if routine is None or routine.user_id != user_id:
            raise HTTPException(status_code=404, detail="Routine not found")
        result = await db.execute(
            select(BuildJob)
            .where(
                BuildJob.source_kind == "routine",
                BuildJob.source_id == routine_id,
            )
            .order_by(desc(BuildJob.created_at))
            .limit(limit)
            .offset(offset)
        )
        runs = list(result.scalars().all())
    return [_run_to_response(r) for r in runs]
