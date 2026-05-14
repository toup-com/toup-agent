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
from datetime import date, datetime
from typing import Any, Optional

from apscheduler.triggers.cron import CronTrigger
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
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


# ── Pydantic models ────────────────────────────────────────────────────


_VALID_DELIVERY_CHANNELS = {"website", "telegram", "whatsapp"}
_VALID_SCHEDULE_KINDS = {"cron", "at", "every"}
_VALID_ROUTINE_KINDS = {"email_briefing", "agent_task", "reminder"}
_MIN_INTERVAL_SECONDS = 60


def _validate_window_str(v: Optional[str], field: str) -> Optional[str]:
    """HH:MM or HH:MM:SS, both inclusive. Returns canonical HH:MM:SS."""
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
            "Routine handler kind. One of: 'email_briefing' (Gmail "
            "summary preset), 'agent_task' (generic LLM-driven), "
            "'reminder' (text-only delivery, no LLM)."
        ),
    )
    # Phase A — schedule shape selector. Existing API callers default
    # to 'cron' so legacy `POST /routines` requests carrying only
    # `schedule_cron_local` keep working unchanged.
    schedule_kind: str = Field(
        default="cron",
        description="'cron' (default) | 'at' (one-shot) | 'every' (interval).",
    )
    schedule_cron_local: Optional[str] = Field(
        default=None,
        description=(
            "5-part cron expression evaluated in the user's tz. "
            "Required when schedule_kind='cron'."
        ),
        examples=["30 6 * * *"],
    )
    schedule_at: Optional[datetime] = Field(
        default=None,
        description=(
            "One-shot fire datetime (UTC). Required when "
            "schedule_kind='at'. Must be in the future."
        ),
    )
    schedule_interval_seconds: Optional[int] = Field(
        default=None,
        description=(
            "Interval between fires in seconds. Required when "
            "schedule_kind='every'. Minimum 60."
        ),
    )
    schedule_window_start_local: Optional[str] = Field(
        default=None,
        description=(
            "Optional active-window start, HH:MM or HH:MM:SS in the "
            "user's tz. Used with schedule_kind='every' to gate fires."
        ),
    )
    schedule_window_end_local: Optional[str] = Field(
        default=None,
        description="Optional active-window end (see start_local).",
    )
    auto_disable_after_fire: Optional[bool] = Field(
        default=None,
        description=(
            "When true, the runner disables the routine after a "
            "successful fire. Server-default true for schedule_kind='at'."
        ),
    )
    reminder_text: Optional[str] = Field(
        default=None, max_length=4000,
        description=(
            "Literal text to deliver (no LLM, no MCP). Required when "
            "kind='reminder'."
        ),
    )
    name: Optional[str] = Field(default=None, max_length=100,
        description="User-visible name. Defaults to a kind-based label.")
    prompt_text: Optional[str] = Field(default=None,
        description="Required when kind='agent_task'. The natural-language "
                    "task the agent runs at fire time.")
    enabled: bool = True
    # Where to send the routine's output. Website (Day-as-Chat) is always
    # written; this list controls EXTRA outbound channels — Telegram and
    # WhatsApp. Default ["website"] preserves legacy behaviour. Stored
    # into `config_json.delivery_channels` server-side so the handler can
    # read it at fire time without a join.
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
                f"Unknown schedule_kind {v!r}. "
                f"Allowed: {sorted(_VALID_SCHEDULE_KINDS)}"
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
        """Phase A — cross-field invariants. Mirrors the Postgres CHECK
        constraint from migration 042 so the API surfaces clean 400s
        instead of relying on the DB to 500."""
        sk = self.schedule_kind
        if sk == "cron":
            if not (self.schedule_cron_local or "").strip():
                raise ValueError(
                    "schedule_cron_local is required when "
                    "schedule_kind='cron'."
                )
        elif sk == "at":
            if self.schedule_at is None:
                raise ValueError(
                    "schedule_at is required when schedule_kind='at'."
                )
            # Reject past one-shots at create-time. The 30-second slack
            # lets clock-skew between client and server round-trip
            # without false-positives on "fire 1 minute from now".
            now = datetime.utcnow()
            scheduled = self.schedule_at
            if scheduled.tzinfo is not None:
                # Convert tz-aware to naive UTC for comparison.
                from datetime import timezone as _tz
                scheduled = scheduled.astimezone(_tz.utc).replace(tzinfo=None)
            if scheduled <= now:
                raise ValueError(
                    f"schedule_at must be in the future "
                    f"(got {scheduled.isoformat()} vs now {now.isoformat()})."
                )
        elif sk == "every":
            if self.schedule_interval_seconds is None:
                raise ValueError(
                    "schedule_interval_seconds is required when "
                    "schedule_kind='every'."
                )
            if self.schedule_interval_seconds < _MIN_INTERVAL_SECONDS:
                raise ValueError(
                    f"schedule_interval_seconds must be >= "
                    f"{_MIN_INTERVAL_SECONDS} (got "
                    f"{self.schedule_interval_seconds})."
                )

        # Reminder kind needs literal text. Empty/whitespace-only fails.
        if self.kind == "reminder" and not (self.reminder_text or "").strip():
            raise ValueError(
                "reminder_text is required when kind='reminder'."
            )

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
                f"Unknown schedule_kind {v!r}. "
                f"Allowed: {sorted(_VALID_SCHEDULE_KINDS)}"
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


class RoutineResponse(BaseModel):
    id: str
    kind: str
    name: Optional[str] = None
    prompt_text: Optional[str] = None
    enabled: bool
    # Phase A — schedule shape and shape-specific fields.
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


def _run_to_response(r) -> RoutineRunResponse:
    """Build a RoutineRunResponse from a RoutineRun row, including the
    Ticket 2.1 / 2.3 / 2.5 columns. `getattr` for the new fields so
    pre-migration rows (NULL columns) and older legacy tests don't break."""
    return RoutineRunResponse(
        id=r.id,
        scheduled_for_local_date=r.scheduled_for_local_date,
        started_at=r.started_at,
        finished_at=r.finished_at,
        fire_instant=getattr(r, "fire_instant", None),
        finished_local_at=getattr(r, "finished_local_at", None),
        status=r.status,
        outcome=getattr(r, "outcome", None),
        error_class=r.error_class,
        error_detail=r.error_detail,
        error_json=getattr(r, "error_json", None),
        channel_results_json=getattr(r, "channel_results_json", None),
        tools_invoked_json=getattr(r, "tools_invoked_json", None),
        emails_fetched=r.emails_fetched,
        attempt=r.attempt,
        summary_message_id=r.summary_message_id,
    )


def _row_to_response(routine, recent_runs=()) -> RoutineResponse:
    return RoutineResponse(
        id=routine.id,
        kind=routine.kind,
        name=getattr(routine, "name", None),
        prompt_text=getattr(routine, "prompt_text", None),
        enabled=bool(routine.enabled),
        schedule_kind=getattr(routine, "schedule_kind", "cron") or "cron",
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
    consumers want an empty list, not an error."""
    from app.db.database import async_session_maker
    from app.db.models import Routine, RoutineRun

    user_id = _user_id()
    async with async_session_maker() as db:
        result = await db.execute(
            select(Routine)
            .where(Routine.user_id == user_id)
            .order_by(Routine.created_at)
        )
        routines = list(result.scalars().all())

        out: list[RoutineResponse] = []
        for r in routines:
            runs_result = await db.execute(
                select(RoutineRun)
                .where(RoutineRun.routine_id == r.id)
                .order_by(desc(RoutineRun.started_at))
                .limit(7)
            )
            recent = list(runs_result.scalars().all())
            out.append(_row_to_response(r, recent))
    return out


@router.post("", response_model=RoutineResponse, status_code=201)
async def create_routine(req: RoutineCreate):
    """Create one routine. 409 if an enabled routine of the same kind
    already exists for this user — v1 UX limits each kind to one active
    routine. The schema permits more so the constraint can be relaxed
    later without a migration."""
    _validate_kind(req.kind)
    _kind_enabled_or_404(req.kind)
    # Phase A — cron expression validated only when schedule_kind='cron'.
    # 'at' and 'every' kinds carry their own schedule_at /
    # schedule_interval_seconds (validated by the Pydantic
    # model_validator above).
    if req.schedule_kind == "cron":
        _validate_cron(req.schedule_cron_local)

    # Generic agent_task routines NEED a prompt — that's the whole point.
    # Preset kinds (email_briefing) don't use prompt_text; the handler
    # has its own hard-coded prompt template. Reminders carry their
    # delivery text in `reminder_text` (enforced by the model_validator).
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

        # Phase A — auto-disable defaults to True for one-shot ('at')
        # reminders. The user said "remind me at 5pm" — they expect ONE
        # delivery, not a daily repeat. Explicit True/False from the
        # caller always wins.
        auto_disable = req.auto_disable_after_fire
        if auto_disable is None:
            auto_disable = (req.schedule_kind == "at")

        # `schedule_cron_local` is required by the SQLAlchemy column
        # (nullable=False on legacy schema). For non-cron shapes we
        # store a placeholder string so the column stays satisfied;
        # the runner ignores it whenever schedule_kind != 'cron'.
        # This avoids a destructive schema change for legacy rows.
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

        # Phase A — `schedule_changed` covers any shape edit: cron
        # expression change, kind switch, schedule_at move, interval
        # change, window adjustment. Each invalidates today's failed
        # routine_run rows so a new fire can claim a fresh slot.
        schedule_changed = (
            (req.schedule_cron_local is not None
             and req.schedule_cron_local != routine.schedule_cron_local)
            or (req.schedule_kind is not None
                and req.schedule_kind != getattr(routine, "schedule_kind", "cron"))
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
        # (routine_id, scheduled_for_local_date)), clear any
        # FAILED/SKIPPED run rows for today's local date so the new
        # cron tick can claim a fresh row. We do NOT touch successful
        # runs — those represent real work the user already received.
        # Running rows are also left alone (race-safe).
        if schedule_changed:
            from app.agent.routines.runner import _resolve_tz
            from app.db.models import RoutineRun, User as _User
            user = await db.get(_User, user_id)
            tz, _ = _resolve_tz(getattr(user, "timezone", None), user_id)
            local_today = datetime.now(tz).date()
            await db.execute(
                delete(RoutineRun).where(
                    RoutineRun.routine_id == routine.id,
                    RoutineRun.scheduled_for_local_date == local_today,
                    RoutineRun.status.in_(("failed", "skipped_reauth")),
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
        await db.delete(routine)
        await db.commit()

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
    timing — we run immediately instead of waiting for the cron tick."""
    from app.agent.routines.runner import _resolve_tz
    from app.db.database import async_session_maker
    from app.db.models import Routine, RoutineRun, User

    user_id = _user_id()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
        if routine is None or routine.user_id != user_id:
            raise HTTPException(status_code=404, detail="Routine not found")
        _kind_enabled_or_404(routine.kind)

        # Compute today in the user's tz — same path the runner uses,
        # so the idempotency key matches a scheduled fire on this day.
        user = await db.get(User, user_id)
        tz, _ = _resolve_tz(getattr(user, "timezone", None), user_id)
        local_today = datetime.now(tz).date()

        existing = await db.execute(
            select(RoutineRun).where(
                RoutineRun.routine_id == routine_id,
                RoutineRun.scheduled_for_local_date == local_today,
            )
        )
        prior = existing.scalar_one_or_none()
        if prior is not None:
            # Re-run policy: if today's prior run ended in a recoverable
            # terminal state (failed, skipped_reauth), the user explicitly
            # asking to re-run is a retry — drop the prior row so the
            # idempotency claim in `_fire` succeeds with a fresh row. We
            # KEEP success rows (don't double-deliver to Telegram /
            # WhatsApp) and KEEP running rows (race-safe — another
            # request is already mid-fire).
            if prior.status in ("failed", "skipped_reauth"):
                await db.execute(
                    delete(RoutineRun).where(RoutineRun.id == prior.id)
                )
                await db.commit()
            else:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Today's run already exists for this routine.",
                        "run_id": prior.id,
                        "status": prior.status,
                    },
                )

    if _runner is None:
        raise HTTPException(status_code=503, detail="Routine runner not started")

    # Dispatch synchronously — _fire handles its own idempotency claim,
    # which will succeed since we just verified no prior row exists.
    await _runner._fire(routine_id)

    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineRun).where(
                RoutineRun.routine_id == routine_id,
                RoutineRun.scheduled_for_local_date == local_today,
            )
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
    """Paginated run history. Most-recent first."""
    from app.db.database import async_session_maker
    from app.db.models import Routine, RoutineRun

    user_id = _user_id()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
        if routine is None or routine.user_id != user_id:
            raise HTTPException(status_code=404, detail="Routine not found")
        result = await db.execute(
            select(RoutineRun)
            .where(RoutineRun.routine_id == routine_id)
            .order_by(desc(RoutineRun.started_at))
            .limit(limit)
            .offset(offset)
        )
        runs = list(result.scalars().all())
    return [_run_to_response(r) for r in runs]
