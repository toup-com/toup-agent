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
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, desc, select


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


class RoutineCreate(BaseModel):
    kind: str = Field(..., description="Routine handler kind: 'email_briefing' or 'agent_task'")
    schedule_cron_local: str = Field(
        ...,
        description="5-part cron expression evaluated in the user's tz",
        examples=["30 6 * * *"],
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


class RoutineUpdate(BaseModel):
    schedule_cron_local: Optional[str] = None
    enabled: Optional[bool] = None
    config: Optional[dict] = None
    name: Optional[str] = Field(default=None, max_length=100)
    prompt_text: Optional[str] = None
    delivery_channels: Optional[list[str]] = None

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
    schedule_cron_local: str
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
        schedule_cron_local=routine.schedule_cron_local,
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
    _validate_cron(req.schedule_cron_local)

    # Generic agent_task routines NEED a prompt — that's the whole point.
    # Preset kinds (email_briefing) don't use prompt_text; the handler
    # has its own hard-coded prompt template.
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
        # distinct task ("morning briefing", "noon GitHub check", etc.)
        # and per-task uniqueness on (kind=agent_task) would be useless.
        if req.kind != "agent_task":
            existing = await db.execute(
                select(Routine).where(
                    Routine.user_id == user_id,
                    Routine.kind == req.kind,
                    Routine.enabled == True,  # noqa: E712
                )
            )
            if existing.scalar_one_or_none() is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"An enabled {req.kind!r} routine already exists. "
                           f"Disable or delete it first.",
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

        routine = Routine(
            user_id=user_id,
            kind=req.kind,
            name=(req.name or "").strip() or None,
            prompt_text=(req.prompt_text or "").strip() or None,
            schedule_cron_local=req.schedule_cron_local,
            enabled=bool(req.enabled),
            config_json=merged_config,
            last_status="never_run",
        )
        db.add(routine)
        await db.commit()
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

        schedule_changed = (
            req.schedule_cron_local is not None
            and req.schedule_cron_local != routine.schedule_cron_local
        )

        if req.schedule_cron_local is not None:
            routine.schedule_cron_local = req.schedule_cron_local
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

        await db.commit()
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
