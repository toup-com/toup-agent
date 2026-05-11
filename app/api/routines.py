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

from datetime import date, datetime
from typing import Any, Optional

from apscheduler.triggers.cron import CronTrigger
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import desc, select


router = APIRouter(prefix="/routines", tags=["routines"])

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


class RoutineCreate(BaseModel):
    kind: str = Field(..., description="Routine handler kind, e.g. 'email_briefing'")
    schedule_cron_local: str = Field(
        ...,
        description="5-part cron expression evaluated in the user's tz",
        examples=["30 6 * * *"],
    )
    enabled: bool = True
    config: Optional[dict] = None


class RoutineUpdate(BaseModel):
    schedule_cron_local: Optional[str] = None
    enabled: Optional[bool] = None
    config: Optional[dict] = None


class RoutineRunResponse(BaseModel):
    id: str
    scheduled_for_local_date: date
    started_at: datetime
    finished_at: Optional[datetime]
    status: str
    error_class: Optional[str]
    error_detail: Optional[str]
    emails_fetched: int
    attempt: int
    summary_message_id: Optional[str]


class RoutineResponse(BaseModel):
    id: str
    kind: str
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


def _row_to_response(routine, recent_runs=()) -> RoutineResponse:
    return RoutineResponse(
        id=routine.id,
        kind=routine.kind,
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
        recent_runs=[
            RoutineRunResponse(
                id=r.id,
                scheduled_for_local_date=r.scheduled_for_local_date,
                started_at=r.started_at,
                finished_at=r.finished_at,
                status=r.status,
                error_class=r.error_class,
                error_detail=r.error_detail,
                emails_fetched=r.emails_fetched,
                attempt=r.attempt,
                summary_message_id=r.summary_message_id,
            )
            for r in recent_runs
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

    from app.db.database import async_session_maker
    from app.db.models import Routine

    user_id = _user_id()
    async with async_session_maker() as db:
        # One-active-per-kind enforcement
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

        routine = Routine(
            user_id=user_id,
            kind=req.kind,
            schedule_cron_local=req.schedule_cron_local,
            enabled=bool(req.enabled),
            config_json=req.config or None,
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

        if req.schedule_cron_local is not None:
            routine.schedule_cron_local = req.schedule_cron_local
        if req.enabled is not None:
            routine.enabled = bool(req.enabled)
        if req.config is not None:
            routine.config_json = req.config
        routine.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(routine)

    if _runner is not None:
        try:
            await _runner.reload_routine(routine.id)
        except Exception:
            pass

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
    return RoutineRunResponse(
        id=run.id,
        scheduled_for_local_date=run.scheduled_for_local_date,
        started_at=run.started_at,
        finished_at=run.finished_at,
        status=run.status,
        error_class=run.error_class,
        error_detail=run.error_detail,
        emails_fetched=run.emails_fetched,
        attempt=run.attempt,
        summary_message_id=run.summary_message_id,
    )


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
    return [
        RoutineRunResponse(
            id=r.id,
            scheduled_for_local_date=r.scheduled_for_local_date,
            started_at=r.started_at,
            finished_at=r.finished_at,
            status=r.status,
            error_class=r.error_class,
            error_detail=r.error_detail,
            emails_fetched=r.emails_fetched,
            attempt=r.attempt,
            summary_message_id=r.summary_message_id,
        )
        for r in runs
    ]
