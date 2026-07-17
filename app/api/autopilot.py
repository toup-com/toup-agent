"""Autopilot approvals API — agent entrypoint (Autopilot PR7).

Mounted in agent_main (single-tenant container; user resolved from
``settings.user_id`` like jobs_events.py — the platform proxy adds
X-Agent-Key upstream). Mission Control and push-notification deep
links reach these through the platform proxy (PR8).

- ``GET  /autopilot/approvals``            — list (default: pending)
- ``POST /autopilot/approvals/{id}/decision`` — approve / deny / answer

A decision RESUMES the mission: the answer lands in the mission's
``user_answers`` (injected into the next tick's prompt, one-shot),
status flips back to ``active``, ``next_due_at`` clears so the very
next scheduler fire runs tier 2. Idempotent: deciding an already
decided approval returns the stored decision (push action buttons can
fire twice, Expo killed-state caveat).

Route order note: literal paths only under the /autopilot prefix — no
``{param}`` catch-alls that could shadow (the /jobs/events lesson).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    AutopilotApproval,
    APPROVAL_ANSWERED, APPROVAL_APPROVED, APPROVAL_DENIED, APPROVAL_PENDING,
    Routine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/autopilot", tags=["Autopilot"])


def _get_user_id() -> str:
    """Single-tenant agent mode — mirror of jobs_events._get_user_id."""
    return settings.user_id


# ── Mission creation (shared by the API and the start_mission tool) ─


class MissionCreateError(Exception):
    def __init__(self, reason: str, message: str):
        self.reason = reason
        self.message = message
        super().__init__(message)


async def create_mission(
    *,
    user_id: str,
    goal: str,
    name: Optional[str] = None,
    budget_credits: Optional[float] = None,
    urgent: bool = False,
    interval_seconds: Optional[int] = None,
) -> Routine:
    """Persist a mission (Routine kind='autopilot') and register its
    tick trigger. The hand-off primitive — used by POST
    /autopilot/missions and the chat start_mission tool.

    Raises MissionCreateError('missing_timezone') for tz-less users:
    RoutineRunner SILENTLY skips trigger registration without a tz
    (runner.py registration guard), which would create a mission that
    never ticks — fail loudly instead (the routines-create lesson)."""
    from app.db.models import User

    goal = (goal or "").strip()
    if not goal:
        raise MissionCreateError("no_goal", "A mission needs a goal.")

    async with async_session_maker() as db:
        user = await db.get(User, user_id)
    if user is None or not getattr(user, "timezone", None):
        raise MissionCreateError(
            "missing_timezone",
            "Your timezone isn't set, so mission ticks can't be "
            "scheduled. Open the chat once (we capture it from your "
            "browser) or set it in account settings, then try again.",
        )

    budget = float(budget_credits or settings.autopilot_default_budget_credits)
    interval = int(interval_seconds or settings.autopilot_base_interval_seconds)
    interval = max(60, interval)

    routine = Routine(
        user_id=user_id,
        kind="autopilot",
        name=(name or goal[:80]).strip(),
        # Non-nullable legacy column; ignored for schedule_kind='every'
        # (same placeholder the routines API writes).
        schedule_cron_local="* * * * *",
        schedule_kind="every",
        schedule_interval_seconds=interval,
        enabled=True,
        config_json={
            "goal": goal,
            "budget_credits": budget,
            "urgent": bool(urgent),
        },
        last_state_json={"status": "active"},
        last_status="never_run",
    )
    async with async_session_maker() as db:
        db.add(routine)
        await db.commit()
        await db.refresh(routine)
        db.expunge(routine)

    # Register the tick trigger now (the reconcile loop would catch it
    # within 10 minutes anyway — this makes hand-off immediate).
    try:
        from app.api import routines as routines_api
        if routines_api._runner is not None:
            await routines_api._runner.reload_routine(routine.id)
    except Exception as e:  # noqa: BLE001 — reconcile loop is the backstop
        logger.warning("autopilot: trigger reload failed for %s: %s",
                       routine.id, e)

    # Announce the hand-off through the durable outbox: on the platform
    # this starts the phone's Live Activity (lock-screen progress card +
    # Dynamic Island) via APNs push-to-start. Best-effort — a mission is
    # never blocked on notification plumbing.
    try:
        from app.services.agent_notify_client import notify
        await notify(
            event_kind="mission_started",
            title=f"🚀 Autopilot engaged: {routine.name}"[:200],
            body=goal[:300],
            data={
                "route": "mission-control",
                "mission_id": routine.id,
                "mission_title": routine.name,
                "progress": 0,
                "urgent": bool(urgent),
            },
            priority="default",
            dedup_key=f"{routine.id}:mission_started",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("autopilot: mission_started notify failed for %s: %s",
                       routine.id, e)
    return routine


class MissionCreateIn(BaseModel):
    goal: str = Field(min_length=3, max_length=4000)
    name: Optional[str] = Field(default=None, max_length=100)
    budget_credits: Optional[float] = Field(default=None, ge=1, le=10000)
    urgent: bool = False
    interval_seconds: Optional[int] = Field(default=None, ge=60, le=86400)


class MissionOut(BaseModel):
    id: str
    name: Optional[str]
    goal: str
    enabled: bool
    status: str
    status_reason: Optional[str] = None
    progress: int = 0
    budget_credits: float
    spent_credits: float
    ticks_run: int
    last_summary: Optional[str] = None
    next_due_at: Optional[str] = None
    last_run_at: Optional[str] = None
    created_at: Optional[str] = None
    pending_approval_id: Optional[str] = None


def _progress_int(state: dict, status: str) -> int:
    """Progress as a guaranteed finite int 0-100.

    completed → always 100, even for missions finished under images
    that predate the progress field (their last_state_json has no key
    forever, which rendered as 0% — or NaN% on unguarded clients,
    founder bug 2026-07-16). Junk state values clamp to 0 instead of
    500ing the whole missions list."""
    if status == "completed":
        return 100
    try:
        return max(0, min(100, int(float(state.get("progress", 0) or 0))))
    except (TypeError, ValueError):
        return 0


def _mission_out(r: Routine) -> MissionOut:
    cfg = r.config_json or {}
    state = r.last_state_json or {}
    _status = state.get("status", "active")
    return MissionOut(
        id=r.id,
        name=r.name,
        goal=cfg.get("goal") or (r.prompt_text or ""),
        enabled=bool(r.enabled),
        status=_status,
        status_reason=state.get("status_reason"),
        progress=_progress_int(state, _status),
        budget_credits=float(cfg.get("budget_credits")
                             or settings.autopilot_default_budget_credits),
        spent_credits=float(state.get("spent_credits", 0.0)),
        ticks_run=int(state.get("ticks_run", 0)),
        last_summary=state.get("last_summary"),
        next_due_at=state.get("next_due_at"),
        last_run_at=r.last_run_at.isoformat() if r.last_run_at else None,
        created_at=r.created_at.isoformat() if getattr(r, "created_at", None) else None,
        pending_approval_id=state.get("pending_approval_id"),
    )


@router.post("/missions", response_model=MissionOut)
async def create_mission_endpoint(body: MissionCreateIn) -> MissionOut:
    try:
        routine = await create_mission(
            user_id=_get_user_id(),
            goal=body.goal,
            name=body.name,
            budget_credits=body.budget_credits,
            urgent=body.urgent,
            interval_seconds=body.interval_seconds,
        )
    except MissionCreateError as e:
        raise HTTPException(400, {"reason": e.reason, "message": e.message})
    return _mission_out(routine)


@router.get("/missions", response_model=List[MissionOut])
async def list_missions(
    include_done: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=200),
) -> List[MissionOut]:
    user_id = _get_user_id()
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Routine)
            .where(Routine.user_id == user_id, Routine.kind == "autopilot")
            .order_by(Routine.created_at.desc())
            .limit(limit)
        )).scalars().all()
    out = [_mission_out(r) for r in rows]
    if not include_done:
        out = [m for m in out if m.status not in ("completed", "failed", "cancelled")]
    return out


async def _get_mission(mission_id: str) -> Routine:
    user_id = _get_user_id()
    async with async_session_maker() as db:
        r = await db.get(Routine, mission_id)
        if r is None or r.user_id != user_id or r.kind != "autopilot":
            raise HTTPException(404, "Mission not found")
        db.expunge(r)
        return r


@router.get("/missions/{mission_id}", response_model=MissionOut)
async def get_mission(mission_id: str) -> MissionOut:
    return _mission_out(await _get_mission(mission_id))


async def _set_mission(
    mission_id: str, *, enabled: Optional[bool] = None,
    state_patch: Optional[Dict[str, Any]] = None,
) -> MissionOut:
    r = await _get_mission(mission_id)
    now = datetime.utcnow()
    state = dict(r.last_state_json or {})
    if state_patch:
        state.update(state_patch)
    values: Dict[str, Any] = {"last_state_json": state, "updated_at": now}
    if enabled is not None:
        values["enabled"] = enabled
    async with async_session_maker() as db:
        await db.execute(
            update(Routine).where(Routine.id == mission_id).values(**values)
        )
        await db.commit()
    try:
        from app.api import routines as routines_api
        if routines_api._runner is not None:
            await routines_api._runner.reload_routine(mission_id)
    except Exception:  # noqa: BLE001 — reconcile loop is the backstop
        pass
    return _mission_out(await _get_mission(mission_id))


@router.post("/missions/{mission_id}/pause", response_model=MissionOut)
async def pause_mission(mission_id: str) -> MissionOut:
    """Pause = stop ticking, keep all state. Resume continues where it
    left off."""
    return await _set_mission(mission_id, enabled=False)


@router.post("/missions/{mission_id}/resume", response_model=MissionOut)
async def resume_mission(mission_id: str) -> MissionOut:
    """Resume ticking. A blocked mission (budget/platform) is set back
    to active — the user pressing resume IS the human decision; the
    tier-1 gates re-evaluate immediately on the next fire anyway."""
    r = await _get_mission(mission_id)
    state = dict(r.last_state_json or {})
    patch: Dict[str, Any] = {"next_due_at": None}
    if state.get("status") == "blocked":
        patch.update({"status": "active", "status_reason": "resumed_by_user",
                      "platform_fail_streak": 0})
    return await _set_mission(mission_id, enabled=True, state_patch=patch)


@router.post("/missions/{mission_id}/cancel", response_model=MissionOut)
async def cancel_mission(mission_id: str) -> MissionOut:
    return await _set_mission(
        mission_id, enabled=False,
        state_patch={
            "status": "cancelled",
            "status_reason": "cancelled_by_user",
            "next_due_at": None,
        },
    )


class BudgetIn(BaseModel):
    budget_credits: float = Field(ge=1, le=10000)


@router.post("/missions/{mission_id}/budget", response_model=MissionOut)
async def raise_mission_budget(mission_id: str, body: BudgetIn) -> MissionOut:
    """Raise (or set) the mission's credit budget. The 'hit its budget'
    push tells the user to do exactly this — a budget above what's
    already spent un-blocks a budget_exhausted mission and re-enables
    ticking, same reactivation semantics as resume."""
    r = await _get_mission(mission_id)
    state = dict(r.last_state_json or {})
    spent = float(state.get("spent_credits", 0.0))
    if body.budget_credits <= spent:
        raise HTTPException(
            422,
            f"budget_credits must exceed the {spent:.1f} already spent",
        )
    # MERGE into config_json — never wholesale-replace (goal/urgent
    # live there too).
    cfg = dict(r.config_json or {})
    cfg["budget_credits"] = body.budget_credits
    now = datetime.utcnow()
    values: Dict[str, Any] = {"config_json": cfg, "updated_at": now}
    patch: Dict[str, Any] = {}
    if state.get("status") == "blocked" and state.get("status_reason") == "budget_exhausted":
        patch = {"status": "active", "status_reason": "budget_raised",
                 "platform_fail_streak": 0, "next_due_at": None}
        state.update(patch)
        values["last_state_json"] = state
        values["enabled"] = True
    async with async_session_maker() as db:
        await db.execute(
            update(Routine).where(Routine.id == mission_id).values(**values)
        )
        await db.commit()
    try:
        from app.api import routines as routines_api
        if routines_api._runner is not None:
            await routines_api._runner.reload_routine(mission_id)
    except Exception:  # noqa: BLE001 — reconcile loop is the backstop
        pass
    return _mission_out(await _get_mission(mission_id))


class ApprovalOut(BaseModel):
    id: str
    mission_id: str
    kind: str
    title: str
    payload: Optional[Dict[str, Any]] = None
    status: str
    answer_text: Optional[str] = None
    created_at: str
    decided_at: Optional[str] = None


class DecisionIn(BaseModel):
    # 'approve' | 'deny' | 'answer' (answer requires answer_text)
    decision: str
    answer_text: Optional[str] = Field(default=None, max_length=4000)
    decided_via: str = Field(default="mission_control", max_length=32)


def _to_out(a: AutopilotApproval) -> ApprovalOut:
    return ApprovalOut(
        id=a.id,
        mission_id=a.mission_id,
        kind=a.kind,
        title=a.title,
        payload=a.payload_json,
        status=a.status,
        answer_text=a.answer_text,
        created_at=a.created_at.isoformat() if a.created_at else "",
        decided_at=a.decided_at.isoformat() if a.decided_at else None,
    )


@router.get("/approvals", response_model=List[ApprovalOut])
async def list_approvals(
    status: Optional[str] = Query(default=APPROVAL_PENDING),
    mission_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> List[ApprovalOut]:
    user_id = _get_user_id()
    stmt = (
        select(AutopilotApproval)
        .where(AutopilotApproval.user_id == user_id)
        .order_by(AutopilotApproval.created_at.desc())
        .limit(limit)
    )
    if status and status != "all":
        stmt = stmt.where(AutopilotApproval.status == status)
    if mission_id:
        stmt = stmt.where(AutopilotApproval.mission_id == mission_id)
    async with async_session_maker() as db:
        rows = (await db.execute(stmt)).scalars().all()
    return [_to_out(a) for a in rows]


_DECISION_TO_STATUS = {
    "approve": APPROVAL_APPROVED,
    "deny": APPROVAL_DENIED,
    "answer": APPROVAL_ANSWERED,
}


@router.post("/approvals/{approval_id}/decision", response_model=ApprovalOut)
async def decide_approval(approval_id: str, body: DecisionIn) -> ApprovalOut:
    if body.decision not in _DECISION_TO_STATUS:
        raise HTTPException(422, f"decision must be one of {sorted(_DECISION_TO_STATUS)}")
    if body.decision == "answer" and not (body.answer_text or "").strip():
        raise HTTPException(422, "answer_text required for decision='answer'")

    user_id = _get_user_id()
    now = datetime.utcnow()
    async with async_session_maker() as db:
        approval = await db.get(AutopilotApproval, approval_id)
        if approval is None or approval.user_id != user_id:
            raise HTTPException(404, "Approval not found")

        # Idempotent replay: push action buttons can fire twice.
        if approval.status != APPROVAL_PENDING:
            return _to_out(approval)

        approval.status = _DECISION_TO_STATUS[body.decision]
        approval.answer_text = (body.answer_text or "").strip() or None
        approval.decided_at = now
        approval.decided_via = body.decided_via

        # Resume the mission: answer into user_answers (one-shot tick
        # context), status back to active, due immediately. Merge into
        # the existing watermark — the engine owns other keys.
        routine = await db.get(Routine, approval.mission_id)
        if routine is not None and routine.kind == "autopilot":
            state = dict(routine.last_state_json or {})
            answers = list(state.get("user_answers") or [])
            answers.append({
                "question": approval.title,
                "decision": body.decision,
                "answer": approval.answer_text or body.decision,
                "decided_at": now.replace(microsecond=0).isoformat(),
            })
            state.update({
                "user_answers": answers[-10:],
                "status": "active",
                "status_reason": f"resumed_by_{body.decision}",
                "next_due_at": None,
                "no_progress_streak": 0,
                "pending_approval_id": None,
            })
            await db.execute(
                update(Routine)
                .where(Routine.id == routine.id)
                .values(last_state_json=state, enabled=True, updated_at=now)
            )

        out = _to_out(approval)
        await db.commit()

    logger.info(
        "[autopilot] approval %s → %s (mission %s resumed)",
        approval_id, body.decision, out.mission_id,
    )
    return out
