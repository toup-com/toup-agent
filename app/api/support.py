"""Maintenance / support agent API (platform).

Intake is available to any authenticated user; everything else is admin-only
(reuses ``require_admin``). The whole feature is dark-launched behind
``settings.support_agent_enabled`` (503 when off).

Routes (mounted at settings.api_prefix, e.g. /api):
  POST /support/issues                  — intake a problem report (auth user)
  GET  /support/issues                  — list issues (admin)
  GET  /support/issues/{id}             — issue + audit events (admin)
  POST /support/issues/{id}/diagnose    — (re)run classify→route→diagnose (admin)
  POST /support/issues/{id}/decision    — APPROVAL GATE: approve|reject|request_changes (admin)
  GET  /support/skills                  — the skills index the agent routes against (admin)

The approval gate: nothing implements code until an admin POSTs
decision=approve here. On approve (and only if implementation is enabled)
the fix pipeline is spawned off the request path.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, User
from app.support import repository as repo
from app.support import pipeline, skills_index
from app.support.enums import (
    AdminDecision,
    SupportIssueStatus as S,
    SupportClassification,
    SupportEventType as E,
)
from app.support.schemas import (
    IssueIntakeRequest,
    DecisionRequest,
    IntakeResponse,
    IssueOut,
    IssueDetailOut,
    IssueEventOut,
    IssueListResponse,
)

router = APIRouter(prefix="/support", tags=["Support"])


def _require_enabled() -> None:
    if not getattr(settings, "support_agent_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Support agent is disabled (set SUPPORT_AGENT_ENABLED=true).",
        )


# ── Intake (authenticated user) ──────────────────────────────────────

@router.post("/issues", response_model=IntakeResponse)
async def intake_issue(
    body: IssueIntakeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> IntakeResponse:
    """Ingest a problem report and kick off classify→route→diagnose."""
    _require_enabled()
    max_chars = int(getattr(settings, "support_intake_max_chars", 8000))
    if len(body.raw_report) > max_chars:
        raise HTTPException(status_code=413, detail=f"raw_report exceeds {max_chars} chars")

    issue = await repo.create_issue(
        db,
        raw_report=body.raw_report,
        channel=body.channel or "api",
        reporter_user_id=current_user.id,
        reporter_email=body.reporter_email or getattr(current_user, "email", None),
        tenant_id=body.tenant_id,
        repro_info=body.repro_info,
        severity=body.severity.value,
    )
    # Diagnose off the request path (own DB session).
    pipeline.spawn(pipeline.run_diagnosis_pipeline(issue.id, actor_user_id=current_user.id))
    return IntakeResponse(
        id=issue.id, status=issue.status,
        message="Report received. Diagnosis is running; an admin will review before any fix.",
    )


# ── Admin: read ──────────────────────────────────────────────────────

@router.get("/issues", response_model=IssueListResponse)
async def list_issues(
    status_filter: str | None = Query(default=None, alias="status"),
    classification: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> IssueListResponse:
    _require_enabled()
    rows = await repo.list_issues(
        db, status=status_filter, classification=classification, limit=limit, offset=offset,
    )
    return IssueListResponse(issues=[IssueOut.from_model(r) for r in rows], total=len(rows))


@router.get("/issues/{issue_id}", response_model=IssueDetailOut)
async def get_issue(
    issue_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> IssueDetailOut:
    _require_enabled()
    issue = await repo.get_issue(db, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    events = await repo.list_events(db, issue_id)
    base = IssueOut.from_model(issue).model_dump()
    return IssueDetailOut(
        **base,
        events=[
            IssueEventOut(
                id=e.id, created_at=e.created_at, event_type=e.event_type,
                actor=e.actor, actor_user_id=e.actor_user_id, message=e.message, detail=e.detail,
            )
            for e in events
        ],
    )


# ── Admin: (re)diagnose ──────────────────────────────────────────────

@router.post("/issues/{issue_id}/diagnose", response_model=IssueOut)
async def rediagnose(
    issue_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> IssueOut:
    """Manually (re)run the diagnosis pipeline (e.g. after a FAILED LLM call)."""
    _require_enabled()
    issue = await repo.get_issue(db, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    if issue.status in (S.IMPLEMENTING.value, S.VERIFYING.value):
        raise HTTPException(status_code=409, detail=f"cannot re-diagnose while {issue.status}")
    pipeline.spawn(pipeline.run_diagnosis_pipeline(issue_id))
    return IssueOut.from_model(issue)


# ── Admin: THE APPROVAL GATE ─────────────────────────────────────────

@router.post("/issues/{issue_id}/decision", response_model=IssueOut)
async def decide(
    issue_id: str,
    body: DecisionRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> IssueOut:
    """Approve / reject / request changes. NO code is touched before approve.

    On approve: if implementation is enabled, the fix pipeline is spawned;
    otherwise the issue is marked APPROVED and an operator runs the
    implement step in an environment that has a git checkout + gh.
    """
    _require_enabled()
    issue = await repo.get_issue(db, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    if issue.status not in (S.AWAITING_APPROVAL.value, S.CHANGES_REQUESTED.value, S.FAILED.value):
        raise HTTPException(
            status_code=409,
            detail=f"issue is {issue.status}; decisions are only valid while awaiting approval.",
        )
    if issue.classification != SupportClassification.BUG.value:
        raise HTTPException(status_code=409, detail="only BUG issues can be approved for a fix.")

    # Persist the decision (durable, audited).
    issue.decision = body.decision.value
    issue.decision_by_user_id = admin.id
    issue.decision_at = datetime.utcnow()
    issue.decision_notes = body.notes
    db.add(issue)

    if body.decision is AdminDecision.REJECT:
        await repo.set_status(db, issue, S.REJECTED, event_type=E.DECISION, actor="admin",
                              actor_user_id=admin.id, message="Rejected by admin",
                              detail={"decision": "reject", "notes": body.notes})
        return IssueOut.from_model(issue)

    if body.decision is AdminDecision.REQUEST_CHANGES:
        await repo.set_status(db, issue, S.CHANGES_REQUESTED, event_type=E.DECISION, actor="admin",
                              actor_user_id=admin.id, message="Changes requested",
                              detail={"decision": "request_changes", "notes": body.notes})
        # Re-diagnose taking the admin's notes into account.
        pipeline.spawn(pipeline.run_diagnosis_pipeline(issue_id, actor_user_id=admin.id))
        return IssueOut.from_model(issue)

    # APPROVE
    await repo.set_status(db, issue, S.APPROVED, event_type=E.DECISION, actor="admin",
                          actor_user_id=admin.id, message="Approved by admin",
                          detail={"decision": "approve", "notes": body.notes})
    if getattr(settings, "support_auto_implement_on_approve", True):
        pipeline.spawn(pipeline.run_implementation(issue_id, admin_user_id=admin.id))
    return IssueOut.from_model(issue)


# ── Admin: skills index visibility ───────────────────────────────────

@router.get("/skills")
async def list_skills(
    _admin: User = Depends(require_admin),
) -> dict:
    """Expose the skills source-of-truth the agent routes against."""
    _require_enabled()
    return {
        "baseline_sha": skills_index.baseline_sha(),
        "skills_dir": str(skills_index.skills_dir()),
        "skills": [
            {"name": s.name, "path": s.path, "description": s.description}
            for s in skills_index.list_skills()
        ],
        "router_rows": len(skills_index.parse_router_table()),
    }
