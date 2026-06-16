"""Maintenance / support agent API (platform).

Intake is available to any authenticated user; everything else is admin-only
(reuses ``require_admin``). The whole feature is dark-launched behind
``settings.support_agent_enabled`` (503 when off).

Routes (mounted at settings.api_prefix, e.g. /api):
  POST /support/issues                              — intake a problem report (auth user)
  POST /support/issues/{id}/attachment             — upload a screenshot (reporter or admin)
  GET  /support/issues/{id}/attachments/{att_id}   — fetch screenshot bytes (reporter or admin)
  GET  /support/issues                              — list issues (admin)
  GET  /support/issues/{id}                         — issue + events + attachments (admin)
  POST /support/issues/{id}/diagnose               — (re)run classify→route→diagnose (admin)
  POST /support/issues/{id}/decision               — APPROVAL GATE: approve|reject|request_changes (admin)
  GET  /support/skills                             — the skills index the agent routes against (admin)

The approval gate: nothing implements code until an admin POSTs
decision=approve here. On approve (and only if implementation is enabled)
the fix pipeline is spawned off the request path.

Attachments (e.g. mobile screenshots) are stored in the platform DB and served
ONLY through the auth'd, ownership-checked GET endpoint above (reporter who
owns the issue, or an admin) — never a world-readable URL, never a token in a
query string (use the Authorization header; the web admin fetches as a blob).
On intake, a "card arrived" alert goes to settings.support_notify_email
(default mrhx@toup.ai) — distinct from the reporter and the admin account.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, User
from app.services.email_service import render_support_card_email, send_email
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
    GradeRequest,
    IntakeResponse,
    IssueOut,
    IssueDetailOut,
    IssueEventOut,
    IssueListResponse,
    AttachmentMeta,
    AttachmentUploadResponse,
    CorpusResponse,
    GradeTally,
)

logger = logging.getLogger("support.api")

router = APIRouter(prefix="/support", tags=["Support"])


def _is_admin(user: User) -> bool:
    return getattr(user, "role", None) == "admin"


async def _load_issue_for_user(db: AsyncSession, issue_id: str, user: User):
    """Return the issue if the user may see it (reporter who owns it, or admin);
    404 otherwise (no existence leak)."""
    issue = await repo.get_issue(db, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    if _is_admin(user) or issue.reporter_user_id == user.id:
        return issue
    raise HTTPException(status_code=404, detail="issue not found")


async def _notify_admin_new_card(issue_id: str) -> None:
    """Fire the 'a support card arrived' alert to the configurable notify
    address. Own DB session (off the request path). Never raises."""
    if not getattr(settings, "support_notify_enabled", True):
        return
    to_addr = getattr(settings, "support_notify_email", "mrhx@toup.ai")
    try:
        from app.db import async_session_maker
        async with async_session_maker() as db:
            issue = await repo.get_issue(db, issue_id)
            if not issue:
                return
            atts = await repo.list_attachments(db, issue_id)
            base = getattr(settings, "app_public_base_url", "https://toup.ai").rstrip("/")
            queue_url = f"{base}/admin?tab=support&issue={issue_id}"
            subject, html, text = render_support_card_email(
                issue_id=issue.id,
                symptom=issue.symptom,
                severity=issue.severity,
                channel=issue.channel,
                classification=issue.classification,
                reporter=issue.reporter_email or issue.reporter_user_id,
                queue_url=queue_url,
                has_screenshot=bool(atts),
            )
        result = await send_email(to=to_addr, subject=subject, html=html, text=text)
        if not getattr(result, "success", False):
            logger.warning("[support] notify email to %s failed for %s: %s",
                           to_addr, issue_id, getattr(result, "error", "?"))
        else:
            logger.info("[support] notify email sent to %s for %s (%s)",
                        to_addr, issue_id, getattr(result, "provider", "?"))
    except Exception as e:  # pragma: no cover - defensive; alert must never break intake
        logger.warning("[support] notify email errored for %s: %s", issue_id, e)


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
        # Identity is the session user. Only an admin may report on behalf of
        # someone else (support-desk use); non-admins can't spoof reporter_email.
        reporter_email=(
            body.reporter_email if (body.reporter_email and _is_admin(current_user))
            else getattr(current_user, "email", None)
        ),
        tenant_id=body.tenant_id,
        repro_info=body.repro_info,
        severity=body.severity.value,
    )
    # Diagnose off the request path (own DB session).
    pipeline.spawn(pipeline.run_diagnosis_pipeline(issue.id, actor_user_id=current_user.id))
    # Alert the notify target (mrhx@toup.ai by default) that a card arrived —
    # off the request path, never blocks/raises. Distinct from reporter + admin.
    pipeline.spawn(_notify_admin_new_card(issue.id))
    return IntakeResponse(
        id=issue.id, status=issue.status,
        message="Report received. Diagnosis is running; an admin will review before any fix.",
    )


# ── Attachments (reporter who owns the issue, or admin) ───────────────

@router.post("/issues/{issue_id}/attachment", response_model=AttachmentUploadResponse)
async def upload_attachment(
    issue_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AttachmentUploadResponse:
    """Upload a screenshot for an issue. Reporter (owns the issue) or admin.

    Authenticated via the Authorization header — no token in the URL. Bytes
    are validated (mime + size cap) and stored in the platform DB.
    """
    _require_enabled()
    issue = await _load_issue_for_user(db, issue_id, current_user)

    mime = (file.content_type or "").split(";")[0].strip().lower()
    allowed = getattr(settings, "support_attachment_allowed_mime",
                      ["image/png", "image/jpeg", "image/webp"])
    if mime not in allowed:
        raise HTTPException(status_code=415, detail=f"unsupported media type: {mime or 'unknown'}")

    max_bytes = int(getattr(settings, "support_attachment_max_bytes", 3_000_000))
    data = await file.read(max_bytes + 1)
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"attachment exceeds {max_bytes} bytes")

    att = await repo.create_attachment(
        db, issue_id=issue.id, data=data, mime_type=mime, kind="screenshot",
        sha256=hashlib.sha256(data).hexdigest(),
        uploaded_by_user_id=current_user.id,
    )
    return AttachmentUploadResponse(
        id=att.id, issue_id=issue.id, mime_type=att.mime_type, size_bytes=att.size_bytes,
    )


@router.get("/issues/{issue_id}/attachments/{att_id}")
async def get_attachment_bytes(
    issue_id: str,
    att_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Stream an attachment's bytes. Reporter (owns the issue) or admin only.

    Auth via the Authorization header (the web admin fetches as a blob, so no
    token ever appears in a URL). 404 (not 403) on cross-user access — no leak.
    """
    _require_enabled()
    await _load_issue_for_user(db, issue_id, current_user)
    att = await repo.get_attachment(db, att_id)
    if not att or att.issue_id != issue_id:
        raise HTTPException(status_code=404, detail="attachment not found")
    return Response(
        content=att.data,
        media_type=att.mime_type,
        headers={
            "Content-Disposition": f'inline; filename="support-{att_id[:8]}"',
            "Cache-Control": "private, no-store",
        },
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
    counts = await repo.attachment_counts(db, [r.id for r in rows])
    out = []
    for r in rows:
        o = IssueOut.from_model(r)
        o.attachment_count = counts.get(r.id, 0)
        out.append(o)
    return IssueListResponse(issues=out, total=len(out))


@router.get("/corpus", response_model=CorpusResponse)
async def grading_corpus(
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> CorpusResponse:
    """Phase-0 read-back: graded issues + the actionable-rate tally.

    Computed from real DB records (not scattered notes) so the ≥80%-actionable
    gate is a real number before deciding whether to build the autonomous-fixer
    runner. Declared before ``/issues/{issue_id}`` so the literal path wins.
    """
    _require_enabled()
    rows = await repo.list_graded(db)
    tally = await repo.grade_tally(db)
    return CorpusResponse(
        issues=[IssueOut.from_model(r) for r in rows],
        tally=GradeTally.from_dict(tally),
    )


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
    atts = await repo.list_attachments(db, issue_id)
    base = IssueOut.from_model(issue).model_dump()
    base["attachment_count"] = len(atts)
    return IssueDetailOut(
        **base,
        events=[
            IssueEventOut(
                id=e.id, created_at=e.created_at, event_type=e.event_type,
                actor=e.actor, actor_user_id=e.actor_user_id, message=e.message, detail=e.detail,
            )
            for e in events
        ],
        attachments=[AttachmentMeta.from_model(a).model_dump() for a in atts],
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


# ── Admin: Phase-0 diagnosis-quality grade (SEPARATE from approval) ───

@router.post("/issues/{issue_id}/grade", response_model=IssueOut)
async def grade_issue(
    issue_id: str,
    body: GradeRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> IssueOut:
    """Record a Phase-0 diagnosis-quality verdict on an issue.

    ANNOTATION ONLY — this records how good the *diagnosis* was (the ≥80%-
    actionable validation gate) and is STRICTLY SEPARATE from the approval gate:
    it never changes the issue's status, never sets the decision columns, and
    never spawns the implement/fix pipeline. Gradeable in ANY state — you also
    flag a real bug that was wrongly parked here (verdict=other + a note). The
    grade can be overwritten; each grade appends a GRADED audit event.
    """
    _require_enabled()
    issue = await repo.get_issue(db, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="issue not found")
    await repo.set_grade(
        db, issue, verdict=body.verdict.value, note=body.note, actor_user_id=admin.id,
    )
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
