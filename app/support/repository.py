"""DB access for the support agent: issues, status transitions, audit events.

Thin async helpers over SupportIssue / SupportIssueEvent. Every status
change goes through ``set_status`` which also writes an audit event, so the
event trail is never out of sync with the issue state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.support import SupportIssue, SupportIssueEvent, SupportAttachment
from app.support.enums import SupportEventType, GRADE_ACTIONABLE


async def create_issue(
    db: AsyncSession,
    *,
    raw_report: str,
    channel: str = "api",
    reporter_user_id: Optional[str] = None,
    reporter_email: Optional[str] = None,
    tenant_id: Optional[str] = None,
    repro_info: Optional[str] = None,
    symptom: Optional[str] = None,
    severity: str = "medium",
    status: str = "intake",
) -> SupportIssue:
    issue = SupportIssue(
        raw_report=raw_report,
        channel=channel,
        reporter_user_id=reporter_user_id,
        reporter_email=reporter_email,
        tenant_id=tenant_id,
        repro_info=repro_info,
        symptom=symptom or (raw_report[:200] if raw_report else None),
        severity=severity,
        status=status,
    )
    db.add(issue)
    await db.flush()  # populate issue.id
    await add_event(
        db, issue.id, SupportEventType.CREATED, actor="user",
        actor_user_id=reporter_user_id,
        message=f"Issue intake via {channel}",
        detail={"severity": severity},
    )
    await db.commit()
    await db.refresh(issue)
    return issue


async def get_issue(db: AsyncSession, issue_id: str) -> Optional[SupportIssue]:
    return await db.get(SupportIssue, issue_id)


async def list_issues(
    db: AsyncSession,
    *,
    status: Optional[str] = None,
    classification: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list:
    stmt = select(SupportIssue)
    if status:
        stmt = stmt.where(SupportIssue.status == status)
    if classification:
        stmt = stmt.where(SupportIssue.classification == classification)
    stmt = stmt.order_by(desc(SupportIssue.created_at)).limit(limit).offset(offset)
    return list((await db.execute(stmt)).scalars().all())


async def list_events(db: AsyncSession, issue_id: str) -> list:
    stmt = (
        select(SupportIssueEvent)
        .where(SupportIssueEvent.issue_id == issue_id)
        .order_by(SupportIssueEvent.created_at)
    )
    return list((await db.execute(stmt)).scalars().all())


async def add_event(
    db: AsyncSession,
    issue_id: str,
    event_type,  # SupportEventType | str
    *,
    actor: str = "system",
    actor_user_id: Optional[str] = None,
    message: Optional[str] = None,
    detail: Optional[dict] = None,
) -> SupportIssueEvent:
    ev = SupportIssueEvent(
        issue_id=issue_id,
        event_type=event_type.value if isinstance(event_type, SupportEventType) else str(event_type),
        actor=actor,
        actor_user_id=actor_user_id,
        message=message,
        detail=detail,
    )
    db.add(ev)
    await db.flush()
    return ev


async def set_status(
    db: AsyncSession,
    issue: SupportIssue,
    new_status,  # SupportIssueStatus | str
    *,
    event_type=SupportEventType.NOTE,
    actor: str = "system",
    actor_user_id: Optional[str] = None,
    message: Optional[str] = None,
    detail: Optional[dict] = None,
    commit: bool = True,
) -> SupportIssue:
    """Transition status + write an audit event in one shot."""
    status_str = new_status.value if hasattr(new_status, "value") else str(new_status)
    prev = issue.status
    issue.status = status_str
    issue.updated_at = datetime.utcnow()
    db.add(issue)
    payload = {"from": prev, "to": status_str}
    if detail:
        payload.update(detail)
    await add_event(
        db, issue.id, event_type, actor=actor, actor_user_id=actor_user_id,
        message=message or f"{prev} → {status_str}", detail=payload,
    )
    if commit:
        await db.commit()
        await db.refresh(issue)
    return issue


# ── Phase-0 diagnosis-quality grading (admin annotation) ──────────────

async def set_grade(
    db: AsyncSession,
    issue: SupportIssue,
    *,
    verdict: str,
    note: Optional[str] = None,
    actor_user_id: Optional[str] = None,
    commit: bool = True,
) -> SupportIssue:
    """Record a Phase-0 diagnosis-quality grade + an audit event.

    Annotation ONLY: this NEVER changes ``issue.status`` and NEVER touches the
    decision/approval columns. Grading how good a diagnosis was is strictly
    separate from approving a fix (which is ``set_status(..., APPROVED)`` +
    the implement pipeline). Re-grading overwrites the verdict and appends a
    fresh GRADED event so the history is preserved.
    """
    issue.grade_verdict = verdict
    issue.grade_note = note
    issue.graded_by_user_id = actor_user_id
    issue.graded_at = datetime.utcnow()
    # NB: issue.updated_at is intentionally left alone — grading is annotation,
    # not a lifecycle change, and must not look like a state transition.
    db.add(issue)
    await add_event(
        db, issue.id, SupportEventType.GRADED, actor="admin",
        actor_user_id=actor_user_id,
        message=f"Graded: {verdict}",
        detail={"verdict": verdict, "note": note},
    )
    if commit:
        await db.commit()
        await db.refresh(issue)
    return issue


async def list_graded(db: AsyncSession, *, limit: int = 500) -> list:
    """Graded issues only, most-recently-graded first (the Phase-0 corpus)."""
    stmt = (
        select(SupportIssue)
        .where(SupportIssue.grade_verdict.isnot(None))
        .order_by(desc(SupportIssue.graded_at))
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


async def grade_tally(db: AsyncSession) -> dict:
    """Counts of graded issues by verdict + the actionable rate, computed from
    real DB records (one grouped query) so the ≥80% gate is a real number."""
    stmt = (
        select(SupportIssue.grade_verdict, func.count(SupportIssue.id))
        .where(SupportIssue.grade_verdict.isnot(None))
        .group_by(SupportIssue.grade_verdict)
    )
    rows = (await db.execute(stmt)).all()
    by_verdict = {v: c for v, c in rows}
    total = sum(by_verdict.values())
    actionable = by_verdict.get(GRADE_ACTIONABLE, 0)
    return {
        "by_verdict": by_verdict,
        "total_graded": total,
        "actionable": actionable,
        "actionable_rate": (actionable / total) if total else 0.0,
    }


# ── Attachments ───────────────────────────────────────────────────────

async def create_attachment(
    db: AsyncSession,
    *,
    issue_id: str,
    data: bytes,
    mime_type: str,
    kind: str = "screenshot",
    sha256: Optional[str] = None,
    uploaded_by_user_id: Optional[str] = None,
    commit: bool = True,
) -> SupportAttachment:
    att = SupportAttachment(
        issue_id=issue_id,
        data=data,
        mime_type=mime_type,
        size_bytes=len(data),
        kind=kind,
        sha256=sha256,
        uploaded_by_user_id=uploaded_by_user_id,
    )
    db.add(att)
    await db.flush()
    await add_event(
        db, issue_id, SupportEventType.NOTE, actor="user",
        actor_user_id=uploaded_by_user_id,
        message=f"Attachment uploaded ({kind}, {len(data)} bytes)",
        detail={"attachment_id": att.id, "mime_type": mime_type, "size_bytes": len(data)},
    )
    if commit:
        await db.commit()
        await db.refresh(att)
    return att


async def get_attachment(db: AsyncSession, att_id: str) -> Optional[SupportAttachment]:
    return await db.get(SupportAttachment, att_id)


async def list_attachments(db: AsyncSession, issue_id: str) -> list:
    stmt = (
        select(SupportAttachment)
        .where(SupportAttachment.issue_id == issue_id)
        .order_by(SupportAttachment.created_at)
    )
    return list((await db.execute(stmt)).scalars().all())


async def attachment_counts(db: AsyncSession, issue_ids: list) -> dict:
    """Map issue_id -> attachment count, in one grouped query (no N+1)."""
    if not issue_ids:
        return {}
    stmt = (
        select(SupportAttachment.issue_id, func.count(SupportAttachment.id))
        .where(SupportAttachment.issue_id.in_(issue_ids))
        .group_by(SupportAttachment.issue_id)
    )
    rows = (await db.execute(stmt)).all()
    return {iid: cnt for iid, cnt in rows}
