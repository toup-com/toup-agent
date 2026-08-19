"""Pydantic request/response models for the support API."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field

from app.support.enums import AdminDecision, IssueSeverity, SupportGradeVerdict


# ── Intake ───────────────────────────────────────────────────────────

class IssueIntakeRequest(BaseModel):
    """A problem report. ``raw_report`` is required; everything else helps
    triage. Length-capped by settings.support_intake_max_chars at the API."""
    raw_report: str = Field(..., min_length=3)
    channel: str = Field(default="api", max_length=32)
    repro_info: Optional[str] = Field(default=None)
    severity: IssueSeverity = Field(default=IssueSeverity.MEDIUM)
    tenant_id: Optional[str] = Field(default=None, max_length=36)
    # Optional: report on behalf of another user (admin/support desk use).
    reporter_email: Optional[str] = Field(default=None, max_length=255)
    # Optional structured capture context — `screen`, `app_version`, `build`,
    # `platform`, `device`, `os` — for a client that would rather not encode
    # them as `repro_info` lines. Unknown keys are dropped and every value is
    # capped by `report_thread.parse_report_context`; the shipped mobile client
    # sends none of this and its `repro_info` block is parsed instead.
    context: Optional[Dict[str, str]] = Field(default=None)


class DecisionRequest(BaseModel):
    decision: AdminDecision
    notes: Optional[str] = Field(default=None, max_length=4000)


class GradeRequest(BaseModel):
    """Phase-0 diagnosis-quality grade. Annotation only — does NOT advance the
    lifecycle or touch the approval gate (see api/support.py grade_issue)."""
    verdict: SupportGradeVerdict
    note: Optional[str] = Field(default=None, max_length=4000)


# ── Responses ────────────────────────────────────────────────────────

class AttachmentMeta(BaseModel):
    """Attachment metadata — never the bytes. Bytes come from the auth'd
    GET /support/issues/{id}/attachments/{att_id} endpoint."""
    id: str
    kind: str
    mime_type: str
    size_bytes: int
    created_at: Optional[datetime]

    @classmethod
    def from_model(cls, m) -> "AttachmentMeta":
        return cls(
            id=m.id, kind=m.kind, mime_type=m.mime_type,
            size_bytes=m.size_bytes, created_at=m.created_at,
        )


class AttachmentUploadResponse(BaseModel):
    id: str
    issue_id: str
    mime_type: str
    size_bytes: int


class IssueEventOut(BaseModel):
    id: str
    created_at: Optional[datetime]
    event_type: str
    actor: str
    actor_user_id: Optional[str]
    message: Optional[str]
    detail: Optional[dict]


class IssueOut(BaseModel):
    id: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    status: str
    channel: str
    severity: str
    symptom: Optional[str]
    raw_report: str
    repro_info: Optional[str]
    reporter_user_id: Optional[str]
    reporter_email: Optional[str]
    tenant_id: Optional[str]
    classification: Optional[str]
    classification_rationale: Optional[str]
    routed_subsystems: Optional[list]
    routing_rationale: Optional[str]
    skills_baseline_sha: Optional[str]
    diagnosis_report: Optional[dict]
    diagnosis_summary: Optional[str]
    decision: Optional[str]
    decision_by_user_id: Optional[str]
    decision_at: Optional[datetime]
    decision_notes: Optional[str]
    branch_name: Optional[str]
    pr_url: Optional[str]
    verification: Optional[dict]
    error: Optional[str]
    # Phase-0 diagnosis-quality grade (admin annotation; separate from decision).
    grade_verdict: Optional[str]
    grade_note: Optional[str]
    graded_by_user_id: Optional[str]
    graded_at: Optional[datetime]
    # Populated by the endpoints (not from_model) — attachments are a
    # separate platform-DB query.
    attachment_count: int = 0

    @classmethod
    def from_model(cls, m) -> "IssueOut":
        return cls(
            id=m.id, created_at=m.created_at, updated_at=m.updated_at,
            status=m.status, channel=m.channel, severity=m.severity,
            symptom=m.symptom, raw_report=m.raw_report, repro_info=m.repro_info,
            reporter_user_id=m.reporter_user_id, reporter_email=m.reporter_email,
            tenant_id=m.tenant_id, classification=m.classification,
            classification_rationale=m.classification_rationale,
            routed_subsystems=m.routed_subsystems, routing_rationale=m.routing_rationale,
            skills_baseline_sha=m.skills_baseline_sha,
            diagnosis_report=m.diagnosis_report, diagnosis_summary=m.diagnosis_summary,
            decision=m.decision, decision_by_user_id=m.decision_by_user_id,
            decision_at=m.decision_at, decision_notes=m.decision_notes,
            branch_name=m.branch_name, pr_url=m.pr_url,
            verification=m.verification, error=m.error,
            grade_verdict=m.grade_verdict, grade_note=m.grade_note,
            graded_by_user_id=m.graded_by_user_id, graded_at=m.graded_at,
        )


class IssueDetailOut(IssueOut):
    events: list = Field(default_factory=list)
    attachments: list = Field(default_factory=list)  # list[AttachmentMeta]


class IntakeResponse(BaseModel):
    id: str
    status: str
    message: str


class IssueListResponse(BaseModel):
    issues: list
    total: int


class GradeTally(BaseModel):
    """Phase-0 corpus tally, computed from real DB records."""
    by_verdict: dict          # {verdict: count}
    total_graded: int
    actionable: int
    actionable_rate: float    # 0.0–1.0; the ≥0.80 gate

    @classmethod
    def from_dict(cls, d: dict) -> "GradeTally":
        return cls(
            by_verdict=d.get("by_verdict", {}),
            total_graded=d.get("total_graded", 0),
            actionable=d.get("actionable", 0),
            actionable_rate=d.get("actionable_rate", 0.0),
        )


class CorpusResponse(BaseModel):
    """Phase-0 read-back: graded issues + the actionable-rate tally."""
    issues: list              # list[IssueOut] (graded only)
    tally: GradeTally
