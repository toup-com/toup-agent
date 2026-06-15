"""Maintenance / support agent models (PLATFORM-only).

Two tables back the admin maintenance-and-support system that ingests
user-reported problems, diagnoses them against the platform skills in
``docs/skills/`` (its single source of truth), and — only after admin
approval — implements and verifies a fix as one PR per issue.

* ``support_issues``        — one row per reported problem; carries the
                              normalized report, classification, skill
                              routing, the diagnosis report, the admin
                              decision, and the implementation outcome.
* ``support_issue_events``  — append-only audit trail of every state
                              transition and admin decision (who/what/when).

Both are PLATFORM-only: this is an operator/admin tool that reasons about
the platform's own code, not per-tenant agent data. See
``docs/support-agent/README.md`` for the full workflow and the
``app.support`` package for the pipeline that drives the state machine.

The status/classification/severity/decision string values are the
canonical enums in ``app.support.enums``; columns are plain ``String``
(mirrors ManagedContainer.status) so the evolving state machine never
needs a migration to add a status.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, ForeignKey, Index, JSON, Integer, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SupportIssue(Base):
    """A user-reported problem moving through intake → diagnose → approve → fix."""

    __tablename__ = "support_issues"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )

    # ── Intake (who/what was reported) ────────────────────────────────
    reporter_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    reporter_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="api")
    raw_report: Mapped[str] = mapped_column(Text, nullable=False)
    repro_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Normalized one-line symptom (filled at intake, refined by the LLM).
    symptom: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="medium")

    # ── State machine ────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="intake", index=True)

    # ── Classification (BUG | MISSING_FEATURE | UNCLEAR) ─────────────
    classification: Mapped[Optional[str]] = mapped_column(String(24), nullable=True, index=True)
    classification_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Triage / skill routing (the single source of truth) ──────────
    routed_subsystems: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # list[str] skill names
    routing_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Provenance: the commit the skills reflected when this issue was routed.
    skills_baseline_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    # ── Diagnosis report ─────────────────────────────────────────────
    # {root_cause, blast_radius, proposed_fix, affected_files[], test_plan,
    #  rollback_path, confidence, skills_referenced[]}
    diagnosis_report: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    diagnosis_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Admin approval gate ──────────────────────────────────────────
    decision: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)  # approve|reject|request_changes
    decision_by_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    decision_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    decision_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Implementation / delivery ────────────────────────────────────
    branch_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    pr_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    verification: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True,
    )

    # status + classification are indexed via index=True on their columns
    # (auto-named ix_support_issues_status / _classification); only the
    # composite/extra index is declared here.
    __table_args__ = (
        Index("ix_support_issues_created_at", "created_at"),
    )


class SupportIssueEvent(Base):
    """Append-only audit trail of one SupportIssue's lifecycle."""

    __tablename__ = "support_issue_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    # CASCADE: deleting an issue removes its event trail atomically.
    issue_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("support_issues.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    actor: Mapped[str] = mapped_column(String(16), nullable=False, default="system")  # system|admin|user
    actor_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # issue_id is indexed via index=True on its column
    # (auto-named ix_support_issue_events_issue_id).
    __table_args__ = (
        Index("ix_support_issue_events_created_at", "created_at"),
    )


class SupportAttachment(Base):
    """A binary attachment (e.g. a mobile screenshot) for a SupportIssue.

    Stored in the PLATFORM DB rather than the agent-side file_storage backend:
    Message/Conversation attachments are AGENT_ONLY (per-tenant DB + agent
    workspace disk), but a support card is platform-side admin data and the
    Railway platform-api has ephemeral disk. Bytes live here, bounded by
    settings.support_attachment_max_bytes; they are served ONLY through an
    auth'd, ownership-checked endpoint (reporter or admin) — never a
    world-readable URL and never a token in the query string. Low volume +
    small (capped) images make DB storage appropriate; the S3Backend stub in
    file_storage.py is the documented scale-up seam.
    """

    __tablename__ = "support_attachments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    # CASCADE: deleting an issue removes its attachments atomically.
    issue_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("support_issues.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False, default="screenshot")
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    uploaded_by_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # issue_id is indexed via index=True on its column
    # (auto-named ix_support_attachments_issue_id).
    __table_args__ = (
        Index("ix_support_attachments_created_at", "created_at"),
    )
