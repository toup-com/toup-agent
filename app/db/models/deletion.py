"""Account deletion: audit receipts + sensitive-action token redemptions.

Two tables, both platform-only, both written under the §1.3 fresh-JWT +
audit-envelope account-deletion design. They are co-located here because
they only exist to make `cascade_delete_user` safe and durable.

  - DeletionAuditEvent  — one row per delete attempt. Outlives the user.
                          Receipt-of-deletion artifact for compliance.
  - SensitiveActionRedemption — single-use jti redemption set for the
                          fresh-JWT replay defense on POST /auth/delete-
                          account (and any future sensitive endpoint).
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Boolean, Index, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# ── Constants for status / actor / failure-step columns ──────────────────
# Defined as module-level tuples so the CheckConstraint and the
# StrEnum (in app.services.user_deletion) can share one source of truth.
DELETION_STATUSES = ("in_progress", "succeeded", "failed")
DELETION_ACTORS = ("self", "admin")

# Failure-step values that DeletionAbortedError.step.value can take.
# Listed here mostly as documentation; the column is free-text in case
# future steps are added without migration. New step values: add to the
# StrEnum in user_deletion.py + remember to update client UX mappings.
# Known values: stripe, container, openai, cascade, user_row.


class DeletionAuditEvent(Base):
    """Receipt of a delete-user attempt. INSERTed at step 0 with
    status='in_progress', UPDATEd at step 8 with the final status.

    Retention: indefinite for non-PII fields; 7 years for the three
    PII columns (user_email, request_ip, request_user_agent), then a
    sweep job (TODO §1.3-followup) hashes them irreversibly.
    """

    __tablename__ = "deletion_audit_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    # PII — subject to 7-year retention then irreversibly hashed.
    user_email: Mapped[str] = mapped_column(Text, nullable=False)
    request_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    request_user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Who triggered. actor_user_id == user_id for actor='self'.
    actor: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    initiated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # State machine: in_progress → succeeded | failed.
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="in_progress")

    # Populated on failure.
    failure_step: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    failure_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Per-step success flags. NULL means "not reached yet" or "n/a".
    container_destroyed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    stripe_cleared: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    # NULL openai_archived means n/a (user had no per-tenant project).
    openai_archived: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Marker for the redaction sweep job (TODO §1.3-followup). When set,
    # user_email/request_ip/request_user_agent have been replaced with
    # sha256(value || internal_salt).
    pii_redacted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            f"status IN {DELETION_STATUSES}",
            name="ck_deletion_audit_events_status",
        ),
        CheckConstraint(
            f"actor IN {DELETION_ACTORS}",
            name="ck_deletion_audit_events_actor",
        ),
        Index("ix_deletion_audit_events_initiated_at", "initiated_at"),
    )


class SensitiveActionRedemption(Base):
    """Redeemed-jti set for fresh-JWT replay defense on sensitive
    endpoints. One row per successful sensitive-action token use.

    Lifecycle: INSERTed by `verify_sensitive_action_token` on a
    successful redeem. Periodic sweep (`DELETE WHERE expires_at <
    now() - interval '1 day'`) keeps the table small. The sweep is
    out of scope for §1.3 — TODO §1.3-followup. A few weeks of dead
    rows is harmless; keying by jti and indexing on expires_at means
    queries stay O(log n).
    """

    __tablename__ = "sensitive_action_redemptions"

    jti: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    redeemed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_sensitive_action_redemptions_expires_at", "expires_at"),
    )
