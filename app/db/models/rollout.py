"""
Rollout models — Phase 3 automated deployment pipeline.

A `Rollout` is one attempt to move every tenant container onto a new agent
image SHA. A `RolloutAttempt` is the per-tenant sub-record: which container,
prior tag, new tag, outcome, error, timings.

The rollout service (app/services/rollout_service.py) orchestrates:
  canary → wait → batch-parallel rest → per-tenant rollback on failure.

See docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md §5.1 for the algorithm
and §6.1 for the schema rationale.

Platform DB only — agents don't care about rollout history.
"""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import String, Text, DateTime, Integer, ForeignKey, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base


# Status enums as plain strings (not SQLEnum) for easy extension and
# round-trips through JSON / logs. Valid values enforced at the service
# layer, not at the DB schema — we want to be able to add new states
# without a migration.
ROLLOUT_STATUSES = {
    "pending",                 # row created, worker not yet picked up
    "running",                 # worker is actively upgrading tenants
    "complete",                # all tenants processed (some may have failed)
    "aborted_canary_failed",   # canary failed; no further tenants touched
    "cancelled",               # operator cancelled mid-flight
}

ROLLOUT_ATTEMPT_STATUSES = {
    "pending",
    "upgrading",
    "ok",                      # tenant on new_tag, healthy
    "failed",                  # tenant upgrade failed, rollback not yet attempted
    "rolled_back",             # upgrade failed, rolled back to prior_tag successfully
    "rollback_failed",         # both upgrade AND rollback failed; manual intervention needed
    "skipped_pinned",          # tenant had pin_image_tag set; not touched
    "skipped_canary",          # canary tenant skipped during Phase B (already upgraded in Phase A)
}

ROLLOUT_TRIGGERS = {
    "ci",                      # GitHub Actions webhook after build
    "manual",                  # operator via admin UI or POST /api/admin/rollout/manual
    "rollback",                # operator-triggered rollback to a known-good SHA
}


class Rollout(Base):
    """One rollout attempt — moving every tenant to a new agent image SHA."""

    __tablename__ = "rollouts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # ghcr.io/toup-com/toup-agent:<short-sha> — validated at API boundary.
    # Prefix is checked to prevent arbitrary images via ROLLOUT_SECRET leak.
    image_tag: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # See ROLLOUT_STATUSES above.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)

    # See ROLLOUT_TRIGGERS above.
    trigger: Mapped[str] = mapped_column(String(32), nullable=False)

    # User ID for manual triggers; NULL for CI-driven rollouts.
    triggered_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True
    )

    # 8-char user_id prefix of the canary tenant for this rollout. Captured
    # at rollout-start time so the canary is stable even if is_canary moves
    # mid-rollout.
    canary_prefix: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    # Configurable per-rollout; 10 min default from the API.
    canary_wait_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=10)

    # Persisted rollout phase. Used by the startup resumer to figure out where
    # to pick up an orphan: "" (just created) → "canary_upgrading" →
    # "canary_observing" → "batching" → "" (terminal). The resumer only ever
    # touches rollouts with status='running' AND phase='canary_observing'.
    phase: Mapped[str] = mapped_column(String(32), nullable=False, default="", server_default="")

    # Wall-clock deadline for the canary observation window. Set when entering
    # phase='canary_observing'. The startup resumer uses this to compute
    # remaining wait (or skip the wait if the deadline has already passed).
    resume_after: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Free-form operator notes; reason for manual rollouts, aborted cause, etc.
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship to attempts (eager-loaded via joinedload in service code)
    attempts: Mapped[List["RolloutAttempt"]] = relationship(
        "RolloutAttempt",
        back_populates="rollout",
        cascade="all, delete-orphan",
        order_by="RolloutAttempt.started_at",
    )

    __table_args__ = (
        Index("ix_rollouts_started_at", "started_at"),
    )


class RolloutAttempt(Base):
    """One tenant's slice of a rollout — upgrade + optional rollback."""

    __tablename__ = "rollout_attempts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    rollout_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("rollouts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    container_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("managed_containers.id"), nullable=False, index=True
    )

    # Tag the container was on BEFORE this attempt. Needed for rollback.
    prior_tag: Mapped[str] = mapped_column(String(255), nullable=False)
    # Tag the container is being upgraded TO. Matches rollouts.image_tag unless
    # this is a rollback attempt.
    new_tag: Mapped[str] = mapped_column(String(255), nullable=False)

    # See ROLLOUT_ATTEMPT_STATUSES above.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)

    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Bridge-returned error or internal exception, truncated to 1000 chars.
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Number of consecutive /agent/health 200s observed. ≥3 = success.
    health_checks_passed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Bridge-reported duration for the swap (pull + recreate + health).
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    rollout: Mapped[Rollout] = relationship("Rollout", back_populates="attempts")
