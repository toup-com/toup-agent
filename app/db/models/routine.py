"""Agent routines — system-managed scheduled actions (email briefing, etc.).

Kept separate from CronJob: CronJobs are user-authored prompts that post to
Telegram. Routines are system-defined automations that post into the
Day-as-Chat as `role=assistant`, `channel=routine`, `source=<kind>`.

The runner (`app.agent.routines.runner.RoutineRunner`) owns its own
APScheduler instance — distinct from CronService.scheduler — and dispatches
to per-kind handlers in `app.agent.routines.registry`.
"""

from datetime import datetime, date
from typing import Optional
import uuid

from sqlalchemy import (
    String, Text, DateTime, Date, Integer, Boolean, ForeignKey, Index,
    UniqueConstraint, JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Routine(Base):
    """One system-managed scheduled action for a user.

    A user may have multiple routines of the same kind (e.g., a weekday
    morning briefing and a weekend morning briefing). The DB does not
    enforce one-per-kind — that constraint lives in the API layer
    (`POST /api/routines` returns 409 if an enabled routine of the same
    kind already exists for v1).
    """

    __tablename__ = "routines"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # Routine kind discriminator.
    #   "agent_task"     — generic. Runs `prompt_text` against the agent at
    #                      the scheduled time. Output posted to day-chat.
    #   "email_briefing" — Gmail-specialised preset (MCP fetch + summary).
    # Adding a new kind = add a handler + register; no migrations.
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")

    # User-visible name, e.g. "Morning briefing" / "Check deploys". Used
    # as the Conversation title for posted messages + Mission Control
    # card title. NULL for legacy rows (pre-2026-05-12); the UI falls
    # back to the kind discriminator when empty.
    name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Natural-language prompt the agent runs at fire time. Only used by
    # `kind="agent_task"`. NULL for handler-specific kinds like
    # `email_briefing` whose flow is hard-coded in the handler.
    prompt_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 5-part cron expression evaluated in User.timezone. Stored as the user's
    # original choice so a later tz change can re-resolve without losing intent.
    schedule_cron_local: Mapped[str] = mapped_column(String(100), nullable=False)

    # Per-kind config blob. For email_briefing:
    #   {connector_identity_id, send_minutes_before_wake, priority_filters}
    config_json: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )

    # Per-kind watermark / cursor. For email_briefing:
    #   {gmail_history_id, last_processed_internal_date}
    last_state_json: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )

    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # "never_run" | "success" | "partial" | "failed" | "skipped_reauth"
    last_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="never_run", server_default="never_run"
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_routines_user_kind_enabled", "user_id", "kind", "enabled"),
    )


class RoutineRun(Base):
    """One execution attempt of a routine for a specific user-local calendar day.

    `UNIQUE (routine_id, scheduled_for_local_date)` is the idempotency gate.
    The runner does an `INSERT ... ON CONFLICT DO NOTHING` keyed on this pair
    before any work; a conflict means another fire already claimed this day
    and we exit silently. Calendar day is always in `User.timezone`, never
    UTC, so a user east of UTC cannot get two briefings near midnight.

    `summary_message_id` uses `ON DELETE SET NULL`: deleting a Message must
    not nuke historical run rows. The summary link is a soft pointer.
    """

    __tablename__ = "routine_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    routine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("routines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalized so Mission Control's per-user scan doesn't have to join.
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # Calendar date in the user's tz. Distinct from started_at — a run that
    # fires at 06:55 local for the 2026-05-12 briefing carries
    # scheduled_for_local_date=2026-05-12 regardless of UTC clock skew.
    scheduled_for_local_date: Mapped[date] = mapped_column(Date, nullable=False)

    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # "running" | "success" | "partial" | "failed" | "skipped_reauth"
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="running", server_default="running"
    )
    error_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    emails_fetched: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    attempt: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )

    # Message.id is String(50), not 36 — FK type matches the target column.
    summary_message_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "routine_id", "scheduled_for_local_date",
            name="uq_routine_runs_routine_date",
        ),
        Index("ix_routine_runs_user_started", "user_id", "started_at"),
        Index("ix_routine_runs_status_started", "status", "started_at"),
    )
