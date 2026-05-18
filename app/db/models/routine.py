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

    # Schedule shape (mig 042). NULLABLE in DB on purpose — the migration
    # that adds these columns is metadata-only (no NOT NULL, no
    # server_default) to avoid AccessExclusiveLock conflicts during
    # blue-green deploys. Application code treats NULL `schedule_kind`
    # as 'cron' (the legacy default), so rows that pre-date the
    # migration keep working unchanged.
    #
    # `default="cron"` is the Python-side default applied to NEW rows
    # inserted via the ORM; existing rows stay NULL in the DB. Reading
    # code uses `(routine.schedule_kind or "cron")` everywhere.
    schedule_kind: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True, default="cron"
    )
    # One-shot fire time (UTC). Required when schedule_kind='at'.
    schedule_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Interval between fires in seconds. Required when schedule_kind='every'.
    # Min 60 enforced at the API layer (Pydantic model_validator).
    schedule_interval_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Optional daily active-window for interval reminders. HH:MM:SS strings
    # in User.timezone. NULL = always active.
    schedule_window_start_local: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    schedule_window_end_local: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    # When true, the runner flips enabled=false after a successful fire.
    # API sets it to true automatically for schedule_kind='at'. Python
    # default False so writes via the ORM always have a concrete value.
    auto_disable_after_fire: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True, default=False
    )
    # For kind='reminder': literal text to deliver. No LLM, no MCP.
    # Required when kind='reminder' (enforced at the API layer).
    reminder_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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

    # Phase 2 / Ticket 2.1: APScheduler's scheduled trigger fire time
    # (UTC). Distinct from started_at = datetime.utcnow() at the head of
    # _fire — a slow-handler run can have finished_at - started_at ≈
    # handler_latency, but `last_run_at` (= fire_instant after Ticket 2.3)
    # is the user-visible "when the routine fired" instant. Nullable for
    # backward compat with pre-migration rows.
    fire_instant: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # finished_at rendered in the user's tz, ISO8601 — render-time helper
    # for dashboards so they don't have to join through User.timezone.
    finished_local_at: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    # "running" | "success" | "partial" | "failed" | "skipped_reauth"
    # Legacy column. Kept for backward compat with `/api/routines/{id}/runs`
    # consumers, Mission Control dashboards, and any operator query that
    # filters on it. New code should branch on `outcome` instead — see
    # Ticket 2.1 in docs/routines/.
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="running", server_default="running"
    )
    # New (Ticket 2.1) — richer state machine:
    #   "success"        — full success, handler ran AND all configured
    #                      channels confirmed delivery.
    #   "success_empty"  — handler ran cleanly, nothing to deliver
    #                      (empty Gmail inbox / no new emails since
    #                      watermark). Distinct from "success" so the
    #                      dashboard can show "delivered" vs "ran".
    #   "partial"        — message landed on at least one channel but
    #                      another channel skipped silently (most common
    #                      Defect-A symptom).
    #   "tool_error"     — Gmail returned reauth_required / scope_missing
    #                      / tool_missing. User-actionable: needs to
    #                      reconnect. Distinct from "failure" so the
    #                      nudge / reconnect notice flow can branch on it.
    #   "failure"        — handler exhausted its retry budget on a
    #                      retryable error (rate_limited / provider_down /
    #                      handler exception). Not user-actionable;
    #                      operator gets the alert.
    outcome: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    error_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Structured error blob — superset of error_class+error_detail. Carries
    # `class`, `detail`, `attempt`, optional `tool_name`, optional `tool_call_id`.
    error_json: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )
    # Per-channel delivery results. Shape:
    #   {
    #     "website": {"status": "delivered", "message_id": "...",
    #                 "ws_count": 2, "day_chat_id": "..."},
    #     "telegram": {"status": "delivered", "telegram_message_id": 12345,
    #                  "chat_id": 987654321},
    #     "whatsapp": {"status": "skipped", "reason": "no_self_e164"},
    #   }
    # Empty/null on tool_error / failure outcomes (nothing was sent).
    channel_results_json: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )
    # List of MCP tool names the handler invoked during this run. Helpful
    # for "did this routine actually call Gmail?" forensics on a failure.
    tools_invoked_json: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )

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

    # Soft pointer to the mirrored ``build_jobs`` row created at fire
    # time (PR 4b of the unified-jobs arc). No FK — same precedent as
    # ``cron_jobs.migrated_to_routine_id`` and ``trigger_events.job_id``:
    # the linkage survives a future Phase-D drop of this table without
    # CASCADE firing on the Job side. Populated by ``RoutineRunner._fire``
    # immediately after the idempotency claim; consumed by the runner's
    # status-sync path (``_post_terminal``, ``_finalize_run``) so the
    # Job row mirrors the legacy row's terminal state.
    job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "routine_id", "scheduled_for_local_date",
            name="uq_routine_runs_routine_date",
        ),
        Index("ix_routine_runs_user_started", "user_id", "started_at"),
        Index("ix_routine_runs_status_started", "status", "started_at"),
        Index("ix_routine_runs_outcome_started", "outcome", "started_at"),
    )


class RoutineNotificationDedupe(Base):
    """Per-routine, per-kind, per-local-day rate-limit row for the
    reconnect / failure notice path (Ticket 2.2).

    The UNIQUE (routine_id, kind, scope_date) constraint is the gate:
    `INSERT … ON UNIQUE → IntegrityError` means "already sent today,
    don't send again." Cron / dashboards don't read this directly — it's
    a write-once dedupe table consumed exclusively by `_write_nudge`.
    """

    __tablename__ = "routine_notification_dedupe"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    routine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("routines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Notice kind: "reauth" | "failure" | future kinds. Keep loose so a
    # new notice type can be added without a migration.
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    # Local calendar day the notice is scoped to. Same semantic as
    # `routine_runs.scheduled_for_local_date` — the user's day.
    scope_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "routine_id", "kind", "scope_date",
            name="uq_routine_notification_dedupe_key",
        ),
    )
