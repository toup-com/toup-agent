"""
DayChat — the canonical conversation container for a calendar day.

A DayChat groups all sessions (Conversation rows) and messages across every
channel (web, telegram, vibecoding, voice, etc.) for a single user on a
single local calendar date. The agent loads the entire DayChat's history
when responding to any message on that day.

Key invariants:
  - Unique constraint on (user_id, local_date) — one DayChat per user per day.
  - `local_date` is the user's local calendar date, NOT UTC. The `timezone`
    field records which timezone was used at creation so historical days don't
    shift retroactively.
  - Sessions (Conversation rows) are sub-streams inside a DayChat. They exist
    for UI thread organization and channel tracking, but the agent's context
    scope is always the full DayChat.
  - A "New Thread" button in the UI creates a new Conversation inside the
    current DayChat — it does NOT reset agent context. The agent still sees
    every message from every session in the day.
"""

from datetime import datetime, date
from typing import Optional, List
import uuid

from sqlalchemy import String, Text, DateTime, Date, Integer, Boolean, ForeignKey, Index, UniqueConstraint, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base


class DayChat(Base):
    """Parent container for all conversations within a user's calendar day."""
    __tablename__ = "day_chats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)

    # The user's local calendar date (NOT UTC)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Timezone captured at creation — historical days don't shift
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_message_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Aggregated stats across all sessions in this day
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Rolling summary of older messages (generated async, never blocks a turn)
    rolling_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_up_to_message_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    summary_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Summary lifecycle: up_to_date | pending | failed | stale
    summary_status: Mapped[str] = mapped_column(String(20), default="up_to_date")

    # Archival summary — permanent, fact-dense index entry produced by the
    # end-of-day job. Distinct from rolling_summary (which compresses active
    # context). Returned by recall_day in summary-only mode.
    archival_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    archival_summary_generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # not_needed | pending | up_to_date | failed
    archival_summary_status: Mapped[str] = mapped_column(String(20), default="not_needed")

    # Relationships
    user: Mapped["User"] = relationship("User")
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="day_chat", order_by="Conversation.started_at"
    )

    __table_args__ = (
        UniqueConstraint("user_id", "local_date", name="uq_day_chat_user_date"),
        Index("ix_day_chats_user_date", "user_id", "local_date"),
    )


class ContextBudgetLog(Base):
    """Telemetry: token budget breakdown per LLM turn.

    Written on every turn before the LLM call. Enables monitoring of context
    usage patterns and summary staleness.

    Example admin queries (run against Postgres):

        -- Day chats where total tokens > 80% of model context window
        SELECT dc.id, dc.local_date, dc.total_tokens, cbl.model, cbl.total_tokens as turn_tokens
        FROM context_budget_logs cbl
        JOIN day_chats dc ON dc.id = cbl.day_chat_id
        WHERE cbl.total_tokens > 0.8 * CASE
            WHEN cbl.model LIKE 'claude%' THEN 200000
            WHEN cbl.model LIKE 'gpt-4o%' THEN 128000
            ELSE 128000
        END
        ORDER BY cbl.created_at DESC;

        -- Day chats where summary has been stale for >1 hour
        SELECT dc.id, dc.local_date, dc.user_id, dc.summary_status, dc.summary_updated_at
        FROM day_chats dc
        WHERE dc.summary_status IN ('stale', 'pending')
          AND (dc.summary_updated_at IS NULL OR dc.summary_updated_at < NOW() - INTERVAL '1 hour')
        ORDER BY dc.last_message_at DESC;
    """
    __tablename__ = "context_budget_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    day_chat_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("day_chats.id"), nullable=True, index=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    turn_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # message ID of user turn

    # Token breakdown
    system_tokens: Mapped[int] = mapped_column(Integer, default=0)
    summary_tokens: Mapped[int] = mapped_column(Integer, default=0)
    history_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tool_tokens: Mapped[int] = mapped_column(Integer, default=0)
    memory_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Model used for this turn
    model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Was the rolling summary stale when this turn loaded context?
    summary_was_stale: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_cbl_day_chat_created", "day_chat_id", "created_at"),
        Index("ix_cbl_user_created", "user_id", "created_at"),
    )


class MigrationStatus(Base):
    """Tracks one-time migration jobs that run at agent startup.

    Each migration is identified by name (e.g. 'day_chat_backfill').
    Status lifecycle: not_started → in_progress → completed | failed.

    On agent startup:
      - completed → skip (no-op)
      - not_started → fire background backfill task
      - in_progress → previous run crashed, resume from progress_json
      - failed → log loudly, do not auto-retry, surface alert

    Env var FORCE_DAY_CHAT_BACKFILL=true resets status to not_started
    and re-runs. Useful when timezone changes require re-bucketing.
    """
    __tablename__ = "migration_status"

    migration_name: Mapped[str] = mapped_column(String(100), primary_key=True)

    # not_started | in_progress | completed | failed
    status: Mapped[str] = mapped_column(String(20), default="not_started")

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Resumable state: {"last_conversation_id": "...", "processed": N, "total": M}
    progress_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
