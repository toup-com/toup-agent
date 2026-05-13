"""Conversation and Message models."""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base, Vector


class Conversation(Base):
    """Conversation/Session record for tracking message history.

    A session is a sub-stream inside a DayChat. It tracks one continuous UI
    thread on one channel. The agent's context scope is the parent DayChat
    (all sessions for the day), not individual sessions.

    Reading-A invariant (decided 2026-05-13, see
    docs/bug-sweep-2026-05-13.md): for SYSTEM-DRIVEN channels —
    `routine`, `trigger`, `api`, `digest` — there is exactly ONE active
    Conversation per `(user_id, day_chat_id, channel)`. Every fire on
    the same day appends to the same row instead of spawning a new one.
    Enforced by partial unique index
    `ix_conversations_system_channel_per_day` and the canonical resolver
    `app.agent.conversation_resolver.resolve_or_create_day_conversation`.

    USER-DRIVEN channels (`web`, `telegram`, `whatsapp`, `voice`,
    `extension`, `app`) keep the per-day-with-`New Thread`-override
    behavior and do NOT go through that resolver.
    """
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    title: Mapped[Optional[str]] = mapped_column(String(500))

    # Parent DayChat — LOOSE HINT, may be stale for long-lived Telegram sessions.
    # Telegram sessions span multiple days; this field points to the day the session
    # was created, not necessarily today. For determining which day a message belongs to,
    # ALWAYS use Message.day_chat_id (canonical) — never Conversation.day_chat_id.
    day_chat_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("day_chats.id"), nullable=True, index=True)

    # Channel tracking
    channel: Mapped[str] = mapped_column(String(50), default="api")  # api, telegram, discord, web

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Stats
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Builder mode (per-chat toggle: NULL/'auto' = App Builder, 'vibe' = vibecoding)
    builder_mode: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # Metadata
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON stored as text

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    day_chat: Mapped[Optional["DayChat"]] = relationship("DayChat", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="conversation", order_by="Message.created_at")


class Message(Base):
    """Individual message in a conversation"""
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"), index=True)

    # Denormalized from parent session for fast day-level loading
    day_chat_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("day_chats.id"), nullable=True, index=True)

    # Channel this specific message was sent from. Denormalized at write
    # time from the ingress handler (not from Conversation.channel) so the
    # agent can see per-message channel in history — a user who starts a
    # day on mobile and switches to web has the two channels visible in
    # chronological order, not blended under the conversation's channel.
    # Null on pre-migration rows; resolver treats null as
    # conversation.channel then "unknown". See channel_util.resolve_channel.
    channel: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)

    # Sub-channel discriminator for non-user-originated messages. NULL for
    # normal user/assistant turns. Set to a routine kind id (e.g.
    # "email_briefing") on routine-generated messages so the frontend can
    # render them as a distinct card and operators can filter the audit
    # trail. DB column added by `init_db()` ALTER; the ORM declaration
    # here lets SQLAlchemy actually read/write it.
    source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    role: Mapped[str] = mapped_column(String(20))  # "user", "assistant", "system", "job"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Token tracking (for cost analysis)
    tokens_prompt: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_completion: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Model tracking
    model_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Memory retrieval tracking (JSON array of memory IDs)
    memories_retrieved_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Processing metadata
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Rich content metadata (JSON): media cards, tool results, etc.
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Generated-file attachments (array of {id, filename, mime_type, size_bytes, storage_path, created_at}).
    # Populated by agent generate_* tools; rendered in the web two-pane DocumentSplit.
    # JSONB on Postgres (binary, indexable), JSON/TEXT on SQLite dev.
    # Pass Python lists/dicts directly — SQLAlchemy handles serialization.
    attachments: Mapped[Optional[list]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )

    # Embedding stored as JSON array (for SQLite compatibility)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Native pgvector column
    embedding = Column(Vector(), nullable=True) if Vector else None

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    extracted_memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="source_message")

    __table_args__ = (
        Index("ix_messages_day_chat_created", "day_chat_id", "created_at"),
    )
