"""Conversation and Message models."""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base, Vector

# Channels whose rows must never render in — or be read back into — the
# main day chat. One tuple, four readers (CONTRACTS-R31 §4.1; the
# repo's "fixing one reader leaves the other doors open" class).
#
#   autopilot  — raw tick turns persisted before 2026-07-16; ticks are
#                headless now and mission outcomes arrive as 'routine'.
#   automation — an automation's OWN thread. The R28 session path wrote
#                those turns as ordinary Messages carrying a real
#                day_chat_id, which is how a thread question, its
#                answer and its "Memory updated" chip all appeared in
#                the founder's main chat on 26 August. The one
#                sanctioned automation row in the day is the
#                notification card, written on channel='routine'.
#
# Every day-scoped Message reader filters on this; the isolation guard
# test asserts the reader list has not grown a fifth member that forgot.
HIDDEN_DAY_CHANNELS: tuple = ("autopilot", "automation")


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
    # delete-orphan is load-bearing: deleting a Conversation has to take its
    # messages with it. Without it SQLAlchemy's default is to NULL out
    # messages.conversation_id, which is NOT NULL — so DELETE /api/sessions/{id}
    # raised IntegrityError and answered 500 for every session that had ever
    # carried a message. Empty sessions deleted fine, which is how a route
    # documented as "delete a session and all its messages" stayed broken.
    # Account deletion is unaffected: user_deletion.py removes messages with
    # raw SQL rather than through this relationship.
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )


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

    # WHO the message is FROM, as a party — not which surface produced it
    # (`channel`) and not which feature (`source`). NULL means the ordinary
    # case: the user and their agent talking. 'admin' means a human operator
    # wrote it and the agent must never see it.
    #
    # This is a SECURITY BOUNDARY, not a rendering hint: `load_day_context`
    # excludes 'admin' rows from the assembled LLM context, so an operator's
    # words can never become instructions the agent reads. A real column
    # rather than a `metadata_json` key precisely because it is a filter
    # predicate — a security check that has to reach into JSON is one that
    # eventually gets written differently in two places.
    #
    # 'agent' is reserved and deliberately unused today: an agent-initiated
    # proactive message belongs in context, so it must be able to ride this
    # same delivery path by writing a DIFFERENT origin and nothing else (see
    # docs/AGENT_INITIATED_CONTACT.md). That is why the exclusion filters on
    # the VALUE and never on "was this row written by the dispatch path".
    #
    # DB column added by `init_db()` ALTER (agent DBs have no alembic — see
    # database.py `_alter_statements`); the ORM declaration here is what lets
    # SQLAlchemy read and write it.
    origin: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)

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

    # Soft pointer to the message this row is a direct reply to. Set on
    # user-originated rows when the user picked a target via the "Reply"
    # affordance (web/mobile) or replied natively (Telegram, WhatsApp). The
    # referenced message can live in any session/channel on the same day —
    # threading spans the whole DayChat. No FK: stale targets stay
    # readable as deleted-message stubs in the UI rather than vanishing.
    reply_to_message_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )

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


class ProcessedMessage(Base):
    """Durable exactly-once ledger for inbound chat messages.

    Why this exists (the 2026-06 replay-loop credit burn):
    A brand-new user's FIRST message is sent with ``session_id=null`` (the
    agent creates the session on receipt), so it bypasses the session-gated
    DB replay check in ``ws_chat``. The in-process dedup guard there only
    survives within ONE process for a short TTL — but during the
    post-onboarding boot window the agent container is swapped (pool claim /
    blue-green upgrade), wiping that in-memory state. The client then
    replays its queued, un-acked first message on every reconnect/refresh,
    and each replay spawned a fresh session + a fresh LLM call + a fresh
    credit charge → an at-least-once loop the user watched burn credit.

    This table makes the first dispatch claim ``(user_id, client_msg_id)``
    ATOMICALLY before the agent runs: the primary key is a deterministic
    ``uuid5(NAMESPACE_OID, "toup-msg:{user_id}:{client_msg_id}")`` — the same
    derivation the session-gated path already uses for ``messages.id`` — so a
    replay collides on INSERT and is dropped *before* it can trigger a second
    LLM call or charge. Exactly-once survives container restarts and
    reconnects because the claim lives in the tenant DB, not memory.

    Agent-only table created by ``init_db()``'s ``create_all`` (agents boot
    via ``init_db``, NOT alembic), so recycled pool DBs self-provision it on
    their next boot. No FK on ``user_id`` — the ledger is a pure dedup marker
    and must never block on the parent row's existence/visibility.
    """
    __tablename__ = "processed_messages"

    # Deterministic uuid5(NAMESPACE_OID, "toup-msg:{user_id}:{client_msg_id}").
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    client_msg_id: Mapped[str] = mapped_column(String(100))
    # The session the first dispatch created/used, stamped opportunistically
    # once known. Nullable: the very first message claims the ledger before a
    # session exists.
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
