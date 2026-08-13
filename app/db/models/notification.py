"""Proactive-notification models (Autopilot arc, PR2 — platform-only).

Two tables, both PLATFORM_ONLY (see base.py):

- ``push_devices``: one row per mobile device that registered an Expo
  push token. Modeled on ``ExtensionDevice`` (agent.py) — soft-delete
  via ``revoked_at``, ``last_seen_at`` bumped on re-registration.
  Tokens live ONLY on the platform: agent containers must never hold
  push credentials (same isolation stance as connector tokens,
  base.py:137-141).

- ``notification_queue``: the durable send queue the platform
  dispatcher (PR3) claims rows from. Producers: the agent's notify
  outbox via ``POST /api/agent/notify`` (this PR) and, later,
  platform-side events. The queue is the idempotency point —
  UNIQUE(user_id, idempotency_key) — so the agent outbox can flush
  at-least-once and replays land as no-ops. Rows are CLAIMED with a
  status CAS (``UPDATE … WHERE status='queued'``) because platform-api
  runs 2 Railway replicas with no leader election (see
  docs/autopilot/PLAN.md D5 and scheduled_tasks.py:354-382 precedent).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    DateTime, ForeignKey, Index, Integer, JSON, String, Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# ── Event-kind vocabulary ─────────────────────────────────────────
# Closed set: the dispatcher routes on these (priority tier, channel
# choice, dedup window). Unknown kinds are rejected at ingest so a
# typo'd producer fails loudly instead of silently never delivering.
NOTIFY_KIND_NEEDS_INPUT = "needs_input"          # agent blocked on a question
NOTIFY_KIND_NEEDS_APPROVAL = "needs_approval"    # gated action awaits approve/deny
NOTIFY_KIND_MISSION_STARTED = "mission_started"  # drives the Live Activity start
NOTIFY_KIND_MISSION_COMPLETED = "mission_completed"
NOTIFY_KIND_MISSION_FAILED = "mission_failed"
NOTIFY_KIND_PROGRESS = "progress"                # Live Activity update only, never an OS alert push
NOTIFY_KIND_DIGEST = "digest"                    # batched, low priority
NOTIFY_KIND_GENERIC = "generic"                  # non-Autopilot producers
# Operator → user (admin dispatch): the only PLATFORM-authored alert
# kind. Every other kind describes something the user's own agent is
# doing, so the tenant ingest route (agent_notify) rejects this one.
NOTIFY_KIND_ANNOUNCEMENT = "announcement"

KNOWN_NOTIFY_KINDS: set[str] = {
    NOTIFY_KIND_NEEDS_INPUT,
    NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_MISSION_STARTED,
    NOTIFY_KIND_MISSION_COMPLETED,
    NOTIFY_KIND_MISSION_FAILED,
    NOTIFY_KIND_PROGRESS,
    NOTIFY_KIND_DIGEST,
    NOTIFY_KIND_GENERIC,
    NOTIFY_KIND_ANNOUNCEMENT,
}

# ── Queue statuses ────────────────────────────────────────────────
NQ_QUEUED = "queued"          # awaiting dispatch
NQ_SENDING = "sending"        # claimed by a dispatcher tick (CAS)
NQ_SENT = "sent"              # delivered to ≥1 channel
NQ_SUPPRESSED = "suppressed"  # prefs/dedup/cap said no — terminal, kept for audit
NQ_FAILED = "failed"          # exhausted attempts on every channel
NQ_EXPIRED = "expired"        # aged out before it could be sent

# ── Priority tiers (map to OS primitives in the dispatcher) ───────
NQ_PRIORITY_HIGH = "high"        # iOS time-sensitive / Android high channel
NQ_PRIORITY_DEFAULT = "default"
NQ_PRIORITY_LOW = "low"          # digest tier

KNOWN_NQ_PRIORITIES: set[str] = {
    NQ_PRIORITY_HIGH, NQ_PRIORITY_DEFAULT, NQ_PRIORITY_LOW,
}


class PushDevice(Base):
    """A mobile device registered for Expo push.

    Upsert key is ``expo_push_token`` (Expo tokens are stable per
    app-install and never expire): re-registration bumps
    ``last_seen_at`` and re-activates a previously revoked row; a
    token that moves to a different account is re-bound to the new
    user (last sign-in wins — the previous owner signed out of that
    device). Hard delivery failures (``DeviceNotRegistered`` receipts,
    PR3) set ``revoked_at`` so we stop harming Expo delivery
    reputation with dead tokens.
    """

    __tablename__ = "push_devices"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx] — ~40-60 chars today,
    # 200 leaves headroom for format evolution.
    expo_push_token: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True,
    )
    platform: Mapped[str] = mapped_column(String(16), nullable=False)  # ios | android
    device_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    app_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, index=True,
    )

    __table_args__ = (
        Index("ix_push_devices_user_revoked", "user_id", "revoked_at"),
    )


class NotificationQueue(Base):
    """One queued proactive notification.

    Lifecycle: ``queued`` → (CAS claim) ``sending`` → ``sent`` |
    ``suppressed`` | ``failed`` | ``expired``. ``scheduled_for`` is the
    quiet-hours deferral point: NULL means "as soon as possible"; the
    dispatcher's claim query only picks rows whose ``scheduled_for`` is
    NULL or in the past. A row stuck in ``sending`` past the stuck
    cutoff is re-claimed (dispatcher crash between claim and send —
    at-least-once by design, receipts dedup on the device).
    """

    __tablename__ = "notification_queue"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Producer: 'agent' (outbox flush) or 'platform' (future platform
    # events). Kept for audit/debug filtering, not routing.
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="agent")
    event_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Deep-link payload: {"route": "mission-control" | "job:<id>" |
    # "approval:<id>", "mission_id": ..., "approval_id": ...}. Scalars
    # only, capped at ingest — this rides inside the 4KB Expo payload.
    data_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )
    priority: Mapped[str] = mapped_column(
        String(12), nullable=False, default=NQ_PRIORITY_DEFAULT,
    )
    # Suppression key for the dispatcher's dedup window, e.g.
    # "<mission_id>:<event_kind>". NULL = never deduped.
    dedup_key: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True,
    )
    # Producer-supplied replay guard (the agent outbox row id).
    # UNIQUE(user_id, idempotency_key) makes ingest at-least-once safe.
    # Nullable: platform-side producers that enqueue inside their own
    # transaction don't need one (Postgres allows multiple NULLs).
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=NQ_QUEUED, index=True,
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Quiet-hours deferral: dispatch no earlier than this (UTC).
    scheduled_for: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Per-channel delivery results the dispatcher fills in:
    # {"expo": {"status": "ok", "ticket": ...}, "telegram": {...}}.
    channels_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "idempotency_key",
            name="uq_notification_queue_user_idem",
        ),
        # The dispatcher's claim scan: status + deferral window.
        Index("ix_notification_queue_claim", "status", "scheduled_for"),
        # Dedup-window lookups: recent rows for (user, dedup_key).
        Index(
            "ix_notification_queue_user_dedup",
            "user_id", "dedup_key", "created_at",
        ),
    )
