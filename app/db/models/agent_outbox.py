"""Agent-side notification outbox (Autopilot PR4 — AGENT_ONLY).

The durable half of agent→platform notification delivery. The credit
reporter's fail-open-drop semantics are fine for metering but wrong
for notifications ("mission needs your approval" must not vanish on a
2s network blip), so producers write HERE first — in the tenant DB,
surviving container restarts and blue-green swaps — and a flush loop
POSTs rows to the platform's /api/agent/notify, which dedupes on
(user_id, idempotency_key=row.id). A row is done only when the
platform has acked queued/duplicate.

AGENT_ONLY: created by init_db create_all on agent boot (new tables
need no alembic mirror — the mirror rule is for COLUMNS on existing
tables).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class AgentNotifyOutbox(Base):
    __tablename__ = "agent_notify_outbox"

    # The row id doubles as the platform-side idempotency key — a
    # replayed flush of the same row is a no-op there.
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )
    priority: Mapped[str] = mapped_column(String(12), nullable=False, default="default")
    dedup_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    # NULL = not yet acked by the platform. Set on ack — including the
    # permanent-rejection and expiry cases, where last_error says why
    # (a poison row must not be retried forever).
    flushed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Backoff cursor — the flush loop skips rows until this passes.
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_agent_notify_outbox_flush", "flushed_at", "next_attempt_at"),
    )
