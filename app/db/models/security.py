"""Per-tenant security audit log.

`AgentSecurityEvent` records destructive / security-sensitive actions
against a user's tenant: agent_api_key rotation (T0b), and — once the
connector work lands — connector quarantine, force-disconnect, and any
future security primitive.

Append-only. Reads are: "show last rotated timestamp on /agent/settings"
(latest row of type `agent_key_rotated`), and "show recent security
activity" (last N rows for the user). Both indexed by (user_id,
occurred_at desc).

Why a new table instead of reusing `deletion_audit_events`:
deletion is one-shot per user; this table grows over the user's
lifetime. Different retention, different schema (no PII to redact,
freer-form metadata). Keeping them separate means each table's
constraints stay tight to its purpose.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# Event types (kept as string constants rather than an enum so future
# values can be added without migrations — append-only matters here).
EVENT_AGENT_KEY_ROTATED = "agent_key_rotated"
EVENT_AGENT_KEY_ROTATION_FAILED = "agent_key_rotation_failed"
# Future:
# EVENT_CONNECTOR_QUARANTINED = "connector_quarantined"
# EVENT_CONNECTOR_FORCE_DISCONNECTED = "connector_force_disconnected"


class AgentSecurityEvent(Base):
    """One row per security-sensitive action against a tenant.

    `metadata_json` is a free-form Text column holding a JSON-encoded
    payload (we use Text rather than JSONB so SQLite tests can still
    seed rows; production runs Postgres which handles JSON in Text just
    as well at this volume). Schema for the payload is per-event-type
    and documented at the call site.
    """

    __tablename__ = "agent_security_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    # JSON-encoded metadata. Text not JSONB — see class docstring.
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_agent_security_events_user_occurred", "user_id", "occurred_at"),
        Index("ix_agent_security_events_event_type", "event_type"),
    )
