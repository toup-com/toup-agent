"""Autopilot approvals — the durable ask-the-user store (PR7, AGENT_ONLY).

The investigation found no reusable approval primitive: the Toup Code
PermissionBroker is an in-memory Future that auto-denies after 600s
and dies with the WebSocket, and the two older approval frameworks are
dead code. A sleeping user needs an approval that survives WS
disconnects, agent restarts, and blue-green swaps — a table.

One row per question/approval a mission raises. The mission pauses
(``waiting_input``/``waiting_approval`` in its Routine state) until a
decision arrives via ``POST /api/autopilot/approvals/{id}/decision``
(platform-proxied from Mission Control or a push notification's deep
link), which also resumes the mission's tick loop.

AGENT_ONLY: created by init_db create_all on agent boot; the platform
reads it only through the agent's HTTP API (split-DB rule).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Index, JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# kinds
APPROVAL_KIND_QUESTION = "question"    # agent needs information
APPROVAL_KIND_APPROVAL = "approval"    # agent proposes a gated action

# statuses
APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_DENIED = "denied"
APPROVAL_ANSWERED = "answered"
APPROVAL_CANCELLED = "cancelled"

TERMINAL_APPROVAL_STATUSES = frozenset({
    APPROVAL_APPROVED, APPROVAL_DENIED, APPROVAL_ANSWERED, APPROVAL_CANCELLED,
})


class AutopilotApproval(Base):
    __tablename__ = "autopilot_approvals"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # The mission's Routine.id. Plain column (no FK) — approvals must
    # survive mission deletion for audit, and Routine rows have no
    # ondelete story on agent DBs.
    mission_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # What the user sees (push notification body / Mission Control card).
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    # Structured context: proposed action, tool payload, options, the
    # tick that raised it.
    payload_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=APPROVAL_PENDING, index=True,
    )
    # The user's free-text answer (kind=question) or decision note.
    answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # 'mission_control' | 'push_action' | 'chat' — where the decision came from.
    decided_via: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )

    __table_args__ = (
        Index("ix_autopilot_approvals_mission_status", "mission_id", "status"),
    )
