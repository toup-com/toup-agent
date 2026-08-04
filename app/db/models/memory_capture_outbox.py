"""Durable outbox for facts that were extracted but could not be stored.

Capture runs fire-and-forget in turn post-processing, after the reply has been
streamed. That is the right place for it — it keeps extraction off the
user-visible path — but it also means nothing is watching when it fails. A
connection blip while writing, and every fact the user stated that turn is gone
with no error surfaced to anyone and no way to notice afterwards.

The LLM call itself already retries once (A6-2), and an embedding failure
already degrades to an unembedded write (W1.5). This covers the third case: the
extraction SUCCEEDED and the write did not.

The payload is the extracted memories, not the raw turn, so a retry costs no
LLM call — the expensive half is already done and paid for.

Modelled on agent_notify_outbox: attempts, a backoff cursor, and a terminal
state so a poison row is not retried forever.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db.models.base import Base


class MemoryCaptureOutbox(Base):
    __tablename__ = "memory_capture_outbox"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    # The turn this came from, for provenance. Nullable because the message row
    # is not guaranteed to have been committed when capture ran.
    source_message_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # The already-extracted memories, serialized. Retrying replays THESE.
    payload_json: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    # NULL = still owed. Set when the facts land, or when the row is abandoned
    # after MAX_ATTEMPTS — last_error says which.
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_memory_capture_outbox_due", "resolved_at", "next_attempt_at"),
    )
