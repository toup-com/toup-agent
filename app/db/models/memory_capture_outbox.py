"""Durable outbox for turns the memory curator could not write (v3 §2.1.6).

The curator runs fire-and-forget in turn post-processing, after the reply has
been streamed. That is the right place for it — it keeps the writer off the
user-visible path — but it also means nothing is watching when it fails. A
connection blip, a rate limit, a bad JSON reply, and everything the user
stated that turn is gone with no error surfaced to anyone.

**The payload is the TURN, not a set of facts.** Round 8 parked serialized
`MemoryCreate` rows here, which made the retry free — the extraction was
already paid for. v3 has no such intermediate: the curator reads the files as
they are NOW and decides what to change, so a replay a turn later must re-run
it against the state it will actually be applied to. Replaying an old op set
would rewrite bullets that have since moved. The retry therefore costs one
model call, and `REPLAY_PER_TURN` is 1 for that reason.

The table and its columns are UNCHANGED — only what `payload_json` holds
changes, from a list of facts to `{user_text, assistant_text, channel, ts}`.
Keeping the DDL still means no `_alter_statements` entry and no migration on
54 tenants for a table that holds, in practice, nothing.

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
    # The turn: {"user_text", "assistant_text", "channel", "ts"}. Retrying
    # re-runs the curator over it against the CURRENT files (see the module
    # docstring for why a stored op set would be wrong).
    payload_json: Mapped[Dict[str, Any]] = mapped_column(
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
