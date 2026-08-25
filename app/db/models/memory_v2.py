"""Memory architecture v2 — one platform memory, scoped views (R30).

CONTRACTS-R30 §4.5. Three ideas:

  - **Facts** are what the agent believes about the user — curated,
    categorised (the five canvas keys), scoped (`global` or one
    automation's id), optionally about an entity, each carrying the
    evidence sentence (`why`) that taught it.
  - **Episodes** are what happened in the jobs — written by the ENGINE
    at ledger close (one per run outcome, one per item that needed the
    user, one per write), linked to the exact thread turn. Never
    curator-written, never individually forgettable (they ride the
    run's 30-day archive).
  - **Entities** are the index over both, so "what did Marcus want?"
    resolves person → facts → episodes → turn ids to open.

These tables are the UI/API/recall source of truth. The brain
(`memory_files`) stays the agent-prompt projection with
`memory_curator` its ONLY writer — v2 fact writes project through the
sanctioned `instruct_file` seam exactly as the R29 facts ledger did
(`test_curator_producers._ALLOWED_WRITE_SITES` is unchanged by R30).
The engine `{{memory.<key>}}` state row stays separate, internal, and
out of every UI (D-07).

All AGENT_ONLY (tenant DB); listed in AGENT_ONLY_TABLES. The dormant
extraction graph (`entities`) is deliberately NOT reused —
`memory_entities` is a curated register; `extraction_entity_id` is an
optional link if the graph ever revives.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# The five canvas categories (§3.10) — fixed; the serializer refuses
# anything else. Order is render order.
MEMORY_V2_CATEGORIES = (
    "people", "team_workspace", "your_time", "work_you_own",
    "noise_filters",
)
MEMORY_V2_CATEGORY_LABELS = {
    "people": "PEOPLE",
    "team_workspace": "TEAM & WORKSPACE",
    "your_time": "YOUR TIME",
    "work_you_own": "WORK YOU OWN",
    "noise_filters": "NOISE IT FILTERS",
}
MEMORY_V2_CATEGORY_TONES = {
    "people": "blue",
    "team_workspace": "violet",
    "your_time": "warning",
    "work_you_own": "success",
    "noise_filters": "neutral",
}

MEMORY_V2_SOURCES = frozenset({"reaction", "told", "agent"})
MEMORY_V2_SCOPE_GLOBAL = "global"

MEMORY_ENTITY_KINDS = frozenset({
    "person", "channel", "ticket", "repo", "project", "account",
})

# Forget signals suppress relearning for this long (§4.5).
MEMORY_FORGET_SUPPRESS_DAYS = 30


class MemoryFact(Base):
    """One curated fact. `scope` is `"global"` or an automation id —
    an automation's "What it remembers" is `WHERE scope = <id>`; the
    global Memory screen reads everything."""

    __tablename__ = "memory_facts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    domain: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    scope: Mapped[str] = mapped_column(
        String(36), nullable=False, default=MEMORY_V2_SCOPE_GLOBAL,
        server_default=MEMORY_V2_SCOPE_GLOBAL,
    )
    subject_entity_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("memory_entities.id", ondelete="SET NULL"),
        nullable=True,
    )

    text: Mapped[str] = mapped_column(String(400), nullable=False)
    # The evidence sentence that taught it ("You replied within the hour
    # four times running.") — rendered under the statement.
    why: Mapped[Optional[str]] = mapped_column(String(400), nullable=True)

    learned_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.7, server_default="0.7",
    )
    last_confirmed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        Index("ix_memory_facts_user_scope", "user_id", "scope"),
        Index("ix_memory_facts_user_category", "user_id", "category"),
    )


class MemoryEpisode(Base):
    """One thing that happened — engine-written at ledger close."""

    __tablename__ = "memory_episodes"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    domain: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Soft pointers (build_jobs / threads have their own retention).
    automation_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, index=True,
    )
    run_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    turn_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    item_ref: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    text: Mapped[str] = mapped_column(String(400), nullable=False)
    outcome: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    subject_entity_ids_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]",
    )


class MemoryEntity(Base):
    """Curated entity register — people, channels, tickets, repos,
    projects, accounts the memory talks about."""

    __tablename__ = "memory_entities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Lower-cased key for the uniqueness constraint (SQLite has no
    # functional unique index through create_all; we store the norm).
    name_norm: Mapped[str] = mapped_column(String(200), nullable=False)
    aliases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # Optional link into the dormant extraction graph.
    extraction_entity_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "kind", "name_norm", name="uq_memory_entities_name",
        ),
    )


class MemoryForget(Base):
    """A forget signal — Forget removes the fact everywhere and writes
    one of these so the curator does not relearn it for 30 days."""

    __tablename__ = "memory_forgets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    scope: Mapped[str] = mapped_column(String(36), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    # Normalized text hash — the dedupe key the curator checks.
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    until: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )

    __table_args__ = (
        Index("ix_memory_forgets_user_hash", "user_id", "text_hash"),
    )
