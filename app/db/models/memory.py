"""Memory model and related audit/stats models."""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, Boolean,
    ForeignKey, Table, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR

from .base import Base, Vector


# Association table for memory relationships (many-to-many)
memory_relationships = Table(
    "memory_relationships",
    Base.metadata,
    Column("source_id", String(36), ForeignKey("memories.id"), primary_key=True),
    Column("target_id", String(36), ForeignKey("memories.id"), primary_key=True),
    Column("relationship_type", String(50)),
    Column("strength", Float, default=1.0),
    Column("created_at", DateTime, default=datetime.utcnow),
)


class Memory(Base):
    """
    Core memory unit - represents a piece of information stored in the brain.
    Combines structured data with vector embedding for hybrid retrieval.
    Supports multiple brain types: User, Agent, Work
    """
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)

    # Brain type - which brain this memory belongs to
    brain_type: Mapped[str] = mapped_column(String(20), default="user", index=True)

    # Content
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Classification
    category: Mapped[str] = mapped_column(String(20), index=True)
    memory_type: Mapped[str] = mapped_column(String(20), index=True)

    # The memory file this row belongs to (app.memory_files). Every ACTIVE
    # row gets one when the tenant's organize pass runs; NULL means the row
    # predates that pass and readers fall back to the category→section map.
    # A slug, not an FK — it survives file-row recreation and lets the
    # platform's read-fallback group rows without the memory_files table
    # (AGENT_ONLY, deliberately no platform mirror).
    file_slug: Mapped[Optional[str]] = mapped_column(String(160), nullable=True, index=True)
    # Curated order within the file. Small integers, renumbered on write —
    # a file holds tens of entries, not thousands.
    file_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Embedding for semantic search (stored as JSON for SQLite)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Native pgvector column
    embedding = Column(Vector(), nullable=True) if Vector else None

    # Full-text search vector (auto-updated by PostgreSQL trigger).
    # `with_variant(TSVECTOR(), "postgresql")` keeps the prod schema as
    # tsvector while letting SQLite test runs compile the column as Text.
    # The trigger that maintains it is Postgres-only — on sqlite the
    # column is just inert ballast.
    search_vector = Column(Text().with_variant(TSVECTOR(), "postgresql"), nullable=True)

    # Importance and confidence
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    # Memory Decay & Reinforcement System
    strength: Mapped[float] = mapped_column(Float, default=1.0)
    memory_level: Mapped[str] = mapped_column(String(20), default="episodic", index=True)
    emotional_salience: Mapped[float] = mapped_column(Float, default=0.5)
    last_reinforced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    consolidation_count: Mapped[int] = mapped_column(Integer, default=0)
    decay_rate: Mapped[float] = mapped_column(Float, default=0.1)

    # The instant a decay pass last WROTE `strength`, and therefore the point
    # the stored value is accurate as of. DecayService multiplies the stored
    # strength by e^(-elapsed/stability); without this stamp every pass
    # re-applies the WHOLE elapsed exponent to an already-decayed number, so
    # the total decay grows with how often the job ran rather than with how
    # much time passed. NULL = never decayed, which is correct for every row
    # written before 2026-08-06 — those fall back to the reinforcement /
    # creation reference and receive exactly one catch-up curve.
    last_decayed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Temporal validity. NULL = this memory never expires, which is correct
    # for durable facts and for every row written before 2026-07-29. Set only
    # when the extractor flags a memory transient AND its category permits
    # expiry (app.memory_taxonomy.resolve_ttl_days) — reminders, one-off
    # errands, passing status. The expiry sweep ARCHIVES (is_active=False)
    # rather than deleting, so nothing is ever lost.
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)

    # Temporal data
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    access_count: Mapped[int] = mapped_column(Integer, default=0)

    # Source tracking
    source_message_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("messages.id"), nullable=True)
    source_type: Mapped[str] = mapped_column(String(50), default="conversation")

    # Explicit linkage to platform entities (Ticket 2, 2026-05-13). When
    # a memory is "about" a specific routine/trigger/connector, set
    # ref_kind + ref_id so subsequent calls about the same entity
    # UPSERT instead of duplicating. Backed by a partial unique index
    # on (user_id, ref_kind, ref_id) installed by init_db.
    ref_kind: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    ref_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Metadata (flexible JSON for additional attributes)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Memory Evolution System
    canonical_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    history_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    merged_from_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    superseded_by: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="memories")
    source_message: Mapped[Optional["Message"]] = relationship("Message", back_populates="extracted_memories")
    entity_links: Mapped[List["EntityLink"]] = relationship("EntityLink", back_populates="memory")

    # Self-referential many-to-many for memory connections
    related_memories: Mapped[List["Memory"]] = relationship(
        "Memory",
        secondary=memory_relationships,
        primaryjoin=id == memory_relationships.c.source_id,
        secondaryjoin=id == memory_relationships.c.target_id,
        backref="referenced_by"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("ix_memories_user_brain", "user_id", "brain_type"),
        Index("ix_memories_user_brain_category", "user_id", "brain_type", "category"),
        Index("ix_memories_user_category", "user_id", "category"),
        Index("ix_memories_user_type", "user_id", "memory_type"),
        Index("ix_memories_user_created", "user_id", "created_at"),
    )


class MemoryFile(Base):
    """A curated memory file — the organizing unit above memory rows.

    One row per (user, slug). System files (profile, preferences, knowledge,
    learned, working, plus the people/areas catch-alls) always exist after
    the organize pass; person and topic files are created on demand. Entries
    are Memory rows carrying this slug in `memories.file_slug` — the file
    row itself holds only identity and curation metadata, so entry counts
    and freshness are always computed live and can never lie.

    AGENT_ONLY with no platform mirror: the platform's read fallback
    synthesizes a virtual file view from memories rows alone.
    """
    __tablename__ = "memory_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)

    slug: Mapped[str] = mapped_column(String(160), index=True)
    section: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(200))
    # The one-line "read when…" header. Shown on the file page and injected
    # into the prompt's memory-file index.
    purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Cross-links: JSON array of sibling file slugs ("see also").
    related_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Order within the section (system files first, then by creation).
    position: Mapped[int] = mapped_column(Integer, default=0)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)

    # When the LLM consolidation pass last curated this file. NULL = only the
    # deterministic organize pass has touched it (entries assigned, nothing
    # merged). Per-file so a failed pass retries file by file.
    consolidated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_memory_files_user_slug", "user_id", "slug", unique=True),
        Index("ix_memory_files_user_section", "user_id", "section"),
    )


class BrainStats(Base):
    """Cached statistics for brain visualization"""
    __tablename__ = "brain_stats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), unique=True, index=True)

    # Stats by region (JSON object)
    region_counts_json: Mapped[str] = mapped_column(Text, default="{}")
    region_sizes_json: Mapped[str] = mapped_column(Text, default="{}")

    # Overall stats
    total_memories: Mapped[int] = mapped_column(Integer, default=0)
    total_entities: Mapped[int] = mapped_column(Integer, default=0)
    total_connections: Mapped[int] = mapped_column(Integer, default=0)

    # Last update
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MemoryEvent(Base):
    """
    Immutable audit log of all memory operations.
    Every operation on a memory is logged here and CANNOT be modified or deleted.
    """
    __tablename__ = "memory_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    memory_id: Mapped[str] = mapped_column(String(36), ForeignKey("memories.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)

    # Event details
    event_type: Mapped[str] = mapped_column(String(20), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Event-specific data (JSON)
    event_data_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trigger_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("ix_memory_events_memory_time", "memory_id", "timestamp"),
        Index("ix_memory_events_user_time", "user_id", "timestamp"),
        Index("ix_memory_events_type_time", "event_type", "timestamp"),
    )


class RetrievalEvent(Base):
    """Tracks retrieval quality for the self-improvement feedback loop."""
    __tablename__ = "retrieval_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    strategies_used_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Memory IDs (all JSON arrays)
    retrieved_memory_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    used_memory_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    irrelevant_memory_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metrics
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    has_extraction_gap: Mapped[bool] = mapped_column(Boolean, default=False)
    gap_topics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_correction: Mapped[bool] = mapped_column(Boolean, default=False)
    correction_data_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quality_signal: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    retrieval_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_retrieval_events_user_created", "user_id", "created_at"),
        Index("ix_retrieval_events_quality", "user_id", "quality_signal"),
        Index("ix_retrieval_events_gaps", "user_id", "has_extraction_gap"),
    )
