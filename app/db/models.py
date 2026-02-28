"""
Database models for Hex Brain Memory System

This implements a hybrid storage system with:
- Vector embeddings for semantic search
- Relational structure for entities, relationships, and metadata
- Audit trail and timestamps
- Multi-brain support: User, Agent, Work
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, BigInteger, Boolean,
    ForeignKey, Table, Enum as SQLEnum, JSON, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # Fallback for environments without pgvector

Base = declarative_base()


class BrainType(str, Enum):
    """Types of brains in the HexBrain system"""
    USER = "user"       # User's personal memories
    AGENT = "agent"     # Hex agent's learned knowledge  
    WORK = "work"       # Workflows and operational processes


class MemoryCategory(str, Enum):
    """Memory categories for USER brain - organizing personal information"""
    # Personal Profile
    IDENTITY = "identity"           # Who the user is - name, age, background
    PREFERENCES = "preferences"     # Likes, dislikes, favorites
    BELIEFS = "beliefs"             # Values, opinions, worldview
    EMOTIONS = "emotions"           # Emotional states, moods, feelings
    
    # Relationships
    PEOPLE = "people"               # Friends, colleagues, contacts
    PLACES = "places"               # Locations, addresses, venues
    FAMILY = "family"               # Family members, relationships
    
    # Experiences
    EXPERIENCES = "experiences"     # Past events, memories, stories
    PROJECTS = "projects"           # Active or past projects
    SCHEDULE = "schedule"           # Appointments, reminders, calendar
    WORK = "work"                   # Job, career, professional
    
    # Knowledge
    LEARNING = "learning"           # Skills being learned, courses
    KNOWLEDGE = "knowledge"         # Facts, information, expertise
    TOOLS = "tools"                 # Software, tools, configurations
    MEDIA = "media"                 # Books, movies, music, articles
    
    # Lifestyle
    HEALTH = "health"               # Medical, fitness, wellness
    HABITS = "habits"               # Routines, habits, rituals
    FOOD = "food"                   # Diet, recipes, restaurants
    TRAVEL = "travel"               # Trips, destinations, travel plans
    GOALS = "goals"                 # Objectives, aspirations, dreams
    CONTEXT = "context"             # Conversation context, general


class AgentCategory(str, Enum):
    """Memory categories for AGENT brain - Hex's learned capabilities"""
    AGENT_TOOLS = "agent_tools"           # 🛠️ Commands, APIs, skills Hex knows
    AGENT_SKILLS = "agent_skills"         # 🎯 Learned capabilities and competencies
    AGENT_SOUL = "agent_soul"             # 💫 Personality, values, communication style
    AGENT_PROCEDURES = "agent_procedures" # 📋 How to do things, step-by-step guides
    AGENT_PATTERNS = "agent_patterns"     # 🧠 Learned behaviors, recognition patterns
    AGENT_DECISIONS = "agent_decisions"   # 📝 Past choices made, decision history


class WorkCategory(str, Enum):
    """Memory categories for WORK brain - workflows and operations"""
    WORKFLOW = "workflow"           # 🔄 Multi-step processes
    PROCESS = "process"             # ⚡ Business processes


# Alias for backward compatibility
BrainRegion = MemoryCategory


class MemoryType(str, Enum):
    """Types of memories that can be stored"""
    FACT = "fact"               # Factual information
    PREFERENCE = "preference"   # User preference/opinion
    TASK = "task"               # Task or todo
    EVENT = "event"             # Event or happening
    PERSON = "person"           # Person entity
    PLACE = "place"             # Location
    PROJECT = "project"         # Project or initiative
    DECISION = "decision"       # Decision made
    SKILL = "skill"             # Learned skill or procedure
    FILE = "file"               # Document or file reference
    NOTE = "note"               # General note
    CONVERSATION = "conversation"  # Conversation summary


class MemoryLevel(str, Enum):
    """
    Cognitive hierarchy of memory levels.
    Based on cognitive science: episodic → semantic consolidation.
    """
    EPISODIC = "episodic"       # Specific experiences with time/place context
    SEMANTIC = "semantic"       # General facts/knowledge (consolidated)
    PROCEDURAL = "procedural"   # How-to knowledge, skills, procedures
    META = "meta"               # Knowledge about knowledge (metacognition)


class MemoryEventType(str, Enum):
    """
    Types of events that can occur to a memory.
    Used for immutable audit log tracking.
    """
    CREATED = "created"           # Memory was created
    ACCESSED = "accessed"         # Memory was retrieved/read
    REINFORCED = "reinforced"     # Memory was strengthened (spaced repetition)
    DECAYED = "decayed"           # Memory strength decreased (forgetting curve)
    CONSOLIDATED = "consolidated" # Memory was consolidated (episodic→semantic)
    UPDATED = "updated"           # Memory content was modified
    DELETED = "deleted"           # Memory was soft-deleted
    LINKED = "linked"             # Memory was linked to another memory
    UNLINKED = "unlinked"         # Memory link was removed


# Association table for memory relationships (many-to-many)
memory_relationships = Table(
    "memory_relationships",
    Base.metadata,
    Column("source_id", String(36), ForeignKey("memories.id"), primary_key=True),
    Column("target_id", String(36), ForeignKey("memories.id"), primary_key=True),
    Column("relationship_type", String(50)),  # e.g., "related_to", "derived_from", "mentions"
    Column("strength", Float, default=1.0),
    Column("created_at", DateTime, default=datetime.utcnow),
)


class UserRole(str, Enum):
    """User roles for access control"""
    ADMIN = "admin"          # Full platform access + user management
    BETA_USER = "beta_user"  # Standard closed-beta access


class User(Base):
    """User model for multi-user isolation"""
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    password_plain: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Admin-visible plaintext (closed beta only)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="beta_user", index=True)  # admin | beta_user
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="user")
    conversations: Mapped[List["Conversation"]] = relationship("Conversation", back_populates="user")
    identities: Mapped[List["Identity"]] = relationship("Identity", back_populates="user")


class IdentityType(str, Enum):
    """Types of identity documents"""
    SOUL = "soul"                     # Agent personality, tone, values
    USER_PROFILE = "user_profile"     # Info about the human user
    AGENT_INSTRUCTIONS = "agent_instructions"  # Behavioral rules
    TOOLS = "tools"                   # Available tools/skills documentation
    SYSTEM = "system"                 # System-level instructions
    CONTEXT = "context"               # Dynamic runtime context


class Identity(Base):
    """
    Identity documents that define the agent's personality and behavior.
    Equivalent to SOUL.md, USER.md, AGENTS.md in other systems.
    """
    __tablename__ = "identities"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    
    # Identity type and name
    identity_type: Mapped[str] = mapped_column(String(50), index=True)  # IdentityType enum value
    name: Mapped[str] = mapped_column(String(255))  # Human-readable name
    
    # Content
    content: Mapped[str] = mapped_column(Text)  # The actual identity document
    
    # Priority (higher = loaded first in prompt)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="identities")
    
    # Indexes
    __table_args__ = (
        Index("ix_identities_user_type", "user_id", "identity_type"),
    )


class Conversation(Base):
    """Conversation/Session record for tracking message history"""
    __tablename__ = "conversations"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    
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
    
    # Metadata
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON stored as text
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="conversation", order_by="Message.created_at")


class Message(Base):
    """Individual message in a conversation"""
    __tablename__ = "messages"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user", "assistant", "system"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Token tracking (for cost analysis)
    tokens_prompt: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Tokens in prompt
    tokens_completion: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Tokens in response
    
    # Model tracking
    model_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # e.g., "gpt-4o"
    
    # Memory retrieval tracking (JSON array of memory IDs)
    memories_retrieved_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Processing metadata
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Embedding stored as JSON array (for SQLite compatibility)
    # For PostgreSQL with pgvector, this would be: Vector(384)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Native pgvector column
    embedding = Column(Vector(1536), nullable=True) if Vector else None
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    extracted_memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="source_message")


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
    brain_type: Mapped[str] = mapped_column(String(20), default="user", index=True)  # user, agent, work
    
    # Content
    content: Mapped[str] = mapped_column(Text)  # Main memory content
    summary: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Short summary
    
    # Classification
    category: Mapped[str] = mapped_column(String(20), index=True)     # Category depends on brain_type
    memory_type: Mapped[str] = mapped_column(String(20), index=True)   # MemoryType enum value
    
    # Embedding for semantic search (stored as JSON for SQLite)
    # For PostgreSQL: embedding = Column(Vector(384))
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Native pgvector column — used for all vector operations (100-1000x faster than JSON)
    # DEPRECATED: embedding_json is kept for backward compat, will be removed in future
    embedding = Column(Vector(1536), nullable=True) if Vector else None
    
    # Full-text search vector (auto-updated by PostgreSQL trigger)
    search_vector = Column(TSVECTOR, nullable=True)
    
    # Importance and confidence
    importance: Mapped[float] = mapped_column(Float, default=0.5)  # 0-1 scale
    confidence: Mapped[float] = mapped_column(Float, default=1.0)  # How confident we are in this memory
    
    # === NEW: Memory Decay & Reinforcement System ===
    # Based on Ebbinghaus forgetting curve and spaced repetition research
    strength: Mapped[float] = mapped_column(Float, default=1.0)  # Memory strength (0-1), decays over time
    memory_level: Mapped[str] = mapped_column(String(20), default="episodic", index=True)  # MemoryLevel enum
    emotional_salience: Mapped[float] = mapped_column(Float, default=0.5)  # Emotional weight (0-1)
    last_reinforced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # Last strengthening
    consolidation_count: Mapped[int] = mapped_column(Integer, default=0)  # Times consolidated
    decay_rate: Mapped[float] = mapped_column(Float, default=0.1)  # Individual decay rate modifier
    
    # Temporal data
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Source tracking
    source_message_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("messages.id"), nullable=True)
    source_type: Mapped[str] = mapped_column(String(50), default="conversation")  # conversation, import, manual
    
    # Metadata (flexible JSON for additional attributes)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Tags for quick filtering
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of tags
    
    # === Memory Evolution System ===
    # For tracking how memories evolve and merge over time
    canonical_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Current truth (most up-to-date version)
    history_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of version history
    merged_from_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of merged memory IDs
    superseded_by: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)  # ID of memory this was merged into
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)  # False if superseded/archived
    
    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="memories")
    source_message: Mapped[Optional["Message"]] = relationship("Message", back_populates="extracted_memories")
    
    # Entity links
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


class Entity(Base):
    """
    Named entities extracted from conversations.
    Examples: people, places, organizations, projects
    """
    __tablename__ = "entities"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    
    # Entity details
    name: Mapped[str] = mapped_column(String(255), index=True)
    entity_type: Mapped[str] = mapped_column(String(50), index=True)  # person, place, org, project, etc.
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Embedding for the entity
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Native pgvector column
    embedding = Column(Vector(1536), nullable=True) if Vector else None
    
    # Schema-enforced extraction type (PersonEntity, OrganizationEntity, etc.)
    schema_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    
    # Metadata
    attributes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Flexible attributes
    
    # Full-text search on name (auto-maintained by PostgreSQL GENERATED ALWAYS AS)
    name_search = Column(TSVECTOR, nullable=True)
    
    # Stats
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Links to memories
    memory_links: Mapped[List["EntityLink"]] = relationship("EntityLink", back_populates="entity")
    
    __table_args__ = (
        Index("ix_entities_user_name", "user_id", "name"),
        Index("ix_entities_user_type", "user_id", "entity_type"),
    )


class EntityLink(Base):
    """Links between entities and memories"""
    __tablename__ = "entity_links"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    memory_id: Mapped[str] = mapped_column(String(36), ForeignKey("memories.id"), index=True)
    entity_id: Mapped[str] = mapped_column(String(36), ForeignKey("entities.id"), index=True)
    
    # How is the entity related to the memory
    role: Mapped[str] = mapped_column(String(50), default="mentioned")  # mentioned, subject, object, etc.
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    memory: Mapped["Memory"] = relationship("Memory", back_populates="entity_links")
    entity: Mapped["Entity"] = relationship("Entity", back_populates="memory_links")


class EntityRelationship(Base):
    """
    Direct entity-to-entity relationships for the knowledge graph.
    Replaces the hack of storing relationships as Memory records.
    Enables efficient graph traversal between entities.
    """
    __tablename__ = "entity_relationships"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    source_entity_id: Mapped[str] = mapped_column(String(36), ForeignKey("entities.id"), nullable=False, index=True)
    target_entity_id: Mapped[str] = mapped_column(String(36), ForeignKey("entities.id"), nullable=False, index=True)
    relationship_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    relationship_label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Human-readable: "Alice works at Google"
    properties_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_entity: Mapped["Entity"] = relationship("Entity", foreign_keys=[source_entity_id], backref="outgoing_relationships")
    target_entity: Mapped["Entity"] = relationship("Entity", foreign_keys=[target_entity_id], backref="incoming_relationships")
    
    __table_args__ = (
        Index("ix_entity_rel_src_tgt", "source_entity_id", "target_entity_id"),
    )


class BrainStats(Base):
    """Cached statistics for brain visualization"""
    __tablename__ = "brain_stats"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), unique=True, index=True)
    
    # Stats by region (JSON object)
    region_counts_json: Mapped[str] = mapped_column(Text, default="{}")
    region_sizes_json: Mapped[str] = mapped_column(Text, default="{}")  # Normalized sizes for visualization
    
    # Overall stats
    total_memories: Mapped[int] = mapped_column(Integer, default=0)
    total_entities: Mapped[int] = mapped_column(Integer, default=0)
    total_connections: Mapped[int] = mapped_column(Integer, default=0)
    
    # Last update
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MemoryEvent(Base):
    """
    Immutable audit log of all memory operations.
    
    This implements the "10-Year Memory Architecture" event sourcing pattern.
    Every operation on a memory is logged here and CANNOT be modified or deleted.
    This provides:
    - Complete audit trail for debugging
    - Replay capability for analytics
    - Compliance/accountability tracking
    """
    __tablename__ = "memory_events"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    memory_id: Mapped[str] = mapped_column(String(36), ForeignKey("memories.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(20), index=True)  # MemoryEventType enum
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Event-specific data (JSON)
    # For DECAYED: {"old_strength": 0.8, "new_strength": 0.6, "decay_amount": 0.2}
    # For REINFORCED: {"old_strength": 0.6, "new_strength": 0.85, "access_context": "search"}
    # For CONSOLIDATED: {"from_level": "episodic", "to_level": "semantic", "related_memories": [...]}
    event_data_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Optional: What triggered this event
    trigger_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "api", "scheduled", "consolidation"
    
    # NOTE: This table should have DELETE and UPDATE triggers disabled in production
    # to maintain immutability. For now, we enforce this at the application layer.
    
    __table_args__ = (
        Index("ix_memory_events_memory_time", "memory_id", "timestamp"),
        Index("ix_memory_events_user_time", "user_id", "timestamp"),
        Index("ix_memory_events_type_time", "event_type", "timestamp"),
    )




class RetrievalEvent(Base):
    """
    Tracks retrieval quality for the self-improvement feedback loop (Phase 5).
    
    Every time the agent retrieves memories to answer a query, an event is logged
    recording which memories were retrieved, which were useful, and quality signals.
    This data powers weekly quality analysis and extraction improvement suggestions.
    """
    __tablename__ = "retrieval_events"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("conversations.id"), nullable=True)
    
    # The query that triggered retrieval
    query: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Which strategies were used (JSON array: ["vector", "keyword", "graph"])
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
    quality_signal: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # good, partial, miss, empty
    retrieval_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_retrieval_events_user_created", "user_id", "created_at"),
        Index("ix_retrieval_events_quality", "user_id", "quality_signal"),
        Index("ix_retrieval_events_gaps", "user_id", "has_extraction_gap"),
    )


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    CODE = "code"
    DOCX = "docx"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"


class MediaType(str, Enum):
    """Supported media types"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Document(Base):
    """
    Uploaded documents (PDFs, Markdown, code files, etc.)
    Documents are chunked and converted into memories for semantic search.
    """
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    
    # Brain assignment
    brain_type: Mapped[str] = mapped_column(String(20), default="user", index=True)
    category: Mapped[str] = mapped_column(String(50), index=True)
    
    # File info
    filename: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(20), index=True)  # DocumentType enum
    mime_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Storage path
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # SHA256 for deduplication
    
    # Content info
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Document-specific metadata
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    encoding: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    programming_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # For code files
    
    # Processing results
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    memories_created: Mapped[int] = mapped_column(Integer, default=0)
    entities_extracted: Mapped[int] = mapped_column(Integer, default=0)
    
    # AI-generated content
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_topics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    
    # Processing options used
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=200)
    
    # Status
    processing_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, processing, completed, failed
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Metadata
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document")
    
    __table_args__ = (
        Index("ix_documents_user_brain", "user_id", "brain_type"),
        Index("ix_documents_user_type", "user_id", "file_type"),
        Index("ix_documents_hash", "file_hash"),
    )


class DocumentChunk(Base):
    """
    Individual chunks of a document for embedding.
    Each chunk becomes a searchable memory.
    """
    __tablename__ = "document_chunks"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
    memory_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("memories.id"), nullable=True, index=True)
    
    # Chunk content
    content: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    
    # Page info (for PDFs)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Embedding
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Native pgvector column
    embedding = Column(Vector(1536), nullable=True) if Vector else None
    
    # Metadata
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    
    __table_args__ = (
        Index("ix_document_chunks_doc_index", "document_id", "chunk_index"),
    )


class Media(Base):
    """
    Uploaded media files (images, videos, audio).
    Media is processed with AI to extract descriptions and transcripts.
    """
    __tablename__ = "media"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    memory_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("memories.id"), nullable=True, index=True)
    
    # Brain assignment
    brain_type: Mapped[str] = mapped_column(String(20), default="user", index=True)
    category: Mapped[str] = mapped_column(String(50), index=True)
    
    # File info
    filename: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(20), index=True)  # MediaType enum
    mime_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    # Content info
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Media-specific metadata
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    format: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Thumbnail
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # AI-generated content
    ai_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Processing results
    memories_created: Mapped[int] = mapped_column(Integer, default=0)
    
    # Status
    processing_status: Mapped[str] = mapped_column(String(20), default="pending")
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Metadata
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    __table_args__ = (
        Index("ix_media_user_brain", "user_id", "brain_type"),
        Index("ix_media_user_type", "user_id", "media_type"),
        Index("ix_media_hash", "file_hash"),
    )


class CronJob(Base):
    """Scheduled tasks for the HexBrain agent runtime."""
    __tablename__ = "cron_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))

    # Schedule
    schedule_kind: Mapped[str] = mapped_column(String(20))  # "at", "every", "cron"
    schedule_spec: Mapped[str] = mapped_column(String(200))  # Original schedule string
    schedule_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    schedule_interval_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schedule_cron_expr: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Payload
    payload_text: Mapped[str] = mapped_column(Text)

    # Telegram chat to send results to
    telegram_chat_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # State
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    run_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TelegramUserMapping(Base):
    """Maps Telegram user IDs to HexBrain user IDs for multi-user support."""
    __tablename__ = "telegram_user_mappings"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    telegram_username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    telegram_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    is_paired: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentError(Base):
    """Logged agent errors for monitoring and debugging."""
    __tablename__ = "agent_errors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    error_type: Mapped[str] = mapped_column(String(100))  # llm_error, tool_error, timeout, etc.
    error_message: Mapped[str] = mapped_column(Text)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Workflow(Base):
    """Agentic workflows — n8n-style node graphs for automation."""
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft, active, paused, error
    nodes_json: Mapped[str] = mapped_column(Text, default="[]")   # JSON array of WorkflowNode
    edges_json: Mapped[str] = mapped_column(Text, default="[]")   # JSON array of WorkflowEdge
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ApiKey(Base):
    """API keys for programmatic access to the HexBrain API."""
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. "My CI Key"
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)  # SHA-256 hash
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)  # First 8 chars for display
    scopes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of scopes
    rate_limit: Mapped[int] = mapped_column(Integer, default=60)  # requests per minute
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class VPSPlan(Base):
    """Available VPS plans that users can choose during signup."""
    __tablename__ = "vps_plans"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # "starter" | "standard" | "pro"
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    instance_type: Mapped[str] = mapped_column(String(20), nullable=False)  # EC2 instance type
    vcpu: Mapped[int] = mapped_column(Integer, nullable=False)
    ram_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)  # Monthly price in cents
    stripe_price_id: Mapped[str] = mapped_column(String(100), nullable=False, default="")

    instances: Mapped[List["VPSInstance"]] = relationship("VPSInstance", back_populates="plan")


class VPSInstance(Base):
    """A provisioned EC2 instance assigned to a user."""
    __tablename__ = "vps_instances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(20), ForeignKey("vps_plans.id"), nullable=False)
    # Lifecycle status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | provisioning | active | error | terminated
    # AWS details (populated after provisioning)
    aws_instance_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    aws_region: Mapped[str] = mapped_column(String(20), default="us-east-1")
    public_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    public_dns: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ami_id: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    # Credentials (SSH password stored plaintext — user's own VPS)
    ssh_password: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Agent API key — used by platform to authenticate requests to this user's Agent VPS
    agent_api_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Stripe
    stripe_session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    provisioned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    terminated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    plan: Mapped["VPSPlan"] = relationship("VPSPlan", back_populates="instances")

    __table_args__ = (
        Index("ix_vps_instances_user_id", "user_id"),
        Index("ix_vps_instances_status", "status"),
        Index("ix_vps_instances_stripe_session", "stripe_session_id"),
    )


class Invite(Base):
    """
    Closed-beta invite tokens.
    Admins create invites → system emails/shares a link → recipient signs up.
    Token is one-time-use and expires after `expires_at`.
    """
    __tablename__ = "invites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # The invite token (URL-safe, unique)
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # Who created this invite
    created_by: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    # Invite metadata
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Pre-assigned email (optional)
    role: Mapped[str] = mapped_column(String(20), default="beta_user")  # Role to assign on signup
    note: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Admin note
    # Lifecycle
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | used | revoked | expired
    used_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)  # User who redeemed
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # Must be set
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_invites_status", "status"),
    )


class AgentConfig(Base):
    """
    Per-user agent configuration.
    Stores the setup wizard state and all config needed to deploy
    and connect the user's personal AI agent.
    """
    __tablename__ = "agent_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), unique=True, nullable=False)

    # Step 1: Machine / Hosting
    hosting_mode: Mapped[str] = mapped_column(String(20), default="self-hosted")  # "vps" | "self-hosted"
    ssh_host: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ssh_port: Mapped[int] = mapped_column(Integer, default=22)
    ssh_user: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default="ubuntu")
    ssh_password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ssh_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # PEM key content

    # Step 2: LLM
    llm_mode: Mapped[str] = mapped_column(String(20), default="manual")  # "manual" | "bundle"
    openai_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    anthropic_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    google_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    mistral_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    groq_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    xai_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    deepseek_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    agent_model: Mapped[str] = mapped_column(String(50), default="gpt-5.2")

    # LLM Bundle subscription
    bundle_stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    bundle_status: Mapped[str] = mapped_column(String(20), default="none")  # none | active | cancelled | past_due
    bundle_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    bundle_current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Step 3: Channels
    telegram_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    discord_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    slack_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    slack_app_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    whatsapp_phone_number_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    whatsapp_access_token: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Step 4: Services
    brave_api_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    elevenlabs_api_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Deploy state
    agent_api_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # toup_ak_...
    agent_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Auto-registered by agent
    deploy_status: Mapped[str] = mapped_column(String(20), default="none")  # none | deploying | active | error
    deploy_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Wizard state
    setup_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    setup_step: Mapped[int] = mapped_column(Integer, default=1)  # Current step 1-5
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Tool access control (JSON list of disabled tool names)
    disabled_tools: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")

    # Connect token — used by terminal agent to authenticate tunnel connection
    connect_token: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_agent_configs_user_id", "user_id"),
    )


class LLMBundleAllocation(Base):
    """Per-user, per-provider budget allocation for LLM bundle subscription."""
    __tablename__ = "llm_bundle_allocations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # openai | anthropic | google | ...
    allocation_cents: Mapped[int] = mapped_column(Integer, default=0)  # Budget in cents
    used_cents: Mapped[int] = mapped_column(Integer, default=0)  # Spent in current period
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_alloc_user_provider", "user_id", "provider", unique=True),
    )


class LLMUsageRecord(Base):
    """Individual LLM API call usage record for bundle billing."""
    __tablename__ = "llm_usage_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)  # Computed cost in USD
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_usage_user_created", "user_id", "created_at"),
        Index("ix_llm_usage_user_provider", "user_id", "provider"),
    )