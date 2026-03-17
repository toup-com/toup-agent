"""Entity and knowledge graph models."""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, ForeignKey, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR

from .base import Base, Vector


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
    entity_type: Mapped[str] = mapped_column(String(50), index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Embedding for the entity
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding = Column(Vector(), nullable=True) if Vector else None

    # Schema-enforced extraction type
    schema_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)

    # Metadata
    attributes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Full-text search on name
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

    role: Mapped[str] = mapped_column(String(50), default="mentioned")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    memory: Mapped["Memory"] = relationship("Memory", back_populates="entity_links")
    entity: Mapped["Entity"] = relationship("Entity", back_populates="memory_links")


class EntityRelationship(Base):
    """
    Direct entity-to-entity relationships for the knowledge graph.
    Enables efficient graph traversal between entities.
    """
    __tablename__ = "entity_relationships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    source_entity_id: Mapped[str] = mapped_column(String(36), ForeignKey("entities.id"), nullable=False, index=True)
    target_entity_id: Mapped[str] = mapped_column(String(36), ForeignKey("entities.id"), nullable=False, index=True)
    relationship_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    relationship_label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
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
