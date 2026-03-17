"""Document and Media models."""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base, Vector


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
    file_type: Mapped[str] = mapped_column(String(20), index=True)
    mime_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Content info
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Document-specific metadata
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    encoding: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    programming_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Processing results
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    memories_created: Mapped[int] = mapped_column(Integer, default=0)
    entities_extracted: Mapped[int] = mapped_column(Integer, default=0)

    # AI-generated content
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_topics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Processing options used
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=200)

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

    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document")

    __table_args__ = (
        Index("ix_documents_user_brain", "user_id", "brain_type"),
        Index("ix_documents_user_type", "user_id", "file_type"),
        Index("ix_documents_hash", "file_hash"),
    )


class DocumentChunk(Base):
    """Individual chunks of a document for embedding."""
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
    embedding = Column(Vector(), nullable=True) if Vector else None

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
    """Uploaded media files (images, videos, audio)."""
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
    media_type: Mapped[str] = mapped_column(String(20), index=True)
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
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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
