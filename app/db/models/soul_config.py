"""SoulConfig model — structured agent personality configuration."""

from datetime import datetime
import uuid

from sqlalchemy import String, Text, DateTime, Boolean, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base


class SoulConfig(Base):
    """
    Structured soul configuration for an agent.
    Compiles into an Identity(type='soul') record for the system prompt.
    """
    __tablename__ = "soul_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), unique=True, index=True)

    # Identity
    name: Mapped[str] = mapped_column(String(50), default="Agent")
    color: Mapped[str] = mapped_column(String(7), default="#9B59B6")  # hex color
    pronouns: Mapped[str] = mapped_column(String(10), default="they")

    # Communication style: casual, professional, mentor, creative
    style: Mapped[str] = mapped_column(String(20), default="casual")

    # Personality traits — JSON array of enabled trait keys
    traits: Mapped[str] = mapped_column(JSON, default=list)

    # Free-text custom instructions (max 500 chars)
    custom_instructions: Mapped[str] = mapped_column(Text, default="")

    # Compiled text (cached output of compile_soul)
    compiled_text: Mapped[str] = mapped_column(Text, default="")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="soul_config")
