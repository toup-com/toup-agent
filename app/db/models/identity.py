"""Identity model — agent personality and behavior documents."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Integer, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base


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
