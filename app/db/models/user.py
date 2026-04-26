"""User model."""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import String, DateTime, Boolean, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base


class User(Base):
    """User model for multi-user isolation"""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    name: Mapped[Optional[str]] = mapped_column(String(255))
    password_changed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="beta_user", index=True)  # admin | beta_user
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Stripe
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True)

    # Timezone (IANA format, e.g. "America/Toronto") — used for day boundaries
    timezone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # is_canary: designates this user as the rollout canary. Rollouts upgrade
    # this tenant FIRST, wait out the canary window watching /agent/health,
    # and only proceed to the rest of the fleet if it stays healthy. If no
    # user has is_canary=True (or the canary user isn't in the running set),
    # rollouts ABORT before touching anyone.
    # Partial unique index (in __table_args__) guarantees at most one
    # user carries is_canary=True at a time.
    is_canary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="user")
    conversations: Mapped[List["Conversation"]] = relationship("Conversation", back_populates="user")
    identities: Mapped[List["Identity"]] = relationship("Identity", back_populates="user")
    soul_config: Mapped[Optional["SoulConfig"]] = relationship("SoulConfig", back_populates="user", uselist=False)

    __table_args__ = (
        # Enforce a single canary user. Postgres partial unique index: only
        # rows WHERE is_canary=true participate in the uniqueness check, so
        # the default false-for-everyone state stays conflict-free.
        Index(
            "uq_users_is_canary_partial",
            "is_canary",
            unique=True,
            postgresql_where="is_canary = true",
        ),
    )
