"""User model."""

from datetime import datetime
from typing import Any, Dict, Optional, List
import uuid

from sqlalchemy import String, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base


# Default notification preferences. Source of truth for "what's a
# brand-new user's email opt-in state" — referenced by the account
# preferences endpoint when a User row has notification_preferences=NULL.
# - morning_briefing: bridges to the user's email_briefing Routine.
#   Off by default; user has to actively turn it on (routines are a
#   flag-gated feature anyway).
# - security_alerts: critical-only (new device, password change). True
#   so users always get the safety net unless they explicitly opt out.
# - product_updates: marketing/changelog. False — opt-in only, CASL/CAN-SPAM friendly.
DEFAULT_NOTIFICATION_PREFERENCES: Dict[str, bool] = {
    "morning_briefing": False,
    "security_alerts": True,
    "product_updates": False,
}


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

    # Account-level notification opt-in toggles. JSONB lets us add new
    # toggles (eg. weekly_digest, mention_emails) without an ALTER per
    # toggle. Shape: {"morning_briefing": bool, "security_alerts":
    # bool, "product_updates": bool, ...}. NULL is treated as
    # DEFAULT_NOTIFICATION_PREFERENCES on read so existing rows don't
    # need a backfill.
    notification_preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )

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
