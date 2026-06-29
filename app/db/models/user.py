"""User model."""

from datetime import datetime
from typing import Any, Dict, Optional, List
import uuid

from sqlalchemy import String, DateTime, Boolean, JSON, Text
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

    # ── Email verification (alembic 055, F13) ─────────────
    # NULL = unverified. Set to UTC timestamp when the user clicks the
    # link in their verification email. CreditService refuses chargeable
    # activity for unverified users when
    # ``settings.require_email_verification_for_credits=True``.
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Opaque urlsafe-base64 token (32 bytes raw → 43 chars). Cleared on
    # successful verify so the link is single-use. Rotated by
    # ``POST /api/auth/send-verification`` to invalidate the old link.
    email_verification_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Throttle resends — /auth/send-verification refuses with 429 if a
    # send already happened in the last 60s.
    email_verification_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Sign in with Apple refresh token, Fernet-encrypted (credential_crypto).
    # Captured at sign-in by exchanging the one-time authorization_code so
    # account deletion can revoke the Apple grant (Guideline 5.1.1(v)).
    # NULL for email/Google users and for Apple sign-ins that happened
    # before the SIWA signing key was configured. Migration 061.
    apple_refresh_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Stable Sign-in-with-Apple subject identifier (`sub` claim). Unique
    # per Apple-ID per app and STABLE even when the user toggles
    # Hide-My-Email (the relay address can change; `sub` does not). Stored
    # so Apple sign-ins can dedupe on the durable identity instead of the
    # mutable relay email. NULL for email/Google users and for Apple
    # sign-ins captured before this column existed. Migration 064.
    apple_sub: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True, index=True)

    # Account-level notification opt-in toggles. JSONB lets us add new
    # toggles (eg. weekly_digest, mention_emails) without an ALTER per
    # toggle. Shape: {"morning_briefing": bool, "security_alerts":
    # bool, "product_updates": bool, ...}. NULL is treated as
    # DEFAULT_NOTIFICATION_PREFERENCES on read so existing rows don't
    # need a backfill.
    #
    # `with_variant(JSONB(), "postgresql")` keeps the prod schema as
    # JSONB (binary, indexable) while letting SQLite-based test runs
    # (billing CI uses sqlite+aiosqlite) actually compile the column.
    # Without this variant, every push has hit
    # `SQLiteTypeCompiler … can't render element of type JSONB` and
    # the CI Backend Tests workflow fails before the first assertion.
    notification_preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
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

    # NOTE: The partial unique index `uq_users_is_canary_partial` lives
    # only in alembic migration 0022 (Postgres-specific via `postgresql_where`).
    # We deliberately do NOT declare it in `__table_args__` because
    # SQLAlchemy's `Base.metadata.create_all` (used by the test conftest and
    # by `init_db()` on sqlite) ignores `postgresql_where` and emits a
    # TOTAL unique index on `is_canary` — every test that creates a second
    # user (default `is_canary=False` × 2) then fails with
    # `UNIQUE constraint failed: users.is_canary`. Production Postgres still
    # gets the partial index from the migration; sqlite test runs get no
    # index, which is the right behavior (canary uniqueness is irrelevant
    # in a single-tenant test DB).
