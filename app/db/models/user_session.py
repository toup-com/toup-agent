"""User session model — tracks active JWT sessions per user.

Every successful login mints a JWT (with a `jti` claim) AND inserts a
row here. `get_current_user` checks the jti against this table on each
authenticated request, so revoking a row immediately logs the device
out — used by the account-page "active sessions" surface and by
password change to invalidate every other device.

We don't try to be exhaustive (no full audit log of every action);
the row is just a session presence record. last_seen_at gets bumped
on each authenticated hit so the user can see "Mac · last active
2 minutes ago" without a separate heartbeat.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # JWT jti claim — natural key. Unique across all sessions so the
    # middleware can do a single lookup by jti.
    jti: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    # Human-readable device label, derived from the User-Agent at login
    # time. "Chrome on macOS", "Safari on iPhone", "API client". Falls
    # back to "Unknown device" when the UA is empty or unparseable.
    device_label: Mapped[str] = mapped_column(String(120), nullable=False, default="Unknown device")
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    __table_args__ = (
        # Common query: list a user's non-revoked sessions ordered by
        # last activity. Covering index keeps it cheap as the table
        # grows (one row per active login forever, but cascade-delete
        # on user deletion + periodic reaping keeps it bounded).
        Index("ix_user_sessions_user_active", "user_id", "is_revoked", "last_seen_at"),
    )
