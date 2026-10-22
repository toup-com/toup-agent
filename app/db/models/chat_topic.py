"""Platform-owned derived chat labels; never a mirror of tenant messages."""

from datetime import date, datetime

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base


class ChatTopicTitle(Base):
    """One current title/lease per user, day and canonical conversation anchor.

    Only the generated title and a source fingerprint leave the tenant's chat
    store. The anchor is deliberately not a FK: the platform has no messages
    table. Account deletion still removes every derived label via users.
    """

    __tablename__ = "chat_topic_titles"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True,
    )
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    canonical_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    topic: Mapped[str | None] = mapped_column(String(160), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        CheckConstraint("version > 0", name="ck_chat_topic_titles_version"),
        Index("ix_chat_topic_titles_expires_at", "expires_at"),
    )
