"""Durable signup-attempt log for IP rate limiting.

The in-memory ``SignupRateLimiter`` (app/services/rate_limiter.py) resets on
every deploy and is per-process, so it splits across replicas and is
defeated by a burst right after a rollout. This table backs a persistent,
cross-replica limiter (app/services/signup_guard.py) so the per-IP signup
cap actually holds.

One row per consumed signup slot. Rows are pruned per-IP on write (rows
older than the window are deleted), so the table stays bounded to roughly
``active_ips × max_per_window``. Platform-only; absent on agent DBs.
"""

from datetime import datetime
import uuid

from sqlalchemy import String, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SignupAttempt(Base):
    __tablename__ = "signup_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ip: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("ix_signup_attempts_ip_created", "ip", "created_at"),)
