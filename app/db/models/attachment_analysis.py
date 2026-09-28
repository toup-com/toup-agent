"""Agent-only durable claim and checkpoint for full chat attachments."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class AttachmentAnalysisJob(Base):
    __tablename__ = "attachment_analysis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    attachment_id: Mapped[str] = mapped_column(String(32), nullable=False)
    analysis_id: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    delivered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    state_json: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    claim_owner: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
