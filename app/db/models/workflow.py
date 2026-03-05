"""Workflow model — n8n-style node graphs for automation."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Workflow(Base):
    """Agentic workflows — n8n-style node graphs for automation."""
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft, active, paused, error
    nodes_json: Mapped[str] = mapped_column(Text, default="[]")   # JSON array of WorkflowNode
    edges_json: Mapped[str] = mapped_column(Text, default="[]")   # JSON array of WorkflowEdge
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
