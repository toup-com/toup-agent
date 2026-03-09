"""App Builder models — user-built apps and build jobs."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class App(Base):
    """A user-built app (React Native/Expo) running on the VPS."""
    __tablename__ = "apps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    slug: Mapped[str] = mapped_column(String(60), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(20), default="building")  # building, ready, running, stopped, error
    port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Metro/mobile port 3001-3050
    web_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Expo web port 4001-4050
    app_dir: Mapped[str] = mapped_column(Text, nullable=False)  # /opt/toup-agent/apps/{id}
    metro_pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    web_pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    build_job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    files_json: Mapped[str] = mapped_column(Text, default="{}")  # backup of files dict
    deps_json: Mapped[str] = mapped_column(Text, default="{}")  # npm dependencies
    db_type: Mapped[str] = mapped_column(String(20), default="none")  # sqlite, supabase, none
    db_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # SQLite path or Supabase URL
    storage_dir: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    github_repo: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # "user/repo-name"
    github_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    publish_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Custom domain or published URL
    platforms: Mapped[str] = mapped_column(String(50), default="web,ios")  # Comma-separated
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BuildJob(Base):
    """A background job that builds an app."""
    __tablename__ = "build_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    app_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued, running, completed, failed
    steps_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of BuildStep dicts
    model: Mapped[str] = mapped_column(String(50), default="")
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
