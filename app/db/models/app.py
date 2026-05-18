"""App Builder models — user-built apps, build jobs, and reconciliation logs."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Integer, Float, ForeignKey
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
    status: Mapped[str] = mapped_column(String(20), default="building")  # building, ready, running, stopped, error, untracked, orphaned
    source: Mapped[str] = mapped_column(String(30), default="app_builder")  # app_builder, vibecoding, filesystem_discovered
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
    plan_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Approved plan from conversation
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
    job_type: Mapped[str] = mapped_column(String(20), default="auto_builder")  # auto_builder, vibe_code, agent_task
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued, running, completed, failed
    steps_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of BuildStep dicts
    model: Mapped[str] = mapped_column(String(50), default="")
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    build_logs_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of BuildLog entries
    paused_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # When build was paused (token limit)
    resume_after: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # When tokens reset
    checkpoint_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Serialized build state for resume
    layer: Mapped[int] = mapped_column(Integer, default=1)  # 1 = app builder, 2 = user customization via agent
    layer2_changes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of Layer 2 changes
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # ── Unified-jobs arc (migration 046) ─────────────────────────────
    # Discriminator + back-link columns added so triggers and routines
    # can materialise a BuildJob row on every fire (PR 4 onwards). All
    # nullable, no behavior change in this PR — population starts when
    # JobRunner.create_job (PR 3) replaces the inline BuildJob(...)
    # constructions across the codebase. See
    # docs/architecture/jobs-investigation-2026-05-18.md D1+D3.
    source_kind: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    source_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    summary_message_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Richer per-type sub-state — `status` is the closed enum across
    # all job_types; `outcome` covers per-type specifics like
    # success_empty / skipped_rate_limit / skipped_filter /
    # skipped_reauth / coalesced / partial.
    outcome: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    # Composite UNIQUE with `source_id` (partial index in migration 046,
    # WHERE idempotency_key IS NOT NULL). Carries the existing
    # UNIQUE (routine_id, scheduled_for_local_date) and
    # UNIQUE (trigger_id, event_dedupe_id) semantics forward.
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)


class JobEvent(Base):
    """One material event emitted during a job's lifecycle.

    The activity-feed surface for the dashboard's Mission Control
    "ACTIVITY" section. One row per phase boundary, tool call, error,
    or output post — see migration 046 for the closed `kind` enum.
    Read via ``SELECT FROM job_events WHERE user_id=? ORDER BY ts
    DESC LIMIT 50`` (joined to ``build_jobs.title`` for attribution).

    Independent from ``BuildJob.build_logs_json`` (the per-job
    verbose log capped at 500 entries, JSON blob on the row):
    ``job_events`` is cross-job and indexable; ``build_logs_json``
    is per-job and high-cardinality. They answer different questions.
    """

    __tablename__ = "job_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("build_jobs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Denormalised so the cross-job activity-feed query doesn't have
    # to JOIN build_jobs to filter by user. Same pattern as
    # trigger_events.user_id and routine_runs.user_id.
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    # Closed enum (application-validated):
    #   phase_started | phase_completed | tool_call | error | output_posted
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # Mirrors build_jobs.status enum where applicable; NULL for kinds
    # that don't carry a status (tool_call, error).
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False, default="info")
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class BuildUsage(Base):
    """Per-LLM-call usage row emitted during an auto-builder build.

    Lets admins aggregate tokens/cost per user per provider (anthropic, openai, etc.).
    One row per LLM call. Written at job persist time in a single bulk insert to keep
    the hot build loop allocation-free. BuildJob stays the aggregate owner via
    `total_tokens`; this table is the breakdown."""
    __tablename__ = "build_usage"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("build_jobs.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(30), nullable=False, index=True)  # anthropic, openai, other
    model: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    phase: Mapped[str] = mapped_column(String(30), nullable=False, default="")  # planning, code_gen, repair, research, building, layer2
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ReconciliationLog(Base):
    """Audit log for workspace reconciliation passes."""
    __tablename__ = "reconciliation_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    trigger: Mapped[str] = mapped_column(String(20), nullable=False)  # lazy, sweep, manual
    dirs_scanned: Mapped[str] = mapped_column(Text, default="[]")  # JSON: directories scanned
    dirs_on_disk: Mapped[str] = mapped_column(Text, default="[]")  # JSON: all dirs found on disk
    dirs_in_db: Mapped[str] = mapped_column(Text, default="[]")  # JSON: all app_dirs in DB
    created_apps: Mapped[str] = mapped_column(Text, default="[]")  # JSON: new App records created
    orphaned_apps: Mapped[str] = mapped_column(Text, default="[]")  # JSON: apps marked orphaned (dir disappeared)
    excluded_dirs: Mapped[str] = mapped_column(Text, default="[]")  # JSON: dirs skipped by ignore patterns
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
