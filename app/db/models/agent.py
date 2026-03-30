"""Agent runtime models — cron jobs, telegram mapping, errors, API keys, config."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Integer, BigInteger, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CronJob(Base):
    """Scheduled tasks for the Toup agent runtime."""
    __tablename__ = "cron_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))

    # Schedule
    schedule_kind: Mapped[str] = mapped_column(String(20))  # "at", "every", "cron"
    schedule_spec: Mapped[str] = mapped_column(String(200))
    schedule_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    schedule_interval_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schedule_cron_expr: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Payload
    payload_text: Mapped[str] = mapped_column(Text)

    # Telegram chat to send results to
    telegram_chat_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # State
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    run_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TelegramUserMapping(Base):
    """Maps Telegram user IDs to Toup user IDs for multi-user support."""
    __tablename__ = "telegram_user_mappings"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    telegram_username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    telegram_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    is_paired: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentError(Base):
    """Logged agent errors for monitoring and debugging."""
    __tablename__ = "agent_errors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    error_type: Mapped[str] = mapped_column(String(100))
    error_message: Mapped[str] = mapped_column(Text)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ApiKey(Base):
    """API keys for programmatic access to the Toup API."""
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(10), nullable=False)
    scopes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rate_limit: Mapped[int] = mapped_column(Integer, default=60)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentConfig(Base):
    """
    Per-user agent configuration.
    Stores the setup wizard state and all config needed to deploy
    and connect the user's personal AI agent.
    """
    __tablename__ = "agent_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), unique=True, nullable=False)

    # Step 1: Machine / Hosting
    hosting_mode: Mapped[str] = mapped_column(String(20), default="self-hosted")
    ssh_host: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ssh_port: Mapped[int] = mapped_column(Integer, default=22)
    ssh_user: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default=None)
    ssh_password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ssh_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    target_os: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Step 2: LLM
    llm_mode: Mapped[str] = mapped_column(String(20), default="manual")
    openai_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    anthropic_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    google_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    mistral_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    groq_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    xai_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    deepseek_api_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    agent_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    agent_model: Mapped[str] = mapped_column(String(50), default="claude-opus-4-6")

    # LLM Bundle subscription
    bundle_stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    bundle_status: Mapped[str] = mapped_column(String(20), default="none")
    bundle_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    bundle_current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Step 3: Channels
    telegram_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    discord_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    slack_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    slack_app_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    whatsapp_phone_number_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    whatsapp_access_token: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Step 4: Services
    brave_api_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    elevenlabs_api_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Deploy state
    agent_api_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    agent_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    deploy_status: Mapped[str] = mapped_column(String(20), default="none")
    deploy_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Wizard state
    setup_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    setup_step: Mapped[int] = mapped_column(Integer, default=1)
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    agent_color: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)

    # Tool access control
    disabled_tools: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")

    # Connect token
    connect_token: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Database mode
    db_mode: Mapped[str] = mapped_column(String(20), default="auto")
    supabase_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_agent_configs_user_id", "user_id"),
    )
