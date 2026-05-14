"""Agent runtime models — cron jobs, telegram mapping, errors, API keys, config."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Integer, BigInteger, Boolean, ForeignKey, Index, PrimaryKeyConstraint
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

    # Phase B (mig 042): pointer to the Routine that replaced this row.
    # NULL until mig 043 backfills. When set, CronService skips the row
    # at load time (Routine runner owns delivery now — anti-double-fire).
    migrated_to_routine_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True,
    )

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


class ExtensionDevice(Base):
    """A Chrome browser paired to a Toup user's tenant agent.

    One row per paired device — same user can install the Toup extension on
    their laptop + desktop and we keep both. `paired_at` is the canonical
    "this is paired" signal that the settings page reads from, replacing the
    in-memory `extension_bridge.is_connected` check which lied because the
    bridge runs on the tenant agent process and the settings endpoint runs
    on the platform process — different processes, different memory.

    `last_seen_at` is bumped by the tenant agent every time the extension's
    /api/ws/extension socket connects, sends a heartbeat, or closes; the
    settings page renders "online" / "offline since X" from the delta.

    `revoked_at` is set when the user explicitly clicks Unpair on the
    settings page. Soft-delete so we keep audit history of which device
    was used when.
    """
    __tablename__ = "extension_devices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    # Short user-supplied name from pair flow, e.g. "MacBook · Chrome".
    device_name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Free-form version string the extension self-reports on pair.
    extension_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    # Truncated JWT jti so we can revoke a specific issued token without
    # storing the full bearer. ON revoke, future WS auths with this jti
    # are rejected via a denylist (added when we wire revocation; today
    # the token TTL is short enough that simple soft-delete is enough).
    token_jti: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    paired_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)

    __table_args__ = (
        Index("ix_extension_devices_user_revoked", "user_id", "revoked_at"),
    )


class WhatsAppInboundDedupe(Base):
    """Persistent dedupe of WhatsApp Cloud API inbound webhook messages.

    Meta retries webhook deliveries when the receiving endpoint returns a
    non-2xx, times out, or the connection drops mid-response. The retry
    cadence is roughly 0s / 30s / 5min / 30min / ... within a 7-day window.
    Without this table every retry would re-run the LLM, double-charge the
    user, and post duplicate replies.

    On every inbound webhook the channel adapter calls ``claim(user_id,
    message_id)`` BEFORE dispatch. ``claim`` is a single ``INSERT ... ON
    CONFLICT DO NOTHING`` that returns True only the first time the
    composite key is seen. A periodic sweep deletes rows older than the
    retention window (default 20 minutes — comfortably longer than Meta's
    initial retry burst, short enough to keep the table tiny).

    Survives container restarts on purpose: an in-memory dedupe would lose
    state during the agent's boot window, which is exactly when Meta is
    most likely to redeliver (if the previous instance went down mid-POST).
    """

    __tablename__ = "whatsapp_inbound_dedupe"

    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    message_id: Mapped[str] = mapped_column(String(200), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        PrimaryKeyConstraint("user_id", "message_id", name="pk_whatsapp_inbound_dedupe"),
        Index("ix_whatsapp_inbound_dedupe_received_at", "received_at"),
    )


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
    # NULL = "follow platform default at use-time via model_resolver.default_model()".
    # The old "claude-opus-4-6" hardcoded default predated the resolver; new rows
    # now correctly inherit from settings.agent_model. Migration 029 dropped the
    # stale server_default and backfilled lingering gpt-5.2 rows to NULL.
    agent_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Auto-builder Planner/Builder split (migration 029). NULL on either column
    # means "fall through to agent_model" — see model_resolver.app_builder_*_model().
    app_builder_planner_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    app_builder_builder_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # User's preferred LLM provider for bundle mode (anthropic | openai).
    # Routes which model the model_router picks when no explicit override.
    # Bundle subscribers can flip this from /agent/settings without code
    # changes — useful for "I want my agent on GPT" preferences. Default
    # 'anthropic' matches today's behavior.
    preferred_provider: Mapped[str] = mapped_column(
        String(20), nullable=False, default="anthropic", server_default="anthropic"
    )

    # User's preferred LLM provider for bundle mode (anthropic | openai).
    # Routes which model the model_router picks when no explicit override.
    # Bundle subscribers can flip this from /agent/settings without code
    # changes — useful for "I want my agent on GPT" preferences. Default
    # 'anthropic' matches today's behavior.
    preferred_provider: Mapped[str] = mapped_column(
        String(20), nullable=False, default="anthropic", server_default="anthropic"
    )

    # LLM Bundle subscription
    bundle_stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    bundle_status: Mapped[str] = mapped_column(String(20), default="none")
    bundle_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    bundle_current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # When the subscription transitioned to bundle_status='cancelled'.
    # Drives the managed-container reaper grace period — see
    # `scheduled_tasks.run_managed_container_reaper`. NULL until the
    # first cancellation; reset on resubscribe via _handle_invoice_succeeded.
    bundle_cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # LLM Proxy auth + budget
    llm_token_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    bundle_anthropic_budget_cents: Mapped[int] = mapped_column(Integer, default=3000)
    bundle_openai_budget_cents: Mapped[int] = mapped_column(Integer, default=1000)
    bundle_anthropic_daily_cap_cents: Mapped[int] = mapped_column(Integer, default=100)
    bundle_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    bundle_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Per-user OpenAI project + key, auto-provisioned on bundle activation via
    # OpenAI Admin API. The proxy uses bundle_openai_api_key on outbound (so
    # OpenAI bills per project for granular usage attribution) without ever
    # putting the key in the agent's container env. On admin user-delete,
    # bundle_openai_project_id is used to archive the project (cascade-revokes
    # all keys, stops billing). Distinct from BYOK `openai_api_key` above.
    bundle_openai_project_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    bundle_openai_api_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Step 3: Channels
    telegram_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    discord_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    slack_bot_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    slack_app_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    whatsapp_phone_number_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    whatsapp_access_token: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # Webhook verification token. User picks any string and pastes the same
    # value into the Meta App webhook configuration. Compared in
    # constant-time during the GET /api/whatsapp/webhook handshake.
    whatsapp_verify_token: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # App Secret from Meta App → Settings → Basic. Used to verify the
    # X-Hub-Signature-256 header on every inbound webhook POST. If null,
    # signature verification is skipped (and a warning is logged at boot
    # so operators see the security degrade explicitly).
    whatsapp_app_secret: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # Set the first time the agent successfully receives a non-handshake
    # POST on /api/whatsapp/webhook. Surfaces a "Connected" badge in the
    # Settings UI; cleared on disconnect.
    whatsapp_connected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Which WhatsApp transport this tenant uses.
    #   "cloud_api"  — Path A / BYOA Cloud API (Meta-sanctioned, user's own Meta App)
    #   "qr_link"    — Path C / Baileys-style QR pairing via neonize
    #   None         — WhatsApp not configured at all
    # Mode is mutually exclusive: only one channel adapter spawns per
    # container. The other mode's columns are ignored.
    whatsapp_mode: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    # E.164 of the dedicated phone number that holds the linked-device
    # session (number "B" in the user-facing docs). Display only — the
    # JID derived from this is what neonize actually uses.
    whatsapp_self_e164: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    # Comma-separated E.164 list of numbers the agent will respond to.
    # Inbound from any other sender is silently dropped at the ACL gate.
    # Empty / NULL = "block everything", which matches the secure default.
    whatsapp_baileys_allowlist: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Lifecycle of the QR-link session.
    #   "not_linked"  — never paired, or logged out
    #   "linking"     — QR rendered, waiting for the user to scan
    #   "linked"      — paired, session live (or transiently disconnected;
    #                   reconnect logic recovers automatically)
    #   "logged_out"  — Meta forced a 401; user must re-pair
    whatsapp_session_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # ── Toup Code (experimental Claude Code IDE, 2026-05-12) ──
    # Long-lived OAuth token from `claude setup-token` on the user's
    # local machine. When set, the Toup Code page can spawn Claude
    # Code sessions authenticated against the user's Claude
    # Pro/Max subscription quota — Toup never bills for these calls.
    # Stored plaintext (same trust model as the other channel tokens
    # above); rotate to column-level encryption before opening this
    # to public if the experiment graduates.
    claude_code_oauth_token: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)

    # OpenAI API key for the `codex` CLI (the OpenAI Codex companion to
    # Claude Code, npm i -g @openai/codex). Same trust + storage model
    # as the Anthropic side; Toup Code routes /code/spawn to whichever
    # CLI the user picked. Migration 039 added the column.
    openai_codex_token: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)

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
    setup_step: Mapped[int] = mapped_column(Integer, default=0)
    setup_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # 'auto' | 'manual' | None
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
