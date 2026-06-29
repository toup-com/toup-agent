"""Shared SQLAlchemy base and imports for all model files."""

from datetime import datetime
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, BigInteger, Boolean,
    ForeignKey, Table, Enum as SQLEnum, JSON, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR

try:
    from pgvector.sqlalchemy import Vector as _PgVector
except ImportError:
    _PgVector = None  # Fallback for environments without pgvector


def _get_embedding_dim() -> int:
    """Get configured embedding dimension (avoids circular import at class-body time)."""
    try:
        from app.config import settings
        return settings.embedding_dimension
    except Exception:
        return 1536  # safe default


def Vector(dim: int = 0):
    """Return a pgvector Vector column type with the configured dimension."""
    if _PgVector is None:
        return None
    if dim <= 0:
        dim = _get_embedding_dim()
    return _PgVector(dim)


Base = declarative_base()


# ── Table Partitioning: Platform vs Agent DBs ─────────────────────────
#
# Toup has a hybrid database architecture:
#   - Platform DB (Railway/Supabase): central, shared across all users
#   - Agent DB (per-user Docker container): isolated per user
#
# When you add a new model, add its __tablename__ to exactly ONE of these
# sets, or to SHARED_TABLES if it genuinely lives in both databases.
# If you skip this step, test_table_partitioning.py will fail.
#
# init_db() uses these sets to decide which tables to create in each mode.
# ──────────────────────────────────────────────────────────────────────

AGENT_ONLY_TABLES: set[str] = {
    # Day-as-Chat context architecture
    "day_chats",
    "context_budget_logs",
    "migration_status",
    # Conversation & messages (agent stores all chat data)
    "conversations",
    "messages",
    # Durable exactly-once ledger for inbound chat (see ProcessedMessage).
    "processed_messages",
    # Memory system
    "memories",
    "memory_relationships",
    "brain_stats",
    "memory_events",
    "retrieval_events",
    # Entity & knowledge graph
    "entities",
    "entity_links",
    "entity_relationships",
    # Documents & media
    "documents",
    "document_chunks",
    "media",
    # Apps & build jobs
    "apps",
    "build_jobs",
    "build_usage",
    "reconciliation_logs",
    # Agent runtime
    "cron_jobs",
    "telegram_user_mappings",
    "agent_errors",
    "api_keys",
    # System-managed routines (email briefing, calendar briefing, …).
    # User-facing config in /agent/settings/routines; operator monitoring
    # in Mission Control reads via authenticated API, not direct DB query.
    "routines",
    "routine_runs",
    "routine_notification_dedupe",
    # Event-driven automations (Gmail Pub/Sub etc.). Platform-side
    # webhook authenticates the Pub/Sub push and dispatches an envelope
    # to the user's per-tenant agent container via the bridge; the
    # agent owns the trigger config + event audit trail. Tokens never
    # leave the platform — same isolation guarantee as Routines.
    "triggers",
    "trigger_events",
    # WhatsApp Cloud API webhook dedupe (agent-side; survives container
    # restarts so Meta retries within a 7-day window cannot re-run the LLM)
    "whatsapp_inbound_dedupe",
    # Identity & soul
    "identities",
    "soul_configs",
}

PLATFORM_ONLY_TABLES: set[str] = {
    # VPS & infrastructure
    "vps_plans",
    "vps_instances",
    "managed_containers",
    # Automated deployment pipeline (Phase 3)
    "rollouts",
    "rollout_attempts",
    # Billing & invites
    "invites",
    "llm_bundle_allocations",
    "llm_usage_records",
    "llm_proxy_events",
    # Agent configuration (platform manages, agent reads via API)
    "agent_configs",
    # Streaming credentials (platform stores, agent fetches via API)
    "streaming_credentials",
    "credential_access_log",
    # Platform-wide admin settings (editable via admin panel)
    "platform_settings",
    # Account deletion audit + sensitive-action replay defense (§1.3).
    "deletion_audit_events",
    "sensitive_action_redemptions",
    # Per-tenant security audit (T0b — agent_api_key rotation, future
    # connector quarantine, future security primitives). Append-only.
    "agent_security_events",
    # Third-party connector vault (T1a). Tokens stay platform-side
    # only — the bridge never sees them, tenant containers never see
    # them. Reaches providers via the platform's MCP server (T0c-gated
    # auth) → connector dispatcher (T1e).
    "connector_identities",
    "connector_oauth_sessions",
    "connector_events",
    "connector_user_preferences",
    # OAuth provider-app credentials (Google/GitHub/etc client_id+secret).
    # Persisted in DB so the operator can paste them through the admin
    # UI instead of editing env vars + redeploying.
    "provider_app_credentials",
    # Active user JWT sessions. Auth router is platform-only (users
    # don't log into agent containers — agent_api_key auth is used
    # there instead), so the table only exists on the platform side.
    "user_sessions",
    # Maintenance / support agent: ingests user-reported problems,
    # diagnoses against docs/skills, and (after admin approval) opens a
    # fix PR. Operator/admin tool — never lives on tenant agent DBs.
    "support_issues",
    "support_issue_events",
    # Support card attachments (e.g. mobile screenshots) — bytes live in the
    # platform DB, served only via an auth'd reporter/admin endpoint.
    "support_attachments",
    # Free-credit grant eligibility tombstone + persistent signup-attempt
    # log. Signup, the free-credit grant, and IP rate limiting all live on
    # the platform — agent containers neither create accounts nor grant
    # credits — so these are platform-only (mirrors the migration guards
    # that skip agent DBs).
    "grant_eligibility",
    "signup_attempts",
}

SHARED_TABLES: set[str] = {
    # Users table exists in both: platform has all users, agent has its owner
    "users",
}
