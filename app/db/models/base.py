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
    # Durable agent→platform notification outbox (Autopilot PR4) —
    # flushed to the platform's /api/agent/notify, acked rows only.
    "agent_notify_outbox",
    # Autopilot approvals — durable ask-the-user store (Autopilot PR7);
    # survives WS disconnects/restarts unlike the in-memory
    # PermissionBroker. Platform reads via the agent HTTP API only.
    "autopilot_approvals",
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
    # Search gateway telemetry. Platform-only by construction: the whole point
    # of the gateway is that the tenant container no longer sees the upstream
    # call, so it has nothing to write here.
    "search_events",
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
    # Proactive notifications (Autopilot arc PR2). Expo push tokens are
    # per-device credentials the platform owns — agent containers must
    # never hold them (same isolation stance as connector tokens above).
    # The queue is claimed/sent by the platform dispatcher; agents only
    # reach it through POST /api/agent/notify.
    "push_devices",
    "notification_queue",
    # iOS Live Activity push-to-start tokens + per-mission activity
    # state (Autopilot phone surface). APNs tokens are platform-only
    # for the same reason as Expo tokens above.
    "live_activity_devices",
    "live_activities",
}

SHARED_TABLES: set[str] = {
    # Users table exists in both: platform has all users, agent has its owner
    "users",
    # agent_configs was PLATFORM_ONLY on the theory "agent reads it via API",
    # but the agent chat/runner/tool hot path in fact reads it by DIRECT query:
    # SELECT AgentConfig WHERE user_id=… at ws_chat.py:1564/2548,
    # agent_runner.py:587/3091/3439, tool_executor.py:1091, chat.py:189/438
    # (BYOK OpenAI key + per-user disabled-tools). Older agent DBs carried the
    # table as a monolith leftover, so this was latent; newer partitioned agent
    # DBs that init_db never created it on hit `UndefinedTable`, which aborts the
    # surrounding transaction and — because the callers catch the Python error
    # without rolling back — poisons the very next write, so chat persistence
    # broke fleet-wide (2026-07 incident). SHARED creates an EMPTY table on the
    # agent: every touch site is a read that already handles the no-row case by
    # falling back (platform key / no disabled tools). No secret is ever written
    # agent-side (no INSERT/UPDATE against AgentConfig runs in agent mode), so the
    # platform-owns-the-config isolation still holds. This is the permanent fix
    # for that incident; the live fleet was already hot-patched with the table.
    "agent_configs",
}
