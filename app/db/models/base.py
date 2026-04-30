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
}

SHARED_TABLES: set[str] = {
    # Users table exists in both: platform has all users, agent has its owner
    "users",
}
