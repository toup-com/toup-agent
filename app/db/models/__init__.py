"""
Toup Database Models — split by domain for clarity.

All models are re-exported here so existing imports continue to work:
    from app.db.models import User, Memory, Conversation, ...
"""

# Base
from .base import Base, Vector

# Enums
from .enums import (
    BrainType, MemoryCategory, BrainRegion, WorkCategory, AgentCategory,
    MemoryType, EntityType, MemoryLevel, MemoryEventType,
    UserRole, IdentityType, DocumentType, MediaType,
)

# User
from .user import User

# Identity
from .identity import Identity

# Conversation & Messages
from .conversation import Conversation, Message

# Day-as-Chat (day-level conversation container + telemetry + migrations)
from .day_chat import DayChat, ContextBudgetLog, MigrationStatus

# Memory system
from .memory import Memory, memory_relationships, BrainStats, MemoryEvent, RetrievalEvent

# Entity & Knowledge Graph
from .entity import Entity, EntityLink, EntityRelationship

# Documents & Media
from .document import Document, DocumentChunk, Media

# Apps & Build Jobs
from .app import App, BuildJob, BuildUsage, ReconciliationLog

# Agent runtime
from .agent import CronJob, TelegramUserMapping, AgentError, ApiKey, AgentConfig

# Soul config
from .soul_config import SoulConfig

# Platform (VPS, invites, billing)
from .platform import VPSPlan, VPSInstance, ManagedContainer, Invite, LLMBundleAllocation, LLMUsageRecord, LLMProxyEvent, PlatformSetting

# Rollouts (Phase 3 automated deployment)
from .rollout import (
    Rollout, RolloutAttempt,
    ROLLOUT_STATUSES, ROLLOUT_ATTEMPT_STATUSES, ROLLOUT_TRIGGERS,
)

# Streaming credentials
from .streaming import StreamingCredential, CredentialAccessLog

# Account deletion (§1.3): audit receipt + sensitive-action redemption set
from .deletion import (
    DeletionAuditEvent, SensitiveActionRedemption,
    DELETION_STATUSES, DELETION_ACTORS,
)

__all__ = [
    # Base
    "Base", "Vector",
    # Enums
    "BrainType", "MemoryCategory", "BrainRegion", "WorkCategory", "AgentCategory",
    "MemoryType", "EntityType", "MemoryLevel", "MemoryEventType",
    "UserRole", "IdentityType", "DocumentType", "MediaType",
    # Models
    "User", "Identity",
    "Conversation", "Message",
    "Memory", "memory_relationships", "BrainStats", "MemoryEvent", "RetrievalEvent",
    "Entity", "EntityLink", "EntityRelationship",
    "Document", "DocumentChunk", "Media",
    "App", "BuildJob", "BuildUsage", "ReconciliationLog",
    "CronJob", "TelegramUserMapping", "AgentError", "ApiKey", "AgentConfig",
    "SoulConfig",
    "VPSPlan", "VPSInstance", "ManagedContainer", "Invite", "LLMBundleAllocation", "LLMUsageRecord", "LLMProxyEvent", "PlatformSetting",
    "Rollout", "RolloutAttempt",
    "ROLLOUT_STATUSES", "ROLLOUT_ATTEMPT_STATUSES", "ROLLOUT_TRIGGERS",
    "StreamingCredential", "CredentialAccessLog",
    "DayChat", "ContextBudgetLog", "MigrationStatus",
    "DeletionAuditEvent", "SensitiveActionRedemption",
    "DELETION_STATUSES", "DELETION_ACTORS",
]
