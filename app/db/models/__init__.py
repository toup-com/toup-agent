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
from .user_session import UserSession

# Identity
from .identity import Identity

# Conversation & Messages
from .conversation import Conversation, Message, ProcessedMessage

# Day-as-Chat (day-level conversation container + telemetry + migrations)
from .day_chat import DayChat, ContextBudgetLog, MigrationStatus

# Memory system
from .memory import Memory, memory_relationships, BrainStats, MemoryEvent, RetrievalEvent

# Entity & Knowledge Graph
from .entity import Entity, EntityLink, EntityRelationship

# Documents & Media
from .document import Document, DocumentChunk, Media

# Apps & Build Jobs
from .app import App, BuildJob, BuildUsage, JobEvent, ReconciliationLog

# Agent runtime
from .agent import CronJob, TelegramUserMapping, AgentError, ApiKey, AgentConfig, ExtensionDevice

# Routines (system-managed scheduled actions — email briefing, etc.)
from .routine import Routine, RoutineRun, RoutineNotificationDedupe

# Triggers (event-driven automations — Gmail Pub/Sub, etc.)
from .trigger import (
    Trigger, TriggerEvent,
    TRIGGER_KINDS, TRIGGER_ACTIONS, TRIGGER_STATUSES, TRIGGER_EVENT_STATUSES,
)

# Soul config
from .soul_config import SoulConfig

# Platform (VPS, invites, billing)
from .platform import VPSPlan, VPSInstance, ManagedContainer, Invite, LLMBundleAllocation, LLMUsageRecord, LLMProxyEvent, PlatformSetting

# Credits (free-for-everyone billing — docs/credits/design.md)
from .credit import (
    SubscriptionPlan, CreditBalance, CreditLedger, CreditReservation,
    AppleSubscription,
    APPLE_SUB_ACTIVE, APPLE_SUB_EXPIRED, APPLE_SUB_BILLING_RETRY,
    APPLE_SUB_GRACE, APPLE_SUB_REVOKED,
    BUCKET_MESSAGE, BUCKET_INTEGRATION,
    LEDGER_CHAT_MESSAGE, LEDGER_ROUTINE_RUN, LEDGER_TRIGGER_RUN, LEDGER_BUILD_STEP,
    LEDGER_TOOL_CALL, LEDGER_BROWSER_ACTION, LEDGER_DOC_GEN,
    LEDGER_RESERVATION, LEDGER_SETTLEMENT, LEDGER_REFUND, LEDGER_PLAN_GRANT,
    LEDGER_PLAN_CHANGE, LEDGER_DAILY_RESET, LEDGER_PERIOD_RENEWAL, LEDGER_MANUAL_ADJUST,
    LEDGER_IAP_PURCHASE,
    RESERVATION_OPEN, RESERVATION_SETTLED, RESERVATION_REFUNDED, RESERVATION_EXPIRED,
)

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

# Per-tenant security audit (T0b) — agent_api_key rotation, future
# connector quarantine, future security primitives. Append-only.
from .security import (
    AgentSecurityEvent,
    EVENT_AGENT_KEY_ROTATED,
    EVENT_AGENT_KEY_ROTATION_FAILED,
)

# Third-party connector vault (T1a). Platform-only.
from .connectors import (
    ConnectorIdentity,
    ConnectorOAuthSession,
    ConnectorEvent,
    ConnectorUserPreference,
    ProviderAppCredential,
    CONNECTOR_IDENTITY_STATUSES,
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_REAUTH_STARTED,
    EVENT_REAUTH_COMPLETED,
    EVENT_REFRESH_SUCCEEDED,
    EVENT_REFRESH_FAILED,
    EVENT_TOOL_CALLED,
    EVENT_TOOL_SUCCEEDED,
    EVENT_TOOL_FAILED,
    EVENT_TOOL_ELEVATION_REQUIRED,
    EVENT_REVOCATION_PROVIDER_SUCCEEDED,
    EVENT_REVOCATION_PROVIDER_FAILED,
    EVENT_HEALTH_PROBE_SUCCEEDED,
    EVENT_HEALTH_PROBE_FAILED,
    EVENT_FORCE_QUARANTINED,
    EVENT_FORCE_RELEASED,
)

__all__ = [
    # Base
    "Base", "Vector",
    # Enums
    "BrainType", "MemoryCategory", "BrainRegion", "WorkCategory", "AgentCategory",
    "MemoryType", "EntityType", "MemoryLevel", "MemoryEventType",
    "UserRole", "IdentityType", "DocumentType", "MediaType",
    # Models
    "User", "UserSession", "Identity",
    "Conversation", "Message", "ProcessedMessage",
    "Memory", "memory_relationships", "BrainStats", "MemoryEvent", "RetrievalEvent",
    "Entity", "EntityLink", "EntityRelationship",
    "Document", "DocumentChunk", "Media",
    "App", "BuildJob", "BuildUsage", "ReconciliationLog",
    "CronJob", "TelegramUserMapping", "AgentError", "ApiKey", "AgentConfig", "ExtensionDevice",
    "Routine", "RoutineRun", "RoutineNotificationDedupe",
    "Trigger", "TriggerEvent",
    "TRIGGER_KINDS", "TRIGGER_ACTIONS", "TRIGGER_STATUSES", "TRIGGER_EVENT_STATUSES",
    "SoulConfig",
    "VPSPlan", "VPSInstance", "ManagedContainer", "Invite", "LLMBundleAllocation", "LLMUsageRecord", "LLMProxyEvent", "PlatformSetting",
    "Rollout", "RolloutAttempt",
    "ROLLOUT_STATUSES", "ROLLOUT_ATTEMPT_STATUSES", "ROLLOUT_TRIGGERS",
    "StreamingCredential", "CredentialAccessLog",
    "DayChat", "ContextBudgetLog", "MigrationStatus",
    "DeletionAuditEvent", "SensitiveActionRedemption",
    "DELETION_STATUSES", "DELETION_ACTORS",
    "AgentSecurityEvent",
    "EVENT_AGENT_KEY_ROTATED", "EVENT_AGENT_KEY_ROTATION_FAILED",
    "ConnectorIdentity", "ConnectorOAuthSession", "ConnectorEvent",
    "ConnectorUserPreference", "ProviderAppCredential",
    "CONNECTOR_IDENTITY_STATUSES",
    "EVENT_CONNECTED", "EVENT_DISCONNECTED",
    "EVENT_REAUTH_STARTED", "EVENT_REAUTH_COMPLETED",
    "EVENT_REFRESH_SUCCEEDED", "EVENT_REFRESH_FAILED",
    "EVENT_TOOL_CALLED", "EVENT_TOOL_SUCCEEDED", "EVENT_TOOL_FAILED",
    "EVENT_TOOL_ELEVATION_REQUIRED",
    "EVENT_REVOCATION_PROVIDER_SUCCEEDED", "EVENT_REVOCATION_PROVIDER_FAILED",
    "EVENT_HEALTH_PROBE_SUCCEEDED", "EVENT_HEALTH_PROBE_FAILED",
    "EVENT_FORCE_QUARANTINED", "EVENT_FORCE_RELEASED",
    "SubscriptionPlan", "CreditBalance", "CreditLedger", "CreditReservation",
    "AppleSubscription",
    "APPLE_SUB_ACTIVE", "APPLE_SUB_EXPIRED", "APPLE_SUB_BILLING_RETRY",
    "APPLE_SUB_GRACE", "APPLE_SUB_REVOKED",
]
