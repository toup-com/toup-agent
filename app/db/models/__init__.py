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
from .media_playlist import MediaPlaylist

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
from .platform import VPSPlan, VPSInstance, ManagedContainer, Invite, LLMBundleAllocation, LLMUsageRecord, LLMProxyEvent, SearchEvent, PlatformSetting

# Maintenance / support agent (platform-only — admin maintenance system)
from .support import SupportIssue, SupportIssueEvent, SupportAttachment

# Agent-side notification outbox (agent-only — Autopilot arc PR4)
from .agent_outbox import AgentNotifyOutbox
from .memory_capture_outbox import MemoryCaptureOutbox

# Autopilot approvals (agent-only — Autopilot arc PR7)
from .autopilot import (
    AutopilotApproval,
    APPROVAL_KIND_QUESTION, APPROVAL_KIND_APPROVAL,
    APPROVAL_PENDING, APPROVAL_APPROVED, APPROVAL_DENIED,
    APPROVAL_ANSWERED, APPROVAL_CANCELLED, TERMINAL_APPROVAL_STATUSES,
)

# Proactive notifications (platform-only — Autopilot arc PR2)
from .notification import (
    PushDevice, NotificationQueue,
    KNOWN_NOTIFY_KINDS, KNOWN_NQ_PRIORITIES,
    NOTIFY_KIND_NEEDS_INPUT, NOTIFY_KIND_NEEDS_APPROVAL,
    NOTIFY_KIND_MISSION_STARTED,
    NOTIFY_KIND_MISSION_COMPLETED, NOTIFY_KIND_MISSION_FAILED,
    NOTIFY_KIND_PROGRESS, NOTIFY_KIND_DIGEST, NOTIFY_KIND_GENERIC,
    NOTIFY_KIND_ANNOUNCEMENT,
    NQ_QUEUED, NQ_SENDING, NQ_SENT, NQ_SUPPRESSED, NQ_FAILED, NQ_EXPIRED,
    NQ_PRIORITY_HIGH, NQ_PRIORITY_DEFAULT, NQ_PRIORITY_LOW,
)

# Admin dispatch (platform-only — operator→user announcements + thread)
from .admin_dispatch import (
    AdminDispatch, AdminDispatchTarget, AdminThreadMessage,
    DISPATCH_MODE_ONCE, DISPATCH_MODE_PERSISTENT, DISPATCH_MODES,
    DISPATCH_AUDIENCE_USER, DISPATCH_AUDIENCE_ALL, DISPATCH_AUDIENCES,
    DISPATCH_QUEUED, DISPATCH_SENDING, DISPATCH_SENT, DISPATCH_FAILED,
    DISPATCH_STATUSES,
    TARGET_PENDING, TARGET_SENDING, TARGET_DONE, TARGET_FAILED, TARGET_STATES,
    CHAT_PENDING, CHAT_DELIVERED, CHAT_NO_AGENT, CHAT_FAILED, CHAT_RETRACTED,
    CHAT_STATUSES,
    THREAD_OUT, THREAD_IN, THREAD_DIRECTIONS,
)

# iOS Live Activities (platform-only — Autopilot phone surface)
from .live_activity import (
    LiveActivityDevice, LiveActivity,
    LA_STARTED, LA_ENDED, LA_FAILED,
    APNS_ENV_DEVELOPMENT, APNS_ENV_PRODUCTION, KNOWN_APNS_ENVIRONMENTS,
)

# Credits (free-for-everyone billing — docs/credits/design.md)
from .credit import (
    SubscriptionPlan, CreditBalance, CreditLedger, CreditReservation,
    AppleSubscription,
    APPLE_SUB_ACTIVE, APPLE_SUB_EXPIRED, APPLE_SUB_BILLING_RETRY,
    APPLE_SUB_GRACE, APPLE_SUB_REVOKED,
    BUCKET_MESSAGE, BUCKET_INTEGRATION,
    LEDGER_CHAT_MESSAGE, LEDGER_ROUTINE_RUN, LEDGER_TRIGGER_RUN, LEDGER_BUILD_STEP,
    LEDGER_TOOL_CALL, LEDGER_BROWSER_ACTION, LEDGER_DOC_GEN, LEDGER_IMAGE_GEN,
    LEDGER_RESERVATION, LEDGER_SETTLEMENT, LEDGER_REFUND, LEDGER_PLAN_GRANT,
    LEDGER_PLAN_CHANGE, LEDGER_DAILY_RESET, LEDGER_PERIOD_RENEWAL, LEDGER_MANUAL_ADJUST,
    LEDGER_IAP_PURCHASE,
    RESERVATION_OPEN, RESERVATION_SETTLED, RESERVATION_REFUNDED, RESERVATION_EXPIRED,
)

# Free-credit grant eligibility tombstone (Sybil / multi-account resistance)
from .grant_eligibility import GrantEligibility

# Durable signup-attempt log (persistent IP rate limiting)
from .signup_attempt import SignupAttempt

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
    ConnectorPendingAction,
    ProviderAppCredential,
    CONNECTOR_IDENTITY_STATUSES,
    PENDING_ACTION_STATUSES,
    PENDING_ACTION_TERMINAL,
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
    "MediaPlaylist",
    "Entity", "EntityLink", "EntityRelationship",
    "Document", "DocumentChunk", "Media",
    "App", "BuildJob", "BuildUsage", "ReconciliationLog",
    "CronJob", "TelegramUserMapping", "AgentError", "ApiKey", "AgentConfig", "ExtensionDevice",
    "Routine", "RoutineRun", "RoutineNotificationDedupe",
    "Trigger", "TriggerEvent",
    "TRIGGER_KINDS", "TRIGGER_ACTIONS", "TRIGGER_STATUSES", "TRIGGER_EVENT_STATUSES",
    "SoulConfig",
    "VPSPlan", "VPSInstance", "ManagedContainer", "Invite", "LLMBundleAllocation", "LLMUsageRecord", "LLMProxyEvent", "SearchEvent", "PlatformSetting",
    "Rollout", "RolloutAttempt",
    "ROLLOUT_STATUSES", "ROLLOUT_ATTEMPT_STATUSES", "ROLLOUT_TRIGGERS",
    "StreamingCredential", "CredentialAccessLog",
    "GrantEligibility", "SignupAttempt",
    "DayChat", "ContextBudgetLog", "MigrationStatus",
    "DeletionAuditEvent", "SensitiveActionRedemption",
    "DELETION_STATUSES", "DELETION_ACTORS",
    "AgentSecurityEvent",
    "EVENT_AGENT_KEY_ROTATED", "EVENT_AGENT_KEY_ROTATION_FAILED",
    "ConnectorIdentity", "ConnectorOAuthSession", "ConnectorEvent",
    "ConnectorUserPreference", "ConnectorPendingAction", "ProviderAppCredential",
    "PENDING_ACTION_STATUSES", "PENDING_ACTION_TERMINAL",
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
    # Maintenance / support agent
    "SupportIssue", "SupportIssueEvent", "SupportAttachment",
    # Agent-side notification outbox (Autopilot arc PR4)
    "AgentNotifyOutbox",
    "MemoryCaptureOutbox",
    # Autopilot approvals (Autopilot arc PR7)
    "AutopilotApproval",
    "APPROVAL_KIND_QUESTION", "APPROVAL_KIND_APPROVAL",
    "APPROVAL_PENDING", "APPROVAL_APPROVED", "APPROVAL_DENIED",
    "APPROVAL_ANSWERED", "APPROVAL_CANCELLED", "TERMINAL_APPROVAL_STATUSES",
    # Proactive notifications (Autopilot arc PR2)
    "PushDevice", "NotificationQueue",
    "KNOWN_NOTIFY_KINDS", "KNOWN_NQ_PRIORITIES",
    "NOTIFY_KIND_NEEDS_INPUT", "NOTIFY_KIND_NEEDS_APPROVAL",
    "NOTIFY_KIND_MISSION_STARTED",
    "NOTIFY_KIND_MISSION_COMPLETED", "NOTIFY_KIND_MISSION_FAILED",
    "NOTIFY_KIND_PROGRESS", "NOTIFY_KIND_DIGEST", "NOTIFY_KIND_GENERIC",
    "NOTIFY_KIND_ANNOUNCEMENT",
    "NQ_QUEUED", "NQ_SENDING", "NQ_SENT", "NQ_SUPPRESSED", "NQ_FAILED",
    "NQ_EXPIRED",
    "NQ_PRIORITY_HIGH", "NQ_PRIORITY_DEFAULT", "NQ_PRIORITY_LOW",
    # Admin dispatch (operator→user announcements + thread)
    "AdminDispatch", "AdminDispatchTarget", "AdminThreadMessage",
    "DISPATCH_MODE_ONCE", "DISPATCH_MODE_PERSISTENT", "DISPATCH_MODES",
    "DISPATCH_AUDIENCE_USER", "DISPATCH_AUDIENCE_ALL", "DISPATCH_AUDIENCES",
    "DISPATCH_QUEUED", "DISPATCH_SENDING", "DISPATCH_SENT", "DISPATCH_FAILED",
    "DISPATCH_STATUSES",
    "TARGET_PENDING", "TARGET_SENDING", "TARGET_DONE", "TARGET_FAILED",
    "TARGET_STATES",
    "CHAT_PENDING", "CHAT_DELIVERED", "CHAT_NO_AGENT", "CHAT_FAILED",
    "CHAT_RETRACTED", "CHAT_STATUSES",
    "THREAD_OUT", "THREAD_IN", "THREAD_DIRECTIONS",
    # iOS Live Activities (Autopilot phone surface)
    "LiveActivityDevice", "LiveActivity",
    "LA_STARTED", "LA_ENDED", "LA_FAILED",
    "APNS_ENV_DEVELOPMENT", "APNS_ENV_PRODUCTION", "KNOWN_APNS_ENVIRONMENTS",
]
