from app.db.models import (
    Base, User, UserRole, Conversation, Message, Memory, Entity, EntityLink, EntityRelationship, BrainStats,
    MemoryCategory, BrainRegion, MemoryType, MemoryLevel, MemoryEventType, MemoryEvent,
    memory_relationships,
    # Document & Media models
    Document, DocumentChunk, Media, DocumentType, MediaType,
    # Identity system
    Identity, IdentityType,
    # Agent platform
    CronJob, TelegramUserMapping, AgentError, ApiKey,
    # Beta access
    Invite,
    # VPS provisioning
    VPSPlan, VPSInstance,
    # Agent setup
    AgentConfig,
    LLMBundleAllocation,
    LLMUsageRecord,
)
from app.db.database import get_db, init_db, drop_db, async_session_maker, engine

__all__ = [
    "Base",
    "User",
    "Conversation", 
    "Message",
    "Memory",
    "Entity",
    "EntityLink",
    "EntityRelationship",
    "BrainStats",
    "MemoryCategory",
    "BrainRegion",  # backwards compatibility
    "MemoryType",
    "MemoryLevel",  # NEW: cognitive hierarchy
    "MemoryEventType",  # NEW: audit event types
    "MemoryEvent",  # NEW: immutable audit log
    "memory_relationships",
    # Document & Media
    "Document",
    "DocumentChunk",
    "Media",
    "DocumentType",
    "MediaType",
    # Identity system
    "Identity",
    "IdentityType",
    # Agent platform
    "CronJob",
    "TelegramUserMapping",
    "AgentError",
    "ApiKey",
    # Beta access
    "UserRole",
    "Invite",
    # VPS provisioning
    "VPSPlan",
    "VPSInstance",
    # Agent setup
    "AgentConfig",
    "LLMBundleAllocation",
    "LLMUsageRecord",
    # Database
    "get_db",
    "init_db",
    "drop_db",
    "async_session_maker",
    "engine",
]
