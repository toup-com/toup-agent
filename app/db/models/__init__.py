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

# Memory system
from .memory import Memory, memory_relationships, BrainStats, MemoryEvent, RetrievalEvent

# Entity & Knowledge Graph
from .entity import Entity, EntityLink, EntityRelationship

# Documents & Media
from .document import Document, DocumentChunk, Media

# Apps & Build Jobs
from .app import App, BuildJob

# Agent runtime
from .agent import CronJob, TelegramUserMapping, AgentError, ApiKey, AgentConfig

# Soul config
from .soul_config import SoulConfig

# Platform (VPS, invites, billing)
from .platform import VPSPlan, VPSInstance, Invite, LLMBundleAllocation, LLMUsageRecord

# Streaming credentials
from .streaming import StreamingCredential

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
    "App", "BuildJob",
    "CronJob", "TelegramUserMapping", "AgentError", "ApiKey", "AgentConfig",
    "SoulConfig",
    "VPSPlan", "VPSInstance", "Invite", "LLMBundleAllocation", "LLMUsageRecord",
    "StreamingCredential",
]
