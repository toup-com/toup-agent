"""All enum types used across Toup models.

The memory taxonomy (BrainType / MemoryCategory / AgentCategory /
WorkCategory / MemoryType / MemoryLevel) is NOT declared here — it is
re-exported from `app.memory_taxonomy`, which is the single source of truth.

Before 2026-07-29 this module and `app/schemas.py` each declared their own
copy with divergent values, so the extractor wrote categories the app had no
label for. Import from either place; there is now only one definition.
"""

from enum import Enum

from app.memory_taxonomy import (  # noqa: F401  (re-exported)
    BrainType,
    MemoryCategory,
    BrainRegion,
    WorkCategory,
    AgentCategory,
    MemoryType,
    MemoryLevel,
    normalize_category,
    normalize_memory_type,
)


class EntityType(str, Enum):
    """Types of named entities"""
    PERSON = "person"           # Person
    ORGANIZATION = "organization"  # Organization
    PLACE = "place"             # Location
    PROJECT = "project"         # Project or initiative
    DECISION = "decision"       # Decision made
    SKILL = "skill"             # Learned skill or procedure
    FILE = "file"               # Document or file reference
    NOTE = "note"               # General note
    CONVERSATION = "conversation"  # Conversation summary


class MemoryEventType(str, Enum):
    """
    Types of events that can occur to a memory.
    Used for immutable audit log tracking.
    """
    CREATED = "created"
    ACCESSED = "accessed"
    REINFORCED = "reinforced"
    DECAYED = "decayed"
    CONSOLIDATED = "consolidated"
    UPDATED = "updated"
    DELETED = "deleted"
    LINKED = "linked"
    UNLINKED = "unlinked"


class UserRole(str, Enum):
    """User roles for access control"""
    ADMIN = "admin"
    BETA_USER = "beta_user"


class IdentityType(str, Enum):
    """Types of identity documents"""
    SOUL = "soul"
    USER_PROFILE = "user_profile"
    AGENT_INSTRUCTIONS = "agent_instructions"
    TOOLS = "tools"
    SYSTEM = "system"
    CONTEXT = "context"


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    CODE = "code"
    DOCX = "docx"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"


class MediaType(str, Enum):
    """Supported media types"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
