"""All enum types used across Toup models."""

from enum import Enum


class BrainType(str, Enum):
    """Types of brains in the Toup system"""
    USER = "user"       # User's personal memories
    AGENT = "agent"     # Agent's learned knowledge
    WORK = "work"       # Operational processes


class MemoryCategory(str, Enum):
    """Memory categories for USER brain - organizing personal information"""
    # Personal Profile
    IDENTITY = "identity"           # Who the user is - name, age, background
    PREFERENCES = "preferences"     # Likes, dislikes, favorites
    BELIEFS = "beliefs"             # Values, opinions, worldview
    EMOTIONS = "emotions"           # Emotional states, moods, feelings

    # Relationships
    PEOPLE = "people"               # Friends, colleagues, contacts
    RELATIONSHIPS = "relationships" # How people are connected

    # Life & Work
    EXPERIENCES = "experiences"     # Past events, stories, life history
    GOALS = "goals"                 # Ambitions, plans, desires
    HABITS = "habits"               # Routines, patterns, behaviors
    SKILLS = "skills"               # Abilities, education, expertise
    WORK = "work"                   # Job, career, professional info
    FINANCE = "finance"             # Financial info, budgets, accounts
    HEALTH = "health"               # Medical, fitness, wellness

    # Knowledge & Environment
    KNOWLEDGE = "knowledge"         # Facts, information, learnings
    LOCATIONS = "locations"         # Places, addresses, geography
    POSSESSIONS = "possessions"     # Things owned, subscriptions, accounts
    MEDIA = "media"                 # Books, movies, music, content consumed

    # System
    INTERACTION = "interaction"     # How the user prefers to interact
    OTHER = "other"                 # Uncategorized


# Backwards compatibility alias
BrainRegion = MemoryCategory


class WorkCategory(str, Enum):
    """Memory categories for WORK brain - operational knowledge"""
    PROCESS = "process"         # Business processes


class AgentCategory(str, Enum):
    """Memory categories for AGENT brain - agent's learned knowledge"""
    TOOL_USAGE = "tool_usage"
    USER_PATTERNS = "user_patterns"
    CORRECTIONS = "corrections"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CONVERSATION_STYLE = "conversation_style"
    PREFERENCES = "preferences"
    SKILLS_LEARNED = "skills_learned"


class MemoryType(str, Enum):
    """What kind of information a memory captures"""
    FACT = "fact"               # Concrete fact
    OPINION = "opinion"         # Subjective view
    EVENT = "event"             # Something that happened
    TASK = "task"               # To-do or action item
    ENTITY = "entity"           # Person, place, or thing


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


class MemoryLevel(str, Enum):
    """
    Cognitive hierarchy of memory levels.
    Based on cognitive science: episodic -> semantic consolidation.
    """
    EPISODIC = "episodic"       # Specific experiences with time/place context
    SEMANTIC = "semantic"       # General facts/knowledge (consolidated)
    PROCEDURAL = "procedural"   # How-to knowledge, skills, procedures
    META = "meta"               # Knowledge about knowledge (metacognition)


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
