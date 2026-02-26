"""Pydantic schemas for API request/response validation"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class BrainType(str, Enum):
    """Types of brains in the HexBrain system"""
    USER = "user"       # User's personal memories
    AGENT = "agent"     # Hex agent's learned knowledge
    WORK = "work"       # Workflows and operational processes


class MemoryCategory(str, Enum):
    """Memory categories for USER brain - organizing personal information"""
    # Personal Profile
    IDENTITY = "identity"           # Who the user is - name, age, background
    PREFERENCES = "preferences"     # Likes, dislikes, favorites
    BELIEFS = "beliefs"             # Values, opinions, worldview
    EMOTIONS = "emotions"           # Emotional states, moods, feelings
    
    # Relationships
    PEOPLE = "people"               # Friends, colleagues, contacts
    PLACES = "places"               # Locations, addresses, venues
    FAMILY = "family"               # Family members, relationships
    
    # Experiences
    EXPERIENCES = "experiences"     # Past events, memories, stories
    PROJECTS = "projects"           # Active or past projects
    SCHEDULE = "schedule"           # Appointments, reminders, calendar
    WORK = "work"                   # Job, career, professional
    
    # Knowledge
    LEARNING = "learning"           # Skills being learned, courses
    KNOWLEDGE = "knowledge"         # Facts, information, expertise
    TOOLS = "tools"                 # Software, tools, configurations
    MEDIA = "media"                 # Books, movies, music, articles
    
    # Lifestyle
    HEALTH = "health"               # Medical, fitness, wellness
    HABITS = "habits"               # Routines, habits, rituals
    FOOD = "food"                   # Diet, recipes, restaurants
    TRAVEL = "travel"               # Trips, destinations, travel plans
    GOALS = "goals"                 # Objectives, aspirations, dreams
    CONTEXT = "context"             # Conversation context, general


class AgentCategory(str, Enum):
    """Memory categories for AGENT brain - Hex's learned capabilities"""
    AGENT_TOOLS = "agent_tools"           # 🛠️ Commands, APIs, skills Hex knows
    AGENT_SKILLS = "agent_skills"         # 🎯 Learned capabilities and competencies
    AGENT_SOUL = "agent_soul"             # 💫 Personality, values, communication style
    AGENT_PROCEDURES = "agent_procedures" # 📋 How to do things, step-by-step guides
    AGENT_PATTERNS = "agent_patterns"     # 🧠 Learned behaviors, recognition patterns
    AGENT_DECISIONS = "agent_decisions"   # 📝 Past choices made, decision history


class WorkCategory(str, Enum):
    """Memory categories for WORK brain - workflows and operations"""
    WORKFLOW = "workflow"           # 🔄 Multi-step processes
    PROCESS = "process"             # ⚡ Business processes


# Alias for backward compatibility
BrainRegion = MemoryCategory


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    EVENT = "event"
    PERSON = "person"
    PLACE = "place"
    PROJECT = "project"
    DECISION = "decision"
    SKILL = "skill"
    FILE = "file"
    NOTE = "note"
    CONVERSATION = "conversation"


class MemoryLevel(str, Enum):
    """
    Cognitive hierarchy of memory levels.
    Based on cognitive science: episodic → semantic consolidation.
    """
    EPISODIC = "episodic"       # Specific experiences with time/place context
    SEMANTIC = "semantic"       # General facts/knowledge (consolidated)
    PROCEDURAL = "procedural"   # How-to knowledge, skills, procedures
    META = "meta"               # Knowledge about knowledge (metacognition)


class MemoryEventType(str, Enum):
    """
    Types of events in the memory audit log.
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


# ============ User Schemas ============

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: Optional[str] = None


class UserLogin(BaseModel):
    email: str  # Accepts email or username
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str = "beta_user"
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    exp: datetime


# ============ Memory Schemas ============

class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    summary: Optional[str] = Field(None, max_length=500)
    brain_type: BrainType = BrainType.USER  # Which brain this belongs to
    category: str  # Category depends on brain_type (MemoryCategory, AgentCategory, or WorkCategory)
    memory_type: MemoryType
    importance: float = Field(0.5, ge=0, le=1)
    confidence: float = Field(1.0, ge=0, le=1)
    # NEW: Memory enhancement fields
    memory_level: MemoryLevel = MemoryLevel.EPISODIC  # Default to episodic
    emotional_salience: float = Field(0.5, ge=0, le=1)  # Emotional weight
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    related_memory_ids: Optional[List[str]] = None
    source_type: Optional[str] = None  # conversation, import, manual


class MemoryUpdate(BaseModel):
    content: Optional[str] = Field(None, max_length=10000)
    summary: Optional[str] = Field(None, max_length=500)
    brain_type: Optional[BrainType] = None
    category: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    importance: Optional[float] = Field(None, ge=0, le=1)
    # NEW: Memory enhancement fields
    memory_level: Optional[MemoryLevel] = None
    emotional_salience: Optional[float] = Field(None, ge=0, le=1)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    id: str
    content: str
    canonical_content: Optional[str] = None  # Current truth (most up-to-date version)
    summary: Optional[str]
    brain_type: str = "user"  # BrainType enum value
    category: str
    memory_type: str
    importance: float
    confidence: float
    # NEW: Memory enhancement fields
    strength: float = 1.0  # Memory strength (0-1)
    memory_level: str = "episodic"  # MemoryLevel enum value
    emotional_salience: float = 0.5  # Emotional weight
    last_reinforced_at: Optional[datetime] = None
    consolidation_count: int = 0
    decay_rate: float = 0.1
    # Memory Evolution fields
    history: Optional[List[Dict[str, Any]]] = None  # Version history
    merged_from: Optional[List[str]] = None  # IDs of memories merged into this
    superseded_by: Optional[str] = None  # ID of memory this was merged into
    is_active: bool = True  # False if superseded/archived
    # Timestamps
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime]
    access_count: int
    source_type: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def current_content(self) -> str:
        """Get the most current version of this memory."""
        return self.canonical_content or self.content
    
    @property
    def version_count(self) -> int:
        """Number of versions in history."""
        return len(self.history) if self.history else 1

    class Config:
        from_attributes = True


class MemoryWithScore(MemoryResponse):
    """Memory with similarity score for search results"""
    similarity_score: float
    explanation: Optional[str] = None  # Why this was retrieved


class MemoryWithRelations(MemoryResponse):
    """Memory with related memories and entities"""
    related_memories: List["MemoryResponse"] = []
    entities: List["EntityResponse"] = []


# ============ Search Schemas ============

class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    brain_type: Optional[BrainType] = None  # Filter by brain type
    categories: Optional[List[str]] = None  # Categories depend on brain_type
    memory_types: Optional[List[MemoryType]] = None
    memory_levels: Optional[List[MemoryLevel]] = None  # NEW: Filter by cognitive level
    tags: Optional[List[str]] = None
    min_importance: Optional[float] = Field(None, ge=0, le=1)
    min_similarity: float = Field(0.1, ge=0, le=1)  # Minimum similarity threshold for results
    min_strength: Optional[float] = Field(None, ge=0, le=1)  # NEW: Filter by memory strength
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(10, ge=1, le=100)
    include_explanation: bool = False
    # NEW: Weighted search options
    use_weighted_scoring: bool = True  # Use decay-aware scoring
    boost_recent_access: bool = True  # Boost recently accessed memories
    # Hybrid retrieval strategies
    strategies: Optional[List[str]] = None  # ["vector", "keyword", "graph"] — None = all


class MemorySearchResponse(BaseModel):
    query: str
    results: List[MemoryWithScore]
    total_count: int
    search_time_ms: float


# ============ Entity Schemas ============

class EntityCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    entity_type: str = Field(max_length=50)
    description: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None


class EntityResponse(BaseModel):
    id: str
    name: str
    entity_type: str
    description: Optional[str]
    mention_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    attributes: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


# ============ Entity Graph Schemas ============

class EntityBriefResponse(BaseModel):
    """Minimal entity info for graph edges"""
    id: str
    name: str
    entity_type: str


class EntityRelationshipResponse(BaseModel):
    """A single entity-to-entity relationship (knowledge graph edge)"""
    id: str
    relationship_type: str
    relationship_label: Optional[str] = None
    confidence: float
    mention_count: int
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    properties: Optional[Dict[str, Any]] = None
    source_entity: Optional[EntityBriefResponse] = None
    target_entity: Optional[EntityBriefResponse] = None


class GraphTraversalNode(BaseModel):
    """A node discovered during graph traversal"""
    entity_id: str
    entity_name: str
    entity_type: str
    depth: int
    relationship_type: Optional[str] = None
    relationship_label: Optional[str] = None
    from_entity_id: Optional[str] = None
    from_entity_name: Optional[str] = None


class GraphTraversalRequest(BaseModel):
    """Request for recursive graph traversal from seed entities"""
    entity_names: Optional[List[str]] = None
    entity_ids: Optional[List[str]] = None
    max_depth: int = Field(2, ge=1, le=5)
    limit: int = Field(50, ge=1, le=200)
    include_memories: bool = False


class GraphTraversalResponse(BaseModel):
    """Response from graph traversal"""
    seed_entities: List[EntityBriefResponse]
    nodes: List[GraphTraversalNode]
    relationships: List[EntityRelationshipResponse] = []
    total_entities: int
    total_relationships: int
    memories: Optional[List["MemoryResponse"]] = None


class GraphExplorationResponse(BaseModel):
    """Response for graph exploration endpoints"""
    entities: List[EntityResponse] = []
    relationships: List[EntityRelationshipResponse] = []
    total_entities: int
    total_relationships: int


# ============ Memory Event Schemas (Audit Log) ============

class MemoryEventResponse(BaseModel):
    """Response for memory audit log events"""
    id: str
    memory_id: str
    event_type: str  # MemoryEventType enum value
    timestamp: datetime
    event_data: Optional[Dict[str, Any]] = None
    trigger_source: Optional[str] = None

    class Config:
        from_attributes = True


class MemoryEventsResponse(BaseModel):
    """Response for listing memory events"""
    memory_id: str
    events: List[MemoryEventResponse]
    total_count: int

    class Config:
        from_attributes = True


# ============ Conversation Schemas ============

class MessageCreate(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    message_count: int

    class Config:
        from_attributes = True


# ============ Ingestion Schemas ============

class IngestMessageRequest(BaseModel):
    """Request to ingest a single message turn"""
    conversation_id: Optional[str] = None  # Create new if not provided
    user_message: str
    assistant_response: str
    extract_memories: bool = True


class IngestConversationRequest(BaseModel):
    """Request to ingest a full conversation"""
    messages: List[MessageCreate]
    title: Optional[str] = None
    extract_memories: bool = True


class IngestResponse(BaseModel):
    conversation_id: str
    messages_ingested: int
    memories_extracted: int
    entities_extracted: int
    memories: List[MemoryResponse] = []


# ============ Agent API Schemas ============

class AgentStoreRequest(BaseModel):
    """Request from agent to store memories"""
    memories: List[MemoryCreate]
    conversation_id: Optional[str] = None


class AgentRecallRequest(BaseModel):
    """Request from agent to recall relevant memories"""
    query: str
    context: Optional[str] = None  # Additional context
    categories: Optional[List[MemoryCategory]] = None
    limit: int = Field(5, ge=1, le=20)
    min_similarity: float = Field(0.5, ge=0, le=1)


class AgentRecallResponse(BaseModel):
    memories: List[MemoryWithScore]
    context_summary: Optional[str] = None


class AgentGraphRequest(BaseModel):
    """Request for graph traversal"""
    memory_id: str
    depth: int = Field(2, ge=1, le=5)
    include_entities: bool = True


class AgentGraphResponse(BaseModel):
    root: MemoryResponse
    related: List[MemoryWithRelations]
    entities: List[EntityResponse]


# ============ Stats Schemas ============

class CategoryStats(BaseModel):
    category: str
    count: int
    size: float  # Normalized size for visualization


# Alias for backward compatibility
RegionStats = CategoryStats


class BrainStatsResponse(BaseModel):
    total_memories: int
    total_entities: int
    total_connections: int
    categories: List[CategoryStats]
    updated_at: datetime


class TimelineEntry(BaseModel):
    date: str
    count: int
    categories: Dict[str, int]


class TimelineResponse(BaseModel):
    entries: List[TimelineEntry]
    start_date: str
    end_date: str


class ConnectionData(BaseModel):
    source_id: str
    target_id: str
    strength: float
    type: str


class ConnectionsResponse(BaseModel):
    nodes: List[MemoryResponse]
    connections: List[ConnectionData]


# ============ Document & Media Schemas ============

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


class DocumentChunk(BaseModel):
    """A chunk of document content for embedding"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Any]] = None


class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents"""
    filename: str
    file_type: DocumentType
    file_size: int
    mime_type: str
    page_count: Optional[int] = None  # For PDFs
    word_count: Optional[int] = None
    language: Optional[str] = None
    encoding: Optional[str] = None
    # Code-specific
    programming_language: Optional[str] = None
    # Additional metadata
    extra: Optional[Dict[str, Any]] = None


class MediaMetadata(BaseModel):
    """Metadata for uploaded media files"""
    filename: str
    media_type: MediaType
    file_size: int
    mime_type: str
    # Image specific
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    # Video/Audio specific
    duration: Optional[float] = None  # seconds
    # AI-generated
    ai_description: Optional[str] = None
    ai_transcript: Optional[str] = None
    ai_tags: Optional[List[str]] = None
    # Additional metadata
    extra: Optional[Dict[str, Any]] = None


class DocumentUploadRequest(BaseModel):
    """Request for document upload metadata"""
    brain_type: BrainType = BrainType.USER
    category: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: float = Field(0.5, ge=0, le=1)
    # Processing options
    chunk_size: int = Field(1000, ge=100, le=5000)
    chunk_overlap: int = Field(200, ge=0, le=500)
    extract_entities: bool = True
    generate_summary: bool = True


class DocumentResponse(BaseModel):
    """Response after document upload"""
    id: str
    filename: str
    file_type: str
    file_size: int
    brain_type: str
    category: str
    title: Optional[str]
    description: Optional[str]
    # Processing results
    chunk_count: int
    memories_created: int
    entities_extracted: int
    # AI-generated
    summary: Optional[str] = None
    key_topics: Optional[List[str]] = None
    # Timestamps
    created_at: datetime
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class MediaUploadRequest(BaseModel):
    """Request for media upload metadata"""
    brain_type: BrainType = BrainType.USER
    category: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: float = Field(0.5, ge=0, le=1)
    # Processing options
    generate_description: bool = True  # Use AI to describe image
    transcribe_audio: bool = True  # Transcribe video/audio


class MediaResponse(BaseModel):
    """Response after media upload"""
    id: str
    filename: str
    media_type: str
    file_size: int
    brain_type: str
    category: str
    title: Optional[str]
    description: Optional[str]
    # Media info
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    # AI-generated
    ai_description: Optional[str] = None
    ai_transcript: Optional[str] = None
    # Processing results
    memories_created: int
    # Timestamps
    created_at: datetime
    processed_at: Optional[datetime] = None
    # URL to access the file
    file_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    
    class Config:
        from_attributes = True


class IngestDocumentResponse(BaseModel):
    """Response from document ingestion"""
    document_id: str
    filename: str
    file_type: str
    chunks_processed: int
    memories_created: int
    entities_extracted: int
    summary: Optional[str] = None
    key_topics: Optional[List[str]] = None
    processing_time_ms: int


class IngestMediaResponse(BaseModel):
    """Response from media ingestion"""
    media_id: str
    filename: str
    media_type: str
    memories_created: int
    ai_description: Optional[str] = None
    ai_transcript: Optional[str] = None
    processing_time_ms: int


# ============ Identity System Schemas ============

class IdentityType(str, Enum):
    """Types of identity documents that define agent behavior"""
    SOUL = "soul"                       # Core personality, values, communication style
    USER_PROFILE = "user_profile"       # Information about the user being served
    AGENT_INSTRUCTIONS = "agent_instructions"  # Specific behavioral instructions
    TOOLS = "tools"                     # Available tools/capabilities description
    CONTEXT = "context"                 # Dynamic context (e.g., current project)


class IdentityCreate(BaseModel):
    """Request to create an identity document"""
    identity_type: IdentityType
    name: str = Field(min_length=1, max_length=255, description="Human-readable name for this identity")
    content: str = Field(min_length=1, max_length=50000, description="The identity content/document")
    priority: int = Field(0, ge=0, le=100, description="Higher priority = loaded first in prompt")
    is_active: bool = Field(True, description="Whether this identity is currently active")


class IdentityUpdate(BaseModel):
    """Request to update an identity document"""
    name: Optional[str] = Field(None, max_length=255)
    content: Optional[str] = Field(None, max_length=50000)
    priority: Optional[int] = Field(None, ge=0, le=100)
    is_active: Optional[bool] = None


class IdentityResponse(BaseModel):
    """Response with identity details"""
    id: str
    user_id: str
    identity_type: str
    name: str
    content: str
    priority: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class IdentityListResponse(BaseModel):
    """Response with list of identities"""
    identities: List[IdentityResponse]
    total_count: int


# ============ Session Management Schemas ============

class SessionCreate(BaseModel):
    """Request to create a new conversation session"""
    title: Optional[str] = Field(None, max_length=500)
    channel: str = Field("api", description="Channel: api, telegram, discord, web")
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response with session details"""
    id: str
    user_id: str
    title: Optional[str]
    channel: str
    is_active: bool
    started_at: datetime
    ended_at: Optional[datetime]
    updated_at: datetime
    message_count: int
    total_tokens: int
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class SessionWithMessages(SessionResponse):
    """Session with full message history"""
    messages: List["ChatMessageResponse"]


class SessionListResponse(BaseModel):
    """Response with list of sessions"""
    sessions: List[SessionResponse]
    total_count: int


# ============ Chat Orchestration Schemas ============

class ChatRequest(BaseModel):
    """Main chat request - the core of the agent platform"""
    message: str = Field(min_length=1, max_length=10000, description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID (creates new if not provided)")
    
    # Memory retrieval options
    include_memories: bool = Field(True, description="Auto-retrieve relevant memories")
    memory_limit: int = Field(15, ge=0, le=50, description="Max memories to retrieve")
    min_similarity: float = Field(0.1, ge=0, le=1, description="Min similarity for memory retrieval")
    brain_types: Optional[List[BrainType]] = Field(None, description="Brain types to search")
    
    # Response options
    model: Optional[str] = Field(None, description="Override default model (e.g., 'gpt-4o', 'gpt-4o-mini')")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Override temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=8192, description="Override max tokens")
    stream: bool = Field(False, description="Stream the response")
    
    # Memory extraction options
    auto_extract_memories: bool = Field(True, description="Auto-extract memories from response")


class ChatMessageResponse(BaseModel):
    """Individual message in chat response"""
    id: str
    role: str
    content: str
    created_at: datetime
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    model_used: Optional[str] = None
    memories_retrieved: Optional[List[str]] = None
    processing_time_ms: Optional[int] = None

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    session_id: str
    message_id: str
    response: str
    
    # Token usage
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    
    # Processing info
    model_used: str
    processing_time_ms: int
    
    # Memory info
    memories_retrieved: List[MemoryWithScore] = []
    memories_extracted: List[MemoryResponse] = []
    
    # Session context
    is_new_session: bool
    session_message_count: int


class ChatStreamChunk(BaseModel):
    """Streaming chunk for chat response"""
    chunk_type: str  # "content", "memory", "done", "error"
    content: Optional[str] = None
    memory: Optional[MemoryWithScore] = None
    metadata: Optional[Dict[str, Any]] = None


# Forward references
MemoryWithRelations.model_rebuild()
SessionWithMessages.model_rebuild()
