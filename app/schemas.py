"""Pydantic schemas for API request/response validation"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


# ── Memory taxonomy ───────────────────────────────────────────────────
# Re-exported from app.memory_taxonomy, the single source of truth. This
# module used to declare its OWN copy of these enums with values that diverged
# from app/db/models/enums.py (places/family/projects/schedule/learning/tools/
# food/travel/context existed only here), which is why 28% of production rows
# carried a category the mobile app had no label for and rendered as "Other".
# Import them from here or from app.db.models.enums — same objects now.
from app.memory_taxonomy import (  # noqa: E402,F401  (re-exported)
    BrainType,
    MemoryCategory,
    AgentCategory,
    WorkCategory,
    MemoryType,
    MemoryLevel,
    normalize_category,
    normalize_memory_type,
)


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
    password: str = Field(min_length=8)
    name: Optional[str] = None
    # Cloudflare Turnstile token. Required in production (when
    # TURNSTILE_SECRET_KEY is set on the backend); ignored in dev.
    # Phase-3 of the prewarm hardening work — see
    # docs/onboarding/prewarm-phase0.md.
    cf_turnstile_token: Optional[str] = None


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
    # IANA tz name. Captured silently on every login via /auth/profile
    # (Intl-resolved) and overridable from the account page's "Share
    # precise location" flow. Exposed here so the account UI can render
    # the current value without a second round trip.
    timezone: Optional[str] = None
    # Email verification state (F13). NULL = unverified. Surfaced so the
    # frontend can render the verify-your-email banner without a second
    # round trip.
    email_verified_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    exp: datetime


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    # IANA tz name (e.g. "America/Toronto"). Frontend captures this via
    # `Intl.DateTimeFormat().resolvedOptions().timeZone` on app boot and
    # PATCHes it silently — no UI, no permission popup. Server validates
    # via zoneinfo.ZoneInfo and rejects unknown values.
    timezone: Optional[str] = Field(None, max_length=50)


class AdminResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=8)


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
    # Temporal validity. None = never expires (correct for durable facts and
    # for every manually-created memory). Set by the extractor for transient
    # content — reminders, one-off errands, passing status.
    expires_at: Optional[datetime] = None


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
    expires_at: Optional[datetime] = None  # None = never expires
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


class SessionMessageCreate(BaseModel):
    """Request to append one message to a session.

    The route also still reads these as query parameters, which is how every
    caller reached it before this model existed. Voice transcripts can be
    multi-kilobyte and non-Latin (a Farsi reply is 6 URL-encoded chars per
    character), which overruns the URL well before it overruns a body — so the
    body is the supported form and the query params are compatibility only.
    """
    role: Optional[str] = None
    content: Optional[str] = None
    model_used: Optional[str] = None
    day_chat_id: Optional[str] = None
    # Media the turn started, persisted into Message.metadata_json as
    # {"media": ...} — the exact shape AgentRunner._save_messages writes for a
    # chat turn and the only thing the mobile client renders a media card from.
    # Without it a voice play left nothing but plain text in the thread, so a
    # song the agent actually started looked, on reopening the app, like it had
    # never happened. Body-only (a nested object can't ride the query shim).
    media: Optional[dict] = None


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
    # Media metadata (YouTube/Netflix cards)
    media: Optional[dict] = None
    # A staged `elevation: true` connector call awaiting the user's
    # confirmation ({action_id, tool_name, summary, payload, ...}).
    # Sent so the card re-renders on reload — an approval prompt that
    # vanishes on refresh is worse than none, because the user assumes
    # it went through. Live status comes from
    # GET /api/connectors/pending-actions; this is the anchor.
    pending_action: Optional[dict] = None
    # Generated-file attachments (doc-delivery feature). List of
    # {id, filename, mime_type, size_bytes, created_at}. storage_path
    # is stripped server-side — it's an internal key.
    attachments: Optional[List[dict]] = None
    # Which surface this row came from ("voice", "telegram", "web", …).
    # api/day_chats.py has always sent this; these session routes did not, and
    # the mobile client falls back to /api/sessions/by-date/{date}/messages
    # whenever day-chats fails — so on that path every voice row arrived
    # indistinguishable from typed web chat and the day chat's voice-session
    # divider silently disappeared. Null when neither the conversation nor the
    # row records one; the client applies its own default rather than the API
    # guessing (see agent/channel_util.py).
    channel: Optional[str] = None
    # Job card fields (role == 'job')
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    job_status: Optional[str] = None
    job_step: Optional[str] = None
    job_total_steps: Optional[int] = None
    job_completed_steps: Optional[int] = None
    job_app_id: Optional[str] = None
    # Cross-channel reply-to (migration 049): soft pointer to a previous
    # message anywhere in this user's day. Frontend resolves the target
    # client-side from its in-memory day-chat list and renders a quoted
    # card on the bubble.
    reply_to_message_id: Optional[str] = None
    # Denormalized reply target — when reply_to_message_id is set, the
    # serializer inlines {id, role, content, created_at} of the target
    # so the frontend can render the quoted card on first paint instead
    # of waiting for older days to hydrate. None when the target is
    # unresolvable (deleted / column missing). content is excerpted to
    # 240 chars; see app/agent/reply_quote.py:serialize_reply_target.
    reply_to: Optional[dict] = None

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


# ============ Soul Config Schemas ============

VALID_SOUL_STYLES = {"casual", "professional", "mentor", "creative"}

VALID_SOUL_TRAITS = {
    # Tone
    "uses_humor", "stays_serious",
    "uses_emoji", "no_emoji",
    "concise", "detailed",
    "direct", "diplomatic",
    # Behavior
    "proactive", "reactive",
    "references_past", "fresh_each_time",
    "asks_questions", "assumes",
    "challenges", "supportive",
}


class SoulConfigUpdate(BaseModel):
    """Request to save/update soul configuration"""
    name: str = Field(min_length=1, max_length=50)
    color: str = Field(default="#9B59B6", pattern=r"^#[0-9A-Fa-f]{6}$")
    pronouns: str = Field(default="they")
    style: str = Field(default="casual")
    traits: List[str] = Field(default_factory=list)
    custom_instructions: str = Field(default="", max_length=500)
    # When true, save the soul without flipping `onboarding_completed=true`.
    # Required by the Phase-2 unified onboarding flow where Soul is the
    # FIRST step (not the last like in the legacy modal flow). Defaults
    # to false so legacy callers — `/agent/soul?onboarding=true` and
    # plain `/agent/soul` edit mode — keep their existing behaviour:
    # first save still terminates the post-install Soul prompt, just
    # like before.
    defer_onboarding_complete: bool = False


class SoulConfigResponse(BaseModel):
    """Response with soul configuration"""
    name: str
    color: str
    pronouns: str
    style: str
    traits: List[str]
    custom_instructions: str
    compiled_text: str
    updated_at: datetime

    class Config:
        from_attributes = True


class PlaylistTrack(BaseModel):
    """One entry of a saved Toup Media playlist (music only today)."""
    video_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    thumbnail_url: Optional[str] = None
    length: Optional[str] = None
    video_type: Optional[str] = None


class PlaylistCreate(BaseModel):
    """Create a playlist — either from explicit tracks or by capturing the
    live radio station on `from_channel` (exact tracks, exact order)."""
    name: Optional[str] = Field(default=None, max_length=200)
    media_type: str = Field(default="music", max_length=20)
    from_channel: Optional[str] = Field(default=None, max_length=16)
    tracks: Optional[List[PlaylistTrack]] = None


class PlaylistUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class PlaylistSummaryResponse(BaseModel):
    id: str
    name: str
    media_type: str
    source: str
    seed_intent: Optional[str] = None
    track_count: int
    artwork_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class PlaylistResponse(PlaylistSummaryResponse):
    tracks: List[PlaylistTrack]


class PlaylistPlayRequest(BaseModel):
    channel: str = Field(default="app", max_length=16)
    # Start from this track instead of the top, so tapping a song in the detail
    # view plays THAT song with the rest of the playlist behind it.
    start_video_id: Optional[str] = Field(default=None, max_length=32)


class PlaylistPlayResponse(BaseModel):
    ok: bool
    first: Optional[PlaylistTrack] = None
    upcoming: List[PlaylistTrack] = []


class LiveStationResponse(BaseModel):
    """Snapshot of the live radio station on a channel — what "Save this
    station" captures. active=False when nothing is playing."""
    active: bool
    seed_intent: Optional[str] = None
    current: Optional[PlaylistTrack] = None
    track_count: int = 0


# Forward references
MemoryWithRelations.model_rebuild()
SessionWithMessages.model_rebuild()
