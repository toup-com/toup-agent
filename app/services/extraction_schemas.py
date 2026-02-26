"""
Phase 4: Schema-enforced entity extraction.

Defines Pydantic schemas for 7 entity types. These schemas are included
in the LLM extraction prompt so that entities are returned with typed,
structured attributes instead of free-text.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ── Entity Schemas ──────────────────────────────────────────────────

class PersonEntity(BaseModel):
    """A person mentioned by the user."""
    name: str = Field(description="Full name of the person")
    relationship_to_user: Optional[str] = Field(None, description="e.g. friend, colleague, brother, manager")
    occupation: Optional[str] = None
    organization: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


class OrganizationEntity(BaseModel):
    """An organization, company, or institution."""
    name: str
    org_type: Optional[str] = None       # company, university, nonprofit, government
    industry: Optional[str] = None
    location: Optional[str] = None
    role: Optional[str] = None           # user's role within this org
    notes: Optional[str] = None


class ProjectEntity(BaseModel):
    """A project, app, or initiative the user is involved with."""
    name: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    status: Optional[str] = None         # active, completed, planned, paused
    url: Optional[str] = None
    role: Optional[str] = None           # user's role on this project
    notes: Optional[str] = None


class PlaceEntity(BaseModel):
    """A place or location."""
    name: str
    place_type: Optional[str] = None     # city, country, restaurant, office, park
    address: Optional[str] = None
    significance: Optional[str] = None   # "where I grew up", "favorite café"
    notes: Optional[str] = None


class EventEntity(BaseModel):
    """An event, meeting, or occasion."""
    name: str
    date: Optional[str] = None
    location: Optional[str] = None
    participants: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    notes: Optional[str] = None


class TopicEntity(BaseModel):
    """A topic, subject, or area of interest."""
    name: str
    domain: Optional[str] = None         # technology, science, art, finance, etc.
    user_interest_level: Optional[str] = None  # passionate, curious, learning, professional
    notes: Optional[str] = None


class ToolEntity(BaseModel):
    """A tool, software, or technology the user uses."""
    name: str
    tool_type: Optional[str] = None      # programming_language, framework, app, service
    proficiency: Optional[str] = None    # expert, intermediate, beginner
    usage: Optional[str] = None          # daily, weekly, occasional
    notes: Optional[str] = None


# ── Schema Registry ─────────────────────────────────────────────────

class EntitySchemaType(str, Enum):
    PERSON = "PersonEntity"
    ORGANIZATION = "OrganizationEntity"
    PROJECT = "ProjectEntity"
    PLACE = "PlaceEntity"
    EVENT = "EventEntity"
    TOPIC = "TopicEntity"
    TOOL = "ToolEntity"


ENTITY_SCHEMA_MAP: Dict[str, type] = {
    "PersonEntity": PersonEntity,
    "OrganizationEntity": OrganizationEntity,
    "ProjectEntity": ProjectEntity,
    "PlaceEntity": PlaceEntity,
    "EventEntity": EventEntity,
    "TopicEntity": TopicEntity,
    "ToolEntity": ToolEntity,
}

# Map loose entity_type strings → schema class name
ENTITY_TYPE_TO_SCHEMA: Dict[str, str] = {
    "person": "PersonEntity",
    "people": "PersonEntity",
    "organization": "OrganizationEntity",
    "company": "OrganizationEntity",
    "university": "OrganizationEntity",
    "institution": "OrganizationEntity",
    "project": "ProjectEntity",
    "app": "ProjectEntity",
    "application": "ProjectEntity",
    "place": "PlaceEntity",
    "location": "PlaceEntity",
    "city": "PlaceEntity",
    "country": "PlaceEntity",
    "restaurant": "PlaceEntity",
    "event": "EventEntity",
    "meeting": "EventEntity",
    "conference": "EventEntity",
    "topic": "TopicEntity",
    "subject": "TopicEntity",
    "interest": "TopicEntity",
    "tool": "ToolEntity",
    "technology": "ToolEntity",
    "framework": "ToolEntity",
    "language": "ToolEntity",
    "software": "ToolEntity",
}


def get_schema_for_type(entity_type: str) -> Optional[str]:
    """Map a loose entity type string to a schema class name."""
    return ENTITY_TYPE_TO_SCHEMA.get(entity_type.lower())


def validate_entity_data(schema_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate entity data against its schema and return cleaned dict."""
    schema_cls = ENTITY_SCHEMA_MAP.get(schema_type)
    if not schema_cls:
        return data  # Unknown schema, pass through
    try:
        validated = schema_cls(**data)
        return validated.model_dump(exclude_none=True)
    except Exception:
        return data  # Validation failed, pass through raw


# ── Relationship Schema ──────────────────────────────────────────────

class RelationshipExtraction(BaseModel):
    """Structured relationship between two entities."""
    source: str
    source_type: str
    target: str
    target_type: str
    relationship: str          # snake_case verb phrase: works_at, lives_in
    confidence: float = 0.7
    properties: Dict[str, Any] = Field(default_factory=dict)


class StructuredEntity(BaseModel):
    """An entity with its schema type and typed data."""
    schema_type: str           # e.g. "PersonEntity"
    data: Dict[str, Any]       # Fields from the schema


class ExtractionResult(BaseModel):
    """Complete extraction output from a conversation turn."""
    memories: List[Dict[str, Any]]
    entities: List[StructuredEntity] = Field(default_factory=list)
    relationships: List[RelationshipExtraction] = Field(default_factory=list)


# ── Prompt Generation ────────────────────────────────────────────────

def generate_entity_schemas_prompt() -> str:
    """Generate a human-readable description of all entity schemas for the LLM prompt."""
    lines = []
    for schema_name, schema_cls in ENTITY_SCHEMA_MAP.items():
        fields = []
        for field_name, field_info in schema_cls.model_fields.items():
            required = field_info.is_required()
            desc = field_info.description or ""
            type_str = "required" if required else "optional"
            fields.append(f"    - {field_name} ({type_str}){': ' + desc if desc else ''}")
        lines.append(f"**{schema_name}:**")
        lines.extend(fields)
        lines.append("")
    return "\n".join(lines)
