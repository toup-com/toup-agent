"""
Toup Platform MCP Server.

Exposes platform operations (memory, session, entity, identity, graph)
as MCP tools that per-user Agents can call from their VPS.

The Agent connects to this MCP server as a client, using its AGENT_API_KEY
for authentication. Each tool operates on the authenticated user's data.

Mounted at /api/mcp in platform_main.py via FastMCP's integration.
"""

from typing import Optional
from fastmcp import FastMCP

from app.config import settings
from app.db.database import async_session_maker
from app.services.memory_service import MemoryService

mcp = FastMCP(
    "Toup Platform",
    instructions=(
        "Toup Platform MCP server. Provides tools for memory management, "
        "knowledge graph traversal, identity documents, and session handling. "
        "All operations are scoped to the authenticated user."
    ),
)


# ── Helper: resolve user_id from context ──────────────────────────────

def _get_user_id() -> str:
    """Get user_id from settings (set via AGENT_API_KEY → user binding).

    In MCP mode, the Agent's USER_ID env var identifies the owner.
    """
    if settings.user_id:
        return settings.user_id
    raise ValueError("USER_ID not configured — cannot identify user for MCP operations")


# ── Memory Tools ──────────────────────────────────────────────────────

@mcp.tool()
async def memory_search(
    query: str,
    limit: int = 10,
    brain_type: str = "user",
    categories: Optional[list[str]] = None,
    min_importance: Optional[float] = None,
    min_similarity: float = 0.1,
    include_explanation: bool = False,
) -> dict:
    """Search user memories by semantic similarity.

    Returns ranked results with relevance scores. Supports filtering
    by brain_type (user/agent/work), categories, and importance.
    """
    from app.api.memories import MemorySearchRequest

    user_id = _get_user_id()
    request = MemorySearchRequest(
        query=query,
        brain_type=brain_type,
        categories=categories,
        min_importance=min_importance,
        min_similarity=min_similarity,
        limit=min(limit, 50),
        include_explanation=include_explanation,
    )

    async with async_session_maker() as db:
        svc = MemoryService(db)
        results, total, search_time = await svc.search_memories(user_id, request)
        return {
            "results": [
                {
                    "id": str(r.memory.id),
                    "content": r.memory.content,
                    "summary": r.memory.summary,
                    "category": r.memory.category,
                    "importance": float(r.memory.importance),
                    "score": float(r.score),
                    "explanation": getattr(r, "explanation", None),
                }
                for r in results
            ],
            "total": total,
            "search_time_ms": search_time,
        }


@mcp.tool()
async def memory_create(
    content: str,
    category: str,
    brain_type: str = "user",
    memory_type: str = "episodic",
    importance: float = 0.5,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Store a new memory in the user's brain.

    Automatically generates embeddings and deduplicates against existing
    memories. Returns the created (or reinforced) memory.
    """
    from app.api.memories import MemoryCreate

    user_id = _get_user_id()
    memory_data = MemoryCreate(
        content=content,
        category=category,
        brain_type=brain_type,
        memory_type=memory_type,
        importance=importance,
        tags=tags or [],
        metadata=metadata or {},
    )

    async with async_session_maker() as db:
        svc = MemoryService(db)
        memory = await svc.create_memory(user_id, memory_data)
        return {
            "id": str(memory.id),
            "content": memory.content,
            "category": memory.category,
            "importance": float(memory.importance),
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
        }


@mcp.tool()
async def memory_update(
    memory_id: str,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Update an existing memory's content, category, or importance.

    Only provided fields are updated. Re-generates embedding if content changes.
    """
    from app.api.memories import MemoryUpdate

    user_id = _get_user_id()
    update_data = MemoryUpdate()
    if content is not None:
        update_data.content = content
    if category is not None:
        update_data.category = category
    if importance is not None:
        update_data.importance = importance
    if tags is not None:
        update_data.tags = tags

    async with async_session_maker() as db:
        svc = MemoryService(db)
        memory = await svc.update_memory(memory_id, user_id, update_data)
        if not memory:
            return {"error": "Memory not found", "id": memory_id}
        return {
            "id": str(memory.id),
            "content": memory.content,
            "category": memory.category,
            "importance": float(memory.importance),
            "updated": True,
        }


@mcp.tool()
async def memory_delete(memory_id: str) -> dict:
    """Delete a memory by ID (soft delete)."""
    user_id = _get_user_id()
    async with async_session_maker() as db:
        svc = MemoryService(db)
        deleted = await svc.delete_memory(memory_id, user_id)
        return {"id": memory_id, "deleted": deleted}


@mcp.tool()
async def memory_list(
    limit: int = 20,
    brain_type: Optional[str] = None,
    category: Optional[str] = None,
    min_importance: Optional[float] = None,
) -> dict:
    """List memories with optional filters.

    Returns memories ordered by creation date (newest first).
    """
    user_id = _get_user_id()
    async with async_session_maker() as db:
        svc = MemoryService(db)
        memories, total = await svc.list_memories(
            user_id,
            limit=min(limit, 100),
            brain_type=brain_type,
            category=category,
            min_importance=min_importance,
        )
        return {
            "memories": [
                {
                    "id": str(m.id),
                    "content": m.content,
                    "summary": m.summary,
                    "category": m.category,
                    "importance": float(m.importance),
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in memories
            ],
            "total": total,
        }


# ── Session Tools ─────────────────────────────────────────────────────

@mcp.tool()
async def session_create(
    title: Optional[str] = None,
    channel: str = "agent",
) -> dict:
    """Create a new conversation session.

    Returns the session ID for subsequent message tracking.
    """
    import uuid
    from datetime import datetime
    from app.db.models import Conversation

    user_id = _get_user_id()
    session_id = str(uuid.uuid4())

    async with async_session_maker() as db:
        session = Conversation(
            id=session_id,
            user_id=user_id,
            title=title,
            channel=channel,
            is_active=True,
            started_at=datetime.utcnow(),
        )
        db.add(session)
        await db.commit()
        return {
            "id": session_id,
            "title": title,
            "channel": channel,
            "is_active": True,
        }


@mcp.tool()
async def session_list(
    limit: int = 20,
    channel: Optional[str] = None,
    active_only: bool = False,
) -> dict:
    """List conversation sessions, optionally filtered by channel or active status."""
    from sqlalchemy import select
    from app.db.models import Conversation

    user_id = _get_user_id()
    async with async_session_maker() as db:
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.started_at.desc())
            .limit(min(limit, 100))
        )
        if channel:
            stmt = stmt.where(Conversation.channel == channel)
        if active_only:
            stmt = stmt.where(Conversation.is_active == True)

        result = await db.execute(stmt)
        sessions = result.scalars().all()
        return {
            "sessions": [
                {
                    "id": str(s.id),
                    "title": s.title,
                    "channel": s.channel,
                    "is_active": s.is_active,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                }
                for s in sessions
            ],
        }


# ── Entity / Graph Tools ─────────────────────────────────────────────

@mcp.tool()
async def entity_search(
    query: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """Search the knowledge graph for entities by name or type.

    Entity types include: person, place, organization, concept, event, etc.
    """
    user_id = _get_user_id()
    async with async_session_maker() as db:
        svc = MemoryService(db)
        entities = await svc.get_entities(
            user_id,
            entity_type=entity_type,
            search=query,
            limit=min(limit, 100),
        )
        return {
            "entities": [
                {
                    "id": str(e["id"]),
                    "name": e["name"],
                    "entity_type": e["entity_type"],
                    "description": e.get("description"),
                    "mention_count": e.get("mention_count", 0),
                }
                for e in entities
            ],
        }


@mcp.tool()
async def graph_traverse(
    entity_name: str,
    max_depth: int = 2,
    limit: int = 50,
) -> dict:
    """Traverse the knowledge graph from a seed entity.

    Follows relationships bidirectionally up to max_depth hops.
    Returns connected entities and their relationships.
    """
    user_id = _get_user_id()
    async with async_session_maker() as db:
        svc = MemoryService(db)
        nodes = await svc.traverse_entity_graph(
            user_id,
            entity_names=[entity_name],
            max_depth=min(max_depth, 5),
            limit=min(limit, 200),
        )
        return {
            "seed": entity_name,
            "nodes": [
                {
                    "entity_id": str(n["entity_id"]),
                    "entity_name": n["entity_name"],
                    "entity_type": n["entity_type"],
                    "depth": n["depth"],
                    "relationship_type": n.get("relationship_type"),
                    "from_entity": n.get("from_entity_name"),
                }
                for n in nodes
            ],
            "total_nodes": len(nodes),
        }


@mcp.tool()
async def entity_relationship_create(
    source_name: str,
    source_type: str,
    target_name: str,
    target_type: str,
    relationship: str,
    confidence: float = 0.7,
) -> dict:
    """Create or update a relationship between two entities in the knowledge graph.

    Entities are created automatically if they don't exist.
    Repeated calls increment the mention count.
    """
    user_id = _get_user_id()
    async with async_session_maker() as db:
        svc = MemoryService(db)
        await svc.store_entity_relationship(
            user_id,
            source_name=source_name,
            source_type=source_type,
            target_name=target_name,
            target_type=target_type,
            relationship=relationship,
            confidence=confidence,
        )
        return {
            "source": {"name": source_name, "type": source_type},
            "target": {"name": target_name, "type": target_type},
            "relationship": relationship,
            "created": True,
        }


# ── Identity Tools ────────────────────────────────────────────────────

@mcp.tool()
async def identity_get(
    identity_type: Optional[str] = None,
    active_only: bool = True,
) -> dict:
    """Get the agent's identity documents (SOUL, instructions, context, etc.).

    Identity types: soul, user_profile, agent_instructions, tools, context.
    Returns all matching documents ordered by priority.
    """
    from sqlalchemy import select
    from app.db.models import IdentityDocument

    user_id = _get_user_id()
    async with async_session_maker() as db:
        stmt = (
            select(IdentityDocument)
            .where(IdentityDocument.user_id == user_id)
            .order_by(IdentityDocument.priority.desc())
        )
        if identity_type:
            stmt = stmt.where(IdentityDocument.identity_type == identity_type)
        if active_only:
            stmt = stmt.where(IdentityDocument.is_active == True)

        result = await db.execute(stmt)
        docs = result.scalars().all()
        return {
            "documents": [
                {
                    "id": str(d.id),
                    "identity_type": d.identity_type,
                    "name": d.name,
                    "content": d.content,
                    "priority": d.priority,
                    "is_active": d.is_active,
                }
                for d in docs
            ],
        }


@mcp.tool()
async def identity_update(
    identity_id: str,
    content: Optional[str] = None,
    name: Optional[str] = None,
    priority: Optional[int] = None,
) -> dict:
    """Update an identity document's content, name, or priority."""
    from sqlalchemy import select
    from app.db.models import IdentityDocument

    user_id = _get_user_id()
    async with async_session_maker() as db:
        result = await db.execute(
            select(IdentityDocument)
            .where(IdentityDocument.id == identity_id)
            .where(IdentityDocument.user_id == user_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            return {"error": "Identity document not found", "id": identity_id}

        if content is not None:
            doc.content = content
        if name is not None:
            doc.name = name
        if priority is not None:
            doc.priority = priority
        await db.commit()
        return {
            "id": str(doc.id),
            "identity_type": doc.identity_type,
            "name": doc.name,
            "updated": True,
        }
