"""
Toup Platform MCP Server.

Exposes platform operations (memory, session, entity, identity, graph)
as MCP tools that per-user Agents can call from their VPS.

The Agent connects to this MCP server as a client, using its AGENT_API_KEY
for authentication. Each tool operates on the authenticated user's data.

Mounted at /api/mcp in platform_main.py via FastMCP's integration.
"""

import logging
from typing import Optional
from fastmcp import FastMCP

from app.config import settings
from app.db.database import async_session_maker
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

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
    """Resolve the MCP request's user_id.

    Primary path (post-T0c): the MCPAuthMiddleware (`app/mcp_auth.py`)
    validated the request's X-Agent-Key against `agent_configs` and bound
    the user_id to a contextvar. We read it here.

    Fallback path: `settings.user_id` is set on agent-side processes that
    happen to import this module (currently no such caller exists, but
    the constant is retained so this resolver doesn't break if one
    appears). On the platform process `settings.user_id` is always
    empty — so without middleware binding, this raises ValueError, which
    FastMCP serialises as a clean tool-call error.

    The failure mode during warn-only soak: an unauthenticated MCP call
    reaches a tool handler, `get_mcp_user_id()` raises, the client sees
    a structured error. That's the correct behavior — we want
    unauthenticated callers to fail loudly with a clear reason, not
    silently return wrong-tenant data (the pre-T0c bug).
    """
    from app.mcp_auth import get_mcp_user_id

    try:
        return get_mcp_user_id()
    except ValueError:
        # No middleware-bound user. Fall back to the legacy single-tenant
        # path only if settings.user_id is set (agent-side processes).
        if settings.user_id:
            return settings.user_id
        raise


# ── Memory Tools (v3 §2.1.6 / §4) ─────────────────────────────────────
#
# What used to be here: `memory_search` / `memory_create` / `memory_update` /
# `memory_delete` / `memory_list`, five row tools against `GET|POST|PATCH|
# DELETE /api/memories[/{id}]`. Every one of those routes is DELETED in v3 —
# the product's unit is a FILE, and `memories` is retired. A tool that proxies
# a route which no longer exists does not fail loudly; it returns the
# platform's own stale monolith-era rows and calls them the user's memory.
#
# Worse, `memory_create`'s own docstring instructed the model to store a
# memory "about a specific routine, trigger, or connector" with `ref_kind`
# and `ref_id` — that guidance is a named producer of the routine-prompt junk
# v3 exists to remove (write-path recon, answer (c)), and routines are
# scheduler objects that were never memories.
#
# What remains: a READ over files, and one WRITE that is an instruction to
# the curator — the same two shapes the in-agent tools have. Both proxy to
# the tenant, because that is where the files live; unlike round 8, the read
# does NOT fall back to the platform session, because there is nothing there
# to fall back to.

async def _proxy_files_read(db, user_id: str, path: str, params: Optional[dict] = None):
    """Read a v3 file route on the tenant agent. None when unreachable."""
    from app.api.memories import _get_agent_proxy_info, _proxy_memories

    try:
        proxy = await _get_agent_proxy_info(user_id, db)
    except Exception:
        return None
    if not proxy:
        return None
    try:
        data = await _proxy_memories(proxy[0], proxy[1], path, params=params or {})
    except Exception as e:  # noqa: BLE001
        logger.warning("[mcp] memory %s proxy failed: %s", path, e)
        return None
    if isinstance(data, dict) and "__status__" in data:
        return None
    return data


@mcp.tool()
async def memory_search(query: str, limit: int = 10) -> dict:
    """Search the user's memory FILES.

    Returns file-attributed snippets: each result names the file it came
    from ("areas/toup"), which can then be opened whole. There are no memory
    ids and no per-row categories — the file is the unit.
    """
    user_id = _get_user_id()
    async with async_session_maker() as db:
        data = await _proxy_files_read(
            db, user_id, "search", {"q": query, "limit": min(limit, 50)},
        )
    if data is None:
        return {"error": "Your agent isn't reachable right now.", "results": []}
    results = data.get("results") or []
    return {"results": results, "total": len(results)}


@mcp.tool()
async def memory_files(slug: Optional[str] = None) -> dict:
    """List the user's memory files, or read ONE of them in full.

    Without `slug`: every file, grouped by section, with its title and the
    one-line description that says when to read it. With `slug` (e.g.
    "areas/toup", "people/majid-tajik", "you/profile"): that file's whole
    markdown body plus the files it links to.
    """
    user_id = _get_user_id()
    async with async_session_maker() as db:
        path = f"files/{slug.strip()}" if (slug or "").strip() else "files"
        data = await _proxy_files_read(db, user_id, path)
    if data is None:
        return {"error": "Your agent isn't reachable right now."}
    return data


@mcp.tool()
async def memory_remember(instruction: str) -> dict:
    """Tell the user's agent to remember, change or forget something.

    Plain language, exactly as the user would say it: "remember that she
    uses an Android phone", "the IELTS date moved to Sept 12", "forget that
    I live in Toronto". The agent's own writer decides which file it belongs
    in, rewrites it in the house voice, merges it with anything it already
    knows, and records a line in the user's memory log.

    It may decline — one-off requests, reminders, transient states and tool
    output are never stored. Read `applied` and `rejected` in the reply
    rather than assuming the write happened.
    """
    from app.api.memories import _get_agent_proxy_info, _proxy_memories_write

    user_id = _get_user_id()
    text = (instruction or "").strip()
    if not text:
        return {"error": "instruction is required"}

    async with async_session_maker() as db:
        try:
            proxy = await _get_agent_proxy_info(user_id, db)
        except Exception:
            proxy = None
        if not proxy:
            # A WRITE never falls back to the platform session. The files are
            # AGENT_ONLY; "succeeding" against the wrong database is how a
            # user comes to believe something was saved that was not.
            return {"error": "Your agent isn't reachable right now.", "applied": 0}
        try:
            data = await _proxy_memories_write(
                proxy[0], proxy[1], "instruct", "POST",
                body={"instruction": text},
            )
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}", "applied": 0}
    return data or {"applied": 0}


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
        # NOTE (2026-08-07 audit): this writes the edge into the PLATFORM
        # database, and a user with a tenant agent has their knowledge graph in
        # the TENANT database — so the agent never reads this row back.
        #
        # Left as-is deliberately. `entity_search` and `graph_traverse` are also
        # platform-side, so an external MCP client gets a self-consistent silo:
        # what it writes, it can read. Making this tool refuse would break that
        # working flow, and routing it to the tenant needs agent-side entity
        # REST routes that do not exist yet. That gap is tracked by
        # test_remaining_mcp_tools_on_agent_only_tables_are_known, which fails
        # if the unproxied set grows OR shrinks, so it cannot be forgotten.
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
