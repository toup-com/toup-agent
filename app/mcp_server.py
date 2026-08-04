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
from app.services.memory_gate import MemoryRejected

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

# Sentinel: "this user has no tenant agent, use the local session".
# Distinguished from None, which means "the tenant answered, nothing found".
_NO_TENANT = object()


async def _proxy_memory_write_to_tenant(
    db,
    user_id: str,
    memory_id: str,
    method: str,
    body: Optional[dict] = None,
):
    """Route an MCP memory write to the tenant agent that owns the row.

    D-mem-B (2026-07-29): `memories` is an AGENT_ONLY table. It exists in each
    tenant container's own Postgres, NOT in the platform DB this MCP server is
    bound to. So `memory_update` on an id the agent had just been handed
    returned "Memory not found" — the row is real, it is simply in another
    database. #375 fixed this for the REST routes in `app/api/memories.py` but
    missed this MCP surface.

    Returns `_NO_TENANT` when the user has no active agent (caller falls back
    to the local session), None when the tenant reports 404, else the decoded
    body. Never raises: an MCP tool must return a structured error, not a 500.
    """
    from app.api.memories import _get_agent_proxy_info, _proxy_memories_write

    try:
        proxy = await _get_agent_proxy_info(user_id, db)
    except Exception:
        return _NO_TENANT
    if not proxy:
        return _NO_TENANT

    try:
        return await _proxy_memories_write(
            proxy[0], proxy[1], memory_id, method, body=body
        )
    except Exception as e:
        # _proxy_memories_write raises HTTPException(404) when the tenant does
        # not have the row, and 502 when it is unreachable. Both collapse to
        # "not found" for the tool's caller, which is the honest answer.
        if getattr(e, "status_code", None) == 404:
            return None
        logger.warning(
            "[mcp] memory write proxy %s %s failed: %s", method, memory_id, e
        )
        return None


async def _proxy_memory_list_from_tenant(
    db,
    user_id: str,
    params: Optional[dict] = None,
):
    """Read a memory list from the tenant agent that actually holds the rows.

    Read counterpart to `_proxy_memory_write_to_tenant`. Returns None when
    there is no tenant agent or the tenant is unreachable — for a READ that is
    an acceptable fall-back to the platform session, which is the deliberate
    asymmetry documented on `_proxy_memories`. Never raises.
    """
    from app.api.memories import _get_agent_proxy_info, _proxy_memories

    try:
        proxy = await _get_agent_proxy_info(user_id, db)
    except Exception:
        return None
    if not proxy:
        return None

    try:
        return await _proxy_memories(proxy[0], proxy[1], "", params=params or {})
    except Exception as e:
        logger.warning("[mcp] memory list proxy failed: %s", e)
        return None


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
    memory_type: str = "note",
    memory_level: str = "episodic",
    importance: float = 0.5,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    ref_kind: Optional[str] = None,
    ref_id: Optional[str] = None,
) -> dict:
    """Store a memory in the user's brain.

    Type vs Level (Ticket 2.1 — these are different things, don't conflate):
      `memory_type`  — WHAT KIND of fact this is. One of:
                       fact, preference, task, event, person, place,
                       project, decision, skill, file, note, conversation.
                       Default: "note".
      `memory_level` — WHICH COGNITIVE LAYER it lives in. One of:
                       episodic, semantic, procedural, meta.
                       Default: "episodic".

    Upsert by entity (Ticket 2): if this memory is about a specific
    routine, trigger, or connector, set BOTH `ref_kind` (e.g.
    "routine", "trigger", "connector") AND `ref_id` (the entity's
    id/uuid). When a memory with the same (user, ref_kind, ref_id)
    already exists, the new content REPLACES it instead of creating
    a duplicate. Without ref_kind+ref_id, semantic-similarity dedup
    is used (less reliable).

    Returns the created or updated memory.
    """
    # Fast-fail validation BEFORE any heavier imports so the LLM gets a
    # parseable error in <100ms instead of a 3 s Pydantic round-trip.
    # Ticket 2.1 — the "semantic" mistake the agent made was passing a
    # MemoryLevel value into a MemoryType slot. Schema imports are
    # lightweight (no FastAPI / DB side-effects).
    from app.schemas import MemoryType as _MT, MemoryLevel as _ML

    _valid_types = [t.value for t in _MT]
    if memory_type not in _valid_types:
        return {
            "error": f"Invalid memory_type {memory_type!r}. Must be one of: "
                     f"{', '.join(_valid_types)}. Note: cognitive layers "
                     f"like 'episodic' / 'semantic' go in `memory_level`, "
                     f"not `memory_type`.",
        }
    _valid_levels = [l.value for l in _ML]
    if memory_level not in _valid_levels:
        return {
            "error": f"Invalid memory_level {memory_level!r}. Must be one of: "
                     f"{', '.join(_valid_levels)}.",
        }

    # Heavier imports only after the cheap validation passes.
    from app.api.memories import MemoryCreate, MemoryUpdate
    from app.db.models import Memory as _Memory
    from sqlalchemy import select as _select

    user_id = _get_user_id()

    memory_data = MemoryCreate(
        content=content,
        category=category,
        brain_type=brain_type,
        memory_type=memory_type,
        memory_level=memory_level,
        importance=importance,
        tags=tags or [],
        metadata=metadata or {},
    )

    async with async_session_maker() as db:
        # The tenant decides FIRST — before any query against this session.
        #
        # Same reason as memory_update/memory_delete below: `memories` is an
        # AGENT_ONLY table living in the tenant's own Postgres, not in the
        # platform DB this MCP server is bound to. #377 proxied update and
        # delete but left create writing locally, which is worse than a plain
        # bug: the row lands in the shared platform DB (exactly what
        # AGENT_ONLY_TABLES exists to prevent), the owner never sees it —
        # tenant hybrid_search and the Memory screen read the tenant DB — and
        # once update/delete proxy, it can no longer be edited or removed.
        #
        # The ref_kind/ref_id upsert below stays on the no-tenant branch on
        # purpose. It queries `memories` directly, so against a tenant-having
        # user it would be interrogating the wrong database — the lookup that
        # is supposed to PREVENT duplicate rows would miss every time and
        # create them instead. MemoryCreate carries no ref_kind/ref_id (the
        # REST surface never had them), so the tenant cannot be asked to do
        # the upsert either without a schema change and a fleet rollout;
        # until then a tenant-backed create falls back to the semantic dedup
        # inside MemoryService.create_memory, which this tool's own docstring
        # already describes as the weaker path.
        proxied = await _proxy_memory_write_to_tenant(
            db, user_id, "", "POST",
            body=memory_data.model_dump(mode="json", exclude_none=True),
        )
        if proxied is not _NO_TENANT:
            if proxied is None:
                # Unreachable tenant. Do NOT fall through to the local session:
                # reporting success against a database the agent never reads is
                # how a user comes to believe something was saved when it was
                # not. Reads may fall back; writes must not.
                return {"error": "Could not reach your agent to save this memory."}
            if ref_kind and ref_id:
                proxied = {**proxied, "ref_kind": ref_kind, "ref_id": ref_id}
            return {**proxied, "action": "created"}

        # ── no tenant agent: legacy platform-local path ──────────────
        # Upsert path: when ref_kind+ref_id are set, look up the existing
        # memory and UPDATE it. This is the Ticket 2 fix: routines + triggers
        # that change schedule update their canonical memory instead of
        # producing N stale rows.
        if ref_kind and ref_id:
            existing = (await db.execute(
                _select(_Memory).where(
                    _Memory.user_id == user_id,
                    _Memory.ref_kind == ref_kind,
                    _Memory.ref_id == ref_id,
                    _Memory.is_deleted.is_(False),
                ).limit(1)
            )).scalar_one_or_none()
            if existing is not None:
                svc = MemoryService(db)
                update_data = MemoryUpdate()
                update_data.content = content
                update_data.category = category
                update_data.importance = importance
                if tags is not None:
                    update_data.tags = tags
                updated = await svc.update_memory(existing.id, user_id, update_data)
                if updated is not None:
                    return {
                        "id": str(updated.id),
                        "content": updated.content,
                        "category": updated.category,
                        "importance": float(updated.importance),
                        "ref_kind": ref_kind,
                        "ref_id": ref_id,
                        "action": "updated",
                    }

        svc = MemoryService(db)
        try:
            memory = await svc.create_memory(user_id, memory_data)
        except MemoryRejected as exc:
            # Model-controlled path: say why, so it does not retry verbatim.
            return {"action": "rejected", "reason": exc.reason}
        # Stamp ref linkage if provided. Service-level path doesn't know
        # about it; we apply it post-create so the partial unique index
        # catches future collisions.
        if ref_kind and ref_id:
            memory.ref_kind = ref_kind
            memory.ref_id = ref_id
            await db.commit()
        return {
            "id": str(memory.id),
            "content": memory.content,
            "category": memory.category,
            "importance": float(memory.importance),
            "ref_kind": ref_kind,
            "ref_id": ref_id,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "action": "created",
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
        # D-mem-B: `memories` is AGENT_ONLY — it lives in the tenant
        # container's DB, not in the platform's. Running this against the
        # platform session returned "Memory not found" for every id the agent
        # had just been handed, because the row genuinely is not here. Same
        # class as the REST write routes fixed in #375; this MCP surface was
        # missed. Route to the tenant, exactly as api/memories.py does.
        proxied = await _proxy_memory_write_to_tenant(
            db, user_id, memory_id, "PATCH",
            body=update_data.model_dump(mode="json", exclude_none=True),
        )
        if proxied is not _NO_TENANT:
            if proxied is None:
                return {"error": "Memory not found", "id": memory_id}
            return {**proxied, "updated": True}

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
        # See memory_update: the row lives in the tenant DB, not here.
        proxied = await _proxy_memory_write_to_tenant(
            db, user_id, memory_id, "DELETE"
        )
        if proxied is not _NO_TENANT:
            return {"id": memory_id, "deleted": True}

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
        # Ask the tenant first — `memories` is AGENT_ONLY, so listing against
        # the platform session queries the wrong database. It does not error;
        # it returns an EMPTY list, so an agent asking "what do I already know
        # about this person?" concludes "nothing" while their real memories sit
        # in the tenant DB. Unlike the write paths this MAY fall back (worst
        # case is an empty list either way) — but it has to try first.
        proxied = await _proxy_memory_list_from_tenant(
            db, user_id,
            params={
                k: v for k, v in {
                    "limit": min(limit, 100),
                    "brain_type": brain_type,
                    "category": category,
                    "min_importance": min_importance,
                }.items() if v is not None
            },
        )
        if isinstance(proxied, dict) and isinstance(proxied.get("memories"), list):
            rows = proxied["memories"]
            return {
                "memories": [
                    {
                        "id": str(m.get("id")),
                        "content": m.get("content"),
                        "summary": m.get("summary"),
                        "category": m.get("category"),
                        "importance": float(m.get("importance") or 0.0),
                        "created_at": m.get("created_at"),
                    }
                    for m in rows
                ],
                "total": proxied.get("total", len(rows)),
            }

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
