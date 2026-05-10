"""Agent API endpoints - for Toup agent to store and retrieve memories"""

import json
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, select, and_

from app.db import get_db, Memory, Entity, EntityLink
from app.db.models import AgentSecurityEvent, EVENT_AGENT_KEY_ROTATED
from app.schemas import (
    AgentStoreRequest, AgentRecallRequest, AgentRecallResponse,
    AgentGraphRequest, AgentGraphResponse,
    MemoryCreate, MemoryResponse, MemoryWithScore, MemoryWithRelations,
    EntityResponse, MemorySearchRequest
)
from app.api.auth import get_current_user
from app.api.memories import memory_to_response
from app.config import settings
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/agent", tags=["Agent API"])


@router.post("/store", response_model=List[MemoryResponse])
async def agent_store_memories(
    request: AgentStoreRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Store multiple memories from the agent.
    Used when the agent wants to explicitly save information.
    """
    service = MemoryService(db)
    created_memories = []
    
    for memory_data in request.memories:
        memory = await service.create_memory(
            current_user.id,
            memory_data,
            source_message_id=None  # Agent-created memories
        )
        created_memories.append(memory)
    
    return [memory_to_response(m) for m in created_memories]


@router.post("/recall", response_model=AgentRecallResponse)
async def agent_recall_memories(
    request: AgentRecallRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Recall relevant memories for the agent during a conversation.
    Returns top-k memories ranked by semantic similarity.
    """
    service = MemoryService(db)
    
    # Build search query
    search_request = MemorySearchRequest(
        query=request.query + (" " + request.context if request.context else ""),
        categories=request.categories,
        limit=request.limit,
        include_explanation=True
    )
    
    results, _, _ = await service.search_memories(current_user.id, search_request)
    
    # Filter by minimum similarity
    filtered_results = [
        r for r in results
        if r.similarity_score >= request.min_similarity
    ]
    
    # Generate context summary
    context_summary = None
    if filtered_results:
        # Create a brief summary of retrieved memories
        summaries = [r.summary or r.content[:100] for r in filtered_results[:3]]
        context_summary = " | ".join(summaries)
    
    return AgentRecallResponse(
        memories=filtered_results,
        context_summary=context_summary
    )


@router.post("/graph", response_model=AgentGraphResponse)
async def agent_graph_traversal(
    request: AgentGraphRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Traverse the memory graph starting from a specific memory.
    Returns connected memories and entities up to specified depth.
    """
    service = MemoryService(db)
    
    # Get root memory
    root_memory = await service.get_memory(request.memory_id, current_user.id)
    if not root_memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    # Get related memories
    related_memories = await service.get_related_memories(
        request.memory_id,
        current_user.id,
        depth=request.depth
    )
    
    # Get entities from root memory
    all_entities = []
    if request.include_entities:
        result = await db.execute(
            select(EntityLink).where(EntityLink.memory_id == request.memory_id)
        )
        links = result.scalars().all()
        
        for link in links:
            result = await db.execute(
                select(Entity).where(Entity.id == link.entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                all_entities.append(EntityResponse(
                    id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    mention_count=entity.mention_count,
                    first_seen_at=entity.first_seen_at,
                    last_seen_at=entity.last_seen_at,
                    attributes=json.loads(entity.attributes_json) if entity.attributes_json else None
                ))
    
    # Build response with relations
    related_with_relations = []
    for memory in related_memories:
        # Get entities for each related memory
        entities = []
        if request.include_entities:
            result = await db.execute(
                select(EntityLink).where(EntityLink.memory_id == memory.id)
            )
            links = result.scalars().all()
            for link in links:
                result = await db.execute(
                    select(Entity).where(Entity.id == link.entity_id)
                )
                entity = result.scalar_one_or_none()
                if entity:
                    entities.append(EntityResponse(
                        id=entity.id,
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        mention_count=entity.mention_count,
                        first_seen_at=entity.first_seen_at,
                        last_seen_at=entity.last_seen_at,
                        attributes=json.loads(entity.attributes_json) if entity.attributes_json else None
                    ))
        
        related_with_relations.append(MemoryWithRelations(
            **memory_to_response(memory).model_dump(),
            related_memories=[],  # Don't recurse further
            entities=entities
        ))
    
    return AgentGraphResponse(
        root=memory_to_response(root_memory),
        related=related_with_relations,
        entities=all_entities
    )


@router.get("/context", response_model=AgentRecallResponse)
async def agent_get_context(
    query: str,
    limit: int = 5,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Quick context retrieval for the agent.
    Simpler endpoint for getting relevant memories during conversation.
    """
    service = MemoryService(db)
    
    search_request = MemorySearchRequest(
        query=query,
        limit=limit,
        include_explanation=True
    )
    
    results, _, _ = await service.search_memories(current_user.id, search_request)
    
    context_summary = None
    if results:
        summaries = [r.summary or r.content[:100] for r in results[:3]]
        context_summary = " | ".join(summaries)
    
    return AgentRecallResponse(
        memories=results,
        context_summary=context_summary
    )


@router.get("/entity/{entity_name}/memories", response_model=List[MemoryResponse])
async def get_memories_by_entity(
    entity_name: str,
    limit: int = 20,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all memories related to a specific entity.
    Useful for retrieving everything known about a person, project, etc.
    """
    # Find entity
    result = await db.execute(
        select(Entity).where(
            and_(
                Entity.user_id == current_user.id,
                Entity.name.ilike(f"%{entity_name}%")
            )
        )
    )
    entities = result.scalars().all()
    
    if not entities:
        return []
    
    # Get all memory IDs linked to these entities
    entity_ids = [e.id for e in entities]
    result = await db.execute(
        select(EntityLink.memory_id).where(
            EntityLink.entity_id.in_(entity_ids)
        )
    )
    memory_ids = [row[0] for row in result.fetchall()]
    
    if not memory_ids:
        return []
    
    # Fetch memories
    result = await db.execute(
        select(Memory)
        .where(
            and_(
                Memory.id.in_(memory_ids),
                Memory.user_id == current_user.id,
                Memory.is_deleted == False
            )
        )
        .order_by(Memory.created_at.desc())
        .limit(limit)
    )
    memories = result.scalars().all()

    return [memory_to_response(m) for m in memories]


# ─── T0b: agent_api_key rotation ──────────────────────────────────────


class RotateAgentKeyResponse(BaseModel):
    new_agent_api_key: str
    rotated_at: str  # ISO 8601 UTC
    key_prefix: str  # First 8 chars for display


class LastRotationResponse(BaseModel):
    last_rotated_at: Optional[str] = None  # ISO 8601 UTC, or None if never


@router.post("/rotate-key", response_model=RotateAgentKeyResponse)
async def rotate_agent_key(
    x_sensitive_action_token: Optional[str] = Header(default=None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rotate this user's `agent_api_key`.

    Authentication chain:
      1. The standard JWT/session auth identifies the user (`current_user`).
      2. `X-Sensitive-Action-Token` carries a fresh `ROTATE_AGENT_KEY`-purpose
         token issued by `POST /auth/reauth` with body
         `{"purpose": "rotate_agent_key"}`. Single-use; replay-protected.

    Effect:
      * New `agent_api_key` is generated, persisted to `agent_configs`,
        and pushed into the tenant container's env (via bridge recreate).
      * The new container is verified by polling `/agent/health` with
        the new key before this endpoint returns success.
      * Active WS chat sessions disconnect when the old container is
        killed by recreate. This is the desired behavior for a security
        action — the user is told this on the confirm modal.
      * On any failure (bridge recreate fails, verify times out), the
        DB is rolled back to the old key and the bridge is restored to
        the old env. The old key remains valid.

    Response:
      * The new key is returned EXACTLY ONCE in this response. The
        client must display it for the user to copy and never request
        it again. The DB does not persist any plaintext copy beyond
        this row.
    """
    # Routing guard: agent_router is mounted on both platform and agent
    # containers. Rotation only makes sense from the platform side
    # (it pushes new env to the bridge, which the platform owns the
    # mTLS cert for). On the agent process, surface a 404 so this
    # endpoint is invisible to anyone probing the tenant URL.
    if settings.run_mode == "agent":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )
    if not settings.enable_agent_key_rotation:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Agent key rotation is not enabled in this environment. "
                "Contact the operator to flip ENABLE_AGENT_KEY_ROTATION."
            ),
        )

    # Verify the sensitive-action token (replay-protected, purpose-bound).
    from app.services.sensitive_action_token import (
        SensitiveActionPurpose,
        verify_sensitive_action_token,
    )
    await verify_sensitive_action_token(
        db,
        x_sensitive_action_token or "",
        expected_user_id=str(current_user.id),
        expected_purpose=SensitiveActionPurpose.ROTATE_AGENT_KEY,
    )

    # Atomic rotation. Raises AgentKeyRotationError on any failure;
    # rollback (DB + bridge restore) is handled inside.
    from app.services.agent_key_rotation import (
        AgentKeyRotationError,
        rotate_agent_api_key,
    )
    try:
        result = await rotate_agent_api_key(db, str(current_user.id))
    except AgentKeyRotationError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        )

    return RotateAgentKeyResponse(
        new_agent_api_key=result.new_agent_api_key,
        rotated_at=result.rotated_at.isoformat() + "Z",
        key_prefix=result.key_prefix,
    )


@router.get("/key-last-rotated", response_model=LastRotationResponse)
async def get_last_key_rotation(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return when this user last rotated their agent_api_key, or None
    if never. Drives the "Last rotated: X ago" line on /agent/settings."""
    if settings.run_mode == "agent":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )
    row = (
        await db.execute(
            select(AgentSecurityEvent.occurred_at)
            .where(AgentSecurityEvent.user_id == str(current_user.id))
            .where(AgentSecurityEvent.event_type == EVENT_AGENT_KEY_ROTATED)
            .order_by(desc(AgentSecurityEvent.occurred_at))
            .limit(1)
        )
    ).scalar_one_or_none()
    return LastRotationResponse(
        last_rotated_at=(row.isoformat() + "Z") if row else None,
    )


# ─── T1h: connector tools cache invalidation ───────────────────────────


class RefreshToolsResponse(BaseModel):
    status: str
    tool_count: int


@router.put("/refresh-tools", response_model=RefreshToolsResponse)
async def refresh_connector_tools(request: Request):
    """T1h — Drop the agent's MCP tools cache and refetch immediately.

    Called by the platform after an OAuth connect / disconnect so the
    agent's tool list reflects the change without waiting for the 60s
    TTL. Best-effort from the platform's side: if this 500s or times
    out, the TTL is the safety net.

    Auth model:
      * Tenant agents have a unique `agent_api_key`. The platform
        knows it (it minted it at provision time and pushes it into
        the tenant container's env). The endpoint accepts the same
        `X-Agent-Key` header the soul-sync path uses
        (`app/api/soul.py:353`). Any other caller — JWT user, CSRF —
        is rejected.
      * Run-mode guard: only meaningful on agent containers
        (`run_mode == "agent"`). On the platform process itself,
        return 404 so the endpoint is invisible to anyone probing
        `<platform>/api/agent/refresh-tools`.

    Idempotent: multiple PUTs in quick succession are coalesced by
    the cache's per-instance asyncio.Lock — only one MCP round-trip
    fires regardless of how many platform callers race.
    """
    if settings.run_mode != "agent":
        # On the platform, this endpoint is meaningless — there's no
        # MCP tools cache here. Surface 404 so probers don't learn
        # the endpoint exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )

    # X-Agent-Key auth — same primitive as soul.py:353. Reject if
    # missing, malformed, or doesn't match the tenant's key.
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent key",
        )

    # Reach the cache via app.state. The agent's lifespan wires it
    # there at boot if MCP discovery succeeded. If it's missing,
    # the agent is in a degraded mode (no MCP client) — return 200
    # with tool_count=0 so the platform's call site doesn't treat
    # missing-cache as a failure (the platform already logs WARN on
    # any non-200; a 200 with zero tools accurately reflects the
    # agent's state).
    cache = getattr(request.app.state, "mcp_tools_cache", None)
    if cache is None:
        return RefreshToolsResponse(status="no_cache", tool_count=0)

    # Force an immediate refetch (bypasses TTL). Lock inside the cache
    # coalesces concurrent platform notifications.
    try:
        await cache.refresh()
    except Exception as e:
        # Refetch failed — DB / network blip on the platform side.
        # Cache.invalidate() ensures the next list_tools() retries.
        cache.invalidate()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Refresh failed: {type(e).__name__}: {str(e)[:200]}",
        )
    return RefreshToolsResponse(
        status="invalidated",
        tool_count=len(cache.tools),
    )


class MCPCacheStatusResponse(BaseModel):
    """Snapshot of the agent's MCP tools cache for ops diagnostics.

    Returned by /api/agent/mcp-cache-status. The whole point is that
    when a user reports "the agent says no Gmail tool but the platform
    shows it Connected", an operator can hit this endpoint and see at
    a glance whether the cache is healthy, never-fetched, or
    persistently failing — without having to docker-exec into the
    tenant container to grep stdout.
    """

    has_cache: bool
    tool_count: int
    tools: list[str]
    cached_at_monotonic: float
    last_attempt_at_monotonic: float
    consecutive_failures: int
    last_error: Optional[str]
    is_fresh: bool


@router.get("/mcp-cache-status", response_model=MCPCacheStatusResponse)
async def mcp_cache_status(request: Request):
    """Read-only view of the MCP tools cache state.

    Same X-Agent-Key gate as /refresh-tools. The platform's admin
    dashboard polls this when an operator clicks "Diagnose connectors"
    on a tenant; if `consecutive_failures > 0` we surface the
    `last_error` next to the tile so the failure mode is visible
    instead of hidden behind a generic "no tools available".
    """
    if settings.run_mode != "agent":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )

    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent key",
        )

    cache = getattr(request.app.state, "mcp_tools_cache", None)
    if cache is None:
        return MCPCacheStatusResponse(
            has_cache=False,
            tool_count=0,
            tools=[],
            cached_at_monotonic=0.0,
            last_attempt_at_monotonic=0.0,
            consecutive_failures=0,
            last_error=None,
            is_fresh=False,
        )

    return MCPCacheStatusResponse(
        has_cache=True,
        tool_count=len(cache.tools),
        tools=list(cache.tools),
        cached_at_monotonic=cache.cached_at,
        last_attempt_at_monotonic=getattr(cache, "last_attempt_at", 0.0),
        consecutive_failures=getattr(cache, "consecutive_failures", 0),
        last_error=getattr(cache, "last_error", None),
        is_fresh=cache.is_fresh(),
    )
