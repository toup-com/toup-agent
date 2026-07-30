"""Memory CRUD and search endpoints"""

import json
import logging
from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import httpx

from app.db import get_db, Memory, MemoryEvent, AgentConfig
from app.schemas import (
    MemoryCreate, MemoryUpdate, MemoryResponse, MemoryWithScore,
    MemoryWithRelations, MemorySearchRequest, MemorySearchResponse,
    MemoryCategory, MemoryType, EntityResponse,
    MemoryEventResponse, MemoryEventsResponse
)
from app.api.auth import get_current_user
from app.memory_taxonomy import normalize_category
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


async def _get_user_api_key(db: AsyncSession, user_id: str) -> Optional[str]:
    """Fetch the user's OpenAI API key from agent_configs."""
    try:
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
            )
            return result.scalar_one_or_none()
    except Exception:
        return None


# ── Agent proxy helpers ────────────────────────────────────────────

async def _get_agent_proxy_info(
    user_id: str, db: AsyncSession
) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a remote agent."""
    try:
        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass  # agent_configs table may not exist on agent DBs
    return None


async def _proxy_memories(
    agent_url: str, agent_api_key: str, path: str,
    params: Optional[dict] = None, method: str = "GET", body: Optional[dict] = None,
):
    """Proxy a memories READ to the VPS agent.

    Returns None on any failure, which makes the caller fall back to the
    platform DB. That is acceptable for reads (worst case: an empty list) but
    NOT for writes — see `_proxy_memories_write`.

    TKT-LAT-007 (wave 3): shared agent_http client.
    """
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/memories/{path}" if path else f"{agent_url}/api/memories"
    try:
        client = get_agent_http_client()
        if method == "GET":
            resp = await client.get(
                url, headers={"X-Agent-Key": agent_api_key},
                params=params or {}, timeout=10.0,
            )
        else:
            resp = await client.post(
                url, headers={"X-Agent-Key": agent_api_key},
                params=params or {}, json=body or {}, timeout=10.0,
            )
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Agent memories proxy %s returned %s", url, resp.status_code)
    except Exception as e:
        logger.warning("Agent memories proxy %s failed: %s", url, e)
    return None


async def _proxy_memories_write(
    agent_url: str, agent_api_key: str, path: str,
    method: str, body: Optional[dict] = None, params: Optional[dict] = None,
):
    """Proxy a memories WRITE (POST/PATCH/DELETE) to the tenant agent.

    Backlog BE-1. `memories` is an AGENT_ONLY table: it exists in the tenant
    container's DB, not in the platform's Supabase DB. Reads have been proxied
    for a while, but writes were not — so every PATCH/DELETE executed against a
    database with no such table and could not possibly affect what the user
    sees. That is why the mobile app shipped with FEATURE_FLAGS.MEMORY_WRITES
    hard-off and no way to correct a wrong memory.

    Unlike the read helper this NEVER falls back to the platform DB. A write
    that cannot reach the tenant must surface as an error: silently "succeeding"
    against the wrong database is how a user comes to believe they deleted
    something that is still there. Raises HTTPException on failure; propagates
    the agent's own status code when it returns one.
    """
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/memories/{path}" if path else f"{agent_url}/api/memories"
    try:
        client = get_agent_http_client()
        resp = await client.request(
            method,
            url,
            headers={"X-Agent-Key": agent_api_key},
            params=params or {},
            json=body,
            timeout=15.0,
        )
    except Exception as e:
        logger.warning("Agent memories write proxy %s %s failed: %s", method, url, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach your agent to save this change. Please try again.",
        )

    if resp.status_code in (200, 201, 204):
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    if resp.status_code == 404:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found"
        )

    logger.warning(
        "Agent memories write proxy %s %s returned %s", method, url, resp.status_code
    )
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Your agent rejected this change. Please try again.",
    )


router = APIRouter(prefix="/memories", tags=["Memories"])


def memory_to_response(memory: Memory) -> MemoryResponse:
    """Convert Memory model to response schema with new enhancement fields"""
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        canonical_content=getattr(memory, 'canonical_content', None),  # Memory evolution
        summary=memory.summary,
        brain_type=getattr(memory, 'brain_type', 'user'),  # Default to 'user' for backward compatibility
        category=memory.category,
        memory_type=memory.memory_type,
        importance=memory.importance,
        confidence=memory.confidence,
        # NEW: Memory enhancement fields
        strength=memory.strength,
        memory_level=memory.memory_level,
        emotional_salience=memory.emotional_salience,
        last_reinforced_at=memory.last_reinforced_at,
        consolidation_count=memory.consolidation_count,
        decay_rate=memory.decay_rate,
        # Memory Evolution fields
        history=json.loads(memory.history_json) if getattr(memory, 'history_json', None) else None,
        merged_from=json.loads(memory.merged_from_json) if getattr(memory, 'merged_from_json', None) else None,
        superseded_by=getattr(memory, 'superseded_by', None),
        is_active=getattr(memory, 'is_active', True),
        expires_at=getattr(memory, 'expires_at', None),
        # Timestamps
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        last_accessed_at=memory.last_accessed_at,
        access_count=memory.access_count,
        source_type=memory.source_type,
        tags=json.loads(memory.tags_json) if memory.tags_json else None,
        metadata=json.loads(memory.metadata_json) if memory.metadata_json else None,
    )


def event_to_response(event: MemoryEvent) -> MemoryEventResponse:
    """Convert MemoryEvent model to response schema"""
    return MemoryEventResponse(
        id=event.id,
        memory_id=event.memory_id,
        event_type=event.event_type,
        timestamp=event.timestamp,
        event_data=json.loads(event.event_data_json) if event.event_data_json else None,
        trigger_source=event.trigger_source,
    )


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_data: MemoryCreate,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new memory"""
    # BE-1: route to the tenant agent, where the memories table actually lives.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories_write(
            proxy[0], proxy[1], "", "POST",
            body=memory_data.model_dump(mode="json", exclude_none=True),
        )
        if data is None:
            # The agent accepted the write but returned no body. The app types
            # this response as a MemoryDetail, so returning `null` would crash
            # the client — surface it as an error instead.
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Your agent saved the memory but returned no details.",
            )
        return JSONResponse(content=data, status_code=status.HTTP_201_CREATED)

    _key = await _get_user_api_key(db, current_user.id)
    service = MemoryService(db, api_key=_key)
    memory = await service.create_memory(current_user.id, memory_data)
    return memory_to_response(memory)


@router.get("", response_model=dict)
async def list_memories(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    brain_type: Optional[str] = Query(None, description="Filter by brain type: user, agent, work"),
    category: Optional[str] = Query(None, description="Filter by category"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    min_importance: Optional[float] = Query(None, ge=0, le=1),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all memories for the current user with optional filters.

    This is different from /search which does semantic similarity search.
    This endpoint returns memories in chronological order without embedding search.
    """
    # Try proxying to remote agent first
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        params = {"limit": limit, "offset": offset}
        if brain_type:
            params["brain_type"] = brain_type
        if category:
            params["category"] = category
        if memory_type:
            params["memory_type"] = memory_type
        if min_importance is not None:
            params["min_importance"] = str(min_importance)
        data = await _proxy_memories(proxy[0], proxy[1], "", params)
        if data is not None:
            return JSONResponse(content=data)

    service = MemoryService(db)
    memories, total_count = await service.list_memories(
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        brain_type=brain_type,
        category=category,
        memory_type=memory_type,
        min_importance=min_importance
    )
    
    return {
        "memories": [memory_to_response(m) for m in memories],
        "total_count": total_count,
        "limit": limit,
        "offset": offset
    }


# NOTE: /search routes must be defined BEFORE /{memory_id} to prevent route shadowing
@router.get("/search", response_model=MemorySearchResponse)
async def search_memories_get(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    brain_type: Optional[str] = Query(None, description="Filter by brain type: user, agent, work"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    min_importance: Optional[float] = Query(None, ge=0, le=1),
    min_similarity: float = Query(0.1, ge=0, le=1, description="Minimum similarity threshold"),
    min_strength: Optional[float] = Query(None, ge=0, le=1),
    memory_level: Optional[str] = Query(None),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Semantic search for memories (GET method for browser/frontend compatibility)"""
    # Try proxying to remote agent first
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        params: dict = {"query": query, "limit": limit, "min_similarity": str(min_similarity)}
        if brain_type:
            params["brain_type"] = brain_type
        if categories:
            params["categories"] = categories
        if min_importance is not None:
            params["min_importance"] = str(min_importance)
        if min_strength is not None:
            params["min_strength"] = str(min_strength)
        if memory_level:
            params["memory_level"] = memory_level
        data = await _proxy_memories(proxy[0], proxy[1], "search", params)
        if data is not None:
            return JSONResponse(content=data)

    from app.schemas import MemorySearchRequest, BrainType
    
    # Parse categories if provided.
    # Normalize first, and drop anything still unrecognised. A bare
    # `MemoryCategory(c)` raises ValueError -> 500 on any retired value, and
    # retired values are exactly what old clients send: the web Brain page's
    # filter bar still enumerates the pre-unification names. A stale filter
    # should degrade, not crash the search.
    category_list = None
    if categories:
        category_list = []
        for raw in categories.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                category_list.append(MemoryCategory(normalize_category(raw)))
            except ValueError:
                logger.info(
                    "[memories] ignoring unrecognised category filter %r", raw
                )
        category_list = category_list or None

    # Parse brain_type if provided
    brain_type_enum = None
    if brain_type:
        try:
            brain_type_enum = BrainType(brain_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown brain_type {brain_type!r}.",
            )
    
    # Build search request
    request = MemorySearchRequest(
        query=query,
        limit=limit,
        brain_type=brain_type_enum,
        categories=category_list,
        min_importance=min_importance,
        min_similarity=min_similarity,
        min_strength=min_strength,
        memory_level=memory_level
    )
    
    _key = await _get_user_api_key(db, current_user.id)
    service = MemoryService(db, api_key=_key)
    results, total_count, search_time_ms = await service.search_memories(
        current_user.id, request
    )
    
    return MemorySearchResponse(
        query=request.query,
        results=results,
        total_count=total_count,
        search_time_ms=search_time_ms
    )


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: MemorySearchRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Semantic search for memories with filters"""
    # Try proxying to remote agent first
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories(
            proxy[0], proxy[1], "search", method="POST",
            body=request.model_dump(exclude_none=True),
        )
        if data is not None:
            return JSONResponse(content=data)

    _key = await _get_user_api_key(db, current_user.id)
    service = MemoryService(db, api_key=_key)
    results, total_count, search_time_ms = await service.search_memories(
        current_user.id, request
    )

    return MemorySearchResponse(
        query=request.query,
        results=results,
        total_count=total_count,
        search_time_ms=search_time_ms
    )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a memory by ID"""
    # This read never proxied, so opening a memory from search hit the platform
    # DB and 404'd for every user with an active agent — i.e. all of them.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories(proxy[0], proxy[1], memory_id)
        if data is not None:
            return JSONResponse(content=data)

    service = MemoryService(db)
    memory = await service.get_memory(memory_id, current_user.id)

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )

    return memory_to_response(memory)


@router.get("/{memory_id}/related", response_model=MemoryWithRelations)
async def get_memory_with_relations(
    memory_id: str,
    depth: int = Query(1, ge=1, le=3),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a memory with its related memories and entities"""
    service = MemoryService(db)
    memory = await service.get_memory(memory_id, current_user.id)
    
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    related = await service.get_related_memories(memory_id, current_user.id, depth)
    
    # Get entities (from entity_links)
    entities = []
    for link in memory.entity_links:
        entity = link.entity
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
    
    return MemoryWithRelations(
        **memory_to_response(memory).model_dump(),
        related_memories=[memory_to_response(m) for m in related],
        entities=entities
    )


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    update_data: MemoryUpdate,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a memory"""
    # BE-1: writes must reach the tenant DB — see _proxy_memories_write.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories_write(
            proxy[0], proxy[1], memory_id, "PATCH",
            body=update_data.model_dump(mode="json", exclude_none=True),
        )
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Your agent saved the change but returned no details.",
            )
        return JSONResponse(content=data)

    service = MemoryService(db)
    memory = await service.update_memory(memory_id, current_user.id, update_data)

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )

    return memory_to_response(memory)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a memory (soft delete)"""
    # BE-1: writes must reach the tenant DB — see _proxy_memories_write.
    # Deleting against the platform DB previously reported success while the
    # memory stayed visible on the user's phone.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        await _proxy_memories_write(proxy[0], proxy[1], memory_id, "DELETE")
        return

    service = MemoryService(db)
    deleted = await service.delete_memory(memory_id, current_user.id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )


@router.get("/category/{category}", response_model=List[MemoryResponse])
async def get_memories_by_category(
    category: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get memories by category (supports user, agent, and work categories)"""
    # Try proxying to remote agent first
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        params = {"limit": limit, "offset": offset}
        data = await _proxy_memories(proxy[0], proxy[1], f"category/{category}", params)
        if data is not None:
            return JSONResponse(content=data)

    service = MemoryService(db)
    memories = await service.get_memories_by_category(
        current_user.id, category, limit, offset
    )
    return [memory_to_response(m) for m in memories]


@router.get("/region/{region}", response_model=List[MemoryResponse])
async def get_memories_by_region(
    region: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get memories by brain region (deprecated, use /category/{category})"""
    service = MemoryService(db)
    memories = await service.get_memories_by_category(
        current_user.id, region, limit, offset
    )
    return [memory_to_response(m) for m in memories]


@router.get("/type/{memory_type}", response_model=List[MemoryResponse])
async def get_memories_by_type(
    memory_type: MemoryType,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get memories by type"""
    from sqlalchemy import select, and_
    
    result = await db.execute(
        select(Memory)
        .where(
            and_(
                Memory.user_id == current_user.id,
                Memory.memory_type == memory_type.value,
                Memory.is_deleted == False,
                Memory.is_active == True  # Only active memories
            )
        )
        .order_by(Memory.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    memories = result.scalars().all()
    return [memory_to_response(m) for m in memories]


# ============ Memory Events (Audit Log) Endpoints ============

@router.get("/{memory_id}/events", response_model=MemoryEventsResponse)
async def get_memory_events(
    memory_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the audit trail (events) for a specific memory.
    
    Events include: created, accessed, reinforced, decayed, consolidated, updated, deleted.
    This provides a complete history of all operations on the memory.
    """
    service = MemoryService(db)
    
    # First verify the memory belongs to this user
    memory = await service.get_memory(memory_id, current_user.id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    events = await service.get_memory_events(memory_id, current_user.id, limit)
    
    return MemoryEventsResponse(
        memory_id=memory_id,
        events=[event_to_response(e) for e in events],
        total_count=len(events)
    )


# ============ Memory Reinforcement Endpoints ============

@router.post("/{memory_id}/reinforce")
async def reinforce_memory(
    memory_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Reinforce a memory to increase its strength.
    
    This implements spaced repetition - accessing a memory makes it stronger
    and more resistant to decay. Useful when a memory is actively recalled
    or confirmed by the user.
    """
    # BE-1: "Keep" on the Memory screen lands here; it must reach the tenant.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        data = await _proxy_memories_write(
            proxy[0], proxy[1], f"{memory_id}/reinforce", "POST"
        )
        return JSONResponse(content=data or {"success": True, "memory_id": memory_id})

    from app.services.decay_service import DecayService

    decay_service = DecayService(db)

    memory = await decay_service.reinforce_memory(
        memory_id=memory_id,
        user_id=str(current_user.id),
        access_context="user_reinforce"
    )
    
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    return {
        "success": True,
        "memory_id": memory_id,
        "new_strength": memory.strength,
        "access_count": memory.access_count,
        "message": "Memory reinforced successfully"
    }


# ============ Memory Evolution Endpoints ============

@router.get("/{memory_id}/history")
async def get_memory_history(
    memory_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the evolution history of a specific memory.
    
    Returns the complete history of how this memory changed over time,
    including all versions, merge events, and change summaries.
    """
    service = MemoryService(db)
    history = await service.get_memory_history(memory_id, current_user.id)
    
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    return history


@router.post("/deduplicate")
async def deduplicate_memories(
    category: Optional[str] = Query(None, description="Optional category filter"),
    brain_type: Optional[str] = Query(None, description="Optional brain type filter"),
    dry_run: bool = Query(True, description="If True, just report what would be merged"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Scan and merge duplicate memories.
    
    This endpoint finds memories that are similar enough to be considered
    duplicates and either reports them (dry_run=True) or merges them (dry_run=False).
    
    When memories are merged:
    - The primary memory absorbs the content from duplicates
    - History is preserved showing the merge
    - Duplicates are marked as superseded (is_active=False)
    
    Args:
        category: Optional category to limit scan
        brain_type: Optional brain type to limit scan
        dry_run: If True, just report what would be merged without doing it
    """
    from app.services.memory_dedup_service import MemoryDedupService
    _key = await _get_user_api_key(db, current_user.id)
    dedup_service = MemoryDedupService(db, api_key=_key)
    results = await dedup_service.find_and_merge_duplicates(
        user_id=current_user.id,
        category=category,
        brain_type=brain_type,
        dry_run=dry_run
    )
    
    return {
        "dry_run": dry_run,
        "operations": results,
        "total_merge_groups": len(results),
        "total_duplicates_found": sum(len(op.get("duplicates", [])) for op in results),
        "message": "Dry run complete - no changes made" if dry_run else f"Merged {len(results)} groups of duplicates"
    }


@router.get("/duplicates/report")
async def get_duplicate_report(
    category: Optional[str] = Query(None, description="Optional category filter"),
    threshold: float = Query(0.85, ge=0.5, le=1.0, description="Similarity threshold"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a report of potential duplicate memories.
    
    This is a read-only operation that analyzes your memories and
    identifies groups of similar memories that could be merged.
    
    Returns statistics and detailed groups of potential duplicates.
    """
    from app.services.memory_dedup_service import MemoryDedupService
    _key = await _get_user_api_key(db, current_user.id)
    dedup_service = MemoryDedupService(db, api_key=_key)
    report = await dedup_service.get_duplicate_report(
        user_id=current_user.id,
        category=category,
        threshold=threshold
    )
    
    return report


@router.post("/{memory_id}/merge")
async def merge_into_memory(
    memory_id: str,
    new_content: str = Query(..., description="New content to merge into the memory"),
    source_type: str = Query("manual", description="Source of this merge"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Manually merge new content into an existing memory.
    
    This uses the LLM to intelligently combine the existing memory content
    with the new information, creating a single coherent memory.
    
    The merge is tracked in the memory's history.
    """
    from app.services.memory_dedup_service import MemoryDedupService
    from app.schemas import MemoryCreate, BrainType, MemoryType
    
    service = MemoryService(db)
    
    # First verify the memory exists and belongs to user
    memory = await service.get_memory(memory_id, current_user.id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    _key = await _get_user_api_key(db, current_user.id)
    dedup_service = MemoryDedupService(db, api_key=_key)

    # Create a temporary MemoryCreate for the merge
    merge_data = MemoryCreate(
        content=new_content,
        category=memory.category,
        memory_type=MemoryType(memory.memory_type),
        brain_type=BrainType(memory.brain_type) if memory.brain_type else BrainType.USER,
        source_type=source_type
    )
    
    # Use the internal merge method
    updated = await dedup_service._merge_memories(
        existing_memory_id=memory_id,
        new_content=new_content,
        new_memory_data=merge_data,
        user_id=current_user.id
    )
    
    return {
        "success": True,
        "memory_id": memory_id,
        "merged_content": updated.canonical_content or updated.content,
        "history_count": len(json.loads(updated.history_json)) if updated.history_json else 1,
        "message": "Content merged successfully"
    }
