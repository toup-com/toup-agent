"""Memory CRUD and search endpoints"""

import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, Memory, MemoryEvent
from app.schemas import (
    MemoryCreate, MemoryUpdate, MemoryResponse, MemoryWithScore,
    MemoryWithRelations, MemorySearchRequest, MemorySearchResponse,
    MemoryCategory, MemoryType, EntityResponse,
    MemoryEventResponse, MemoryEventsResponse
)
from app.api.auth import get_current_user
from app.services.memory_service import MemoryService

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
    service = MemoryService(db)
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
    from app.schemas import MemorySearchRequest, BrainType
    
    # Parse categories if provided
    category_list = None
    if categories:
        category_list = [MemoryCategory(c.strip()) for c in categories.split(",") if c.strip()]
    
    # Parse brain_type if provided
    brain_type_enum = None
    if brain_type:
        brain_type_enum = BrainType(brain_type)
    
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
    
    service = MemoryService(db)
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
    service = MemoryService(db)
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
    
    dedup_service = MemoryDedupService(db)
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
    
    dedup_service = MemoryDedupService(db)
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
    
    dedup_service = MemoryDedupService(db)
    
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
