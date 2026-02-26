"""Agent API endpoints - for HexBrain agent to store and retrieve memories"""

import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db import get_db, Memory, Entity, EntityLink
from app.schemas import (
    AgentStoreRequest, AgentRecallRequest, AgentRecallResponse,
    AgentGraphRequest, AgentGraphResponse,
    MemoryCreate, MemoryResponse, MemoryWithScore, MemoryWithRelations,
    EntityResponse, MemorySearchRequest
)
from app.api.auth import get_current_user
from app.api.memories import memory_to_response
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
