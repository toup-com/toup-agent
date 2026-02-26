"""Knowledge Graph API endpoints — entity and relationship exploration"""

import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, Entity, EntityRelationship
from app.schemas import (
    EntityResponse,
    EntityBriefResponse,
    EntityRelationshipResponse,
    GraphTraversalRequest,
    GraphTraversalResponse,
    GraphTraversalNode,
    GraphExplorationResponse,
)
from app.api.auth import get_current_user
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


@router.get("/entities", response_model=GraphExplorationResponse)
async def list_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type: person, place, organization, etc."),
    search: Optional[str] = Query(None, description="Search entity names"),
    limit: int = Query(100, ge=1, le=500),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all entities in the user's knowledge graph.
    Optionally filter by entity_type or search by name.
    """
    service = MemoryService(db)
    entities = await service.get_entities(
        user_id=current_user.id,
        entity_type=entity_type,
        search=search,
        limit=limit,
    )

    return GraphExplorationResponse(
        entities=[
            EntityResponse(
                id=e["id"],
                name=e["name"],
                entity_type=e["entity_type"],
                description=e.get("description"),
                mention_count=e.get("mention_count", 0),
                first_seen_at=e["first_seen_at"],
                last_seen_at=e["last_seen_at"],
                attributes=e.get("attributes"),
            )
            for e in entities
        ],
        relationships=[],
        total_entities=len(entities),
        total_relationships=0,
    )


@router.get("/relationships", response_model=GraphExplorationResponse)
async def list_relationships(
    entity_id: Optional[str] = Query(None, description="Filter to relationships involving this entity"),
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
    limit: int = Query(100, ge=1, le=500),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List entity-to-entity relationships in the knowledge graph.
    Optionally filter by entity or relationship type.
    """
    service = MemoryService(db)
    rels = await service.get_entity_relationships(
        user_id=current_user.id,
        entity_id=entity_id,
        relationship_type=relationship_type,
        limit=limit,
    )

    return GraphExplorationResponse(
        entities=[],
        relationships=[
            EntityRelationshipResponse(
                id=r["id"],
                relationship_type=r["relationship_type"],
                relationship_label=r.get("relationship_label"),
                confidence=r["confidence"],
                mention_count=r["mention_count"],
                first_seen_at=r.get("first_seen_at"),
                last_seen_at=r.get("last_seen_at"),
                properties=r.get("properties"),
                source_entity=EntityBriefResponse(**r["source_entity"]) if r.get("source_entity") else None,
                target_entity=EntityBriefResponse(**r["target_entity"]) if r.get("target_entity") else None,
            )
            for r in rels
        ],
        total_entities=0,
        total_relationships=len(rels),
    )


@router.post("/traverse", response_model=GraphTraversalResponse)
async def traverse_graph(
    request: GraphTraversalRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Traverse the entity knowledge graph using a recursive CTE.
    
    Provide either `entity_names` (resolved by name search) or `entity_ids`
    (pre-resolved IDs) as seed nodes. The traversal follows EntityRelationship
    edges bidirectionally up to `max_depth` hops.
    
    Returns discovered entities, the relationships connecting them, and
    optionally linked memories.
    """
    if not request.entity_names and not request.entity_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either entity_names or entity_ids",
        )

    service = MemoryService(db)

    # Resolve seed entity IDs
    seed_entity_ids = list(request.entity_ids) if request.entity_ids else None

    nodes = await service.traverse_entity_graph(
        user_id=current_user.id,
        seed_entity_ids=seed_entity_ids,
        entity_names=request.entity_names,
        max_depth=request.max_depth,
        limit=request.limit,
    )

    # Build seed entity briefs
    seed_ids = set(request.entity_ids or [])
    if not seed_ids and nodes:
        seed_ids = {n["entity_id"] for n in nodes if n["depth"] == 0}

    seed_entities = [
        EntityBriefResponse(
            id=n["entity_id"],
            name=n["entity_name"],
            entity_type=n["entity_type"],
        )
        for n in nodes
        if n["entity_id"] in seed_ids
    ]

    traversal_nodes = [
        GraphTraversalNode(**n)
        for n in nodes
    ]

    # Optionally fetch relationships between discovered entities
    discovered_ids = [n["entity_id"] for n in nodes]
    rels_data = []
    if discovered_ids:
        rels_data = await service.get_entity_relationships(
            user_id=current_user.id,
            limit=request.limit * 3,
        )
        # Filter to only relationships between discovered entities
        discovered_set = set(discovered_ids)
        rels_data = [
            r for r in rels_data
            if (r.get("source_entity", {}) or {}).get("id") in discovered_set
            and (r.get("target_entity", {}) or {}).get("id") in discovered_set
        ]

    rels_response = [
        EntityRelationshipResponse(
            id=r["id"],
            relationship_type=r["relationship_type"],
            relationship_label=r.get("relationship_label"),
            confidence=r["confidence"],
            mention_count=r["mention_count"],
            first_seen_at=r.get("first_seen_at"),
            last_seen_at=r.get("last_seen_at"),
            properties=r.get("properties"),
            source_entity=EntityBriefResponse(**r["source_entity"]) if r.get("source_entity") else None,
            target_entity=EntityBriefResponse(**r["target_entity"]) if r.get("target_entity") else None,
        )
        for r in rels_data
    ]

    return GraphTraversalResponse(
        seed_entities=seed_entities,
        nodes=traversal_nodes,
        relationships=rels_response,
        total_entities=len(nodes),
        total_relationships=len(rels_response),
        memories=None,  # TODO: optionally fetch linked memories if request.include_memories
    )


@router.get("/entity/{entity_id}")
async def get_entity_detail(
    entity_id: str,
    include_relationships: bool = Query(True, description="Include relationships this entity participates in"),
    include_memories: bool = Query(False, description="Include memories linked to this entity"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a single entity, including its
    relationships and optionally linked memories.
    """
    from sqlalchemy import select, and_
    from app.db import Entity, EntityLink, Memory

    result = await db.execute(
        select(Entity).where(
            and_(Entity.id == entity_id, Entity.user_id == current_user.id)
        )
    )
    entity = result.scalar_one_or_none()
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    response: dict = {
        "id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "description": entity.description,
        "mention_count": entity.mention_count,
        "first_seen_at": entity.first_seen_at.isoformat() if entity.first_seen_at else None,
        "last_seen_at": entity.last_seen_at.isoformat() if entity.last_seen_at else None,
        "attributes": json.loads(entity.attributes_json) if entity.attributes_json else None,
    }

    if include_relationships:
        service = MemoryService(db)
        rels = await service.get_entity_relationships(
            user_id=current_user.id, entity_id=entity_id, limit=50
        )
        response["relationships"] = rels

    if include_memories:
        result = await db.execute(
            select(EntityLink.memory_id).where(EntityLink.entity_id == entity_id)
        )
        memory_ids = [row.memory_id for row in result.all()]
        if memory_ids:
            result = await db.execute(
                select(Memory).where(
                    and_(
                        Memory.id.in_(memory_ids),
                        Memory.user_id == current_user.id,
                        Memory.is_deleted == False,
                        Memory.is_active == True,
                    )
                )
            )
            memories = result.scalars().all()
            response["memories"] = [
                {
                    "id": m.id,
                    "content": m.content,
                    "summary": m.summary,
                    "category": m.category,
                    "importance": m.importance,
                    "strength": m.strength,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in memories
            ]

    return response
