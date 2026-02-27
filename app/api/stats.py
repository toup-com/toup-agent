"""Stats endpoints for brain visualization"""

import json
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.db import get_db, Memory, Entity, BrainStats, memory_relationships
from app.schemas import (
    BrainStatsResponse, CategoryStats, TimelineResponse, TimelineEntry,
    ConnectionsResponse, ConnectionData, MemoryResponse, MemoryCategory, AgentCategory
)
from app.api.auth import get_current_user
from app.api.memories import memory_to_response

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("/categories", response_model=BrainStatsResponse)
async def get_category_stats(
    brain_type: str = Query(None, description="Filter by brain type: user, agent, work"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get memory statistics by category for visualization"""
    
    # Build query filters
    filters = [
        Memory.user_id == current_user.id,
        Memory.is_deleted == False,
        Memory.is_active == True
    ]
    
    if brain_type:
        filters.append(Memory.brain_type == brain_type)
    
    # Calculate fresh stats - only count active, non-deleted memories
    result = await db.execute(
        select(Memory.category, func.count(Memory.id))
        .where(and_(*filters))
        .group_by(Memory.category)
    )
    category_counts = {row[0]: row[1] for row in result.fetchall()}
    
    total_memories = sum(category_counts.values())
    max_count = max(category_counts.values()) if category_counts else 1
    
    # Choose the right category enum based on brain_type
    if brain_type == 'agent':
        category_enum = AgentCategory
    else:
        category_enum = MemoryCategory
    
    categories = []
    for category in category_enum:
        count = category_counts.get(category.value, 0)
        size = count / max_count if max_count > 0 else 0.1
        size = max(size, 0.1) if count > 0 else 0.05
        categories.append(CategoryStats(category=category.value, count=count, size=size))
    
    # Also include any categories in the data that aren't in the enum
    for cat, count in category_counts.items():
        if cat not in [c.value for c in category_enum]:
            size = count / max_count if max_count > 0 else 0.1
            categories.append(CategoryStats(category=cat, count=count, size=size))
    
    # Count entities
    result = await db.execute(
        select(func.count(Entity.id)).where(Entity.user_id == current_user.id)
    )
    total_entities = result.scalar() or 0
    
    # Count connections
    result = await db.execute(
        select(func.count())
        .select_from(memory_relationships)
        .join(Memory, Memory.id == memory_relationships.c.source_id)
        .where(Memory.user_id == current_user.id)
    )
    total_connections = result.scalar() or 0
    
    return BrainStatsResponse(
        total_memories=total_memories,
        total_entities=total_entities,
        total_connections=total_connections,
        categories=categories,
        updated_at=datetime.utcnow()
    )


# Alias for backward compatibility
@router.get("/regions", response_model=BrainStatsResponse)
async def get_region_stats(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Alias for get_category_stats"""
    return await get_category_stats(current_user, db)


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    days: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get memory activity timeline for the past N days"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get memories grouped by date and category
    result = await db.execute(
        select(
            func.date(Memory.created_at).label("date"),
            Memory.category,
            func.count(Memory.id).label("count")
        )
        .where(
            and_(
                Memory.user_id == current_user.id,
                Memory.created_at >= start_date,
                Memory.is_deleted == False
            )
        )
        .group_by(func.date(Memory.created_at), Memory.category)
        .order_by(func.date(Memory.created_at))
    )
    rows = result.fetchall()
    
    # Aggregate by date
    date_data = {}
    for row in rows:
        date_str = str(row[0])
        if date_str not in date_data:
            date_data[date_str] = {"count": 0, "categories": {}}
        date_data[date_str]["count"] += row[2]
        date_data[date_str]["categories"][row[1]] = row[2]
    
    entries = [
        TimelineEntry(
            date=date_str,
            count=data["count"],
            categories=data["categories"]
        )
        for date_str, data in sorted(date_data.items())
    ]
    
    return TimelineResponse(
        entries=entries,
        start_date=start_date.date().isoformat(),
        end_date=end_date.date().isoformat()
    )


@router.get("/connections", response_model=ConnectionsResponse)
async def get_connections(
    limit: int = Query(100, ge=10, le=500),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get memory connection graph data for visualization"""
    # Get connections
    result = await db.execute(
        select(
            memory_relationships.c.source_id,
            memory_relationships.c.target_id,
            memory_relationships.c.strength,
            memory_relationships.c.relationship_type
        )
        .join(Memory, Memory.id == memory_relationships.c.source_id)
        .where(Memory.user_id == current_user.id)
        .limit(limit)
    )
    connection_rows = result.fetchall()
    
    # Get unique memory IDs
    memory_ids = set()
    for row in connection_rows:
        memory_ids.add(row[0])
        memory_ids.add(row[1])
    
    # Fetch memories
    if memory_ids:
        result = await db.execute(
            select(Memory).where(
                and_(
                    Memory.id.in_(memory_ids),
                    Memory.user_id == current_user.id,
                    Memory.is_deleted == False
                )
            )
        )
        memories = result.scalars().all()
    else:
        memories = []
    
    # Build response
    nodes = [memory_to_response(m) for m in memories]
    connections = [
        ConnectionData(
            source_id=row[0],
            target_id=row[1],
            strength=row[2] or 1.0,
            type=row[3] or "related_to"
        )
        for row in connection_rows
    ]
    
    return ConnectionsResponse(nodes=nodes, connections=connections)


@router.get("/recent", response_model=List[MemoryResponse])
async def get_recent_memories(
    limit: int = Query(20, ge=1, le=100),
    brain_type: Optional[str] = Query(None),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get most recently created memories, optionally filtered by brain_type."""
    filters = [Memory.user_id == current_user.id, Memory.is_deleted == False]
    if brain_type:
        filters.append(Memory.brain_type == brain_type)
    result = await db.execute(
        select(Memory)
        .where(and_(*filters))
        .order_by(Memory.created_at.desc())
        .limit(limit)
    )
    memories = result.scalars().all()
    return [memory_to_response(m) for m in memories]


@router.get("/entities", response_model=List[dict])
async def get_entity_stats(
    limit: int = Query(50, ge=1, le=200),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get entity statistics for visualization"""
    result = await db.execute(
        select(Entity)
        .where(Entity.user_id == current_user.id)
        .order_by(Entity.mention_count.desc())
        .limit(limit)
    )
    entities = result.scalars().all()
    
    return [
        {
            "id": e.id,
            "name": e.name,
            "type": e.entity_type,
            "mentions": e.mention_count,
            "first_seen": e.first_seen_at.isoformat(),
            "last_seen": e.last_seen_at.isoformat(),
        }
        for e in entities
    ]
