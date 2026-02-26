"""Memory service - CRUD operations and search for memories"""

import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import uuid
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, text
from sqlalchemy.orm import selectinload

from app.db.models import (
    Memory, Entity, EntityLink, EntityRelationship, BrainStats,
    MemoryEvent, MemoryEventType, memory_relationships,
)
from app.schemas import (
    MemoryCreate, MemoryUpdate, MemoryResponse, MemoryWithScore,
    MemorySearchRequest, MemoryCategory, MemoryType, MemoryLevel, BrainType
)
from app.services.embedding_service import get_embedding_service



def extract_temporal_filters(query: str) -> dict:
    """
    Detect temporal references in a query and return date filters.
    
    Parses natural language time expressions and converts them to
    created_after / created_before datetime bounds.
    
    Examples:
        "last week"        → created_after = now - 7 days
        "yesterday"        → created_after = start of yesterday, created_before = start of today
        "in January"       → created_after = Jan 1, created_before = Jan 31
        "3 days ago"       → created_after = now - 3 days
        "since December"   → created_after = Dec 1
        "before I moved"   → {} (can't resolve, returns empty)
    
    Returns:
        {
            "created_after": datetime | None,
            "created_before": datetime | None,
            "temporal_keywords": list[str],  # matched temporal phrases
            "has_temporal": bool,
        }
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    result = {"created_after": None, "created_before": None, "temporal_keywords": [], "has_temporal": False}
    query_lower = query.lower()
    
    # --- Relative time patterns (most specific first) ---
    
    # "N days/weeks/months ago"
    m = re.search(r'(\d+)\s+days?\s+ago', query_lower)
    if m:
        days = int(m.group(1))
        result["created_after"] = now - timedelta(days=days)
        result["created_before"] = now - timedelta(days=max(0, days - 1))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    m = re.search(r'(\d+)\s+weeks?\s+ago', query_lower)
    if m:
        weeks = int(m.group(1))
        result["created_after"] = now - timedelta(weeks=weeks)
        result["created_before"] = now - timedelta(weeks=max(0, weeks - 1))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    m = re.search(r'(\d+)\s+months?\s+ago', query_lower)
    if m:
        months = int(m.group(1))
        year = now.year
        month = now.month - months
        while month <= 0:
            month += 12
            year -= 1
        result["created_after"] = now.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "today"
    if re.search(r'\btoday\b', query_lower):
        result["created_after"] = today_start
        result["temporal_keywords"].append("today")
        result["has_temporal"] = True
        return result
    
    # "yesterday"
    if re.search(r'\byesterday\b', query_lower):
        result["created_after"] = today_start - timedelta(days=1)
        result["created_before"] = today_start
        result["temporal_keywords"].append("yesterday")
        result["has_temporal"] = True
        return result
    
    # "last week"
    if re.search(r'\blast\s+week\b', query_lower):
        # Last Monday through last Sunday
        days_since_monday = now.weekday()
        this_monday = today_start - timedelta(days=days_since_monday)
        last_monday = this_monday - timedelta(weeks=1)
        result["created_after"] = last_monday
        result["created_before"] = this_monday
        result["temporal_keywords"].append("last week")
        result["has_temporal"] = True
        return result
    
    # "this week"
    if re.search(r'\bthis\s+week\b', query_lower):
        days_since_monday = now.weekday()
        this_monday = today_start - timedelta(days=days_since_monday)
        result["created_after"] = this_monday
        result["temporal_keywords"].append("this week")
        result["has_temporal"] = True
        return result
    
    # "last month"
    if re.search(r'\blast\s+month\b', query_lower):
        first_of_this_month = today_start.replace(day=1)
        if now.month == 1:
            first_of_last_month = first_of_this_month.replace(year=now.year - 1, month=12)
        else:
            first_of_last_month = first_of_this_month.replace(month=now.month - 1)
        result["created_after"] = first_of_last_month
        result["created_before"] = first_of_this_month
        result["temporal_keywords"].append("last month")
        result["has_temporal"] = True
        return result
    
    # "this month"
    if re.search(r'\bthis\s+month\b', query_lower):
        result["created_after"] = today_start.replace(day=1)
        result["temporal_keywords"].append("this month")
        result["has_temporal"] = True
        return result
    
    # "last N days/weeks"
    m = re.search(r'(?:in\s+the\s+)?last\s+(\d+)\s+days?', query_lower)
    if m:
        result["created_after"] = now - timedelta(days=int(m.group(1)))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    m = re.search(r'(?:in\s+the\s+)?last\s+(\d+)\s+weeks?', query_lower)
    if m:
        result["created_after"] = now - timedelta(weeks=int(m.group(1)))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "recently" / "recent"
    if re.search(r'\brecent(?:ly)?\b', query_lower):
        result["created_after"] = now - timedelta(days=7)
        result["temporal_keywords"].append("recently")
        result["has_temporal"] = True
        return result
    
    # "since <month>" / "after <month>"
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }
    
    # "since January" / "after February" / "from March"
    m = re.search(r'(?:since|after|from)\s+(' + '|'.join(months_map.keys()) + r')(?:\s+(\d{4}))?', query_lower)
    if m:
        month_num = months_map[m.group(1)]
        year = int(m.group(2)) if m.group(2) else now.year
        # If month is in the future this year, assume last year
        if year == now.year and month_num > now.month:
            year -= 1
        result["created_after"] = datetime(year, month_num, 1)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "in January" / "in February 2025" / "during March"
    m = re.search(r'(?:in|during)\s+(' + '|'.join(months_map.keys()) + r')(?:\s+(\d{4}))?', query_lower)
    if m:
        month_num = months_map[m.group(1)]
        year = int(m.group(2)) if m.group(2) else now.year
        # If month is in the future this year, assume last year
        if year == now.year and month_num > now.month:
            year -= 1
        result["created_after"] = datetime(year, month_num, 1)
        # End of that month
        if month_num == 12:
            result["created_before"] = datetime(year + 1, 1, 1)
        else:
            result["created_before"] = datetime(year, month_num + 1, 1)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "before <month>"
    m = re.search(r'before\s+(' + '|'.join(months_map.keys()) + r')(?:\s+(\d{4}))?', query_lower)
    if m:
        month_num = months_map[m.group(1)]
        year = int(m.group(2)) if m.group(2) else now.year
        result["created_before"] = datetime(year, month_num, 1)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "last year"
    if re.search(r'\blast\s+year\b', query_lower):
        result["created_after"] = datetime(now.year - 1, 1, 1)
        result["created_before"] = datetime(now.year, 1, 1)
        result["temporal_keywords"].append("last year")
        result["has_temporal"] = True
        return result
    
    # "this year"
    if re.search(r'\bthis\s+year\b', query_lower):
        result["created_after"] = datetime(now.year, 1, 1)
        result["temporal_keywords"].append("this year")
        result["has_temporal"] = True
        return result
    
    return result


class MemoryService:
    """Service for memory CRUD operations and search"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = get_embedding_service()
    
    async def create_memory(
        self,
        user_id: str,
        memory_data: MemoryCreate,
        source_message_id: Optional[str] = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.9
    ) -> Memory:
        """Create a new memory with embedding, with deduplication support.
        
        Args:
            user_id: The user ID
            memory_data: Memory creation data
            source_message_id: Optional source message ID
            deduplicate: If True, check for similar memories first
            similarity_threshold: Minimum similarity to consider as duplicate (default 0.9)
            
        Returns:
            Memory: The created or reinforced memory
        """
        # Generate embedding
        embedding = self.embedding_service.embed(memory_data.content)
        
        # Deduplication: Check for similar existing memories
        if deduplicate:
            similar_memory = await self._find_similar_memory(
                user_id=user_id,
                embedding=embedding,
                threshold=similarity_threshold
            )
            if similar_memory:
                # Reinforce existing memory instead of creating duplicate
                return await self._reinforce_memory(similar_memory, memory_data)
        
        # Handle memory_level - can be MemoryLevel enum or string
        memory_level_value = (
            memory_data.memory_level.value 
            if hasattr(memory_data.memory_level, 'value') 
            else memory_data.memory_level
        )
        
        # Handle brain_type - can be BrainType enum or string
        brain_type_value = (
            memory_data.brain_type.value 
            if hasattr(memory_data.brain_type, 'value') 
            else memory_data.brain_type
        ) if memory_data.brain_type else 'user'
        
        # Handle category - can be string or enum
        category_value = (
            memory_data.category.value 
            if hasattr(memory_data.category, 'value') 
            else memory_data.category
        )
        
        # Handle memory_type - can be MemoryType enum or string
        memory_type_value = (
            memory_data.memory_type.value 
            if hasattr(memory_data.memory_type, 'value') 
            else memory_data.memory_type
        )
        
        memory = Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=memory_data.content,
            summary=memory_data.summary,
            brain_type=brain_type_value,
            category=category_value,
            memory_type=memory_type_value,
            importance=memory_data.importance,
            confidence=memory_data.confidence,
            # NEW: Memory enhancement fields
            strength=1.0,  # Start at full strength
            memory_level=memory_level_value,
            emotional_salience=memory_data.emotional_salience,
            last_reinforced_at=datetime.utcnow(),
            consolidation_count=0,
            decay_rate=0.1,  # Default decay rate
            # Embedding and metadata
            embedding=embedding,  # Native pgvector column
            embedding_json=json.dumps(embedding),  # Backward compat (DEPRECATED)
            tags_json=json.dumps(memory_data.tags) if memory_data.tags else None,
            metadata_json=json.dumps(memory_data.metadata) if memory_data.metadata else None,
            source_message_id=source_message_id,
            source_type=memory_data.source_type or ("conversation" if source_message_id else "manual"),
            # Memory Evolution fields
            canonical_content=memory_data.content,  # Initially same as content
            history_json=json.dumps([{
                "date": datetime.utcnow().isoformat(),
                "content": memory_data.content,
                "source": memory_data.source_type or ("conversation" if source_message_id else "manual"),
                "action": "created"
            }]),
            merged_from_json=json.dumps([]),
            is_active=True,
        )
        
        self.db.add(memory)
        await self.db.flush()  # Get the ID before committing
        
        # Log CREATED event to audit trail
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.CREATED,
            event_data={
                "category": memory.category,
                "memory_type": memory.memory_type,
                "memory_level": memory.memory_level,
                "importance": memory.importance,
                "source_type": memory.source_type,
            },
            trigger_source="api"
        )
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        # Link related memories
        if memory_data.related_memory_ids:
            await self._link_memories(memory.id, memory_data.related_memory_ids)
        
        # Update brain stats
        await self._update_brain_stats(user_id)
        
        return memory
    
    async def _find_similar_memory(
        self,
        user_id: str,
        embedding: List[float],
        threshold: float = 0.9
    ) -> Optional[Memory]:
        """
        Find an existing memory that is very similar to the new one.
        Uses pgvector for fast similarity search.
        
        Args:
            user_id: User ID
            embedding: Embedding of the new memory content
            threshold: Minimum similarity to consider as duplicate
            
        Returns:
            The most similar existing memory if above threshold, else None
        """
        # Use pgvector if the Memory model has a native embedding column
        if hasattr(Memory, 'embedding') and Memory.embedding is not None:
            try:
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = embedding  # raw list for pgvector
                query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.is_deleted == False,
                            Memory.embedding.isnot(None),
                        )
                    )
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(1)
                )
                result = await self.db.execute(query)
                row = result.first()
                if row and row.similarity >= threshold:
                    return row.Memory
                return None
            except Exception:
                pass  # Fall back to Python-side search
        
        # Fallback: Python-side cosine similarity (for SQLite or if pgvector fails)
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                )
            )
        )
        memories = result.scalars().all()
        
        best_match = None
        best_similarity = threshold
        
        for memory in memories:
            if memory.embedding_json:
                memory_embedding = json.loads(memory.embedding_json)
                similarity = self.embedding_service.cosine_similarity(
                    embedding, memory_embedding
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = memory
        
        return best_match
    
    async def _reinforce_memory(
        self,
        memory: Memory,
        new_data: MemoryCreate
    ) -> Memory:
        """
        Reinforce an existing memory instead of creating a duplicate.
        
        Updates the memory's strength, importance, and last_reinforced_at.
        
        Args:
            memory: The existing memory to reinforce
            new_data: The new memory data (used to potentially update importance)
            
        Returns:
            The reinforced memory
        """
        # Increase strength (capped at 1.0)
        memory.strength = min(1.0, memory.strength + 0.1)
        
        # Update importance if the new one is higher
        if new_data.importance > memory.importance:
            memory.importance = new_data.importance
        
        # Update confidence if the new one is higher
        if new_data.confidence > memory.confidence:
            memory.confidence = new_data.confidence
        
        # Track reinforcement
        memory.last_reinforced_at = datetime.utcnow()
        memory.consolidation_count += 1
        memory.access_count += 1
        memory.updated_at = datetime.utcnow()
        
        # Log REINFORCED event
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=memory.user_id,
            event_type=MemoryEventType.REINFORCED,
            event_data={
                "strength": memory.strength,
                "consolidation_count": memory.consolidation_count,
                "importance": memory.importance,
            },
            trigger_source="deduplication"
        )
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory

    async def merge_memory(
        self,
        memory_id: str,
        new_content: str,
        change_summary: str,
        source_type: str = "merge",
        new_embedding: Optional[List[float]] = None
    ) -> Memory:
        """
        Merge new information into an existing memory, updating history.
        
        Args:
            memory_id: ID of the memory to update
            new_content: The new merged content
            change_summary: Summary of what changed
            source_type: Source of this update (merge, conversation, manual)
            new_embedding: Pre-computed embedding for new content (optional)
            
        Returns:
            The updated memory
        """
        result = await self.db.execute(
            select(Memory).where(Memory.id == memory_id)
        )
        memory = result.scalar_one_or_none()
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        
        # Get existing history
        existing_history = json.loads(memory.history_json) if memory.history_json else []
        
        # Create history entry
        history_entry = {
            "date": datetime.utcnow().isoformat(),
            "content": new_content,
            "previous_content": memory.canonical_content or memory.content,
            "source": source_type,
            "action": "updated",
            "change_summary": change_summary
        }
        existing_history.append(history_entry)
        
        # Generate new embedding if not provided
        if new_embedding is None:
            new_embedding = self.embedding_service.embed(new_content)
        
        # Update memory
        memory.content = new_content
        memory.canonical_content = new_content
        memory.history_json = json.dumps(existing_history)
        memory.embedding_json = json.dumps(new_embedding)  # Backward compat
        if hasattr(memory, 'embedding'):
            memory.embedding = new_embedding  # Native pgvector
        memory.updated_at = datetime.utcnow()
        memory.consolidation_count += 1
        
        # Log UPDATED event
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=memory.user_id,
            event_type=MemoryEventType.UPDATED,
            event_data={
                "action": "merged",
                "change_summary": change_summary,
                "version_count": len(existing_history),
            },
            trigger_source=source_type
        )
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def supersede_memory(
        self,
        old_memory_id: str,
        new_memory_id: str
    ) -> Memory:
        """
        Mark a memory as superseded by another (after merging).
        
        Args:
            old_memory_id: ID of the memory being superseded
            new_memory_id: ID of the memory that supersedes it
            
        Returns:
            The updated old memory
        """
        result = await self.db.execute(
            select(Memory).where(Memory.id == old_memory_id)
        )
        memory = result.scalar_one_or_none()
        if not memory:
            raise ValueError(f"Memory {old_memory_id} not found")
        
        memory.superseded_by = new_memory_id
        memory.is_active = False
        memory.updated_at = datetime.utcnow()
        
        # Update the target memory's merged_from
        target_result = await self.db.execute(
            select(Memory).where(Memory.id == new_memory_id)
        )
        target_memory = target_result.scalar_one_or_none()
        if target_memory:
            merged_from = json.loads(target_memory.merged_from_json) if target_memory.merged_from_json else []
            if old_memory_id not in merged_from:
                merged_from.append(old_memory_id)
                target_memory.merged_from_json = json.dumps(merged_from)
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def get_memory_history(self, memory_id: str, user_id: str) -> Optional[Dict]:
        """
        Get the evolution history of a memory.
        
        Returns:
            Dict with memory details and history, or None if not found
        """
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id == memory_id,
                    Memory.user_id == user_id
                )
            )
        )
        memory = result.scalar_one_or_none()
        if not memory:
            return None
        
        history = json.loads(memory.history_json) if memory.history_json else []
        merged_from = json.loads(memory.merged_from_json) if memory.merged_from_json else []
        
        return {
            "memory_id": memory.id,
            "current_content": memory.canonical_content or memory.content,
            "original_content": memory.content,
            "history": history,
            "version_count": len(history) if history else 1,
            "merged_from": merged_from,
            "superseded_by": memory.superseded_by,
            "is_active": memory.is_active,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "last_updated": memory.updated_at.isoformat() if memory.updated_at else None
        }
    
    async def get_user_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        brain_type: Optional[str] = None,
        is_active: bool = True,
        include_superseded: bool = False
    ) -> List[Memory]:
        """
        Get all memories for a user with optional filters.
        
        Args:
            user_id: User ID
            category: Optional category filter
            brain_type: Optional brain type filter
            is_active: If True, only return active memories
            include_superseded: If True, include superseded memories
            
        Returns:
            List of memories
        """
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False
        ]
        
        if category:
            conditions.append(Memory.category == category)
        if brain_type:
            conditions.append(Memory.brain_type == brain_type)
        if is_active and not include_superseded:
            conditions.append(Memory.is_active == True)
        
        result = await self.db.execute(
            select(Memory).where(and_(*conditions)).order_by(Memory.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Get a memory by ID"""
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id == memory_id,
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                )
            )
        )
        memory = result.scalar_one_or_none()
        
        if memory:
            # Update access stats
            memory.last_accessed_at = datetime.utcnow()
            memory.access_count += 1
            
            # Log ACCESSED event to audit trail
            await self._log_memory_event(
                memory_id=memory.id,
                user_id=user_id,
                event_type=MemoryEventType.ACCESSED,
                event_data={
                    "access_count": memory.access_count,
                    "strength": memory.strength,
                },
                trigger_source="api"
            )
            
            await self.db.commit()
        
        return memory
    
    async def list_memories(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        brain_type: Optional[str] = None,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
        include_inactive: bool = False
    ) -> Tuple[List[Memory], int]:
        """
        List memories with optional filters, ordered by creation date (newest first).
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            offset: Offset for pagination
            brain_type: Optional brain type filter
            category: Optional category filter
            memory_type: Optional memory type filter
            min_importance: Optional minimum importance filter
            include_inactive: Include superseded/merged memories
            
        Returns:
            Tuple of (memories list, total count)
        """
        # Build conditions - only show active memories by default
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False
        ]
        
        if not include_inactive:
            conditions.append(Memory.is_active == True)
        
        if brain_type:
            conditions.append(Memory.brain_type == brain_type)
        if category:
            conditions.append(Memory.category == category)
        if memory_type:
            conditions.append(Memory.memory_type == memory_type)
        if min_importance is not None:
            conditions.append(Memory.importance >= min_importance)
        
        # Count total
        count_query = select(func.count(Memory.id)).where(and_(*conditions))
        count_result = await self.db.execute(count_query)
        total_count = count_result.scalar() or 0
        
        # Fetch memories
        query = (
            select(Memory)
            .where(and_(*conditions))
            .order_by(Memory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(query)
        memories = list(result.scalars().all())
        
        return memories, total_count

    async def update_memory(
        self,
        memory_id: str,
        user_id: str,
        update_data: MemoryUpdate
    ) -> Optional[Memory]:
        """Update a memory"""
        memory = await self.get_memory(memory_id, user_id)
        if not memory:
            return None
        
        update_dict = update_data.model_dump(exclude_unset=True)
        old_values = {}  # Track changes for audit
        
        # Update content and regenerate embedding if content changed
        if "content" in update_dict:
            old_values["content"] = memory.content[:100]  # Truncate for audit
            embedding = self.embedding_service.embed(update_dict["content"])
            memory.embedding_json = json.dumps(embedding)  # Backward compat
            if hasattr(memory, 'embedding'):
                memory.embedding = embedding  # Native pgvector
            memory.content = update_dict["content"]
        
        if "summary" in update_dict:
            old_values["summary"] = memory.summary
            memory.summary = update_dict["summary"]
        if "category" in update_dict:
            old_values["category"] = memory.category
            memory.category = update_dict["category"].value
        if "memory_type" in update_dict:
            old_values["memory_type"] = memory.memory_type
            memory.memory_type = update_dict["memory_type"].value
        if "importance" in update_dict:
            old_values["importance"] = memory.importance
            memory.importance = update_dict["importance"]
        if "tags" in update_dict:
            memory.tags_json = json.dumps(update_dict["tags"]) if update_dict["tags"] else None
        if "metadata" in update_dict:
            memory.metadata_json = json.dumps(update_dict["metadata"]) if update_dict["metadata"] else None
        
        # NEW: Handle memory enhancement fields
        if "memory_level" in update_dict:
            old_values["memory_level"] = memory.memory_level
            memory.memory_level = update_dict["memory_level"].value if hasattr(update_dict["memory_level"], 'value') else update_dict["memory_level"]
        if "emotional_salience" in update_dict:
            old_values["emotional_salience"] = memory.emotional_salience
            memory.emotional_salience = update_dict["emotional_salience"]
        
        memory.updated_at = datetime.utcnow()
        
        # Log UPDATED event to audit trail
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.UPDATED,
            event_data={
                "old_values": old_values,
                "fields_updated": list(update_dict.keys()),
            },
            trigger_source="api"
        )
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Soft delete a memory"""
        # First get the memory to log it
        result = await self.db.execute(
            select(Memory).where(
                and_(Memory.id == memory_id, Memory.user_id == user_id)
            )
        )
        memory = result.scalar_one_or_none()
        
        if not memory:
            return False
        
        # Log DELETED event to audit trail BEFORE deleting
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.DELETED,
            event_data={
                "category": memory.category,
                "memory_type": memory.memory_type,
                "content_preview": memory.content[:100],
                "strength_at_deletion": memory.strength,
            },
            trigger_source="api"
        )
        
        # Now perform soft delete
        memory.is_deleted = True
        memory.deleted_at = datetime.utcnow()
        await self.db.commit()
        
        await self._update_brain_stats(user_id)
        return True
    
    async def search_memories(
        self,
        user_id: str,
        request: MemorySearchRequest
    ) -> Tuple[List[MemoryWithScore], int, float]:
        """
        Semantic search for memories with optional filters.
        Returns (results, total_count, search_time_ms)
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed(request.query)
        
        # Build base query with filters - only search active, non-deleted memories
        query = select(Memory).where(
            and_(
                Memory.user_id == user_id,
                Memory.is_deleted == False,
                Memory.is_active == True  # Only search active memories
            )
        )
        
        # Apply filters
        # Filter by brain_type
        if request.brain_type:
            brain_type_value = (
                request.brain_type.value 
                if hasattr(request.brain_type, 'value') 
                else request.brain_type
            )
            query = query.where(Memory.brain_type == brain_type_value)
        
        if request.categories:
            categories = [c.value for c in request.categories]
            query = query.where(Memory.category.in_(categories))
        
        if request.memory_types:
            types = [t.value for t in request.memory_types]
            query = query.where(Memory.memory_type.in_(types))
        
        # NEW: Filter by memory levels
        if request.memory_levels:
            levels = [l.value for l in request.memory_levels]
            query = query.where(Memory.memory_level.in_(levels))
        
        if request.min_importance is not None:
            query = query.where(Memory.importance >= request.min_importance)
        
        # NEW: Filter by minimum strength (exclude decayed memories)
        if request.min_strength is not None:
            query = query.where(Memory.strength >= request.min_strength)
        
        if request.start_date:
            query = query.where(Memory.created_at >= request.start_date)
        
        if request.end_date:
            query = query.where(Memory.created_at <= request.end_date)
        
        if request.tags:
            # Check if any tag matches (JSON array search)
            for tag in request.tags:
                query = query.where(Memory.tags_json.contains(f'"{tag}"'))
        
        # Try pgvector-accelerated search first
        use_pgvector = hasattr(Memory, 'embedding') and Memory.embedding is not None
        use_weighted = getattr(request, 'use_weighted_scoring', True)
        scored_memories = []
        total_count = 0

        if use_pgvector:
            try:
                # Add pgvector filter
                query = query.where(Memory.embedding.isnot(None))
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = query_embedding  # raw list for pgvector
                
                # Query with pgvector cosine similarity
                pgvector_query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(query.whereclause)
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(request.limit * 2)  # Over-fetch for weighted re-ranking
                )
                
                result = await self.db.execute(pgvector_query)
                rows = result.all()
                total_count = len(rows)
                
                for memory, similarity in rows:
                    if similarity < request.min_similarity:
                        continue
                    
                    if use_weighted:
                        final_score = (
                            similarity * 0.4 +
                            memory.strength * 0.25 +
                            memory.importance * 0.2 +
                            memory.emotional_salience * 0.15
                        )
                        if getattr(request, 'boost_recent_access', True) and memory.last_accessed_at:
                            days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                            if days_since_access < 7:
                                recency_boost = (7 - days_since_access) / 7 * 0.1
                                final_score += recency_boost
                    else:
                        final_score = similarity
                    
                    explanation = None
                    if request.include_explanation:
                        explanation = self._generate_explanation(memory, similarity, final_score if use_weighted else None)
                    
                    scored_memories.append((memory, final_score, similarity, explanation))
                
                scored_memories.sort(key=lambda x: x[1], reverse=True)
                scored_memories = scored_memories[:request.limit]
            except Exception:
                scored_memories = []  # Fall through to Python-side search

        # Fallback: Python-side cosine similarity (SQLite or pgvector failure)
        if not scored_memories and not use_pgvector:
            result = await self.db.execute(query)
            memories = result.scalars().all()
            total_count = len(memories)
            
            for memory in memories:
                if memory.embedding_json:
                    memory_embedding = json.loads(memory.embedding_json)
                    similarity = self.embedding_service.cosine_similarity(
                        query_embedding, memory_embedding
                    )
                    
                    if use_weighted:
                        final_score = (
                            similarity * 0.4 +
                            memory.strength * 0.25 +
                            memory.importance * 0.2 +
                            memory.emotional_salience * 0.15
                        )
                        if getattr(request, 'boost_recent_access', True) and memory.last_accessed_at:
                            days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                            if days_since_access < 7:
                                recency_boost = (7 - days_since_access) / 7 * 0.1
                                final_score += recency_boost
                    else:
                        final_score = similarity
                    
                    if similarity >= request.min_similarity:
                        explanation = None
                        if request.include_explanation:
                            explanation = self._generate_explanation(memory, similarity, final_score if use_weighted else None)
                        
                        scored_memories.append((memory, final_score, similarity, explanation))
            
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            scored_memories = scored_memories[:request.limit]
        
        # Convert to response objects with new fields
        results = []
        for m, final_score, raw_similarity, explanation in scored_memories:
            results.append(MemoryWithScore(
                id=m.id,
                content=m.content,
                summary=m.summary,
                category=m.category,
                memory_type=m.memory_type,
                importance=m.importance,
                confidence=m.confidence,
                # NEW: Memory enhancement fields in response
                strength=m.strength,
                memory_level=m.memory_level,
                emotional_salience=m.emotional_salience,
                last_reinforced_at=m.last_reinforced_at,
                consolidation_count=m.consolidation_count,
                decay_rate=m.decay_rate,
                # Timestamps
                created_at=m.created_at,
                updated_at=m.updated_at,
                last_accessed_at=m.last_accessed_at,
                access_count=m.access_count,
                source_type=m.source_type,
                tags=json.loads(m.tags_json) if m.tags_json else None,
                metadata=json.loads(m.metadata_json) if m.metadata_json else None,
                similarity_score=raw_similarity,  # Use raw similarity for compatibility
                explanation=explanation
            ))
        
        search_time_ms = (time.time() - start_time) * 1000
        return results, total_count, search_time_ms
    
    async def get_memories_by_category(
        self,
        user_id: str,
        category,
        limit: int = 50,
        offset: int = 0
    ) -> List[Memory]:
        """Get memories by category (only active, non-deleted)"""
        # Handle both enum and string categories
        cat_value = category.value if hasattr(category, 'value') else str(category)
        result = await self.db.execute(
            select(Memory)
            .where(
                and_(
                    Memory.user_id == user_id,
                    Memory.category == cat_value,
                    Memory.is_deleted == False,
                    Memory.is_active == True  # Only active memories
                )
            )
            .order_by(Memory.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_memories_by_brain_type(
        self,
        user_id: str,
        brain_type: str = "agent",
        limit: int = 50,
    ) -> List[dict]:
        """Get all active memories for a brain type, returned as dicts."""
        result = await self.db.execute(
            select(Memory)
            .where(
                and_(
                    Memory.user_id == user_id,
                    Memory.brain_type == brain_type,
                    Memory.is_deleted == False,
                    Memory.is_active == True,
                )
            )
            .order_by(Memory.strength.desc(), Memory.created_at.desc())
            .limit(limit)
        )
        memories = result.scalars().all()
        return [
            {
                "id": m.id,
                "content": m.content,
                "category": m.category,
                "brain_type": m.brain_type,
                "strength": m.strength,
                "importance": m.importance,
            }
            for m in memories
        ]

    # Alias for backward compatibility
    async def get_memories_by_region(
        self,
        user_id: str,
        region: MemoryCategory,
        limit: int = 50,
        offset: int = 0
    ) -> List[Memory]:
        """Alias for get_memories_by_category"""
        return await self.get_memories_by_category(user_id, region, limit, offset)
    
    async def get_related_memories(
        self,
        memory_id: str,
        user_id: str,
        depth: int = 1
    ) -> List[Memory]:
        """Get memories related to a given memory (graph traversal)"""
        visited = set()
        to_visit = [memory_id]
        related = []
        
        for _ in range(depth):
            next_level = []
            for mid in to_visit:
                if mid in visited:
                    continue
                visited.add(mid)
                
                # Get directly related memories
                result = await self.db.execute(
                    select(memory_relationships.c.target_id)
                    .where(memory_relationships.c.source_id == mid)
                )
                targets = [row[0] for row in result.fetchall()]
                
                result = await self.db.execute(
                    select(memory_relationships.c.source_id)
                    .where(memory_relationships.c.target_id == mid)
                )
                sources = [row[0] for row in result.fetchall()]
                
                next_level.extend(targets + sources)
            
            to_visit = [m for m in next_level if m not in visited]
        
        # Fetch the actual memory objects
        if visited:
            visited.discard(memory_id)  # Don't include the original
            result = await self.db.execute(
                select(Memory).where(
                    and_(
                        Memory.id.in_(visited),
                        Memory.user_id == user_id,
                        Memory.is_deleted == False
                    )
                )
            )
            related = result.scalars().all()
        
        return related
    
    async def search_by_entity_graph(
        self,
        user_id: str,
        entity_name: str,
        depth: int = 2,
        limit: int = 20,
    ) -> List[dict]:
        """
        Search memories by traversing the entity knowledge graph.
        
        1. Find the entity by name
        2. Find all memories linked to that entity
        3. Find related entities (via shared memories)
        4. Find memories linked to those related entities (depth traversal)
        
        Returns memory dicts sorted by relevance (direct links first).
        """
        # Step 1: Find matching entities
        result = await self.db.execute(
            select(Entity).where(
                and_(
                    Entity.user_id == user_id,
                    func.lower(Entity.name).contains(entity_name.lower()),
                )
            )
        )
        entities = list(result.scalars().all())
        
        if not entities:
            return []
        
        entity_ids = {e.id for e in entities}
        all_memory_ids = set()
        memory_depth = {}  # memory_id -> depth (lower = more relevant)
        
        for current_depth in range(depth):
            # Find memories linked to current entities
            result = await self.db.execute(
                select(EntityLink.memory_id, EntityLink.entity_id).where(
                    EntityLink.entity_id.in_(entity_ids)
                )
            )
            links = result.fetchall()
            
            new_memory_ids = set()
            for memory_id, entity_id in links:
                if memory_id not in all_memory_ids:
                    new_memory_ids.add(memory_id)
                    memory_depth[memory_id] = current_depth
            
            all_memory_ids.update(new_memory_ids)
            
            if current_depth < depth - 1 and new_memory_ids:
                # Find new entities connected through these memories
                result = await self.db.execute(
                    select(EntityLink.entity_id).where(
                        and_(
                            EntityLink.memory_id.in_(new_memory_ids),
                            ~EntityLink.entity_id.in_(entity_ids),
                        )
                    )
                )
                new_entity_ids = {row[0] for row in result.fetchall()}
                entity_ids.update(new_entity_ids)
        
        if not all_memory_ids:
            return []
        
        # Fetch actual memories
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id.in_(all_memory_ids),
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    Memory.is_active == True,
                )
            )
        )
        memories = list(result.scalars().all())
        
        # Score and sort: direct links first, then by importance
        scored = []
        for m in memories:
            d = memory_depth.get(m.id, depth)
            score = (1.0 / (1 + d)) * 0.5 + m.importance * 0.3 + m.strength * 0.2
            scored.append({
                "id": m.id,
                "content": m.content,
                "summary": m.summary,
                "brain_type": m.brain_type,
                "category": m.category,
                "importance": m.importance,
                "strength": m.strength,
                "graph_depth": d,
                "graph_score": round(score, 3),
                "created_at": m.created_at.isoformat() if m.created_at else None,
            })
        
        scored.sort(key=lambda x: x["graph_score"], reverse=True)
        return scored[:limit]
    
    async def find_similar_memories(
        self,
        user_id: str,
        content: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Tuple[Memory, float]]:
        """Find memories similar to given content. Uses pgvector when available."""
        embedding = self.embedding_service.embed(content)
        
        # Try pgvector-accelerated search
        if hasattr(Memory, 'embedding') and Memory.embedding is not None:
            try:
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = embedding  # raw list for pgvector
                query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.is_deleted == False,
                            Memory.embedding.isnot(None),
                        )
                    )
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(limit)
                )
                result = await self.db.execute(query)
                rows = result.all()
                
                similar = []
                for memory, similarity in rows:
                    if similarity >= min_similarity:
                        similar.append((memory, similarity))
                return similar
            except Exception:
                pass  # Fall through to Python-side search
        
        # Fallback: Python-side cosine similarity
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                )
            )
        )
        memories = result.scalars().all()
        
        similar = []
        for memory in memories:
            if memory.embedding_json:
                memory_embedding = json.loads(memory.embedding_json)
                similarity = self.embedding_service.cosine_similarity(
                    embedding, memory_embedding
                )
                if similarity >= min_similarity:
                    similar.append((memory, similarity))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:limit]
    
    async def store_entity_relationship(
        self,
        user_id: str,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        relationship: str,
        confidence: float = 0.7,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store an entity-entity relationship in the knowledge graph.
        
        Uses the dedicated entity_relationships table for efficient graph traversal.
        Also creates/updates a Memory record for backward compatibility and
        vector search (so the relationship is discoverable via semantic search too).
        """
        # Upsert source and target entities
        source_entity = await self._upsert_entity(user_id, source_name, source_type)
        target_entity = await self._upsert_entity(user_id, target_name, target_type)
        
        # Upsert into entity_relationships table (knowledge graph edge)
        result = await self.db.execute(
            select(EntityRelationship).where(
                and_(
                    EntityRelationship.source_entity_id == source_entity.id,
                    EntityRelationship.target_entity_id == target_entity.id,
                    EntityRelationship.relationship_type == relationship,
                )
            )
        )
        existing_rel = result.scalar_one_or_none()
        
        if existing_rel:
            existing_rel.mention_count += 1
            existing_rel.confidence = max(existing_rel.confidence, confidence)
            existing_rel.last_seen_at = datetime.utcnow()
            existing_rel.updated_at = datetime.utcnow()
        else:
            new_rel = EntityRelationship(
                id=str(uuid.uuid4()),
                user_id=user_id,
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relationship_type=relationship,
                relationship_label=f"{source_name} {relationship.replace('_', ' ')} {target_name}",
                confidence=confidence,
                mention_count=1,
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
            )
            self.db.add(new_rel)
        
        # Also maintain backward-compatible relationship memory for vector search
        relationship_content = f"{source_name} {relationship.replace('_', ' ')} {target_name}"
        existing = await self.find_similar_memories(
            user_id=user_id,
            content=relationship_content,
            limit=1,
            min_similarity=0.85,
        )
        
        if existing:
            mem, _ = existing[0]
            mem.access_count += 1
            mem.strength = min(1.0, mem.strength + 0.1)
            mem.last_accessed_at = datetime.utcnow()
            mem.confidence = max(mem.confidence, confidence)
        else:
            embedding = self.embedding_service.embed(relationship_content)
            
            rel_memory = Memory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=relationship_content,
                summary=f"{source_name} → {relationship} → {target_name}",
                brain_type="user",
                category="people" if source_type == "person" else "knowledge",
                memory_type="fact",
                memory_level="episodic",
                embedding_json=json.dumps(embedding),
                embedding=embedding,
                importance=0.6,
                confidence=confidence,
                strength=1.0,
                emotional_salience=0.3,
                source_type="entity_extraction",
                tags_json=json.dumps([source_type, target_type, "relationship"]),
                metadata_json=json.dumps({
                    "relationship_type": relationship,
                    "source_entity": source_name,
                    "target_entity": target_name,
                    "extracted_by": "llm",
                }),
            )
            self.db.add(rel_memory)
            await self.db.flush()
            
            # Link memory to both entities
            source_link = EntityLink(
                id=str(uuid.uuid4()),
                memory_id=rel_memory.id,
                entity_id=source_entity.id,
                role="subject",
            )
            target_link = EntityLink(
                id=str(uuid.uuid4()),
                memory_id=rel_memory.id,
                entity_id=target_entity.id,
                role="object",
            )
            self.db.add(source_link)
            self.db.add(target_link)
        
        await self.db.commit()
    
    async def _upsert_entity(
        self,
        user_id: str,
        name: str,
        entity_type: str,
        schema_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Find or create an entity by name and type.
        
        Phase 4: Now accepts schema_type (e.g. 'PersonEntity') and
        structured attributes dict to deep-merge into attributes_json.
        """
        result = await self.db.execute(
            select(Entity).where(
                and_(
                    Entity.user_id == user_id,
                    func.lower(Entity.name) == name.lower(),
                )
            )
        )
        entity = result.scalar_one_or_none()
        
        if entity:
            entity.mention_count += 1
            entity.last_seen_at = datetime.utcnow()
            # Update type if we have a more specific one
            if entity_type != "unknown" and entity.entity_type == "unknown":
                entity.entity_type = entity_type
            # Update schema_type if provided and not set yet
            if schema_type and not entity.schema_type:
                entity.schema_type = schema_type
            # Deep-merge attributes
            if attributes:
                existing = {}
                if entity.attributes_json:
                    try:
                        existing = json.loads(entity.attributes_json)
                    except (json.JSONDecodeError, TypeError):
                        existing = {}
                # Merge: new values overwrite, but don't erase existing keys
                merged = {**existing, **{k: v for k, v in attributes.items() if v is not None}}
                entity.attributes_json = json.dumps(merged)
        else:
            entity = Entity(
                id=str(uuid.uuid4()),
                user_id=user_id,
                name=name,
                entity_type=entity_type,
                schema_type=schema_type,
                attributes_json=json.dumps(attributes) if attributes else None,
            )
            self.db.add(entity)
            await self.db.flush()
        
        return entity
    
    async def _link_memories(
        self,
        source_id: str,
        target_ids: List[str],
        relationship_type: str = "related_to"
    ):
        """Link memories together"""
        for target_id in target_ids:
            await self.db.execute(
                memory_relationships.insert().values(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                    strength=1.0,
                    created_at=datetime.utcnow()
                )
            )
        await self.db.commit()
    
    async def _update_brain_stats(self, user_id: str):
        """Update cached brain statistics"""
        # Count memories by category
        result = await self.db.execute(
            select(Memory.category, func.count(Memory.id))
            .where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                )
            )
            .group_by(Memory.category)
        )
        category_counts = {row[0]: row[1] for row in result.fetchall()}
        
        # Count total
        total_memories = sum(category_counts.values())
        
        # Count entities
        result = await self.db.execute(
            select(func.count(Entity.id)).where(Entity.user_id == user_id)
        )
        total_entities = result.scalar() or 0
        
        # Count connections
        result = await self.db.execute(
            select(func.count())
            .select_from(memory_relationships)
            .join(Memory, Memory.id == memory_relationships.c.source_id)
            .where(Memory.user_id == user_id)
        )
        total_connections = result.scalar() or 0
        
        # Normalize sizes for visualization
        max_count = max(category_counts.values()) if category_counts else 1
        category_sizes = {
            cat: count / max_count
            for cat, count in category_counts.items()
        }
        
        # Upsert brain stats
        result = await self.db.execute(
            select(BrainStats).where(BrainStats.user_id == user_id)
        )
        stats = result.scalar_one_or_none()
        
        if stats:
            stats.region_counts_json = json.dumps(category_counts)
            stats.region_sizes_json = json.dumps(category_sizes)
            stats.total_memories = total_memories
            stats.total_entities = total_entities
            stats.total_connections = total_connections
            stats.updated_at = datetime.utcnow()
        else:
            stats = BrainStats(
                user_id=user_id,
                region_counts_json=json.dumps(category_counts),
                region_sizes_json=json.dumps(category_sizes),
                total_memories=total_memories,
                total_entities=total_entities,
                total_connections=total_connections,
            )
            self.db.add(stats)
        
        await self.db.commit()
    
    async def _log_memory_event(
        self,
        memory_id: str,
        user_id: str,
        event_type: MemoryEventType,
        event_data: Dict[str, Any],
        trigger_source: str
    ) -> MemoryEvent:
        """
        Log an event to the immutable memory audit trail.
        This is called automatically by create/update/delete operations.
        """
        event = MemoryEvent(
            id=str(uuid.uuid4()),
            memory_id=memory_id,
            user_id=user_id,
            event_type=event_type.value,
            timestamp=datetime.utcnow(),
            event_data_json=json.dumps(event_data) if event_data else None,
            trigger_source=trigger_source,
        )
        self.db.add(event)
        # Don't commit here - let the calling method handle the transaction
        return event
    
    async def get_memory_events(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 100
    ) -> List[MemoryEvent]:
        """Get the event audit trail for a memory."""
        result = await self.db.execute(
            select(MemoryEvent).where(
                and_(
                    MemoryEvent.memory_id == memory_id,
                    MemoryEvent.user_id == user_id,
                )
            ).order_by(MemoryEvent.timestamp.desc()).limit(limit)
        )
        return list(result.scalars().all())
    
    def _generate_explanation(self, memory: Memory, similarity: float, weighted_score: Optional[float] = None) -> str:
        """Generate explanation for why a memory was retrieved"""
        parts = []
        
        # Similarity explanation
        if similarity >= 0.9:
            parts.append("Very high semantic match")
        elif similarity >= 0.7:
            parts.append("Good semantic match")
        else:
            parts.append("Partial semantic match")
        
        parts.append(f"({similarity:.0%} similar)")
        
        # NEW: Add strength/decay info if using weighted scoring
        if weighted_score is not None:
            if memory.strength >= 0.8:
                parts.append("• Strong memory")
            elif memory.strength < 0.4:
                parts.append("• Fading memory")
            
            if memory.memory_level == "semantic":
                parts.append("• Consolidated knowledge")
            elif memory.memory_level == "procedural":
                parts.append("• Procedural skill")
        
        # Add category context
        category_names = {
            "identity": "Identity",
            "preferences": "Preferences",
            "beliefs": "Beliefs",
            "emotions": "Emotions",
            "people": "People",
            "places": "Places",
            "family": "Family",
            "experiences": "Experiences",
            "projects": "Projects",
            "schedule": "Schedule",
            "work": "Work",
            "learning": "Learning",
            "knowledge": "Knowledge",
            "tools": "Tools",
            "media": "Media",
            "health": "Health",
            "habits": "Habits",
            "food": "Food",
            "travel": "Travel",
            "goals": "Goals",
            "context": "Context",
        }
        category_name = category_names.get(memory.category, memory.category)
        parts.append(f"• From {category_name}")
        
        # Add recency
        age_days = (datetime.utcnow() - memory.created_at).days
        if age_days == 0:
            parts.append("• Created today")
        elif age_days == 1:
            parts.append("• Created yesterday")
        elif age_days < 7:
            parts.append(f"• Created {age_days} days ago")
        
        return " ".join(parts)

    # ==================================================================
    # HYBRID RETRIEVAL ENGINE — Multi-strategy search with RRF fusion
    # ==================================================================

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[List[Tuple[str, float]]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked result lists using Reciprocal Rank Fusion.
        
        For each document d, its RRF score is:
        RRF(d) = Σ 1/(k + rank_i(d))  for each ranked list i
        
        k=60 is the standard constant from the RRF paper (Cormack et al., 2009).
        """
        scores: Dict[str, float] = defaultdict(float)
        for ranked_list in ranked_lists:
            for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    async def hybrid_search(
        self,
        user_id: str,
        query: str,
        limit: int = 15,
        min_similarity: float = 0.3,
        brain_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        strategies: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Multi-strategy memory retrieval with Reciprocal Rank Fusion.
        
        Strategies:
          - "vector": pgvector cosine similarity (semantic search)
          - "keyword": tsvector full-text search (BM25-style keyword matching)
          - "graph": entity graph traversal (knowledge graph walk)
          - "temporal": recency-weighted search (auto-activated on time queries)
        
        Temporal reasoning: Automatically detects time references in queries
        (e.g., "last week", "yesterday", "in January") and applies date filters
        to all strategies + adds a temporal strategy for recency-weighted results.
        
        Results from all strategies are fused with RRF, then weighted
        by Brain's proprietary scoring (strength, importance, emotion, recency).
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if strategies is None:
            strategies = ["vector", "keyword", "graph"]
        
        # Build common filter conditions
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.is_active == True,
        ]
        if brain_types:
            conditions.append(Memory.brain_type.in_(brain_types))
        if categories:
            conditions.append(Memory.category.in_(categories))
        if created_after:
            conditions.append(Memory.created_at >= created_after)
        if created_before:
            conditions.append(Memory.created_at <= created_before)
        
        # Phase 7: Auto-detect temporal intent from query
        temporal = extract_temporal_filters(query)
        temporal_after = created_after  # Preserve explicit params
        temporal_before = created_before
        if temporal["has_temporal"]:
            logger.info(f"[HYBRID] Temporal detected: {temporal['temporal_keywords']} → after={temporal.get('created_after')}, before={temporal.get('created_before')}")
            if temporal["created_after"] and not created_after:
                conditions.append(Memory.created_at >= temporal["created_after"])
                temporal_after = temporal["created_after"]
            if temporal["created_before"] and not created_before:
                conditions.append(Memory.created_at <= temporal["created_before"])
                temporal_before = temporal["created_before"]
        
        # Generate query embedding once (shared by vector strategy)
        query_embedding = self.embedding_service.embed(query)
        
        # Run enabled strategies in parallel
        tasks = {}
        fetch_limit = limit * 4  # Over-fetch per strategy for re-ranker input
        
        if "vector" in strategies:
            tasks["vector"] = self._vector_search(
                user_id, query_embedding, fetch_limit, conditions
            )
        if "keyword" in strategies:
            tasks["keyword"] = self._keyword_search(
                user_id, query, fetch_limit, conditions
            )
        if "graph" in strategies:
            tasks["graph"] = self._graph_search(
                user_id, query, fetch_limit
            )
        if "temporal" in strategies or temporal.get("has_temporal"):
            tasks["temporal"] = self._temporal_search(
                user_id, query, fetch_limit, temporal_after, temporal_before
            )
        
        # Execute all strategies concurrently
        task_names = list(tasks.keys())
        task_coros = list(tasks.values())
        
        try:
            results = await asyncio.gather(*task_coros, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Hybrid search gather failed: {e}")
            results = []
        
        # Collect valid ranked lists
        ranked_lists = []
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Strategy '{name}' failed: {result}")
                continue
            if result:
                ranked_lists.append(result)
                logger.info(f"[HYBRID] Strategy '{name}' returned {len(result)} results")
        
        if not ranked_lists:
            return []
        
        # Fuse with RRF
        fused = self.reciprocal_rank_fusion(ranked_lists)
        fused_ids = [doc_id for doc_id, _ in fused[:limit * 3]]  # Keep 3x for re-ranker
        
        if not fused_ids:
            return []
        
        # Fetch the actual Memory objects for the top results
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id.in_(fused_ids),
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    Memory.is_active == True,
                )
            )
        )
        memories_map = {m.id: m for m in result.scalars().all()}
        
        # Compute per-memory vector similarity (for scoring + response)
        similarity_map: Dict[str, float] = {}
        if hasattr(Memory, 'embedding') and Memory.embedding is not None and fused_ids:
            try:
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = query_embedding  # raw list for pgvector
                sim_query = (
                    select(
                        Memory.id,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("sim"),
                    )
                    .where(
                        and_(
                            Memory.id.in_(fused_ids),
                            Memory.embedding.isnot(None),
                        )
                    )
                )
                sim_result = await self.db.execute(sim_query)
                for row in sim_result.all():
                    similarity_map[row.id] = row.sim
            except Exception:
                pass  # Fall through — similarity will be 0
        
        # Apply Brain's weighted scoring on top of RRF
        rrf_scores = dict(fused)
        scored_memories = []
        
        for memory_id in fused_ids:
            memory = memories_map.get(memory_id)
            if not memory:
                continue
            
            similarity = similarity_map.get(memory_id, 0.0)
            rrf_score = rrf_scores.get(memory_id, 0.0)
            
            # Weighted score: combines semantic similarity + memory quality
            final_score = (
                similarity * 0.30 +
                rrf_score * 10.0 * 0.25 +  # Normalize RRF (typically 0.01-0.05 range)
                memory.strength * 0.20 +
                memory.importance * 0.15 +
                memory.emotional_salience * 0.10
            )
            
            # Recency boost
            if memory.last_accessed_at:
                days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                if days_since_access < 7:
                    final_score += (7 - days_since_access) / 7 * 0.1
            
            scored_memories.append({
                "id": memory.id,
                "content": memory.content,
                "summary": memory.summary,
                "brain_type": memory.brain_type,
                "category": memory.category,
                "memory_type": memory.memory_type,
                "importance": memory.importance,
                "confidence": memory.confidence,
                "strength": memory.strength,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                "access_count": memory.access_count,
                "source_type": memory.source_type,
                "similarity_score": similarity,
                "final_score": final_score,
                "retrieval_strategies": [n for n, r in zip(task_names, results) if not isinstance(r, Exception)],
            })
        
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)

        # Phase 6: Cross-encoder re-ranking for improved precision
        try:
            from app.config import settings as _settings
            if _settings.enable_reranker and len(scored_memories) > limit:
                from app.services.reranker_service import get_reranker_service
                reranker = get_reranker_service()
                reranked = await reranker.rerank(
                    query=query,
                    candidates=scored_memories,
                    top_k=limit,
                )
                logger.info(
                    f"[HYBRID] Re-ranked {len(scored_memories)}→{len(reranked)} candidates"
                )
                return reranked
        except Exception as e:
            logger.warning(f"[HYBRID] Re-ranker failed, using score-based ranking: {e}")

        return scored_memories[:limit]

    async def _vector_search(
        self,
        user_id: str,
        query_embedding: List[float],
        limit: int,
        conditions: list,
    ) -> List[Tuple[str, float]]:
        """
        pgvector cosine similarity search.
        Returns [(memory_id, similarity_score)] sorted by similarity desc.
        """
        if not (hasattr(Memory, 'embedding') and Memory.embedding is not None):
            return []
        
        try:
            extra_conditions = list(conditions) + [Memory.embedding.isnot(None)]
            # pgvector accepts Python list directly (no string conversion needed)
            embedding_vec = query_embedding  # raw list for pgvector
            
            query = (
                select(
                    Memory.id,
                    (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity"),
                )
                .where(and_(*extra_conditions))
                .order_by(Memory.embedding.cosine_distance(embedding_vec))
                .limit(limit)
            )
            
            result = await self.db.execute(query)
            return [(row.id, row.similarity) for row in result.all() if row.similarity > 0.05]
        except Exception:
            return []

    async def _keyword_search(
        self,
        user_id: str,
        query: str,
        limit: int,
        conditions: list,
    ) -> List[Tuple[str, float]]:
        """
        PostgreSQL tsvector full-text search with ts_rank scoring.
        Falls back to ILIKE if tsvector column doesn't exist.
        Returns [(memory_id, rank_score)] sorted by rank desc.
        """
        # Clean query for tsquery — remove special chars
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not clean_query:
            return []
        
        try:
            # Try tsvector path first
            ts_query = func.plainto_tsquery('english', clean_query)
            
            stmt = (
                select(
                    Memory.id,
                    func.ts_rank(Memory.search_vector, ts_query).label("rank"),
                )
                .where(
                    and_(
                        *conditions,
                        Memory.search_vector.op('@@')(ts_query),
                    )
                )
                .order_by(func.ts_rank(Memory.search_vector, ts_query).desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            rows = result.all()
            if rows:
                return [(row.id, float(row.rank)) for row in rows]
        except Exception:
            pass  # Fall through to ILIKE
        
        # Fallback: simple ILIKE search
        try:
            keywords = clean_query.split()[:5]  # Max 5 keywords
            ilike_conditions = list(conditions)
            for kw in keywords:
                if len(kw) >= 3:
                    ilike_conditions.append(
                        or_(
                            Memory.content.ilike(f"%{kw}%"),
                            Memory.summary.ilike(f"%{kw}%"),
                        )
                    )
            
            if len(ilike_conditions) == len(conditions):
                return []  # No valid keywords
            
            stmt = (
                select(Memory.id)
                .where(and_(*ilike_conditions))
                .order_by(Memory.importance.desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            # Give ILIKE results a uniform rank (no ts_rank available)
            return [(row.id, 0.5) for row in result.all()]
        except Exception:
            return []

    async def _graph_search(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """
        Entity-graph search using recursive CTE for multi-hop traversal.
        
        1. Extract entity names from query
        2. Find seed entities in DB
        3. Traverse EntityRelationship edges (up to 2 hops) via recursive CTE
        4. Find memories linked to all discovered entities
        5. Score: direct matches > 1-hop > 2-hop
        
        Returns [(memory_id, graph_score)] sorted by graph_score desc.
        """
        # Step 1: Extract candidate entity names from the query
        entity_names = self._extract_entity_names(query)
        if not entity_names:
            return []
        
        # Step 2: Find matching entities in the database
        seed_entity_ids: set = set()
        for name in entity_names:
            result = await self.db.execute(
                select(Entity.id).where(
                    and_(
                        Entity.user_id == user_id,
                        func.lower(Entity.name).contains(name.lower()),
                    )
                )
            )
            for row in result.all():
                seed_entity_ids.add(row.id)
        
        if not seed_entity_ids:
            return []
        
        # Step 3: Recursive CTE traversal through entity_relationships (up to 2 hops)
        traversal_results = await self.traverse_entity_graph(
            user_id=user_id,
            seed_entity_ids=list(seed_entity_ids),
            max_depth=2,
            limit=limit * 3,
        )
        
        # Build entity→depth map (depth 0 = seed, 1 = 1-hop, 2 = 2-hop)
        entity_depth: Dict[str, int] = {}
        for eid in seed_entity_ids:
            entity_depth[eid] = 0
        for item in traversal_results:
            eid = item["entity_id"]
            depth = item["depth"]
            if eid not in entity_depth or depth < entity_depth[eid]:
                entity_depth[eid] = depth
        
        all_entity_ids = set(entity_depth.keys())
        
        # Step 4: Find memories linked to these entities via EntityLink
        if not all_entity_ids:
            return []
        
        result = await self.db.execute(
            select(EntityLink.memory_id, EntityLink.entity_id).where(
                EntityLink.entity_id.in_(all_entity_ids)
            )
        )
        links = result.all()
        
        memory_scores: Dict[str, float] = defaultdict(float)
        for memory_id, entity_id in links:
            depth = entity_depth.get(entity_id, 3)
            # Decay score by depth: direct=1.0, 1-hop=0.5, 2-hop=0.25
            memory_scores[memory_id] += 1.0 / (1 + depth)
        
        # Filter to active, non-deleted memories owned by this user
        if memory_scores:
            result = await self.db.execute(
                select(Memory.id).where(
                    and_(
                        Memory.id.in_(list(memory_scores.keys())),
                        Memory.user_id == user_id,
                        Memory.is_deleted == False,
                        Memory.is_active == True,
                    )
                )
            )
            valid_ids = {row.id for row in result.all()}
            memory_scores = {k: v for k, v in memory_scores.items() if k in valid_ids}
        
        # Sort by score descending
        scored = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return scored[:limit]


    async def _temporal_search(
        self,
        user_id: str,
        query: str,
        limit: int,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
    ) -> List[Tuple[str, float]]:
        """
        Temporal retrieval strategy — prioritizes recency within a time window.
        
        When a user asks "what happened recently?" or "what did I discuss last week?",
        this strategy surfaces memories sorted by recency + importance, independent
        of semantic similarity. This catches memories that vector search might miss
        because the query terms don't semantically match the content.
        
        Score formula:
            score = recency_weight * (1 / (1 + days_ago)) + importance * 0.3 + strength * 0.2
        
        Returns [(memory_id, temporal_score)] sorted by score desc.
        """
        try:
            conditions = [
                Memory.user_id == user_id,
                Memory.is_deleted == False,
                Memory.is_active == True,
            ]
            
            if created_after:
                conditions.append(Memory.created_at >= created_after)
            if created_before:
                conditions.append(Memory.created_at <= created_before)
            
            stmt = (
                select(
                    Memory.id,
                    Memory.created_at,
                    Memory.importance,
                    Memory.strength,
                    Memory.emotional_salience,
                )
                .where(and_(*conditions))
                .order_by(Memory.created_at.desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            rows = result.all()
            
            if not rows:
                return []
            
            now = datetime.utcnow()
            scored = []
            for row in rows:
                days_ago = max(0.01, (now - row.created_at).total_seconds() / 86400.0)
                recency_score = 1.0 / (1.0 + days_ago)
                importance = row.importance if row.importance else 0.5
                strength = row.strength if row.strength else 0.5
                emotion = row.emotional_salience if row.emotional_salience else 0.0
                
                temporal_score = (
                    recency_score * 0.50 +
                    importance * 0.25 +
                    strength * 0.15 +
                    emotion * 0.10
                )
                scored.append((row.id, temporal_score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        
        except Exception:
            return []

    async def traverse_entity_graph(
        self,
        user_id: str,
        seed_entity_ids: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> List[dict]:
        """
        Multi-hop BFS traversal through EntityRelationship edges using
        a PostgreSQL recursive CTE for maximum efficiency.
        
        Accepts either seed_entity_ids (pre-resolved) or entity_names (to resolve).
        Returns a list of dicts: {entity_id, entity_name, entity_type, depth,
            relationship_type, relationship_label, from_entity_id, from_entity_name}
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Resolve entity names to IDs if needed
        if seed_entity_ids is None and entity_names:
            seed_entity_ids = []
            for name in entity_names:
                result = await self.db.execute(
                    select(Entity.id).where(
                        and_(
                            Entity.user_id == user_id,
                            func.lower(Entity.name).contains(name.lower()),
                        )
                    )
                )
                for row in result.all():
                    seed_entity_ids.append(row.id)
        
        if not seed_entity_ids:
            return []
        
        # Build the recursive CTE query
        # This traverses entity_relationships in BOTH directions (bidirectional graph)
        seed_ids_str = ",".join(f"'{eid}'" for eid in seed_entity_ids)
        
        cte_sql = text(f"""
            WITH RECURSIVE graph_walk AS (
                -- Base case: seed entities at depth 0
                SELECT
                    e.id AS entity_id,
                    e.name AS entity_name,
                    e.entity_type AS entity_type,
                    0 AS depth,
                    NULL::varchar AS relationship_type,
                    NULL::varchar AS relationship_label,
                    NULL::varchar AS from_entity_id,
                    NULL::varchar AS from_entity_name
                FROM entities e
                WHERE e.id IN ({seed_ids_str})
                  AND e.user_id = :user_id

                UNION ALL

                -- Forward edges: source → target
                SELECT
                    e2.id AS entity_id,
                    e2.name AS entity_name,
                    e2.entity_type AS entity_type,
                    gw.depth + 1 AS depth,
                    er.relationship_type,
                    er.relationship_label,
                    gw.entity_id AS from_entity_id,
                    gw.entity_name AS from_entity_name
                FROM graph_walk gw
                JOIN entity_relationships er ON er.source_entity_id = gw.entity_id
                JOIN entities e2 ON e2.id = er.target_entity_id
                WHERE gw.depth < :max_depth
                  AND er.user_id = :user_id

                UNION ALL

                -- Reverse edges: target → source (bidirectional traversal)
                SELECT
                    e2.id AS entity_id,
                    e2.name AS entity_name,
                    e2.entity_type AS entity_type,
                    gw.depth + 1 AS depth,
                    er.relationship_type,
                    er.relationship_label,
                    gw.entity_id AS from_entity_id,
                    gw.entity_name AS from_entity_name
                FROM graph_walk gw
                JOIN entity_relationships er ON er.target_entity_id = gw.entity_id
                JOIN entities e2 ON e2.id = er.source_entity_id
                WHERE gw.depth < :max_depth
                  AND er.user_id = :user_id
            )
            SELECT DISTINCT ON (entity_id)
                entity_id, entity_name, entity_type, depth,
                relationship_type, relationship_label,
                from_entity_id, from_entity_name
            FROM graph_walk
            ORDER BY entity_id, depth ASC
            LIMIT :limit
        """)
        
        try:
            result = await self.db.execute(
                cte_sql,
                {"user_id": user_id, "max_depth": max_depth, "limit": limit},
            )
            rows = result.fetchall()
            
            return [
                {
                    "entity_id": row.entity_id,
                    "entity_name": row.entity_name,
                    "entity_type": row.entity_type,
                    "depth": row.depth,
                    "relationship_type": row.relationship_type,
                    "relationship_label": row.relationship_label,
                    "from_entity_id": row.from_entity_id,
                    "from_entity_name": row.from_entity_name,
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Recursive CTE graph traversal failed: {e}")
            # Fallback to simple 1-hop if CTE fails
            expanded = set(seed_entity_ids)
            try:
                result = await self.db.execute(
                    select(
                        EntityRelationship.target_entity_id,
                        EntityRelationship.relationship_type,
                    ).where(EntityRelationship.source_entity_id.in_(seed_entity_ids))
                )
                for row in result.all():
                    expanded.add(row.target_entity_id)
                
                result = await self.db.execute(
                    select(
                        EntityRelationship.source_entity_id,
                        EntityRelationship.relationship_type,
                    ).where(EntityRelationship.target_entity_id.in_(seed_entity_ids))
                )
                for row in result.all():
                    expanded.add(row.source_entity_id)
            except Exception:
                pass
            
            # Fetch entity details
            result = await self.db.execute(
                select(Entity.id, Entity.name, Entity.entity_type).where(
                    Entity.id.in_(expanded)
                )
            )
            return [
                {
                    "entity_id": row.id,
                    "entity_name": row.name,
                    "entity_type": row.entity_type,
                    "depth": 0 if row.id in seed_entity_ids else 1,
                    "relationship_type": None,
                    "relationship_label": None,
                    "from_entity_id": None,
                    "from_entity_name": None,
                }
                for row in result.all()
            ]

    async def get_entity_relationships(
        self,
        user_id: str,
        entity_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Get entity relationships, optionally filtered by entity or type.
        Returns relationship details with source/target entity info.
        """
        conditions = [EntityRelationship.user_id == user_id]
        if entity_id:
            conditions.append(
                or_(
                    EntityRelationship.source_entity_id == entity_id,
                    EntityRelationship.target_entity_id == entity_id,
                )
            )
        if relationship_type:
            conditions.append(EntityRelationship.relationship_type == relationship_type)
        
        result = await self.db.execute(
            select(EntityRelationship)
            .where(and_(*conditions))
            .order_by(EntityRelationship.mention_count.desc())
            .limit(limit)
        )
        rels = result.scalars().all()
        
        # Batch-load referenced entities
        entity_ids = set()
        for r in rels:
            entity_ids.add(r.source_entity_id)
            entity_ids.add(r.target_entity_id)
        
        entities_map: Dict[str, Entity] = {}
        if entity_ids:
            result = await self.db.execute(
                select(Entity).where(Entity.id.in_(entity_ids))
            )
            for e in result.scalars().all():
                entities_map[e.id] = e
        
        output = []
        for r in rels:
            src = entities_map.get(r.source_entity_id)
            tgt = entities_map.get(r.target_entity_id)
            output.append({
                "id": r.id,
                "relationship_type": r.relationship_type,
                "relationship_label": r.relationship_label,
                "confidence": r.confidence,
                "mention_count": r.mention_count,
                "first_seen_at": r.first_seen_at.isoformat() if r.first_seen_at else None,
                "last_seen_at": r.last_seen_at.isoformat() if r.last_seen_at else None,
                "properties": json.loads(r.properties_json) if r.properties_json else None,
                "source_entity": {
                    "id": src.id,
                    "name": src.name,
                    "entity_type": src.entity_type,
                } if src else None,
                "target_entity": {
                    "id": tgt.id,
                    "name": tgt.name,
                    "entity_type": tgt.entity_type,
                } if tgt else None,
            })
        
        return output

    async def get_entities(
        self,
        user_id: str,
        entity_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        List entities for a user, optionally filtered by type or name search.
        """
        conditions = [Entity.user_id == user_id]
        if entity_type:
            conditions.append(Entity.entity_type == entity_type)
        if search:
            conditions.append(
                func.lower(Entity.name).contains(search.lower())
            )
        
        result = await self.db.execute(
            select(Entity)
            .where(and_(*conditions))
            .order_by(Entity.mention_count.desc())
            .limit(limit)
        )
        entities = result.scalars().all()
        
        output = []
        for e in entities:
            output.append({
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type,
                "description": e.description,
                "mention_count": e.mention_count,
                "first_seen_at": e.first_seen_at.isoformat() if e.first_seen_at else None,
                "last_seen_at": e.last_seen_at.isoformat() if e.last_seen_at else None,
                "attributes": json.loads(e.attributes_json) if e.attributes_json else None,
            })
        
        return output

    @staticmethod
    def _extract_entity_names(query: str) -> List[str]:
        """
        Extract candidate entity names from a query string.
        
        Uses heuristics:
        1. Capitalized words/phrases (proper nouns)
        2. Quoted strings
        3. Words after entity-indicating prepositions
        
        Fast and local — no LLM call needed.
        """
        names = set()
        
        # 1. Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        names.update(quoted)
        quoted_single = re.findall(r"'([^']+)'", query)
        names.update(quoted_single)
        
        # 2. Extract capitalized words/phrases (consecutive capitalized words)
        # Skip first word (may be capitalized because it starts a sentence)
        words = query.split()
        if len(words) > 1:
            cap_phrase = []
            for word in words[1:]:
                clean = re.sub(r'[^\w]', '', word)
                if clean and clean[0].isupper() and len(clean) > 1:
                    cap_phrase.append(clean)
                else:
                    if cap_phrase:
                        names.add(' '.join(cap_phrase))
                        cap_phrase = []
            if cap_phrase:
                names.add(' '.join(cap_phrase))
        
        # 3. Also check the first word if it looks like a name (>= 2 chars, capitalized)
        if words:
            first = re.sub(r'[^\w]', '', words[0])
            if first and first[0].isupper() and len(first) > 1 and first.lower() not in {
                'what', 'when', 'where', 'who', 'how', 'why', 'the', 'this', 'that',
                'which', 'tell', 'show', 'find', 'get', 'do', 'does', 'did', 'is',
                'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should',
                'have', 'has', 'had', 'my', 'your', 'their', 'our', 'its',
            }:
                names.add(first)
        
        return list(names)

    # ==================================================================
    # END HYBRID RETRIEVAL ENGINE
    # ==================================================================

    async def search_memories_by_embedding(
        self,
        user_id: str,
        embedding: List[float],
        limit: int = 15,
        min_similarity: float = 0.5,
        brain_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Search memories using a pre-computed embedding.
        Uses pgvector for fast similarity search when available.
        
        Args:
            user_id: User ID to search memories for
            embedding: Pre-computed query embedding
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            brain_types: Optional list of brain types to filter by
            categories: Optional list of categories to filter by
            created_after: Optional temporal filter — only memories created after this datetime
            created_before: Optional temporal filter — only memories created before this datetime
        
        Returns:
            List of memory dicts with similarity scores
        """
        # Build base conditions
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.is_active == True,
        ]
        
        if brain_types:
            conditions.append(Memory.brain_type.in_(brain_types))
        if categories:
            conditions.append(Memory.category.in_(categories))
        if created_after:
            conditions.append(Memory.created_at >= created_after)
        if created_before:
            conditions.append(Memory.created_at <= created_before)
        
        scored_memories = []
        
        # Try pgvector-accelerated search
        if hasattr(Memory, 'embedding') and Memory.embedding is not None:
            try:
                conditions.append(Memory.embedding.isnot(None))
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = embedding  # raw list for pgvector
                
                query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(and_(*conditions))
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(limit * 2)  # Over-fetch for weighted re-ranking
                )
                
                result = await self.db.execute(query)
                rows = result.all()
                
                for memory, similarity in rows:
                    if similarity < min_similarity:
                        continue
                    
                    final_score = (
                        similarity * 0.4 +
                        memory.strength * 0.25 +
                        memory.importance * 0.2 +
                        memory.emotional_salience * 0.15
                    )
                    
                    if memory.last_accessed_at:
                        days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                        if days_since_access < 7:
                            recency_boost = (7 - days_since_access) / 7 * 0.1
                            final_score += recency_boost
                    
                    scored_memories.append({
                        "id": memory.id,
                        "content": memory.content,
                        "summary": memory.summary,
                        "brain_type": memory.brain_type,
                        "category": memory.category,
                        "memory_type": memory.memory_type,
                        "importance": memory.importance,
                        "confidence": memory.confidence,
                        "strength": memory.strength,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                        "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                        "access_count": memory.access_count,
                        "source_type": memory.source_type,
                        "similarity_score": similarity,
                        "final_score": final_score,
                    })
                
                scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
                return scored_memories[:limit]
            except Exception:
                scored_memories = []  # Fall through to Python-side search
        
        # Fallback: Python-side cosine similarity
        # Remove the pgvector-specific condition if it was added
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.is_active == True,
        ]
        if brain_types:
            conditions.append(Memory.brain_type.in_(brain_types))
        if categories:
            conditions.append(Memory.category.in_(categories))
        if created_after:
            conditions.append(Memory.created_at >= created_after)
        if created_before:
            conditions.append(Memory.created_at <= created_before)
        
        query = select(Memory).where(and_(*conditions))
        result = await self.db.execute(query)
        memories = result.scalars().all()
        
        for memory in memories:
            if memory.embedding_json:
                memory_embedding = json.loads(memory.embedding_json)
                similarity = self.embedding_service.cosine_similarity(
                    embedding, memory_embedding
                )
                
                if similarity >= min_similarity:
                    final_score = (
                        similarity * 0.4 +
                        memory.strength * 0.25 +
                        memory.importance * 0.2 +
                        memory.emotional_salience * 0.15
                    )
                    
                    if memory.last_accessed_at:
                        days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                        if days_since_access < 7:
                            recency_boost = (7 - days_since_access) / 7 * 0.1
                            final_score += recency_boost
                    
                    scored_memories.append({
                        "id": memory.id,
                        "content": memory.content,
                        "summary": memory.summary,
                        "brain_type": memory.brain_type,
                        "category": memory.category,
                        "memory_type": memory.memory_type,
                        "importance": memory.importance,
                        "confidence": memory.confidence,
                        "strength": memory.strength,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                        "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                        "access_count": memory.access_count,
                        "source_type": memory.source_type,
                        "similarity_score": similarity,
                        "final_score": final_score,
                    })
        
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_memories[:limit]
