"""
Memory Consolidation Service - Promotes episodic memories to semantic

This service implements the cognitive science principle of memory consolidation:
- Episodic memories (specific events) get consolidated into semantic memories (general knowledge)
- Memories that are accessed frequently and consistently tend to consolidate
- Consolidation creates summary/abstraction from multiple related episodic memories
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.db.models import Memory, MemoryEvent, MemoryEventType, MemoryLevel, memory_relationships
from app.services.embedding_service import EmbeddingService, get_embedding_service


class ConsolidationService:
    """
    Consolidates episodic memories into semantic memories.
    
    The consolidation process:
    1. Find episodic memories that have been accessed multiple times
    2. Group similar memories using embedding similarity
    3. Create or update semantic memory summaries
    4. Link episodic memories to their semantic parent
    5. Optionally "promote" the memory level
    """
    
    # Minimum access count before considering consolidation
    MIN_ACCESS_COUNT = 3
    
    # Minimum age before considering consolidation (days)
    MIN_AGE_DAYS = 1
    
    # Similarity threshold for grouping memories
    SIMILARITY_THRESHOLD = 0.7
    
    # Minimum group size for consolidation
    MIN_GROUP_SIZE = 2
    
    # Maximum memories to process per run
    MAX_MEMORIES_PER_RUN = 100
    
    def __init__(self, db: AsyncSession, embedding_service: Optional[EmbeddingService] = None):
        self.db = db
        self.embedding_service = embedding_service or get_embedding_service()
    
    async def run_consolidation(
        self,
        user_id: str,
        force: bool = False
    ) -> Tuple[int, int, int]:
        """
        Run the consolidation process for a user.
        
        Returns (memories_considered, groups_found, memories_consolidated).
        """
        # Find episodic memories eligible for consolidation
        cutoff_date = datetime.utcnow() - timedelta(days=self.MIN_AGE_DAYS)
        
        query = select(Memory).where(
            and_(
                Memory.user_id == user_id,
                Memory.is_deleted == False,
                Memory.memory_level == "episodic",
                Memory.access_count >= self.MIN_ACCESS_COUNT if not force else True,
                Memory.created_at <= cutoff_date if not force else True,
            )
        ).order_by(Memory.access_count.desc()).limit(self.MAX_MEMORIES_PER_RUN)
        
        result = await self.db.execute(query)
        episodic_memories = list(result.scalars().all())
        
        if not episodic_memories:
            return 0, 0, 0
        
        # Group memories by category first (only consolidate within same category)
        category_groups = defaultdict(list)
        for memory in episodic_memories:
            category_groups[memory.category].append(memory)
        
        total_groups = 0
        total_consolidated = 0
        
        for category, memories in category_groups.items():
            if len(memories) < self.MIN_GROUP_SIZE:
                continue
            
            # Find similar memories within this category
            similar_groups = await self._find_similar_groups(memories)
            total_groups += len(similar_groups)
            
            # Process each group
            for group in similar_groups:
                consolidated = await self._consolidate_group(user_id, group)
                total_consolidated += consolidated
        
        await self.db.commit()
        return len(episodic_memories), total_groups, total_consolidated
    
    async def _find_similar_groups(
        self,
        memories: List[Memory]
    ) -> List[List[Memory]]:
        """
        Group memories by semantic similarity using pgvector.
        Falls back to Python-side cosine if pgvector is unavailable.
        Uses a simple greedy clustering approach.
        """
        if len(memories) < self.MIN_GROUP_SIZE:
            return []
        
        # Build a map of memory objects by ID
        memory_map = {m.id: m for m in memories}
        memory_ids = list(memory_map.keys())
        
        # Try pgvector-accelerated pairwise similarity
        use_pgvector = hasattr(Memory, 'embedding') and Memory.embedding is not None
        similarity_cache: dict = {}
        
        if use_pgvector:
            try:
                # For each memory, find similar ones in the group using pgvector
                for anchor in memories:
                    if anchor.embedding is None:
                        continue
                    embedding_str = f"[{','.join(str(x) for x in anchor.embedding)}]"
                    result = await self.db.execute(
                        select(
                            Memory.id,
                            (1 - Memory.embedding.cosine_distance(embedding_str)).label("sim"),
                        )
                        .where(
                            and_(
                                Memory.id.in_(memory_ids),
                                Memory.id != anchor.id,
                                Memory.embedding.isnot(None),
                            )
                        )
                    )
                    for row in result.all():
                        pair = tuple(sorted([anchor.id, row.id]))
                        if pair not in similarity_cache:
                            similarity_cache[pair] = row.sim
            except Exception:
                similarity_cache = {}  # Fall through to Python-side
        
        # Fallback: Python-side cosine similarity
        if not similarity_cache:
            embeddings = {}
            for memory in memories:
                if memory.embedding_json:
                    embeddings[memory.id] = json.loads(memory.embedding_json)
            
            if len(embeddings) < self.MIN_GROUP_SIZE:
                return []
            
            ids_with_emb = [mid for mid in memory_ids if mid in embeddings]
            for i, id_a in enumerate(ids_with_emb):
                for id_b in ids_with_emb[i+1:]:
                    sim = self.embedding_service.cosine_similarity(
                        embeddings[id_a], embeddings[id_b]
                    )
                    pair = tuple(sorted([id_a, id_b]))
                    similarity_cache[pair] = sim
        
        # Greedy clustering using cached similarities
        groups = []
        used = set()
        
        for anchor in memories:
            if anchor.id in used:
                continue
            
            group = [anchor]
            used.add(anchor.id)
            
            for candidate in memories:
                if candidate.id in used:
                    continue
                pair = tuple(sorted([anchor.id, candidate.id]))
                sim = similarity_cache.get(pair, 0.0)
                if sim >= self.SIMILARITY_THRESHOLD:
                    group.append(candidate)
                    used.add(candidate.id)
            
            if len(group) >= self.MIN_GROUP_SIZE:
                groups.append(group)
        
        return groups
    
    async def _consolidate_group(
        self,
        user_id: str,
        group: List[Memory]
    ) -> int:
        """
        Consolidate a group of similar episodic memories.
        
        Options:
        1. Create a new semantic memory summarizing the group
        2. Promote the most important episodic memory to semantic
        3. Link all episodic memories to the semantic one
        """
        if len(group) < self.MIN_GROUP_SIZE:
            return 0
        
        # Find the most important/accessed memory in the group
        primary = max(group, key=lambda m: m.importance * m.access_count)
        
        # Check if there's already a semantic memory linked to this group
        existing_semantic = await self._find_linked_semantic(group)
        
        consolidated_count = 0
        now = datetime.utcnow()
        
        if existing_semantic:
            # Update existing semantic memory
            semantic_memory = existing_semantic
            semantic_memory.consolidation_count += 1
            semantic_memory.updated_at = now
            
            # Update content to reflect new consolidated information
            semantic_memory.content = await self._create_consolidated_content(group, semantic_memory.content)
            
            # Re-embed to match updated content
            fresh_embedding = self.embedding_service.embed(semantic_memory.content)
            semantic_memory.embedding_json = json.dumps(fresh_embedding)
            semantic_memory.embedding = fresh_embedding
            
        else:
            # Create new semantic memory from the group
            consolidated_content = await self._create_consolidated_content(group)
            
            # Generate fresh embedding for the consolidated content (not stale primary embedding)
            fresh_embedding = self.embedding_service.embed(consolidated_content)
            
            semantic_memory = Memory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=consolidated_content,
                summary=f"Consolidated from {len(group)} memories: {primary.summary or primary.content[:50]}",
                category=primary.category,
                memory_type=primary.memory_type,
                memory_level="semantic",
                embedding_json=json.dumps(fresh_embedding),
                embedding=fresh_embedding,  # Native pgvector — matches content
                importance=max(m.importance for m in group),
                confidence=sum(m.confidence for m in group) / len(group),
                strength=max(m.strength for m in group),
                emotional_salience=max(m.emotional_salience for m in group),
                consolidation_count=1,
                decay_rate=0.05,  # Semantic memories decay slower
                source_type="consolidation",
                created_at=now,
                tags_json=self._merge_tags(group),
            )
            self.db.add(semantic_memory)
            await self.db.flush()
        
        # Link all episodic memories to the semantic memory
        for memory in group:
            if memory.id == semantic_memory.id:
                continue
            
            # Create relationship link
            await self._create_relationship(
                memory.id,
                semantic_memory.id,
                "consolidated_into"
            )
            
            # Update episodic memory's consolidation count
            memory.consolidation_count += 1
            memory.updated_at = now
            
            # Log consolidation event
            await self._log_event(
                memory_id=memory.id,
                user_id=user_id,
                event_type=MemoryEventType.CONSOLIDATED,
                event_data={
                    "semantic_memory_id": semantic_memory.id,
                    "group_size": len(group),
                    "from_level": memory.memory_level,
                    "to_level": "linked_to_semantic",
                },
                trigger_source="consolidation"
            )
            
            consolidated_count += 1
        
        # Log event for the semantic memory
        await self._log_event(
            memory_id=semantic_memory.id,
            user_id=user_id,
            event_type=MemoryEventType.CREATED if not existing_semantic else MemoryEventType.UPDATED,
            event_data={
                "source": "consolidation",
                "episodic_memory_ids": [m.id for m in group],
                "group_size": len(group),
            },
            trigger_source="consolidation"
        )
        
        return consolidated_count
    
    async def _create_consolidated_content(
        self,
        group: List[Memory],
        existing_content: Optional[str] = None
    ) -> str:
        """Create consolidated content from a group of memories using LLM summarization.
        
        Uses GPT-4o-mini to produce a proper semantic summary that captures
        all unique information from the group, rather than string concatenation.
        Falls back to simple concatenation if LLM fails.
        """
        contents = [m.content for m in group]
        
        # Try LLM-powered summarization
        try:
            from app.services.llm_service import get_llm_service
            llm = get_llm_service()
            
            # Build the prompt
            memories_text = "\n".join(f"- {c}" for c in contents)
            
            if existing_content:
                prompt = f"""You are a memory consolidation system. An existing semantic memory needs to be updated with new episodic memories.

EXISTING SEMANTIC MEMORY:
{existing_content}

NEW EPISODIC MEMORIES TO INCORPORATE:
{memories_text}

Write a single, comprehensive paragraph that merges ALL information from the existing semantic memory and the new episodic memories into one cohesive summary. 
- Preserve EVERY unique fact, name, date, preference, and detail — do NOT drop information.
- Remove redundancy — if the same fact appears multiple times, state it once.
- Write in third person about the user (e.g., "The user prefers..." or "Nariman works at...").
- The result should be a standalone summary that needs no additional context.
- Maximum 300 words."""
            else:
                prompt = f"""You are a memory consolidation system. Multiple related episodic memories need to be consolidated into a single semantic memory.

EPISODIC MEMORIES:
{memories_text}

Write a single, comprehensive paragraph that captures ALL unique information from these memories into one cohesive summary.
- Preserve EVERY unique fact, name, date, preference, and detail — do NOT drop information.
- Remove redundancy — if the same fact appears multiple times, state it once.
- Write in third person about the user (e.g., "The user prefers..." or "Nariman works at...").
- The result should be a standalone summary that needs no additional context.
- Maximum 300 words."""

            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            consolidated = response.content.strip()
            if len(consolidated) > 20:  # Sanity check
                return consolidated[:5000]
        except Exception as e:
            import logging
            logging.warning(f"LLM consolidation failed, falling back to concatenation: {e}")
        
        # Fallback: simple concatenation (original behavior)
        if existing_content:
            base = existing_content + "\n\n[Updated consolidation]\n"
        else:
            base = "[Consolidated memory]\n"
        
        seen_content = set()
        unique_parts = []
        for content in contents:
            key = content[:50].lower().strip()
            if key not in seen_content:
                seen_content.add(key)
                unique_parts.append(content)
        
        consolidated = base + "\n- ".join(unique_parts[:10])
        return consolidated[:5000]
    
    def _merge_tags(self, group: List[Memory]) -> str:
        """Merge tags from all memories in the group."""
        all_tags = set()
        for memory in group:
            if memory.tags_json:
                tags = json.loads(memory.tags_json)
                all_tags.update(tags)
        return json.dumps(list(all_tags)[:20])  # Limit to 20 tags
    
    async def _find_linked_semantic(
        self,
        group: List[Memory]
    ) -> Optional[Memory]:
        """Find if there's already a semantic memory linked to this group."""
        group_ids = [m.id for m in group]
        
        # Look for existing consolidation relationships
        result = await self.db.execute(
            select(Memory).join(
                memory_relationships,
                memory_relationships.c.target_id == Memory.id
            ).where(
                and_(
                    memory_relationships.c.source_id.in_(group_ids),
                    memory_relationships.c.relationship_type == "consolidated_into",
                    Memory.memory_level == "semantic",
                    Memory.is_deleted == False,
                )
            ).limit(1)
        )
        return result.scalar_one_or_none()
    
    async def _create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str
    ) -> None:
        """Create a relationship between memories."""
        # Check if relationship already exists
        result = await self.db.execute(
            select(memory_relationships).where(
                and_(
                    memory_relationships.c.source_id == source_id,
                    memory_relationships.c.target_id == target_id,
                )
            )
        )
        if result.first() is not None:
            return  # Already exists
        
        # Insert new relationship
        await self.db.execute(
            memory_relationships.insert().values(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                strength=1.0,
                created_at=datetime.utcnow(),
            )
        )
    
    async def promote_to_semantic(
        self,
        memory_id: str,
        user_id: str
    ) -> Optional[Memory]:
        """
        Manually promote an episodic memory to semantic level.
        Used when a memory is identified as general knowledge.
        """
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id == memory_id,
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                )
            )
        )
        memory = result.scalar_one_or_none()
        
        if not memory:
            return None
        
        old_level = memory.memory_level
        
        # Promote to semantic
        memory.memory_level = "semantic"
        memory.consolidation_count += 1
        memory.decay_rate = max(0.05, memory.decay_rate * 0.5)  # Slower decay
        memory.updated_at = datetime.utcnow()
        
        # Log event
        await self._log_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.CONSOLIDATED,
            event_data={
                "from_level": old_level,
                "to_level": "semantic",
                "manual_promotion": True,
            },
            trigger_source="api"
        )
        
        await self.db.commit()
        return memory
    
    async def _log_event(
        self,
        memory_id: str,
        user_id: str,
        event_type: MemoryEventType,
        event_data: dict,
        trigger_source: str
    ) -> MemoryEvent:
        """Log an event to the immutable audit trail."""
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
        return event


def get_consolidation_service(
    db: AsyncSession,
    embedding_service: Optional[EmbeddingService] = None
) -> ConsolidationService:
    """Factory function for ConsolidationService."""
    return ConsolidationService(db, embedding_service)
