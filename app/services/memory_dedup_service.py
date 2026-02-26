"""
Memory Deduplication and Evolution Service

This service handles:
1. Detecting duplicate/similar memories before insertion
2. Merging related memories into evolving records
3. Tracking history of memory changes
4. Generating change summaries using LLM
"""

import json
import logging
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.memory_service import MemoryService
from app.services.embedding_service import get_embedding_service
from app.services.llm_service import get_llm_service
from app.schemas import MemoryCreate, MemoryResponse, BrainType
from app.db.models import Memory

logger = logging.getLogger(__name__)

# Similarity thresholds - used as initial filter, LLM makes final decision
# These are deliberately LOW to catch potential matches, LLM decides what to do
CANDIDATE_THRESHOLD = 0.40  # Consider as potential match for LLM analysis
MIN_THRESHOLD = 0.25        # Below this, definitely not related


class MemoryDedupService:
    """
    Service for intelligent memory deduplication and evolution.
    
    This service provides smart memory management by:
    - Detecting duplicates before insertion
    - Merging related information into existing memories
    - Tracking how memories evolve over time
    - Using LLM to intelligently combine information
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.memory_service = MemoryService(db)
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()
    
    async def smart_create_memory(
        self,
        new_memory: MemoryCreate,
        user_id: str
    ) -> Tuple[Memory, str]:
        """
        Intelligently create or merge a memory.
        
        This is the main entry point for creating memories with deduplication.
        It checks for similar existing memories and decides whether to:
        - Create a new memory
        - Reinforce an existing one (exact duplicate)
        - Merge with an existing one (similar but with new info)
        
        Args:
            new_memory: The memory data to create
            user_id: The user's ID
            
        Returns:
            Tuple of (memory, action) where action is one of:
            - "created": New memory created
            - "merged": Merged into existing memory
            - "skipped": Exact duplicate, no action taken (returns existing)
            - "reinforced": Same info, just strengthened existing
        """
        # Step 1: Generate embedding for new content
        new_embedding = self.embedding_service.embed(new_memory.content)
        
        # Handle brain_type - can be enum or string
        brain_type_value = (
            new_memory.brain_type.value 
            if hasattr(new_memory.brain_type, 'value') 
            else new_memory.brain_type
        ) if new_memory.brain_type else 'user'
        
        # Handle category - can be enum or string
        category_value = (
            new_memory.category.value 
            if hasattr(new_memory.category, 'value') 
            else new_memory.category
        )
        
        # Step 2: Search for similar existing memories (cast wide net)
        similar_memories = await self.memory_service.search_memories_by_embedding(
            user_id=user_id,
            embedding=new_embedding,
            limit=5,
            min_similarity=MIN_THRESHOLD,
            brain_types=[brain_type_value] if brain_type_value else None,
            categories=None  # Search across ALL categories to find related memories
        )
        
        if not similar_memories:
            # No similar memories - create new
            memory = await self.memory_service.create_memory(
                user_id=user_id,
                memory_data=new_memory,
                deduplicate=False  # We already checked
            )
            logger.info(f"Created new memory: {memory.id}")
            return memory, "created"
        
        # Step 3: Filter candidates above threshold
        candidates = [m for m in similar_memories if m.get('similarity_score', 0) >= CANDIDATE_THRESHOLD]
        
        if not candidates:
            # Nothing similar enough to consider
            memory = await self.memory_service.create_memory(
                user_id=user_id,
                memory_data=new_memory,
                deduplicate=False
            )
            logger.info(f"No candidates above threshold, created new memory: {memory.id}")
            return memory, "created"
        
        # Step 4: Use LLM to decide what to do with the best candidate
        top_match = candidates[0]
        similarity = top_match.get('similarity_score', 0)
        existing_content = top_match.get('content', '')
        
        logger.info(f"Found candidate with similarity {similarity:.3f}: {existing_content[:50]}...")
        
        # Ask LLM to decide: duplicate, merge, or new
        decision = await self._llm_decide_action(
            existing_content=existing_content,
            new_content=new_memory.content
        )
        
        logger.info(f"LLM decision: {decision['action']} - {decision.get('reason', '')[:50]}")
        
        if decision['action'] == 'duplicate':
            # Same information, just reinforce
            existing = await self._get_memory_by_id(top_match['id'])
            if existing:
                reinforced = await self._reinforce_existing_memory(existing, new_memory)
                return reinforced, "reinforced"
        
        elif decision['action'] == 'merge':
            # Related with new info, merge them
            try:
                merged = await self._merge_memories(
                    existing_memory_id=top_match['id'],
                    new_content=new_memory.content,
                    new_memory_data=new_memory,
                    user_id=user_id
                )
                return merged, "merged"
            except Exception as e:
                logger.error(f"Merge failed: {e}, creating new memory instead")
        
        # decision['action'] == 'new' or merge failed - create new memory
        memory = await self.memory_service.create_memory(
            user_id=user_id,
            memory_data=new_memory,
            deduplicate=False
        )
        logger.info(f"Created new memory: {memory.id}")
        return memory, "created"
    
    async def _get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID without user check (internal use)."""
        from sqlalchemy import select
        result = await self.db.execute(
            select(Memory).where(Memory.id == memory_id)
        )
        return result.scalar_one_or_none()
    
    async def _reinforce_existing_memory(
        self,
        memory: Memory,
        new_data: MemoryCreate
    ) -> Memory:
        """Reinforce an existing memory with new occurrence."""
        # Increase strength (capped at 1.0)
        memory.strength = min(1.0, (memory.strength or 0.5) + 0.1)
        
        # Update importance if the new one is higher
        if new_data.importance > (memory.importance or 0.5):
            memory.importance = new_data.importance
        
        # Update confidence if the new one is higher
        if new_data.confidence > (memory.confidence or 0.5):
            memory.confidence = new_data.confidence
        
        # Track reinforcement
        memory.last_reinforced_at = datetime.utcnow()
        memory.consolidation_count = (memory.consolidation_count or 0) + 1
        memory.access_count = (memory.access_count or 0) + 1
        memory.updated_at = datetime.utcnow()
        
        # Add to history
        history = json.loads(memory.history_json) if memory.history_json else []
        history.append({
            "date": datetime.utcnow().isoformat(),
            "content": memory.canonical_content or memory.content,
            "source": new_data.source_type or "conversation",
            "action": "reinforced",
            "change_summary": "Memory reinforced (duplicate occurrence)"
        })
        memory.history_json = json.dumps(history)
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def _merge_memories(
        self,
        existing_memory_id: str,
        new_content: str,
        new_memory_data: MemoryCreate,
        user_id: str
    ) -> Memory:
        """
        Merge new information into an existing memory.
        
        1. Get the existing memory
        2. Use LLM to generate merged content
        3. Generate change summary
        4. Update memory with new canonical_content and history entry
        5. Re-generate embedding for the merged content
        """
        # Get existing memory
        existing = await self._get_memory_by_id(existing_memory_id)
        if not existing:
            raise ValueError(f"Memory {existing_memory_id} not found")
        
        existing_content = existing.canonical_content or existing.content
        
        # Check if the new content actually adds information
        # If it's essentially the same, just reinforce
        if self._is_same_information(existing_content, new_content):
            return await self._reinforce_existing_memory(existing, new_memory_data)
        
        # Use LLM to merge the contents
        merged_content, change_summary = await self._llm_merge_contents(
            existing_content=existing_content,
            new_content=new_content
        )
        
        # If merge didn't produce new content, just reinforce
        if merged_content.strip() == existing_content.strip():
            return await self._reinforce_existing_memory(existing, new_memory_data)
        
        # Update the memory using memory_service method
        updated = await self.memory_service.merge_memory(
            memory_id=existing_memory_id,
            new_content=merged_content,
            change_summary=change_summary,
            source_type=new_memory_data.source_type or "merge"
        )
        
        # Update importance if new info is significant
        if new_memory_data.importance and new_memory_data.importance > (updated.importance or 0.5):
            updated.importance = new_memory_data.importance
            await self.db.commit()
            await self.db.refresh(updated)
        
        return updated
    
    def _is_same_information(self, content1: str, content2: str) -> bool:
        """
        Quick check if two pieces of content have the same information.
        Uses simple normalization and comparison.
        """
        def normalize(s: str) -> str:
            return ' '.join(s.lower().split())
        
        n1, n2 = normalize(content1), normalize(content2)
        
        # If one contains the other, they might have the same info
        if n1 in n2 or n2 in n1:
            # If lengths are similar, probably same info
            if abs(len(n1) - len(n2)) < 20:
                return True
        
        return n1 == n2
    
    async def _llm_decide_action(
        self,
        existing_content: str,
        new_content: str
    ) -> Dict[str, str]:
        """
        Use LLM to decide what action to take with a potential duplicate/related memory.
        
        This is the "smart" part - instead of relying on similarity thresholds,
        we ask the LLM to understand the semantic relationship.
        
        Returns:
            Dict with 'action' (duplicate|merge|new) and 'reason'
        """
        prompt = f"""You are a memory management system. Compare these two pieces of information and decide what to do.

EXISTING MEMORY:
"{existing_content}"

NEW INFORMATION:
"{new_content}"

Decide ONE of these actions:

1. "duplicate" - The new information says the EXACT SAME FACT, just worded differently.
   Example: "I love pizza" vs "Pizza is my favorite food" → duplicate
   Example: "My name is John" vs "I'm John" → duplicate

2. "merge" - The new information ADDS DETAILS to the SAME SPECIFIC TOPIC.
   Example: "I love pizza" vs "I love pepperoni pizza from Dominos" → merge (both about pizza preference)
   Example: "I play basketball" vs "I play basketball as point guard on Saturdays" → merge (both about basketball)
   
3. "new" - The information is about a DIFFERENT topic, even if about the same person.
   Example: "My name is John" vs "I love pizza" → new (name vs food = different topics)
   Example: "I love pizza" vs "I have a dog named Max" → new (food vs pet = different topics)
   Example: "I play basketball" vs "My favorite movie is Inception" → new (sport vs movie = different topics)
   Example: "My birthday is Feb 19" vs "I work as an engineer" → new (birthday vs job = different topics)

CRITICAL: Just because two facts mention the same person does NOT mean they should merge.
Only merge if they are about the SAME SPECIFIC TOPIC (e.g., both about food, both about sports, both about work).

Respond in JSON:
{{
    "action": "duplicate|merge|new",
    "reason": "Brief explanation"
}}"""

        try:
            response = await self.llm_service.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini"  # Fast and cheap for decisions
            )
            
            # Parse response
            if hasattr(response, 'content'):
                import json as json_module
                parsed = json_module.loads(response.content)
            else:
                parsed = response
            
            action = parsed.get("action", "new")
            reason = parsed.get("reason", "")
            
            # Validate action
            if action not in ["duplicate", "merge", "new"]:
                action = "new"
            
            return {"action": action, "reason": reason}
            
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            # Fallback: use simple heuristic
            if self._is_same_information(existing_content, new_content):
                return {"action": "duplicate", "reason": "Fallback: text similarity"}
            return {"action": "new", "reason": f"Fallback due to error: {e}"}
    
    async def _llm_merge_contents(
        self,
        existing_content: str,
        new_content: str
    ) -> Tuple[str, str]:
        """
        Use LLM to intelligently merge two pieces of information.
        
        Returns:
            Tuple of (merged_content, change_summary)
        """
        prompt = f"""You are merging two pieces of information about the same topic into one coherent memory.

EXISTING MEMORY:
{existing_content}

NEW INFORMATION:
{new_content}

Tasks:
1. Combine these into a single, coherent statement that includes all information from both.
2. Summarize what new information was added.

Respond in this exact JSON format:
{{
    "merged_content": "The combined memory statement",
    "change_summary": "Brief summary of what was added/changed"
}}

Rules:
- Keep it concise but complete
- Don't lose any information from either source
- If there's a contradiction, prefer the new information but note it
- The merged content should read naturally as a single fact/memory
- If the new information doesn't add anything new, just return the existing content"""

        try:
            response = await self.llm_service.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini"  # Use cheaper model for merging
            )
            
            # Parse JSON response
            if hasattr(response, 'content'):
                import json as json_module
                parsed = json_module.loads(response.content)
            else:
                parsed = response
            
            merged = parsed.get("merged_content", new_content)
            summary = parsed.get("change_summary", "Updated with new information")
            
            return merged, summary
            
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            # Fallback: simple concatenation
            if new_content not in existing_content:
                merged = f"{existing_content} Additionally, {new_content}"
                return merged, f"Added: {new_content[:50]}..."
            return existing_content, "No changes (duplicate)"
    
    async def find_and_merge_duplicates(
        self,
        user_id: str,
        category: Optional[str] = None,
        brain_type: Optional[str] = None,
        dry_run: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan existing memories and find/merge duplicates.
        
        Useful for cleaning up existing data.
        
        Args:
            user_id: User whose memories to scan
            category: Optional category filter
            brain_type: Optional brain type filter
            dry_run: If True, just report what would be merged without doing it
            
        Returns:
            List of merge operations (performed or proposed)
        """
        # Get all active memories for user
        memories = await self.memory_service.get_user_memories(
            user_id=user_id,
            category=category,
            brain_type=brain_type,
            is_active=True
        )
        
        merge_operations = []
        processed_ids = set()
        
        for memory in memories:
            if memory.id in processed_ids:
                continue
            
            # Get embedding
            if not memory.embedding_json:
                continue
            embedding = json.loads(memory.embedding_json)
            
            # Find similar memories
            similar = await self.memory_service.search_memories_by_embedding(
                user_id=user_id,
                embedding=embedding,
                limit=10,
                min_similarity=CANDIDATE_THRESHOLD
            )
            
            # Filter out self and already processed
            duplicates = [
                s for s in similar 
                if s['id'] != memory.id and s['id'] not in processed_ids
            ]
            
            if duplicates:
                operation = {
                    "primary_memory": {
                        "id": str(memory.id),
                        "content": memory.canonical_content or memory.content
                    },
                    "duplicates": [
                        {
                            "id": str(d['id']),
                            "content": d.get('content', ''),
                            "similarity": d.get('similarity_score', 0)
                        }
                        for d in duplicates
                    ],
                    "action": "proposed" if dry_run else "merged"
                }
                
                if not dry_run:
                    # Actually merge them
                    for dup in duplicates:
                        try:
                            # Merge duplicate into primary
                            await self._merge_memories(
                                existing_memory_id=memory.id,
                                new_content=dup.get('content', ''),
                                new_memory_data=MemoryCreate(
                                    content=dup.get('content', ''),
                                    category=memory.category,
                                    memory_type=memory.memory_type,
                                    brain_type=BrainType(memory.brain_type) if memory.brain_type else BrainType.USER,
                                    source_type="dedup_merge"
                                ),
                                user_id=user_id
                            )
                            
                            # Mark duplicate as superseded
                            await self.memory_service.supersede_memory(
                                old_memory_id=dup['id'],
                                new_memory_id=memory.id
                            )
                            
                            processed_ids.add(dup['id'])
                            logger.info(f"Merged memory {dup['id']} into {memory.id}")
                            
                        except Exception as e:
                            logger.error(f"Failed to merge {dup['id']} into {memory.id}: {e}")
                            operation["errors"] = operation.get("errors", []) + [str(e)]
                
                merge_operations.append(operation)
                processed_ids.add(memory.id)
        
        return merge_operations
    
    async def get_duplicate_report(
        self,
        user_id: str,
        category: Optional[str] = None,
        threshold: float = CANDIDATE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Generate a report of potential duplicates without making changes.
        
        Args:
            user_id: User whose memories to scan
            category: Optional category filter
            threshold: Similarity threshold for considering duplicates
            
        Returns:
            Report with statistics and duplicate groups
        """
        operations = await self.find_and_merge_duplicates(
            user_id=user_id,
            category=category,
            dry_run=True
        )
        
        total_memories = len(await self.memory_service.get_user_memories(
            user_id=user_id,
            category=category,
            is_active=True
        ))
        
        total_duplicates = sum(len(op.get("duplicates", [])) for op in operations)
        
        return {
            "total_memories": total_memories,
            "duplicate_groups": len(operations),
            "total_duplicates": total_duplicates,
            "potential_reduction": total_duplicates,
            "groups": operations
        }


# Singleton instance
_dedup_service: Optional[MemoryDedupService] = None


def get_memory_dedup_service(db: AsyncSession) -> MemoryDedupService:
    """Get or create a MemoryDedupService instance."""
    return MemoryDedupService(db)
