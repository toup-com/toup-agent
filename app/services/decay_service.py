"""
Memory Decay Service - Implements Ebbinghaus forgetting curve

This service handles memory strength decay based on cognitive science principles:
- R = e^(-t/S) where R=retention, t=time, S=stability
- Spaced repetition strengthens memories
- Emotional salience resists decay
- Important memories decay slower
"""

import json
import math
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.db.models import Memory, MemoryEvent, MemoryEventType


class DecayService:
    """
    Applies memory decay based on the Ebbinghaus Forgetting Curve.
    
    The forgetting curve formula: R = e^(-t/S)
    Where:
    - R = retention (our "strength" field)
    - t = time since last reinforcement
    - S = stability (inverse of decay_rate, modified by importance and emotional_salience)
    """
    
    # Base half-life in days (time for memory to decay to 50% strength)
    BASE_HALF_LIFE_DAYS = 7.0
    
    # Minimum strength before memory is considered "forgotten"
    MIN_STRENGTH = 0.1
    
    # Maximum strength after reinforcement
    MAX_STRENGTH = 1.0
    
    # How much each reinforcement increases strength
    REINFORCEMENT_BOOST = 0.25
    
    # How much importance affects decay resistance (0-1 importance adds 0-100% to stability)
    IMPORTANCE_DECAY_MODIFIER = 1.0
    
    # How much emotional salience affects decay resistance
    EMOTIONAL_DECAY_MODIFIER = 0.5
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def apply_decay_to_user(
        self,
        user_id: str,
        batch_size: int = 100
    ) -> Tuple[int, int]:
        """
        Apply decay to all memories for a user.
        Returns (memories_processed, memories_updated).
        """
        processed = 0
        updated = 0
        
        # Get all non-deleted memories with strength > MIN_STRENGTH
        query = select(Memory).where(
            and_(
                Memory.user_id == user_id,
                Memory.is_deleted == False,
                Memory.strength > self.MIN_STRENGTH
            )
        ).order_by(Memory.last_accessed_at.asc().nullsfirst())
        
        result = await self.db.execute(query)
        memories = result.scalars().all()
        
        now = datetime.utcnow()
        
        for memory in memories:
            processed += 1
            
            # Calculate new strength
            old_strength = memory.strength
            new_strength = self._calculate_decayed_strength(memory, now)
            
            # Only update if strength changed significantly (>1%)
            if abs(new_strength - old_strength) > 0.01:
                memory.strength = new_strength
                updated += 1
                
                # Log decay event
                await self._log_event(
                    memory_id=memory.id,
                    user_id=user_id,
                    event_type=MemoryEventType.DECAYED,
                    event_data={
                        "old_strength": round(old_strength, 4),
                        "new_strength": round(new_strength, 4),
                        "decay_amount": round(old_strength - new_strength, 4),
                    },
                    trigger_source="scheduled"
                )
        
        await self.db.commit()
        return processed, updated
    
    def _calculate_decayed_strength(
        self,
        memory: Memory,
        current_time: datetime
    ) -> float:
        """
        Calculate the new strength of a memory based on time elapsed.
        
        Uses modified Ebbinghaus formula with individual factors.
        """
        # Determine reference time (last reinforcement or creation)
        reference_time = memory.last_reinforced_at or memory.last_accessed_at or memory.created_at
        
        # Calculate time elapsed in days
        time_delta = current_time - reference_time
        days_elapsed = time_delta.total_seconds() / (24 * 3600)
        
        if days_elapsed <= 0:
            return memory.strength
        
        # Calculate stability factor (higher = slower decay)
        # Base stability modified by importance, emotional salience, and individual decay rate
        stability = self.BASE_HALF_LIFE_DAYS / memory.decay_rate
        
        # Importance adds decay resistance (0-1 importance adds 0-100% to stability)
        stability *= (1 + memory.importance * self.IMPORTANCE_DECAY_MODIFIER)
        
        # Emotional salience adds decay resistance
        stability *= (1 + memory.emotional_salience * self.EMOTIONAL_DECAY_MODIFIER)
        
        # Consolidated memories decay slower
        if memory.consolidation_count > 0:
            stability *= (1 + 0.2 * min(memory.consolidation_count, 5))
        
        # Semantic memories decay slower than episodic
        if memory.memory_level == "semantic":
            stability *= 2.0
        elif memory.memory_level == "procedural":
            stability *= 1.5
        elif memory.memory_level == "meta":
            stability *= 3.0  # Meta-knowledge is very stable
        
        # Apply decay formula: R = R_0 * e^(-t/S)
        # Using natural decay: strength = old_strength * e^(-days/stability)
        decay_factor = math.exp(-days_elapsed / stability)
        new_strength = memory.strength * decay_factor
        
        # Ensure minimum threshold
        return max(new_strength, self.MIN_STRENGTH)
    
    async def reinforce_memory(
        self,
        memory_id: str,
        user_id: str,
        access_context: str = "recall"
    ) -> Optional[Memory]:
        """
        Reinforce a memory (strengthen it) when accessed or recalled.
        
        This implements spaced repetition principles:
        - Each access increases strength
        - More recent memories get larger boosts
        - Diminishing returns for very frequent access
        """
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
        
        if not memory:
            return None
        
        old_strength = memory.strength
        now = datetime.utcnow()
        
        # Calculate reinforcement boost based on current strength
        # Lower strength = larger boost (helping weak memories more)
        strength_factor = 1 - memory.strength  # 0 to 1, higher for weaker memories
        boost = self.REINFORCEMENT_BOOST * (1 + strength_factor)
        
        # Diminishing returns for very frequent access
        if memory.last_reinforced_at:
            hours_since_last = (now - memory.last_reinforced_at).total_seconds() / 3600
            if hours_since_last < 1:
                boost *= 0.1  # Very small boost for rapid re-access
            elif hours_since_last < 24:
                boost *= 0.5  # Reduced boost within same day
        
        # Apply boost
        new_strength = min(old_strength + boost, self.MAX_STRENGTH)
        
        # Update memory
        memory.strength = new_strength
        memory.last_reinforced_at = now
        memory.last_accessed_at = now
        memory.access_count += 1
        
        # Adjust decay rate based on access pattern (frequent access = slower decay)
        # This implements the spacing effect
        if memory.access_count > 5:
            memory.decay_rate = max(0.05, memory.decay_rate * 0.95)
        
        # Log reinforcement event
        await self._log_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.REINFORCED,
            event_data={
                "old_strength": round(old_strength, 4),
                "new_strength": round(new_strength, 4),
                "boost_amount": round(new_strength - old_strength, 4),
                "access_context": access_context,
                "access_count": memory.access_count,
            },
            trigger_source="api"
        )
        
        await self.db.commit()
        return memory
    
    async def get_weak_memories(
        self,
        user_id: str,
        threshold: float = 0.3,
        limit: int = 50
    ) -> List[Memory]:
        """
        Get memories that have decayed below a threshold.
        Useful for review suggestions or cleanup.
        """
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    Memory.strength < threshold
                )
            ).order_by(Memory.strength.asc()).limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_memories_to_review(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Memory]:
        """
        Get memories that would benefit from review (spaced repetition).
        Prioritizes memories that:
        - Have moderate importance but declining strength
        - Haven't been accessed recently
        - Are approaching decay threshold
        """
        # Target strength range: memories that are starting to decay but not forgotten
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    Memory.strength.between(0.3, 0.7),
                    Memory.importance > 0.4
                )
            ).order_by(
                # Prioritize by combination of declining strength and importance
                (Memory.importance * (1 - Memory.strength)).desc()
            ).limit(limit)
        )
        return list(result.scalars().all())
    
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


def get_decay_service(db: AsyncSession) -> DecayService:
    """Factory function for DecayService."""
    return DecayService(db)
