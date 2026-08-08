"""
Memory Decay Service - Implements Ebbinghaus forgetting curve

This service handles memory strength decay based on cognitive science principles:
- R = e^(-t/S) where R=retention, t=time, S=stability
- Spaced repetition strengthens memories
- Emotional salience resists decay
- Important memories decay slower
"""

import json
import logging
import math
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.db.models import Memory, MemoryEvent, MemoryEventType

logger = logging.getLogger(__name__)


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
        batch_size: int = 100,
        now: Optional[datetime] = None,
    ) -> Tuple[int, int]:
        """
        Apply decay to all memories for a user.
        Returns (memories_processed, memories_updated).

        `now` is the instant the whole batch is evaluated at. It defaults to
        utcnow() and exists as an explicit seam: the batch must share ONE
        timestamp (a memory's new strength and the `last_decayed_at` stamp
        that says when it was accurate have to agree exactly), and tests need
        to advance the clock across several passes to prove the decay curve
        does not compound.
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

        if now is None:
            now = datetime.utcnow()

        for memory in memories:
            processed += 1

            # Calculate new strength
            old_strength = memory.strength
            reference = self._decay_reference_time(memory)
            new_strength = self._calculate_decayed_strength(memory, now)

            # Only update if strength changed significantly (>1%)
            if abs(new_strength - old_strength) > 0.01:
                memory.strength = new_strength
                # Advance the decay clock IN LOCKSTEP with the write. The
                # stored strength is now the value as of `now`, so the next
                # pass must measure elapsed time from here — otherwise it
                # re-applies the whole elapsed exponent to an already-decayed
                # number and the memory is forgotten at a rate set by the job
                # interval instead of by the forgetting curve.
                #
                # Deliberately INSIDE the branch: when the change is below the
                # 1% write threshold nothing was written, so the strength is
                # still only accurate as of the old reference and the clock
                # must NOT move. Skipped passes then accumulate into the next
                # one that does write, and the product is still a single
                # curve over the whole interval.
                memory.last_decayed_at = now
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
                        # The interval this pass actually charged for. Two
                        # consecutive DECAYED events whose days_elapsed both
                        # measure back to the same reinforcement are the
                        # signature of the compounding bug.
                        "days_elapsed": None if reference is None else round(
                            (now - reference).total_seconds() / 86400.0, 4
                        ),
                    },
                    trigger_source="scheduled"
                )

        await self.db.commit()
        return processed, updated
    
    def _decay_reference_time(self, memory: Memory) -> Optional[datetime]:
        """
        The instant the CURRENTLY STORED `strength` was accurate as of.

        `_calculate_decayed_strength` multiplies the stored strength by
        e^(-elapsed/stability), so "elapsed" has to be measured from the last
        time something WROTE that number — not from the last time the memory
        was reinforced. Every writer of `strength` leaves a timestamp:

            creation        → created_at
            reinforcement   → last_reinforced_at (and last_accessed_at)
            a decay pass    → last_decayed_at

        so the reference is the latest of those. Before `last_decayed_at`
        existed the decay pass wrote strength and left no mark, and run N
        charged the memory for the WHOLE span since the last reinforcement
        all over again: R0·e^(-t1/S)·e^(-t2/S)·… instead of R0·e^(-t/S). The
        accumulated exponent grew with the number of runs, so halving the job
        interval doubled the forgetting.

        `last_accessed_at` keeps its existing place in the precedence chain
        (`or`, not max, so a fixture that back-dates only last_reinforced_at
        still decays); `last_decayed_at` is then folded in with an explicit
        comparison because it is the only one of the four that can legitimately
        be newer than the rest.

        Returns None only for a row with no usable timestamp at all — possible
        on tenant tables healed by init_db, whose ALTER adds `created_at`
        with no default and therefore NULL on legacy rows.
        """
        reference = (
            memory.last_reinforced_at
            or memory.last_accessed_at
            or memory.created_at
        )
        last_decayed = memory.last_decayed_at
        if last_decayed is not None and (reference is None or last_decayed > reference):
            return last_decayed
        return reference

    def _calculate_decayed_strength(
        self,
        memory: Memory,
        current_time: datetime
    ) -> float:
        """
        Calculate the new strength of a memory based on time elapsed.

        Uses modified Ebbinghaus formula with individual factors.
        """
        # Determine reference time: the point the stored strength is accurate
        # as of — last decay pass, else last reinforcement/access, else creation.
        reference_time = self._decay_reference_time(memory)
        if reference_time is None:
            # Nothing to measure from; leave the memory alone rather than
            # crashing the whole user's batch on a legacy NULL.
            return memory.strength

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
        access_context: str = "recall",
        similarity_score: float = 0.5,
        commit: bool = True,
    ) -> Optional[Memory]:
        """
        Reinforce a memory (strengthen it) when accessed or recalled.

        Implements adaptive spaced-repetition:
        - Longer gap since last recall → bigger boost (log-based)
        - Higher similarity match → stronger reinforcement
        - Diminishing returns for very frequent access

        `commit` (default True, i.e. every pre-existing caller is unchanged):
        this method commits the session it was handed, which is fine for the
        API routes that own their request session and call it once, and wrong
        for anything that is a guest in someone else's transaction. Pass
        commit=False there and let the owner decide when to commit — see
        RetrievalFeedback.log_retrieval_feedback, whose docstring contract is
        explicitly "don't commit here, let the caller handle the transaction".
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

        now = datetime.utcnow()

        # An EXPLICIT user "Keep" cancels any expiry lease outright. This runs
        # BEFORE the cooldown return below on purpose: tapping Keep must always
        # save the memory, even if something reinforced it in the last hour.
        #
        # Without this the Memory screen lies. The user sees a card, taps Keep,
        # gets a success response — and the TTL sweep archives it anyway,
        # because reinforcement moved strength and last_reinforced_at but never
        # touched expires_at. There is no restore route, so that is effectively
        # irreversible from the user's side.
        _lease_cleared = False
        if access_context == "user_reinforce" and memory.expires_at is not None:
            memory.expires_at = None
            _lease_cleared = True
            logger.info(
                "[decay] user Keep cleared expiry lease on %s: %s",
                memory.id, (memory.content or "")[:60],
            )

        # Cooldown: skip reinforcement if last reinforced within 1 hour
        if memory.last_reinforced_at:
            hours_since = (now - memory.last_reinforced_at).total_seconds() / 3600
            if hours_since < 1.0:
                # The lease clear above is a USER DECISION, not part of the
                # strength update, so it must survive the cooldown. This
                # returned before the commit at the end of the function, so a
                # Keep within an hour of the row's last reinforcement was
                # discarded when the session closed — and Keep is exactly the
                # button a user presses right after the agent has just recalled
                # the memory, which is what set last_reinforced_at in the first
                # place. The success response was truthful about the intent and
                # wrong about the outcome; the TTL sweep archived it anyway.
                if commit and _lease_cleared:
                    await self.db.commit()
                return memory  # Already reinforced recently, skip

        old_strength = memory.strength

        # Spaced repetition: longer gap = bigger boost (log-based)
        last = memory.last_reinforced_at or memory.created_at
        days_since = max((now - last).total_seconds() / 86400, 0.01)

        # log(days+1)/log(31) gives: <1hr→~0.0, 1day→~0.2, 7days→~0.57, 30days→~1.0
        time_factor = min(math.log(days_since + 1) / math.log(31), 1.0)
        base_boost = 0.05 + (0.45 * time_factor)  # Range: 0.05 to 0.50

        # Similarity factor: higher match = stronger reinforcement
        sim_factor = 0.5 + (0.5 * min(similarity_score, 1.0))  # Range: 0.5 to 1.0

        boost = base_boost * sim_factor

        # Apply boost
        new_strength = min(old_strength + boost, self.MAX_STRENGTH)

        # Update memory
        memory.strength = new_strength
        memory.last_reinforced_at = now
        memory.last_accessed_at = now
        memory.access_count += 1

        # Adjust decay rate based on access pattern (frequent access = slower decay)
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
                "time_factor": round(time_factor, 3),
                "similarity_score": round(similarity_score, 3),
                "access_context": access_context,
                "access_count": memory.access_count,
            },
            trigger_source="api"
        )

        if commit:
            await self.db.commit()
        else:
            # Make the UPDATE + the MemoryEvent INSERT visible inside the
            # caller's transaction without ending it.
            await self.db.flush()
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
