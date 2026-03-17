"""
Backfill missing EntityLink records for memories.

Before this fix, _upsert_entity() created Entity records but never created
EntityLink records connecting those entities to the memories that mentioned them.
This left graph-aware retrieval with nothing to walk.

Strategy per user:
1. Load all entities for that user (name → id mapping)
2. Find all memories that have zero EntityLinks
3. For each memory, check which entity names appear in the content
4. Create an EntityLink for each match

Run: python -m app.scripts.backfill_entity_links
"""

import asyncio
import logging
import re
import time
import uuid

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.db.models import Memory, Entity, EntityLink
from app.db.models.user import User

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Words that happen to be capitalised but are never entity names
STOP_WORDS = {
    "i", "my", "me", "the", "a", "an", "this", "that", "it", "is", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "about", "like", "through", "after", "before", "between",
    "under", "above", "up", "down", "out", "off", "over", "then", "than",
    "so", "if", "or", "and", "but", "not", "no", "yes", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "very", "just", "also", "now", "here", "there", "when",
    "where", "how", "what", "which", "who", "whom", "why",
}

BATCH_SIZE = 100


def _entity_appears_in(entity_name: str, text: str) -> bool:
    """Check if an entity name appears in text as a whole word (case-insensitive)."""
    # Require word boundaries so "Ali" doesn't match "normalize"
    pattern = r"\b" + re.escape(entity_name) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


async def backfill():
    """Create missing EntityLinks by matching entity names against memory content."""
    async with async_session_maker() as db:
        # Get all users that have at least one entity
        user_result = await db.execute(
            select(Entity.user_id).group_by(Entity.user_id)
        )
        user_ids = [row[0] for row in user_result.fetchall()]
        logger.info(f"Found {len(user_ids)} users with entities")

        total_created = 0
        total_memories = 0

        for user_id in user_ids:
            # Load all entities for this user
            ent_result = await db.execute(
                select(Entity).where(Entity.user_id == user_id)
            )
            entities = list(ent_result.scalars().all())
            if not entities:
                continue

            # Filter out very short / stop-word entity names
            valid_entities = [
                e for e in entities
                if len(e.name.strip()) >= 2
                and e.name.strip().lower() not in STOP_WORDS
            ]
            if not valid_entities:
                continue

            # Find memories with zero EntityLinks
            # Use a LEFT JOIN / NOT EXISTS approach
            subq = (
                select(EntityLink.memory_id)
                .where(EntityLink.memory_id == Memory.id)
                .correlate(Memory)
                .exists()
            )
            mem_result = await db.execute(
                select(Memory).where(
                    and_(
                        Memory.user_id == user_id,
                        Memory.is_deleted == False,
                        ~subq,
                    )
                ).order_by(Memory.created_at.desc())
            )
            orphan_memories = list(mem_result.scalars().all())

            if not orphan_memories:
                continue

            logger.info(
                f"User {user_id[:8]}…: {len(orphan_memories)} memories without links, "
                f"{len(valid_entities)} entities to match"
            )

            user_created = 0
            for mem in orphan_memories:
                text = mem.canonical_content or mem.content or ""
                if not text:
                    continue

                for entity in valid_entities:
                    if _entity_appears_in(entity.name, text):
                        db.add(EntityLink(
                            id=str(uuid.uuid4()),
                            memory_id=mem.id,
                            entity_id=entity.id,
                            role="mentioned",
                        ))
                        user_created += 1

            if user_created:
                await db.commit()
                logger.info(f"  → Created {user_created} EntityLinks")

            total_created += user_created
            total_memories += len(orphan_memories)

        logger.info(
            f"\nBackfill complete: {total_created} EntityLinks created "
            f"across {total_memories} orphan memories for {len(user_ids)} users"
        )


if __name__ == "__main__":
    asyncio.run(backfill())
