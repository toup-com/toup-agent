"""
Backfill entity_relationships from existing Memory records.

Finds Memory records with source_type='entity_extraction' and metadata containing
relationship_type, then creates corresponding EntityRelationship records.

Run: python -m app.scripts.backfill_entity_relationships
"""

import asyncio
import json
import logging
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.db.models import Memory, Entity, EntityRelationship

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def backfill():
    """Find relationship memories and create EntityRelationship records."""
    async with async_session_maker() as db:
        # Find all memories that encode entity relationships
        result = await db.execute(
            select(Memory).where(
                and_(
                    Memory.source_type == "entity_extraction",
                    Memory.is_deleted == False,
                    Memory.metadata_json.isnot(None),
                )
            )
        )
        relationship_memories = list(result.scalars().all())
        logger.info(f"Found {len(relationship_memories)} relationship memories to process")

        created = 0
        skipped = 0
        errors = 0

        for mem in relationship_memories:
            try:
                meta = json.loads(mem.metadata_json) if mem.metadata_json else {}
                rel_type = meta.get("relationship_type")
                source_name = meta.get("source_entity")
                target_name = meta.get("target_entity")

                if not (rel_type and source_name and target_name):
                    skipped += 1
                    continue

                # Find source entity
                result = await db.execute(
                    select(Entity).where(
                        and_(
                            Entity.user_id == mem.user_id,
                            Entity.name.ilike(source_name),
                        )
                    )
                )
                source_entity = result.scalar_one_or_none()

                # Find target entity
                result = await db.execute(
                    select(Entity).where(
                        and_(
                            Entity.user_id == mem.user_id,
                            Entity.name.ilike(target_name),
                        )
                    )
                )
                target_entity = result.scalar_one_or_none()

                if not source_entity or not target_entity:
                    logger.debug(
                        f"Skipping: entity not found — source='{source_name}' target='{target_name}'"
                    )
                    skipped += 1
                    continue

                # Check if relationship already exists
                result = await db.execute(
                    select(EntityRelationship).where(
                        and_(
                            EntityRelationship.source_entity_id == source_entity.id,
                            EntityRelationship.target_entity_id == target_entity.id,
                            EntityRelationship.relationship_type == rel_type,
                        )
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Just bump mention count
                    existing.mention_count += 1
                    existing.last_seen_at = datetime.utcnow()
                    skipped += 1
                else:
                    import uuid

                    new_rel = EntityRelationship(
                        id=str(uuid.uuid4()),
                        user_id=mem.user_id,
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relationship_type=rel_type,
                        relationship_label=f"{source_name} {rel_type.replace('_', ' ')} {target_name}",
                        confidence=mem.confidence or 0.7,
                        mention_count=1,
                        first_seen_at=mem.created_at or datetime.utcnow(),
                        last_seen_at=mem.created_at or datetime.utcnow(),
                    )
                    db.add(new_rel)
                    created += 1
                    logger.info(
                        f"  Created: {source_name} --[{rel_type}]--> {target_name}"
                    )

            except Exception as e:
                logger.warning(f"  Error processing memory {mem.id}: {e}")
                errors += 1

        await db.commit()
        logger.info(
            f"\nBackfill complete: {created} created, {skipped} skipped, {errors} errors"
        )


if __name__ == "__main__":
    asyncio.run(backfill())
