"""
Backfill NULL embeddings on memories using the local all-MiniLM-L6-v2 model.

After a vector dimension migration (1536 → 384), the embedding column is dropped
and recreated, leaving all existing memories with NULL embeddings.  This script
re-embeds them so they become visible to vector search again.

Run: python -m app.scripts.backfill_embeddings
"""

import asyncio
import json
import logging
import time

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session_maker
from app.db.models import Memory
from app.services.embedding_service import get_embedding_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50


async def backfill():
    """Re-embed all memories where the pgvector embedding column is NULL."""
    embed_svc = get_embedding_service()

    async with async_session_maker() as db:
        # Count how many need fixing
        count_result = await db.execute(
            select(func.count(Memory.id)).where(
                and_(
                    Memory.is_deleted == False,
                    Memory.embedding == None,  # noqa: E711 — SQLAlchemy IS NULL
                )
            )
        )
        total = count_result.scalar() or 0
        logger.info(f"Found {total} memories with NULL embedding")

        if total == 0:
            logger.info("Nothing to backfill — all memories have embeddings.")
            return

        fixed = 0
        errors = 0
        t0 = time.time()

        # Process in batches
        while True:
            result = await db.execute(
                select(Memory).where(
                    and_(
                        Memory.is_deleted == False,
                        Memory.embedding == None,  # noqa: E711
                    )
                ).limit(BATCH_SIZE)
            )
            batch = list(result.scalars().all())
            if not batch:
                break

            for mem in batch:
                try:
                    text = mem.canonical_content or mem.content
                    if not text or len(text.strip()) < 3:
                        errors += 1
                        continue

                    embedding = embed_svc.embed(text)
                    mem.embedding = embedding
                    mem.embedding_json = json.dumps(embedding)
                    fixed += 1
                except Exception as e:
                    logger.warning(f"  Failed to embed memory {mem.id}: {e}")
                    errors += 1

            await db.commit()
            elapsed = time.time() - t0
            rate = fixed / elapsed if elapsed > 0 else 0
            logger.info(f"  Progress: {fixed}/{total} fixed ({rate:.1f}/s), {errors} errors")

        elapsed = time.time() - t0
        logger.info(
            f"\nBackfill complete: {fixed} re-embedded, {errors} errors, "
            f"{elapsed:.1f}s elapsed"
        )


if __name__ == "__main__":
    asyncio.run(backfill())
