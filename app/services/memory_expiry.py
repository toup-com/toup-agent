"""Per-turn expiry sweep for time-bounded memories.

Why this is not part of the memory-maintenance scheduler
--------------------------------------------------------
Decay and consolidation are registered on the tenant scheduler behind
`agent_memory_maintenance_enabled`, which defaults False and was found unset on
all 54 production containers at the 2026-07-29 audit — those jobs have never
executed against tenant data. Expiry is the one piece of hygiene that must not
depend on a flag being flipped, because it is the only thing standing between
"remind me to eat tea in 2 minutes" and permanent residency in the user's
brain.

So it runs on the per-turn background path (`AgentRunner`'s
`_background_post_processing`), alongside the pre-existing active-task TTL,
which production logs confirm does execute every turn.

Archival, never deletion
------------------------
Expiry sets `is_active = False`. The row stays in the table, stays queryable,
and keeps its audit trail — `list_memories` and `hybrid_search` both filter on
`is_active`, so the memory leaves the UI and the prompt without leaving the
database. Every archival writes a MemoryEvent so the sweep is reversible from
the audit log alone.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Memory, MemoryEvent, MemoryEventType

from app.services.memory_log import describe_memory

logger = logging.getLogger(__name__)

# Safety valve: never archive more than this many rows in a single turn. A
# clock skew or a bad backfill that stamped expires_at across the table should
# degrade into "archives 200/turn and logs loudly", not "wipes the brain in one
# transaction".
MAX_ARCHIVED_PER_SWEEP = 200


async def expire_stale_memories(
    db: AsyncSession,
    user_id: str,
    *,
    now: Optional[datetime] = None,
    dry_run: bool = False,
) -> List[Memory]:
    """Archive active memories whose `expires_at` has passed.

    Scoped to a single `user_id` on every path — this runs inside a tenant
    container whose DB holds one user, but the predicate is explicit so the
    same code is safe if it is ever run against a shared DB.

    Returns the affected memories (the ones it archived, or the ones it *would*
    archive when `dry_run`).
    """
    if not user_id:
        return []

    now = now or datetime.utcnow()

    result = await db.execute(
        select(Memory)
        .where(
            and_(
                Memory.user_id == user_id,
                Memory.expires_at.is_not(None),
                Memory.expires_at <= now,
                Memory.is_active.is_(True),
                Memory.is_deleted.is_(False),
            )
        )
        .order_by(Memory.expires_at)
        .limit(MAX_ARCHIVED_PER_SWEEP + 1)
    )
    stale = list(result.scalars().all())

    if len(stale) > MAX_ARCHIVED_PER_SWEEP:
        logger.warning(
            "[memory_expiry] user=%s has more than %d expired memories; "
            "archiving the oldest %d this turn and deferring the rest",
            user_id, MAX_ARCHIVED_PER_SWEEP, MAX_ARCHIVED_PER_SWEEP,
        )
        stale = stale[:MAX_ARCHIVED_PER_SWEEP]

    if dry_run or not stale:
        return stale

    for memory in stale:
        memory.is_active = False
        db.add(
            MemoryEvent(
                memory_id=memory.id,
                user_id=user_id,
                event_type=MemoryEventType.DECAYED.value,
                event_data_json=(
                    '{"reason": "ttl_expired", "expires_at": "%s"}'
                    % (memory.expires_at.isoformat() if memory.expires_at else "")
                ),
                trigger_source="memory_expiry",
            )
        )
        logger.info(
            "[memory_expiry] archived (expired %s): %s",
            memory.expires_at, describe_memory(memory.content),
        )

    await db.flush()
    return stale
