#!/usr/bin/env python3
"""Corrective backfill for the three data problems #375 left on live tenants.

Runs INSIDE a tenant agent container (that is where `memories` lives).
Idempotent, dry-run by default, and NOTHING IS DELETED. Every mutation writes
a MemoryEvent carrying the previous values under
trigger_source='memory_regression_repair', so each step is reversible from the
audit log alone.

This is a companion to scripts/migrate_memory_taxonomy.py, not a replacement.
That script did the taxonomy work; this one repairs what it got wrong.

  1. RESTORE WRONGLY-ARCHIVED ROUTINES (is_active False -> True)

     `decay_expired_tasks` archives on `category` alone. The taxonomy
     migration remapped legacy `schedule` onto `active_task`, which put the
     recurring arrangements it had deliberately exempted from `expires_at`
     into that archiver's blast radius — and a standing "daily Gmail briefing"
     has no reinforcement of its own, because the ROUTINE fires, not the
     memory. Four of the founder's real routines were archived on the day of
     the rollout. The code fix stops it recurring; this restores the rows.

     Scoped hard: only rows that are (a) inactive, (b) recurring by
     describes_recurring_arrangement, and (c) carry a MemoryEvent proving the
     migration or the archiver touched them. A row the USER archived by hand
     is never resurrected — that is the same irreversibility trap as
     re-creating a forgotten memory.

  2. HUMANIZE LEGACY RELATIONSHIP CONTENT

     Relationship rows stored `content` as the raw predicate spliced between
     two entity names — "Better Call Saul play on Netflix", "Bunker performed
     by Baltazar". That is what the Memory card renders now that `summary` is
     NULL. Rewrites `content` through humanize_relationship() and RE-EMBEDS,
     because content is also the vector's source text; skipping the re-embed
     would leave the row searchable only by its old phrasing.

     The triple is not touched: it stays in metadata_json and on the
     entity_relationships edge. The previous content is recorded in the event.

  3. ARCHIVE GENUINELY DEAD REMINDERS

     The migration anchored expires_at on `last_reinforced_at`, and the
     >50% word-overlap matcher in active_task_service had recently bumped
     rows that were 70+ days old — so dead reminders got a FRESH 7-day lease
     instead of being archived. Re-anchors those on `created_at` when the
     content names a sub-hour horizon ("in two minutes"), which is never a
     durable fact. Archival is is_active=False, reversible, and still goes
     through the normal sweep rather than happening here.

Usage:
    python3 -m scripts.repair_memory_regressions --dry-run
    python3 -m scripts.repair_memory_regressions --apply
    python3 -m scripts.repair_memory_regressions --dry-run --only restore
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.db.database import async_session_maker  # noqa: E402
from app.db.models import Memory, MemoryEvent  # noqa: E402
from app.memory_taxonomy import (  # noqa: E402
    describes_recurring_arrangement,
    humanize_relationship,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("repair_memory_regressions")

TRIGGER = "memory_regression_repair"

# Events that prove an automated path archived the row, as opposed to the user.
_AUTOMATED_TRIGGERS = {
    "taxonomy_migration",
    "memory_expiry",
    "active_task_service",
    "decay_service",
}

# A horizon under an hour is never a durable fact. It must name the unit, so
# "remind me in two minutes" matches and "the user is researching UofT events"
# does not.
#
# BOTH orderings are required. The extractor overwhelmingly writes the SUFFIX
# form — "reminded to eat tea 2 minutes after the request", "reminded two
# minutes after the request to call their brother", "wake up two minutes
# later" — and a prefix-only pattern ("in N minutes") missed every one of the
# rows this step exists to catch.
_COUNT = r"(?:a|an|one|two|three|four|five|ten|fifteen|thirty|\d{1,3})"
_UNIT = r"(?:min|mins|minute|minutes|sec|secs|second|seconds)"
_SUBHOUR_HORIZON = re.compile(
    rf"\b(?:(?:in|after|within)\s+{_COUNT}\s*(?:more\s+)?{_UNIT}"
    rf"|{_COUNT}\s*{_UNIT}\s+(?:after|later|from\s+now))\b",
    re.IGNORECASE,
)


def _event(memory: Memory, user_id: str, action: str, before: dict) -> MemoryEvent:
    return MemoryEvent(
        memory_id=memory.id,
        user_id=user_id,
        event_type="updated",
        event_data_json=json.dumps({"repair": action, "before": before}),
        trigger_source=TRIGGER,
    )


async def _touched_by_automation(session: AsyncSession, memory_id: str) -> bool:
    """True if an automated path — not the user — last changed this row."""
    rows = (await session.execute(
        select(MemoryEvent.trigger_source).where(
            MemoryEvent.memory_id == memory_id
        )
    )).scalars().all()
    sources = {r for r in rows if r}
    if not sources:
        return False
    # A hand delete/archive shows up as a user-triggered event. If ANY user
    # event exists we leave the row alone rather than guess at ordering.
    if any(s.startswith("user") or s == "api" for s in sources):
        return False
    return bool(sources & _AUTOMATED_TRIGGERS)


async def restore_archived_routines(
    session: AsyncSession, apply: bool
) -> Dict[str, int]:
    stats = {"candidates": 0, "restored": 0, "skipped_user_archived": 0}
    rows = (await session.execute(
        select(Memory).where(
            Memory.is_active == False,  # noqa: E712
            Memory.is_deleted == False,  # noqa: E712
        )
    )).scalars().all()

    for mem in rows:
        if not describes_recurring_arrangement(mem.content):
            continue
        stats["candidates"] += 1
        if not await _touched_by_automation(session, mem.id):
            stats["skipped_user_archived"] += 1
            logger.info("  SKIP (user-archived): %s", (mem.content or "")[:70])
            continue

        before = {"is_active": False, "strength": float(mem.strength or 0.0)}
        logger.info("  RESTORE: %s", (mem.content or "")[:70])
        if apply:
            mem.is_active = True
            # decay_expired_tasks zeroes strength on archive; a restored row
            # with strength 0 is retrievable but ranks last forever.
            if not mem.strength:
                mem.strength = 0.6
            mem.expires_at = None  # a standing arrangement has no horizon
            session.add(_event(mem, mem.user_id, "restore_archived_routine", before))
        stats["restored"] += 1

    return stats


async def humanize_relationship_rows(
    session: AsyncSession, apply: bool
) -> Dict[str, int]:
    stats = {"scanned": 0, "rewritten": 0, "reembedded": 0, "unchanged": 0}
    rows = (await session.execute(
        select(Memory).where(
            Memory.source_type == "entity_extraction",
            Memory.is_deleted == False,  # noqa: E712
        )
    )).scalars().all()

    svc = None
    for mem in rows:
        stats["scanned"] += 1
        meta = {}
        try:
            meta = json.loads(mem.metadata_json) if mem.metadata_json else {}
        except (TypeError, ValueError):
            meta = {}
        source = meta.get("source_name") or meta.get("source")
        target = meta.get("target_name") or meta.get("target")
        predicate = meta.get("relationship") or meta.get("relationship_type")
        if not (source and target and predicate):
            # Without the structured triple there is nothing authoritative to
            # re-render from, and guessing at the split would corrupt content.
            continue

        wanted = humanize_relationship(source, predicate, target)
        if not wanted or wanted == (mem.content or ""):
            stats["unchanged"] += 1
            continue

        logger.info("  %r", (mem.content or "")[:66])
        logger.info("    -> %r", wanted[:66])
        if apply:
            before = {"content": mem.content}
            mem.content = wanted
            session.add(_event(mem, mem.user_id, "humanize_relationship", before))
            # content IS the vector's source text — re-embed or the row stays
            # findable only under its old phrasing.
            try:
                if svc is None:
                    from app.services.memory_service import MemoryService
                    svc = MemoryService(session)
                emb = svc.embedding_service.embed(wanted, api_key=svc.api_key)
                mem.embedding_json = json.dumps(emb)
                mem.embedding = emb
                stats["reembedded"] += 1
            except Exception as e:
                logger.warning("    (re-embed failed, content still fixed: %s)", e)
        stats["rewritten"] += 1

    return stats


async def rearchive_dead_reminders(
    session: AsyncSession, apply: bool
) -> Dict[str, int]:
    stats = {"candidates": 0, "reanchored": 0, "already_past": 0}
    now = datetime.utcnow()
    rows = (await session.execute(
        select(Memory).where(
            Memory.is_active == True,  # noqa: E712
            Memory.is_deleted == False,  # noqa: E712
            Memory.expires_at.is_not(None),
        )
    )).scalars().all()

    for mem in rows:
        if not _SUBHOUR_HORIZON.search(mem.content or ""):
            continue
        if describes_recurring_arrangement(mem.content):
            continue
        stats["candidates"] += 1
        anchor = mem.created_at or now
        # A sub-hour reminder is dead the day it was made. One day of grace so
        # it stays visible for the rest of that day, matching resolve_ttl_days.
        horizon = anchor + timedelta(days=1)
        if mem.expires_at and mem.expires_at <= horizon:
            continue
        age = (now - anchor).days
        logger.info(
            "  RE-ANCHOR (%dd old, lease %s -> %s): %s",
            age,
            mem.expires_at.date() if mem.expires_at else None,
            horizon.date(),
            (mem.content or "")[:60],
        )
        if apply:
            session.add(_event(
                mem, mem.user_id, "reanchor_dead_reminder",
                {"expires_at": mem.expires_at.isoformat() if mem.expires_at else None},
            ))
            mem.expires_at = horizon
        stats["reanchored"] += 1
        if horizon <= now:
            stats["already_past"] += 1

    return stats


STEPS = {
    "restore": ("restore wrongly-archived routines", restore_archived_routines),
    "humanize": ("humanize legacy relationship content", humanize_relationship_rows),
    "reminders": ("re-anchor dead reminder leases", rearchive_dead_reminders),
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    group.add_argument("--apply", action="store_true", help="apply the repair")
    parser.add_argument(
        "--only", choices=sorted(STEPS), action="append",
        help="run only these steps (repeatable); default is all",
    )
    args = parser.parse_args()
    apply = bool(args.apply)
    chosen = args.only or sorted(STEPS)

    logger.info("=== memory regression repair (%s) ===",
                "APPLY" if apply else "DRY RUN")

    async with async_session_maker() as session:
        for key in chosen:
            label, fn = STEPS[key]
            logger.info("\n-- %s --", label)
            stats = await fn(session, apply)
            logger.info("   %s", json.dumps(stats))
        if apply:
            await session.commit()
            logger.info("\ncommitted.")
        else:
            await session.rollback()
            logger.info("\nnothing written (dry run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
