#!/usr/bin/env python3
"""One-shot migration for the 2026-07-29 memory taxonomy + hygiene work.

Runs INSIDE a tenant agent container (that is where the `memories` table
lives). Idempotent: safe to run repeatedly, and safe to run on a tenant that
has already been migrated.

What it does, all of it reversible:

  1. CATEGORY REMAP — rewrites categories the app has no label for
     (`schedule`, `projects`, `learning`, `tools`, `family`, `places`, `food`,
     `travel`, `context`) onto canonical values. 28% of rows on the founder's
     tenant were affected and rendered as "Other".

  2. ARROW SUMMARY CLEANUP — nulls `summary` on rows whose summary is a
     machine triple ("Bunker → performed_by → Baltazar"). The readable
     sentence is already in `content`; the structured triple is already in
     `metadata_json`. Nothing is lost. 61% of the founder's rows.

  3. CONSOLIDATION_COUNT RESET — zeroes `consolidation_count` on rows that
     were never actually consolidated (memory_level='episodic'). The
     reinforcement path used to increment this field on every restatement,
     and DecayService reads it as a decay-resistance multiplier worth up to
     2x. Leaving the inflated values in place would neutralise decay the
     moment it is enabled. Prior values are recorded in the audit event.

  4. BACKFILL expires_at — stamps a horizon on legacy reminder rows, i.e.
     those whose EFFECTIVE category is `active_task` (the old `schedule`).
     Deliberately narrow: a `memory_type='task'` rule was tried first and
     rejected because live data showed it also catching durable `goals` and
     `projects` rows. Uses the row's own age, so rows already past their
     horizon are archived by the next per-turn sweep rather than deleted
     here — `--dry-run` lists exactly which ones those are before you apply.

  5. SEARCH_VECTOR BACKFILL — populates the tsvector column for existing rows.
     The maintenance trigger shipped only in Alembic, and agent containers
     boot via create_all, so 100% of tenant rows had it NULL and the keyword
     leg of hybrid_search matched nothing.

NOTHING IS DELETED. No row is removed, no column is dropped, no memory is
hard-deleted. Step 4 can mark rows for later archival (is_active=False by the
sweep), which is itself reversible.

Every mutation writes a MemoryEvent with trigger_source='taxonomy_migration'
carrying the previous values, so the whole migration can be reversed from the
audit log alone even without the backup.

Usage:
    python3 -m scripts.migrate_memory_taxonomy --dry-run     # report only
    python3 -m scripts.migrate_memory_taxonomy --apply
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.db.database import async_session_maker  # noqa: E402
from app.db.models import Memory, MemoryEvent  # noqa: E402
from app.memory_taxonomy import (  # noqa: E402
    MEMORY_CATEGORY_ALIASES,
    MemoryCategory,
    describes_recurring_arrangement,
    normalize_category,
    resolve_ttl_days,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("migrate_memory_taxonomy")

TRIGGER = "taxonomy_migration"

# A triple summary is "A → B → C". Matching the arrow glyph is enough: the
# backend is the only writer of that character into `summary`.
ARROW = "→"


def _audit(memory: Memory, user_id: str, action: str, before: Dict) -> MemoryEvent:
    return MemoryEvent(
        memory_id=memory.id,
        user_id=user_id,
        event_type="updated",
        event_data_json=json.dumps({"migration": action, "before": before}),
        trigger_source=TRIGGER,
    )


async def migrate(session: AsyncSession, apply: bool) -> Dict[str, int]:
    stats = {
        "scanned": 0,
        "category_remapped": 0,
        "summary_cleared": 0,
        "consolidation_reset": 0,
        "expiry_backfilled": 0,
        "will_archive_immediately": 0,
        "recurring_exempted": 0,
    }
    immediate_archives: List[str] = []

    rows = (await session.execute(select(Memory))).scalars().all()
    stats["scanned"] = len(rows)

    now = datetime.utcnow()

    for memory in rows:
        user_id = memory.user_id
        before: Dict = {}
        changed = False

        # ── 1. category remap ────────────────────────────────────────
        raw_category = (memory.category or "").strip().lower()
        if raw_category in MEMORY_CATEGORY_ALIASES:
            canonical = normalize_category(
                raw_category, brain_type=memory.brain_type or "user"
            )
            if canonical != memory.category:
                before["category"] = memory.category
                if apply:
                    memory.category = canonical
                stats["category_remapped"] += 1
                changed = True

        # ── 2. arrow summaries ───────────────────────────────────────
        if memory.summary and ARROW in memory.summary:
            # Only safe to drop the summary when `content` actually holds the
            # readable sentence. Every relationship row does, but check rather
            # than assume — a row with no content would render blank.
            if (memory.content or "").strip():
                before["summary"] = memory.summary
                if apply:
                    memory.summary = None
                stats["summary_cleared"] += 1
                changed = True
            else:
                logger.warning(
                    "  skipped summary cleanup on %s — content is empty", memory.id
                )

        # ── 3. consolidation_count reset ─────────────────────────────
        # Only for rows that never went through real consolidation. A
        # consolidated row has memory_level='semantic'.
        if (memory.consolidation_count or 0) > 0 and memory.memory_level == "episodic":
            before["consolidation_count"] = memory.consolidation_count
            if apply:
                memory.consolidation_count = 0
            stats["consolidation_reset"] += 1
            changed = True

        # ── 4. expiry backfill for legacy transient rows ─────────────
        if memory.expires_at is None and memory.is_active:
            effective_category = (
                normalize_category(memory.category, brain_type=memory.brain_type or "user")
                if memory.category
                else MemoryCategory.OTHER.value
            )
            # Deliberately NARROW: effective category only, not
            # `memory_type == "task"`. Checked against live tenant data —
            # a memory_type-based rule would also have caught a `goals` row
            # ("the user is researching UofT future-student events", 31d) and
            # a `projects` row (66d), both of which are durable context the
            # user would not expect to lose. Every genuinely reminder-shaped
            # row already carries the legacy `schedule` category, which maps
            # to active_task above.
            is_transient = effective_category == MemoryCategory.ACTIVE_TASK.value
            # A standing arrangement ("send me a Gmail briefing every day at
            # 11:49") is phrased like a schedule but is a durable preference —
            # archiving it would stop the agent knowing about a routine the
            # user still relies on. 9 of the founder's 28 legacy schedule rows
            # are of this shape.
            if is_transient and describes_recurring_arrangement(memory.content):
                is_transient = False
                stats["recurring_exempted"] += 1
            if is_transient and (memory.brain_type or "user") == "user":
                ttl_days = resolve_ttl_days(effective_category)
                if ttl_days is not None:
                    anchor = memory.last_reinforced_at or memory.created_at or now
                    horizon = anchor + timedelta(days=ttl_days)
                    before["expires_at"] = None
                    if apply:
                        memory.expires_at = horizon
                    stats["expiry_backfilled"] += 1
                    if horizon <= now:
                        # Already past its horizon: the next per-turn sweep
                        # will archive it. Reported separately so an operator
                        # sees the real UI impact BEFORE applying, rather than
                        # discovering it when rows vanish from the phone.
                        stats["will_archive_immediately"] += 1
                        immediate_archives.append(
                            (memory.content or "")[:70]
                        )
                    changed = True

        if changed and apply:
            session.add(_audit(memory, user_id, "taxonomy_migration", before))

    # ── 5. search_vector backfill ────────────────────────────────────
    # Postgres-only; the trigger keeps it current from here on.
    if apply:
        try:
            await session.execute(
                text(
                    "UPDATE memories SET search_vector = "
                    "setweight(to_tsvector('english', coalesce(content, '')), 'A') || "
                    "setweight(to_tsvector('english', coalesce(summary, '')), 'B') "
                    "WHERE search_vector IS NULL"
                )
            )
        except Exception as e:
            logger.warning("  search_vector backfill skipped: %s", e)

    if apply:
        await session.commit()

    if immediate_archives:
        logger.info("")
        logger.info(
            "  %d row(s) are ALREADY past their horizon and will be archived "
            "(is_active=False, reversible) by the next per-turn sweep:",
            len(immediate_archives),
        )
        for content in immediate_archives:
            logger.info("      - %s", content)

    return stats


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    group.add_argument("--apply", action="store_true", help="apply the migration")
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN"
    logger.info("=== memory taxonomy migration (%s) ===", mode)

    async with async_session_maker() as session:
        stats = await migrate(session, apply=args.apply)

    logger.info("")
    logger.info("  rows scanned:          %d", stats["scanned"])
    logger.info("  categories remapped:   %d", stats["category_remapped"])
    logger.info("  arrow summaries dropped: %d", stats["summary_cleared"])
    logger.info("  consolidation_count reset: %d", stats["consolidation_reset"])
    logger.info("  expiry backfilled:     %d", stats["expiry_backfilled"])
    logger.info("  ...of which already expired: %d", stats["will_archive_immediately"])
    logger.info("  recurring arrangements exempted: %d", stats["recurring_exempted"])
    logger.info("")
    if not args.apply:
        logger.info("  DRY RUN — nothing was written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
