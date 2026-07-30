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
     memory. The code fix stops it recurring; this restores the rows.

     Measured on the founder's tenant: 11 archived recurring rows, but 9 are
     dedup SUPERSESSIONS whose content is still live under `superseded_by`.
     Only 2 are genuine losses. Worth stating plainly, because the raw
     "archived recurring rows" count overstates the harm by 5x.

     Scoped hard: only rows that are (a) `is_active=False` AND
     `is_deleted=False`, (b) recurring by describes_recurring_arrangement, and
     (c) not a dedup loser (`superseded_by IS NULL`). A memory the USER forgot
     sets `is_deleted=True` — a different column — so it is excluded by (a)
     and can never be resurrected here. See restore_archived_routines for why
     the event log cannot be used for this.

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
    _PREDICATE_ALIASES,
    _PREDICATE_TEMPLATES,
    category_for_relationship,
    describes_recurring_arrangement,
    humanize_relationship,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("repair_memory_regressions")

TRIGGER = "memory_regression_repair"

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


async def restore_archived_routines(
    session: AsyncSession, apply: bool
) -> Dict[str, int]:
    """Re-activate recurring arrangements that an automated path archived.

    Distinguishing "the user got rid of this" from "our archiver ate it" does
    NOT come from the event log. Verified against a live tenant: the only
    trigger_source values that exist are `api` (created/reinforced),
    `conversation`, `taxonomy_migration`, `memory_expiry` and
    `conversation_backfill`. There is no user-archive event at all — and
    `decay_expired_tasks` writes no event either, so the archival that caused
    this is invisible in the log. An earlier version of this function keyed off
    trigger_source and rejected all 11 candidates as "user-archived", because
    `api/reinforced` (906 rows of it) is automated reinforcement during
    retrieval, not a user action.

    The real signal is the COLUMN, not the log:
      - a user forgetting a memory sets `is_deleted = True`
      - archival sets `is_active = False`
    so `is_active=False AND is_deleted=False` is already exclusively automated.

    The one case that still must be excluded is dedup supersession, which also
    deactivates the loser — restoring it would resurrect a duplicate of a row
    that is still live under `superseded_by`.
    """
    stats = {"candidates": 0, "restored": 0, "skipped_superseded": 0}
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
        if mem.superseded_by:
            stats["skipped_superseded"] += 1
            logger.info(
                "  SKIP (superseded by %s): %s",
                mem.superseded_by, (mem.content or "")[:60],
            )
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
    stats = {
        "scanned": 0, "rewritten": 0, "reembedded": 0, "unchanged": 0,
        "no_triple": 0, "richer_content_kept": 0,
        "uncurated_predicate_skipped": 0,
    }
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
        # Real key names, read off live rows: source_entity / target_entity /
        # relationship_type (plus extracted_by). Guessing source_name/
        # relationship matched nothing and made this step a silent no-op.
        source = meta.get("source_entity") or meta.get("source_name")
        target = meta.get("target_entity") or meta.get("target_name")
        predicate = meta.get("relationship_type") or meta.get("relationship")
        if not (source and target and predicate):
            # Without the structured triple there is nothing authoritative to
            # re-render from, and guessing at the split would corrupt content.
            stats["no_triple"] += 1
            continue

        # ONLY rewrite rows whose content IS the old machine format. Many
        # entity_extraction rows carry richer prose written by another path —
        # "When handling Gmail, including during the user's daily briefing, ..."
        # for a reconnect_using edge, or "Church's Texas Chicken offers a
        # chicken wrap, and ..." for an offers edge. Re-rendering those from
        # the bare triple would DESTROY content to fix formatting that was
        # never broken. The legacy format is exactly
        # f"{source} {predicate.replace('_',' ')} {target}".
        legacy = f"{source} {str(predicate).replace('_', ' ')} {target}"
        if (mem.content or "").strip() != legacy.strip():
            stats["richer_content_kept"] += 1
            continue

        # Only CURATED predicates may rewrite existing content. The verb-repair
        # fallback in humanize_relationship is a reasonable default for a NEW
        # write, but it is too blunt to run over rows that already exist: on the
        # first dry-run it turned "Rampage might cause problems in Canada" into
        # "Rampage mights cause problems in Canada". The modal bug is fixed, but
        # the general lesson stands — an uncurated guess is not worth overwriting
        # a row that already reads as English. `play_on`, `performed_by`,
        # `owner_of`, `located_in` are in the tables precisely because someone
        # decided what they should say.
        pkey = str(predicate).lower().replace(" ", "_")
        pkey = _PREDICATE_ALIASES.get(pkey, pkey)
        if pkey not in _PREDICATE_TEMPLATES:
            stats["uncurated_predicate_skipped"] += 1
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
    stats = {"candidates": 0, "reanchored": 0, "already_past": 0, "had_no_lease": 0}
    now = datetime.utcnow()
    # NOT filtered on `expires_at IS NOT NULL`. A sub-hour reminder with NO
    # lease is the worse case, not the safe one — it never expires at all.
    # Rows reach that state legitimately (the Keep path and the durable-
    # restatement promotion both clear the lease deliberately) and, in at least
    # one case, because verifying the Keep fix in production cleared the lease
    # on a dead reminder: DecayService.reinforce_memory commits internally, so
    # a probe that rolls back afterwards has already changed the row.
    rows = (await session.execute(
        select(Memory).where(
            Memory.is_active == True,  # noqa: E712
            Memory.is_deleted == False,  # noqa: E712
        )
    )).scalars().all()

    for mem in rows:
        if not _SUBHOUR_HORIZON.search(mem.content or ""):
            continue
        if describes_recurring_arrangement(mem.content):
            continue
        stats["candidates"] += 1
        if mem.expires_at is None:
            stats["had_no_lease"] += 1
        anchor = mem.created_at or now
        # A sub-hour reminder is dead the day it was made. One day of grace so
        # it stays visible for the rest of that day, matching resolve_ttl_days.
        horizon = anchor + timedelta(days=1)
        if mem.expires_at is not None and mem.expires_at <= horizon:
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


async def recategorize_relationship_rows(
    session: AsyncSession, apply: bool
) -> Dict[str, int]:
    """Re-derive the category of legacy relationship rows from entity types.

    Before the taxonomy work, `store_entity_relationship` hardcoded
    `"people" if source_type == "person" else "knowledge"` — a binary that
    produced exactly two of the twenty categories. It still shows: 48
    `knowledge` + 24 `people` and nothing else across 72 relationship rows on
    the founder's tenant, which is why "User owns Toup" is filed under People.

    The row itself cannot answer this: `metadata_json` records
    source_entity / target_entity / relationship_type / extracted_by — names
    but NOT types. The types live on the `entities` table, so recover them by
    name and then apply `category_for_relationship`, exactly as a fresh write
    would.

    Category is a display and filter concern, fully reversible, and the event
    records the previous value.
    """
    from app.db.models import Entity

    stats = {
        "scanned": 0, "recategorized": 0, "unchanged": 0,
        "entity_type_unknown": 0,
    }

    ents = (await session.execute(select(Entity))).scalars().all()
    # Entity names are what the triple stores; fold case so "User"/"user" match.
    by_name = {}
    for e in ents:
        nm = (getattr(e, "name", None) or "").strip().lower()
        et = getattr(e, "entity_type", None) or getattr(e, "type", None)
        if nm and et and nm not in by_name:
            by_name[nm] = et

    rows = (await session.execute(
        select(Memory).where(
            Memory.source_type == "entity_extraction",
            Memory.is_deleted == False,  # noqa: E712
        )
    )).scalars().all()

    for mem in rows:
        stats["scanned"] += 1
        try:
            meta = json.loads(mem.metadata_json) if mem.metadata_json else {}
        except (TypeError, ValueError):
            meta = {}
        src = (meta.get("source_entity") or meta.get("source_name") or "").strip().lower()
        tgt = (meta.get("target_entity") or meta.get("target_name") or "").strip().lower()
        st, tt = by_name.get(src), by_name.get(tgt)
        if not (st or tt):
            stats["entity_type_unknown"] += 1
            continue

        want = category_for_relationship(st, tt)
        if want == mem.category:
            stats["unchanged"] += 1
            continue

        logger.info(
            "  %-14s -> %-14s (%s/%s) %s",
            mem.category, want, st or "?", tt or "?", (mem.content or "")[:44],
        )
        if apply:
            session.add(_event(
                mem, mem.user_id, "recategorize_relationship",
                {"category": mem.category},
            ))
            mem.category = want
        stats["recategorized"] += 1

    return stats


STEPS = {
    "restore": ("restore wrongly-archived routines", restore_archived_routines),
    "humanize": ("humanize legacy relationship content", humanize_relationship_rows),
    "reminders": ("re-anchor dead reminder leases", rearchive_dead_reminders),
    "recategorize": (
        "re-derive relationship categories from entity types",
        recategorize_relationship_rows,
    ),
}

# `recategorize` is NOT in the default run. Its logic is right, but bulk-applying
# it to legacy rows makes the data worse, because it depends on entity TYPES and
# those are unreliable upstream. Measured on the founder's tenant: ~5 rows
# clearly improved ("User owns Toup" people -> work, "Shops at Don Mills is in
# Toronto" knowledge -> locations) and ~8 clearly degraded, all from bad typing
# rather than bad precedence:
#
#   "Better Call Saul is available on Netflix"      topic/organization -> work
#       (the show is typed `topic`, so MEDIA never enters the running)
#   "Arash performed Dooset Daram"                  person/project     -> work
#   "Drake artist of 0-100"                         person/project     -> work
#       (songs are typed `project`)
#   "Church's Texas Chicken offers a chicken wrap"  organization/topic -> work
#       (should be preferences)
#
# No subset of entity-type pairs separates the good cases from the bad — the
# same person/project pair produces both "User owns Toup" and "Drake artist of
# 0-100". Fixing entity typing is the prerequisite, and that is separate work.
# Reachable deliberately with `--only recategorize` once it lands.
DEFAULT_STEPS = ("restore", "humanize", "reminders")


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
    chosen = args.only or list(DEFAULT_STEPS)

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
