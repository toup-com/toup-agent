#!/usr/bin/env python3
"""Archive memories the write-time gate would now reject.

DRY-RUN BY DEFAULT. Nothing is written without --apply.

NEVER DELETES. Sets is_active=False, which is the same non-destructive archive
the TTL sweep uses; the row, its embedding and its history_json stay intact and
`memory_undo`/a restore query can bring any of it back. This matters because
the labelling underneath is a judgement call, and a judgement call must not be
able to destroy user data.

The verdict comes from app.services.memory_gate — the SAME module the write
path now calls. That is deliberate: if cleanup and write-time used two
different notions of junk, the brain would re-accumulate exactly the rows this
script removed, and nobody would notice for weeks.

Usage
-----
    python scripts/archive_junk_memories.py                # dry run, full report
    python scripts/archive_junk_memories.py --apply        # archive
    python scripts/archive_junk_memories.py --apply --only tautology,world_knowledge
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.db.database import async_session_maker  # noqa: E402
from app.db.models import Memory, MemoryEvent  # noqa: E402
from app.services.memory_gate import (  # noqa: E402
    memory_gate_reason,
    relationship_gate_reason,
)

TRIGGER = "junk_memory_archive"


async def _user_aliases(db: AsyncSession, user_id: str) -> list:
    aliases = []
    try:
        from app.db.models.user import User
        row = (await db.execute(
            select(User.name, User.email).where(User.id == user_id)
        )).first()
        if row:
            name, email = row
            if name:
                aliases.append(name)
                first = name.strip().split()[0] if name.strip() else ""
                if first:
                    aliases.append(first)
            if email:
                aliases.append(email.split("@")[0])
    except Exception as exc:      # pragma: no cover - defensive
        print(f"  ! alias lookup failed ({exc}); gate will be stricter")
    return aliases


def _verdict(mem: Memory, aliases: list):
    """Why this row would be rejected today, or None to keep it."""
    if mem.source_type == "entity_extraction":
        meta = {}
        try:
            meta = json.loads(mem.metadata_json or "{}")
        except Exception:
            pass
        source = meta.get("source_entity")
        target = meta.get("target_entity")
        predicate = meta.get("relationship_type")
        # A relationship row with no triple in metadata cannot be re-judged as
        # an edge; fall through to the content gate rather than guessing.
        if source and target and predicate:
            return relationship_gate_reason(
                source, predicate, target,
                user_aliases=aliases, rendered=mem.content,
            )
    # No turn context survives on a stored row, so assistant_echo cannot fire
    # here — only the structural rules (length, scaffolding) apply. Rows that
    # needed echo detection are handled going forward by the write-time gate.
    return memory_gate_reason(mem.content)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually archive (default: dry run)")
    ap.add_argument("--only", default="",
                    help="comma-separated gate reasons to act on")
    ap.add_argument("--user-id", default=None,
                    help="restrict to one tenant (default: every user in this DB)")
    args = ap.parse_args()
    only = {r.strip() for r in args.only.split(",") if r.strip()}

    async with async_session_maker() as db:
        q = select(Memory).where(Memory.is_active == True)  # noqa: E712
        if args.user_id:
            q = q.where(Memory.user_id == args.user_id)
        memories = list((await db.execute(q)).scalars())

        if not memories:
            print("no active memories in this database")
            return 0

        alias_cache = {}
        hits, reasons = [], Counter()
        for mem in memories:
            if mem.user_id not in alias_cache:
                alias_cache[mem.user_id] = await _user_aliases(db, mem.user_id)
            reason = _verdict(mem, alias_cache[mem.user_id])
            if reason and (not only or reason in only):
                hits.append((mem, reason))
                reasons[reason] += 1

        mode = "APPLY" if args.apply else "DRY RUN"
        print(f"\n=== {mode} — {len(memories)} active, {len(hits)} would be archived "
              f"({100 * len(hits) / len(memories):.0f}%) ===\n")
        for reason, n in reasons.most_common():
            print(f"  {reason:<26} {n:>3}")
        print("\n--- rows ---")
        for mem, reason in hits:
            print(f"  [{reason:<24}] acc={mem.access_count:<3} {mem.content[:88]}")

        kept = len(memories) - len(hits)
        print(f"\n  {kept} rows survive.")

        if not args.apply:
            print("\nDRY RUN — nothing written. Re-run with --apply to archive.")
            return 0

        now = datetime.utcnow()
        for mem, reason in hits:
            mem.is_active = False
            mem.updated_at = now
            db.add(MemoryEvent(
                id=__import__("uuid").uuid4().hex,
                memory_id=mem.id,
                user_id=mem.user_id,
                event_type="archived",
                timestamp=now,
                event_data_json=json.dumps({
                    "gate_reason": reason,
                    "source_type": mem.source_type,
                    "reversible": True,
                    "note": "is_active=False only; row and embedding intact",
                }),
                trigger_source=TRIGGER,
            ))
        await db.commit()
        print(f"\nARCHIVED {len(hits)} rows (reversible: is_active=False, "
              f"audited under trigger_source={TRIGGER!r}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
