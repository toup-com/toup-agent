#!/usr/bin/env python3
"""F4 — backfill legacy memory→routine linkage into ref_kind / ref_id.

Before Ticket 2 added the explicit `memories.ref_kind` + `memories.ref_id`
columns (commit 13d5704, 2026-05-13), routine-linked memories carried
the routine id only in the freeform `metadata_json["routine_id"]` blob.
The MCP `memory_create` upsert path doesn't see that — it only consults
the new columns — so legacy rows are still vulnerable to the duplicate
bug.

This script walks every memory whose `metadata_json` parses as JSON and
contains a `routine_id` (or `trigger_id`, `connector_id`) and stamps
the corresponding `ref_kind` + `ref_id`. Where a (user, ref_kind,
ref_id) collision already exists, it picks the newest row as the
keeper and soft-deletes the rest (`is_deleted=TRUE`,
`deleted_at=NOW()`).

Idempotent — re-running with no work to do is a no-op. Dry-run by
default; `--apply` commits.

Usage:
    python scripts/migrate_memory_routine_links.py             # dry-run
    python scripts/migrate_memory_routine_links.py --apply     # commits
    python scripts/migrate_memory_routine_links.py --user UID  # scope to one user
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_memory_routine_links")


# Map metadata_json keys → (ref_kind value). Add to this dict when a
# new entity type starts being linked from agent-created memories.
_LEGACY_KEYS: dict[str, str] = {
    "routine_id": "routine",
    "trigger_id": "trigger",
    "connector_id": "connector",
}


async def main(apply: bool, user_id_filter: str | None) -> int:
    from sqlalchemy import text

    from app.db.database import async_session_maker

    async with async_session_maker() as db:
        sql = """
            SELECT id, user_id, metadata_json
            FROM memories
            WHERE metadata_json IS NOT NULL
              AND is_deleted = FALSE
              AND ref_id IS NULL
        """
        params: dict[str, object] = {}
        if user_id_filter:
            sql += " AND user_id = :uid"
            params["uid"] = user_id_filter

        rows = (await db.execute(text(sql), params)).all()

        if not rows:
            print("No legacy memory rows with metadata_json found. Nothing to do.")
            return 0

        # First pass — parse metadata, pick a (ref_kind, ref_id) per row.
        candidates: list[tuple[str, str, str, str]] = []  # (mem_id, user_id, ref_kind, ref_id)
        skipped_unparseable = 0
        skipped_no_ref = 0
        for row in rows:
            meta_raw = row.metadata_json or ""
            try:
                meta = json.loads(meta_raw)
            except (TypeError, ValueError):
                skipped_unparseable += 1
                continue
            if not isinstance(meta, dict):
                skipped_no_ref += 1
                continue
            ref_kind: str | None = None
            ref_id_val: str | None = None
            for key, kind in _LEGACY_KEYS.items():
                v = meta.get(key)
                if v:
                    ref_kind = kind
                    ref_id_val = str(v)
                    break
            if not (ref_kind and ref_id_val):
                skipped_no_ref += 1
                continue
            candidates.append((row.id, row.user_id, ref_kind, ref_id_val))

        print(
            f"\nLegacy memory rows scanned: {len(rows)}. "
            f"Stamp candidates: {len(candidates)}. "
            f"Skipped (unparseable JSON): {skipped_unparseable}. "
            f"Skipped (no ref key in metadata): {skipped_no_ref}."
            f"{' DRY-RUN — no writes.' if not apply else ' APPLY — committing.'}\n"
        )

        # Second pass — group by (user, ref_kind, ref_id) so we can pick
        # the keeper when multiple legacy rows map to the same entity.
        groups: dict[tuple[str, str, str], list[str]] = {}
        for mem_id, uid, ref_kind, ref_id_val in candidates:
            groups.setdefault((uid, ref_kind, ref_id_val), []).append(mem_id)

        # Order each group by created_at DESC so the most recent stays
        # the active row. Fetch created_at in one query for all members.
        order_sql = """
            SELECT id, created_at
            FROM memories
            WHERE id = ANY(:ids)
            ORDER BY created_at DESC
        """
        total_stamped = 0
        total_soft_deleted = 0
        for (uid, ref_kind, ref_id_val), member_ids in groups.items():
            if len(member_ids) == 1:
                # Solo row — straight stamp, no collision.
                only = member_ids[0]
                print(f"  stamp     {ref_kind}/{ref_id_val[:12]} user={uid[:8]} mem={only[:8]}")
                total_stamped += 1
                if apply:
                    await db.execute(
                        text("UPDATE memories SET ref_kind=:rk, ref_id=:ri WHERE id=:id"),
                        {"rk": ref_kind, "ri": ref_id_val, "id": only},
                    )
                continue
            # Collision — multiple legacy rows pointed at the same
            # entity. Keep the newest, archive the rest.
            ordered = (await db.execute(
                text(order_sql), {"ids": member_ids}
            )).all()
            keeper = ordered[0].id
            archive = [r.id for r in ordered[1:]]
            print(
                f"  collision {ref_kind}/{ref_id_val[:12]} user={uid[:8]} "
                f"keep={keeper[:8]} archive={len(archive)}"
            )
            total_stamped += 1
            total_soft_deleted += len(archive)
            if apply:
                await db.execute(
                    text("UPDATE memories SET ref_kind=:rk, ref_id=:ri WHERE id=:id"),
                    {"rk": ref_kind, "ri": ref_id_val, "id": keeper},
                )
                await db.execute(
                    text("UPDATE memories SET is_deleted=TRUE, deleted_at=NOW() "
                         "WHERE id = ANY(:ids)"),
                    {"ids": archive},
                )

        if apply:
            await db.commit()
            print(
                f"\nCOMMITTED. stamped={total_stamped} "
                f"soft_deleted={total_soft_deleted}"
            )
        else:
            print(
                f"\nDRY-RUN totals: would_stamp={total_stamped} "
                f"would_soft_delete={total_soft_deleted}"
            )
            print("Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually commit. Default is dry-run.",
    )
    parser.add_argument(
        "--user", dest="user_id", default=None,
        help="Limit to one user_id.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(apply=args.apply, user_id_filter=args.user_id)))
