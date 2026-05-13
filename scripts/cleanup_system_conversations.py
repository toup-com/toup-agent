#!/usr/bin/env python3
"""Merge duplicate system-channel Conversations into one row per
(user_id, day_chat_id, channel).

Background: before the Reading-A fix landed
(docs/bug-sweep-2026-05-13.md Ticket 1), `routines.message_writer`
and `triggers.message_writer` inserted a fresh `Conversation` per
fire. This script reconciles existing data so the partial unique index
`ix_conversations_system_channel_per_day` can be installed without
collisions and the sidebar session counter stops over-reporting.

Per (user, day, channel) group of duplicates:
    1. Pick the EARLIEST `started_at` Conversation as the survivor.
    2. Reassign every Message in the losing rows to the survivor.
    3. Soft-delete the losing rows (is_active=FALSE, ended_at=NOW()).

Idempotent — re-running with no duplicates is a no-op. Dry-run by
default; pass `--apply` to actually write.

Usage:
    python scripts/cleanup_system_conversations.py             # dry-run, prints plan
    python scripts/cleanup_system_conversations.py --apply     # commits the merge
    python scripts/cleanup_system_conversations.py --user UID  # limit scope to one user
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cleanup_system_conversations")


SYSTEM_CHANNELS = ("routine", "trigger", "api", "digest")


async def main(apply: bool, user_id_filter: str | None) -> int:
    from sqlalchemy import text

    from app.db.database import async_session_maker

    async with async_session_maker() as db:
        # Find all (user, day, channel) groups with >1 active Conversation.
        # Sort the id list by started_at ASC so the survivor is the earliest.
        sql = """
            SELECT
                user_id,
                day_chat_id,
                channel,
                array_agg(id ORDER BY started_at ASC) AS ids,
                COUNT(*) AS dup_count
            FROM conversations
            WHERE channel = ANY(:channels)
              AND is_active = TRUE
              AND day_chat_id IS NOT NULL
        """
        params: dict[str, object] = {"channels": list(SYSTEM_CHANNELS)}
        if user_id_filter:
            sql += " AND user_id = :uid"
            params["uid"] = user_id_filter
        sql += """
            GROUP BY user_id, day_chat_id, channel
            HAVING COUNT(*) > 1
            ORDER BY dup_count DESC
        """

        rows = (await db.execute(text(sql), params)).all()

        if not rows:
            print("No duplicate system-channel conversations found. Nothing to do.")
            return 0

        total_groups = len(rows)
        total_archive = 0
        total_messages_reassigned = 0

        print(
            f"\nFound {total_groups} (user, day, channel) groups with duplicates."
            f"{' DRY-RUN — no writes.' if not apply else ' APPLY — committing.'}\n"
        )

        for row in rows:
            ids = list(row.ids)
            keep_id = ids[0]
            archive_ids = ids[1:]

            # Count messages we're about to reassign (informational).
            msg_count = (await db.execute(
                text("SELECT COUNT(*) FROM messages WHERE conversation_id = ANY(:archive)"),
                {"archive": archive_ids},
            )).scalar_one()

            print(
                f"  {row.channel:8s} user={row.user_id[:8]} day={str(row.day_chat_id)[:8]} "
                f"keep={keep_id[:8]} archive={len(archive_ids)} msgs={msg_count}"
            )

            total_archive += len(archive_ids)
            total_messages_reassigned += int(msg_count or 0)

            if not apply:
                continue

            # Reassign messages. `day_chat_id` on Message is already
            # canonical (Day-as-Chat invariant), so we only rewrite
            # conversation_id — the cross-Conversation merge doesn't
            # change which DayChat a message belongs to.
            await db.execute(
                text("UPDATE messages SET conversation_id = :keep "
                     "WHERE conversation_id = ANY(:archive)"),
                {"keep": keep_id, "archive": archive_ids},
            )
            # Soft-delete losers.
            await db.execute(
                text("UPDATE conversations SET is_active = FALSE, ended_at = NOW() "
                     "WHERE id = ANY(:archive)"),
                {"archive": archive_ids},
            )

        if apply:
            await db.commit()
            print(f"\nCOMMITTED. groups={total_groups} "
                  f"conversations_archived={total_archive} "
                  f"messages_reassigned={total_messages_reassigned}")
        else:
            print(f"\nDRY-RUN totals: groups={total_groups} "
                  f"would_archive={total_archive} "
                  f"would_reassign_messages={total_messages_reassigned}")
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
        help="Limit cleanup to one user_id (for staged rollouts).",
    )
    args = parser.parse_args()
    rc = asyncio.run(main(apply=args.apply, user_id_filter=args.user_id))
    sys.exit(rc)
