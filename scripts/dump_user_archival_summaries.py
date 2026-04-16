#!/usr/bin/env python3
"""
Admin script — dump every DayChat row for one user with its summary state.

Resolves the user from --user-id or --email, then for each day_chat:
  - local_date, message_count, channels used, status fields
  - full archival_summary text (if present)
  - rolling_summary presence
  - explanatory note when a day has no archival yet

Also reports:
  - DayChats with archival_summary_status='failed'
  - Count of DayChats with zero user/assistant messages

Usage:
  python scripts/dump_user_archival_summaries.py --email ceo@ete.bar
  python scripts/dump_user_archival_summaries.py --user-id <uuid>

The script connects to the DB pointed at by DATABASE_URL (set in the
agent container's .env or exported manually). It works unchanged against
platform Supabase OR per-user Docker Postgres — whatever DATABASE_URL
points to.

  # From inside the agent container:
  DATABASE_URL=postgresql+asyncpg://postgres:postgres@host.docker.internal:5432/toup_agent_<shortid> \\
    python scripts/dump_user_archival_summaries.py --email ceo@ete.bar

Exit 0 on success, 1 on user-not-found, 2 on DB error.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence the SQLAlchemy echo that the app's default engine config turns on —
# this script prints human-readable output, not SQL traces. Must be set BEFORE
# importing app.db.database, because the engine's echo writes directly to the
# logger at INFO once initialized.
logging.basicConfig(level=logging.WARNING)
for _name in (
    "sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine",
    "sqlalchemy.pool", "sqlalchemy.orm",
):
    logging.getLogger(_name).setLevel(logging.WARNING)
    logging.getLogger(_name).propagate = False


async def _resolve_user(db, email: Optional[str], user_id: Optional[str]):
    from sqlalchemy import text

    if user_id:
        row = (await db.execute(
            text("SELECT id, email FROM users WHERE id = :uid"),
            {"uid": user_id},
        )).first()
        return row
    if email:
        row = (await db.execute(
            text("SELECT id, email FROM users WHERE lower(email) = lower(:e)"),
            {"e": email},
        )).first()
        return row
    return None


async def _dump(user_id: str) -> int:
    from sqlalchemy import text

    async with _quiet_session_maker()() as db:
        # All day_chats for this user, newest first.
        rows = (await db.execute(text("""
            SELECT
                dc.id,
                dc.local_date,
                dc.timezone,
                dc.message_count,
                dc.total_tokens,
                dc.started_at,
                dc.last_message_at,
                dc.summary_status,
                dc.summary_updated_at,
                dc.rolling_summary,
                dc.archival_summary,
                dc.archival_summary_status,
                dc.archival_summary_generated_at
            FROM day_chats dc
            WHERE dc.user_id = :uid
            ORDER BY dc.local_date DESC
        """), {"uid": user_id})).all()

        if not rows:
            print("(no DayChat rows for this user)")
            return 0

        # Per-day channel usage (from messages joined to conversations).
        channels_per_dc: dict[str, list[str]] = {}
        counts_per_dc: dict[str, int] = {}
        for r in rows:
            dc_id = r[0]
            ch = (await db.execute(text("""
                SELECT DISTINCT c.channel
                FROM messages m JOIN conversations c ON m.conversation_id = c.id
                WHERE m.day_chat_id = :dc AND m.role IN ('user','assistant')
            """), {"dc": dc_id})).all()
            channels_per_dc[dc_id] = sorted([x[0] for x in ch if x[0]])
            cnt = (await db.execute(text("""
                SELECT COUNT(*) FROM messages
                WHERE day_chat_id = :dc AND role IN ('user','assistant')
            """), {"dc": dc_id})).scalar() or 0
            counts_per_dc[dc_id] = cnt

        # Summary counts.
        total = len(rows)
        have_archival = sum(1 for r in rows if r[10])
        failed = [r for r in rows if r[11] == "failed"]
        empty = [r for r in rows if counts_per_dc[r[0]] == 0]
        only_rolling = [r for r in rows if r[9] and not r[10]]
        no_summary = [r for r in rows if not r[9] and not r[10] and counts_per_dc[r[0]] > 0]

        print("=" * 72)
        print(f"DayChats for user {user_id}")
        print(f"  total:                    {total}")
        print(f"  with archival_summary:    {have_archival}")
        print(f"  with rolling only:        {len(only_rolling)}")
        print(f"  with NO summary (non-0):  {len(no_summary)}")
        print(f"  empty (0 user/assistant): {len(empty)}")
        print(f"  status=failed:            {len(failed)}")
        print("=" * 72)
        print()

        for r in rows:
            (dc_id, local_date, tz_name, cached_count, total_tokens,
             started_at, last_msg_at,
             summary_status, summary_updated_at, rolling_summary,
             archival_summary, archival_status, archival_generated_at) = r
            real_count = counts_per_dc[dc_id]
            channels = channels_per_dc[dc_id]

            print(f"── {local_date.isoformat()}  ({dc_id[:8]})  tz={tz_name or '?'}")
            print(f"   messages (real/cached): {real_count}/{cached_count}  "
                  f"tokens={total_tokens or 0}  channels={','.join(channels) or '-'}")
            print(f"   rolling: status={summary_status} "
                  f"updated={_fmt_ts(summary_updated_at)} "
                  f"({'present' if rolling_summary else 'none'})")
            print(f"   archival: status={archival_status} "
                  f"generated={_fmt_ts(archival_generated_at)} "
                  f"({'present' if archival_summary else 'none'})")

            if archival_summary:
                print()
                print("   ┌── ARCHIVAL SUMMARY ──")
                for line in archival_summary.splitlines():
                    print(f"   │ {line}")
                print("   └──")
            else:
                # Explain why no archival.
                reason = _why_no_archival(
                    local_date, archival_status, real_count,
                )
                print(f"   [no archival — {reason}]")
                if rolling_summary and not archival_summary:
                    preview = rolling_summary[:300].replace("\n", " ")
                    ellipsis = "..." if len(rolling_summary) > 300 else ""
                    print(f"   rolling preview: {preview}{ellipsis}")

            print()

        if failed:
            print()
            print("── FAILED archival summaries ──")
            for r in failed:
                print(f"   {r[1].isoformat()}  dc={r[0][:8]}  generated_at={_fmt_ts(r[12])}")
            print()

        return 0


def _fmt_ts(ts) -> str:
    if not ts:
        return "-"
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _why_no_archival(local_date, status: str, msg_count: int) -> str:
    today_utc = datetime.now(timezone.utc).date()
    if msg_count == 0:
        return "day has no user/assistant messages"
    if local_date >= today_utc:
        return "day is current or future; archival only runs for past days"
    if status == "pending":
        return "archival job claimed the row but hasn't committed result yet"
    if status == "failed":
        return "archival generation failed; admin must reset status to retry"
    if status == "not_needed":
        return "archival job hasn't reached this row yet (next hourly tick)"
    return f"unexpected status={status!r}"


_SESSION_MAKER = None


def _quiet_session_maker():
    """Return a session maker bound to a quiet engine (no SQL echo).

    Reuses DATABASE_URL from app.config so it works identically against both
    platform Supabase and per-user Docker Postgres.
    """
    global _SESSION_MAKER
    if _SESSION_MAKER is not None:
        return _SESSION_MAKER
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from app.config import settings
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    _SESSION_MAKER = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _SESSION_MAKER


async def _main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--user-id", help="Target user UUID")
    g.add_argument("--email", help="Target user email")
    args = ap.parse_args()

    try:
        sm = _quiet_session_maker()
    except Exception as e:
        print(f"ERROR: cannot build DB session (DATABASE_URL set?): {e}", file=sys.stderr)
        return 2

    async with sm() as db:
        row = await _resolve_user(db, args.email, args.user_id)

    if not row:
        print(f"ERROR: user not found (email={args.email!r}, user_id={args.user_id!r})", file=sys.stderr)
        return 1

    uid, email = row[0], row[1]
    print(f"Resolved user: {email}  id={uid}")
    print()
    return await _dump(uid)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
