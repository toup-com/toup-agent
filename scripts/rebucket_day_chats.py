#!/usr/bin/env python3
"""Re-derive day-chat membership from each message's own timestamp.

Runs INSIDE an agent container (`docker exec -it <c> python
scripts/rebucket_day_chats.py …`) — it imports the tenant's own app, so the
repair is the same code the list endpoint and the timezone-learn hook call.
A fourth divergent copy is what this file exists to prevent.

    # what would move, nothing written
    python scripts/rebucket_day_chats.py --dry-run

    # repair every user holding a day dated ahead of their local today
    python scripts/rebucket_day_chats.py --apply

    # re-derive EVERY loaded day for one user
    python scripts/rebucket_day_chats.py --user-id <uuid> --scope all --apply

`--scope future` (the default) touches only days whose `local_date` is still
ahead of the user's local today — the incident shape. `--scope all` re-derives
every day in the window and is the bigger hammer: it can move rows a user has
been reading for months, and it marks every touched day's summary stale, which
re-queues LLM work per day. Read the dry run first.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rebucket")


def _local_today(tz_name):
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        return None


async def _candidate_user_ids(session_maker, explicit):
    """Users to consider: the one named, or everyone holding a day dated
    ahead of their own local today."""
    from sqlalchemy import select

    from app.db.models import User
    from app.db.models.day_chat import DayChat

    if explicit:
        return [explicit]

    async with session_maker() as db:
        rows = (await db.execute(
            select(DayChat.user_id, DayChat.local_date, User.timezone)
            .join(User, DayChat.user_id == User.id)
        )).all()
    out = []
    for user_id, local_date, tz_name in rows:
        if not tz_name or user_id in out:
            continue
        today = _local_today(tz_name)
        if today is not None and local_date and local_date > today:
            out.append(user_id)
    return out


async def _user_tz(session_maker, user_id):
    from sqlalchemy import select

    from app.db.models import User

    async with session_maker() as db:
        return (await db.execute(
            select(User.timezone).where(User.id == user_id)
        )).scalar()


async def main(args) -> int:
    from app.db.database import async_session_maker
    from app.services.day_chat_rebucket import (
        ConvRow,
        CblRow,
        DayRow,
        MsgRow,
        apply_rebucket,
        plan_rebucket,
    )

    if args.apply:
        from app.services.day_chat_rebucket import rebucket_enabled
        if not rebucket_enabled() and not getattr(args, "force", False):
            print("day_chat_rebucket_enabled is OFF for this process — refusing --apply. "
                  "Pass --force to override the kill switch deliberately.")
            return 2

    now = datetime.now(timezone.utc)
    user_ids = await _candidate_user_ids(async_session_maker, args.user_id)
    if not user_ids:
        print("No users with a future-dated day chat. Nothing to do.")
        return 0

    report = {"generated_at": now.isoformat(), "scope": args.scope,
              "applied": bool(args.apply), "users": []}

    for user_id in user_ids:
        tz_name = await _user_tz(async_session_maker, user_id)
        if not tz_name:
            print(f"{user_id[:8]}: no timezone on the users row — SKIPPED "
                  f"(a UTC guess is what creates these rows)")
            report["users"].append({"user_id": user_id, "skipped": "no timezone"})
            continue

        # Read through the same loaders the service uses, so a dry run and an
        # apply can never disagree about what the plan is.
        from sqlalchemy import select

        from app.db.models import Conversation, Message
        from app.db.models.day_chat import ContextBudgetLog, DayChat

        async with async_session_maker() as db:
            day_rows = (await db.execute(
                select(DayChat.id, DayChat.local_date)
                .where(DayChat.user_id == user_id)
                .order_by(DayChat.local_date.desc())
                .limit(args.limit_days)
            )).all()
            days = [DayRow(id=r[0], local_date=r[1]) for r in day_rows]
            local_today = _local_today(tz_name)
            if args.scope == "all":
                scope_ids = {d.id for d in days}
            else:
                scope_ids = {d.id for d in days if local_today and d.local_date > local_today}
            if not scope_ids:
                report["users"].append({"user_id": user_id, "skipped": "nothing in scope"})
                continue
            messages = [
                MsgRow(id=r[0], created_at=r[1], day_chat_id=r[2], conversation_id=r[3])
                for r in (await db.execute(
                    select(Message.id, Message.created_at, Message.day_chat_id,
                           Message.conversation_id)
                    .where(Message.day_chat_id.in_(sorted(scope_ids)))
                )).all()
            ]
            conversations = [
                ConvRow(id=r[0], day_chat_id=r[1], channel=r[2], is_active=bool(r[3]))
                for r in (await db.execute(
                    select(Conversation.id, Conversation.day_chat_id,
                           Conversation.channel, Conversation.is_active)
                    .where(Conversation.user_id == user_id)
                )).all()
            ]
            cbls = [
                CblRow(id=r[0], day_chat_id=r[1], created_at=r[2], turn_id=r[3])
                for r in (await db.execute(
                    select(ContextBudgetLog.id, ContextBudgetLog.day_chat_id,
                           ContextBudgetLog.created_at, ContextBudgetLog.turn_id)
                    .where(ContextBudgetLog.day_chat_id.in_(sorted(scope_ids)))
                )).all()
            ]

        plan = plan_rebucket(
            days=days, messages=messages, conversations=conversations, cbls=cbls,
            tz_name=tz_name, now_utc=now, scope_day_ids=scope_ids,
        )
        day_date = {d.id: d.local_date for d in days}
        msg_day = {m.id: m.day_chat_id for m in messages}
        entry = {
            "user_id": user_id,
            "timezone": tz_name,
            "local_today": local_today.isoformat() if local_today else None,
            "messages": {
                mid: {
                    "from": (
                        day_date[src].isoformat()
                        if (src := msg_day.get(mid)) in day_date else None
                    ),
                    "to": d.isoformat(),
                }
                for mid, d in plan.message_moves.items()
            },
            "conversations": {k: v.isoformat() for k, v in plan.conversation_moves.items()},
            "deactivate": {k: v.isoformat() for k, v in plan.conversation_deactivate.items()},
            "context_budget_logs": {k: v.isoformat() for k, v in plan.cbl_moves.items()},
            "days_needed": sorted(d.isoformat() for d in plan.days_needed),
            "days_deletable": sorted(plan.days_deletable),
            "days_retained": plan.days_retained,
            "noop": plan.noop,
        }

        print(f"\n── {user_id[:8]} ({tz_name}, today={entry['local_today']}) ──")
        if plan.noop:
            print("  nothing to do")
        else:
            print(f"  messages      {len(plan.message_moves)}")
            print(f"  conversations {len(plan.conversation_moves)} "
                  f"(+{len(plan.conversation_deactivate)} deactivated by the "
                  f"partial unique index)")
            print(f"  budget logs   {len(plan.cbl_moves)}")
            print(f"  days to create {sorted(d.isoformat() for d in plan.days_needed)}")
            print(f"  days to delete {sorted(i[:8] for i in plan.days_deletable)}")
            for did, reason in sorted(plan.days_retained.items()):
                print(f"  RETAINED {did[:8]}: {reason}")
            for mid, move in sorted(entry["messages"].items()):
                print(f"    {mid[:12]}  {move['from']} -> {move['to']}")

        if args.apply and not plan.noop:
            async with async_session_maker() as db:
                res = await apply_rebucket(db, user_id, plan, tz_name=tz_name)
            entry["result"] = {
                "changed": res.changed,
                "moved_messages": res.moved_messages,
                "moved_conversations": res.moved_conversations,
                "deactivated": res.deactivated,
                "moved_cbls": res.moved_cbls,
                "days_created": res.days_created,
                "days_deleted": res.days_deleted,
                "days_retained": res.days_retained,
                "per_date_errors": res.per_date_errors,
            }
            print(f"  APPLIED: {entry['result']}")
            if res.per_date_errors:
                print("  !! some target dates failed; the others committed")
        report["users"].append(entry)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-derive day-chat membership")
    ap.add_argument("--user-id", default=None,
                    help="Only this user (default: every user with a future-dated day)")
    ap.add_argument("--scope", choices=("future", "all"), default="future")
    ap.add_argument("--limit-days", type=int, default=90)
    ap.add_argument("--json", default=None, help="Write the full report here")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Apply even while DAY_CHAT_REBUCKET_ENABLED=0 (the kill switch)")
    a = ap.parse_args()
    if a.apply:
        a.dry_run = False
    sys.exit(asyncio.run(main(a)))
