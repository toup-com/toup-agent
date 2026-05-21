"""
Recent Days Service — day-boundary warm-start recap.

When a user starts a new day, the agent has no in-day rolling summary yet,
no recent message history, and the only continuity surface is the
<active_tasks> block (open threads). That misses everything that already
*wrapped up*: yesterday's wins, decisions made, conclusions reached.

This service surfaces the last 1–2 days' archival summaries as a
<recent_days> system-prompt block, ONLY when:
  - day-chat mode is on,
  - today's day-chat is "fresh" (no rolling summary, low message count),
  - prior days actually have archival summaries to surface.

Once a day is active enough to have its own rolling summary, that takes
over — we don't double-up archival recap on top of an active conversation.

F8 (2026-05-08): closes the day-boundary continuity gap. The archival
summary table was already populated nightly by scheduled_tasks.py; the
data was sitting unused until this surface was added.
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Default: surface up to the last 2 calendar days. Going further is noise —
# anything older that still matters should already be a memory or active task.
DEFAULT_LIMIT_DAYS = 2

# Hard cap on rows fetched from day_chats irrespective of date-range
# math (TKT-LAT-009). The date-range already bounds the result tightly
# in normal use, but a stale local_date on a row + a malformed
# limit_days override could in principle scan the whole table for that
# user. Belt-and-suspenders.
DAY_CHATS_ROW_HARD_CAP = 30

# Inject only when today's day-chat has fewer than this many messages.
# Two turns of a back-and-forth = 4 messages; once we're past that, the
# active conversation has its own continuity and the archival recap
# starts competing with live context.
FRESH_DAY_MESSAGE_THRESHOLD = 4

# Truncate each day's recap to this many chars to keep total prompt
# overhead bounded (~250 tokens for 2 days). Archival summaries are
# typically ~400-800 chars; this gives us "lead paragraph" semantics.
PER_DAY_CHAR_CAP = 600


def should_inject_recent_days(day_context: Optional[Dict]) -> bool:
    """Gate the <recent_days> block on "is this a fresh day?".

    Fresh = no in-day rolling summary AND few messages so far. Once the
    day grows its own summary, we stop injecting — the active conversation
    is the better continuity surface.
    """
    if not day_context:
        return False
    if day_context.get("summary"):
        return False
    if day_context.get("message_count", 0) >= FRESH_DAY_MESSAGE_THRESHOLD:
        return False
    return True


async def get_recent_day_summaries(
    db: AsyncSession,
    user_id: str,
    today_local_date: date,
    limit_days: int = DEFAULT_LIMIT_DAYS,
) -> List[Dict]:
    """Fetch archival summaries from the last N calendar days, excluding today.

    Falls back to rolling_summary when archival_summary is missing — this
    matters for days where the archival job hasn't run yet (e.g. yesterday
    if the user is checking in before the 5am UTC archival sweep).

    Returns most-recent-first list of:
        {
            "local_date": date,
            "summary": str,
            "is_archival": bool,  # False if rolling_summary fallback
        }
    """
    from app.db.models.day_chat import DayChat

    cutoff = today_local_date - timedelta(days=limit_days)

    rows = (await db.execute(
        select(DayChat).where(
            and_(
                DayChat.user_id == user_id,
                DayChat.local_date >= cutoff,
                DayChat.local_date < today_local_date,
            )
        ).order_by(DayChat.local_date.desc()).limit(DAY_CHATS_ROW_HARD_CAP)
    )).scalars().all()

    out: List[Dict] = []
    for dc in rows:
        summary_text = None
        is_archival = False
        if dc.archival_summary and dc.archival_summary_status == "up_to_date":
            summary_text = dc.archival_summary
            is_archival = True
        elif dc.rolling_summary:
            summary_text = dc.rolling_summary
            is_archival = False

        if not summary_text:
            continue

        out.append({
            "local_date": dc.local_date,
            "summary": summary_text.strip(),
            "is_archival": is_archival,
        })

    return out


def build_recent_days_block(
    summaries: List[Dict],
    today_local_date: Optional[date] = None,
) -> str:
    """Build the <recent_days> system prompt section.

    Tone-matched to the founder's friend-not-chatbot guards: this is
    context for the agent to draw on, NOT a script to recite. The
    instructions explicitly forbid the "Yesterday you did X, Y, Z"
    list-back behavior.
    """
    if not summaries:
        return ""

    lines = [
        "\n<recent_days>",
        "What happened with this user over the last day or two — for "
        "your own continuity. They didn't ask for a recap; don't give "
        "one. Use this so you don't sound like you're meeting them for "
        "the first time. If something here connects naturally to what "
        "they're saying right now, reference it the way a friend would "
        "(\"oh did the deploy thing finally land?\") — but if it doesn't "
        "fit, just hold the context quietly and move on.",
        "",
        "**Never** open with \"Yesterday you...\" or list these back. "
        "That's robot behavior. The point is that you already know.",
        "",
    ]

    for entry in summaries:
        d: date = entry["local_date"]
        text: str = entry["summary"]
        if len(text) > PER_DAY_CHAR_CAP:
            text = text[:PER_DAY_CHAR_CAP].rstrip() + "…"

        # Relative label is more useful than an ISO date for tonal
        # calibration — "yesterday" reads as recent, "2 days ago" reads
        # as catch-up territory.
        if today_local_date is not None:
            delta = (today_local_date - d).days
            if delta == 1:
                label = "Yesterday"
            elif delta == 2:
                label = "2 days ago"
            else:
                label = f"{delta} days ago"
            label = f"{label} ({d.isoformat()})"
        else:
            label = d.isoformat()

        lines.append(f"**{label}:** {text}")
        lines.append("")

    lines.append("</recent_days>")
    return "\n".join(lines) + "\n"
