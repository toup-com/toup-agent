"""Reply-to quote rendering — shared between live turn + history annotation.

When a user replies to a specific earlier message (via the web Reply
affordance, Telegram native reply, etc.), the threading link is stored as
``messages.reply_to_message_id``. To make the LLM understand the
relationship — not just see two unrelated turns — we prefix the replying
turn's content with a short quoted preamble before handing it to the
model.

Two call sites use this:

1. ``api/ws_chat.py`` — the current turn's user text, before passing to
   ``agent_runner.run(user_message=...)``.
2. ``agent/day_context_loader.py`` — historical user turns when assembling
   day context for the LLM.

The preamble is LLM-only. The DB row stores the user's clean original
content; the structured pointer (``reply_to_message_id``) is what the
frontend reads to render the quoted card.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Keep the inline quote short so it doesn't dominate the turn's token
# budget on long-original replies. UI shows the full content on hover.
_MAX_QUOTE_CHARS = 240


def _format_quote_excerpt(content: str) -> str:
    text = (content or "").strip().replace("\n", " ")
    if len(text) > _MAX_QUOTE_CHARS:
        text = text[: _MAX_QUOTE_CHARS - 1] + "…"
    return text


def _format_local_time(dt: Optional[datetime], tz_name: Optional[str]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            dt = dt.astimezone(ZoneInfo(tz_name))
        except Exception:
            pass
    return dt.strftime("%-I:%M%p").lower()


def render_reply_preamble(
    target_role: str,
    target_content: str,
    target_created_at: Optional[datetime] = None,
    tz_name: Optional[str] = None,
) -> str:
    """Render the quoted-reply block that prefixes a replying turn.

    Uses XML-style tags rather than the looser ``[Replying to …]`` inline
    form because frontier models attend to tagged blocks much more
    reliably than to bracketed prose buried in a paragraph. The user's
    actual question follows the block, separated by a blank line, so the
    model treats the quoted content as context for that question.
    """
    quote = _format_quote_excerpt(target_content)
    who = "your earlier message" if target_role == "assistant" else "their earlier message"
    time_str = _format_local_time(target_created_at, tz_name)
    when = f" sent at {time_str}" if time_str else ""
    return (
        "<reply_to>\n"
        f"The user is directly replying to {who}{when}. "
        f"Their question below is about THIS specific message, not earlier "
        f"unrelated topics in the day. Quoted content:\n"
        f'"""\n{quote}\n"""\n'
        "</reply_to>"
    )


async def fetch_reply_targets(
    db: AsyncSession,
    message_ids: Sequence[str],
) -> Dict[str, "ReplyTarget"]:
    """Bulk-fetch reply target rows for use in history annotation.

    Returns a map id → ReplyTarget so the loader avoids N+1 queries when a
    day has multiple reply rows.
    """
    ids = [m for m in message_ids if m]
    if not ids:
        return {}
    from app.db.models import Message  # local import: avoid agent boot cycle

    rows = (
        await db.execute(
            select(Message.id, Message.role, Message.content, Message.created_at)
            .where(Message.id.in_(ids))
        )
    ).all()
    return {
        rid: ReplyTarget(id=rid, role=rrole, content=rcontent, created_at=rcreated)
        for (rid, rrole, rcontent, rcreated) in rows
    }


class ReplyTarget:
    __slots__ = ("id", "role", "content", "created_at")

    def __init__(
        self,
        id: str,
        role: str,
        content: str,
        created_at: Optional[datetime],
    ):
        self.id = id
        self.role = role
        self.content = content
        self.created_at = created_at
