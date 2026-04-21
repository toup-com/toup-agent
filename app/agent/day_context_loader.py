"""
Day Context Loader — loads ALL messages for a DayChat across all sessions/channels.

This replaces _load_history() when USE_DAY_CHAT_CONTEXT is enabled.
Messages are returned in strict chronological order with channel annotations
so the LLM knows where each message came from.

Feature flag discipline: this module is ONLY called when both:
  1. USE_DAY_CHAT_CONTEXT=true
  2. day_chat_backfill migration is completed
If either condition fails, the caller falls back to the old _load_history().
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tool_elision import elide_tool_results

logger = logging.getLogger(__name__)

# Token budget: 60% of model context window for history + summary
HISTORY_BUDGET_RATIO = 0.60
# Recent messages to keep verbatim when over budget
VERBATIM_WINDOW = 20
# Tool elision: keep full results for last N tool turns
TOOL_ELISION_RECENT = 10


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _format_time(dt: datetime) -> str:
    """Format datetime as h:MMam/pm for message annotations."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%-I:%M%p").lower()


def annotate_message(content: str, channel: str, created_at: datetime) -> str:
    """Prefix message content with channel and time annotation.

    Example: [web 9:14am] Original message content here...
    """
    time_str = _format_time(created_at)
    tag = f"[{channel} {time_str}]" if time_str else f"[{channel}]"
    return f"{tag} {content}"


async def load_day_context(
    db: AsyncSession,
    day_chat_id: str,
    model: str = "claude-opus-4-6",
    model_context_tokens: int = 200_000,
    calling_channel: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the full day's message history for context assembly.

    Returns:
        {
            "summary": str or None,       # Rolling summary of older messages
            "messages": List[Dict],        # Chronological message list (annotated)
            "raw_messages": List[Dict],    # Unannotated for save/reference
            "total_tokens": int,           # Estimated token count
            "summary_was_stale": bool,     # True if summary_status != 'up_to_date'
            "message_count": int,          # Total messages loaded
        }
    """
    from app.db.models import Message, Conversation
    from app.db.models.day_chat import DayChat

    # Entry-log so operators can see per-channel load behavior in prod
    # without speculation. Part of the Rule 12 observability sweep.
    logger.info(
        "[day_ctx] load_day_context entry day_chat_id=%s calling_channel=%s model=%s",
        day_chat_id[:8] if day_chat_id else None, calling_channel, model,
    )

    # Load the DayChat row
    dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
    if not dc:
        logger.warning(
            "[day_ctx] load_day_context no_day_chat day_chat_id=%s calling_channel=%s — "
            "empty history returned",
            day_chat_id[:8] if day_chat_id else None, calling_channel,
        )
        return {"summary": None, "messages": [], "raw_messages": [], "total_tokens": 0,
                "summary_was_stale": False, "message_count": 0}

    summary = dc.rolling_summary
    summary_was_stale = dc.summary_status not in ("up_to_date", None)

    # Load ALL messages for this day, across all sessions, chronologically.
    # Select Conversation.channel as a secondary hint — per-message Message.channel
    # is preferred (denormalized at write time, survives channel switches
    # mid-day) but the backfill covers old rows via Conversation.channel.
    result = await db.execute(
        select(Message, Conversation.channel)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.day_chat_id == day_chat_id)
        .order_by(Message.created_at.asc())
    )
    rows = result.all()

    # Build annotated message list
    raw_messages: List[Dict[str, Any]] = []
    annotated_messages: List[Dict[str, Any]] = []

    from app.agent.channel_util import resolve_channel
    for msg, conv_channel in rows:
        if msg.role not in ("user", "assistant"):
            continue
        raw = {"role": msg.role, "content": msg.content}
        raw_messages.append(raw)

        # Annotate with channel and time. resolve_channel prefers the
        # per-message channel (Rule 12), falls back to the conversation's
        # channel, logs a WARNING on fallback so silent drift is visible.
        _msg_channel = resolve_channel(
            payload_hint=getattr(msg, "channel", None),
            conversation_hint=conv_channel,
            site="history_annotation",
        )
        annotated_content = annotate_message(msg.content, _msg_channel, msg.created_at)
        annotated_messages.append({"role": msg.role, "content": annotated_content})

    # Estimate total tokens
    total_tokens = sum(_estimate_tokens(m["content"]) for m in annotated_messages)
    if summary:
        total_tokens += _estimate_tokens(summary)

    budget = int(model_context_tokens * HISTORY_BUDGET_RATIO)

    # If within budget, return everything
    if total_tokens <= budget:
        # Still apply tool elision for very long tool results
        annotated_messages = elide_tool_results(annotated_messages, keep_recent_turns=TOOL_ELISION_RECENT, format="anthropic")
        return {
            "summary": summary,
            "messages": annotated_messages,
            "raw_messages": raw_messages,
            "total_tokens": total_tokens,
            "summary_was_stale": summary_was_stale,
            "message_count": len(annotated_messages),
        }

    # Over budget: use summary + recent verbatim window
    verbatim = annotated_messages[-VERBATIM_WINDOW:]
    verbatim_tokens = sum(_estimate_tokens(m["content"]) for m in verbatim)

    # Apply tool elision to the verbatim window
    verbatim = elide_tool_results(verbatim, keep_recent_turns=TOOL_ELISION_RECENT, format="anthropic")

    return {
        "summary": summary,  # May be stale — async summarizer will update it
        "messages": verbatim,
        "raw_messages": raw_messages[-VERBATIM_WINDOW:],
        "total_tokens": verbatim_tokens + _estimate_tokens(summary or ""),
        "summary_was_stale": True,  # We're over budget, summary needs regeneration
        "message_count": len(rows),
    }


def build_today_so_far_block(summary: Optional[str]) -> str:
    """Build the <today_so_far> system prompt section from rolling summary.

    Only returns content when a summary exists. Injected into the system prompt
    to give the LLM awareness of earlier activity today.
    """
    if not summary:
        return ""

    return (
        "\n<today_so_far>\n"
        "Summary of earlier activity today across all channels. "
        "Channel tags show where each interaction happened.\n\n"
        f"{summary}\n"
        "</today_so_far>\n"
    )
