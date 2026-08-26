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

from sqlalchemy import select, and_, func, or_
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


def _format_time(dt: datetime, tz_name: Optional[str] = None) -> str:
    """Format datetime as h:MMam/pm for message annotations, in the user's
    local timezone when provided.

    Every stored `created_at` is tz-naive UTC from the DB. Historically
    this function also formatted in UTC regardless of user tz, so a
    Toronto user saw labels like `[mobile 6:41pm]` when their local clock
    said 2:41pm. That conflicted with the Runtime Context (which was
    correctly local) and GPT-class models echoed the UTC number as "your
    time." Local annotation is the fix — the agent should NEVER see a
    UTC timestamp in its prompt.

    Falls back to UTC only if tz_name is None or invalid (last resort).
    """
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            dt = dt.astimezone(ZoneInfo(tz_name))
        except Exception:
            # Invalid tz → stay on UTC rather than crashing history load.
            # Caller's tz resolution chain should log the failure upstream.
            pass
    return dt.strftime("%-I:%M%p").lower()


def annotate_message(
    content: str,
    channel: str,
    created_at: datetime,
    tz_name: Optional[str] = None,
) -> str:
    """Prefix message content with channel and time annotation.

    Example: [web 9:14am] Original message content here...

    `tz_name` formats the time in the user's local zone. When None, times
    render in UTC — keep this as a last-resort fallback, not the default.
    """
    time_str = _format_time(created_at, tz_name=tz_name)
    tag = f"[{channel} {time_str}]" if time_str else f"[{channel}]"
    return f"{tag} {content}"


async def load_day_context(
    db: AsyncSession,
    day_chat_id: str,
    model: str = "",  # empty → resolves to model_resolver.default_model() below
    model_context_tokens: int = 200_000,
    calling_channel: Optional[str] = None,
    tz_name: Optional[str] = None,
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
            "history_is_complete": bool,   # True unless the over-budget branch
                                           # truncated to summary + verbatim window
        }
    """
    from app.db.models import Message, Conversation
    from app.db.models.conversation import HIDDEN_DAY_CHANNELS
    from app.db.models.admin_dispatch import ORIGINS_EXCLUDED_FROM_CONTEXT
    from app.db.models.day_chat import DayChat

    if not model:
        from app.services.model_resolver import default_model
        model = default_model()

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
                "summary_was_stale": False, "message_count": 0,
                "history_is_complete": True}

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
        # An operator's words must never become instructions the model reads.
        # This is the R2 prompt-injection boundary and it is deliberately a
        # SECOND, independent guard: an admin notice is also written with
        # `day_chat_id = NULL`, so the predicate above already misses it.
        #
        # Both are kept because they fail differently. The NULL-day trick is
        # structural — nothing can accidentally re-include the row — but it
        # cannot survive agent-initiated contact, which needs a REAL day so
        # the agent's own proactive message appears in its context. The
        # moment that ships, the structural guard stops covering this case
        # and only this predicate does. Writing it now means the boundary is
        # already correct when that happens, rather than being the thing
        # someone has to remember.
        #
        # `IS NULL` is part of the test on purpose: every pre-existing row
        # predates the column, and a bare `!= 'admin'` is FALSE for NULL in
        # SQL — which would silently drop the entire history of every user.
        .where(
            or_(
                Message.origin.is_(None),
                Message.origin.notin_(list(ORIGINS_EXCLUDED_FROM_CONTEXT)),
            )
        )
        # CONTRACTS-R31 §4.1: an automation's conversation is its OWN
        # thread and must never be read back into the main chat. The R28
        # session path wrote those turns as ordinary Messages carrying a
        # real `day_chat_id` (session.py's docstring called that "part of
        # the agent's day context by design"), so this predicate is the
        # only thing standing between the thread and the day. Without it
        # the founder's 26 August thread answer was re-asked and
        # re-answered in the main chat (F1, "and back").
        #
        # Keyed on the CHANNEL, not on a title or a metadata key: the
        # channel is the producer, written at row creation, and it is
        # what the clean-up migration keys on too.
        .where(
            or_(
                Conversation.channel.is_(None),
                Conversation.channel.notin_(HIDDEN_DAY_CHANNELS),
            )
        )
        # W2.4(a): id tiebreaker — equal-microsecond inserts otherwise leave
        # the order DB-dependent between loads, and a retroactive reorder of
        # already-serialized history rewrites the cached prompt prefix.
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    rows = result.all()

    # Reply-to: bulk-fetch every distinct target referenced by the day's
    # rows so we can prepend a quoted preamble without N+1 queries. A
    # target may live in any session/channel, so we resolve by id alone.
    _reply_target_ids = [
        getattr(msg, "reply_to_message_id", None)
        for (msg, _conv_channel) in rows
        if getattr(msg, "reply_to_message_id", None)
    ]
    _reply_targets: Dict[str, Any] = {}
    if _reply_target_ids:
        try:
            from app.agent.reply_quote import fetch_reply_targets
            _reply_targets = await fetch_reply_targets(db, _reply_target_ids)
        except Exception as _rq_err:
            logger.warning("[day_ctx] reply target fetch failed: %s", _rq_err)
            _reply_targets = {}

    # Build annotated message list
    raw_messages: List[Dict[str, Any]] = []
    annotated_messages: List[Dict[str, Any]] = []

    from app.agent.channel_util import resolve_channel
    for msg, conv_channel in rows:
        if msg.role not in ("user", "assistant"):
            continue

        # If this row replies to another, prepend a short quoted preamble
        # so the LLM sees the threading link instead of two unrelated
        # turns. UI side reads the structured reply_to_message_id column.
        _content = msg.content
        _rt_id = getattr(msg, "reply_to_message_id", None)
        if _rt_id and _rt_id in _reply_targets:
            try:
                from app.agent.reply_quote import render_reply_preamble
                _target = _reply_targets[_rt_id]
                _preamble = render_reply_preamble(
                    target_role=_target.role,
                    target_content=_target.content,
                    target_created_at=_target.created_at,
                    tz_name=tz_name,
                )
                _content = f"{_preamble}\n\n{_content}"
            except Exception:
                pass

        raw = {"role": msg.role, "content": _content}
        raw_messages.append(raw)

        # Annotate with channel and time. resolve_channel prefers the
        # per-message channel (Rule 12), falls back to the conversation's
        # channel, logs a WARNING on fallback so silent drift is visible.
        _msg_channel = resolve_channel(
            payload_hint=getattr(msg, "channel", None),
            conversation_hint=conv_channel,
            site="history_annotation",
        )
        annotated_content = annotate_message(_content, _msg_channel, msg.created_at, tz_name=tz_name)
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
            # W2.1b: full verbatim history — the rolling summary would be a
            # pure duplicate, so callers skip <today_so_far> injection.
            "history_is_complete": True,
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
        # W2.1b: older messages were elided to the summary + verbatim window
        # — here the summary genuinely replaces history, so it MUST inject.
        "history_is_complete": False,
    }


def should_inject_today_so_far(day_context: Optional[Dict[str, Any]]) -> bool:
    """Whether the <today_so_far> rolling-summary block should be injected.

    W2.1b: on the under-budget path load_day_context returns the day's
    COMPLETE verbatim history, so the rolling summary is a pure duplicate
    of messages the model already sees — up to ~1,000 UNCACHED tokens per
    turn (<turn_context> sits after history and is never prompt-cached).
    Inject only when the loader actually truncated to the summary +
    verbatim-window shape (history_is_complete=False), where the summary
    genuinely replaces elided messages. A missing bit (older return shape)
    keeps the historical behavior: inject.
    """
    if not day_context or not day_context.get("summary"):
        return False
    return not day_context.get("history_is_complete", False)


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
