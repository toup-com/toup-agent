"""
Day Summarizer — generates rolling summaries of DayChat message history.

Uses Haiku for cost efficiency. Runs AFTER the agent response is fully
streamed back to the user — never on the request path. Failure sets
summary_status='failed' and never breaks the user turn.

Debounce: only runs when messages_since_last_summary > 10 OR
tokens_since_last_summary > 5000.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Debounce thresholds
MIN_MESSAGES_SINCE_SUMMARY = 10
MIN_TOKENS_SINCE_SUMMARY = 5000

# Target output size
TARGET_SUMMARY_TOKENS = 1000
TARGET_ARCHIVAL_TOKENS = 1500

SUMMARIZER_SYSTEM_PROMPT = """You are a concise summarizer for a personal AI agent's daily activity log.

Given a list of messages from today's conversations across multiple channels (web, telegram, vibecoding, etc.), produce a structured summary covering:

1. **Ongoing tasks/problems** — what the user is working on, debugging, or trying to accomplish
2. **Decisions made** — choices the user confirmed or the agent recommended
3. **Key facts established** — information the user shared (names, numbers, preferences, configurations)
4. **Tool results worth remembering** — important outputs from tool calls (not every tool call, just significant results)
5. **Open questions** — things left unresolved or that the user said they'd come back to
6. **User's apparent mood/intent** — if notable (frustrated, exploring, in a rush, etc.)

Preserve channel attribution so the agent knows where each thing happened. Use format like:
- [web 9:14am] User asked about X...
- [telegram 12:30pm] User sent a photo of Y...

Keep the summary under 1000 tokens. Be factual and concise — this summary replaces the original messages in the agent's context window, so accuracy matters more than readability."""


# Archival prompt — distinct from the rolling prompt. This produces an INDEX
# ENTRY that will live long-term and be the primary surface when the user or
# agent asks "what happened on {date}" days, weeks, or months later. Different
# goal than the rolling summary: less about keeping the active agent coherent,
# more about making the day recoverable in a few hundred tokens.
ARCHIVAL_SUMMARY_PROMPT = """You are writing a permanent ARCHIVAL SUMMARY of a single day in a user's life with their personal AI agent. This summary will be stored indefinitely and used when the user or agent later asks to recall this day. Future-you will never see the raw messages — only this summary — so make it self-contained and fact-dense.

Produce a structured markdown summary with these sections (omit sections that don't apply, but keep the order):

## Narrative arc
One short paragraph (2–4 sentences) describing the overall shape of the day: what the user was doing, what mood/energy it had, how it started and ended.

## Artifacts produced
Concrete things created or modified: apps built (with names), files written, features shipped, documents drafted, messages sent to others, photos/videos uploaded. Be specific — names, counts, filenames when known.

## Topics taught / learned / discussed
What subjects came up in depth. For tutoring or study sessions, list the actual topics covered (e.g. "derivatives of trigonometric functions", not "math"). For research sessions, list what was investigated. For casual chat, list themes. Be concrete enough that a future quiz or recap could be built from just this section.

## Decisions made
Choices the user committed to or the agent recommended and the user accepted. Include the reasoning when it was stated.

## Problems solved / outcomes reached
Bugs fixed, questions answered, deliverables produced. Note what worked and what didn't.

## Named entities
People, organizations, places, projects, tools, or products mentioned by name. Brief — just a list with a word or two of context each.

## Open threads
Anything left unresolved that the user or agent said they'd return to.

## Channels used
Which surfaces the user was on (web / telegram / voice / app / vibecoding) and roughly when (morning / afternoon / evening).

Rules:
- Be factual. Never invent content not in the messages.
- Prefer specifics over generics: "set up the React Native build for the calculus-tutor app" over "worked on an app".
- Skip conversational filler, greetings, small talk, and repetitive acknowledgements.
- Don't restate what the user already knows about themselves (e.g. "user is Nariman" is not useful). Stick to what was done and discussed today.
- Target under 1500 tokens. Dense paragraphs are fine — this is for retrieval, not readability.
- Do not wrap the output in code blocks or add a preamble. Start directly with the first heading."""


async def should_summarize(db: AsyncSession, day_chat_id: str) -> bool:
    """Check if a DayChat needs summarization based on debounce thresholds.

    Always counts messages from the DB directly — never trusts dc.message_count
    which can be stale (e.g. from backfill or old-path turns).
    """
    from app.db.models.day_chat import DayChat
    from app.db.models import Message

    dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
    if not dc:
        return False

    # If summary is failed, don't auto-retry (per constraint)
    if dc.summary_status == "failed":
        return False

    # Count messages since last summary cutoff — always from DB, never from dc.message_count
    cutoff_filter = Message.day_chat_id == day_chat_id
    if dc.summary_up_to_message_id:
        cutoff_created_at = (await db.execute(
            select(Message.created_at).where(Message.id == dc.summary_up_to_message_id)
        )).scalar_one_or_none()

        if cutoff_created_at:
            cutoff_filter = and_(
                Message.day_chat_id == day_chat_id,
                Message.created_at > cutoff_created_at,
            )

    new_msg_count = (await db.execute(
        select(func.count()).select_from(Message).where(
            and_(
                cutoff_filter,
                Message.role.in_(["user", "assistant"]),
            )
        )
    )).scalar() or 0

    return new_msg_count >= MIN_MESSAGES_SINCE_SUMMARY


async def generate_summary(
    db: AsyncSession,
    day_chat_id: str,
    existing_summary: Optional[str] = None,
) -> Optional[str]:
    """Generate or update the rolling summary for a DayChat.

    If an existing summary covers earlier messages, this merges the new
    messages into it. Otherwise generates from scratch.

    Returns the new summary text, or None if generation fails.
    """
    from app.db.models.day_chat import DayChat
    from app.db.models import Message, Conversation

    dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
    if not dc:
        return None

    # Load messages to summarize
    query = (
        select(Message, Conversation.channel)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            and_(
                Message.day_chat_id == day_chat_id,
                Message.role.in_(["user", "assistant"]),
            )
        )
        .order_by(Message.created_at.asc())
    )

    # If we have a summary already, only summarize new messages
    if dc.summary_up_to_message_id:
        cutoff_msg = (await db.execute(
            select(Message.created_at).where(Message.id == dc.summary_up_to_message_id)
        )).scalar_one_or_none()
        if cutoff_msg:
            query = query.where(Message.created_at > cutoff_msg)

    rows = (await db.execute(query)).all()
    if not rows:
        return existing_summary

    # Format messages for the summarizer
    formatted = []
    last_msg_id = None
    for msg, channel in rows:
        time_str = msg.created_at.strftime("%-I:%M%p").lower() if msg.created_at else ""
        prefix = f"[{channel or 'web'} {time_str}]"
        formatted.append(f"{prefix} {msg.role}: {msg.content[:500]}")  # Truncate very long messages
        last_msg_id = msg.id

    messages_text = "\n".join(formatted)

    # Build the LLM prompt
    if existing_summary:
        user_prompt = (
            f"Here is the existing summary of earlier activity today:\n\n{existing_summary}\n\n"
            f"--- New messages since then ---\n\n{messages_text}\n\n"
            f"Merge the new messages into the existing summary. "
            f"Keep the total under {TARGET_SUMMARY_TOKENS} tokens."
        )
    else:
        user_prompt = (
            f"Summarize today's activity:\n\n{messages_text}\n\n"
            f"Keep the summary under {TARGET_SUMMARY_TOKENS} tokens."
        )

    # Call Haiku for cost efficiency
    try:
        import httpx
        from app.config import settings

        # Use Anthropic API directly for the summarizer
        api_key = settings.anthropic_api_key
        if not api_key:
            # Fallback: try OpenAI
            api_key = settings.openai_api_key
            if api_key:
                return await _summarize_openai(api_key, user_prompt)
            logger.warning("[summarizer] No API key available for summarization")
            return None

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1200,
                    "system": SUMMARIZER_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                summary_text = data.get("content", [{}])[0].get("text", "")
                return summary_text
            else:
                logger.warning("[summarizer] Anthropic API error: %d %s", resp.status_code, resp.text[:200])
                return None

    except Exception as e:
        logger.warning("[summarizer] Failed: %s", e)
        return None


async def _summarize_openai(api_key: str, user_prompt: str) -> Optional[str]:
    """Fallback summarizer using OpenAI API."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 1200,
                    "messages": [
                        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("[summarizer] OpenAI fallback failed: %s", e)
    return None


async def generate_archival_summary(
    db: AsyncSession,
    day_chat_id: str,
) -> Optional[str]:
    """Generate the end-of-day ARCHIVAL summary (distinct from rolling summary).

    This is the long-term index entry returned by recall_day. It uses the
    archival prompt (narrative arc, artifacts, topics, decisions, etc.) rather
    than the rolling prompt (which is tuned for keeping active context coherent).

    Routed through the internal_llm helper with operation_type="system.day_archival"
    so the cost is tracked but does not count toward the user's monthly/daily cap.

    Returns the summary text on success, None on failure. Idempotent at the
    LLM layer — callers are expected to gate on archival_summary_status.
    """
    from app.db.models.day_chat import DayChat
    from app.db.models import Message, Conversation
    from app.services.internal_llm import call_anthropic_system, call_openai_system

    dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
    if not dc:
        return None

    # Load ALL user+assistant messages for the day, chronological, with channel tags.
    rows = (await db.execute(
        select(Message, Conversation.channel)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            and_(
                Message.day_chat_id == day_chat_id,
                Message.role.in_(["user", "assistant"]),
            )
        )
        .order_by(Message.created_at.asc())
    )).all()

    if not rows:
        return None

    # Cap per-message content AND total prompt chars so Haiku's 200k context
    # isn't blown by a 500-message day with long tool transcripts.
    MAX_PER_MESSAGE_CHARS = 2000
    MAX_PROMPT_CHARS = 120_000  # ~30k tokens — Haiku-safe with headroom for output

    formatted: List[str] = []
    running_chars = 0
    truncated_messages = 0
    for msg, channel in rows:
        time_str = msg.created_at.strftime("%-I:%M%p").lower() if msg.created_at else ""
        prefix = f"[{channel or 'web'} {time_str}]"
        content = (msg.content or "")[:MAX_PER_MESSAGE_CHARS]
        line = f"{prefix} {msg.role}: {content}"
        if running_chars + len(line) + 1 > MAX_PROMPT_CHARS:
            truncated_messages = len(rows) - len(formatted)
            break
        formatted.append(line)
        running_chars += len(line) + 1

    if truncated_messages > 0:
        formatted.append(
            f"[... {truncated_messages} later messages omitted from this summary "
            f"input for context-length reasons. Summary should note this.]"
        )
        logger.info(
            "[archival] day_chat=%s dropped %d/%d messages at prompt cap (%d chars)",
            day_chat_id[:8], truncated_messages, len(rows), MAX_PROMPT_CHARS,
        )

    messages_text = "\n".join(formatted)
    user_prompt = (
        f"Produce the archival summary for the following day. "
        f"Local date: {dc.local_date.isoformat()}.\n\n"
        f"Messages (chronological, across all channels):\n\n{messages_text}"
    )

    # Prefer Anthropic Haiku (cheap, strong at structured summarization). Fall
    # back to OpenAI only if no Anthropic key is configured on the platform.
    text = await call_anthropic_system(
        user_id=dc.user_id,
        operation_type="system.day_archival",
        model="claude-haiku-4-5-20251001",
        max_tokens=1800,
        system=ARCHIVAL_SUMMARY_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if text:
        return text

    return await call_openai_system(
        user_id=dc.user_id,
        operation_type="system.day_archival",
        model="gpt-4o-mini",
        max_tokens=1800,
        system=ARCHIVAL_SUMMARY_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )


async def run_summarizer_if_needed(session_maker, day_chat_id: str):
    """Check debounce thresholds and run summarizer if needed.

    Call this as a fire-and-forget background task AFTER the agent response
    is fully streamed. Never blocks the user turn.
    """
    from app.db.models.day_chat import DayChat
    from app.db.models import Message

    try:
        async with session_maker() as db:
            if not await should_summarize(db, day_chat_id):
                return

            dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
            if not dc:
                return

            # Mark as pending
            dc.summary_status = "pending"
            await db.commit()

        # Generate summary in a separate session
        async with session_maker() as db:
            dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
            if not dc:
                return

            new_summary = await generate_summary(db, day_chat_id, dc.rolling_summary)

            if new_summary:
                # Find the last message ID for the cutoff
                last_msg = (await db.execute(
                    select(Message.id)
                    .where(and_(Message.day_chat_id == day_chat_id, Message.role.in_(["user", "assistant"])))
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )).scalar_one_or_none()

                dc.rolling_summary = new_summary
                dc.summary_up_to_message_id = last_msg
                dc.summary_updated_at = datetime.now(timezone.utc)
                dc.summary_status = "up_to_date"
                await db.commit()
                logger.info("[summarizer] Updated summary for day_chat %s", day_chat_id[:8])
            else:
                dc.summary_status = "failed"
                await db.commit()
                logger.warning("[summarizer] Failed to generate summary for day_chat %s", day_chat_id[:8])

    except Exception as e:
        logger.error("[summarizer] Error for day_chat %s: %s", day_chat_id[:8], e)
        try:
            async with session_maker() as db:
                dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
                if dc:
                    dc.summary_status = "failed"
                    await db.commit()
        except Exception:
            pass
