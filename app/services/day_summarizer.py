"""
Day Summarizer — generates rolling summaries of DayChat message history.

Uses Haiku for cost efficiency. Runs AFTER the agent response is fully
streamed back to the user — never on the request path. Failure sets
summary_status='failed' but the row is now ELIGIBLE for retry per the
FF-B.2 backoff policy below — no longer fail-once-fail-forever.

Debounce: only runs when messages_since_last_summary > 10 OR
tokens_since_last_summary > 5000.

FF-B.2 (2026-05-08) — three changes vs the original implementation:
  M1. Auth-failure fallback. If Anthropic returns 401/403 (invalid
      key), fall through to OpenAI before declaring failure. Previously
      the OpenAI fallback only fired when the Anthropic key was
      empty/None — so a rejected key permanent-failed every day.
  M2. Bounded retry with backoff. summary_status='failed' rows are now
      eligible for retry until summary_failure_count hits MAX_RETRIES.
      Backoff schedule defined in RETRY_BACKOFF_SECONDS.
  M3. Structured failure stub. On any failure we record the classified
      reason (auth_error | rate_limit | timeout | server_error |
      parse_error | no_keys | other) so Day Recall can surface
      "summary unavailable for {date} (reason: ...)" and metrics can
      separate credential issues from genuine LLM failures.

See docs/memory/summarizer-payload-analysis.md for the diagnostic that
drove this rewrite — root cause was a 401 from Anthropic that the old
fallback condition didn't catch.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Debounce thresholds
MIN_MESSAGES_SINCE_SUMMARY = 10
MIN_TOKENS_SINCE_SUMMARY = 5000

# Target output size
TARGET_SUMMARY_TOKENS = 1000
TARGET_ARCHIVAL_TOKENS = 1500

# FF-B.2 — bounded retry policy. After MAX_RETRIES the day stays
# permanently failed (loud — surfaced via summary_last_failure_reason).
# Backoff schedule indexed by failure_count: first retry after 1h,
# second after 4h, third after 24h. Then permanent.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [3600, 14400, 86400]  # 1h, 4h, 24h


def _classify_failure(
    *,
    status_code: Optional[int] = None,
    error_body: Optional[str] = None,
    exception: Optional[BaseException] = None,
) -> str:
    """Map a raw failure into the M3 reason taxonomy.

    Returned strings are stored in day_chats.summary_last_failure_reason
    and emitted as metric labels — keep the set small and stable.
    """
    if exception is not None:
        name = type(exception).__name__.lower()
        if "timeout" in name or "timeout" in str(exception).lower():
            return "timeout"
        return "other"
    if status_code is None:
        return "other"
    if status_code in (401, 403):
        return "auth_error"
    if status_code == 429:
        return "rate_limit"
    if 500 <= status_code < 600:
        return "server_error"
    return "other"

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

    FF-B.2 M2: 'failed' rows are now eligible for retry until
    summary_failure_count >= MAX_RETRIES, gated by RETRY_BACKOFF_SECONDS
    based on summary_last_failure_at. Replaces the previous
    fail-once-fail-forever behaviour.
    """
    from app.db.models.day_chat import DayChat
    from app.db.models import Message

    dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
    if not dc:
        return False

    # M2: failed-but-retry-eligible path
    if dc.summary_status == "failed":
        failure_count = dc.summary_failure_count or 0
        if failure_count >= MAX_RETRIES:
            # Permanently failed — surfaced via summary_last_failure_reason
            return False
        # Was a backoff window started? If so, only retry once it's elapsed.
        # Backoff index = failure_count - 1 (clamped) — after the Nth failure
        # we wait RETRY_BACKOFF_SECONDS[N-1] before attempting retry #N.
        # failure_count=1 → backoff[0]=1h, =2 → backoff[1]=4h, =3 → blocked above.
        last_failure_at = dc.summary_last_failure_at
        if last_failure_at is not None and failure_count >= 1:
            backoff_idx = min(failure_count - 1, len(RETRY_BACKOFF_SECONDS) - 1)
            backoff_seconds = RETRY_BACKOFF_SECONDS[backoff_idx]
            cutoff = datetime.utcnow() - timedelta(seconds=backoff_seconds)
            if last_failure_at > cutoff:
                # Inside backoff window — wait
                return False
        # Eligible for retry. Fall through to the message-count gate so we
        # don't pointlessly retry days that have nothing new (the original
        # debounce still applies).

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

    # If we're retrying a failed day with no new messages since last attempt,
    # still allow one retry pass against ALL messages — the failure may have
    # been auth_error and the fallback path now exists. Otherwise apply the
    # standard debounce.
    if dc.summary_status == "failed":
        # Always count ALL user/assistant messages on a retry — not "since last"
        all_msg_count = (await db.execute(
            select(func.count()).select_from(Message).where(
                and_(
                    Message.day_chat_id == day_chat_id,
                    Message.role.in_(["user", "assistant"]),
                )
            )
        )).scalar() or 0
        return all_msg_count >= MIN_MESSAGES_SINCE_SUMMARY

    return new_msg_count >= MIN_MESSAGES_SINCE_SUMMARY


async def generate_summary(
    db: AsyncSession,
    day_chat_id: str,
    existing_summary: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Generate or update the rolling summary for a DayChat.

    If an existing summary covers earlier messages, this merges the new
    messages into it. Otherwise generates from scratch.

    Returns (summary_text, reason). On success: (text, ""). On failure:
    (None, reason) where reason is one of the M3 taxonomy values.
    Caller persists `reason` to DayChat.summary_last_failure_reason.
    """
    from app.db.models.day_chat import DayChat
    from app.db.models import Message, Conversation

    dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
    if not dc:
        return None, "other"

    # FF-B.2: when retrying a previously-failed day, summarize ALL messages —
    # not just "since last cutoff" — because the prior failure may have been
    # an auth_error that prevented any summary from being written. Without
    # this, retries on failed days would only see new messages and miss the
    # accumulated context.
    is_retry = dc.summary_status == "failed" and (dc.summary_failure_count or 0) > 0
    use_full_history = is_retry or dc.summary_up_to_message_id is None

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

    # If we have a summary already AND this isn't a retry, only summarize
    # new messages. Retries need the full history (see comment above).
    if dc.summary_up_to_message_id and not is_retry:
        cutoff_msg = (await db.execute(
            select(Message.created_at).where(Message.id == dc.summary_up_to_message_id)
        )).scalar_one_or_none()
        if cutoff_msg:
            query = query.where(Message.created_at > cutoff_msg)

    rows = (await db.execute(query)).all()
    if not rows:
        # Nothing to summarize — return whatever existed before, no failure.
        return existing_summary, ""

    # Format messages for the summarizer
    formatted = []
    last_msg_id = None
    for msg, channel in rows:
        time_str = msg.created_at.strftime("%-I:%M%p").lower() if msg.created_at else ""
        prefix = f"[{channel or 'web'} {time_str}]"
        formatted.append(f"{prefix} {msg.role}: {msg.content[:500]}")  # Truncate very long messages
        last_msg_id = msg.id

    messages_text = "\n".join(formatted)

    # Build the LLM prompt. On retry we ignore existing_summary (which may
    # be None anyway since the prior attempt failed) and produce a fresh one.
    if existing_summary and not is_retry:
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

    # Call Haiku (with OpenAI fallback) and surface the (text, reason) tuple
    # to the caller — see _try_summarize for fallback semantics.
    return await _try_summarize(user_prompt)


async def _try_summarize(user_prompt: str) -> Tuple[Optional[str], str]:
    """Try Anthropic → OpenAI fallback. Returns (summary_or_None, reason).

    Reason is one of {"" success, "auth_error", "rate_limit", "timeout",
    "server_error", "parse_error", "no_keys", "other"}. Used by
    run_summarizer_if_needed to record the M3 structured failure stub.

    Auth-failure fallback (M1): Anthropic 401/403 → try OpenAI before
    giving up. Same for any non-2xx if an OpenAI key is configured —
    cheaper to fall back than fail-and-retry-from-zero.
    """
    import httpx
    from app.config import settings

    anthropic_key = settings.anthropic_api_key
    openai_key = settings.openai_api_key

    if not anthropic_key and not openai_key:
        logger.warning("[summarizer] No API key configured for summarization")
        return None, "no_keys"

    last_reason = "other"

    # Try Anthropic first (cheaper for this workload)
    if anthropic_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
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
                try:
                    data = resp.json()
                    summary_text = data.get("content", [{}])[0].get("text", "")
                    if summary_text:
                        return summary_text, ""
                    last_reason = "parse_error"
                    logger.warning(
                        "[summarizer] Anthropic returned 200 with empty content — request_id=%s",
                        resp.headers.get("x-request-id", "?"),
                    )
                except Exception as parse_err:
                    last_reason = "parse_error"
                    logger.warning("[summarizer] Anthropic parse failed: %s", parse_err)
            else:
                last_reason = _classify_failure(
                    status_code=resp.status_code, error_body=resp.text[:200],
                )
                logger.warning(
                    "[summarizer] Anthropic %d (%s) — request_id=%s body=%s",
                    resp.status_code, last_reason,
                    resp.headers.get("x-request-id", "?"), resp.text[:200],
                )
        except Exception as e:
            last_reason = _classify_failure(exception=e)
            logger.warning("[summarizer] Anthropic exception (%s): %s", last_reason, e)

        # M1 fallback: any Anthropic failure with an OpenAI key in scope
        # routes through OpenAI. Previously only triggered on empty key.
        if openai_key:
            logger.info(
                "[summarizer] Falling back to OpenAI (anthropic reason=%s)",
                last_reason,
            )
            text, openai_reason = await _summarize_openai(openai_key, user_prompt)
            if text:
                return text, ""
            # If OpenAI also failed, the reason that surfaces is whichever
            # was more diagnostic — but auth_error trumps generic other.
            return None, last_reason if last_reason != "other" else openai_reason

        return None, last_reason

    # Anthropic key missing but OpenAI present
    text, openai_reason = await _summarize_openai(openai_key, user_prompt)
    if text:
        return text, ""
    return None, openai_reason


async def _summarize_openai(api_key: str, user_prompt: str) -> Tuple[Optional[str], str]:
    """Fallback summarizer using OpenAI API.

    Returns (text_or_None, reason). Reason is "" on success, otherwise
    one of the M3 taxonomy values. Same return shape as _try_summarize so
    the auth-fallback path can compose them.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
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
            try:
                text = resp.json()["choices"][0]["message"]["content"]
                if text:
                    return text, ""
                return None, "parse_error"
            except Exception as parse_err:
                logger.warning("[summarizer] OpenAI parse failed: %s", parse_err)
                return None, "parse_error"
        reason = _classify_failure(status_code=resp.status_code, error_body=resp.text[:200])
        logger.warning(
            "[summarizer] OpenAI %d (%s) body=%s",
            resp.status_code, reason, resp.text[:200],
        )
        return None, reason
    except Exception as e:
        reason = _classify_failure(exception=e)
        logger.warning("[summarizer] OpenAI exception (%s): %s", reason, e)
        return None, reason


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

    FF-B.2: on failure, increments summary_failure_count, records
    summary_last_failure_at + summary_last_failure_reason. Retry-eligibility
    is enforced by should_summarize() — once MAX_RETRIES is hit, the day
    stays permanently failed (and surfaces the reason to Day Recall).
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

            # M1 + M3: tuple return. reason is "" on success, taxonomy str on failure.
            new_summary, reason = await generate_summary(db, day_chat_id, dc.rolling_summary)

            if new_summary:
                # Find the last message ID for the cutoff
                last_msg = (await db.execute(
                    select(Message.id)
                    .where(and_(Message.day_chat_id == day_chat_id, Message.role.in_(["user", "assistant"])))
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )).scalar_one_or_none()

                # Snapshot whether this was a recovery from a prior failure
                # before we clear the bookkeeping — for the log line below.
                prior_failure_count = dc.summary_failure_count or 0

                dc.rolling_summary = new_summary
                dc.summary_up_to_message_id = last_msg
                dc.summary_updated_at = datetime.now(timezone.utc)
                dc.summary_status = "up_to_date"
                # Recovery clears failure bookkeeping — a previously-stuck day
                # that just succeeded should not look "failed" forever after.
                dc.summary_failure_count = 0
                dc.summary_last_failure_at = None
                dc.summary_last_failure_reason = None
                await db.commit()
                if prior_failure_count > 0:
                    logger.info(
                        "[summarizer] day_chat=%s status=up_to_date RECOVERED after %d failed attempts",
                        day_chat_id[:8], prior_failure_count,
                    )
                else:
                    logger.info(
                        "[summarizer] day_chat=%s status=up_to_date",
                        day_chat_id[:8],
                    )
            else:
                # M2 + M3: increment failure_count, record reason + timestamp.
                # The retry-eligibility gate in should_summarize() will refuse
                # to call us again until the backoff window elapses (or
                # MAX_RETRIES is reached, at which point the day is permanently
                # failed and the reason is preserved for Day Recall).
                dc.summary_status = "failed"
                dc.summary_failure_count = (dc.summary_failure_count or 0) + 1
                dc.summary_last_failure_at = datetime.utcnow()
                dc.summary_last_failure_reason = reason or "other"
                await db.commit()
                logger.warning(
                    "[summarizer] day_chat=%s status=failed reason=%s attempt=%d/%d",
                    day_chat_id[:8], dc.summary_last_failure_reason,
                    dc.summary_failure_count, MAX_RETRIES,
                )

    except Exception as e:
        logger.error("[summarizer] Error for day_chat %s: %s", day_chat_id[:8], e)
        try:
            async with session_maker() as db:
                dc = (await db.execute(select(DayChat).where(DayChat.id == day_chat_id))).scalar_one_or_none()
                if dc:
                    dc.summary_status = "failed"
                    dc.summary_failure_count = (dc.summary_failure_count or 0) + 1
                    dc.summary_last_failure_at = datetime.utcnow()
                    dc.summary_last_failure_reason = _classify_failure(exception=e)
                    await db.commit()
        except Exception:
            pass
