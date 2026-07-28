"""
Context Manager — Token counting, context window management, and conversation compaction.

Prevents token overflow by:
1. Estimating token counts for messages
2. Summarizing old messages when context grows too large
3. Keeping system prompt + recent messages + summary of older messages
"""

import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4

# Model context windows (tokens)
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-5.5": 1_050_000,
    # gpt-5.6 family — 1M window (G1 prep; bare "gpt-5.6" included so a
    # tier-less id budgets correctly instead of the 128k default).
    "gpt-5.6-terra": 1_000_000,
    "gpt-5.6-sol": 1_000_000,
    "gpt-5.6-luna": 1_000_000,
    "gpt-5.6": 1_000_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.2": 400_000,
    "gpt-5": 128_000,
    "gpt-4.1": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    # A8-3: the canonical Anthropic default (model_resolver
    # _CANONICAL_ANTHROPIC_MODEL) was missing — the failover path would
    # have budgeted it against DEFAULT_CONTEXT_WINDOW with a warning.
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
}

# Default if model not in table
DEFAULT_CONTEXT_WINDOW = 128_000

# We trigger compaction when context usage exceeds this ratio
COMPACTION_THRESHOLD = 0.75

# Number of recent messages to always keep uncompacted
KEEP_RECENT_MESSAGES = 10

# Number of oldest messages preserved byte-untouched at the front of the
# compacted array when `settings.cache_aware_overflow` is on (finding F-6):
# they are the head of the provider's cached prompt prefix — rewriting
# messages[0] with a fresh summary invalidated the cache for the whole
# conversation on every compaction.
HEAD_KEEP_MESSAGES = 2

# Max persisted compaction summaries kept per conversation
# (Conversation.metadata_json["compaction_summary"], flag-gated).
_MAX_PERSISTED_SUMMARIES = 4

# Max tokens for the compaction summary
SUMMARY_MAX_TOKENS = 500


def get_context_window(model: str) -> int:
    """Get the context window size for a model."""
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


def is_context_overflow_error(err: BaseException) -> bool:
    """Classify an LLM API error as a context-window overflow (A8-2).

    OpenAI raises BadRequestError with code ``context_length_exceeded``
    and a message like "This model's maximum context length is N tokens";
    Anthropic phrases it "prompt is too long". These 400s are
    DETERMINISTIC — retrying the identical payload can never succeed.
    """
    code = getattr(err, "code", None)
    if isinstance(code, str) and code == "context_length_exceeded":
        return True
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error_obj = body.get("error")
        if isinstance(error_obj, dict) and error_obj.get("code") == "context_length_exceeded":
            return True
    msg = str(err).lower()
    return (
        "context_length_exceeded" in msg
        or "maximum context length" in msg
        or "prompt is too long" in msg
    )


def _content_has_block(msg: Dict[str, Any], block_type: str) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == block_type for b in content)


def is_plain_text_message(msg: Dict[str, Any]) -> bool:
    """A user/assistant message with no tool_use/tool_result blocks —
    always a safe place for the kept window to start."""
    if msg.get("role") not in ("user", "assistant"):
        return False
    content = msg.get("content")
    if isinstance(content, list):
        return not any(
            isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            for b in content
        )
    return True


def find_safe_cut(messages: List[Dict[str, Any]], desired_cut: int) -> int:
    """Adjust a compaction cut index so it never lands between an
    assistant tool_use message and its following tool_result message.

    A8-4: a kept window starting with an orphaned tool_result converts to
    a ``role:"tool"`` message whose parent ``tool_calls`` were summarized
    away → OpenAI rejects the request with a 400 (the API contract is
    documented in tool_elision.py). Walk FORWARD from ``desired_cut`` to
    the next plain user/assistant text message; fall back to any start
    that is not a tool_result continuation (an assistant tool_use start
    keeps its pair intact on the kept side), then walk backward as a last
    resort. Returns an index in [0, len(messages)].
    """
    n = len(messages)
    if desired_cut <= 0:
        return 0
    if desired_cut >= n:
        return n
    for i in range(desired_cut, n):
        if is_plain_text_message(messages[i]):
            return i
    for i in range(desired_cut, n):
        if not _content_has_block(messages[i], "tool_result"):
            return i
    for i in range(desired_cut - 1, -1, -1):
        if not _content_has_block(messages[i], "tool_result"):
            return i
    return 0


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single message (any format)."""
    content = msg.get("content", "")

    if isinstance(content, str):
        return estimate_tokens(content) + 4  # role overhead

    # List content (tool_use blocks, tool_result blocks, etc.)
    if isinstance(content, list):
        total = 4  # role overhead
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block.get("type") in ("tool_use", "tool_result"):
                    total += estimate_tokens(json.dumps(block, default=str))
                elif block.get("type") == "image_url":
                    total += 300  # rough estimate for image token cost
                else:
                    total += estimate_tokens(json.dumps(block, default=str))
            elif isinstance(block, str):
                total += estimate_tokens(block)
        return total

    return estimate_tokens(str(content)) + 4


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def needs_compaction(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    model: str,
) -> bool:
    """Check if the conversation context needs compaction."""
    context_window = get_context_window(model)
    system_tokens = estimate_tokens(system_prompt)
    messages_tokens = estimate_messages_tokens(messages)
    total = system_tokens + messages_tokens
    ratio = total / context_window

    if ratio > COMPACTION_THRESHOLD:
        logger.info(
            f"[CONTEXT] Compaction needed: {total} tokens / {context_window} window = {ratio:.1%}"
        )
        return True
    return False


async def compact_messages(
    messages: List[Dict[str, Any]],
    model: str,
    *,
    conversation_id: Optional[str] = None,
    on_drop: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Compact conversation by summarizing older messages.

    Keeps the most recent KEEP_RECENT_MESSAGES and summarizes the rest
    into a single system-like message. The cut never splits an assistant
    tool_use from its following tool_result (A8-4).

    With ``settings.cache_aware_overflow`` on (finding F-6), the first
    HEAD_KEEP_MESSAGES stay byte-untouched at the front (they head the
    provider's cached prompt prefix) and only the MIDDLE is summarized:
    output is [head..., summary, recent...]. The summary is generated
    deterministically (temperature 0.0) and persisted per
    (conversation, span-hash) on Conversation.metadata_json so
    re-compacting the same span reuses it instead of paying for a new
    LLM call; ``on_drop`` is invoked with the summarized span so the
    caller can promote it to durable memory before it leaves the
    context window (A8-6).
    """
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages

    cache_aware = bool(getattr(settings, "cache_aware_overflow", False))

    cut = find_safe_cut(messages, len(messages) - KEEP_RECENT_MESSAGES)
    head_end = 0
    if cache_aware:
        # The head boundary must be tool-pair-safe too: a head ending in
        # a dangling tool_use is just as invalid as a window starting
        # with an orphaned tool_result.
        head_end = min(find_safe_cut(messages, HEAD_KEEP_MESSAGES), cut)
    if cut <= head_end:
        # The tool-pair adjustment ate the whole compactable span —
        # nothing can be dropped without orphaning a pair.
        return messages

    head = messages[:head_end]
    old_messages = messages[head_end:cut]
    recent_messages = messages[cut:]

    logger.info(
        f"[CONTEXT] Compacting: summarizing {len(old_messages)} old messages, keeping {len(recent_messages)} recent"
    )

    # Build a text summary of old messages
    summary_text = _build_span_transcript(old_messages, cache_aware=cache_aware)

    if cache_aware:
        # F-6: deterministic + idempotent. Reuse the persisted summary
        # when the exact same span is compacted again; otherwise generate
        # at temperature 0.0 and persist it keyed by the span hash.
        span_key = _span_hash(old_messages)
        cached_summary = (
            await _load_persisted_summary(conversation_id, span_key)
            if conversation_id else None
        )
        if cached_summary is not None:
            summary_text = cached_summary
        else:
            if estimate_tokens(summary_text) > SUMMARY_MAX_TOKENS * 2:
                summary_text = await _llm_summarize(
                    summary_text, model, temperature=0.0, structured=True
                )
            if conversation_id:
                await _persist_summary(conversation_id, span_key, summary_text)
        if on_drop is not None and cached_summary is None:
            # A8-6: promote the span to durable memory BEFORE it leaves
            # the context window. The callback must schedule its own
            # background task — it is invoked synchronously here.
            #
            # Review pr3-#1: the span drifts by a couple of messages every
            # turn once a conversation lives past the threshold, so naive
            # per-compaction promotion would re-extract ~the same span
            # (an LLM call) on EVERY turn — the exact hot-path burn the
            # no-LLM-on-hot-path rule exists for. A persisted
            # promoted-through cursor hands the callback only messages it
            # has never seen; each message is promoted at most once.
            unpromoted = old_messages
            if conversation_id:
                unpromoted = await _filter_unpromoted(conversation_id, old_messages)
            if unpromoted:
                try:
                    on_drop(list(unpromoted))
                except Exception:
                    logger.warning("[CONTEXT] on_drop promotion callback failed", exc_info=True)
    # Use LLM to generate a concise summary if the raw summary is large
    elif estimate_tokens(summary_text) > SUMMARY_MAX_TOKENS * 2:
        summary_text = await _llm_summarize(summary_text, model)

    # Prepend summary as a system message
    summary_msg = {
        "role": "user",
        "content": (
            f"[Conversation summary of {len(old_messages)} earlier messages]\n"
            f"{summary_text}\n"
            f"[End of summary — recent messages follow]"
        ),
    }

    return head + [summary_msg] + recent_messages


# W3.1 (remediation): transcript budget for the structured (cache-aware)
# summarizer input. User/assistant text keeps far more per message than
# tool chatter — the per-message 200-char cap of the legacy builder was
# the recall hole: a long user message carrying five facts got cut at
# 200 chars before the summarizer ever saw it.
_TRANSCRIPT_CHAR_BUDGET = 8000
_STRUCTURED_TEXT_CAP = 500
_TOOL_PREVIEW_CAP = 100


def _build_span_transcript(
    old_messages: List[Dict[str, Any]], *, cache_aware: bool
) -> str:
    """Render the to-be-summarized span as a text transcript.

    ``cache_aware=False`` reproduces the legacy builder byte-for-byte
    (200-char cap on every line) — the flag-off path must not change.

    ``cache_aware=True`` (W3.1 clear-before-compact): user/assistant text
    keeps up to ``_STRUCTURED_TEXT_CAP`` chars; tool_use/tool_result lines
    stay terse. If the whole transcript exceeds ``_TRANSCRIPT_CHAR_BUDGET``,
    tool lines are dropped first (marked), then the OLDEST conversational
    lines — never the newest user/assistant text — so the summarizer's
    fixed input window is spent on recall-critical content instead of
    tool chatter and truncation ellipses.
    """

    def _render(msg: Dict[str, Any], text_cap: int) -> tuple[str, bool]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        has_tool = False
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_bits = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_bits.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        has_tool = True
                        text_bits.append(f"[Tool: {block.get('name', '?')}]")
                    elif block.get("type") == "tool_result":
                        has_tool = True
                        result_preview = str(block.get("content", ""))[:_TOOL_PREVIEW_CAP]
                        text_bits.append(f"[Tool result: {result_preview}]")
            text = " ".join(text_bits)
        else:
            text = str(content)
        if len(text) > text_cap:
            text = text[:text_cap] + "..."
        return f"{role}: {text}", has_tool

    if not cache_aware:
        return "\n".join(_render(m, 200)[0] for m in old_messages)

    lines = [_render(m, _STRUCTURED_TEXT_CAP) for m in old_messages]

    def _total(ls) -> int:
        return sum(len(t) + 1 for t, _ in ls)

    if _total(lines) > _TRANSCRIPT_CHAR_BUDGET:
        # 1st pass: drop tool-bearing lines, oldest first.
        kept = list(lines)
        over = _total(kept) - _TRANSCRIPT_CHAR_BUDGET
        for line in list(kept):
            if over <= 0:
                break
            if line[1]:
                kept.remove(line)
                over -= len(line[0]) + 1
        # 2nd pass: still over → drop oldest conversational lines.
        while len(kept) > 1 and _total(kept) > _TRANSCRIPT_CHAR_BUDGET:
            kept.pop(0)
        dropped = len(lines) - len(kept)
        lines = kept
        if dropped:
            lines = [(f"[{dropped} earlier/tool lines elided before summarization]", False)] + lines

    return "\n".join(t for t, _ in lines)


def _span_hash(span: List[Dict[str, Any]]) -> str:
    """Stable content hash of a summarized message span (persistence key)."""
    payload = json.dumps(span, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


async def _load_persisted_summary(conversation_id: str, span_key: str) -> Optional[str]:
    """Fetch a previously persisted compaction summary for this exact span.

    Stored under Conversation.metadata_json["compaction_summary"] (F-6);
    best-effort — any failure just means the summary is regenerated.
    """
    try:
        from app.db.database import async_session_maker
        from app.db.models.conversation import Conversation

        async with async_session_maker() as db:
            conv = await db.get(Conversation, conversation_id)
            if conv is None or not conv.metadata_json:
                return None
            meta = json.loads(conv.metadata_json)
            summaries = meta.get("compaction_summary") if isinstance(meta, dict) else None
            if isinstance(summaries, dict):
                val = summaries.get(span_key)
                if isinstance(val, str):
                    logger.info(f"[CONTEXT] Reusing persisted compaction summary ({span_key[:8]}…)")
                    return val
    except Exception as e:
        logger.warning(f"[CONTEXT] Compaction summary load failed (non-fatal): {e}")
    return None


async def _persist_summary(conversation_id: str, span_key: str, summary_text: str) -> None:
    """Store a compaction summary on the Conversation row (F-6, idempotent).

    Keyed by span hash so the same span is never re-summarized; bounded
    to _MAX_PERSISTED_SUMMARIES entries (dict order is insertion order
    and survives the JSON round-trip). Best-effort — never raises.

    Concurrency note (review pr3-#2, accepted while flag-gated): this is
    a whole-column read-modify-write of metadata_json with no row lock —
    a concurrent writer of another metadata_json key can be clobbered
    last-write-wins. Acceptable for a best-effort cache under a
    default-off flag; revisit with SELECT ... FOR UPDATE if more writers
    appear on this column.
    """
    try:
        from app.db.database import async_session_maker
        from app.db.models.conversation import Conversation

        async with async_session_maker() as db:
            conv = await db.get(Conversation, conversation_id)
            if conv is None:
                return
            try:
                meta = json.loads(conv.metadata_json) if conv.metadata_json else {}
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            summaries = meta.get("compaction_summary")
            if not isinstance(summaries, dict):
                summaries = {}
            summaries[span_key] = summary_text
            while len(summaries) > _MAX_PERSISTED_SUMMARIES:
                summaries.pop(next(iter(summaries)))
            meta["compaction_summary"] = summaries
            conv.metadata_json = json.dumps(meta, ensure_ascii=False)
            await db.commit()
    except Exception as e:
        logger.warning(f"[CONTEXT] Compaction summary persist failed (non-fatal): {e}")


async def _filter_unpromoted(
    conversation_id: str,
    span: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Suffix of ``span`` that has never been handed to on_drop before
    (review pr3-#1), and advance the persisted cursor past it.

    The cursor is the content hash of the LAST message promoted so far
    (metadata_json["compaction_promoted_through"]). We look for it inside
    the current span: found → only what follows is new; absent (first
    compaction, or the cursor message already scrolled out) → the whole
    span is new. Best-effort: on any failure, return the full span —
    over-promotion degrades to the dedup layer, never loses a fact.
    """
    try:
        from app.db.database import async_session_maker
        from app.db.models.conversation import Conversation

        cursor_key = "compaction_promoted_through"
        async with async_session_maker() as db:
            conv = await db.get(Conversation, conversation_id)
            if conv is None:
                return span
            try:
                meta = json.loads(conv.metadata_json) if conv.metadata_json else {}
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            cursor = meta.get(cursor_key)
            unpromoted = span
            if isinstance(cursor, str):
                for i in range(len(span) - 1, -1, -1):
                    if _span_hash([span[i]]) == cursor:
                        unpromoted = span[i + 1:]
                        break
            if span:
                meta[cursor_key] = _span_hash([span[-1]])
                conv.metadata_json = json.dumps(meta, ensure_ascii=False)
                await db.commit()
            if len(unpromoted) < len(span):
                logger.info(
                    "[CONTEXT] promotion cursor: %d/%d span messages already promoted, handing %d new",
                    len(span) - len(unpromoted), len(span), len(unpromoted),
                )
            return unpromoted
    except Exception as e:
        logger.warning(f"[CONTEXT] promotion cursor load failed (non-fatal): {e}")
        return span


# W3.1 (remediation): structured compaction template for the cache-aware
# path. LongMemEval-class failures come from summaries that narrate the
# conversation instead of preserving retrievable state — the sections
# force the summarizer to keep exactly what a later turn will need to
# recall. Wording is fixed (part of the deterministic persisted-summary
# contract); change it and previously persisted summaries stay valid but
# newly generated ones differ.
_STRUCTURED_SUMMARY_PROMPT = (
    "Compress this conversation transcript into a handoff note for the "
    "same assistant to continue from. Use EXACTLY these four sections, "
    "each a terse dash-list (write 'none' when empty):\n"
    "FACTS: user-stated facts, preferences, names, dates, numbers — keep "
    "exact values verbatim.\n"
    "DECISIONS: what was decided or concluded, with the deciding reason "
    "when stated.\n"
    "OPEN: unresolved questions, promised follow-ups, tasks in progress.\n"
    "ACTIONS: tools/operations performed and their outcomes (one line "
    "each).\n"
    "No preamble, no narration, no information not present in the "
    "transcript."
)

_LEGACY_SUMMARY_PROMPT = (
    "Summarize this conversation transcript into a brief, dense paragraph. "
    "Focus on: key facts discussed, decisions made, user requests, "
    "tool actions taken, and important context. Be concise."
)


async def _llm_summarize(
    text: str, model: str, temperature: float = 0.3, structured: bool = False
) -> str:
    """Use the LLM to generate a concise conversation summary.

    ``temperature=0.0`` (the cache_aware_overflow path) makes the output
    deterministic for identical input so the persisted summary is stable.
    ``structured=True`` (same path, W3.1) swaps the generic paragraph
    prompt for the four-section handoff template; the legacy flag-off
    path keeps the original prompt byte-for-byte.
    """
    try:
        from app.services.bundle_client import make_openai_client
        client = make_openai_client()
        if client is None:
            return ""  # Summarization is best-effort; no key → no summary.

        resp = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheap model for summarization
            max_completion_tokens=SUMMARY_MAX_TOKENS,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        _STRUCTURED_SUMMARY_PROMPT if structured else _LEGACY_SUMMARY_PROMPT
                    ),
                },
                {"role": "user", "content": text[:8000]},  # Cap input
            ],
        )

        summary = resp.choices[0].message.content.strip()
        logger.info(f"[CONTEXT] LLM summary: {len(summary)} chars")
        return summary

    except Exception as e:
        logger.warning(f"[CONTEXT] LLM summarization failed: {e}, using truncation")
        # Fallback: just truncate
        max_chars = SUMMARY_MAX_TOKENS * CHARS_PER_TOKEN
        return text[:max_chars] + "..." if len(text) > max_chars else text
