"""
Context Manager — Token counting, context window management, and conversation compaction.

Prevents token overflow by:
1. Estimating token counts for messages
2. Summarizing old messages when context grows too large
3. Keeping system prompt + recent messages + summary of older messages
"""

import json
import logging
from typing import Dict, Any, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4

# Model context windows (tokens)
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-5.2": 400_000,
    "gpt-5": 128_000,
    "gpt-4.1": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-5-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
}

# Default if model not in table
DEFAULT_CONTEXT_WINDOW = 128_000

# We trigger compaction when context usage exceeds this ratio
COMPACTION_THRESHOLD = 0.75

# Number of recent messages to always keep uncompacted
KEEP_RECENT_MESSAGES = 10

# Max tokens for the compaction summary
SUMMARY_MAX_TOKENS = 500


def get_context_window(model: str) -> int:
    """Get the context window size for a model."""
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


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
) -> List[Dict[str, Any]]:
    """
    Compact conversation by summarizing older messages.
    
    Keeps the most recent KEEP_RECENT_MESSAGES and summarizes the rest
    into a single system-like message.
    """
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages

    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]

    logger.info(
        f"[CONTEXT] Compacting: summarizing {len(old_messages)} old messages, keeping {len(recent_messages)} recent"
    )

    # Build a text summary of old messages
    summary_parts = []
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_bits = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_bits.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        text_bits.append(f"[Tool: {block.get('name', '?')}]")
                    elif block.get("type") == "tool_result":
                        result_preview = str(block.get("content", ""))[:100]
                        text_bits.append(f"[Tool result: {result_preview}]")
            text = " ".join(text_bits)
        else:
            text = str(content)

        # Truncate individual messages in the summary
        if len(text) > 200:
            text = text[:200] + "..."
        summary_parts.append(f"{role}: {text}")

    summary_text = "\n".join(summary_parts)

    # Use LLM to generate a concise summary if the raw summary is large
    if estimate_tokens(summary_text) > SUMMARY_MAX_TOKENS * 2:
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

    return [summary_msg] + recent_messages


async def _llm_summarize(text: str, model: str) -> str:
    """Use the LLM to generate a concise conversation summary."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        resp = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheap model for summarization
            max_completion_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation transcript into a brief, dense paragraph. "
                        "Focus on: key facts discussed, decisions made, user requests, "
                        "tool actions taken, and important context. Be concise."
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
