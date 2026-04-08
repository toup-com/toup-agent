"""
Tool result elision — safely compress old tool results in message history.

Anthropic's API requires every tool_use content block in an assistant message
to have a matching tool_result block in the following user message, keyed by
tool_use_id. Deleting or omitting tool_result blocks causes a 400 error.

This module replaces ONLY the content field of older tool_result blocks with
a short elision placeholder. It preserves:
  - Block type ("tool_result" / "tool")
  - tool_use_id / tool_call_id
  - Parent message structure
  - Message ordering

Supports both Anthropic and OpenAI message formats.
"""

import copy
from typing import List, Dict, Any, Optional


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _find_tool_turn_indices_anthropic(messages: List[Dict]) -> List[int]:
    """Find indices of user messages that contain tool_result blocks (Anthropic format)."""
    indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    indices.append(i)
                    break
    return indices


def _find_tool_turn_indices_openai(messages: List[Dict]) -> List[int]:
    """Find indices of tool role messages (OpenAI format)."""
    return [i for i, msg in enumerate(messages) if msg.get("role") == "tool"]


def _build_elision_text(tool_name: Optional[str], channel: Optional[str] = None,
                        timestamp: Optional[str] = None, original_tokens: int = 0) -> str:
    """Build the elision placeholder string."""
    parts = [f"[tool result elided: {tool_name or 'unknown'} ran"]
    if timestamp:
        parts.append(f" at {timestamp}")
    if channel:
        parts.append(f" on {channel}")
    parts.append(f", was ~{original_tokens} tokens. Ask the user or re-run if needed.]")
    return "".join(parts)


def _get_tool_name_for_use_id(messages: List[Dict], tool_use_id: str) -> Optional[str]:
    """Find the tool name for a given tool_use_id by scanning assistant messages (Anthropic)."""
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return block.get("name")
    return None


def _get_tool_name_for_call_id(messages: List[Dict], call_id: str) -> Optional[str]:
    """Find the tool name for a given tool_call_id (OpenAI)."""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id") == call_id:
                    return tc.get("function", {}).get("name")
    return None


def elide_tool_results(
    messages: List[Dict[str, Any]],
    keep_recent_turns: int = 10,
    format: str = "anthropic",
) -> List[Dict[str, Any]]:
    """Elide old tool results while preserving API contract integrity.

    Args:
        messages: The message history (will be deep-copied, not mutated).
        keep_recent_turns: Number of most recent tool turns to keep full results.
        format: "anthropic" or "openai".

    Returns:
        New message list with older tool results replaced by elision placeholders.
    """
    if not messages:
        return []

    # Deep copy to avoid mutating the original
    result = copy.deepcopy(messages)

    if format == "anthropic":
        return _elide_anthropic(result, keep_recent_turns)
    elif format == "openai":
        return _elide_openai(result, keep_recent_turns)
    else:
        return result


def _elide_anthropic(messages: List[Dict], keep_recent: int) -> List[Dict]:
    """Elide tool_result blocks in Anthropic format messages."""
    tool_turn_indices = _find_tool_turn_indices_anthropic(messages)

    if len(tool_turn_indices) <= keep_recent:
        return messages  # Nothing to elide

    # Indices to elide = everything except the last `keep_recent`
    elide_indices = set(tool_turn_indices[:-keep_recent])

    for i in elide_indices:
        msg = messages[i]
        if not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                original_content = block.get("content", "")
                original_tokens = _estimate_tokens(original_content if isinstance(original_content, str) else str(original_content))
                tool_name = _get_tool_name_for_use_id(messages, tool_use_id)
                block["content"] = _build_elision_text(
                    tool_name=tool_name,
                    original_tokens=original_tokens,
                )

    return messages


def _elide_openai(messages: List[Dict], keep_recent: int) -> List[Dict]:
    """Elide tool role messages in OpenAI format."""
    tool_turn_indices = _find_tool_turn_indices_openai(messages)

    if len(tool_turn_indices) <= keep_recent:
        return messages  # Nothing to elide

    elide_indices = set(tool_turn_indices[:-keep_recent])

    for i in elide_indices:
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id", "")
        original_content = msg.get("content", "")
        original_tokens = _estimate_tokens(original_content if isinstance(original_content, str) else str(original_content))
        tool_name = _get_tool_name_for_call_id(messages, call_id)
        msg["content"] = _build_elision_text(
            tool_name=tool_name,
            original_tokens=original_tokens,
        )

    return messages
