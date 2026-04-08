"""
Tests for tool result elision — the highest-risk piece because it can
silently break the Anthropic/OpenAI API contract.

Rules:
  - Every tool_use block MUST have a matching tool_result block (Anthropic)
  - Every tool_call MUST have a matching tool role message (OpenAI)
  - Elision replaces ONLY the content field, never deletes blocks
  - Recent turns (last 10) keep full tool results
  - Older turns get elision placeholder text

Run: python tests/test_tool_result_elision.py
"""

import sys
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_anthropic_conversation(num_turns: int = 30):
    """Build a multi-turn Anthropic-format conversation with tool calls.

    Anthropic format:
      - assistant message with content=[{type: "text", ...}, {type: "tool_use", id: "X", name: "foo", input: {...}}]
      - user message with content=[{type: "tool_result", tool_use_id: "X", content: "big result..."}]
    """
    messages = []
    for i in range(num_turns):
        tool_id = f"toolu_{i:04d}"
        tool_name = f"tool_{i % 5}"  # cycle through 5 tool names
        timestamp = f"2026-04-07T{10 + i // 6:02d}:{(i * 10) % 60:02d}:00Z"

        # User message
        messages.append({
            "role": "user",
            "content": f"Turn {i}: do something with {tool_name}",
        })

        # Assistant calls tool
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"I'll use {tool_name} for you."},
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {"query": f"turn_{i}_query"},
                },
            ],
        })

        # Tool result
        big_result = f"Result for turn {i}. " + ("x" * 500)  # ~500 chars each
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": big_result,
                },
            ],
        })

        # Assistant response after tool
        messages.append({
            "role": "assistant",
            "content": f"Based on the tool result for turn {i}, here's what I found.",
        })

    return messages


def _make_openai_conversation(num_turns: int = 30):
    """Build a multi-turn OpenAI-format conversation with tool calls.

    OpenAI format:
      - assistant message with tool_calls=[{id: "X", type: "function", function: {name: "foo", arguments: "..."}}]
      - tool role message with tool_call_id="X", content="big result..."
    """
    messages = []
    for i in range(num_turns):
        tool_id = f"call_{i:04d}"
        tool_name = f"tool_{i % 5}"

        # User message
        messages.append({
            "role": "user",
            "content": f"Turn {i}: do something with {tool_name}",
        })

        # Assistant calls tool
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": f'{{"query": "turn_{i}_query"}}',
                    },
                }
            ],
        })

        # Tool result
        big_result = f"Result for turn {i}. " + ("x" * 500)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": big_result,
        })

        # Assistant response
        messages.append({
            "role": "assistant",
            "content": f"Based on the tool result for turn {i}, here's what I found.",
        })

    return messages


# ── Tests ─────────────────────────────────────────────────────────────

async def test_anthropic_every_tool_use_has_matching_result():
    """After elision, every tool_use block still has a matching tool_result."""
    from app.agent.tool_elision import elide_tool_results

    messages = _make_anthropic_conversation(30)
    elided = elide_tool_results(messages, keep_recent_turns=10, format="anthropic")

    # Collect all tool_use IDs
    tool_use_ids = set()
    for msg in elided:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_use_ids.add(block["id"])

    # Collect all tool_result IDs
    tool_result_ids = set()
    for msg in elided:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_ids.add(block["tool_use_id"])

    # Every tool_use must have a matching tool_result
    missing = tool_use_ids - tool_result_ids
    assert not missing, f"tool_use IDs without matching tool_result: {missing}"

    # Every tool_result must have a matching tool_use
    orphaned = tool_result_ids - tool_use_ids
    assert not orphaned, f"tool_result IDs without matching tool_use: {orphaned}"


async def test_anthropic_older_results_are_elided():
    """Older tool_result content is replaced with elision string, not deleted."""
    from app.agent.tool_elision import elide_tool_results

    messages = _make_anthropic_conversation(30)
    elided = elide_tool_results(messages, keep_recent_turns=10, format="anthropic")

    # Count elided vs full results
    elided_count = 0
    full_count = 0
    for msg in elided:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if "[tool result elided:" in content:
                        elided_count += 1
                    else:
                        full_count += 1

    # Older turns should be elided, recent 10 should be full
    assert elided_count > 0, "Expected some elided results"
    assert full_count > 0, "Expected some full results (recent turns)"
    assert full_count <= 10, f"Expected at most 10 full results, got {full_count}"


async def test_anthropic_elision_preserves_block_structure():
    """Elided blocks keep type, tool_use_id — only content changes."""
    from app.agent.tool_elision import elide_tool_results

    messages = _make_anthropic_conversation(15)
    elided = elide_tool_results(messages, keep_recent_turns=5, format="anthropic")

    for msg in elided:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    # Must always have these fields
                    assert "tool_use_id" in block, "tool_result missing tool_use_id"
                    assert "type" in block, "tool_result missing type"
                    assert block["type"] == "tool_result", "type changed"
                    # Content must be a string (not None, not deleted)
                    assert isinstance(block.get("content"), str), \
                        f"tool_result content is not a string: {type(block.get('content'))}"


async def test_openai_every_tool_call_has_matching_result():
    """After elision, every tool_call still has a matching tool role message."""
    from app.agent.tool_elision import elide_tool_results

    messages = _make_openai_conversation(30)
    elided = elide_tool_results(messages, keep_recent_turns=10, format="openai")

    # Collect tool_call IDs
    call_ids = set()
    for msg in elided:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                call_ids.add(tc["id"])

    # Collect tool result IDs
    result_ids = set()
    for msg in elided:
        if msg.get("role") == "tool":
            result_ids.add(msg["tool_call_id"])

    missing = call_ids - result_ids
    assert not missing, f"tool_call IDs without matching tool result: {missing}"

    orphaned = result_ids - call_ids
    assert not orphaned, f"tool result IDs without matching tool_call: {orphaned}"


async def test_openai_older_results_are_elided():
    """Older tool role message content is replaced, not deleted."""
    from app.agent.tool_elision import elide_tool_results

    messages = _make_openai_conversation(30)
    elided = elide_tool_results(messages, keep_recent_turns=10, format="openai")

    elided_count = 0
    full_count = 0
    for msg in elided:
        if msg.get("role") == "tool":
            if "[tool result elided:" in msg.get("content", ""):
                elided_count += 1
            else:
                full_count += 1

    assert elided_count > 0, "Expected some elided tool results"
    assert full_count > 0, "Expected some full results (recent turns)"
    assert full_count <= 10, f"Expected at most 10 full results, got {full_count}"


async def test_openai_elision_preserves_message_structure():
    """Tool role messages keep role, tool_call_id — only content changes."""
    from app.agent.tool_elision import elide_tool_results

    messages = _make_openai_conversation(15)
    elided = elide_tool_results(messages, keep_recent_turns=5, format="openai")

    for msg in elided:
        if msg.get("role") == "tool":
            assert "tool_call_id" in msg, "tool message missing tool_call_id"
            assert msg["role"] == "tool", "role changed"
            assert isinstance(msg.get("content"), str), \
                f"tool content is not a string: {type(msg.get('content'))}"


async def test_no_elision_when_under_threshold():
    """If conversation has fewer than keep_recent_turns tool turns, nothing is elided."""
    from app.agent.tool_elision import elide_tool_results

    messages = _make_anthropic_conversation(5)
    elided = elide_tool_results(messages, keep_recent_turns=10, format="anthropic")

    # No elision should have happened
    for msg in elided:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    assert "[tool result elided:" not in block.get("content", ""), \
                        "Unexpected elision in short conversation"


async def test_empty_messages():
    """Empty message list doesn't crash."""
    from app.agent.tool_elision import elide_tool_results

    result = elide_tool_results([], keep_recent_turns=10, format="anthropic")
    assert result == []

    result2 = elide_tool_results([], keep_recent_turns=10, format="openai")
    assert result2 == []


async def test_mixed_text_and_tool_messages():
    """Messages without tool calls pass through unchanged."""
    from app.agent.tool_elision import elide_tool_results

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "what's up?"},
        {"role": "assistant", "content": "not much"},
    ]
    elided = elide_tool_results(messages, keep_recent_turns=10, format="anthropic")
    assert elided == messages  # Unchanged


# ── Standalone runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    async def _run_all():
        tests = [
            ("test_anthropic_every_tool_use_has_matching_result", test_anthropic_every_tool_use_has_matching_result),
            ("test_anthropic_older_results_are_elided", test_anthropic_older_results_are_elided),
            ("test_anthropic_elision_preserves_block_structure", test_anthropic_elision_preserves_block_structure),
            ("test_openai_every_tool_call_has_matching_result", test_openai_every_tool_call_has_matching_result),
            ("test_openai_older_results_are_elided", test_openai_older_results_are_elided),
            ("test_openai_elision_preserves_message_structure", test_openai_elision_preserves_message_structure),
            ("test_no_elision_when_under_threshold", test_no_elision_when_under_threshold),
            ("test_empty_messages", test_empty_messages),
            ("test_mixed_text_and_tool_messages", test_mixed_text_and_tool_messages),
        ]
        passed = 0
        failed = 0
        for name, fn in tests:
            try:
                await fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
        return failed == 0

    success = asyncio.run(_run_all())
    sys.exit(0 if success else 1)
