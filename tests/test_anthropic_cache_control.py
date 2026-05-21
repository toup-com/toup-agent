"""Regression tests for TKT-LAT-001 — Anthropic prompt caching.

Ensures that:
  * ``_prepare_system`` always wraps a non-empty system in a text block with
    ephemeral ``cache_control`` (for BOTH OAuth and non-OAuth paths).
  * ``_mark_messages_cacheable`` adds ``cache_control`` to the last content
    block of the final message, normalizing string content into a list of
    blocks when needed.

These guard against silent regressions of the caching policy. The actual
cache hit/write counts are observable in production via the ``[PERF]
cache_read=…`` log line emitted by ``create_message`` and the streaming
variant.
"""

from __future__ import annotations

from app.services.anthropic_service import (
    AnthropicService,
    _mark_messages_cacheable,
    _mark_tools_cacheable,
)


def _make_service(*, is_oauth: bool) -> AnthropicService:
    svc = AnthropicService.__new__(AnthropicService)
    svc.is_oauth = is_oauth
    return svc


def test_prepare_system_byok_wraps_with_cache_control():
    svc = _make_service(is_oauth=False)
    out = svc._prepare_system("you are a helpful agent")
    assert isinstance(out, list), "system must be a block list when cached"
    assert len(out) == 1
    block = out[0]
    assert block["type"] == "text"
    assert block["text"] == "you are a helpful agent"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_prepare_system_byok_empty_returns_as_is():
    svc = _make_service(is_oauth=False)
    assert svc._prepare_system("") == ""
    assert svc._prepare_system(None) is None


def test_prepare_system_oauth_has_identity_and_caches_both():
    svc = _make_service(is_oauth=True)
    out = svc._prepare_system("agent system text")
    assert isinstance(out, list)
    assert len(out) == 2
    assert "Claude Code" in out[0]["text"]
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert out[1]["text"] == "agent system text"
    assert out[1]["cache_control"] == {"type": "ephemeral"}


def test_prepare_system_oauth_no_system_only_identity():
    svc = _make_service(is_oauth=True)
    out = svc._prepare_system("")
    assert isinstance(out, list)
    assert len(out) == 1
    assert "Claude Code" in out[0]["text"]


def test_mark_messages_cacheable_string_content_is_wrapped():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what's the weather?"},
    ]
    out = _mark_messages_cacheable(messages)
    # Earlier messages untouched
    assert out[0]["content"] == "hi"
    assert out[1]["content"] == "hello"
    # Last message lifted into a block list with cache_control
    last_content = out[-1]["content"]
    assert isinstance(last_content, list)
    assert last_content[-1]["type"] == "text"
    assert last_content[-1]["text"] == "what's the weather?"
    assert last_content[-1]["cache_control"] == {"type": "ephemeral"}


def test_mark_messages_cacheable_list_content_marks_last_block():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "text", "text": "and answer"},
            ],
        }
    ]
    out = _mark_messages_cacheable(messages)
    blocks = out[-1]["content"]
    # Final block carries cache_control; earlier block does not
    assert "cache_control" not in blocks[0]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert blocks[-1]["text"] == "and answer"


def test_mark_messages_cacheable_empty_input_passthrough():
    assert _mark_messages_cacheable([]) == []


def test_mark_messages_cacheable_empty_string_content_passthrough():
    # No content to cache — leave alone.
    messages = [{"role": "user", "content": ""}]
    out = _mark_messages_cacheable(messages)
    assert out[-1]["content"] == ""


def test_mark_messages_cacheable_does_not_mutate_input():
    original = [{"role": "user", "content": "hi"}]
    snapshot = [dict(m) for m in original]
    _ = _mark_messages_cacheable(original)
    assert original == snapshot


def test_mark_tools_cacheable_marks_last_tool_only():
    tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    out = _mark_tools_cacheable(tools)
    assert "cache_control" not in out[0]
    assert "cache_control" not in out[1]
    assert out[-1]["cache_control"] == {"type": "ephemeral"}


def test_mark_tools_cacheable_empty_passthrough():
    assert _mark_tools_cacheable([]) == []
    assert _mark_tools_cacheable(None) is None
