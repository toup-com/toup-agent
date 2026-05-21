"""Regression tests for TKT-LAT-018 — OpenAI prompt-cache observability.

Anthropic users got prompt caching wired via TKT-LAT-001. Non-Anthropic
users (GPT-5.5 is the platform default for many tenants) ride OpenAI's
*automatic* prompt cache, but the platform was neither observing the
hit rate nor passing ``prompt_cache_key``. Both gaps are closed in
``OpenAIAgentService.create_message_stream``.

These tests are source-level (no live OpenAI call) — they guard
against future refactors that drop the cache_key plumbing or the
[PERF] log line.
"""

from __future__ import annotations

import inspect

from app.services import openai_agent_service as oas
from app.services import anthropic_service as ans


def test_openai_stream_accepts_prompt_cache_key():
    sig = inspect.signature(oas.OpenAIAgentService.create_message_stream)
    assert "prompt_cache_key" in sig.parameters, (
        "TKT-LAT-018: OpenAIAgentService.create_message_stream must accept "
        "prompt_cache_key so agent_runner can pass a stable per-session value."
    )


def test_openai_stream_forwards_cache_key_to_kwargs():
    """If the kwarg is set, the OpenAI API kwargs must carry it."""
    src = inspect.getsource(oas.OpenAIAgentService.create_message_stream)
    assert 'kwargs["prompt_cache_key"] = prompt_cache_key' in src


def test_openai_stream_extracts_cached_tokens():
    """Streaming usage must extract cached_tokens from prompt_tokens_details."""
    src = inspect.getsource(oas.OpenAIAgentService.create_message_stream)
    assert "prompt_tokens_details" in src
    assert "cached_tokens" in src
    assert "cache_read_input_tokens" in src


def test_openai_stream_emits_perf_log_line_in_anthropic_shape():
    """[PERF] cache_read=… line must use the same shape as Anthropic so a
    single grok-friendly query in prod aggregates across providers."""
    src = inspect.getsource(oas.OpenAIAgentService.create_message_stream)
    assert "[PERF] cache_read=" in src
    assert "provider=openai" in src


def test_anthropic_stream_silently_accepts_prompt_cache_key():
    """Cross-provider signature parity: AnthropicService must accept
    prompt_cache_key so agent_runner can always pass it without
    branching on provider before the call."""
    sig = inspect.signature(ans.AnthropicService.create_message_stream)
    assert "prompt_cache_key" in sig.parameters
