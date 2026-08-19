"""Responses-API wire unit (openai_wire_api flag) — G1 blocker.

gpt-5.6-* rejects /v1/chat/completions when function tools are present
("Function tools with reasoning_effort are not supported … use
/v1/responses" — canary abort 2026-07-28). This suite pins the flag-gated
second wire path end to end:

  A. Flag default 'chat' + the DEFAULT path is byte-identical to today:
     full-dict equality on the chat.completions.create kwargs (extra or
     missing keys fail), and the responses client surface is never touched.
  B. Translation units: flattened Responses function tools (strict:False),
     chat→Responses tool_choice shapes, the Responses input-item builder
     (incl. reasoning-item echo + dedupe).
  C. The SSE→StreamEvent adapter: EXACT yielded sequence (interleaved text
     + tool_use_start, buffered args, tool_use_end after stream end, one
     trailing message_end), stop-reason inference, retry/metering/[PERF]
     parity with the chat path.
  D. Proxy: /openai/v1/responses route (401/422/400 — never 405), the
     Responses-shape usage extractor, and metering parity (llm_proxy_events
     row endpoint="responses" + credit ledger keyed on the event id).

The flag flip is a per-tenant bound-.env action (canary 533354ce first) —
these tests only prove the OFF path is unchanged and the ON path honors
the same consumer contract.
"""

from __future__ import annotations

import hashlib
import asyncio
import inspect
import json
import logging
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from app.config import settings
from app.services import openai_agent_service as oas
from app.services.openai_agent_service import (
    OpenAIAgentService,
    _anthropic_tools_to_responses,
    _chat_tool_choice_to_responses,
)
from app.api import llm_proxy as lp


pytestmark = pytest.mark.asyncio


# ── Shared fixtures / helpers ────────────────────────────────────────


class _AsyncStream:
    """Duck-typed stand-in for the SDK's AsyncStream: async-iterates a
    canned event/chunk list (same idiom as test_subagent_context_isolation's
    _FakeLLM)."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item
        return gen()


def _ev(type_: str, **kw):
    return NS(type=type_, **kw)


_TOOLS_ANTHROPIC = [
    {
        "name": "web_search",
        "description": "Search the web",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def _make_service(fake_client) -> OpenAIAgentService:
    svc = OpenAIAgentService()
    # Inject post-__init__: _ensure_client no-ops while keys.version is
    # unchanged (test_internal_llm_dispatch precedent).
    svc.client = fake_client
    return svc


def _chat_chunks():
    """Minimal chat stream: one text chunk + one usage/finish chunk."""
    return [
        NS(
            id="chatcmpl-1",
            usage=None,
            choices=[NS(delta=NS(content="hi", tool_calls=None), finish_reason=None)],
        ),
        NS(
            id="chatcmpl-1",
            usage=NS(
                prompt_tokens=10,
                completion_tokens=2,
                prompt_tokens_details=NS(cached_tokens=4),
            ),
            choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason="stop")],
        ),
    ]


def _chat_client(chunks=None):
    """Fake client exposing ONLY the chat surface — an attribute error on
    .responses proves the default path never touches the new wire."""
    return NS(
        chat=NS(completions=NS(create=AsyncMock(return_value=_AsyncStream(chunks or _chat_chunks())))),
    )


def _responses_client(events):
    """Fake client exposing ONLY the responses surface — an attribute
    error on .chat proves the flag-on path never touches the chat wire."""
    return NS(responses=NS(create=AsyncMock(return_value=_AsyncStream(events))))


async def _drive(svc: OpenAIAgentService, **kwargs):
    events = []
    async for ev in svc.create_message_stream(**kwargs):
        events.append(ev)
    # Round 4: the usage report is scheduled as a background task right
    # before message_end (no longer awaited inline) — yield once so the spy
    # sees it, exactly as the runner's next await would in production.
    await asyncio.sleep(0)
    return events


# ──────────────────────────────────────────────────────────────────────
# A. Flag default + byte-identical default path (the regression pin)
# ──────────────────────────────────────────────────────────────────────


def test_flag_defaults_to_chat():
    """The flag must ship dark: default 'chat', zero behavior change on
    deploy. Flipping it is a per-tenant bound-.env action."""
    from app.config import Settings
    assert Settings.model_fields["openai_wire_api"].default == "chat"


async def test_flag_off_chat_kwargs_full_dict_equality(monkeypatch):
    """FULL-dict equality on the built chat kwargs — an extra, missing, or
    reordered-value key is a regression. This is the byte-identity pin for
    the default wire."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    fake = _chat_client()
    svc = _make_service(fake)

    await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        tools=_TOOLS_ANTHROPIC,
        model="gpt-5.5",
        max_tokens=777,
        tool_choice="auto",
        prompt_cache_key="u1:day-1",
        safety_identifier="u1",
        idempotency_key="u1:sess-1",
        stable_prefix_active=True,
    )

    fake.chat.completions.create.assert_called_once()
    assert fake.chat.completions.create.call_args.kwargs == {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "max_completion_tokens": 777,
        "stream": True,
        "stream_options": {"include_usage": True},
        "prompt_cache_key": "u1:day-1",
        "prompt_cache_retention": "24h",
        "safety_identifier": "u1",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": _TOOLS_ANTHROPIC[0]["input_schema"],
                },
            },
        ],
        "tool_choice": "auto",
    }


async def test_flag_off_minimal_no_tools_no_cache(monkeypatch):
    """No tools → no tools/tool_choice keys; no cache args → no cache keys;
    gpt-5.5 → no temperature key."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    fake = _chat_client()
    svc = _make_service(fake)

    await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.5",
        max_tokens=100,
    )

    assert fake.chat.completions.create.call_args.kwargs == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 100,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def test_flag_off_gpt4o_gets_temperature(monkeypatch):
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    fake = _chat_client()
    svc = _make_service(fake)

    await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
        max_tokens=100,
        temperature=0.3,
    )

    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3


async def test_flag_off_never_touches_responses_surface(monkeypatch):
    """_chat_client has NO .responses attribute — touching it on the
    default path would AttributeError out of this test."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    fake = _chat_client()
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert not hasattr(fake, "responses")
    assert events[-1].type == "message_end"


def test_source_pin_branch_before_chat_kwargs_build():
    """The wire branch must sit ABOVE the chat kwargs build (so flag-off
    reaches the untouched build) and the chat create call must survive
    verbatim (the 4 existing suites pin more of this source)."""
    src = inspect.getsource(OpenAIAgentService.create_message_stream)
    assert "openai_wire_api" in src
    assert src.index("openai_wire_api") < src.index("kwargs: Dict")
    assert "self.client.chat.completions.create(**kwargs)" in src


# ──────────────────────────────────────────────────────────────────────
# B. Translation units
# ──────────────────────────────────────────────────────────────────────


def test_tools_translate_to_flattened_responses_shape():
    out = _anthropic_tools_to_responses(_TOOLS_ANTHROPIC)
    assert out == [
        {
            "type": "function",
            "name": "web_search",
            "description": "Search the web",
            "parameters": _TOOLS_ANTHROPIC[0]["input_schema"],
            "strict": False,
        },
    ]
    # No chat-style nested wrapper, and strict is EXPLICITLY False —
    # Responses defaults strict=true and our schemas are not strict-clean.
    assert "function" not in out[0]
    assert out[0]["strict"] is False


def test_tools_translate_defaults_missing_schema():
    out = _anthropic_tools_to_responses([{"name": "t"}])
    assert out[0]["parameters"] == {"type": "object", "properties": {}}
    assert out[0]["description"] == ""


@pytest.mark.parametrize("choice", ["auto", "required", "none", None])
def test_tool_choice_strings_pass_through(choice):
    assert _chat_tool_choice_to_responses(choice) == choice


def test_tool_choice_allowed_tools_flattens_agent_runner_shape():
    """The EXACT dict agent_runner builds (prefix_stability.
    build_allowed_tools_choice) must flatten to the Responses shape."""
    from app.agent.prefix_stability import build_allowed_tools_choice
    chat_shape = build_allowed_tools_choice(["web_search", "exec"], mode="auto")
    # sanity: this is the nested chat shape
    assert chat_shape["allowed_tools"]["tools"][0] == {
        "type": "function", "function": {"name": "exec"},
    }
    assert _chat_tool_choice_to_responses(chat_shape) == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [
            {"type": "function", "name": "exec"},
            {"type": "function", "name": "web_search"},
        ],
    }


def test_tool_choice_named_function_flattens():
    assert _chat_tool_choice_to_responses(
        {"type": "function", "function": {"name": "web_search"}}
    ) == {"type": "function", "name": "web_search"}


def test_tool_choice_unknown_and_already_flat_shapes_pass_through():
    unknown = {"type": "custom", "weird": 1}
    assert _chat_tool_choice_to_responses(unknown) == unknown
    already_flat = {
        "type": "allowed_tools", "mode": "auto",
        "tools": [{"type": "function", "name": "x"}],
    }
    assert _chat_tool_choice_to_responses(already_flat) == already_flat


def test_input_builder_plain_and_assistant_tool_use():
    svc = _make_service(_chat_client())
    items = svc._build_responses_input([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me search"},
            {"type": "tool_use", "id": "call_1", "name": "web_search",
             "input": {"query": "x"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "found it"},
        ]},
    ])
    assert items == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Let me search"},
        {"type": "function_call", "call_id": "call_1", "name": "web_search",
         "arguments": json.dumps({"query": "x"})},
        {"type": "function_call_output", "call_id": "call_1", "output": "found it"},
    ]


def test_input_builder_assistant_tool_only_emits_no_empty_message():
    svc = _make_service(_chat_client())
    items = svc._build_responses_input([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_9", "name": "exec", "input": {}},
        ]},
    ])
    assert items == [
        {"type": "function_call", "call_id": "call_9", "name": "exec",
         "arguments": "{}"},
    ]


def test_input_builder_translates_image_parts():
    svc = _make_service(_chat_client())
    items = svc._build_responses_input([
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AA==", "detail": "auto"}},
        ]},
    ])
    assert items == [
        {"role": "user", "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA==",
             "detail": "auto"},
        ]},
    ]


def test_input_builder_translates_anthropic_base64_image_spelling():
    svc = _make_service(_chat_client())
    items = svc._build_responses_input([
        {"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "AA==",
            }},
        ]},
    ])
    assert items == [
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        ]},
    ]


def test_input_builder_echoes_cached_reasoning_before_function_call():
    svc = _make_service(_chat_client())
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [],
                 "encrypted_content": "ENC"}
    svc._responses_reasoning["call_1"] = reasoning
    items = svc._build_responses_input([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_1", "name": "web_search",
             "input": {"query": "x"}},
        ]},
    ])
    assert items[0] == reasoning
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_1"


def test_input_builder_dedupes_shared_reasoning_item():
    """Two calls holding the SAME reasoning item (same item id) must emit
    it once per build."""
    svc = _make_service(_chat_client())
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [],
                 "encrypted_content": "ENC"}
    svc._responses_reasoning["call_1"] = reasoning
    svc._responses_reasoning["call_2"] = reasoning
    items = svc._build_responses_input([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "call_2", "name": "b", "input": {}},
        ]},
    ])
    assert [i for i in items if i.get("type") == "reasoning"] == [reasoning]
    assert [i["call_id"] for i in items if i.get("type") == "function_call"] == [
        "call_1", "call_2",
    ]


def test_input_builder_uncached_call_id_emits_function_call_alone():
    """No cached reasoning for a call_id → the function_call goes out
    without it. NOT reachable via restart or pre-flag history (history
    rehydration is string-only — _load_history/day_context_loader never
    emit tool_use blocks); the only real path is FIFO eviction mid-run
    under extreme call volume, which _REASONING_CACHE_MAX=2048 is sized
    to prevent. Kept as the defensive-degradation pin: emit the bare
    function_call rather than crash the input build."""
    svc = _make_service(_chat_client())
    items = svc._build_responses_input([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_cold", "name": "a", "input": {}},
        ]},
    ])
    assert items == [
        {"type": "function_call", "call_id": "call_cold", "name": "a",
         "arguments": "{}"},
    ]


def test_reasoning_cache_is_bounded():
    svc = _make_service(_chat_client())
    for i in range(oas._REASONING_CACHE_MAX + 10):
        svc._remember_reasoning(f"call_{i}", {"type": "reasoning", "id": f"rs_{i}",
                                              "summary": []})
    assert len(svc._responses_reasoning) == oas._REASONING_CACHE_MAX
    assert "call_0" not in svc._responses_reasoning  # FIFO eviction
    assert f"call_{oas._REASONING_CACHE_MAX + 9}" in svc._responses_reasoning


def test_reasoning_cache_cap_exceeds_worst_case_run_loop():
    """Evicting a call_id whose tool_use block is still in the live run
    loop's messages list resubmits a function_call without its reasoning
    item — reasoning models 400 the whole turn. Pin the cap well above
    agent_max_tool_iterations × a generous parallel batch, with headroom
    for concurrent runs (chat + routines) sharing the process."""
    assert oas._REASONING_CACHE_MAX >= settings.agent_max_tool_iterations * 16


# ──────────────────────────────────────────────────────────────────────
# C. Adapter: Responses SSE → StreamEvent contract (flag ON)
# ──────────────────────────────────────────────────────────────────────


_USAGE = NS(
    input_tokens=1200,
    output_tokens=80,
    input_tokens_details=NS(cached_tokens=1024),
)


def _tool_turn_events():
    return [
        _ev("response.created", response=NS(id="resp_123")),
        _ev("response.in_progress"),  # ignored
        _ev("response.output_text.delta", delta="Hello "),
        _ev("response.output_item.added", output_index=1,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="web_search")),
        _ev("response.function_call_arguments.delta", item_id="fc_1",
            delta='{"query":'),
        _ev("response.output_text.delta", delta="world"),
        _ev("response.function_call_arguments.delta", item_id="fc_1",
            delta=' "x"}'),
        _ev("response.function_call_arguments.done", item_id="fc_1",
            arguments='{"query": "x"}'),
        _ev("response.output_item.done", output_index=1,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="web_search", arguments='{"query": "x"}')),
        _ev("response.content_part.done"),  # ignored
        _ev("response.completed", response=NS(id="resp_123", usage=_USAGE)),
    ]


@pytest.fixture
def responses_flag(monkeypatch):
    monkeypatch.setattr(settings, "openai_wire_api", "responses", raising=False)


async def test_adapter_exact_event_sequence(responses_flag):
    fake = _responses_client(_tool_turn_events())
    svc = _make_service(fake)
    events = await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        tools=_TOOLS_ANTHROPIC,
        model="gpt-5.5",
        max_tokens=777,
        prompt_cache_key="u1:day-1",
        safety_identifier="u1",
        idempotency_key="u1:sess-1",
        stable_prefix_active=True,
    )

    # Exact sequence: interleaved text + tool_use_start; partial args never
    # yielded; tool_use_end only after the wire stream ends; one trailing
    # message_end.
    assert [e.type for e in events] == [
        "text", "tool_use_start", "text", "tool_use_end", "message_end",
    ]
    assert events[0].text == "Hello "
    assert events[2].text == "world"

    start = events[1]
    assert start.tool_name == "web_search"
    assert start.tool_id == "call_1"  # call_id, NEVER the fc_… item id

    end = events[3]
    assert end.tool_name == "web_search"
    assert end.tool_id == "call_1"
    assert end.tool_input == {"query": "x"}

    tail = events[4]
    assert tail.stop_reason == "tool_use"
    assert tail.usage == {
        "input_tokens": 1200,
        "output_tokens": 80,
        "cache_read_input_tokens": 1024,
        "cache_creation_input_tokens": 0,
    }


async def test_adapter_responses_create_kwargs(responses_flag):
    fake = _responses_client(_tool_turn_events())
    svc = _make_service(fake)
    await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        tools=_TOOLS_ANTHROPIC,
        model="gpt-5.5",
        max_tokens=777,
        tool_choice="auto",
        prompt_cache_key="u1:day-1",
        safety_identifier="u1",
        idempotency_key="u1:sess-1",
        stable_prefix_active=True,
    )

    fake.responses.create.assert_called_once()
    kwargs = fake.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["store"] is False  # stateless mode — no server-side state
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["max_output_tokens"] == 777
    assert kwargs["stream"] is True
    assert kwargs["instructions"] == "sys"
    assert kwargs["input"] == [{"role": "user", "content": "hi"}]
    assert kwargs["prompt_cache_key"] == "u1:day-1"
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["safety_identifier"] == "u1"
    assert kwargs["tools"] == _anthropic_tools_to_responses(_TOOLS_ANTHROPIC)
    assert kwargs["tool_choice"] == "auto"
    # Never sent on this wire:
    assert "stream_options" not in kwargs   # usage rides response.completed
    assert "reasoning" not in kwargs        # parity: server default effort
    assert "temperature" not in kwargs      # gpt-5.x gate, same as chat
    assert "previous_response_id" not in kwargs
    assert "conversation" not in kwargs
    assert "truncation" not in kwargs
    assert "parallel_tool_calls" not in kwargs
    # The chat surface must never be touched on this path.
    assert not hasattr(fake, "chat")


async def test_adapter_text_only_end_turn(responses_flag):
    fake = _responses_client([
        _ev("response.created", response=NS(id="resp_1")),
        _ev("response.output_text.delta", delta="just text"),
        _ev("response.completed", response=NS(id="resp_1", usage=_USAGE)),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert [e.type for e in events] == ["text", "message_end"]
    assert events[-1].stop_reason == "end_turn"


async def test_adapter_incomplete_maps_to_max_tokens(responses_flag):
    fake = _responses_client([
        _ev("response.created", response=NS(id="resp_1")),
        _ev("response.output_text.delta", delta="cut of"),
        _ev("response.incomplete", response=NS(
            id="resp_1", usage=_USAGE,
            incomplete_details=NS(reason="max_output_tokens"),
        )),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert events[-1].stop_reason == "max_tokens"
    assert events[-1].usage["input_tokens"] == 1200


async def test_adapter_incomplete_content_filter_maps_to_end_turn(responses_flag):
    fake = _responses_client([
        _ev("response.incomplete", response=NS(
            id="resp_1", usage=_USAGE,
            incomplete_details=NS(reason="content_filter"),
        )),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert events[-1].stop_reason == "end_turn"


async def test_adapter_dead_stream_maps_to_empty_stop_reason(responses_flag):
    """Stream ends with no terminal event → '' (chat parity: unlatched
    finish_reason)."""
    fake = _responses_client([
        _ev("response.created", response=NS(id="resp_1")),
        _ev("response.output_text.delta", delta="then it died"),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert events[-1].type == "message_end"
    assert events[-1].stop_reason == ""
    assert events[-1].usage == {}


async def test_adapter_truncated_fn_call_incomplete_wins_over_tool_use(responses_flag):
    """max_output_tokens hit mid-function-call: the terminal status must
    outrank the buffered fn_call — stop_reason='max_tokens', NEVER
    'tool_use', so agent_runner's exec gate (stop_reason == 'tool_use')
    stays closed and the truncated call is dropped, not executed with
    garbage args. Chat parity: finish_reason='length'→'max_tokens' latches
    even when tool_calls were partially streamed."""
    fake = _responses_client([
        _ev("response.created", response=NS(id="resp_1")),
        _ev("response.output_item.added", output_index=0,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="exec")),
        _ev("response.function_call_arguments.delta", item_id="fc_1",
            delta='{"cmd": "rm -r'),
        _ev("response.incomplete", response=NS(
            id="resp_1", usage=_USAGE,
            incomplete_details=NS(reason="max_output_tokens"),
        )),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert events[-1].type == "message_end"
    assert events[-1].stop_reason == "max_tokens"
    # The truncated call still surfaces as tool_use_end with raw args
    # (chat parity: ends are emitted for every buffered call), but the
    # stop_reason keeps it from ever executing.
    end = [e for e in events if e.type == "tool_use_end"][0]
    assert end.tool_input == {"raw": '{"cmd": "rm -r'}


async def test_adapter_dead_stream_with_buffered_fn_call_keeps_empty_stop(responses_flag):
    """Stream dies (no terminal event) after buffering a function_call:
    stop_reason='' (chat parity: unlatched finish_reason) — the exec gate
    stays closed, the buffered call is never executed."""
    fake = _responses_client([
        _ev("response.created", response=NS(id="resp_1")),
        _ev("response.output_item.added", output_index=0,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="exec")),
        _ev("response.function_call_arguments.done", item_id="fc_1",
            arguments='{"cmd": "ls"}'),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert events[-1].type == "message_end"
    assert events[-1].stop_reason == ""


async def test_adapter_incomplete_content_filter_with_fn_call_maps_to_end_turn(responses_flag):
    """incomplete/content_filter with a buffered fn_call: 'end_turn' (chat
    parity: content_filter→end_turn), never 'tool_use'."""
    fake = _responses_client([
        _ev("response.output_item.added", output_index=0,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="exec")),
        _ev("response.function_call_arguments.done", item_id="fc_1",
            arguments='{"cmd": "ls"}'),
        _ev("response.incomplete", response=NS(
            id="resp_1", usage=_USAGE,
            incomplete_details=NS(reason="content_filter"),
        )),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert events[-1].stop_reason == "end_turn"


async def test_adapter_malformed_args_yield_raw(responses_flag):
    fake = _responses_client([
        _ev("response.output_item.added", output_index=0,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="exec")),
        _ev("response.function_call_arguments.delta", item_id="fc_1",
            delta="{oops"),
        _ev("response.completed", response=NS(id="resp_1", usage=_USAGE)),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    end = [e for e in events if e.type == "tool_use_end"][0]
    assert end.tool_input == {"raw": "{oops"}


async def test_adapter_multiple_calls_end_in_output_index_order(responses_flag):
    fake = _responses_client([
        _ev("response.output_item.added", output_index=0,
            item=NS(type="function_call", id="fc_a", call_id="call_a",
                    name="first")),
        _ev("response.function_call_arguments.done", item_id="fc_a",
            arguments="{}"),
        _ev("response.output_item.added", output_index=1,
            item=NS(type="function_call", id="fc_b", call_id="call_b",
                    name="second")),
        _ev("response.function_call_arguments.done", item_id="fc_b",
            arguments="{}"),
        _ev("response.completed", response=NS(id="resp_1", usage=_USAGE)),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    ends = [e for e in events if e.type == "tool_use_end"]
    assert [e.tool_id for e in ends] == ["call_a", "call_b"]
    # Both starts were emitted live, before any end.
    types = [e.type for e in events]
    assert types.index("tool_use_end") > types.index("tool_use_start")
    assert events[-1].stop_reason == "tool_use"


async def test_adapter_unknown_event_types_ignored(responses_flag):
    fake = _responses_client([
        _ev("response.created", response=NS(id="resp_1")),
        _ev("response.output_text.annotation.added", annotation={"x": 1}),
        _ev("response.reasoning_summary_text.delta", delta="thinking…"),
        _ev("some.future.event"),
        _ev("response.output_text.delta", delta="ok"),
        _ev("response.output_text.done", text="ok"),
        _ev("response.completed", response=NS(id="resp_1", usage=_USAGE)),
    ])
    svc = _make_service(fake)
    events = await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert [e.type for e in events] == ["text", "message_end"]


async def test_adapter_failed_event_raises(responses_flag):
    fake = _responses_client([
        _ev("response.created", response=NS(id="resp_1")),
        _ev("response.failed", response=NS(
            id="resp_1",
            error=NS(code="server_error", message="boom"),
        )),
    ])
    svc = _make_service(fake)
    with pytest.raises(RuntimeError, match="server_error.*boom"):
        await _drive(
            svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
            max_tokens=10,
        )


async def test_adapter_error_event_raises(responses_flag):
    """The literal 'error' SSE event (not 'response.error')."""
    fake = _responses_client([
        _ev("error", code="rate_limit_exceeded", message="slow down",
            param=None),
    ])
    svc = _make_service(fake)
    with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
        await _drive(
            svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
            max_tokens=10,
        )


async def test_adapter_caches_reasoning_item_for_following_call(responses_flag):
    fake = _responses_client([
        _ev("response.output_item.done", output_index=0,
            item=NS(type="reasoning", id="rs_1",
                    summary=[{"type": "summary_text", "text": "thought"}],
                    encrypted_content="ENC123")),
        _ev("response.output_item.added", output_index=1,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="web_search")),
        _ev("response.output_item.done", output_index=1,
            item=NS(type="function_call", id="fc_1", call_id="call_1",
                    name="web_search", arguments='{"query": "x"}')),
        _ev("response.completed", response=NS(id="resp_1", usage=_USAGE)),
    ])
    svc = _make_service(fake)
    await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10,
    )
    assert svc._responses_reasoning["call_1"] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "thought"}],
        "encrypted_content": "ENC123",
    }


# ── C-metering: [PERF] + report_llm_usage parity ─────────────────────


@pytest.fixture
def metering_spy(monkeypatch):
    calls: list[dict] = []

    async def fake_report(**kw):
        calls.append(kw)

    monkeypatch.setattr(
        "app.services.credit_reporter.report_llm_usage", fake_report,
    )
    monkeypatch.setattr(settings, "user_id", "user-mx", raising=False)
    return calls


async def test_responses_metering_flag_off_key_is_legacy(
    responses_flag, metering_spy, monkeypatch, caplog,
):
    monkeypatch.setattr(settings, "metering_correctness_v2", False, raising=False)
    caplog.set_level(logging.INFO, logger="app.services.openai_agent_service")
    fake = _responses_client(_tool_turn_events())
    svc = _make_service(fake)
    await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10, prompt_cache_key="u1:day-1", idempotency_key="u1:sess-1",
    )

    assert len(metering_spy) == 1
    report = metering_spy[0]
    assert report["input_tokens"] == 1200
    assert report["output_tokens"] == 80
    assert report["cached_tokens"] == 1024
    assert report["provider"] == "openai"
    # flag OFF: byte-identical legacy key (idempotency_key or prompt_cache_key)
    assert report["idempotency_key"] == "u1:sess-1"

    perf = [r.getMessage() for r in caplog.records if "[PERF]" in r.getMessage()]
    assert len(perf) == 1
    assert "cache_read=1024" in perf[0]
    assert "input=1200" in perf[0]
    assert "output=80" in perf[0]
    assert "provider=openai" in perf[0]


async def test_responses_metering_v2_uses_response_id(
    responses_flag, metering_spy, monkeypatch,
):
    monkeypatch.setattr(settings, "metering_correctness_v2", True, raising=False)
    fake = _responses_client(_tool_turn_events())
    svc = _make_service(fake)
    await _drive(
        svc, messages=[{"role": "user", "content": "hi"}], model="gpt-5.5",
        max_tokens=10, prompt_cache_key="u1:day-1", idempotency_key="u1:sess-1",
    )
    assert metering_spy[0]["idempotency_key"] == "oaireq:resp_123"


def test_source_pin_responses_metering_routes_through_helper():
    """Same pin test_metering_correctness_v2 applies to the chat path: the
    responses path must derive its billing key via _metering_idempotency_key
    and never re-inline the legacy expression."""
    src = inspect.getsource(OpenAIAgentService._create_responses_stream)
    assert "_metering_idempotency_key(" in src
    assert "idempotency_key=idempotency_key or prompt_cache_key" not in src


# ──────────────────────────────────────────────────────────────────────
# D. Proxy: usage extractor + /openai/v1/responses route + metering parity
# ──────────────────────────────────────────────────────────────────────


def _sse(*objs: str) -> bytes:
    return ("".join(f"data: {o}\n\n" for o in objs) + "data: [DONE]\n\n").encode()


_RESPONSES_SSE = _sse(
    '{"type": "response.created", "response": {"id": "resp_1"}}',
    '{"type": "response.output_text.delta", "delta": "hi"}',
    '{"type": "response.completed", "response": {"usage": {'
    '"input_tokens": 1200, "output_tokens": 80,'
    ' "input_tokens_details": {"cached_tokens": 1024,'
    ' "cache_write_tokens": 176}}}}',
)


def test_extract_responses_usage_reads_completed_event():
    assert lp._extract_responses_usage(_RESPONSES_SSE) == (1200, 80, 1024, 176)


def test_extract_responses_usage_incomplete_event_counts():
    raw = _sse(
        '{"type": "response.incomplete", "response": {"usage": {'
        '"input_tokens": 500, "output_tokens": 9}}}',
    )
    assert lp._extract_responses_usage(raw) == (500, 9, 0, 0)


def test_extract_responses_usage_last_usage_wins():
    raw = _sse(
        '{"type": "response.completed", "response": {"usage": {'
        '"input_tokens": 1, "output_tokens": 1}}}',
        '{"type": "response.completed", "response": {"usage": {'
        '"input_tokens": 2, "output_tokens": 3}}}',
    )
    assert lp._extract_responses_usage(raw) == (2, 3, 0, 0)


def test_extract_responses_usage_garbage_is_zeros():
    assert lp._extract_responses_usage(b"data: {not json\n\nnonsense") == (0, 0, 0, 0)
    assert lp._extract_responses_usage(b"") == (0, 0, 0, 0)
    # response present but not a dict — must not raise
    assert lp._extract_responses_usage(
        _sse('{"type": "response.completed", "response": "bogus"}')
    ) == (0, 0, 0, 0)


def test_chat_extractor_meters_zero_on_responses_stream():
    """THE reason the new extractor exists: the chat extractor reads
    prompt_tokens/completion_tokens and silently returns zeros on a
    Responses stream — the exact failure class G2 exists to prevent.
    Conversely the Responses extractor reads zeros off a chat stream."""
    assert lp._extract_openai_usage(_RESPONSES_SSE) == (0, 0, 0)
    chat_sse = _sse(
        '{"usage": {"prompt_tokens": 1200, "completion_tokens": 40,'
        ' "prompt_tokens_details": {"cached_tokens": 1024}}}',
    )
    assert lp._extract_responses_usage(chat_sse) == (0, 0, 0, 0)


def test_cache_write_helper_default_arg_byte_identity():
    """The details_key generalization must keep every existing chat call
    site byte-identical (test_g1_model_gate pins the behavior; this pins
    the signature default)."""
    sig = inspect.signature(lp._extract_openai_cache_write_tokens)
    assert sig.parameters["details_key"].default == "prompt_tokens_details"
    assert lp._extract_openai_cache_write_tokens(
        {"prompt_tokens_details": {"cache_write_tokens": 4096}}
    ) == 4096


@pytest.mark.parametrize("spelling", [
    "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens",
])
def test_cache_write_helper_responses_details_key_all_spellings(spelling):
    assert lp._extract_openai_cache_write_tokens(
        {"input_tokens_details": {spelling: 128}},
        details_key="input_tokens_details",
    ) == 128


def test_extract_responses_cached_tokens_shapes():
    assert lp._extract_responses_cached_tokens(
        {"input_tokens_details": {"cached_tokens": 448}}
    ) == 448
    assert lp._extract_responses_cached_tokens(
        NS(input_tokens_details=NS(cached_tokens=64))
    ) == 64
    for garbage in (
        None, {}, {"input_tokens_details": None}, {"input_tokens_details": {}},
        {"input_tokens_details": {"cached_tokens": None}},
        {"input_tokens_details": {"cached_tokens": "bogus"}},
        {"input_tokens_details": "bogus"},
    ):
        assert lp._extract_responses_cached_tokens(garbage) == 0


# ── Route behavior ───────────────────────────────────────────────────


@pytest_asyncio.fixture
async def responses_agent(test_user_id):
    """AgentConfig with a hashed LLM token + active bundle, plus a credit
    balance so _log_event's try_charge writes a real ledger row."""
    from app.db import AgentConfig, async_session_maker
    from app.services.credit_service import credit_service

    token = f"toup-tok-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=test_user_id,
            llm_token_hash=hashlib.sha256(token.encode()).hexdigest(),
            bundle_status="active",
            # _route_chat needs an outbound key (per-project or platform
            # master); the backend is monkeypatched so it's never used.
            bundle_openai_api_key="sk-test-outbound",
        ))
        await credit_service.get_or_create_balance(db, test_user_id)
        await db.commit()
    return test_user_id, token


async def test_responses_route_unauthenticated_is_401_never_405(client):
    res = await client.post("/api/llm/openai/v1/responses", json={"model": "gpt-5.5"})
    assert res.status_code == 401, res.text
    assert res.status_code not in (404, 405)  # repo rule: route registered


async def test_responses_route_missing_model_is_422(client, responses_agent):
    _uid, token = responses_agent
    res = await client.post(
        "/api/llm/openai/v1/responses",
        json={"input": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 422, res.text


async def test_responses_route_claude_model_is_400(client, responses_agent):
    """No claude default and no Anthropic fallthrough — /responses is
    OpenAI-only."""
    _uid, token = responses_agent
    res = await client.post(
        "/api/llm/openai/v1/responses",
        json={"model": "claude-opus-4-6", "input": []},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400, res.text
    assert "OpenAI-only" in res.text


async def test_responses_stream_passthrough_and_metering(
    client, responses_agent, monkeypatch,
):
    """E2E metering parity: byte-exact SSE passthrough, ONE llm_proxy_events
    row (endpoint='responses', Responses-shape usage, cached_tokens
    persisted, cost from _calc_cost_cents incl. cache columns), and the
    credit ledger row keyed on the event id."""
    from sqlalchemy import select
    from app.db import LLMProxyEvent, async_session_maker
    from app.db.models.credit import CreditLedger

    user_id, token = responses_agent
    seen_bodies: list[dict] = []

    async def fake_stream(self, body, api_key):
        seen_bodies.append(dict(body))
        yield _RESPONSES_SSE

    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)

    res = await client.post(
        "/api/llm/openai/v1/responses",
        json={
            "model": "gpt-5.5",
            "stream": True,
            "input": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "u:day",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200, res.text
    assert res.content == _RESPONSES_SSE  # byte-exact passthrough
    assert seen_bodies and seen_bodies[0]["model"] == "gpt-5.5"

    async with async_session_maker() as db:
        events = (await db.execute(
            select(LLMProxyEvent).where(LLMProxyEvent.user_id == user_id)
        )).scalars().all()
        assert len(events) == 1
        event = events[0]
        assert event.provider == "openai"
        assert event.endpoint == "responses"
        assert event.status == "ok"
        assert event.input_tokens == 1200
        assert event.output_tokens == 80
        assert event.cached_tokens == 1024  # persisted (mig 075 column)
        assert event.operation_type is None  # user-attributable
        assert event.cost_cents == lp._calc_cost_cents(
            "gpt-5.5", 1200, 80, cached_tokens=1024, cache_write_tokens=176,
        )

        # get_or_create_balance writes grant rows too — assert on the ONE
        # deduction row tied to this proxy event.
        from app.db.models import LEDGER_CHAT_MESSAGE
        ledger = (await db.execute(
            select(CreditLedger).where(
                CreditLedger.user_id == user_id,
                CreditLedger.event_type == LEDGER_CHAT_MESSAGE,
            )
        )).scalars().all()
        assert len(ledger) == 1
        # Per-request key: the event id — proxy metering needs no v2 flag.
        assert ledger[0].idempotency_key == event.id
        assert ledger[0].event_id == event.id


async def test_responses_upstream_error_logs_zero_token_error_event(
    client, responses_agent, monkeypatch,
):
    from sqlalchemy import select
    from app.db import LLMProxyEvent, async_session_maker

    user_id, token = responses_agent

    async def fail_stream(self, body, api_key):
        raise lp.UpstreamProviderError(400, b'{"error": {"message": "bad"}}', "openai")
        yield b""  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fail_stream)

    res = await client.post(
        "/api/llm/openai/v1/responses",
        json={"model": "gpt-5.5", "stream": True, "input": []},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400, res.text

    async with async_session_maker() as db:
        events = (await db.execute(
            select(LLMProxyEvent).where(LLMProxyEvent.user_id == user_id)
        )).scalars().all()
        assert len(events) == 1
        assert events[0].status == "error"
        assert events[0].endpoint == "responses"
        assert events[0].input_tokens == 0
        assert events[0].output_tokens == 0


async def test_responses_route_does_not_inject_stream_options(
    client, responses_agent, monkeypatch,
):
    """Responses streams deliver usage in response.completed — the proxy
    must NOT inject the chat-style stream_options include_usage."""
    user_id, token = responses_agent
    seen_bodies: list[dict] = []

    async def fake_stream(self, body, api_key):
        seen_bodies.append(dict(body))
        yield _RESPONSES_SSE

    monkeypatch.setattr(lp.OpenAIBackend, "responses_stream", fake_stream)
    res = await client.post(
        "/api/llm/openai/v1/responses",
        json={"model": "gpt-5.5", "stream": True, "input": []},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert "stream_options" not in seen_bodies[0]


def test_backend_responses_stream_source_pins():
    """The real (unmonkeypatched) backend method must not inject
    stream_options, must hit /v1/responses, and must raise
    UpstreamProviderError before yielding on a non-2xx."""
    src = inspect.getsource(lp.OpenAIBackend.responses_stream)
    assert "/v1/responses" in src
    assert 'body["stream_options"]' not in src
    assert "UpstreamProviderError" in src
    assert "_debug_log_upstream_cache_headers" in src


def test_proxy_responses_uses_fresh_session_and_shield():
    """The stream finally must own a fresh session (post-0.106 the request
    session is closed before the body streams) and shield the write."""
    src = inspect.getsource(lp.proxy_responses)
    assert "async_session_maker" in src
    assert "asyncio.shield" in src
    assert "_extract_responses_usage" in src
    # never the chat extractor — it meters zeros on a Responses stream
    assert "_extract_openai_usage(" not in src


# ── prompt_cache_key 64-char limit (/v1/responses only) ─────────────────
# The Responses endpoint rejects keys over 64 chars (string_above_max_length,
# hit in production 2026-07-29 with a 73-char day-scoped key). Chat
# completions never enforced a limit — its wire stays byte-identical.

def test_responses_cache_key_short_passthrough():
    from app.services.openai_agent_service import _responses_cache_key
    assert _responses_cache_key("u1:day-1") == "u1:day-1"
    exactly_64 = "k" * 64
    assert _responses_cache_key(exactly_64) == exactly_64


def test_responses_cache_key_long_is_64_deterministic_and_distinct():
    from app.services.openai_agent_service import _responses_cache_key
    long_a = "5deca34a:day:2026-07-29:" + "a" * 49  # 73 chars, like the incident
    long_b = "5deca34a:day:2026-07-29:" + "b" * 49  # same 31-char prefix
    ka, kb = _responses_cache_key(long_a), _responses_cache_key(long_b)
    assert len(ka) == 64 and len(kb) == 64
    assert ka == _responses_cache_key(long_a)          # deterministic
    assert ka != kb                                    # full-key hash → no prefix collision
    assert ka.startswith(long_a[:31])                  # readable routing prefix kept


def test_responses_stream_shortens_long_cache_key_chat_source_untouched():
    """The responses kwargs builder must route through _responses_cache_key;
    the chat wire must NOT (its 400-free behavior with long keys is the
    historical baseline the byte-identity suite pins)."""
    import inspect as _inspect
    from app.services import openai_agent_service as svc
    responses_src = _inspect.getsource(svc.OpenAIAgentService._create_responses_stream)
    assert "_responses_cache_key(" in responses_src
    chat_src = _inspect.getsource(svc.OpenAIAgentService.create_message_stream)
    chat_before_branch = chat_src.split("_create_responses_stream", 1)[0]
    assert "_responses_cache_key(" not in chat_before_branch
