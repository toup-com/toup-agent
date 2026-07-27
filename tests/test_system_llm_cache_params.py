"""W1.0 — system-LLM prompt-cache param plumbing.

Every system-LLM caller must be ABLE to opt into OpenAI prompt caching
(`prompt_cache_key` / `prompt_cache_retention` / `safety_identifier`)
without any behavior change for callers that don't. Contract pinned here:

  1. LLMService.complete / complete_with_json / stream forward the params
     to the OpenAI chat.completions call ONLY when set; the Anthropic
     route accepts-and-ignores them (its cache is cache_control-based).
  2. call_system_llm forwards the same params on its OpenAI dispatch
     (bundle SDK path and direct httpx body); the Anthropic dispatch
     accepts-and-ignores them.
  3. First adopter: the Toup-Code supervisor loop passes a session-stable
     prompt_cache_key ("toupcode:<conv_id>") + safety_identifier.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _openai_response(text: str = "ok", in_tok: int = 10, out_tok: int = 5):
    """Fake AsyncOpenAI .chat.completions.create() return value.

    SimpleNamespace (not MagicMock) so absent attributes like
    prompt_tokens_details behave like a real older-SDK response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok, total_tokens=in_tok + out_tok),
        model="gpt-4o-mini",
    )


class _FakeAsyncStream:
    """Minimal async iterator standing in for an OpenAI stream."""

    def __init__(self):
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="hi"),
            finish_reason="stop",
        )])


def _make_openai_llm_service(fake_client):
    """Build an LLMService wired to a fake OpenAI client, bypassing
    __init__ (which needs real keys + tiktoken)."""
    from app.services.llm_service import LLMService

    svc = LLMService.__new__(LLMService)
    svc._use_anthropic = False
    svc._anthropic_svc = None
    svc._openai_client = fake_client
    svc.default_model = "gpt-4o-mini"
    svc.default_temperature = 0.2
    svc.default_max_tokens = 256
    svc._encoding = None
    return svc


def _fake_openai_client(response=None):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response or _openai_response())
    return client


_CACHE_KWARGS = dict(
    prompt_cache_key="test:cache-key",
    prompt_cache_retention="24h",
    safety_identifier="user-abc",
)

_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture
def bundle_settings(monkeypatch):
    """Force bundle mode (same shape as test_internal_llm_dispatch)."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "toup_ct_test", raising=False)
    monkeypatch.setattr(settings, "platform_api_url", "https://test/api", raising=False)
    yield settings


# ── LLMService.complete ──────────────────────────────────────────────


async def test_complete_passes_cache_params_when_set():
    client = _fake_openai_client()
    svc = _make_openai_llm_service(client)

    await svc.complete(_MSGS, model="gpt-4o-mini", **_CACHE_KWARGS)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["prompt_cache_key"] == "test:cache-key"
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["safety_identifier"] == "user-abc"


async def test_complete_omits_cache_params_when_unset():
    """No behavior change for callers that don't opt in — the request
    must not grow new keys."""
    client = _fake_openai_client()
    svc = _make_openai_llm_service(client)

    await svc.complete(_MSGS, model="gpt-4o-mini")

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_retention" not in kwargs
    assert "safety_identifier" not in kwargs


async def test_complete_anthropic_route_accepts_and_ignores():
    """A claude-* model routes to AnthropicService; the OpenAI-only
    params must be swallowed, never forwarded (would TypeError)."""
    from app.services.llm_service import LLMService

    svc = LLMService.__new__(LLMService)
    svc._use_anthropic = True
    svc._openai_client = None
    svc.default_model = "claude-haiku-4-5-20251001"
    svc.default_temperature = 0.2
    svc.default_max_tokens = 256
    svc._encoding = None
    ant = MagicMock()
    ant.create_message = AsyncMock(return_value=SimpleNamespace(
        content="ok", model="claude-haiku-4-5-20251001",
        tokens_input=3, tokens_output=2, tokens_total=5, stop_reason="end_turn",
    ))
    svc._anthropic_svc = ant

    resp = await svc.complete(_MSGS, model="claude-haiku-4-5-20251001", **_CACHE_KWARGS)

    assert resp.content == "ok"
    kwargs = ant.create_message.call_args.kwargs
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_retention" not in kwargs
    assert "safety_identifier" not in kwargs


# ── LLMService.complete_with_json / stream ───────────────────────────


async def test_complete_with_json_passes_cache_params():
    client = _fake_openai_client()
    svc = _make_openai_llm_service(client)

    await svc.complete_with_json(_MSGS, model="gpt-4o-mini", **_CACHE_KWARGS)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["prompt_cache_key"] == "test:cache-key"
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["safety_identifier"] == "user-abc"


async def test_stream_passes_cache_params_when_set():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_FakeAsyncStream())
    svc = _make_openai_llm_service(client)

    chunks = [c async for c in svc.stream(_MSGS, model="gpt-4o-mini", **_CACHE_KWARGS)]

    assert chunks  # generator actually ran
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["prompt_cache_key"] == "test:cache-key"
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["safety_identifier"] == "user-abc"


async def test_stream_omits_cache_params_when_unset():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_FakeAsyncStream())
    svc = _make_openai_llm_service(client)

    _ = [c async for c in svc.stream(_MSGS, model="gpt-4o-mini")]

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_retention" not in kwargs
    assert "safety_identifier" not in kwargs


# ── call_system_llm ──────────────────────────────────────────────────


async def test_call_system_llm_openai_passes_cache_params(bundle_settings):
    fake_client = _fake_openai_client()

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=100,
            system="sys",
            messages=_MSGS,
            model="gpt-4o-mini",
            **_CACHE_KWARGS,
        )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["prompt_cache_key"] == "test:cache-key"
    assert kwargs["prompt_cache_retention"] == "24h"
    assert kwargs["safety_identifier"] == "user-abc"


async def test_call_system_llm_openai_omits_cache_params_when_unset(bundle_settings):
    fake_client = _fake_openai_client()

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=100,
            system="sys",
            messages=_MSGS,
            model="gpt-4o-mini",
        )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_retention" not in kwargs
    assert "safety_identifier" not in kwargs


async def test_call_system_llm_anthropic_accepts_and_ignores(bundle_settings):
    """The Anthropic dispatch must swallow the OpenAI-only params —
    passing them through would TypeError / 400 on messages.create."""
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=SimpleNamespace(
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        content=[SimpleNamespace(text="ok")],
    ))

    with patch("app.services.bundle_client.make_anthropic_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        text = await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=100,
            system="sys",
            messages=_MSGS,
            model="claude-haiku-4-5-20251001",
            **_CACHE_KWARGS,
        )

    assert text == "ok"
    kwargs = fake_client.messages.create.call_args.kwargs
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_retention" not in kwargs
    assert "safety_identifier" not in kwargs


async def test_direct_openai_body_includes_cache_params(monkeypatch):
    """BYOK (non-bundle) path posts raw JSON — the params must land in
    the request body when set."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_mode", "byok", raising=False)
    monkeypatch.setattr(settings, "toup_token", None, raising=False)
    monkeypatch.setattr(settings, "platform_openai_api_key", "test-key-not-real", raising=False)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "choices": [{"message": {"content": "ok"}}],
    }
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=fake_resp)
    fake_http.__aenter__.return_value = fake_http

    with patch("app.services.internal_llm.httpx.AsyncClient", return_value=fake_http):
        from app.services.internal_llm import call_system_llm
        text = await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=100,
            system="sys",
            messages=_MSGS,
            model="gpt-4o-mini",
            **_CACHE_KWARGS,
        )

    assert text == "ok"
    body = fake_http.post.call_args.kwargs["json"]
    assert body["prompt_cache_key"] == "test:cache-key"
    assert body["prompt_cache_retention"] == "24h"
    assert body["safety_identifier"] == "user-abc"


async def test_direct_openai_body_unchanged_when_unset(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "llm_mode", "byok", raising=False)
    monkeypatch.setattr(settings, "toup_token", None, raising=False)
    monkeypatch.setattr(settings, "platform_openai_api_key", "test-key-not-real", raising=False)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "choices": [{"message": {"content": "ok"}}],
    }
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=fake_resp)
    fake_http.__aenter__.return_value = fake_http

    with patch("app.services.internal_llm.httpx.AsyncClient", return_value=fake_http):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=100,
            system="sys",
            messages=_MSGS,
            model="gpt-4o-mini",
        )

    body = fake_http.post.call_args.kwargs["json"]
    assert "prompt_cache_key" not in body
    assert "prompt_cache_retention" not in body
    assert "safety_identifier" not in body


# ── Telemetry parity ─────────────────────────────────────────────────


async def test_system_llm_perf_log_line(bundle_settings, caplog):
    """W1.0(d): a usage-bearing response must emit the cross-provider
    [PERF] system_llm line so cache dashboards see system calls."""
    import logging

    caplog.set_level(logging.INFO, logger="app.services.internal_llm")
    fake_client = _fake_openai_client(_openai_response(in_tok=42, out_tok=7))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=100,
            system="sys",
            messages=_MSGS,
            model="gpt-4o-mini",
        )

    perf = [r.getMessage() for r in caplog.records if "[PERF] system_llm" in r.getMessage()]
    assert perf, "expected a [PERF] system_llm log line"
    assert "cache_read=0" in perf[0]
    assert "input=42" in perf[0]


# ── First adopter: Toup-Code supervisor ──────────────────────────────


async def test_toup_code_supervisor_passes_session_cache_key():
    """The supervisor loop's growing session.messages conversation must
    self-cache: session-stable prompt_cache_key + per-user
    safety_identifier on every call_system_llm."""
    from app.api.toup_code import SupervisorSession, _run_supervisor_loop

    session = SupervisorSession(
        conv_id="conv-123",
        user_id="user-1",
        provider="claude",
        workspace=Path("/tmp"),
        workspace_root=Path("/tmp"),
        token="test-token",
        messages=[{"role": "user", "content": "build me a thing"}],
    )
    queue: asyncio.Queue = asyncio.Queue()
    done = '{"action": "done", "summary": "all set"}'

    with patch(
        "app.services.internal_llm.call_system_llm",
        new=AsyncMock(return_value=done),
    ) as mock_call:
        await _run_supervisor_loop(session, queue)

    mock_call.assert_called_once()
    kwargs = mock_call.call_args.kwargs
    assert kwargs["prompt_cache_key"] == "toupcode:conv-123"
    assert kwargs["safety_identifier"] == "user-1"
