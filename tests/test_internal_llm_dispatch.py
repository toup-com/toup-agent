"""Tests for `call_system_llm` — provider + bundle routing.

The contract we pin:

  1. Caller passes no `model` → resolver default (settings.agent_model)
     is used. A GPT-5.5 tenant gets GPT-5.5, a Claude tenant gets Claude.
  2. Caller passes an explicit `model` → that wins (per-routine override).
  3. Model id classification picks the SDK:
       • is_openai_model → OpenAI path
       • is_claude_model OR unknown → Anthropic path (back-compat)
  4. Bundle vs direct is decided by LLM_MODE + TOUP_TOKEN, NOT by the
     model. A GPT-5.5 tenant in bundle mode hits the proxy; the same
     tenant in BYOK mode would hit api.openai.com directly. We mock at
     the bundle-client boundary so neither path actually talks to the
     network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _ant_response(text: str = "ok", in_tok: int = 10, out_tok: int = 5):
    """Build a fake AsyncAnthropic .messages.create() return value."""
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=in_tok, output_tokens=out_tok)
    return resp


def _openai_response(text: str = "ok", in_tok: int = 10, out_tok: int = 5):
    """Build a fake AsyncOpenAI .chat.completions.create() return value."""
    choice = MagicMock()
    choice.message = MagicMock(content=text)
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=in_tok, completion_tokens=out_tok)
    return resp


@pytest.fixture
def bundle_settings(monkeypatch):
    """Force bundle mode for the duration of the test."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_mode", "bundle", raising=False)
    monkeypatch.setattr(settings, "toup_token", "toup_ct_test", raising=False)
    monkeypatch.setattr(settings, "platform_api_url", "https://test/api", raising=False)
    yield settings


@pytest.fixture
def byok_settings(monkeypatch):
    """Force BYOK mode."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_mode", "byok", raising=False)
    monkeypatch.setattr(settings, "toup_token", None, raising=False)
    yield settings


# ── Operation-type guard ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejects_non_system_operation_type():
    """The system-budget exemption depends on the `system.` prefix —
    a caller who passes anything else is either misusing the helper or
    trying to bypass user-cap billing. Reject loudly."""
    from app.services.internal_llm import call_system_llm

    with pytest.raises(ValueError, match="system\\."):
        await call_system_llm(
            user_id="u",
            operation_type="memory.extract",  # missing system. prefix
            max_tokens=10,
            system="x",
            messages=[{"role": "user", "content": "y"}],
        )


# ── Model resolution + provider classification ───────────────────────


@pytest.mark.asyncio
async def test_default_model_resolves_to_settings_agent_model(monkeypatch, bundle_settings):
    """The whole point of the refactor: when the caller omits `model`,
    we hit settings.agent_model — same source the chat uses."""
    from app.config import settings
    monkeypatch.setattr(settings, "agent_model", "gpt-5.5", raising=False)

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_openai_response("hello"))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        text = await call_system_llm(
            user_id="u",
            operation_type="system.routine.email_briefing",
            max_tokens=100,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            # model intentionally omitted
        )

    assert text == "hello"
    # Verify it routed to OpenAI, not Anthropic, because gpt-5.5 is OpenAI.
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.5"


@pytest.mark.asyncio
async def test_caller_provided_model_overrides_resolver(monkeypatch, bundle_settings):
    """Per-routine pin must win. The user said "use Haiku for this one"
    — resolver default is irrelevant."""
    from app.config import settings
    monkeypatch.setattr(settings, "agent_model", "gpt-5.5", raising=False)

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_ant_response("from haiku"))

    with patch("app.services.bundle_client.make_anthropic_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        text = await call_system_llm(
            user_id="u",
            operation_type="system.routine.email_briefing",
            max_tokens=100,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            model="claude-haiku-4-5-20251001",
        )

    assert text == "from haiku"
    fake_client.messages.create.assert_called_once()
    assert fake_client.messages.create.call_args.kwargs["model"] == "claude-haiku-4-5-20251001"


# ── Bundle vs direct dispatch ────────────────────────────────────────


@pytest.mark.asyncio
async def test_bundle_mode_routes_through_make_anthropic_client(bundle_settings):
    """When bundle mode is active we MUST hit bundle_client and never
    call api.anthropic.com directly. Asserting via mock — if we patched
    httpx, the original direct-API code path would never be triggered
    and the test would pass by accident."""
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_ant_response("ok"))

    with patch("app.services.bundle_client.make_anthropic_client", return_value=fake_client) as mk:
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=10,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="claude-haiku-4-5-20251001",
        )

    mk.assert_called_once()


@pytest.mark.asyncio
async def test_bundle_mode_routes_through_make_openai_client(bundle_settings):
    """Symmetric for OpenAI."""
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_openai_response("ok"))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client) as mk:
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=10,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="gpt-5.5",
        )

    mk.assert_called_once()


@pytest.mark.asyncio
async def test_gpt5_uses_max_completion_tokens_not_max_tokens(bundle_settings):
    """gpt-5.x rejects `max_tokens` with HTTP 400 — we must send the
    request with `max_completion_tokens`. This is the kind of subtle bug
    that only shows up against the live API; pin it here."""
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_openai_response("ok"))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=500,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="gpt-5.5",
        )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["max_completion_tokens"] == 500


@pytest.mark.asyncio
async def test_classic_openai_uses_max_tokens(bundle_settings):
    """Symmetric — gpt-4o-style models still take the legacy parameter
    name. Don't over-correct."""
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_openai_response("ok"))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=500,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="gpt-4o-mini",
        )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "max_tokens" in kwargs
    assert "max_completion_tokens" not in kwargs


@pytest.mark.asyncio
async def test_bundle_call_failure_returns_none_not_raises(bundle_settings):
    """The contract is `Optional[str]` — callers expect None on any
    failure (timeout, auth, network), never an exception. The retry
    loop in RoutineRunner depends on this."""
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("app.services.bundle_client.make_anthropic_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        result = await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=10,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="claude-opus-4-7",
        )

    assert result is None


# ── System prompt placement (OpenAI vs Anthropic) ────────────────────


@pytest.mark.asyncio
async def test_openai_system_prompt_goes_into_messages(bundle_settings):
    """OpenAI's chat completions API takes system in-band via a
    {role: system} message — NOT a separate `system` kwarg like
    Anthropic. If we swap this we silently lose the system prompt."""
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_openai_response("ok"))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=10,
            system="MY SYSTEM PROMPT",
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-5.5",
        )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "system" not in kwargs  # Anthropic-style kwarg would be a bug here
    msgs = kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "MY SYSTEM PROMPT"}
    assert msgs[1]["content"] == "hello"


@pytest.mark.asyncio
async def test_gpt5_passes_reasoning_effort_low(bundle_settings):
    """Reasoning models silently spend output-token budget on hidden
    chain-of-thought. For a system-op summarisation there's nothing to
    reason hard about — we must pin `reasoning_effort="low"` so the
    visible-output budget isn't eaten. Real production failure: a 13k-
    token email summary on gpt-5.5 hit the 1200-token cap entirely
    inside the reasoning step and returned empty content."""
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_openai_response("ok"))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=1200,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="gpt-5.5",
        )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_non_reasoning_openai_omits_reasoning_effort(bundle_settings):
    """gpt-4o doesn't accept reasoning_effort — passing it would 400.
    Only reasoning models (gpt-5.x, o-series) get the flag."""
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_openai_response("ok"))

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=1200,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="gpt-4o-mini",
        )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_empty_content_with_nonzero_output_logs_warning(bundle_settings, caplog):
    """When a reasoning model burns through its budget without emitting
    visible text, we must log a clear warning so triage isn't blind."""
    import logging

    # Build a response where content is empty but output_tokens > 0
    block = MagicMock()
    block.message = MagicMock(content="")
    block.finish_reason = "length"
    resp = MagicMock()
    resp.choices = [block]
    resp.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=1200,
        completion_tokens_details=MagicMock(reasoning_tokens=1200),
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=resp)
    caplog.set_level(logging.WARNING, logger="app.services.internal_llm")

    with patch("app.services.bundle_client.make_openai_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        text = await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=1200,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            model="gpt-5.5",
        )

    assert text is None  # contract: empty content surfaces as None
    truncation_logs = [
        r for r in caplog.records
        if "reasoning consumed the budget" in r.getMessage()
    ]
    assert truncation_logs, "expected an operator-visible warning when reasoning ate the budget"


@pytest.mark.asyncio
async def test_anthropic_system_prompt_is_separate_kwarg(bundle_settings):
    """Mirror of the above — Anthropic takes `system` as a top-level
    parameter, not in messages."""
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_ant_response("ok"))

    with patch("app.services.bundle_client.make_anthropic_client", return_value=fake_client):
        from app.services.internal_llm import call_system_llm
        await call_system_llm(
            user_id="u",
            operation_type="system.test",
            max_tokens=10,
            system="MY SYSTEM PROMPT",
            messages=[{"role": "user", "content": "hello"}],
            model="claude-haiku-4-5-20251001",
        )

    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "MY SYSTEM PROMPT"
    # No system role injected into messages.
    assert all(m["role"] != "system" for m in kwargs["messages"])
