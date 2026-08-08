"""The OpenAI wire is derived from the model, not configured beside it.

gpt-5.6-* returns 400 on /v1/chat/completions whenever function tools are
present (canary abort 2026-07-28, config.py). Every agent turn sends function
tools. Before 2026-08-07 the wire came from `settings.openai_wire_api` alone,
which made "which model" and "which wire" two independent settings that had to
agree. A container with one flipped and not the other fails EVERY turn while
every health check stays green — the exact shape that has cost this codebase
fleets before (the four bind gates, the append-only bridge env, the pool
slots that never upgraded once claimed).

So `model_resolver.wire_api_for()` derives it. This suite pins that the broken
half-flipped state is unreachable, and — just as important — that the derived
rule did not quietly move every OTHER model onto the Responses wire.

Sections:
  A. The predicate, including the ids that must NOT match.
  B. Derivation precedence: the mandate wins over settings; settings still
     govern every other family.
  C. The integration pin — drive create_message_stream and prove which wire
     the client actually touched, with a chat-path control.
  D. The shipped configuration is self-consistent, and every per-model
     registry knows the default model.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from app.config import Settings, settings
from app.services import model_resolver as mr
from app.services.openai_agent_service import OpenAIAgentService


# ── Shared harness (same idiom as test_openai_responses_wire) ─────────
# Each fake client exposes exactly ONE wire surface, so touching the other
# raises AttributeError rather than silently passing.


class _AsyncStream:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item
        return gen()


def _ev(type_: str, **kw):
    return NS(type=type_, **kw)


_USAGE = NS(input_tokens=10, output_tokens=2,
            input_tokens_details=NS(cached_tokens=4))


def _chat_only_client():
    """.responses is absent — reaching for it is an AttributeError."""
    chunks = [
        NS(id="chatcmpl-1", usage=None,
           choices=[NS(delta=NS(content="hi", tool_calls=None),
                       finish_reason=None)]),
        NS(id="chatcmpl-1",
           usage=NS(prompt_tokens=10, completion_tokens=2,
                    prompt_tokens_details=NS(cached_tokens=4)),
           choices=[NS(delta=NS(content=None, tool_calls=None),
                       finish_reason="stop")]),
    ]
    return NS(chat=NS(completions=NS(
        create=AsyncMock(return_value=_AsyncStream(chunks)))))


def _responses_only_client():
    """.chat is absent — reaching for it is an AttributeError."""
    events = [
        _ev("response.output_text.delta", delta="hi"),
        _ev("response.completed", response=NS(id="resp_1", usage=_USAGE)),
    ]
    return NS(responses=NS(create=AsyncMock(return_value=_AsyncStream(events))))


def _make_service(fake_client) -> OpenAIAgentService:
    svc = OpenAIAgentService()
    svc.client = fake_client
    return svc


async def _drive(svc, **kwargs):
    return [ev async for ev in svc.create_message_stream(**kwargs)]


_TOOLS = [{
    "name": "web_search",
    "description": "Search the web",
    "input_schema": {"type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"]},
}]


# ──────────────────────────────────────────────────────────────────────
# A. The predicate
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", [
    "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6",
    "GPT-5.6-TERRA", "  gpt-5.6-terra  ",
])
def test_the_5_6_family_mandates_the_responses_wire(model):
    assert mr.requires_responses_wire(model) is True


@pytest.mark.parametrize("model", [
    "gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-4o", "gpt-4o-mini",
    "claude-opus-4-7", "", None,
])
def test_nothing_else_mandates_it(model):
    """The anti-vacuity half of the predicate. A rule that returns True for
    everything would pass every test in section B while silently moving the
    whole platform onto a wire it was never canaried on."""
    assert mr.requires_responses_wire(model) is False


def test_the_prefix_is_matched_at_the_start_not_anywhere(model=None):
    """`in` instead of `startswith` would match a tenant-supplied id like
    'my-gpt-5.6-clone' or a vendor-prefixed 'openrouter/gpt-5.6'."""
    assert mr.requires_responses_wire("not-gpt-5.6-really") is False
    assert mr.requires_responses_wire("openrouter/gpt-5.6") is False


# ──────────────────────────────────────────────────────────────────────
# B. Derivation precedence
# ──────────────────────────────────────────────────────────────────────


def test_the_mandate_beats_a_chat_setting(monkeypatch):
    """THE point of this module. Someone sets AGENT_MODEL=gpt-5.6-terra and
    leaves OPENAI_WIRE_API=chat; before, that container 400s every turn."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    assert mr.wire_api_for("gpt-5.6-terra") == "responses"


def test_the_mandate_beats_an_absent_setting(monkeypatch):
    monkeypatch.setattr(settings, "openai_wire_api", "", raising=False)
    assert mr.wire_api_for("gpt-5.6-terra") == "responses"
    monkeypatch.setattr(settings, "openai_wire_api", None, raising=False)
    assert mr.wire_api_for("gpt-5.6-terra") == "responses"


def test_other_families_still_honour_the_setting(monkeypatch):
    """The derived rule must not take the setting away from an operator who
    wants gpt-4o or gpt-5.5 on the Responses wire."""
    monkeypatch.setattr(settings, "openai_wire_api", "responses", raising=False)
    assert mr.wire_api_for("gpt-5.5") == "responses"
    assert mr.wire_api_for("gpt-4o") == "responses"


def test_other_families_default_to_chat(monkeypatch):
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    assert mr.wire_api_for("gpt-5.5") == "chat"
    assert mr.wire_api_for("gpt-4o") == "chat"
    assert mr.wire_api_for(None) == "chat"


def test_an_unset_setting_falls_back_to_chat_not_to_empty(monkeypatch):
    monkeypatch.setattr(settings, "openai_wire_api", "", raising=False)
    assert mr.wire_api_for("gpt-4o") == "chat"


def test_the_setting_is_normalised(monkeypatch):
    monkeypatch.setattr(settings, "openai_wire_api", "  RESPONSES ",
                        raising=False)
    assert mr.wire_api_for("gpt-4o") == "responses"


def test_the_cross_provider_fallback_keeps_the_chat_wire(monkeypatch):
    """Deliberate: the gpt-4o fallback fired 0 times in the 14-day window, so
    it has no production evidence on the Responses wire. Moving the primary
    must not silently move the untested escape hatch with it."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    assert mr.requires_responses_wire(mr._CANONICAL_FALLBACK_MODEL) is False
    assert mr.wire_api_for(mr._CANONICAL_FALLBACK_MODEL) == "chat"


# ──────────────────────────────────────────────────────────────────────
# C. Integration — which wire did the client actually get called on
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_5_6_model_reaches_the_responses_wire_despite_chat_settings(
    monkeypatch,
):
    """End to end through create_message_stream, with tools present — the
    exact condition under which the chat wire 400s. The client has no .chat
    attribute at all, so taking the old path raises instead of passing."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    fake = _responses_only_client()
    svc = _make_service(fake)

    events = await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        tools=_TOOLS,
        model="gpt-5.6-terra",
        max_tokens=64,
    )

    assert fake.responses.create.await_count == 1
    assert [e.type for e in events][-1] == "message_end"


@pytest.mark.asyncio
async def test_control_a_5_5_model_still_reaches_the_chat_wire(monkeypatch):
    """The anti-vacuity control for the test above. Without it, that test
    also passes in the world where EVERY model was routed to Responses —
    which is the regression this whole design is trying to avoid. This
    client has no .responses attribute."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    fake = _chat_only_client()
    svc = _make_service(fake)

    events = await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        tools=_TOOLS,
        model="gpt-5.5",
        max_tokens=64,
    )

    assert fake.chat.completions.create.await_count == 1
    assert [e.type for e in events][-1] == "message_end"


@pytest.mark.asyncio
async def test_the_shipped_default_model_reaches_the_responses_wire(
    monkeypatch,
):
    """model=None resolves to the service default, which resolves through
    model_resolver to the shipped default. This is the path a real container
    with no AGENT_MODEL env var takes — 53 of 57 at the time of writing."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    fake = _responses_only_client()
    svc = _make_service(fake)

    await _drive(
        svc,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        tools=_TOOLS,
        model=None,
        max_tokens=64,
    )

    assert fake.responses.create.await_count == 1


# ──────────────────────────────────────────────────────────────────────
# D. The shipped configuration is self-consistent
# ──────────────────────────────────────────────────────────────────────


def test_the_shipped_default_is_terra():
    """Pins the G1 decision itself (docs/audits/2026-08-g1-cost-and-latency.md
    §8). Changing it should be a deliberate edit that updates this line."""
    assert Settings.model_fields["agent_model"].default == "gpt-5.6-terra"
    assert mr._CANONICAL_AGENT_MODEL == "gpt-5.6-terra"


def test_the_default_is_not_barred_by_the_g1_pro_guard():
    """default_model() silently falls through for any candidate with no
    cached-input rate. A default that tripped its own guard would resolve to
    itself and look fine, so assert the predicate directly."""
    assert mr.has_cached_input_rate(mr._CANONICAL_AGENT_MODEL) is True


def test_the_default_and_the_canonical_openai_model_agree():
    assert mr._CANONICAL_OPENAI_MODEL == mr._CANONICAL_AGENT_MODEL


def test_every_per_model_registry_knows_the_default_model():
    """A model bump touches several independent registries. Missing one does
    not raise — it silently falls back to a 128k context window, a $0 cost, or
    an unlabelled entry in the model picker. Assert each explicitly."""
    model = mr._CANONICAL_AGENT_MODEL

    from app.agent.context_manager import (
        DEFAULT_CONTEXT_WINDOW, MODEL_CONTEXT_WINDOWS,
    )
    assert model in MODEL_CONTEXT_WINDOWS, "context window registry"
    assert MODEL_CONTEXT_WINDOWS[model] != DEFAULT_CONTEXT_WINDOW, (
        "present but equal to the fallback is indistinguishable from absent"
    )

    from app.agent.token_tracker import MODEL_PRICING
    assert model in MODEL_PRICING, "token_tracker pricing"

    from app.agent.model_session import AVAILABLE_MODELS
    assert model in AVAILABLE_MODELS, "model_session registry"

    from app.api.models import _LABEL_MAP
    assert model in _LABEL_MAP, "the /api/models display label"

    assert model in settings.pricing_per_1k, "config.pricing_per_1k"


def test_the_default_has_a_cached_input_rate_in_the_pricing_table():
    """G1's whole cost case rests on the cached-input discount. A pricing
    entry without the column bills cached reads at the full input rate — the
    live defect gpt-5.5 has (docs/audits/2026-08-g1-cost-and-latency.md §3.2).
    The model we are moving TO must not repeat it."""
    entry = settings.pricing_per_1k[mr._CANONICAL_AGENT_MODEL]
    assert entry.get("cached_input"), (
        f"{mr._CANONICAL_AGENT_MODEL} has no cached_input rate; cached reads "
        f"would be billed at the full input rate"
    )
