"""R44 — a greeting stops paying default reasoning effort.

Measured 2026-09-13/14: LLM ttft p50 2,947 ms is 75% of the turn's 3,923 ms,
the median OUTPUT is 30 tokens, and 76 of 79 turns called no tool at all. Even
a warm-cache turn (40,192 cached of ~41,000 input) spent 1.5–4.4 s before its
first token, so the gap is deliberation, not prefill — "hi" was being answered
at the model's default effort.

The policy is deliberately narrow: `greeting` and `media` only, low (never
minimal), everything else unchanged. The tests pin all three halves —

  * the map itself, including the flag-off case;
  * the wire: `reasoning` present for those two categories and absent
    otherwise, and absent on a model family that cannot carry it;
  * that it is a REQUEST parameter and not prompt input, which is what makes
    it free to vary per turn against the cached prefix.
"""

from __future__ import annotations

import pytest

from app.config import (
    REASONING_EFFORT_BY_INTENT,
    reasoning_effort_for_intent,
    settings,
)
from app.services.openai_agent_service import (
    OpenAIAgentService,
    supports_reasoning_effort,
)


# ── the map ───────────────────────────────────────────────────────────

def test_only_greeting_and_media_map_to_low():
    assert REASONING_EFFORT_BY_INTENT == {"greeting": "low", "media": "low"}


@pytest.mark.parametrize("category", ["greeting", "media"])
def test_low_effort_categories(category):
    assert reasoning_effort_for_intent(category) == "low"


@pytest.mark.parametrize(
    "category", ["question", "full", "code", "web", "memory", "scheduling",
                 "agent", "", None, "unknown"],
)
def test_every_other_category_sends_no_parameter(category):
    assert reasoning_effort_for_intent(category) is None


def test_flag_off_sends_nothing_anywhere(monkeypatch):
    monkeypatch.setattr(settings, "reasoning_effort_by_intent", False)
    for category in ("greeting", "media", "question", "full"):
        assert reasoning_effort_for_intent(category) is None


# ── the model gate ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "model", ["gpt-5.6-terra", "gpt-5.6", "gpt-5", "o1", "o3-mini", "o4-mini"],
)
def test_reasoning_families_accept_effort(model):
    assert supports_reasoning_effort(model) is True


@pytest.mark.parametrize(
    "model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "claude-opus-4-7", "", None],
)
def test_non_reasoning_models_never_get_the_parameter(model):
    assert supports_reasoning_effort(model) is False


# ── the wire ──────────────────────────────────────────────────────────

class _RecordingClient:
    """Captures the kwargs of the single `responses.create` call."""

    def __init__(self, sink):
        self.responses = self
        self._sink = sink

    async def create(self, **kwargs):
        self._sink.append(kwargs)

        class _Empty:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _Empty()


async def _capture(model: str, reasoning_effort):
    sink: list[dict] = []
    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    svc._responses_reasoning = {}
    svc.client = _RecordingClient(sink)
    stream = svc._create_responses_stream(
        messages=[{"role": "user", "content": "hi"}],
        system="you are an agent",
        tools=None,
        model=model,
        max_tokens=512,
        reasoning_effort=reasoning_effort,
    )
    async for _ in stream:
        pass
    assert len(sink) == 1
    return sink[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "medium"])
async def test_effort_reaches_the_responses_body(effort):
    kwargs = await _capture("gpt-5.6-terra", effort)
    assert kwargs["reasoning"] == {"effort": effort}


@pytest.mark.asyncio
async def test_no_effort_means_no_reasoning_key():
    kwargs = await _capture("gpt-5.6-terra", None)
    assert "reasoning" not in kwargs, (
        "absence must be absence — sending reasoning=None is a different "
        "request from today's and is not what the flag-off path promises"
    )


@pytest.mark.asyncio
async def test_unsupported_family_drops_the_effort():
    kwargs = await _capture("gpt-4o", "low")
    assert "reasoning" not in kwargs


@pytest.mark.asyncio
async def test_effort_is_a_request_param_not_prompt_input():
    """The cached prefix is tools + instructions + input. Effort is in none
    of the three, which is why it can vary per turn for free."""
    with_effort = await _capture("gpt-5.6-terra", "low")
    without = await _capture("gpt-5.6-terra", None)
    for field in ("input", "instructions", "tools"):
        assert with_effort.get(field) == without.get(field), (
            f"`{field}` changed when only the effort changed — effort has "
            f"leaked into the cached prefix"
        )


# ── the caller ────────────────────────────────────────────────────────
#
# The wire tests above pass just as happily when nothing calls the policy.
# These read agent_runner itself, because the whole feature is one call site
# and a guard that cannot see the call site cannot see it disappear.

def _runner_source() -> str:
    from pathlib import Path

    import app.agent.agent_runner as runner

    return Path(runner.__file__).read_text(encoding="utf-8")


def test_runner_derives_effort_from_the_classifier_category():
    src = _runner_source()
    assert "reasoning_effort_for_intent(" in src, (
        "agent_runner no longer asks for an effort — the feature is dead"
    )
    assert 'getattr(query_intent, "category", None)' in src, (
        "the effort must come from the SAME classifier output that "
        "[PERF] query_intent logs, or the trail cannot explain a turn"
    )


def test_runner_withholds_the_kwarg_on_the_anthropic_path():
    src = _runner_source()
    assert '_is_claude_model(active_model)' in src
    i = src.index("_llm_extra_kwargs")
    window = src[i : i + 400]
    assert "_is_claude_model(active_model)" in window, (
        "AnthropicService.create_message_stream has no reasoning_effort "
        "parameter — passing one is a TypeError on every Claude turn"
    )
