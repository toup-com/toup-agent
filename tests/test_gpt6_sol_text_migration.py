"""Current GPT-6 Sol chat defaults, wire parameters, and long-context billing."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from app.agent.model_session import ModelSessionManager
from app.agent.token_tracker import UsageRecord
from app.api.llm_proxy import _calc_cost_cents
from app.services.credit_service import tokens_to_credits_raw
from app.services.llm_service import LLMService
from app.services.model_resolver import LONG_CONTEXT_THRESHOLD, long_context_price_multipliers


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    # These tests exercise cost arithmetic and a mocked OpenAI stream.
    yield


@pytest.mark.parametrize("model,short_out,long_out", [
    ("gpt-6-sol", Decimal("70"), Decimal("135")),
    ("gpt-5.6-terra", Decimal("72"), Decimal("138")),
])
def test_long_context_surcharge_applies_to_the_whole_request(model, short_out, long_out):
    assert long_context_price_multipliers(model, LONG_CONTEXT_THRESHOLD) == (1.0, 1.0)
    assert long_context_price_multipliers(model, LONG_CONTEXT_THRESHOLD + 1) == (2.0, 1.5)

    # 300K input, 10K output. All 300K input tokens receive the 2x
    # multiplier; output receives 1.5x once the threshold is crossed.
    assert _calc_cost_cents(model, 300_000, 10_000) == long_out
    assert tokens_to_credits_raw(model, 300_000, 10_000) == long_out
    assert _calc_cost_cents(model, 270_000, 10_000) < short_out

    # Both simple analytics registries must use long-context rates as well.
    usage = UsageRecord(session_id="s", model=model, input_tokens=300_000, output_tokens=10_000)
    assert usage.cost_usd == float(long_out) / 100
    tracked = ModelSessionManager().track_usage("s", model, 300_000, 10_000)
    assert tracked.cost == pytest.approx(float(long_out) / 100)


def test_long_context_cache_read_and_write_rates_match_provider(model="gpt-6-sol"):
    # 100K uncached + 100K read + 100K written: short input is
    # $0.20 + $0.02 + $0.25 = $0.47; long input $0.94.
    # 10K output is $0.15 at long-context rates.
    expected_cents = Decimal("109")
    kwargs = dict(cached_tokens=100_000, cache_write_tokens=100_000)
    # The proxy's existing R-3 cap truncates a floating-point cent at the
    # whole-cent boundary; the credit ledger retains the precise 0.1c value.
    assert Decimal("108") <= _calc_cost_cents(model, 300_000, 10_000, **kwargs) <= expected_cents
    assert tokens_to_credits_raw(model, 300_000, 10_000, **kwargs) == expected_cents


@pytest.mark.asyncio
@pytest.mark.parametrize("model,expected_arg", [
    ("gpt-6-sol", "max_completion_tokens"),
    ("gpt-4o", "max_tokens"),
])
async def test_http_chat_stream_uses_the_model_compatible_token_parameter(model, expected_arg):
    class Stream:
        def __aiter__(self):
            async def generate():
                yield NS(choices=[NS(delta=NS(content="hello"), finish_reason="stop")])
            return generate()

    create = AsyncMock(return_value=Stream())
    service = LLMService.__new__(LLMService)
    service._use_anthropic = False
    service._openai_client = NS(chat=NS(completions=NS(create=create)))
    service.default_model = model
    service.default_temperature = 0.7
    service.default_max_tokens = 4096

    chunks = [chunk async for chunk in service.stream(
        messages=[{"role": "user", "content": "hello"}],
        model=model,
        max_tokens=64,
    )]

    sent = create.await_args.kwargs
    assert sent[expected_arg] == 64
    assert ("max_tokens" in sent) != ("max_completion_tokens" in sent)
    assert ("temperature" in sent) is (model == "gpt-4o")
    assert chunks[0].content == "hello"


def test_openai_only_byok_chat_uses_primary_model(monkeypatch):
    from app.config import settings
    from app.services import key_provider, model_router

    # Exercise the BYOK branch explicitly; Anthropic is normally disabled
    # fleet-wide and would otherwise short-circuit before it.
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "llm_mode", "manual")
    monkeypatch.setattr(settings, "agent_model", "gpt-6-sol")
    monkeypatch.setattr(key_provider, "keys", NS(has_anthropic=False, has_openai=True))

    decision = model_router.classify_request("hello")
    assert decision.model == "gpt-6-sol"
