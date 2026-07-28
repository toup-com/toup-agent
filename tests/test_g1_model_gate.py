"""G1 gate preparation (W4.1) — gpt-5.6-terra migration plumbing, all dark.

The fleet default does NOT change in this unit (the gate's stop line:
FLEET DEFAULT CHANGES ONLY ON WRITTEN APPROVAL — see
docs/audits/2026-07-g1-model-gate.md). These tests pin the preparation:

  1. Pricing: gpt-5.6-terra/sol/luna present in all three pricing dicts
     (settings.pricing_per_1k, token_tracker.MODEL_PRICING,
     model_session.AVAILABLE_MODELS) with the cached_input column and the
     NEW cache_write column at exactly 1.25x input (5.6 bills cache
     writes; 5.5 writes are free).
  2. Cost math: _calc_cost_cents + tokens_to_credits apply cached/write
     rates ONLY for models carrying the columns; legacy models bill
     byte-identically with or without the new kwargs.
  3. Usage extraction: cache_write_tokens read defensively from OpenAI
     usage payloads (dict AND SDK-object shapes, several field
     spellings, garbage-safe).
  4. Guard: gpt-5.5-pro (no cached-input rate, $30/$180) can never
     resolve as the chat/agent model — resolver logs + falls back.
  5. Capability plumbing: 1M context window, max_completion_tokens,
     reasoning-model + temperature handling for the gpt-5.6 family.

Pure unit tests — no DB, no network.
"""

from __future__ import annotations

import logging
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest_asyncio

from app.services import model_resolver as mr


# Override the suite-wide `_reset_database` autouse fixture from
# conftest.py — these are pure unit tests and don't touch the DB.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


# ── 1. Pricing dicts: the three 5.6 tiers everywhere ─────────────────

# (input, cached_input, cache_write, output) in USD per 1M tokens.
_EXPECTED_PER_1M = {
    "gpt-5.6-terra": (2.50, 0.25, 3.125, 15.00),
    "gpt-5.6-sol": (5.00, 0.50, 6.25, 30.00),
    "gpt-5.6-luna": (1.00, 0.10, 1.25, 6.00),
}


class TestPricingDicts(unittest.TestCase):
    def test_settings_pricing_per_1k_has_all_three_tiers(self):
        from app.config import settings
        for model, (inp, cached, write, out) in _EXPECTED_PER_1M.items():
            entry = settings.pricing_per_1k[model]
            # settings table is per-1K → per-1M / 1000.
            self.assertAlmostEqual(entry["input"], inp / 1000)
            self.assertAlmostEqual(entry["cached_input"], cached / 1000)
            self.assertAlmostEqual(entry["cache_write"], write / 1000)
            self.assertAlmostEqual(entry["output"], out / 1000)

    def test_cache_write_is_exactly_1_25x_input(self):
        """The 5.6 regime bills cache writes at 1.25x input — pin the ratio
        so a price edit can't silently break the economics note."""
        from app.config import settings
        for model in _EXPECTED_PER_1M:
            entry = settings.pricing_per_1k[model]
            self.assertAlmostEqual(entry["cache_write"], entry["input"] * 1.25)

    def test_live_models_have_no_cache_columns(self):
        """Legacy/live models must NOT grow the cache columns in this PR —
        their presence is what activates the new billing math."""
        from app.config import settings
        for model in ("gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-5.4",
                      "claude-opus-4-6", "claude-sonnet-4-6"):
            entry = settings.pricing_per_1k[model]
            self.assertNotIn("cached_input", entry, model)
            self.assertNotIn("cache_write", entry, model)

    def test_token_tracker_pricing_has_all_three_tiers(self):
        from app.agent.token_tracker import MODEL_PRICING
        for model, (inp, cached, write, out) in _EXPECTED_PER_1M.items():
            entry = MODEL_PRICING[model]
            self.assertAlmostEqual(entry["input"], inp)
            self.assertAlmostEqual(entry["cached_input"], cached)
            self.assertAlmostEqual(entry["cache_write"], write)
            self.assertAlmostEqual(entry["output"], out)

    def test_model_session_registry_has_all_three_tiers(self):
        from app.agent.model_session import AVAILABLE_MODELS
        for model, (inp, _cached, _write, out) in _EXPECTED_PER_1M.items():
            entry = AVAILABLE_MODELS[model]
            self.assertEqual(entry["provider"], "openai")
            self.assertAlmostEqual(entry["cost_in"], inp)
            self.assertAlmostEqual(entry["cost_out"], out)
            self.assertEqual(entry["context"], 1_000_000)

    def test_resolver_pricing_for_resolves_terra(self):
        result = mr.pricing_for("gpt-5.6-terra")
        self.assertIsNotNone(result)
        inp, out = result
        self.assertAlmostEqual(inp, 0.0025)
        self.assertAlmostEqual(out, 0.015)


# ── 2. Cost math: cache-aware ONLY for models with the columns ───────


class TestProxyCostMath(unittest.TestCase):
    def test_terra_cold_turn_prices_at_base_input(self):
        from app.api.llm_proxy import _calc_cost_cents
        # 100k in / 1k out, nothing cached: 100k*$2.5/M + 1k*$15/M
        # = $0.25 + $0.015 = 26.5c → int() → 26.
        self.assertEqual(_calc_cost_cents("gpt-5.6-terra", 100_000, 1_000), 26)

    def test_terra_cached_read_bills_at_cached_rate(self):
        from app.api.llm_proxy import _calc_cost_cents
        # 100k in of which 80k cached: 20k*$2.5/M + 80k*$0.25/M + 1k*$15/M
        # = $0.05 + $0.02 + $0.015 = 8.5c → 8.
        self.assertEqual(
            _calc_cost_cents("gpt-5.6-terra", 100_000, 1_000, cached_tokens=80_000), 8
        )

    def test_terra_cache_write_bills_at_1_25x(self):
        from app.api.llm_proxy import _calc_cost_cents
        # 100k in of which 80k written to cache: 20k*$2.5/M + 80k*$3.125/M
        # + 1k*$15/M = $0.05 + $0.25 + $0.015 = 31.5c → 31 (a write turn
        # costs MORE than an uncached turn — the 1.25x economics).
        self.assertEqual(
            _calc_cost_cents(
                "gpt-5.6-terra", 100_000, 1_000, cache_write_tokens=80_000
            ),
            31,
        )

    def test_terra_read_and_write_disjoint_and_clamped(self):
        from app.api.llm_proxy import _calc_cost_cents
        # Bogus provider usage claiming more cached+written than prompt
        # tokens must clamp, never go negative: 10k in, "20k cached, 20k
        # written" → cached clamps to 10k, written to 0.
        self.assertEqual(
            _calc_cost_cents(
                "gpt-5.6-terra", 10_000, 0,
                cached_tokens=20_000, cache_write_tokens=20_000,
            ),
            # 10k * $0.25/M = $0.0025 → 0.25c → floor max(1, 0) = 1.
            1,
        )

    def test_gpt55_billing_byte_identical_with_cache_kwargs(self):
        """The live default model's billing must not move in this PR."""
        from app.api.llm_proxy import _calc_cost_cents
        base = _calc_cost_cents("gpt-5.5", 27_200, 500)
        self.assertEqual(
            _calc_cost_cents(
                "gpt-5.5", 27_200, 500,
                cached_tokens=22_144, cache_write_tokens=5_000,
            ),
            base,
        )


class TestCreditMath(unittest.TestCase):
    def test_terra_credits_discount_cached_reads(self):
        from app.services.credit_service import tokens_to_credits
        cold = tokens_to_credits("gpt-5.6-terra", 100_000, 1_000)
        warm = tokens_to_credits("gpt-5.6-terra", 100_000, 1_000, cached_tokens=80_000)
        self.assertEqual(cold, Decimal("26.5"))
        self.assertEqual(warm, Decimal("8.5"))

    def test_terra_credits_bill_cache_writes(self):
        from app.services.credit_service import tokens_to_credits
        write_turn = tokens_to_credits(
            "gpt-5.6-terra", 100_000, 1_000, cache_write_tokens=80_000
        )
        self.assertEqual(write_turn, Decimal("31.5"))

    def test_gpt55_credits_identical_with_cache_kwargs(self):
        from app.services.credit_service import tokens_to_credits, tokens_to_credits_raw
        for fn in (tokens_to_credits, tokens_to_credits_raw):
            base = fn("gpt-5.5", 27_200, 500)
            self.assertEqual(
                fn("gpt-5.5", 27_200, 500,
                   cached_tokens=22_144, cache_write_tokens=5_000),
                base,
            )

    def test_unknown_model_fallback_unchanged(self):
        from app.services.credit_service import tokens_to_credits
        base = tokens_to_credits("gpt-9000-not-real", 1000, 1000)
        self.assertEqual(
            tokens_to_credits("gpt-9000-not-real", 1000, 1000,
                              cached_tokens=500, cache_write_tokens=500),
            base,
        )


# ── 3. cache_write_tokens extraction (defensive, both shapes) ────────


class TestCacheWriteExtraction(unittest.TestCase):
    def test_dict_shape_cache_write_tokens(self):
        from app.api.llm_proxy import _extract_openai_cache_write_tokens
        usage = {"prompt_tokens_details": {"cache_write_tokens": 4096}}
        self.assertEqual(_extract_openai_cache_write_tokens(usage), 4096)

    def test_dict_shape_fallback_spellings(self):
        from app.api.llm_proxy import _extract_openai_cache_write_tokens
        self.assertEqual(
            _extract_openai_cache_write_tokens(
                {"prompt_tokens_details": {"cache_creation_tokens": 128}}
            ),
            128,
        )
        self.assertEqual(
            _extract_openai_cache_write_tokens(
                {"prompt_tokens_details": {"cache_creation_input_tokens": 256}}
            ),
            256,
        )

    def test_sdk_object_shape_via_getattr(self):
        from app.api.llm_proxy import _extract_openai_cache_write_tokens
        usage = SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=1024, cache_write_tokens=2048
            )
        )
        self.assertEqual(_extract_openai_cache_write_tokens(usage), 2048)

    def test_absent_and_garbage_default_to_zero(self):
        from app.api.llm_proxy import _extract_openai_cache_write_tokens
        for usage in (
            None,
            {},
            {"prompt_tokens_details": None},
            {"prompt_tokens_details": {}},
            {"prompt_tokens_details": {"cached_tokens": 512}},  # reads only
            {"prompt_tokens_details": {"cache_write_tokens": None}},
            {"prompt_tokens_details": {"cache_write_tokens": "bogus"}},
            {"prompt_tokens_details": "bogus"},
            SimpleNamespace(prompt_tokens_details=None),
        ):
            self.assertEqual(
                _extract_openai_cache_write_tokens(usage), 0, repr(usage)
            )

    def test_sse_twin_reads_last_usage_frame(self):
        from app.api.llm_proxy import _extract_openai_cache_write_from_sse
        raw = (
            'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            'data: {"usage": {"prompt_tokens": 1200, "completion_tokens": 40,'
            ' "prompt_tokens_details": {"cached_tokens": 1024,'
            ' "cache_write_tokens": 176}}}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.assertEqual(_extract_openai_cache_write_from_sse(raw), 176)

    def test_sse_twin_zero_when_absent(self):
        from app.api.llm_proxy import _extract_openai_cache_write_from_sse
        raw = (
            'data: {"usage": {"prompt_tokens": 300, "completion_tokens": 7}}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.assertEqual(_extract_openai_cache_write_from_sse(raw), 0)

    def test_streaming_3_tuple_contract_untouched(self):
        """The pinned _extract_openai_usage 3-tuple contract (W0.2/F-7
        tests) must stay byte-stable — the write count rides a twin."""
        from app.api.llm_proxy import _extract_openai_usage
        raw = (
            'data: {"usage": {"prompt_tokens": 1200, "completion_tokens": 40,'
            ' "prompt_tokens_details": {"cached_tokens": 1024,'
            ' "cache_write_tokens": 176}}}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.assertEqual(_extract_openai_usage(raw), (1200, 40, 1024))


# ── 4. The no-cached-rate guard (gpt-5.5-pro et al.) ─────────────────


def _settings_with(**overrides):
    return patch.object(mr, "settings", SimpleNamespace(**overrides))


class TestProModelGuard(unittest.TestCase):
    def test_has_cached_input_rate_denies_pro_tier(self):
        for model in ("gpt-5.5-pro", "GPT-5.5-PRO", "gpt-5.6-pro",
                      "o1-pro", "o3-pro"):
            self.assertFalse(mr.has_cached_input_rate(model), model)

    def test_has_cached_input_rate_allows_everything_else(self):
        for model in ("gpt-5.5", "gpt-5.6-terra", "gpt-5.6-sol",
                      "gpt-5.6-luna", "gpt-4o", "claude-opus-4-7",
                      "", None):
            self.assertTrue(mr.has_cached_input_rate(model), model)

    def test_settings_override_to_pro_falls_back_to_canonical(self):
        with _settings_with(agent_model="gpt-5.5-pro"):
            self.assertEqual(mr.default_model(), "gpt-5.5")

    def test_per_tenant_pro_falls_back_to_settings_layer(self):
        cfg = SimpleNamespace(agent_model="gpt-5.5-pro")
        with _settings_with(agent_model="gpt-5.6-terra"):
            self.assertEqual(mr.default_model(cfg), "gpt-5.6-terra")

    def test_both_layers_pro_falls_back_to_canonical(self):
        cfg = SimpleNamespace(agent_model="o1-pro")
        with _settings_with(agent_model="gpt-5.5-pro"):
            self.assertEqual(mr.default_model(cfg), "gpt-5.5")

    def test_guard_logs_an_error(self):
        with _settings_with(agent_model="gpt-5.5-pro"):
            with self.assertLogs(mr.logger, level=logging.ERROR) as captured:
                mr.default_model()
        joined = "\n".join(captured.output)
        self.assertIn("gpt-5.5-pro", joined)
        self.assertIn("G1 gate", joined)

    def test_valid_override_still_resolves_normally(self):
        """Regression: the guard must not disturb the existing chain."""
        cfg = SimpleNamespace(agent_model="gpt-4o")
        with _settings_with(agent_model="gpt-5.4"):
            self.assertEqual(mr.default_model(cfg), "gpt-4o")
        with _settings_with(agent_model="gpt-5.4"):
            self.assertEqual(mr.default_model(), "gpt-5.4")
        with _settings_with():
            self.assertEqual(mr.default_model(), "gpt-5.5")


# ── 5. Capability plumbing for the 5.6 family ────────────────────────


class TestCapabilityPlumbing(unittest.TestCase):
    def test_context_windows_1m(self):
        from app.agent.context_manager import MODEL_CONTEXT_WINDOWS
        for model in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6"):
            self.assertEqual(MODEL_CONTEXT_WINDOWS[model], 1_000_000, model)

    def test_resolver_context_window_for_terra(self):
        self.assertEqual(mr.context_window_for("gpt-5.6-terra"), 1_000_000)

    def test_uses_max_completion_tokens(self):
        for model in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna"):
            self.assertTrue(mr.uses_max_completion_tokens(model), model)

    def test_is_reasoning_model(self):
        for model in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna"):
            self.assertTrue(mr.is_reasoning_model(model), model)

    def test_no_custom_temperature(self):
        for model in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna"):
            self.assertFalse(mr.supports_custom_temperature(model), model)

    def test_classified_as_openai(self):
        for model in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna"):
            self.assertTrue(mr.is_openai_model(model), model)
            self.assertFalse(mr.is_claude_model(model), model)

    def test_public_label_tiers(self):
        """Leak-filter tiering: terra/sol are Deep, luna is Fast — no 5.6
        id may fall through to a raw provider string."""
        from app.services.model_alias import (
            public_model_label, TIER_DEEP, TIER_FAST,
        )
        self.assertEqual(public_model_label("gpt-5.6-terra"), TIER_DEEP)
        self.assertEqual(public_model_label("gpt-5.6-sol"), TIER_DEEP)
        self.assertEqual(public_model_label("gpt-5.6-luna"), TIER_FAST)


if __name__ == "__main__":
    unittest.main()
