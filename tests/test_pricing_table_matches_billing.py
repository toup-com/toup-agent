"""The pricing table against what OpenAI actually bills.

Every cost number this platform produces — `llm_proxy_events.cost_cents`, the
credit ledger, every model-gate comparison — is computed from
`settings.pricing_per_1k` by `_calc_cost_cents`. When a column is missing or
wrong there is no error and no warning: just a wrong number that everything
downstream inherits, which is how two models stayed mispriced for months.

Found 2026-08-07 by reading OpenAI's organization billing directly
(docs/audits/2026-08-g1-cost-and-latency.md §8):

  * `gpt-5.5` had NO `cached_input` column while the provider was discounting
    56.8% of its input tokens. Measured rate $0.5493/M = 0.098x the same
    window's uncached rate.
  * `gpt-4o-mini` — the memory-extraction model, 43.3% of 46.2M input tokens
    cached — same defect. Measured $0.0750/M = exactly 0.500x uncached.
  * `gpt-5.6-terra`'s `cache_write` was 0.003125, a MODELLED 1.25x input
    surcharge. Measured, that billed line prices at $2.548/M — terra's list
    input rate — while the separate `terra, input` line is $0.0045. There is
    no surcharge; OpenAI files terra's ordinary uncached input under that
    label.

Sections:
  A. The measured rates, pinned as ratios (robust to a list-price change).
  B. The three pricing dicts agree.
  C. `_calc_cost_cents` actually applies the columns, with controls.
  D. The detection guard for the NEXT missing column.
"""

from __future__ import annotations

import logging

import pytest

from app.config import settings
from app.api.llm_proxy import _calc_cost_cents


# ── A. Measured rates ─────────────────────────────────────────────────


@pytest.mark.parametrize("model,ratio", [
    # ratio = cached_input / input, MEASURED against org billing 2026-08-07.
    ("gpt-5.5", 0.10),        # $0.5493/M measured vs $5.591/M uncached -> 0.098
    ("gpt-4o-mini", 0.50),    # $0.0750/M measured vs $0.1502/M uncached -> 0.500
    ("gpt-5.6-terra", 0.10),  # published 10%, consistent with $0.206/M measured
])
def test_cached_input_is_the_measured_fraction_of_input(model, ratio):
    """Asserted as a RATIO, not an absolute. A list-price change moves both
    numbers together; a dropped or mistyped column does not."""
    entry = settings.pricing_per_1k[model]
    assert "cached_input" in entry, (
        f"{model} has no cached_input rate — every cached token is billed at "
        f"the full input rate"
    )
    assert entry["cached_input"] == pytest.approx(entry["input"] * ratio, rel=1e-6)


def test_terra_cache_write_is_the_input_rate_not_a_surcharge():
    """MEASURED: the `gpt-5.6-terra, cache writes` billing line prices at
    $2.548/M against the tokens it covers — terra's $2.50/M list input rate.
    It was encoded as 1.25x (0.003125), which overstated it by 25%."""
    terra = settings.pricing_per_1k["gpt-5.6-terra"]
    assert terra["cache_write"] == pytest.approx(terra["input"], rel=1e-6)


def test_unmeasured_models_keep_their_modelled_rate():
    """Anti-overreach control. sol and luna have carried no production
    traffic, so there is nothing to measure and nothing to correct. Changing
    them by analogy with terra would be inventing a number and presenting it
    as a measurement — the failure mode this whole audit exists to end."""
    for model in ("gpt-5.6-sol", "gpt-5.6-luna"):
        entry = settings.pricing_per_1k[model]
        assert entry["cache_write"] == pytest.approx(entry["input"] * 1.25, rel=1e-6)


# ── B. The dicts agree ────────────────────────────────────────────────


def test_the_two_pricing_dicts_agree_on_every_shared_model():
    """settings.pricing_per_1k is per 1K tokens; token_tracker.MODEL_PRICING is
    per 1M. They drift silently because nothing reads both."""
    from app.agent.token_tracker import MODEL_PRICING

    shared = set(settings.pricing_per_1k) & set(MODEL_PRICING)
    assert shared, "no overlap — this test would be vacuous"
    for model in sorted(shared):
        per_1k = settings.pricing_per_1k[model]
        per_1m = MODEL_PRICING[model]
        for column in ("input", "output", "cached_input", "cache_write"):
            in_a, in_b = column in per_1k, column in per_1m
            assert in_a == in_b, (
                f"{model}: '{column}' present in "
                f"{'pricing_per_1k' if in_a else 'MODEL_PRICING'} only"
            )
            if in_a:
                assert per_1m[column] == pytest.approx(per_1k[column] * 1000, rel=1e-6), (
                    f"{model}.{column}: {per_1m[column]}/M vs "
                    f"{per_1k[column]}/1K"
                )


# ── C. The columns are actually applied ───────────────────────────────


def test_a_cached_token_is_cheaper_than_an_uncached_one():
    """The integration that matters: same token count, different cached split.
    Sized well above the 1-cent floor so the comparison is not swallowed."""
    all_uncached = _calc_cost_cents("gpt-5.5", input_tokens=1_000_000,
                                    output_tokens=0, cached_tokens=0)
    all_cached = _calc_cost_cents("gpt-5.5", input_tokens=1_000_000,
                                  output_tokens=0, cached_tokens=1_000_000)
    assert all_cached < all_uncached
    # 10% rate -> a tenth of the cost, within the truncation slop of one cent.
    assert all_cached == pytest.approx(all_uncached // 10, abs=1)


def test_control_a_model_without_the_column_is_unaffected():
    """Anti-vacuity control for the test above. `claude-opus-4-6` has no
    cached_input column, so cached_tokens must change nothing — that is the
    documented pre-G1 behaviour for every model without the columns, and it
    proves the test above is measuring the column rather than some unrelated
    discount applied everywhere."""
    without = _calc_cost_cents("claude-opus-4-6", input_tokens=1_000_000,
                               output_tokens=0, cached_tokens=0)
    with_cached = _calc_cost_cents("claude-opus-4-6", input_tokens=1_000_000,
                                   output_tokens=0, cached_tokens=1_000_000)
    assert without == with_cached


def test_gpt_4o_mini_cached_reads_are_now_halved():
    mini_uncached = _calc_cost_cents("gpt-4o-mini", input_tokens=20_000_000,
                                     output_tokens=0, cached_tokens=0)
    mini_cached = _calc_cost_cents("gpt-4o-mini", input_tokens=20_000_000,
                                   output_tokens=0, cached_tokens=20_000_000)
    assert mini_cached == pytest.approx(mini_uncached // 2, abs=1)


# ── D. The guard for the next one ─────────────────────────────────────


def test_a_model_missing_the_column_warns_when_cached_tokens_arrive(caplog):
    """Detection-only. The defect is invisible by construction, so the only
    thing that can surface the next instance is the provider telling us it
    cached tokens for a model we have no cached rate for."""
    from app.api import llm_proxy

    llm_proxy._MISSING_CACHED_RATE_WARNED.discard("claude-opus-4-6")
    with caplog.at_level(logging.WARNING, logger=llm_proxy.logger.name):
        _calc_cost_cents("claude-opus-4-6", input_tokens=1000,
                         output_tokens=0, cached_tokens=400)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "claude-opus-4-6" in joined
    assert "cached_input" in joined


def test_the_warning_does_not_fire_for_a_model_that_has_the_column(caplog):
    """Anti-vacuity control: a warning that fired unconditionally would pass
    the test above while telling an operator nothing."""
    from app.api import llm_proxy

    with caplog.at_level(logging.WARNING, logger=llm_proxy.logger.name):
        _calc_cost_cents("gpt-5.5", input_tokens=1000,
                         output_tokens=0, cached_tokens=400)
    assert "no cached_input rate" not in caplog.text


def test_the_warning_does_not_fire_when_no_tokens_were_cached(caplog):
    """A model with no cached rate AND no cached tokens is not a defect —
    most models genuinely have no caching. Warning there would train
    operators to ignore the message."""
    from app.api import llm_proxy

    llm_proxy._MISSING_CACHED_RATE_WARNED.discard("claude-sonnet-4-6")
    with caplog.at_level(logging.WARNING, logger=llm_proxy.logger.name):
        _calc_cost_cents("claude-sonnet-4-6", input_tokens=1000,
                         output_tokens=0, cached_tokens=0)
    assert "claude-sonnet-4-6" not in caplog.text


def test_the_warning_is_once_per_model_not_once_per_call(caplog):
    """This is a hot path; a per-call warning buries itself."""
    from app.api import llm_proxy

    llm_proxy._MISSING_CACHED_RATE_WARNED.discard("gpt-4o")
    with caplog.at_level(logging.WARNING, logger=llm_proxy.logger.name):
        for _ in range(5):
            _calc_cost_cents("gpt-4o", input_tokens=1000,
                             output_tokens=0, cached_tokens=400)
    assert sum("gpt-4o" in r.getMessage() for r in caplog.records) == 1


def test_the_guard_does_not_change_the_arithmetic():
    """Detection-only, deliberately. Guessing an unmeasured discount would
    replace a known-wrong number with an unknown-wrong one."""
    from app.api import llm_proxy

    llm_proxy._MISSING_CACHED_RATE_WARNED.discard("gpt-4o")
    with_cached = _calc_cost_cents("gpt-4o", input_tokens=1_000_000,
                                   output_tokens=0, cached_tokens=900_000)
    without = _calc_cost_cents("gpt-4o", input_tokens=1_000_000,
                               output_tokens=0, cached_tokens=0)
    assert with_cached == without
