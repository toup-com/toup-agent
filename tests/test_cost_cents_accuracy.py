"""R-3 (GA run) — recorded cost accuracy: remove the 1¢/call floor.

``cost_cents`` was ``max(1, int(cost_usd * 100))`` at three sites
(llm_proxy._calc_cost_cents, the embeddings route, internal_llm), so a
15-token embedding whose true cost is 0.03¢ was recorded as 1¢ — 55
embedding calls totalling 630 tokens were recorded as 55¢ (~33x). The
monthly bundle budget gate sums this column, so the floor consumed real
user budget with fake spend.

The authorized fix is bounded: recorded cost may only ever go DOWN or
stay equal, never up. Exact fractional accounting would RAISE mid-range
rows (int() truncates 3.7¢ to 3¢), so the shipped rule is

    recorded = min(exact_fractional_4dp, legacy_floor_value)

which removes the floor's inflation and pins the truncation half in
place until a change that may raise costs is separately approved. The
truncation understatement (up to 33% on 3-8¢ calls) is therefore a
DOCUMENTED, DELIBERATE residual, not an oversight.

Red-first evidence (run against pre-fix code): the fractional and
zero-cost tests fail with `assert 1 == Decimal(...)`, and the schema pin
fails because the column is still Integer.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.config import settings


TEST_MODEL = "ga-cost-test-model"
# input $0.001/1k, output $0.002/1k → 10 input tokens = 0.001¢
TEST_PRICING = {"input": 0.001, "output": 0.002}


@pytest.fixture(autouse=True)
def _pin_pricing(monkeypatch):
    pricing = dict(settings.pricing_per_1k)
    pricing[TEST_MODEL] = TEST_PRICING
    monkeypatch.setattr(settings, "pricing_per_1k", pricing, raising=False)
    yield


# ──────────────────────────────────────────────────────────────────────
# llm_proxy._calc_cost_cents
# ──────────────────────────────────────────────────────────────────────


def test_subcent_call_records_fractional_cost_not_one_cent():
    """10 input tokens at $0.001/1k = $0.00001 = 0.001¢ — not 1¢."""
    from app.api.llm_proxy import _calc_cost_cents

    cost = _calc_cost_cents(TEST_MODEL, 10, 0)
    assert cost == Decimal("0.001"), (
        f"a 0.001-cent call was recorded as {cost} — the 1-cent floor "
        "inflates sub-cent calls up to 1000x and the budget gate sums it"
    )


def test_zero_token_call_costs_zero():
    from app.api.llm_proxy import _calc_cost_cents

    assert _calc_cost_cents(TEST_MODEL, 0, 0) == Decimal("0")


def test_multicent_call_is_never_raised_above_legacy():
    """37,000 input tokens = 3.7¢ exact; legacy recorded int() = 3¢.

    The authorization forbids recording MORE than before, so 3¢ stays —
    the truncation understatement is pinned deliberately, not fixed.
    """
    from app.api.llm_proxy import _calc_cost_cents

    cost = _calc_cost_cents(TEST_MODEL, 37_000, 0)
    assert cost == Decimal("3"), (
        f"expected the legacy truncated 3 (never-higher bound), got {cost}"
    )


def test_never_higher_property_across_token_grid():
    """For every (input, output) shape: new value <= the legacy value."""
    from app.api.llm_proxy import _calc_cost_cents

    for inp in (0, 1, 10, 500, 1_000, 9_999, 37_000, 250_000):
        for out in (0, 3, 700, 12_000):
            exact_usd = (inp * TEST_PRICING["input"] / 1000) + (
                out * TEST_PRICING["output"] / 1000
            )
            legacy = max(1, int(exact_usd * 100))
            new = _calc_cost_cents(TEST_MODEL, inp, out)
            assert new <= legacy, (inp, out, new, legacy)


def test_cached_token_discount_still_applies():
    """The G1 cache-aware arithmetic is untouched — only the floor went."""
    from app.api.llm_proxy import _calc_cost_cents

    pricing = dict(settings.pricing_per_1k)
    pricing[TEST_MODEL] = {**TEST_PRICING, "cached_input": 0.0001}
    settings.pricing_per_1k = pricing

    # 10,000 prompt tokens, 9,000 cached: 1,000*0.001 + 9,000*0.0001 per 1k
    cost = _calc_cost_cents(TEST_MODEL, 10_000, 0, cached_tokens=9_000)
    exact = Decimal("0.19")  # (1000*0.001 + 9000*0.0001)/1000 * 100
    assert cost == exact, cost


# ──────────────────────────────────────────────────────────────────────
# internal_llm._calc_cost_cents (system-LLM events share the column)
# ──────────────────────────────────────────────────────────────────────


def test_internal_llm_subcent_call_is_fractional(monkeypatch):
    from app.services import internal_llm

    pricing = dict(settings.pricing_per_1k)
    pricing[TEST_MODEL] = TEST_PRICING
    monkeypatch.setattr(settings, "pricing_per_1k", pricing, raising=False)

    cost = internal_llm._calc_cost_cents(TEST_MODEL, 10, 0)
    assert cost == Decimal("0.001"), cost


def test_internal_llm_never_higher_than_legacy(monkeypatch):
    from app.services import internal_llm

    pricing = dict(settings.pricing_per_1k)
    pricing[TEST_MODEL] = TEST_PRICING
    monkeypatch.setattr(settings, "pricing_per_1k", pricing, raising=False)

    assert internal_llm._calc_cost_cents(TEST_MODEL, 37_000, 0) == Decimal("3")


# ──────────────────────────────────────────────────────────────────────
# Schema: the column must be able to HOLD a fraction
# ──────────────────────────────────────────────────────────────────────


def test_event_cost_column_is_fractional():
    """Integer storage rounds every fractional cost back out of existence."""
    import sqlalchemy as sa

    from app.db.models import LLMProxyEvent

    col = LLMProxyEvent.__table__.c.cost_cents
    assert isinstance(col.type, sa.Numeric) and not isinstance(col.type, sa.Integer), (
        f"llm_proxy_events.cost_cents is {col.type!r} — an Integer column "
        "silently re-floors every fractional cost on INSERT"
    )
    assert col.type.scale == 4, col.type


# ──────────────────────────────────────────────────────────────────────
# The embeddings route and the spend reader route through the same rule
# ──────────────────────────────────────────────────────────────────────


def test_embeddings_route_uses_the_shared_unfloored_helper():
    """The embeddings route hand-rolled `max(1, int(...))`; it must route
    through the same never-higher helper as chat."""
    import inspect

    import app.api.llm_proxy as proxy

    # proxy_openai_embeddings is a one-line wrapper; proxy_embeddings is
    # where the cost is computed.
    src = inspect.getsource(proxy.proxy_embeddings)
    assert "_embedding_cost_cents(" in src, (
        "the embeddings route no longer routes through the shared "
        "unfloored cost helper"
    )
    assert "max(1" not in src, "a 1-cent floor is back in the embeddings route"


def test_embedding_cost_helper_is_unfloored_and_never_higher():
    from app.api.llm_proxy import _embedding_cost_cents

    # 630 tokens at $0.00002/token = 1.26¢ — the finish-run example that
    # was recorded as 55¢ across 55 calls.
    assert _embedding_cost_cents(630) == Decimal("1")  # min(1.26, legacy int 1)
    assert _embedding_cost_cents(11) == Decimal("0.022")
    assert _embedding_cost_cents(0) == Decimal("0")


def test_get_spend_preserves_fractions():
    """`_get_spend` wrapped the SUM in int(), truncating fractional spend
    before the budget comparison. Decimal in, Decimal out."""
    import inspect

    import app.api.llm_proxy as proxy

    src = inspect.getsource(proxy._get_spend)
    assert "int(result.scalar())" not in src, (
        "_get_spend still truncates the summed spend to whole cents"
    )


# ──────────────────────────────────────────────────────────────────────
# Response models must accept what the Numeric column now produces
# ──────────────────────────────────────────────────────────────────────


def test_usage_response_accepts_fractional_spend():
    """SUM over Numeric returns fractional Decimals; an int field here is
    a ValidationError — a 500 on /usage the first sub-cent call after the
    deploy."""
    from app.api.llm_proxy import UsageResponse

    r = UsageResponse(
        anthropic_monthly_cents=Decimal("0.5"),
        anthropic_daily_cents=Decimal("0.021"),
        openai_monthly_cents=Decimal("123.4567"),
        anthropic_budget_cents=3000,
        anthropic_daily_cap_cents=200,
        openai_budget_cents=1000,
    )
    assert r.openai_monthly_cents == pytest.approx(123.4567)


def test_admin_stats_accepts_fractional_costs():
    from app.api.llm_proxy import AdminStatsResponse

    r = AdminStatsResponse(
        total_requests_today=3,
        total_cost_cents_today=Decimal("1.26"),
        anthropic_cost_cents_today=Decimal("0"),
        openai_cost_cents_today=Decimal("1.26"),
        fallback_count_today=0,
        error_count_today=0,
        top_users=[],
    )
    assert r.total_cost_cents_today == pytest.approx(1.26)
