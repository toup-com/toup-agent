"""GA run — the second cost_cents write surface (#559 fixed only the first).

Voice metering (`_meter_voice_turn`) wrote `cost_cents=int(round(...))`
into the same `llm_proxy_events` column llm_proxy writes, and `round()`
goes UP half the time. Measured on production by joining each voice row
to its own exact cost in `credit_ledger.underlying_cost_cents` (written
one line below from the same variable): 15 of 27 rows recorded HIGHER
than exact, worst +0.47¢ on a single turn. `_get_spend` has no endpoint
filter, so the overstatement was summed into the openai bundle gate for
a real non-admin user.

The fix routes the write through `_voice_recorded_cost_cents`, the same
never-higher rule as R-3: `min(exact_4dp, legacy)` where legacy is THIS
surface's historical expression, `int(round())` — not chat's 1¢ floor.
The round-DOWN half (a 0.35¢ turn recording 0) is deliberately kept:
raising it would record more than legacy, which the authorization
forbids. Documented residual.

Red-first: the never-higher and exactness tests fail against the
pre-fix `int(round())` write with `assert 1 == Decimal('0.6922')`.
"""
from __future__ import annotations

from decimal import Decimal


def test_round_up_half_is_gone():
    """0.6922¢ exact was recorded as 1¢ — the prod worst case's shape."""
    from app.api.ws_realtime import _voice_recorded_cost_cents

    assert _voice_recorded_cost_cents(0.6922) == Decimal("0.6922")
    assert _voice_recorded_cost_cents(1.532) == Decimal("1.532")


def test_round_down_half_is_kept_never_higher():
    """0.3482¢ legacy-recorded 0; raising it to 0.3482 would record MORE
    than legacy, which R-3 forbids. The understatement stays, on purpose."""
    from app.api.ws_realtime import _voice_recorded_cost_cents

    assert _voice_recorded_cost_cents(0.3482) == Decimal("0")


def test_never_higher_than_legacy_across_grid():
    from app.api.ws_realtime import _voice_recorded_cost_cents

    for cents in (0.0, 0.0001, 0.05, 0.3482, 0.5, 0.6922, 0.9999,
                  1.0, 1.4999, 1.5, 1.532, 2.75, 10.499, 10.5):
        legacy = int(round(cents))
        recorded = _voice_recorded_cost_cents(cents)
        assert recorded <= Decimal(legacy), (cents, recorded, legacy)
        # And never above the exact cost either.
        assert recorded <= Decimal(str(cents)).quantize(Decimal("0.0001")) or (
            recorded == Decimal(str(cents)).quantize(Decimal("0.0001"))
        ), (cents, recorded)


def test_zero_and_negative_record_zero():
    from app.api.ws_realtime import _voice_recorded_cost_cents

    assert _voice_recorded_cost_cents(0.0) == Decimal("0")
    assert _voice_recorded_cost_cents(-0.01) == Decimal("0")


def test_write_site_routes_through_the_helper():
    """`int(round(cost_cents))` at the LLMProxyEvent write is the defect;
    it must not come back."""
    import inspect

    from app.api import ws_realtime

    src = inspect.getsource(ws_realtime._meter_voice_turn)
    assert "_voice_recorded_cost_cents(" in src
    assert "int(round(cost_cents))" not in src
