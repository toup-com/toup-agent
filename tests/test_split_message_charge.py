"""Unit tests for credit_service._split_message_charge — the pure helper
that decides how a MESSAGE-bucket charge splits across the plan wallet and
the non-expiring purchased (StoreKit IAP) wallet.

The reviewer's contract:
  1. purchased_remaining == 0 reduces EXACTLY to the pre-IAP behaviour across
     {under cap, over cap, insufficient}.
  2. with purchased > 0 the daily cap limits ONLY plan spend; purchased
     credits cover the overflow and bypass the cap.
  3. reason precedence: a true shortfall reports INSUFFICIENT, otherwise the
     daily cap reports DAILY_CAP_EXCEEDED.

No DB — pure Decimal math.
"""
from __future__ import annotations

from decimal import Decimal

from app.services.credit_service import (
    REASON_DAILY_CAP_EXCEEDED,
    REASON_INSUFFICIENT_MESSAGE,
    _split_message_charge,
)


D = Decimal


# ── (1) purchased == 0 → exact legacy behaviour ──────────────────────


def test_p0_under_cap_feasible():
    # plan 100, used 0, cap 15, charge 10 → all from plan, feasible.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("100"), D("0"), D("0"), D("15"), D("10"),
    )
    assert from_plan == D("10")
    assert from_pur == D("0")
    assert feasible is True
    assert reason is None


def test_p0_no_cap_feasible():
    # daily_cap=None → cap_room infinite; plan covers it.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("100"), D("0"), D("50"), None, D("40"),
    )
    assert from_plan == D("40")
    assert from_pur == D("0")
    assert feasible is True
    assert reason is None


def test_p0_over_cap_denied_with_cap_reason():
    # plan 100 (plenty), but cap 15 with 10 already used → only 5 of room.
    # charge 10 > 5 room → infeasible, and since amount(10) <= plan(100),
    # the reason is the DAILY CAP, not insufficient.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("100"), D("0"), D("10"), D("15"), D("10"),
    )
    assert feasible is False
    assert reason == REASON_DAILY_CAP_EXCEEDED


def test_p0_insufficient_denied_with_insufficient_reason():
    # plan 5, charge 10, no cap → amount > plan → INSUFFICIENT.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("5"), D("0"), D("0"), None, D("10"),
    )
    assert feasible is False
    assert reason == REASON_INSUFFICIENT_MESSAGE


def test_p0_insufficient_takes_precedence_over_cap():
    # plan 5, cap 3, charge 10 → amount(10) > plan(5)+purchased(0) →
    # INSUFFICIENT wins over cap (matches legacy: insufficient checked first).
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("5"), D("0"), D("0"), D("3"), D("10"),
    )
    assert feasible is False
    assert reason == REASON_INSUFFICIENT_MESSAGE


def test_p0_exact_cap_boundary_feasible():
    # cap 15, used 5, charge 10 → projected 15 == cap → still feasible
    # (legacy used strict > for denial).
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("100"), D("0"), D("5"), D("15"), D("10"),
    )
    assert from_plan == D("10")
    assert feasible is True
    assert reason is None


# ── (2) purchased > 0 → cap limits plan only, purchased covers overflow ─


def test_purchased_covers_cap_overflow():
    # plan 100, purchased 50, cap 15 with 10 used → 5 plan room.
    # charge 30 → 5 from plan, 25 from purchased, feasible (bypasses cap).
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("100"), D("50"), D("10"), D("15"), D("30"),
    )
    assert from_plan == D("5")
    assert from_pur == D("25")
    assert feasible is True
    assert reason is None


def test_purchased_covers_when_plan_exhausted():
    # plan 0, purchased 100, no/any cap → whole charge from purchased.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("0"), D("100"), D("0"), D("15"), D("40"),
    )
    assert from_plan == D("0")
    assert from_pur == D("40")
    assert feasible is True
    assert reason is None


def test_purchased_only_user_can_spend_past_cap():
    # plan 0, purchased 1000, cap 15 already fully used (used==cap) → 0 plan
    # room, but purchased covers the full charge regardless of the cap.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("0"), D("1000"), D("15"), D("15"), D("500"),
    )
    assert from_plan == D("0")
    assert from_pur == D("500")
    assert feasible is True
    assert reason is None


def test_no_cap_with_purchased_drains_plan_first():
    # daily_cap=None, plan 30, purchased 100, charge 50 → 30 plan + 20 purchased.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("30"), D("100"), D("0"), None, D("50"),
    )
    assert from_plan == D("30")
    assert from_pur == D("20")
    assert feasible is True
    assert reason is None


# ── (3) reason precedence with purchased > 0 ─────────────────────────


def test_purchased_insufficient_total_reports_insufficient():
    # plan 5, purchased 10 (total 15), charge 20 > 15 → INSUFFICIENT,
    # not daily-cap, even though a cap exists.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("5"), D("10"), D("0"), D("3"), D("20"),
    )
    assert feasible is False
    assert reason == REASON_INSUFFICIENT_MESSAGE


def test_total_sufficient_but_purchased_short_is_feasible():
    # The split never makes purchased the limiting factor when total covers
    # it: plan 5 (cap room 5), purchased 100, charge 10 → 5 plan + 5 purchased,
    # feasible. Confirms cap pushes overflow to purchased, not a denial.
    from_plan, from_pur, feasible, reason = _split_message_charge(
        D("100"), D("100"), D("0"), D("5"), D("10"),
    )
    assert from_plan == D("5")
    assert from_pur == D("5")
    assert feasible is True
    assert reason is None


def test_conservation_from_plan_plus_purchased_equals_amount_when_feasible():
    # For any feasible charge, from_plan + from_purchased == amount.
    for plan_r, pur_r, used, cap, amt in [
        (D("100"), D("0"), D("0"), D("15"), D("10")),
        (D("100"), D("50"), D("10"), D("15"), D("30")),
        (D("0"), D("1000"), D("15"), D("15"), D("500")),
        (D("30"), D("100"), D("0"), None, D("50")),
    ]:
        fp, fpur, feasible, _ = _split_message_charge(plan_r, pur_r, used, cap, amt)
        assert feasible is True
        assert fp + fpur == amt
