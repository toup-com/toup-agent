"""Tests for credit-exhausted copy generation.

Regression bug (2026-05-25, operator-reported): a Builder user with 6.9
message credits remaining sent a chat message and the assistant card
displayed "You've used all your message credits for this period" — a
verifiable lie. The /account#billing page showed 313/320 used, 6.9
remaining, on the same screen. The bug was that
``format_message_text()`` produced copy purely from the reason code
without consulting ``balance_after``, so REASON_INSUFFICIENT_MESSAGE
always rendered as "used all" even when balance > 0.

The fix threads ``balance_after`` through and emits a partial-balance
copy that names the remaining amount honestly. These tests pin both
branches so a future refactor can't silently regress the honest path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal


_NOW = datetime(2026, 5, 25, 18, 0, 0, tzinfo=timezone.utc)
_RENEW = datetime(2026, 6, 20, 0, 0, 0, tzinfo=timezone.utc)


def test_partial_balance_copy_names_remaining_credits():
    """Builder user with 6.9 credits left → copy must mention the
    remaining amount and NOT claim "used all". This is the exact
    operator-reported scenario."""
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_MESSAGE, format_message_text,
    )
    txt = format_message_text(
        reason=REASON_INSUFFICIENT_MESSAGE,
        monthly_reset_at=_RENEW,
        daily_reset_at=None,
        plan_id="builder",
        now=_NOW,
        balance_after=Decimal("6.9"),
    )
    assert "6.9" in txt, f"copy must surface remaining balance, got: {txt}"
    assert "used all" not in txt.lower(), (
        f"partial-balance copy must NOT claim 'used all' — got: {txt}"
    )
    # Sanity: still mentions renewal countdown + upgrade CTA.
    assert "renew" in txt.lower()
    assert "/pricing" in txt


def test_zero_balance_keeps_used_all_copy():
    """When balance is genuinely 0, the original 'used all' copy is
    correct and clearer. Don't accidentally regress this branch."""
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_MESSAGE, format_message_text,
    )
    txt = format_message_text(
        reason=REASON_INSUFFICIENT_MESSAGE,
        monthly_reset_at=_RENEW,
        daily_reset_at=None,
        plan_id="builder",
        now=_NOW,
        balance_after=0,
    )
    assert "used all" in txt.lower(), (
        f"zero-balance must use the canonical 'used all' phrasing, got: {txt}"
    )


def test_partial_balance_renders_whole_numbers_cleanly():
    """A balance of 6.0 should render as '6', not '6.0'."""
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_MESSAGE, format_message_text,
    )
    txt = format_message_text(
        reason=REASON_INSUFFICIENT_MESSAGE,
        monthly_reset_at=_RENEW,
        daily_reset_at=None,
        plan_id="builder",
        now=_NOW,
        balance_after=Decimal("6"),
    )
    # No literal "6.0" — would look unpolished.
    assert "6.0" not in txt
    assert " 6 " in txt or " 6 message" in txt or "remaining 6" in txt


def test_integration_bucket_uses_integration_word():
    """REASON_INSUFFICIENT_INTEGRATION must use 'integration' not
    'message' in the partial-balance copy (and same for zero-balance)."""
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_INTEGRATION, format_message_text,
    )
    partial = format_message_text(
        reason=REASON_INSUFFICIENT_INTEGRATION,
        monthly_reset_at=_RENEW,
        daily_reset_at=None,
        plan_id="builder",
        now=_NOW,
        balance_after=Decimal("12.5"),
    )
    assert "integration" in partial.lower()
    assert "message credit" not in partial.lower()


def test_none_balance_falls_through_to_used_all():
    """Older callers that don't pass balance_after must still get
    valid copy (the legacy 'used all' branch). Guards back-compat
    for any caller that hasn't been updated to thread balance."""
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_MESSAGE, format_message_text,
    )
    txt = format_message_text(
        reason=REASON_INSUFFICIENT_MESSAGE,
        monthly_reset_at=_RENEW,
        daily_reset_at=None,
        plan_id="builder",
        now=_NOW,
        balance_after=None,
    )
    assert "used all" in txt.lower()


def test_build_exhausted_response_threads_balance_into_message():
    """End-to-end: build_exhausted_response must forward balance_after
    to format_message_text. Regression for the original bug — fixing
    only one of the two would still leak the misleading copy via the
    ExhaustedResponse.message field that channels use directly."""
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_MESSAGE, build_exhausted_response,
    )
    resp = build_exhausted_response(
        reason=REASON_INSUFFICIENT_MESSAGE,
        bucket="message",
        balance_after=Decimal("6.9"),
        plan_id="builder",
        plan_display_name="Builder",
        period_end=_RENEW,
        now=_NOW,
    )
    assert resp.balance_after == 6.9
    assert "6.9" in resp.message
    assert "used all" not in resp.message.lower()


def test_build_exhausted_response_uses_real_period_end_not_30_day_fallback():
    """When period_end is supplied, the countdown must derive from it,
    not from a now+30d fallback. Regression for the secondary bug
    where /agent-deduct didn't return period_end and the card showed
    'renew in 30 days, 0 hours' for a user whose Stripe renewal was
    4 weeks away."""
    from app.services.credit_exhausted import (
        REASON_INSUFFICIENT_MESSAGE, build_exhausted_response,
    )
    resp = build_exhausted_response(
        reason=REASON_INSUFFICIENT_MESSAGE,
        bucket="message",
        balance_after=Decimal("0"),
        plan_id="builder",
        plan_display_name="Builder",
        period_end=_RENEW,
        now=_NOW,
    )
    # Real delta: 2026-05-25 18:00 → 2026-06-20 00:00 = 25 days 6h
    assert resp.monthly_reset_at == _RENEW
    # The "in 30 days, 0 hours" fallback string should never appear
    # in the message when period_end is real.
    assert "30 days" not in resp.message, (
        f"period_end was supplied — copy must not show fallback 30-day "
        f"countdown, got: {resp.message}"
    )
