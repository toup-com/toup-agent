"""Hitting the daily cap is not being out of credits, and saying so is worse
than saying nothing.

Found 2026-08-13. A demo account showed:

    "⚠️ Error: You're out of Toup credits. Open Credits to top up or
     upgrade your plan and keep going."

The balance at that moment:

    message_credits_remaining   84.90
    integration_credits         499.00
    message_credits_used_today   15.10
    message_credits_daily_cap    15.00   <- this is what fired

The user had 84.90 credits. They had hit the free tier's DAILY cap, which
resets that night. Being told to buy credits they already own reads as the
product upselling instead of explaining — and it sent the founder hunting a
billing bug that did not exist.

The information was always there: `OutOfCreditsError` carries an
`ExhaustedResponse` with `reason` (`daily_cap_exceeded`) and
`daily_reset_at`. `_friendly_error` never looked. Its str() is
"out_of_credits:<reason>:<bucket>" — which contains "credit" — so the
generic keyword bucket matched first and overwrote the truth with the
wrong sentence.

Structured causes must be read BEFORE keyword sniffing. A keyword bucket is
a fallback for exceptions that carry nothing; it must never outrank one that
knows exactly what happened.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

os.environ.setdefault("ENVIRONMENT", "test")


@dataclass
class _Resp:
    reason: str
    bucket: str = "message"
    plan_display_name: str = "Free"
    daily_reset_at: Optional[datetime] = None
    monthly_reset_at: Optional[datetime] = None


class _OOC(Exception):
    def __init__(self, resp: _Resp):
        self.response = resp
        super().__init__(f"out_of_credits:{resp.reason}:{resp.bucket}")


def _friendly(exc: Exception) -> str:
    from app.api.ws_chat import _friendly_error

    return _friendly_error(exc)


def _in(hours: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def test_the_daily_cap_does_not_claim_the_user_is_out_of_credits():
    out = _friendly(_OOC(_Resp("daily_cap_exceeded", daily_reset_at=_in(3.5))))
    low = out.lower()
    assert "out of toup credits" not in low, (
        f"the daily cap still reports being out of credits: {out!r}. The user "
        f"has credits; they hit a per-day limit."
    )
    assert "limit" in low or "daily" in low, out


def test_the_daily_cap_message_says_WHEN_it_resets():
    """Without a reset time the limit reads as permanent, which is the
    difference between 'wait a few hours' and 'this product is broken'."""
    out = _friendly(_OOC(_Resp("daily_cap_exceeded", daily_reset_at=_in(3.5))))
    assert "3h" in out or "hours" in out.lower(), out


def test_the_daily_cap_message_reassures_that_credits_survive():
    """The user's first fear is that their balance was consumed."""
    out = _friendly(_OOC(_Resp("daily_cap_exceeded", daily_reset_at=_in(2))))
    assert "untouched" in out.lower() or "remaining" in out.lower(), out


def test_genuinely_empty_still_says_out_of_credits():
    """The fix must not make the real exhaustion case vague."""
    out = _friendly(
        _OOC(_Resp("insufficient_message_credits", monthly_reset_at=_in(72)))
    )
    assert "out of toup credits" in out.lower(), out


def test_a_missing_reset_time_never_raises_and_never_says_nothing():
    """Copy must not crash, and 'it resets' with no when is worse than
    'soon'."""
    out = _friendly(_OOC(_Resp("daily_cap_exceeded", daily_reset_at=None)))
    assert out and "soon" in out.lower(), out


def test_the_structured_branch_runs_BEFORE_keyword_matching():
    """The exception's str() contains 'credit'. If the keyword bucket is
    reachable first, it overwrites the accurate message and every test above
    passes only by accident of ordering."""
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat._friendly_error)
    structured = src.index("REASON_DAILY_CAP_EXCEEDED")
    keyword = src.index('"insufficient_quota"')
    assert structured < keyword, (
        "the generic credit/billing keyword bucket is evaluated before the "
        "structured reason — it will swallow OutOfCreditsError again"
    )


def test_an_exception_with_no_structured_reason_still_falls_through():
    """Ordinary exceptions must be unaffected by the new branch."""
    out = _friendly(RuntimeError("rate_limit exceeded"))
    assert "rate limit" in out.lower(), out
