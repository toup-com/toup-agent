"""Shared empty-balance response surface for the credit system.

Every chargeable code path needs the same answer when a user hits
zero: a structured response with a reason, a server-supplied
`reset_at` timestamp, and copy that the channel renderer dresses up
(web chat card, Telegram text, voice synthesis, etc.).

This module centralizes that:

* :class:`OutOfCreditsError`  — exception raised at the call site when
  enforcement is on AND the user is out. Carries the structured fields
  so the catching handler doesn't have to recompute them.
* :class:`ExhaustedResponse` — dataclass for the structured payload.
* :func:`build_exhausted_response`  — synthesises the response from a
  `credit_service.ChargeResult` + a `BalanceView` (or just the reason
  + balance view, when the upstream caller only has those).
* :func:`format_message_text` — channel-agnostic plain-text copy. The
  reset_at timestamps are passed separately so channels with rich UI
  (web chat card, mobile modal) can render a live countdown.

Why a separate module
---------------------
The same exhausted-balance condition can be reached from at least:

* `/api/llm/chat` proxy pre-flight (bundle mode)
* `report_llm_usage` post-flight response (manual mode)
* Connector dispatcher pre-flight (integration bucket)
* App-builder pre-flight (build job gate)
* Routine handlers (`call_system_llm` pre-flight)

Each one needs to render the same copy with the same timestamps. A
shared module means the copy lives in one place — if marketing wants
to change "credits reset on X" to "credits refresh on X", we edit
exactly here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


logger = logging.getLogger(__name__)


# Reason codes — match credit_service.REASON_* constants so callers
# downstream can read either source without translation.
REASON_INSUFFICIENT_MESSAGE = "insufficient_message_credits"
REASON_INSUFFICIENT_INTEGRATION = "insufficient_integration_credits"
REASON_DAILY_CAP_EXCEEDED = "daily_cap_exceeded"
REASON_EMAIL_NOT_VERIFIED = "email_not_verified"


@dataclass
class ExhaustedResponse:
    """Structured response for an exhausted-balance condition.

    Channels render this differently:

    * Web chat: assistant message + embedded card with live countdown,
      "Upgrade plan" CTA.
    * Mobile: modal + countdown + CTA.
    * Telegram: plain text + deep link.
    * Voice: spoken message, ends session.
    * Routines: ledger row marked skipped + one-per-period user notice.
    """
    reason: str
    bucket: str  # "message" | "integration"
    balance_after: float
    plan_id: str
    plan_display_name: str
    # Monthly cap reset — when the user's credit allowance refills.
    # Always UTC. Channels generate countdowns client-side from this.
    monthly_reset_at: datetime
    # Daily-cap reset (free tier hits this BEFORE monthly). When None,
    # the user's tier has no daily cap.
    daily_reset_at: Optional[datetime] = None
    # CTA payload for the upgrade button. URL is relative — channels
    # join with their base URL or wrap in a deep-link signer.
    cta_label: str = "Upgrade plan"
    cta_url: str = "/pricing"
    # Free-form text the channel can use directly (Telegram, voice).
    message: str = ""


class OutOfCreditsError(Exception):
    """Raised by chargeable surfaces when enforcement is on AND user is out.

    Always carries an :class:`ExhaustedResponse` so the catching
    handler doesn't have to re-derive the reason/reset timestamps. The
    catcher's job is just to render — through `agent_runner` for chat
    surfaces, or directly into the response for non-stream endpoints.
    """

    def __init__(self, response: ExhaustedResponse):
        self.response = response
        super().__init__(f"out_of_credits:{response.reason}:{response.bucket}")


def _humanize_delta(delta: timedelta) -> str:
    """Convert a timedelta to a human-readable "in X hours Y minutes" string.

    Used in the static-text copy. Channels with live countdowns
    (web chat, mobile) should compute their own ticker against the
    UTC timestamp instead of using this string.
    """
    total_seconds = max(0, int(delta.total_seconds()))
    if total_seconds < 60:
        return "in less than a minute"
    minutes = total_seconds // 60
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days >= 1:
        return f"in {days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
    if hours >= 1:
        return f"in {hours}h {minutes}m"
    return f"in {minutes} minute{'s' if minutes != 1 else ''}"


def _format_remaining(balance: Optional[float | Decimal]) -> Optional[str]:
    """Render a balance as a short, honest display string ("6.9", "0.4")
    or None when we don't have a balance to show. Hides the cents on
    whole-number balances so "100.0" displays as "100"."""
    if balance is None:
        return None
    try:
        b = float(balance)
    except (TypeError, ValueError):
        return None
    if b <= 0:
        return None
    if abs(b - round(b)) < 0.05:
        return str(int(round(b)))
    return f"{b:.1f}"


def format_message_text(
    reason: str,
    monthly_reset_at: datetime,
    daily_reset_at: Optional[datetime],
    plan_id: str,
    now: Optional[datetime] = None,
    balance_after: Optional[float | Decimal] = None,
) -> str:
    """Plain-text copy for the channel renderer.

    The web chat card overrides this with a live countdown; Telegram /
    voice use this string as-is.

    `balance_after` distinguishes two flavors of INSUFFICIENT:
      * balance > 0 — user has SOME credits but not enough for this
        request (multi-turn agent loop overshoots remaining). Copy
        honestly shows the remaining amount instead of lying about
        "used all".
      * balance == 0 (or unknown) — user truly out, original copy.
    """
    now = now or datetime.now(timezone.utc)
    if monthly_reset_at.tzinfo is None:
        monthly_reset_at = monthly_reset_at.replace(tzinfo=timezone.utc)
    if daily_reset_at is not None and daily_reset_at.tzinfo is None:
        daily_reset_at = daily_reset_at.replace(tzinfo=timezone.utc)

    if reason == REASON_EMAIL_NOT_VERIFIED:
        return (
            "Please verify your email before continuing. Check your "
            "inbox for the verification link — once confirmed, your "
            "credits unlock immediately."
        )

    if reason == REASON_DAILY_CAP_EXCEEDED and daily_reset_at is not None:
        delta_txt = _humanize_delta(daily_reset_at - now)
        return (
            f"You've hit today's free limit. Your daily allowance "
            f"resets {delta_txt}. Upgrade for higher daily limits and "
            f"instant access — visit /pricing."
        )

    if reason in (REASON_INSUFFICIENT_MESSAGE, REASON_INSUFFICIENT_INTEGRATION):
        delta_txt = _humanize_delta(monthly_reset_at - now)
        bucket_word = "message" if reason == REASON_INSUFFICIENT_MESSAGE else "integration"
        remaining_txt = _format_remaining(balance_after)
        if remaining_txt is not None:
            # Honest "partial balance" copy — user has some credits left
            # but the next request needs more than they have. A multi-
            # turn agent task can run several LLM calls per user message
            # so the threshold is per-task, not per-message.
            return (
                f"Your remaining {remaining_txt} {bucket_word} "
                f"credit{'s' if remaining_txt != '1' else ''} aren't enough "
                f"to complete this request. They renew {delta_txt}. "
                f"Upgrade your plan to keep going now — visit /pricing."
            )
        return (
            f"You've used all your {bucket_word} credits for this "
            f"period. They renew {delta_txt}. Upgrade your plan to "
            f"keep going now — visit /pricing."
        )

    # Generic fallback — shouldn't fire in practice, but better than empty.
    return (
        "You're out of credits for the current period. Upgrade your "
        "plan at /pricing to continue, or wait for the next renewal."
    )


# Bounded: a client reporting a different malformed zone on every message must
# not grow this without limit. Past the bound we stop deduplicating and the
# repeated lines are themselves the signal.
_WARNED_ZONES: set[str] = set()
_WARNED_ZONES_MAX = 64


def _zone(user_timezone: Optional[str]) -> tzinfo:
    """Resolve a stored timezone name to a zone, falling back to UTC.

    ``User.timezone`` is client-reported and only the REST profile route
    validates it (``auth.py`` calls ``ZoneInfo`` and 400s a bad value); the chat
    WS persists whatever the client sent after checking ``len < 50`` and nothing
    else. So an unresolvable name can reach this module, and every caller here
    is on a path that must not fail because of one: ``_local_day_iso`` runs
    inside ``try_charge``, where raising turns a bad timezone string into a
    failed turn, and ``get_balance_view`` serves ``/credits/status``, where
    raising 500s the one screen whose job is to explain the limit just hit.

    UTC is the same fallback a NULL timezone already gets, so a bad value
    degrades to the documented default instead of to an outage — but it is
    logged once per distinct value per process, because silently treating
    ``Mars/Olympus`` as UTC would hide a real client bug for as long as it ran.

    Returns ``datetime.timezone.utc`` rather than ``ZoneInfo("UTC")`` on the
    fallback so the recovery path cannot itself raise on a host with no tzdata.
    """
    if not user_timezone:
        return timezone.utc
    try:
        return ZoneInfo(user_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        if user_timezone not in _WARNED_ZONES:
            if len(_WARNED_ZONES) < _WARNED_ZONES_MAX:
                _WARNED_ZONES.add(user_timezone)
            logger.warning(
                "[CREDITS] unresolvable stored timezone %r — treating as UTC",
                user_timezone[:64],
            )
        return timezone.utc


def _daily_rollover_utc(
    user_timezone: Optional[str],
    now: datetime,
    anchor_local_date: Optional[str] = None,
) -> datetime:
    """The UTC instant at which the daily counter next rolls.

    No clock rolls it. ``credit_service._reset_daily_if_needed`` zeroes the
    counter on the first charge whose local day is strictly GREATER than
    ``credit_balances.day_anchor_local_date``, and ``_effective_used_today``
    applies the identical rule for readers. So the moment the cap lifts is the
    first instant of the local day *after that anchor* — not simply "tomorrow".

    The two coincide whenever the anchor is today, and diverge in exactly the
    case the anchor rule was written for: the anchor is seeded from the UTC date
    at signup, so for a user west of UTC it can sit a day AHEAD of their local
    date until the app reports a timezone. "Next local midnight" is then wrong
    by a full 24 hours, in the direction that promises capacity the user does
    not have. Measured, not reasoned — Toronto at 22:30 local with a UTC-seeded
    anchor rolls at 2026-08-14T04:00Z; next-local-midnight answers
    2026-08-13T04:00Z.

    ``anchor_local_date`` is optional because the agent-side reporter renders
    the out-of-credits copy from a cached snapshot with no balance row in it.
    Omitting it yields the next local midnight, which is the same answer for
    every user whose anchor is today. One function either way, so the two
    surfaces cannot quote roll times that disagree for the same user.
    """
    tz = _zone(user_timezone)
    today = now.astimezone(tz).date()
    last_counted = today
    if anchor_local_date:
        try:
            # Forward only, matching _reset_daily_if_needed: an anchor ahead of
            # today is legitimate and the day it names is the one still being
            # counted, so it — not today — is what the next day follows.
            last_counted = max(date.fromisoformat(anchor_local_date), today)
        except ValueError:
            # A malformed anchor is not worth failing a status read over; the
            # ordinary rule is the safe reading.
            last_counted = today
    nxt = last_counted + timedelta(days=1)
    # Building the local date at 00:00 and converting is exact even where local
    # midnight does not exist: on a zone that springs forward AT midnight
    # (Santiago, Havana, Beirut) zoneinfo maps 00:00 onto the post-transition
    # offset, and the resulting instant is verifiably the first of that local
    # day — one second earlier still reads as the previous date.
    return datetime(nxt.year, nxt.month, nxt.day, tzinfo=tz).astimezone(timezone.utc)


def build_exhausted_response(
    *,
    reason: str,
    bucket: str,
    balance_after: Decimal | float,
    plan_id: str,
    plan_display_name: str,
    period_end: Optional[datetime],
    user_timezone: Optional[str] = None,
    has_daily_cap: bool = False,
    now: Optional[datetime] = None,
) -> ExhaustedResponse:
    """Construct an ExhaustedResponse from caller-known credit state.

    Callers typically have a ``BalanceView`` + a denial reason; this
    helper translates them into the rendered payload. ``period_end``
    feeds the monthly_reset_at; daily_reset_at is computed from the
    user's timezone when ``has_daily_cap`` is True.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Monthly: use the credit_balances.period_end. Fall back to "30 days
    # from now" if the caller didn't have it (defensive — every caller
    # should provide it; this prevents NPE in odd edge cases).
    if period_end is None:
        monthly_reset_at = now + timedelta(days=30)
    else:
        if period_end.tzinfo is None:
            period_end = period_end.replace(tzinfo=timezone.utc)
        monthly_reset_at = period_end

    daily_reset_at: Optional[datetime] = None
    if has_daily_cap and reason in (REASON_DAILY_CAP_EXCEEDED, REASON_INSUFFICIENT_MESSAGE):
        daily_reset_at = _daily_rollover_utc(user_timezone, now)

    message = format_message_text(
        reason=reason,
        monthly_reset_at=monthly_reset_at,
        daily_reset_at=daily_reset_at,
        plan_id=plan_id,
        now=now,
        balance_after=balance_after,
    )

    return ExhaustedResponse(
        reason=reason,
        bucket=bucket,
        balance_after=float(balance_after),
        plan_id=plan_id,
        plan_display_name=plan_display_name,
        monthly_reset_at=monthly_reset_at,
        daily_reset_at=daily_reset_at,
        cta_label="Upgrade plan",
        cta_url="/pricing",
        message=message,
    )


def response_to_http_detail(resp: ExhaustedResponse) -> dict:
    """Render an ExhaustedResponse as an HTTP 402 detail dict.

    The shape matches the existing llm_proxy `/api/llm/chat` 402
    response so chat clients have a single decode path regardless of
    which surface raised it.
    """
    return {
        "error": "out_of_credits",
        "reason": resp.reason,
        "bucket": resp.bucket,
        "balance_after": str(resp.balance_after),
        "plan_id": resp.plan_id,
        "plan_display_name": resp.plan_display_name,
        "monthly_reset_at": resp.monthly_reset_at.isoformat(),
        "daily_reset_at": resp.daily_reset_at.isoformat() if resp.daily_reset_at else None,
        "cta_label": resp.cta_label,
        "cta_url": resp.cta_url,
        "message": resp.message,
    }


def response_to_stream_event(resp: ExhaustedResponse) -> dict:
    """Render an ExhaustedResponse as a websocket / stream event.

    Used by agent_runner when an LLM service raises mid-stream so the
    chat client can render the same card as the proxy 402 path.
    """
    return {
        "type": "credit_exhausted",
        **response_to_http_detail(resp),
    }
