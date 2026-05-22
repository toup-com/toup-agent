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

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo


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


def format_message_text(
    reason: str,
    monthly_reset_at: datetime,
    daily_reset_at: Optional[datetime],
    plan_id: str,
    now: Optional[datetime] = None,
) -> str:
    """Plain-text copy for the channel renderer.

    The web chat card overrides this with a live countdown; Telegram /
    voice use this string as-is.
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


def _next_local_midnight_utc(user_timezone: Optional[str], now: datetime) -> datetime:
    """Compute the UTC instant of the user's next local midnight.

    Daily caps reset at the user's stored timezone boundary, falling
    back to UTC. DST transitions handled by zoneinfo.
    """
    tz = ZoneInfo(user_timezone) if user_timezone else ZoneInfo("UTC")
    local_now = now.astimezone(tz)
    next_local = (local_now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return next_local.astimezone(timezone.utc)


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
        daily_reset_at = _next_local_midnight_utc(user_timezone, now)

    message = format_message_text(
        reason=reason,
        monthly_reset_at=monthly_reset_at,
        daily_reset_at=daily_reset_at,
        plan_id=plan_id,
        now=now,
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
