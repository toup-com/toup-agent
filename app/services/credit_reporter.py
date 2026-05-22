"""Agent → platform credit deduction reporter + in-process credit state.

When the tenant agent makes a DIRECT LLM call (manual mode, own API
key) it bypasses the platform's `/api/llm/chat` proxy and its
deduction hook. This module closes that gap by POSTing a deduction
record to `{platform_api_url}/credits/agent-deduct` after every
LLM call, authenticated with the agent's existing `X-Agent-Key`.

The platform's `agent_deduct` endpoint validates the key, routes
through `credit_service.try_charge`, and returns the new balance +
enforcement state.

Honoring the response (added 2026-05-21)
----------------------------------------
The first iteration of this module logged the response and discarded
it. That meant users continued chatting past zero balance — the
deduction was recorded but nothing blocked the next call. Now:

* :func:`report_llm_usage` returns a typed :class:`DeductOutcome`.
* Every successful round-trip updates a module-level
  :class:`CreditState` singleton with the platform's latest view of
  balance + enforcement.
* The LLM services (`anthropic_service`, `openai_agent_service`)
  consult :func:`raise_if_exhausted` BEFORE the next call and short-
  circuit with :class:`OutOfCreditsError` — no extra HTTP roundtrip
  needed for pre-flight after the first deduction lands.

The "one slip" — the call that takes us from positive to zero — is
allowed through (it generated the deduct that revealed the state).
Every subsequent call until the platform replies success=True again
is blocked.

Fail-open by design — a credit-reporting outage must NEVER break the
agent's primary chat loop. Network/HTTP failures leave CreditState
unchanged; subsequent calls proceed under the last-known state.

Used by:
    * services/anthropic_service.py — both streaming and non-streaming
      message paths, after `usage` is known, and via
      :func:`raise_if_exhausted` BEFORE every call.
    * services/openai_agent_service.py — same pattern.
    * services/internal_llm.py — system.* calls are exempt server-
      side, so reporting them is harmless. user.* routine calls flow
      through the same path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

import httpx

from app.config import settings
from app.services.credit_exhausted import (
    ExhaustedResponse,
    OutOfCreditsError,
    REASON_DAILY_CAP_EXCEEDED,
    REASON_EMAIL_NOT_VERIFIED,
    REASON_INSUFFICIENT_INTEGRATION,
    REASON_INSUFFICIENT_MESSAGE,
    build_exhausted_response,
)


logger = logging.getLogger(__name__)


# Bound the per-request timeout so the reporter never adds noticeable
# latency to the user's chat response. The deduction is fire-and-forget
# from the agent's perspective — when the platform is slow, we'd rather
# under-meter for one event than make the user wait.
_REPORT_TIMEOUT_S = 3.0
_STATUS_TIMEOUT_S = 2.0


# ── DeductOutcome — typed response shape ─────────────────────────────


@dataclass
class DeductOutcome:
    """Parsed `/agent-deduct` response.

    `network_ok=False` means we couldn't reach the platform — callers
    should treat this as "no information" and NOT block the user
    (fail-open). All other fields mirror the platform's
    `AgentDeductResponse` plus an `exhausted` convenience flag.
    """
    network_ok: bool
    success: bool = True
    enforcement_enabled: bool = False
    balance_after: float = 0.0
    reason: Optional[str] = None
    bucket: str = "message"
    amount_charged: float = 0.0
    idempotent_hit: bool = False

    @property
    def exhausted(self) -> bool:
        """True iff the platform reports enforcement on AND success=False."""
        return bool(self.network_ok and self.enforcement_enabled and not self.success)


# ── CreditState — in-process tracker ─────────────────────────────────


@dataclass
class CreditState:
    """Snapshot of the user's credit state from the most recent server reply.

    Thread-safe by virtue of a tiny lock around mutations. Read paths
    take a copy snapshot under the same lock to avoid torn reads.
    """
    last_outcome: Optional[DeductOutcome] = None
    last_known_period_end: Optional[datetime] = None
    last_known_plan_id: str = "free"
    last_known_plan_display_name: str = "Free"
    last_known_message_daily_cap: Optional[float] = None
    last_known_user_timezone: Optional[str] = None
    last_updated_at: Optional[datetime] = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_deduct(
        self,
        outcome: DeductOutcome,
        *,
        period_end: Optional[datetime] = None,
        plan_id: Optional[str] = None,
        plan_display_name: Optional[str] = None,
        daily_cap: Optional[float] = None,
        user_timezone: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.last_outcome = outcome
            self.last_updated_at = datetime.now(timezone.utc)
            if period_end is not None:
                self.last_known_period_end = period_end
            if plan_id is not None:
                self.last_known_plan_id = plan_id
            if plan_display_name is not None:
                self.last_known_plan_display_name = plan_display_name
            if daily_cap is not None:
                self.last_known_message_daily_cap = daily_cap
            if user_timezone is not None:
                self.last_known_user_timezone = user_timezone

    def is_exhausted(self) -> bool:
        """True iff the latest known state says the user is out of credits."""
        with self._lock:
            return self.last_outcome is not None and self.last_outcome.exhausted

    def build_exhausted_response(self) -> Optional[ExhaustedResponse]:
        """Render the current exhausted state into a structured response.

        Returns None when state isn't exhausted (caller should not hit
        this branch — :func:`raise_if_exhausted` guards against it).
        """
        with self._lock:
            if self.last_outcome is None or not self.last_outcome.exhausted:
                return None
            return build_exhausted_response(
                reason=self.last_outcome.reason or REASON_INSUFFICIENT_MESSAGE,
                bucket=self.last_outcome.bucket,
                balance_after=self.last_outcome.balance_after,
                plan_id=self.last_known_plan_id,
                plan_display_name=self.last_known_plan_display_name,
                period_end=self.last_known_period_end,
                user_timezone=self.last_known_user_timezone,
                has_daily_cap=self.last_known_message_daily_cap is not None,
            )


_state = CreditState()


def get_state() -> CreditState:
    """Module-level CreditState singleton (per-process)."""
    return _state


def raise_if_exhausted() -> None:
    """Raise :class:`OutOfCreditsError` iff the last known state says exhausted.

    Cheap, no-HTTP guard. Call this BEFORE every chargeable LLM call
    so a user who's been blocked once stays blocked until the platform
    replies success=True again.

    Returns None (no exception) when state is unknown — fail-open so a
    cold start doesn't block the user's first message.
    """
    resp = _state.build_exhausted_response()
    if resp is not None:
        raise OutOfCreditsError(resp)


# ── HTTP surface ────────────────────────────────────────────────────


def _platform_endpoint(path: str) -> Optional[str]:
    """Resolve `{platform_api_url}/{path}`. None when not configured."""
    platform_url = (getattr(settings, "platform_api_url", "") or "").strip()
    if not platform_url:
        return None
    return f"{platform_url.rstrip('/')}/{path.lstrip('/')}"


def _agent_key() -> str:
    return (getattr(settings, "agent_api_key", "") or "").strip()


async def report_llm_usage(
    *,
    user_id: str,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    underlying_cost_cents: Optional[float] = None,
    operation_type: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    event_id: Optional[str] = None,
) -> DeductOutcome:
    """POST a deduction record to the platform.

    Returns a :class:`DeductOutcome` carrying the parsed platform
    response. ``network_ok=False`` indicates the call couldn't reach
    the platform — callers should treat this as "no signal" and NOT
    block the user.

    Side effect: updates the module-level CreditState so
    :func:`raise_if_exhausted` reflects the latest server view.
    """
    url = _platform_endpoint("/credits/agent-deduct")
    agent_key = _agent_key()
    if url is None or not agent_key:
        # No platform configured — silent no-op (dev mode, etc.).
        # Returning network_ok=False keeps CreditState untouched.
        return DeductOutcome(network_ok=False)

    if input_tokens <= 0 and output_tokens <= 0:
        # Empty event — nothing meaningful to deduct. Don't churn the
        # platform with zero-token POSTs (some upstream errors log
        # empty usage which would otherwise spam the ledger).
        return DeductOutcome(network_ok=False)

    payload: dict = {
        "user_id": user_id,
        "model": model,
        "provider": provider,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
    }
    if underlying_cost_cents is not None:
        payload["underlying_cost_cents"] = float(underlying_cost_cents)
    if operation_type:
        payload["operation_type"] = operation_type
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    if event_id:
        payload["event_id"] = event_id

    try:
        async with httpx.AsyncClient(timeout=_REPORT_TIMEOUT_S) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"X-Agent-Key": agent_key},
            )
        if resp.status_code != 200:
            logger.warning(
                "[credits] agent-deduct non-200 status=%d body=%s",
                resp.status_code, resp.text[:300],
            )
            return DeductOutcome(network_ok=False)
        data = resp.json() or {}
    except httpx.TimeoutException:
        logger.warning(
            "[credits] agent-deduct timed out (>%ss) user=%s model=%s",
            _REPORT_TIMEOUT_S, (user_id or "?")[:8], model,
        )
        return DeductOutcome(network_ok=False)
    except Exception:
        logger.exception(
            "[credits] agent-deduct failed user=%s model=%s",
            (user_id or "?")[:8], model,
        )
        return DeductOutcome(network_ok=False)

    outcome = DeductOutcome(
        network_ok=True,
        success=bool(data.get("success", True)),
        enforcement_enabled=bool(data.get("enforcement_enabled", False)),
        balance_after=float(data.get("balance_after") or 0.0),
        reason=data.get("reason"),
        bucket=str(data.get("bucket") or "message"),
        amount_charged=float(data.get("amount_charged") or 0.0),
        idempotent_hit=bool(data.get("idempotent_hit", False)),
    )

    if not outcome.idempotent_hit:
        logger.info(
            "[credits] reported user=%s model=%s tokens=%d/%d "
            "balance_after=%.2f success=%s enforce=%s reason=%s",
            (user_id or "?")[:8], model, input_tokens, output_tokens,
            outcome.balance_after, outcome.success,
            outcome.enforcement_enabled, outcome.reason or "-",
        )

    # Update in-process state for the next call's pre-flight short-
    # circuit. Period_end / plan / timezone come from /credits/status
    # which the LLM service refreshes opportunistically; the deduct
    # response alone doesn't include them.
    _state.record_deduct(outcome)

    if outcome.exhausted:
        # Refresh balance metadata so build_exhausted_response can
        # construct accurate timestamps. Best-effort; failure here is
        # not fatal — we still surface the message with sensible
        # defaults.
        await _refresh_status_metadata(user_id)

    return outcome


async def _refresh_status_metadata(user_id: str) -> None:
    """Pull plan / period_end / timezone from `/credits/status`.

    Called opportunistically after a deduct shows exhaustion so the
    rendered timestamps + plan name are accurate. Failure is silent.
    """
    url = _platform_endpoint("/credits/status")
    agent_key = _agent_key()
    if url is None or not agent_key or not user_id:
        return
    try:
        async with httpx.AsyncClient(timeout=_STATUS_TIMEOUT_S) as client:
            resp = await client.get(
                url,
                # `/credits/status` is a user-auth route, not agent. We
                # ride the agent_api_key as a header so the platform can
                # do the lookup; if rejected, the agent gracefully
                # degrades to default-period rendering.
                headers={"X-Agent-Key": agent_key, "X-Agent-User-Id": user_id},
            )
        if resp.status_code != 200:
            return
        data = resp.json() or {}
    except Exception:
        return

    try:
        period_end_iso = data.get("period_end") or ""
        period_end = (
            datetime.fromisoformat(period_end_iso.replace("Z", "+00:00"))
            if period_end_iso else None
        )
        msg = data.get("message") or {}
        _state.record_deduct(
            outcome=_state.last_outcome or DeductOutcome(network_ok=True),
            period_end=period_end,
            plan_id=data.get("plan_id"),
            plan_display_name=data.get("plan_display_name"),
            daily_cap=(float(msg.get("daily_cap")) if msg.get("daily_cap") is not None else None),
        )
    except Exception:
        # Best-effort enrichment; never block on parse failure.
        pass


def is_out_of_credits(deduct_response: Optional[dict] | DeductOutcome) -> bool:
    """Back-compat helper for callers that hold the raw response.

    New code should prefer :class:`DeductOutcome.exhausted` /
    :func:`raise_if_exhausted`.
    """
    if deduct_response is None:
        return False
    if isinstance(deduct_response, DeductOutcome):
        return deduct_response.exhausted
    return bool(
        deduct_response.get("enforcement_enabled")
        and not deduct_response.get("success", True)
    )
