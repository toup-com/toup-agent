"""CreditService — the single deduction surface for the credit-based billing system.

Every chargeable code path goes through one of three entrypoints:

* :py:meth:`try_charge` — instant deduction (single LLM event, single tool call)
* :py:meth:`reserve` + :py:meth:`settle` — two-phase for long-running tasks
* :py:meth:`grant` — adds credits (plan renewal, manual adjust, refund)

All deductions are atomic and idempotent:

* Atomicity: ``SELECT ... FOR UPDATE`` on the balance row (Postgres) or
  serial writer on SQLite. Balance update + ledger insert commit together.
* Idempotency: ``(user_id, idempotency_key)`` is unique in
  ``credit_ledger``. Replays of the same dispatch hit the constraint and
  short-circuit to the original result.
* Daily / monthly resets happen lazily inside the same transaction — no
  background reset job is required for correctness.

Feature flag
------------
``settings.credit_enforcement_enabled`` gates the deny path. When False,
``try_charge`` writes the ledger row and returns success regardless of
balance — useful for shadow-meter rollout.

Email-verified gate (F13)
-------------------------
``settings.require_email_verification_for_credits`` adds an additional
gate: unverified users get ``REASON_EMAIL_NOT_VERIFIED`` (the frontend
renders a "verify your email" CTA, not "upgrade plan"). Users created
before ``settings.email_verification_required_after_iso`` are exempt
(grandfather).

The peg
-------
**1 credit = 1 cent of underlying LLM / tool cost.** See
:py:func:`underlying_cost_to_credits` and the FLAT_FEES table.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import (
    BUCKET_INTEGRATION, BUCKET_MESSAGE,
    LEDGER_DAILY_RESET, LEDGER_MANUAL_ADJUST, LEDGER_PERIOD_RENEWAL,
    LEDGER_PLAN_GRANT, LEDGER_REFUND, LEDGER_RESERVATION, LEDGER_SETTLEMENT,
    RESERVATION_OPEN, RESERVATION_REFUNDED, RESERVATION_SETTLED,
    CreditBalance, CreditLedger, CreditReservation, SubscriptionPlan, User,
)


logger = logging.getLogger(__name__)


_AMOUNT_QUANTUM = Decimal("0.0001")
_DISPLAY_QUANTUM = Decimal("0.1")
_BALANCE_QUANTUM = Decimal("0.01")


def _q(value: Decimal | float | int, quantum: Decimal = _AMOUNT_QUANTUM) -> Decimal:
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def underlying_cost_to_credits(cost_cents: float | int | Decimal) -> Decimal:
    """Convert underlying LLM cost (cents) to credits. Peg: 1c = 1 credit."""
    return _q(cost_cents, _DISPLAY_QUANTUM)


def tokens_to_credits(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Compute credit cost for an LLM call from token counts.

    Uses Decimal math directly (not ``int(cost_usd*100)``) so sub-cent
    calls don't get rounded up to 1¢. This is what makes the design
    claim "0.2 credit per routine fire" actually true.
    """
    pricing = settings.pricing_per_1k.get(model)
    if not pricing:
        pricing = {"input": 0.003, "output": 0.015}
    cost_usd = (
        (Decimal(str(input_tokens)) * Decimal(str(pricing["input"])) / Decimal("1000"))
        + (Decimal(str(output_tokens)) * Decimal(str(pricing["output"])) / Decimal("1000"))
    )
    cost_cents = cost_usd * Decimal("100")
    # Floor at 0.1 credit so we never write zero-amount deduction rows.
    credits = max(Decimal("0.1"), cost_cents)
    return _q(credits, _DISPLAY_QUANTUM)


# Flat-fee credit costs for non-LLM chargeable events.
FLAT_FEES: dict[str, dict] = {
    "web_search":          {"bucket": BUCKET_INTEGRATION, "credits": Decimal("1.0")},
    "web_fetch":           {"bucket": BUCKET_INTEGRATION, "credits": Decimal("0.5")},
    "connector_call":      {"bucket": BUCKET_INTEGRATION, "credits": Decimal("1.0")},
    "connector_bulk_call": {"bucket": BUCKET_INTEGRATION, "credits": Decimal("5.0")},
    "browser_action":      {"bucket": BUCKET_MESSAGE,     "credits": Decimal("0.5")},
    "browser_screenshot":  {"bucket": BUCKET_MESSAGE,     "credits": Decimal("0.1")},
    "doc_gen_pdf":         {"bucket": BUCKET_MESSAGE,     "credits": Decimal("1.0")},
    "doc_gen_docx":        {"bucket": BUCKET_MESSAGE,     "credits": Decimal("1.0")},
    "doc_gen_xlsx":        {"bucket": BUCKET_MESSAGE,     "credits": Decimal("1.0")},
    "doc_gen_pptx":        {"bucket": BUCKET_MESSAGE,     "credits": Decimal("1.0")},
    "reminder_fire":       {"bucket": BUCKET_MESSAGE,     "credits": Decimal("0.1")},
}


async def load_flat_fee_overrides(db_session_maker) -> list[str]:
    """Overlay platform_settings overrides onto in-process FLAT_FEES.

    Called once on boot. Subsequent edits via the admin endpoint update
    FLAT_FEES in-place AND write a platform_settings row, so restart
    picks them up here.
    """
    import json
    from app.db.models import PlatformSetting
    from sqlalchemy import select as _select

    overridden: list[str] = []
    async with db_session_maker() as db:
        rows = (await db.execute(
            _select(PlatformSetting).where(PlatformSetting.key.startswith("credit.flat_fee."))
        )).scalars().all()
        for row in rows:
            fee_key = row.key.removeprefix("credit.flat_fee.")
            if fee_key not in FLAT_FEES:
                continue
            try:
                payload = json.loads(row.value)
                bucket = payload.get("bucket") or FLAT_FEES[fee_key]["bucket"]
                credits = Decimal(str(payload.get("credits")))
                FLAT_FEES[fee_key] = {"bucket": bucket, "credits": credits}
                overridden.append(fee_key)
            except Exception:
                continue
    return overridden


def _flat_fee_for_tool(tool_name: str, tool_input: Optional[dict]) -> Decimal:
    """Look up the integration-bucket credit cost for a connector tool call."""
    if tool_input is None:
        tool_input = {}
    name_l = (tool_name or "").lower()
    if (("list_" in name_l) or ("_search" in name_l) or name_l.endswith("_bulk")) \
            and tool_input.get("include_body", False):
        return FLAT_FEES["connector_bulk_call"]["credits"]
    if name_l in ("web_search", "search"):
        return FLAT_FEES["web_search"]["credits"]
    if name_l in ("web_fetch", "fetch"):
        return FLAT_FEES["web_fetch"]["credits"]
    return FLAT_FEES["connector_call"]["credits"]


@dataclass
class ChargeResult:
    success: bool
    balance_after: Decimal
    reason: Optional[str] = None
    ledger_id: Optional[str] = None
    idempotent_hit: bool = False


@dataclass
class ReservationResult:
    reservation_id: str
    estimated_amount: Decimal
    balance_after: Decimal
    success: bool = True
    reason: Optional[str] = None


@dataclass
class BalanceView:
    plan_id: str
    plan_display_name: str
    message_credits_remaining: Decimal
    message_credits_monthly: Decimal
    message_credits_used_today: Decimal
    message_credits_daily_cap: Optional[Decimal]
    integration_credits_remaining: Decimal
    integration_credits_monthly: Decimal
    period_start: datetime
    period_end: datetime
    enforcement_enabled: bool


REASON_INSUFFICIENT_MESSAGE = "insufficient_message_credits"
REASON_INSUFFICIENT_INTEGRATION = "insufficient_integration_credits"
REASON_DAILY_CAP_EXCEEDED = "daily_cap_exceeded"
REASON_EMAIL_NOT_VERIFIED = "email_not_verified"


def _is_unlimited_user(user) -> bool:
    """True iff this user has NO usage limits — admins. Such users are never
    denied at the credit gate and never deducted (their balance row is left
    untouched), so an operator/owner account can use the platform freely.
    Centralized so try_charge, reserve, and the /status view all agree."""
    return user is not None and getattr(user, "role", None) == "admin"


def _email_verification_required(user) -> bool:
    """True iff this user must verify email before being charged. F13."""
    if not getattr(settings, "require_email_verification_for_credits", False):
        return False
    if user is None:
        return False
    if user.email_verified_at is not None:
        return False
    cutoff_iso = (getattr(settings, "email_verification_required_after_iso", "") or "").strip()
    if not cutoff_iso:
        return True
    try:
        cutoff = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00"))
        if cutoff.tzinfo is not None:
            cutoff = cutoff.replace(tzinfo=None)
        return user.created_at >= cutoff
    except Exception:
        return True  # malformed cutoff → fail safe (require verification)


def _local_day_iso(user_timezone: Optional[str], now: Optional[datetime] = None) -> str:
    tz = ZoneInfo(user_timezone) if user_timezone else ZoneInfo("UTC")
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    else:
        now = now.astimezone(tz)
    return now.date().isoformat()


def _bucket_remaining(balance: CreditBalance, bucket: str) -> Decimal:
    if bucket == BUCKET_MESSAGE:
        return Decimal(balance.message_credits_remaining)
    if bucket == BUCKET_INTEGRATION:
        return Decimal(balance.integration_credits_remaining)
    raise ValueError(f"unknown bucket {bucket!r}")


def _apply_delta(balance: CreditBalance, bucket: str, delta: Decimal) -> None:
    if bucket == BUCKET_MESSAGE:
        balance.message_credits_remaining = _q(
            Decimal(balance.message_credits_remaining) + delta, _BALANCE_QUANTUM,
        )
        if delta < 0:
            balance.message_credits_used_today = _q(
                Decimal(balance.message_credits_used_today) + (-delta), _BALANCE_QUANTUM,
            )
    elif bucket == BUCKET_INTEGRATION:
        balance.integration_credits_remaining = _q(
            Decimal(balance.integration_credits_remaining) + delta, _BALANCE_QUANTUM,
        )
    else:
        raise ValueError(f"unknown bucket {bucket!r}")


class CreditService:
    """Stateless. Pass an ``AsyncSession`` per call. Caller owns commit."""

    async def get_or_create_balance(self, db: AsyncSession, user_id: str) -> CreditBalance:
        balance = await db.get(CreditBalance, user_id)
        if balance is not None:
            return balance
        user = await db.get(User, user_id)
        plan = await db.get(SubscriptionPlan, "free")
        if plan is None:
            raise RuntimeError("subscription_plans 'free' row missing — alembic 053 not applied?")
        now = datetime.utcnow()
        balance = CreditBalance(
            user_id=user_id, plan_id=plan.id,
            message_credits_remaining=plan.message_credits_monthly,
            integration_credits_remaining=plan.integration_credits_monthly,
            message_credits_used_today=Decimal("0"),
            message_credits_daily_cap=plan.message_credits_daily_cap,
            day_anchor_local_date=_local_day_iso(user.timezone if user else None),
            period_start=now, period_end=now + timedelta(days=30),
        )
        db.add(balance)
        db.add(CreditLedger(
            user_id=user_id, event_type=LEDGER_PLAN_GRANT, bucket=BUCKET_MESSAGE,
            amount=plan.message_credits_monthly, balance_after=plan.message_credits_monthly,
            metadata_json={"plan_id": plan.id, "reason": "initial_grant"},
        ))
        db.add(CreditLedger(
            user_id=user_id, event_type=LEDGER_PLAN_GRANT, bucket=BUCKET_INTEGRATION,
            amount=plan.integration_credits_monthly, balance_after=plan.integration_credits_monthly,
            metadata_json={"plan_id": plan.id, "reason": "initial_grant"},
        ))
        await db.flush()
        return balance

    async def check_balance(
        self, db: AsyncSession, user_id: str, bucket: str,
        required: Decimal | float | int,
    ) -> ChargeResult:
        """Read-only "would this charge succeed?" peek. No ledger row written."""
        balance = await self.get_or_create_balance(db, user_id)
        required_q = _q(required, _AMOUNT_QUANTUM)
        enforcement = getattr(settings, "credit_enforcement_enabled", False)
        remaining = _bucket_remaining(balance, bucket)
        if not enforcement:
            return ChargeResult(success=True, balance_after=remaining)
        if required_q > remaining:
            return ChargeResult(
                success=False, balance_after=remaining,
                reason=(REASON_INSUFFICIENT_MESSAGE if bucket == BUCKET_MESSAGE
                        else REASON_INSUFFICIENT_INTEGRATION),
            )
        return ChargeResult(success=True, balance_after=remaining)

    async def get_balance_view(self, db: AsyncSession, user_id: str) -> BalanceView:
        balance = await self.get_or_create_balance(db, user_id)
        plan = await db.get(SubscriptionPlan, balance.plan_id)
        return BalanceView(
            plan_id=balance.plan_id,
            plan_display_name=plan.display_name if plan else balance.plan_id,
            message_credits_remaining=Decimal(balance.message_credits_remaining),
            message_credits_monthly=Decimal(plan.message_credits_monthly) if plan else Decimal("0"),
            message_credits_used_today=Decimal(balance.message_credits_used_today),
            message_credits_daily_cap=(
                Decimal(balance.message_credits_daily_cap)
                if balance.message_credits_daily_cap is not None else None
            ),
            integration_credits_remaining=Decimal(balance.integration_credits_remaining),
            integration_credits_monthly=Decimal(plan.integration_credits_monthly) if plan else Decimal("0"),
            period_start=balance.period_start, period_end=balance.period_end,
            enforcement_enabled=getattr(settings, "credit_enforcement_enabled", False),
        )

    async def try_charge(
        self, db: AsyncSession, user_id: str, event_type: str, bucket: str,
        amount: Decimal | float | int, *,
        idempotency_key: Optional[str] = None, event_id: Optional[str] = None,
        model: Optional[str] = None, provider: Optional[str] = None,
        input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
        underlying_cost_cents: Optional[Decimal | float | int] = None,
        metadata: Optional[dict] = None,
    ) -> ChargeResult:
        """Instant deduction. Atomic, idempotent.

        When ``settings.credit_enforcement_enabled=False`` always succeeds
        but still writes the ledger row (shadow mode).
        """
        amount_q = _q(amount, _AMOUNT_QUANTUM)
        if amount_q <= 0:
            raise ValueError(f"try_charge requires positive amount, got {amount}")

        balance = await self._lock_balance(db, user_id)
        await self._reset_daily_if_needed(db, balance, user_id)

        # Idempotency short-circuit.
        if idempotency_key:
            prior = await db.execute(
                select(CreditLedger).where(
                    CreditLedger.user_id == user_id,
                    CreditLedger.idempotency_key == idempotency_key,
                )
            )
            existing = prior.scalar_one_or_none()
            if existing is not None:
                return ChargeResult(
                    success=True, balance_after=Decimal(existing.balance_after),
                    ledger_id=existing.id, idempotent_hit=True,
                )

        # Email-verified gate (F13).
        deny_reason: Optional[str] = None
        user = await db.get(User, user_id)
        # Admin accounts are UNLIMITED — never denied, never deducted. The
        # balance row is left untouched (we still write a zero-amount ledger
        # row for an audit trail). Centralized via _is_unlimited so try_charge,
        # reserve, and the /status view all agree. See [[admin-unlimited-credits]].
        is_unlimited = _is_unlimited_user(user)
        if is_unlimited:
            deny_reason = None
        elif _email_verification_required(user):
            deny_reason = REASON_EMAIL_NOT_VERIFIED

        # Insufficient balance check.
        if deny_reason is not None or is_unlimited:
            pass
        elif bucket == BUCKET_MESSAGE:
            if amount_q > Decimal(balance.message_credits_remaining):
                deny_reason = REASON_INSUFFICIENT_MESSAGE
            elif balance.message_credits_daily_cap is not None:
                projected_today = Decimal(balance.message_credits_used_today) + amount_q
                if projected_today > Decimal(balance.message_credits_daily_cap):
                    deny_reason = REASON_DAILY_CAP_EXCEEDED
        elif bucket == BUCKET_INTEGRATION:
            if amount_q > Decimal(balance.integration_credits_remaining):
                deny_reason = REASON_INSUFFICIENT_INTEGRATION
        else:
            raise ValueError(f"unknown bucket {bucket!r}")

        enforcement = getattr(settings, "credit_enforcement_enabled", False)

        if deny_reason and enforcement:
            ledger = CreditLedger(
                user_id=user_id, event_type=event_type, bucket=bucket,
                amount=Decimal("0"), balance_after=_bucket_remaining(balance, bucket),
                idempotency_key=idempotency_key, event_id=event_id, model=model,
                provider=provider, input_tokens=input_tokens, output_tokens=output_tokens,
                underlying_cost_cents=(_q(underlying_cost_cents) if underlying_cost_cents is not None else None),
                metadata_json={"denied": True, "reason": deny_reason, **(metadata or {})},
            )
            try:
                db.add(ledger)
                await db.flush()
            except IntegrityError:
                await db.rollback()
                if idempotency_key:
                    prior = await db.execute(
                        select(CreditLedger).where(
                            CreditLedger.user_id == user_id,
                            CreditLedger.idempotency_key == idempotency_key,
                        )
                    )
                    existing = prior.scalar_one_or_none()
                    if existing:
                        return ChargeResult(
                            success=True, balance_after=Decimal(existing.balance_after),
                            ledger_id=existing.id, idempotent_hit=True,
                        )
                raise
            return ChargeResult(
                success=False, balance_after=_bucket_remaining(balance, bucket),
                reason=deny_reason, ledger_id=ledger.id,
            )

        # Unlimited (admin) charges never deduct — the row stays put.
        will_deduct = ((deny_reason is None) or (not enforcement)) and not is_unlimited
        if will_deduct:
            _apply_delta(balance, bucket, -amount_q)
            actual_amount = -amount_q
        else:
            actual_amount = Decimal("0")

        ledger = CreditLedger(
            user_id=user_id, event_type=event_type, bucket=bucket, amount=actual_amount,
            balance_after=_bucket_remaining(balance, bucket),
            idempotency_key=idempotency_key, event_id=event_id, model=model,
            provider=provider, input_tokens=input_tokens, output_tokens=output_tokens,
            underlying_cost_cents=(_q(underlying_cost_cents) if underlying_cost_cents is not None else None),
            metadata_json={
                **(metadata or {}),
                **({"shadow_would_deny": deny_reason} if (deny_reason and not enforcement) else {}),
                **({"admin_unlimited": True} if is_unlimited else {}),
            } or None,
        )
        try:
            db.add(ledger)
            await db.flush()
        except IntegrityError:
            await db.rollback()
            if idempotency_key:
                prior = await db.execute(
                    select(CreditLedger).where(
                        CreditLedger.user_id == user_id,
                        CreditLedger.idempotency_key == idempotency_key,
                    )
                )
                existing = prior.scalar_one_or_none()
                if existing:
                    return ChargeResult(
                        success=True, balance_after=Decimal(existing.balance_after),
                        ledger_id=existing.id, idempotent_hit=True,
                    )
            raise

        return ChargeResult(success=True, balance_after=Decimal(ledger.balance_after), ledger_id=ledger.id)

    async def reserve(
        self, db: AsyncSession, user_id: str, event_type: str, bucket: str,
        estimated_amount: Decimal | float | int, *, ttl_seconds: int = 1800,
        idempotency_key: Optional[str] = None, event_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ReservationResult:
        amount_q = _q(estimated_amount, _BALANCE_QUANTUM)
        if amount_q <= 0:
            raise ValueError("reserve requires positive estimated_amount")
        balance = await self._lock_balance(db, user_id)
        await self._reset_daily_if_needed(db, balance, user_id)

        if idempotency_key:
            existing = await db.execute(
                select(CreditReservation).where(
                    CreditReservation.user_id == user_id,
                    CreditReservation.idempotency_key == idempotency_key,
                )
            )
            row = existing.scalar_one_or_none()
            if row is not None:
                return ReservationResult(
                    reservation_id=row.id, estimated_amount=Decimal(row.estimated_amount),
                    balance_after=_bucket_remaining(balance, bucket), success=True,
                )

        enforcement = getattr(settings, "credit_enforcement_enabled", False)
        remaining = _bucket_remaining(balance, bucket)
        # Admins are UNLIMITED — never gated, never deducted. Reserve a
        # zero-amount hold so the row exists for settle() (which clamps actual
        # to estimated=0 → the whole reserve/settle cycle is a balance no-op).
        is_unlimited = _is_unlimited_user(await db.get(User, user_id))
        if enforcement and not is_unlimited and amount_q > remaining:
            return ReservationResult(
                reservation_id="", estimated_amount=amount_q, balance_after=remaining,
                success=False,
                reason=(REASON_INSUFFICIENT_MESSAGE if bucket == BUCKET_MESSAGE
                        else REASON_INSUFFICIENT_INTEGRATION),
            )

        deducted = Decimal("0") if is_unlimited else amount_q
        _apply_delta(balance, bucket, -deducted)
        now = datetime.utcnow()
        reservation_id = str(uuid.uuid4())
        db.add(CreditReservation(
            id=reservation_id, user_id=user_id, event_type=event_type, bucket=bucket,
            estimated_amount=deducted, status=RESERVATION_OPEN,
            idempotency_key=idempotency_key, event_id=event_id,
            metadata_json={**(metadata or {}), **({"admin_unlimited": True} if is_unlimited else {})} or None,
            expires_at=now + timedelta(seconds=ttl_seconds),
        ))
        db.add(CreditLedger(
            user_id=user_id, event_type=LEDGER_RESERVATION, bucket=bucket,
            amount=-deducted, balance_after=_bucket_remaining(balance, bucket),
            reservation_id=reservation_id, idempotency_key=idempotency_key,
            event_id=event_id, metadata_json={"action": "reserve", **(metadata or {}),
                                              **({"admin_unlimited": True} if is_unlimited else {})},
        ))
        await db.flush()
        return ReservationResult(
            reservation_id=reservation_id, estimated_amount=amount_q,
            balance_after=_bucket_remaining(balance, bucket),
        )

    async def settle(
        self, db: AsyncSession, reservation_id: str,
        actual_amount: Decimal | float | int, *, metadata: Optional[dict] = None,
    ) -> Decimal:
        reservation = await db.get(CreditReservation, reservation_id)
        if reservation is None:
            raise ValueError(f"reservation {reservation_id} not found")
        if reservation.status != RESERVATION_OPEN:
            return Decimal(reservation.settled_amount or 0)

        actual_q = _q(actual_amount, _AMOUNT_QUANTUM)
        estimated_q = Decimal(reservation.estimated_amount)
        if actual_q < 0:
            actual_q = Decimal("0")
        if actual_q > estimated_q:
            actual_q = estimated_q

        balance = await self._lock_balance(db, reservation.user_id)
        refund = estimated_q - actual_q
        if refund > 0:
            _apply_delta(balance, reservation.bucket, refund)
            db.add(CreditLedger(
                user_id=reservation.user_id, event_type=LEDGER_REFUND, bucket=reservation.bucket,
                amount=refund, balance_after=_bucket_remaining(balance, reservation.bucket),
                reservation_id=reservation_id,
                metadata_json={"reason": "settlement_overage", **(metadata or {})},
            ))
        db.add(CreditLedger(
            user_id=reservation.user_id, event_type=LEDGER_SETTLEMENT, bucket=reservation.bucket,
            amount=Decimal("0"), balance_after=_bucket_remaining(balance, reservation.bucket),
            reservation_id=reservation_id,
            metadata_json={"estimated": str(estimated_q), "actual": str(actual_q), **(metadata or {})},
        ))
        reservation.settled_amount = actual_q
        reservation.status = RESERVATION_SETTLED
        reservation.settled_at = datetime.utcnow()
        await db.flush()
        return actual_q

    async def refund(self, db: AsyncSession, reservation_id: str, *, reason: str = "manual_refund") -> None:
        reservation = await db.get(CreditReservation, reservation_id)
        if reservation is None or reservation.status != RESERVATION_OPEN:
            return
        balance = await self._lock_balance(db, reservation.user_id)
        amount = Decimal(reservation.estimated_amount)
        _apply_delta(balance, reservation.bucket, amount)
        db.add(CreditLedger(
            user_id=reservation.user_id, event_type=LEDGER_REFUND, bucket=reservation.bucket,
            amount=amount, balance_after=_bucket_remaining(balance, reservation.bucket),
            reservation_id=reservation_id, metadata_json={"reason": reason},
        ))
        reservation.status = RESERVATION_REFUNDED
        reservation.settled_at = datetime.utcnow()
        await db.flush()

    async def grant(
        self, db: AsyncSession, user_id: str, bucket: str,
        amount: Decimal | float | int, *,
        event_type: str = LEDGER_MANUAL_ADJUST, metadata: Optional[dict] = None,
    ) -> Decimal:
        amount_q = _q(amount, _AMOUNT_QUANTUM)
        balance = await self._lock_balance(db, user_id)
        _apply_delta(balance, bucket, amount_q)
        db.add(CreditLedger(
            user_id=user_id, event_type=event_type, bucket=bucket, amount=amount_q,
            balance_after=_bucket_remaining(balance, bucket), metadata_json=metadata,
        ))
        await db.flush()
        return _bucket_remaining(balance, bucket)

    async def apply_plan_change(
        self, db: AsyncSession, user_id: str, new_plan_id: str, *,
        reason: str = "stripe_subscription_updated",
    ) -> None:
        new_plan = await db.get(SubscriptionPlan, new_plan_id)
        if new_plan is None:
            raise ValueError(f"unknown plan {new_plan_id}")
        balance = await self._lock_balance(db, user_id)
        old_plan = await db.get(SubscriptionPlan, balance.plan_id)

        old_msg = Decimal(old_plan.message_credits_monthly) if old_plan else Decimal("0")
        old_int = Decimal(old_plan.integration_credits_monthly) if old_plan else Decimal("0")

        period_total = balance.period_end - balance.period_start
        remaining = balance.period_end - datetime.utcnow()
        if period_total.total_seconds() <= 0:
            ratio = Decimal("1")
        else:
            ratio = max(Decimal("0"), min(Decimal("1"),
                Decimal(str(remaining.total_seconds())) / Decimal(str(period_total.total_seconds()))))

        msg_delta = (Decimal(new_plan.message_credits_monthly) - old_msg) * ratio
        int_delta = (Decimal(new_plan.integration_credits_monthly) - old_int) * ratio

        if msg_delta != 0:
            _apply_delta(balance, BUCKET_MESSAGE, _q(msg_delta, _BALANCE_QUANTUM))
            db.add(CreditLedger(
                user_id=user_id, event_type="plan_change", bucket=BUCKET_MESSAGE,
                amount=_q(msg_delta, _AMOUNT_QUANTUM),
                balance_after=_bucket_remaining(balance, BUCKET_MESSAGE),
                metadata_json={"reason": reason, "from": balance.plan_id, "to": new_plan_id,
                               "proration_ratio": str(ratio)},
            ))
        if int_delta != 0:
            _apply_delta(balance, BUCKET_INTEGRATION, _q(int_delta, _BALANCE_QUANTUM))
            db.add(CreditLedger(
                user_id=user_id, event_type="plan_change", bucket=BUCKET_INTEGRATION,
                amount=_q(int_delta, _AMOUNT_QUANTUM),
                balance_after=_bucket_remaining(balance, BUCKET_INTEGRATION),
                metadata_json={"reason": reason, "from": balance.plan_id, "to": new_plan_id,
                               "proration_ratio": str(ratio)},
            ))
        balance.plan_id = new_plan_id
        balance.message_credits_daily_cap = new_plan.message_credits_daily_cap
        await db.flush()

    async def renew_period(self, db: AsyncSession, user_id: str) -> bool:
        balance = await self._lock_balance(db, user_id)
        now = datetime.utcnow()
        if now < balance.period_end:
            return False
        plan = await db.get(SubscriptionPlan, balance.plan_id)
        if plan is None:
            return False

        msg_carry = Decimal("0")
        int_carry = Decimal("0")
        if plan.rollover_message_credits:
            cap = Decimal(plan.message_credits_monthly) * (Decimal(plan.rollover_max_pct) / Decimal("100"))
            msg_carry = min(Decimal(balance.message_credits_remaining), cap)
        if plan.rollover_integration_credits:
            cap = Decimal(plan.integration_credits_monthly) * (Decimal(plan.rollover_max_pct) / Decimal("100"))
            int_carry = min(Decimal(balance.integration_credits_remaining), cap)

        balance.message_credits_remaining = _q(Decimal(plan.message_credits_monthly) + msg_carry, _BALANCE_QUANTUM)
        balance.integration_credits_remaining = _q(Decimal(plan.integration_credits_monthly) + int_carry, _BALANCE_QUANTUM)
        balance.message_credits_used_today = Decimal("0")
        balance.message_credits_daily_cap = plan.message_credits_daily_cap
        balance.period_start = balance.period_end
        balance.period_end = balance.period_end + timedelta(days=30)

        db.add(CreditLedger(
            user_id=user_id, event_type=LEDGER_PERIOD_RENEWAL, bucket=BUCKET_MESSAGE,
            amount=Decimal(plan.message_credits_monthly) + msg_carry,
            balance_after=balance.message_credits_remaining,
            metadata_json={"plan_id": plan.id, "rollover_amount": str(msg_carry)},
        ))
        db.add(CreditLedger(
            user_id=user_id, event_type=LEDGER_PERIOD_RENEWAL, bucket=BUCKET_INTEGRATION,
            amount=Decimal(plan.integration_credits_monthly) + int_carry,
            balance_after=balance.integration_credits_remaining,
            metadata_json={"plan_id": plan.id, "rollover_amount": str(int_carry)},
        ))
        await db.flush()
        return True

    async def _lock_balance(self, db: AsyncSession, user_id: str) -> CreditBalance:
        result = await db.execute(
            select(CreditBalance).where(CreditBalance.user_id == user_id).with_for_update()
        )
        balance = result.scalar_one_or_none()
        if balance is None:
            balance = await self.get_or_create_balance(db, user_id)
        return balance

    async def _reset_daily_if_needed(self, db: AsyncSession, balance: CreditBalance, user_id: str) -> None:
        user = await db.get(User, user_id)
        today_iso = _local_day_iso(user.timezone if user else None)
        anchor = balance.day_anchor_local_date
        # Reset ONLY when the local day moves FORWARD past the anchor.
        # The anchor can legitimately sit AHEAD of today_iso: at signup
        # the user's timezone is NULL, so the anchor is seeded from the
        # UTC date (_local_day_iso falls back to UTC); once the app later
        # learns a west-of-UTC timezone, today_iso can be the *previous*
        # calendar day. The old `!= today_iso` check then zeroed
        # used_today the moment the timezone landed — wiping a day's
        # counted spend mid-period and, worse, handing every user a
        # cap-evasion lever (move the clock/timezone backward → daily
        # counter resets → cap re-opens). ISO "YYYY-MM-DD" sorts
        # chronologically, so a lexical compare is a correct date compare.
        if anchor is not None and today_iso <= anchor:
            return
        was = Decimal(balance.message_credits_used_today)
        balance.message_credits_used_today = Decimal("0")
        balance.day_anchor_local_date = today_iso
        db.add(CreditLedger(
            user_id=user_id, event_type=LEDGER_DAILY_RESET, bucket=BUCKET_MESSAGE,
            amount=Decimal("0"), balance_after=Decimal(balance.message_credits_remaining),
            metadata_json={"prior_used_today": str(was), "new_anchor": today_iso},
        ))


credit_service = CreditService()
