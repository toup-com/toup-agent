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
    LEDGER_DAILY_RESET, LEDGER_IAP_PURCHASE, LEDGER_MANUAL_ADJUST,
    LEDGER_PERIOD_RENEWAL, LEDGER_PLAN_GRANT, LEDGER_REFUND, LEDGER_RESERVATION,
    LEDGER_SETTLEMENT,
    RESERVATION_OPEN, RESERVATION_REFUNDED, RESERVATION_SETTLED,
    APPLE_SUB_ACTIVE,
    AppleSubscription, CreditBalance, CreditLedger, CreditReservation, GrantEligibility,
    SubscriptionPlan, User,
)
from app.services.email_canonical import canonical_email_hash
from app.services.abuse_metrics import emit as _emit, hp as _hp, uidp as _uidp


# Plan-source markers stamped on credit_balances.plan_source. NULL/'stripe' ⇒
# time-driven renewal (today); 'apple' ⇒ webhook-driven renewal only.
PLAN_SOURCE_APPLE = "apple"
PLAN_SOURCE_STRIPE = "stripe"


logger = logging.getLogger(__name__)


_AMOUNT_QUANTUM = Decimal("0.0001")
_DISPLAY_QUANTUM = Decimal("0.1")
_BALANCE_QUANTUM = Decimal("0.01")


def _q(value: Decimal | float | int, quantum: Decimal = _AMOUNT_QUANTUM) -> Decimal:
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def underlying_cost_to_credits(cost_cents: float | int | Decimal) -> Decimal:
    """Convert underlying LLM cost (cents) to credits. Peg: 1c = 1 credit."""
    return _q(cost_cents, _DISPLAY_QUANTUM)


def image_generation_cost_cents(size: Optional[str], quality: Optional[str]) -> Decimal:
    """True OpenAI cost (in cents) for one gpt-image-1 image at (size, quality).

    Pulls from ``settings.image_gen_pricing_cents`` (keyed ``"<size>:<quality>"``)
    and falls back to ``settings.image_gen_fallback_cents`` for unknown combos.
    Both the bundle LLM proxy (which charges bundle users inline) and the
    agent's generate_image tool (which reports the charge for manual/BYO users)
    call this so the two paths never diverge on price.
    """
    size_n = (size or settings.image_gen_default_size or "1024x1024").strip().lower()
    qual_n = (quality or settings.image_gen_default_quality or "high").strip().lower()
    if qual_n in ("auto", "hd", "standard", ""):
        # Map dall-e / "auto" qualities onto our low/medium/high scale so the
        # price table always resolves. "auto" bills as high (gpt-image-1's auto
        # skews toward high); dall-e-3 "hd"->high, "standard"->medium.
        qual_n = {"hd": "high", "standard": "medium", "auto": "high", "": "high"}[qual_n]
    table = settings.image_gen_pricing_cents or {}
    cents = table.get(f"{size_n}:{qual_n}")
    if cents is None:
        cents = settings.image_gen_fallback_cents
    # Return underlying cost in CENTS (Decimal, 0.1c granularity). Callers pass
    # this as underlying_cost_cents and convert to credits via
    # underlying_cost_to_credits() (1c = 1 credit peg).
    return _q(cents, _DISPLAY_QUANTUM)


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


def tokens_to_credits_raw(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """``tokens_to_credits`` without the 0.1-credit floor.

    For per-call accumulation inside the agent loop (mission budget
    hard-stop, 2026-07-16): a 40-iteration run floored at 0.1/call
    would inflate the ledger by up to 4 credits. Deduction rows keep
    using the floored ``tokens_to_credits``."""
    pricing = settings.pricing_per_1k.get(model)
    if not pricing:
        pricing = {"input": 0.003, "output": 0.015}
    cost_usd = (
        (Decimal(str(input_tokens)) * Decimal(str(pricing["input"])) / Decimal("1000"))
        + (Decimal(str(output_tokens)) * Decimal(str(pricing["output"])) / Decimal("1000"))
    )
    return _q(cost_usd * Decimal("100"), _DISPLAY_QUANTUM)


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
    message_credits_remaining: Decimal  # spendable total = plan wallet + purchased wallet
    message_credits_monthly: Decimal
    message_credits_used_today: Decimal
    message_credits_daily_cap: Optional[Decimal]
    integration_credits_remaining: Decimal
    integration_credits_monthly: Decimal
    period_start: datetime
    period_end: datetime
    enforcement_enabled: bool
    purchased_credits_remaining: Decimal = Decimal("0")
    # Mobile contract (mig-063 §6): 'iap' (apple) | 'web' (stripe/NULL paid) |
    # 'free'. NOT the raw stored plan_source.
    plan_source: Optional[str] = None
    subscription_product_id: Optional[str] = None
    subscription_renews_at: Optional[datetime] = None


REASON_INSUFFICIENT_MESSAGE = "insufficient_message_credits"
REASON_INSUFFICIENT_INTEGRATION = "insufficient_integration_credits"
REASON_DAILY_CAP_EXCEEDED = "daily_cap_exceeded"
REASON_EMAIL_NOT_VERIFIED = "email_not_verified"


def _split_message_charge(
    plan_remaining: Decimal,
    purchased_remaining: Decimal,
    used_today: Decimal,
    daily_cap: Optional[Decimal],
    amount: Decimal,
) -> tuple[Decimal, Decimal, bool, Optional[str]]:
    """Decide how a MESSAGE-bucket charge splits across the two wallets.

    Plan credits are spent first (and are the ONLY thing the daily cap
    limits). Purchased (StoreKit IAP) credits cover any overflow and
    bypass the daily cap entirely.

    Returns ``(from_plan, from_purchased, feasible, reason)``.

    INVARIANT — when ``purchased_remaining == 0`` this reduces EXACTLY to
    the pre-IAP behaviour: ``from_plan = min(amount, plan_remaining,
    cap_room)``, ``from_purchased = 0``, feasible iff ``amount <=
    plan_remaining`` AND ``amount <= cap_room``, with insufficient-vs-cap
    reason precedence matching the old try_charge gate (insufficient
    checked first, then daily-cap).
    """
    cap_room = (
        None if daily_cap is None
        else max(Decimal("0"), daily_cap - used_today)
    )
    if cap_room is None:
        from_plan = min(amount, plan_remaining)
    else:
        from_plan = min(amount, plan_remaining, cap_room)
    if from_plan < 0:
        from_plan = Decimal("0")
    from_purchased = amount - from_plan
    feasible = from_purchased <= purchased_remaining
    reason: Optional[str] = None
    if not feasible:
        # Precedence mirrors the legacy gate: a true shortfall (no amount of
        # cap headroom would help) reports insufficient; otherwise it's the
        # daily cap that's blocking. With purchased_remaining == 0 this
        # collapses to: amount > plan_remaining → insufficient, else cap.
        if amount > plan_remaining + purchased_remaining:
            reason = REASON_INSUFFICIENT_MESSAGE
        else:
            reason = REASON_DAILY_CAP_EXCEEDED
    return from_plan, from_purchased, feasible, reason


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
        # Spendable MESSAGE total = plan wallet + non-expiring purchased wallet.
        return Decimal(balance.message_credits_remaining) + Decimal(
            getattr(balance, "purchased_credits_remaining", 0) or 0
        )
    if bucket == BUCKET_INTEGRATION:
        return Decimal(balance.integration_credits_remaining)
    raise ValueError(f"unknown bucket {bucket!r}")


def _apply_delta(
    balance: CreditBalance, bucket: str, delta: Decimal,
) -> Optional[tuple[Decimal, Decimal]]:
    """Mutate a wallet by ``delta``. Returns the (from_plan, from_purchased)
    split for MESSAGE-bucket charges (delta < 0); None otherwise.

    delta >= 0 on MESSAGE adds to the PLAN wallet only (grants / renewals /
    plan refunds). delta < 0 (a charge of ``-delta``) drains plan first then
    the purchased wallet per :func:`_split_message_charge`; only the plan
    portion advances the daily-used counter so purchased spend bypasses the
    cap. Feasibility is enforced by callers BEFORE this runs.
    """
    if bucket == BUCKET_MESSAGE:
        if delta >= 0:
            balance.message_credits_remaining = _q(
                Decimal(balance.message_credits_remaining) + delta, _BALANCE_QUANTUM,
            )
            return None
        amount = -delta
        from_plan, from_purchased, _feasible, _reason = _split_message_charge(
            Decimal(balance.message_credits_remaining),
            Decimal(getattr(balance, "purchased_credits_remaining", 0) or 0),
            Decimal(balance.message_credits_used_today),
            (Decimal(balance.message_credits_daily_cap)
             if balance.message_credits_daily_cap is not None else None),
            amount,
        )
        balance.message_credits_remaining = _q(
            Decimal(balance.message_credits_remaining) - from_plan, _BALANCE_QUANTUM,
        )
        balance.purchased_credits_remaining = _q(
            Decimal(getattr(balance, "purchased_credits_remaining", 0) or 0) - from_purchased,
            _BALANCE_QUANTUM,
        )
        # Only plan spend counts against the daily cap.
        balance.message_credits_used_today = _q(
            Decimal(balance.message_credits_used_today) + from_plan, _BALANCE_QUANTUM,
        )
        return from_plan, from_purchased
    elif bucket == BUCKET_INTEGRATION:
        balance.integration_credits_remaining = _q(
            Decimal(balance.integration_credits_remaining) + delta, _BALANCE_QUANTUM,
        )
        return None
    else:
        raise ValueError(f"unknown bucket {bucket!r}")


def _restore_message_refund(
    balance: CreditBalance, refund: Decimal, purchased_drained: Decimal,
) -> None:
    """Restore a MESSAGE-bucket reservation refund, purchased wallet FIRST.

    ``purchased_drained`` is how much of the original hold came out of the
    purchased wallet (from the reservation's stored split). We refund up to
    that amount back to purchased, the remainder to plan, and roll back the
    daily-used counter by the plan portion only — the exact inverse of how
    :func:`_apply_delta` charged it, so reserve+settle conserves credits and
    leaves the cap accounting consistent.
    """
    if refund <= 0:
        return
    pur_restore = min(refund, max(Decimal("0"), purchased_drained))
    plan_restore = refund - pur_restore
    balance.purchased_credits_remaining = _q(
        Decimal(getattr(balance, "purchased_credits_remaining", 0) or 0) + pur_restore,
        _BALANCE_QUANTUM,
    )
    if plan_restore > 0:
        balance.message_credits_remaining = _q(
            Decimal(balance.message_credits_remaining) + plan_restore, _BALANCE_QUANTUM,
        )
        balance.message_credits_used_today = _q(
            max(Decimal("0"), Decimal(balance.message_credits_used_today) - plan_restore),
            _BALANCE_QUANTUM,
        )


def _reservation_purchased_drained(reservation: CreditReservation) -> Decimal:
    """How much of a reservation's hold came from the purchased wallet.

    Reads the {"split": {"purchased": ...}} that reserve() stamped. Returns
    0 for pre-IAP reservations (no split key) and admin/zero holds — those
    refund entirely to plan, which is the pre-IAP behaviour.
    """
    meta = reservation.metadata_json or {}
    split = meta.get("split") if isinstance(meta, dict) else None
    if not isinstance(split, dict):
        return Decimal("0")
    try:
        return Decimal(str(split.get("purchased", "0")))
    except Exception:
        return Decimal("0")


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
        # Create the balance row with ZERO credits first; the one-time free
        # grant is applied (or suppressed) below. With all gates off this is
        # byte-for-byte equivalent to the legacy "seed with plan credits"
        # path — the grant fires immediately. The split exists so the grant
        # can be deduped (Phase 2) and deferred to email-verify (Phase 3)
        # WITHOUT ever leaving a user without a balance row (the invariant
        # the provision gate depends on).
        balance = CreditBalance(
            user_id=user_id, plan_id=plan.id,
            message_credits_remaining=Decimal("0"),
            integration_credits_remaining=Decimal("0"),
            message_credits_used_today=Decimal("0"),
            message_credits_daily_cap=plan.message_credits_daily_cap,
            day_anchor_local_date=_local_day_iso(user.timezone if user else None),
            period_start=now, period_end=now + timedelta(days=30),
        )
        db.add(balance)
        await db.flush()
        await self._maybe_grant_initial(db, balance, user, plan)
        return balance

    def _apply_initial_grant(
        self, db: AsyncSession, balance: CreditBalance, plan: SubscriptionPlan,
    ) -> None:
        """Seed the balance with the plan's monthly allotment + write the two
        ``initial_grant`` ledger rows. Pure local mutation; caller flushes."""
        balance.message_credits_remaining = plan.message_credits_monthly
        balance.integration_credits_remaining = plan.integration_credits_monthly
        db.add(CreditLedger(
            user_id=balance.user_id, event_type=LEDGER_PLAN_GRANT, bucket=BUCKET_MESSAGE,
            amount=plan.message_credits_monthly, balance_after=plan.message_credits_monthly,
            metadata_json={"plan_id": plan.id, "reason": "initial_grant"},
        ))
        db.add(CreditLedger(
            user_id=balance.user_id, event_type=LEDGER_PLAN_GRANT, bucket=BUCKET_INTEGRATION,
            amount=plan.integration_credits_monthly, balance_after=plan.integration_credits_monthly,
            metadata_json={"plan_id": plan.id, "reason": "initial_grant"},
        ))

    async def _claim_grant_tombstone(self, db: AsyncSession, user: Optional[User]) -> str:
        """Try to claim the one-time grant for this user's CANONICAL identity.

        Race-safe: the INSERT into grant_eligibility (PK = canonical hash)
        is wrapped in a SAVEPOINT so a collision rolls back ONLY this insert,
        never the in-flight user/balance creation. Two concurrent signups
        with the same canonical email → exactly one claims; the other reads
        the existing row.

        Returns one of:
          * "granted"    — we own the tombstone now → caller should grant.
          * "already"    — THIS user already owns it → idempotent, no re-grant.
          * "suppressed" — a DIFFERENT account already claimed it AND dedupe
                           is enabled → caller must NOT grant.
        With ``free_grant_dedupe_enabled`` OFF a cross-account collision
        returns "granted" (legacy behavior) but is logged as would-suppress.
        """
        if user is None or not getattr(user, "email", None):
            return "granted"  # no email identity to dedupe on → behave as before
        h = canonical_email_hash(user.email)
        uid = str(user.id)
        try:
            async with db.begin_nested():
                await db.execute(
                    GrantEligibility.__table__.insert().values(
                        canonical_email_hash=h, granted_at=datetime.utcnow(),
                        first_user_id=uid,
                    )
                )
            _emit("tombstone_claimed", hp=_hp(h), uid=_uidp(uid))
            return "granted"
        except IntegrityError:
            existing = await db.get(GrantEligibility, h)
            owner = existing.first_user_id if existing else None
            if owner == uid:
                return "already"
            if getattr(settings, "free_grant_dedupe_enabled", False):
                _emit("grant_suppressed", hp=_hp(h), uid=_uidp(uid), owner=_uidp(owner))
                return "suppressed"
            # Flag OFF → still grant (legacy), but record the SHADOW decision so
            # the suppression rate is measurable before enabling dedupe.
            _emit("grant_suppressed_would", hp=_hp(h), uid=_uidp(uid), owner=_uidp(owner))
            return "granted"

    async def _maybe_grant_initial(
        self, db: AsyncSession, balance: CreditBalance,
        user: Optional[User], plan: SubscriptionPlan,
    ) -> None:
        """Apply the one-time free grant at balance-creation time, unless:

        * Phase 3 verified-email gate is on AND the email is not yet
          verified → DEFER (no grant, NO tombstone claim — the grant fires
          later via ``grant_initial_free_credits`` when the user verifies /
          when OAuth stamps email_verified_at). The balance row stays at
          zero so the 'every user has a balance row' invariant holds.
        * Phase 2 dedupe says this canonical identity already claimed it →
          suppress.
        """
        if getattr(settings, "require_verified_email_for_grant", False) and not (
            user is not None and user.email_verified_at is not None
        ):
            _emit("grant_deferred", uid=_uidp(user.id if user else None))
            return
        claim = await self._claim_grant_tombstone(db, user)
        if claim in ("suppressed", "already"):
            return  # leave the zero-credit balance row in place
        self._apply_initial_grant(db, balance, plan)
        await db.flush()

    async def _already_granted(self, db: AsyncSession, user_id: str) -> bool:
        """True iff THIS user already owns a grant tombstone (i.e. its
        one-time grant already fired). Makes grant_initial_free_credits
        idempotent and a no-op when the grant happened at signup."""
        row = await db.execute(
            select(GrantEligibility.canonical_email_hash).where(
                GrantEligibility.first_user_id == str(user_id)
            )
        )
        return row.first() is not None

    async def grant_initial_free_credits(
        self, db: AsyncSession, user_id: str, *, email_verified: bool = False,
    ) -> bool:
        """Apply the DEFERRED one-time free grant once the email is verified.

        Called from the email-verify endpoint and from the OAuth/Apple
        handlers (provider-verified). Idempotent and safe to call from
        multiple channels: returns False without re-granting if this user
        already received its grant, if the grant is still gated unverified,
        or if the canonical identity was already claimed by another account.
        Returns True iff it granted on THIS call. Caller owns commit.
        """
        balance = await self.get_or_create_balance(db, user_id)
        if await self._already_granted(db, user_id):
            return False
        user = await db.get(User, user_id)
        if getattr(settings, "require_verified_email_for_grant", False):
            verified = email_verified or (
                user is not None and user.email_verified_at is not None
            )
            if not verified:
                return False  # still deferred
        plan = await db.get(SubscriptionPlan, balance.plan_id)
        if plan is None:
            return False
        claim = await self._claim_grant_tombstone(db, user)
        if claim in ("suppressed", "already"):
            return False
        self._apply_initial_grant(db, balance, plan)
        await db.flush()
        return True

    async def reconcile_deferred_grants(self, db: AsyncSession, *, limit: int = 500) -> int:
        """Grant anyone left in a deferred-but-now-eligible state.

        Closes the gaps the per-request path can miss so no user is ever
        permanently grant-less due to timing:
          * verified during a window where the on-verify grant call failed;
          * signed up while ``require_verified_email_for_grant`` was ON, then
            the flag was turned OFF (nothing re-triggers their grant).

        Candidates = balances with NO owned tombstone (i.e. never granted).
        Each is re-run through ``grant_initial_free_credits``, which grants
        only if eligible and still suppresses aliases / unverified-and-gated
        users. Idempotent and safe to run on a schedule. Per-candidate commit
        isolates failures. Returns the number actually granted.
        """
        rows = await db.execute(
            select(CreditBalance.user_id)
            .outerjoin(
                GrantEligibility,
                GrantEligibility.first_user_id == CreditBalance.user_id,
            )
            .where(GrantEligibility.canonical_email_hash.is_(None))
            .limit(limit)
        )
        candidate_ids = [r[0] for r in rows.all()]
        granted = 0
        for uid in candidate_ids:
            try:
                user = await db.get(User, uid)
                verified = user is not None and user.email_verified_at is not None
                if await self.grant_initial_free_credits(db, uid, email_verified=verified):
                    await db.commit()
                    granted += 1
                else:
                    await db.rollback()
            except Exception as e:  # one bad row must not abort the sweep
                await db.rollback()
                logger.warning(
                    "[credits] reconcile grant failed user=%s: %s", str(uid)[:8], e,
                )
        logger.info(
            "[credits] reconcile_deferred_grants: %d candidates, %d granted",
            len(candidate_ids), granted,
        )
        return granted

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

        # Map the stored plan_source to the mobile contract:
        #   apple            → 'iap'
        #   stripe/NULL paid → 'web'
        #   free             → 'free'
        raw_source = getattr(balance, "plan_source", None)
        if balance.plan_id == "free":
            mapped_source: Optional[str] = "free"
        elif raw_source == PLAN_SOURCE_APPLE:
            mapped_source = "iap"
        else:  # 'stripe' or NULL (legacy paid) → web
            mapped_source = "web"

        sub_product_id: Optional[str] = None
        sub_renews_at: Optional[datetime] = None
        if raw_source == PLAN_SOURCE_APPLE:
            apple_sub = (await db.execute(
                select(AppleSubscription)
                .where(AppleSubscription.user_id == user_id)
                .where(AppleSubscription.status == APPLE_SUB_ACTIVE)
                .order_by(AppleSubscription.updated_at.desc())
            )).scalars().first()
            if apple_sub is not None:
                sub_product_id = apple_sub.product_id
                sub_renews_at = apple_sub.expires_date

        return BalanceView(
            plan_id=balance.plan_id,
            plan_display_name=plan.display_name if plan else balance.plan_id,
            # Spendable MESSAGE total = plan wallet + purchased wallet, so
            # /status reports what the user can actually spend.
            message_credits_remaining=_bucket_remaining(balance, BUCKET_MESSAGE),
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
            purchased_credits_remaining=Decimal(
                getattr(balance, "purchased_credits_remaining", 0) or 0
            ),
            plan_source=mapped_source,
            subscription_product_id=sub_product_id,
            subscription_renews_at=sub_renews_at,
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
            # _split_message_charge re-derives feasibility + reason from the
            # SAME state _apply_delta will read inside this locked txn, so the
            # gate and the application below agree exactly. With no purchased
            # credits this is byte-for-byte the legacy insufficient-then-cap
            # gate.
            _fp, _fpur, _feasible, _split_reason = _split_message_charge(
                Decimal(balance.message_credits_remaining),
                Decimal(getattr(balance, "purchased_credits_remaining", 0) or 0),
                Decimal(balance.message_credits_used_today),
                (Decimal(balance.message_credits_daily_cap)
                 if balance.message_credits_daily_cap is not None else None),
                amount_q,
            )
            if not _feasible:
                deny_reason = _split_reason
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
        split = _apply_delta(balance, bucket, -deducted)
        now = datetime.utcnow()
        reservation_id = str(uuid.uuid4())
        # For MESSAGE reservations, persist how the hold split across the
        # plan vs purchased wallets so settle()/refund() can restore the
        # purchased portion first (conserving the user's paid credits).
        split_meta = (
            {"split": {"plan": str(split[0]), "purchased": str(split[1])}}
            if split is not None else {}
        )
        db.add(CreditReservation(
            id=reservation_id, user_id=user_id, event_type=event_type, bucket=bucket,
            estimated_amount=deducted, status=RESERVATION_OPEN,
            idempotency_key=idempotency_key, event_id=event_id,
            metadata_json={**(metadata or {}), **split_meta,
                           **({"admin_unlimited": True} if is_unlimited else {})} or None,
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
            if reservation.bucket == BUCKET_MESSAGE:
                # Restore the purchased portion first (up to what was drained
                # from the purchased wallet), remainder to plan.
                _restore_message_refund(
                    balance, refund, _reservation_purchased_drained(reservation),
                )
            else:
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
        if reservation.bucket == BUCKET_MESSAGE:
            _restore_message_refund(
                balance, amount, _reservation_purchased_drained(reservation),
            )
        else:
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

    async def grant_purchased(
        self, db: AsyncSession, user_id: str, amount: Decimal | float | int, *,
        event_type: str = LEDGER_IAP_PURCHASE, idempotency_key: str,
        metadata: Optional[dict] = None,
    ) -> tuple[Decimal, bool]:
        """Mint non-expiring purchased credits into the MESSAGE wallet.

        Idempotent on ``(user_id, idempotency_key)`` — keyed on the StoreKit
        transaction id so a replay returns the SAME balance without granting
        again. Returns ``(message_bucket_remaining_after, already_redeemed)``
        where the balance is the spendable MESSAGE total (plan + purchased).

        The amount lands in the purchased wallet only; the ledger row is
        written against the MESSAGE bucket with balance_after = the spendable
        total, matching how every other MESSAGE row is recorded.
        """
        amount_q = _q(amount, _BALANCE_QUANTUM)
        if amount_q <= 0:
            raise ValueError("grant_purchased requires positive amount")
        balance = await self._lock_balance(db, user_id)

        # Idempotency short-circuit: a prior grant with this key wins.
        prior = await db.execute(
            select(CreditLedger).where(
                CreditLedger.user_id == user_id,
                CreditLedger.idempotency_key == idempotency_key,
            )
        )
        existing = prior.scalar_one_or_none()
        if existing is not None:
            return Decimal(existing.balance_after), True

        balance.purchased_credits_remaining = _q(
            Decimal(getattr(balance, "purchased_credits_remaining", 0) or 0) + amount_q,
            _BALANCE_QUANTUM,
        )
        ledger = CreditLedger(
            user_id=user_id, event_type=event_type, bucket=BUCKET_MESSAGE,
            amount=amount_q, balance_after=_bucket_remaining(balance, BUCKET_MESSAGE),
            idempotency_key=idempotency_key, metadata_json=metadata,
        )
        try:
            db.add(ledger)
            await db.flush()
        except IntegrityError:
            # Concurrent replay raced us to the unique (user_id, key) index.
            await db.rollback()
            balance = await self._lock_balance(db, user_id)
            prior = await db.execute(
                select(CreditLedger).where(
                    CreditLedger.user_id == user_id,
                    CreditLedger.idempotency_key == idempotency_key,
                )
            )
            existing = prior.scalar_one_or_none()
            if existing is not None:
                return Decimal(existing.balance_after), True
            raise
        return _bucket_remaining(balance, BUCKET_MESSAGE), False

    async def clawback_purchased(
        self, db: AsyncSession, user_id: str, amount: Decimal | float | int, *,
        idempotency_key: str, metadata: Optional[dict] = None,
    ) -> Decimal:
        """Reverse a purchased-credit grant (refund / chargeback).

        Subtracts from the purchased wallet, FLOORED at 0 so a user who has
        already spent the credits can't go negative. Idempotent on its own
        ``idempotency_key`` (use ``<transaction_id>:refund``). Returns the
        spendable MESSAGE total after the clawback.
        """
        amount_q = _q(amount, _BALANCE_QUANTUM)
        balance = await self._lock_balance(db, user_id)

        prior = await db.execute(
            select(CreditLedger).where(
                CreditLedger.user_id == user_id,
                CreditLedger.idempotency_key == idempotency_key,
            )
        )
        if prior.scalar_one_or_none() is not None:
            return _bucket_remaining(balance, BUCKET_MESSAGE)

        current = Decimal(getattr(balance, "purchased_credits_remaining", 0) or 0)
        removed = min(current, max(Decimal("0"), amount_q))
        balance.purchased_credits_remaining = _q(current - removed, _BALANCE_QUANTUM)
        ledger = CreditLedger(
            user_id=user_id, event_type=LEDGER_REFUND, bucket=BUCKET_MESSAGE,
            amount=-removed, balance_after=_bucket_remaining(balance, BUCKET_MESSAGE),
            idempotency_key=idempotency_key,
            metadata_json={"reason": "iap_refund", "requested": str(amount_q),
                           "clawed_back": str(removed), **(metadata or {})},
        )
        try:
            db.add(ledger)
            await db.flush()
        except IntegrityError:
            await db.rollback()
            balance = await self._lock_balance(db, user_id)
            return _bucket_remaining(balance, BUCKET_MESSAGE)
        return _bucket_remaining(balance, BUCKET_MESSAGE)

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

    # ── Apple subscription reconciliation + lifecycle ────────────────────

    async def active_paid_source(
        self, db: AsyncSession, user_id: str,
    ) -> Optional[str]:
        """Return ``'stripe'`` | ``'apple'`` | ``None`` — the source of the
        user's CURRENT paid (non-free) plan, or None when on the free tier.

        Reads ``plan_source`` only when ``plan_id != 'free'``. The ``or
        'stripe'`` fallback is load-bearing: a pre-migration paid Stripe user
        has ``plan_source = NULL`` and must read as 'stripe' for the
        RECONCILIATION exclusivity check — while still being treated as NULL
        ("time-driven renewal OK") by the renewal guard. The two reads of NULL
        are deliberately different and both correct (mig-063 §3).
        """
        balance = await self._lock_balance(db, user_id)
        if balance.plan_id == "free":
            return None
        return balance.plan_source or PLAN_SOURCE_STRIPE

    async def activate_subscription(
        self, db: AsyncSession, user_id: str, plan_id: str, source: str,
        period_end: datetime,
    ) -> None:
        """First/again paid, or an immediate upgrade — stamp the plan + source
        + the externally-driven period window.

        Reuses the proration-aware tier math in :meth:`apply_plan_change` (the
        same entrypoint Stripe uses), so free→starter grants the full monthly
        allotment at ratio≈1.0 and a mid-period pro→elite upgrade prorates
        correctly. The ONLY additions are stamping ``plan_source`` and
        overriding the period window to the external (Apple) ``period_end`` so
        the renewal guard's clock matches Apple's. Idempotency is the caller's
        responsibility (per-original_transaction_id on verify, per-
        notificationUUID on notifications) since ``apply_plan_change`` is not
        idempotent.
        """
        await self.apply_plan_change(db, user_id, plan_id, reason=f"{source}:activate")
        balance = await self._lock_balance(db, user_id)
        balance.plan_source = source
        balance.period_start = datetime.utcnow()
        balance.period_end = period_end
        await db.flush()

    async def _apple_renew(
        self, db: AsyncSession, user_id: str, plan_id: str, period_end: datetime,
    ) -> None:
        """Apple DID_RENEW: re-grant the monthly allotment and bump the period
        to Apple's new ``expiresDate``.

        If the renewed product differs from the stored plan (a pending
        downgrade landing at renewal) apply the plan change first, then run the
        shared unguarded renewal body, then override the period window to
        Apple's clock. Only ever invoked from the notificationUUID-deduped
        handler, so the non-idempotent calls below are contained.
        """
        balance = await self._lock_balance(db, user_id)
        if balance.plan_id != plan_id:
            await self.apply_plan_change(db, user_id, plan_id, reason="apple:renew_plan_change")
            balance = await self._lock_balance(db, user_id)
        # Force the renewal grant even if the clock hasn't caught up to the old
        # period_end yet (Apple's DID_RENEW is authoritative for the renewal).
        balance.period_end = datetime.utcnow()
        await self._renew_period_unguarded(db, user_id, balance=balance)
        balance.plan_source = PLAN_SOURCE_APPLE
        balance.period_start = datetime.utcnow()
        balance.period_end = period_end
        await db.flush()

    async def downgrade_to_free(
        self, db: AsyncSession, user_id: str, reason: str,
    ) -> None:
        """Apple lapse/refund/revoke: return the user to the free tier.

        Setting ``plan_source`` back to NULL returns the row to the default
        (free) regime — future behaviour is identical to a never-subscribed
        user. Mirrors the Stripe→free path's OUTCOME. The non-expiring
        purchased (consumable) wallet is untouched — bought credits never
        expire.
        """
        await self.apply_plan_change(db, user_id, "free", reason=reason)
        balance = await self._lock_balance(db, user_id)
        balance.plan_source = None
        await db.flush()

    async def renew_period(self, db: AsyncSession, user_id: str) -> bool:
        """Time-driven monthly renewal (the hourly cron's entrypoint).

        Guards the Apple regime: an ``plan_source == 'apple'`` balance renews
        ONLY on Apple's DID_RENEW notification (via :meth:`_apple_renew`), never
        on the clock — a clock-driven renewal here would grant an unpaid month.
        For NULL/'stripe' rows this is byte-for-byte identical to before: the
        guard falls through and the shared :meth:`_renew_period_unguarded` body
        runs exactly as it did. The guard lives in EXACTLY one place; both this
        path (after the guard) and ``_apple_renew`` call the unguarded body so
        the grant logic is shared.
        """
        balance = await self._lock_balance(db, user_id)
        if (balance.plan_source or None) == PLAN_SOURCE_APPLE:
            # Apple subs renew ONLY on DID_RENEW. The Apple notification handler
            # calls _apple_renew() → _renew_period_unguarded() directly.
            return False
        return await self._renew_period_unguarded(db, user_id, balance=balance)

    async def _renew_period_unguarded(
        self, db: AsyncSession, user_id: str, *,
        balance: Optional[CreditBalance] = None,
    ) -> bool:
        """The renewal grant body, with NO plan_source guard.

        Carries the exact pre-Apple semantics: re-grant the monthly allotment
        (plus capped rollover), reset the daily counter, advance the period by
        30 days. Callers that have already locked the balance pass it in to
        avoid a second lock; ``_apple_renew`` overrides the period window after
        this returns so the renewal lands on Apple's expiresDate.
        """
        if balance is None:
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
        # purchased_credits_remaining is intentionally NOT reset — bought
        # credits never expire and carry across every monthly renewal.
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
