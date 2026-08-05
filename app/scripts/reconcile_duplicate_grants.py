"""Repair balances inflated by the duplicate free-grant loop (2026-08-03).

Background
==========
``reconcile_deferred_grants`` runs hourly and selected every balance with no
``grant_eligibility`` row owned by that user. That tombstone is keyed by
CANONICAL EMAIL HASH, so a user whose hash was claimed by an earlier account of
the same person can never own one: ``_claim_grant_tombstone`` hits the
IntegrityError branch and, with ``free_grant_dedupe_enabled`` False, returns
"granted" WITHOUT writing anything. ``_apply_initial_grant`` then ASSIGNS
``message_credits_remaining = plan.message_credits_monthly`` rather than adding
— so every sweep silently reset the wallet to full and erased the period's
spend, while ``message_credits_used_today`` kept climbing.

The code defect is fixed (``_already_granted`` now consults the ledger, and the
sweep excludes users who already hold a ``plan_grant`` row). This script repairs
the DATA the loop already corrupted; the fix alone stops the bleeding but does
not give back the credits the resets handed out.

Method
======
For each affected user, the truthful balance is:

    plan allowance for the period  −  charges actually made since period_start

Charges are read from ``credit_ledger`` (immutable, and the resets never touched
it — which is why the correct answer is recoverable at all). Only NEGATIVE
amounts in the period count; ``plan_grant`` / ``period_renewal`` rows are the
resets themselves and are deliberately ignored.

A user is "affected" iff they hold MORE THAN ONE ``plan_grant`` row per bucket:
one is the legitimate signup grant, the rest are the loop.

Safety
======
* DRY RUN by default. Pass ``--apply`` to write.
* Never RAISES a balance — if the computed value is above what the user
  currently holds, the row is left alone and reported. This script exists to
  take back credits that were minted in error, never to hand out more.
* Writes a ``plan_change`` ledger row per correction so the adjustment is
  auditable and the user's history explains itself.
* Idempotent: a second run finds nothing to do.

Run:
    python -m app.scripts.reconcile_duplicate_grants            # dry run
    python -m app.scripts.reconcile_duplicate_grants --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from decimal import Decimal

from sqlalchemy import func, select

from app.db.database import async_session_maker
from app.db.models import (
    LEDGER_PLAN_CHANGE, LEDGER_PLAN_GRANT,
    BUCKET_INTEGRATION, BUCKET_MESSAGE,
    CreditBalance, CreditLedger, SubscriptionPlan, User,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _duplicate_grant_users(db) -> list[str]:
    """Users holding more than one message-bucket plan_grant — i.e. the loop
    fired for them at least once beyond the legitimate signup grant."""
    rows = await db.execute(
        select(CreditLedger.user_id)
        .where(
            CreditLedger.event_type == LEDGER_PLAN_GRANT,
            CreditLedger.bucket == BUCKET_MESSAGE,
        )
        .group_by(CreditLedger.user_id)
        .having(func.count() > 1)
    )
    return [r[0] for r in rows.all()]


async def _spent_since(db, user_id: str, bucket: str, since) -> Decimal:
    """Total charged in this bucket since `since`. Only negative amounts —
    positives are grants/renewals, which are what we are correcting for."""
    total = (await db.execute(
        select(func.coalesce(func.sum(CreditLedger.amount), 0)).where(
            CreditLedger.user_id == user_id,
            CreditLedger.bucket == bucket,
            CreditLedger.amount < 0,
            CreditLedger.created_at >= since,
        )
    )).scalar()
    return -Decimal(str(total or 0))


async def reconcile(apply: bool = False) -> dict:
    counters = {"scanned": 0, "corrected": 0, "already_correct": 0, "skipped_would_raise": 0}
    async with async_session_maker() as db:
        user_ids = await _duplicate_grant_users(db)
        logger.info("[reconcile] %d user(s) with duplicate plan grants", len(user_ids))

        for uid in user_ids:
            counters["scanned"] += 1
            balance = await db.get(CreditBalance, uid)
            if balance is None:
                continue
            plan = await db.get(SubscriptionPlan, balance.plan_id)
            if plan is None:
                logger.warning("[reconcile] user=%s has unknown plan %s", uid[:8], balance.plan_id)
                continue
            user = await db.get(User, uid)
            email = getattr(user, "email", "?")

            msg_spent = await _spent_since(db, uid, BUCKET_MESSAGE, balance.period_start)
            int_spent = await _spent_since(db, uid, BUCKET_INTEGRATION, balance.period_start)
            msg_target = Decimal(plan.message_credits_monthly) - msg_spent
            int_target = Decimal(plan.integration_credits_monthly) - int_spent

            msg_now = Decimal(balance.message_credits_remaining)
            int_now = Decimal(balance.integration_credits_remaining)

            # Only ever take back. A computed value ABOVE what they hold means
            # something this script does not model (a refund, a plan change, a
            # purchase) — report it and leave the row alone rather than
            # silently minting credits.
            if msg_target > msg_now or int_target > int_now:
                counters["skipped_would_raise"] += 1
                logger.warning(
                    "[reconcile] SKIP %s — computed balance is HIGHER than held "
                    "(msg %s→%s, int %s→%s); needs a human",
                    email, msg_now, msg_target, int_now, int_target,
                )
                continue

            if msg_target == msg_now and int_target == int_now:
                counters["already_correct"] += 1
                continue

            logger.info(
                "[reconcile] %s  msg %s → %s (spent %s)   int %s → %s (spent %s)%s",
                email, msg_now, msg_target, msg_spent,
                int_now, int_target, int_spent,
                "" if apply else "   [DRY RUN]",
            )
            if not apply:
                continue

            for bucket, now, target in (
                (BUCKET_MESSAGE, msg_now, msg_target),
                (BUCKET_INTEGRATION, int_now, int_target),
            ):
                if target == now:
                    continue
                db.add(CreditLedger(
                    id=str(uuid.uuid4()),
                    user_id=uid,
                    event_type=LEDGER_PLAN_CHANGE,
                    bucket=bucket,
                    amount=(target - now),
                    balance_after=target,
                    metadata_json={
                        "reason": "duplicate_grant_reconciliation",
                        "incident": "2026-08-03",
                        "held_before": str(now),
                    },
                ))
            balance.message_credits_remaining = msg_target
            balance.integration_credits_remaining = int_target
            await db.commit()
            counters["corrected"] += 1

    logger.info("[reconcile] %s", counters)
    return counters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="write the corrections (default is a dry run that changes nothing)",
    )
    args = ap.parse_args()
    asyncio.run(reconcile(apply=args.apply))


if __name__ == "__main__":
    main()
