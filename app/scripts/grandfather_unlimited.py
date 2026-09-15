"""Move the legacy Apple payers onto UNLIMITED at their old price.

    python -m app.scripts.grandfather_unlimited                 # DRY RUN
    python -m app.scripts.grandfather_unlimited --apply
    python -m app.scripts.grandfather_unlimited --revert receipt.json

This is the FIRST production mutation of the Unlimited work, and it is a
command rather than a migration on purpose (design §9.3). The four reasons,
short:

  1. A migration runs unattended, in every environment, with no operator
     present. This mutation deserves a human reading three names first.
  2. Merging platform ``main`` is a deploy AND a fleet rollout. A migration
     would flip the payers' plans at the exact moment every tenant container
     restarts — two irreversible things in one window.
  3. The predicate is a RUNTIME question (``expires_date > now()``), not a
     schema one. Re-run against CI or a fresh environment it would either
     find nothing or grandfather fixtures.
  4. Reversibility needs a receipt of prior values, and writing a receipt
     makes it a command wearing a migration's clothes.

WHAT IT SELECTS — ``apple_subscriptions`` WHERE ``environment='Production'``
AND ``product_id`` in the four legacy ids AND ``expires_date > now()``. Never
``status``: the Sandbox Elite row has read ``status='active'`` for 2.7 months
past its expiry. Never ``plan_id <> 'free'``: that would sweep in
``nariman@toup.ai``'s alembic-054 Builder grant. Never ``plan_source``. The
whole selection is ``entitlement.select_legacy_apple_payers`` — one query,
shared with the reconciler, so the two can never disagree about who is a
payer.

WHAT IT SETS, per user, in one transaction:

  * ``credit_balances``: plan ``unlimited``, wallets ASSIGNED to
    1,000,000 / 1,000,000 (``apply_plan_change``'s absolute arm — proration
    across a gap that size drives ``purchased_credits_remaining`` to roughly
    −670,000), ``used_today`` 0, daily cap NULL, ``plan_source='apple'``.
  * ``purchased_credits_remaining``: UNTOUCHED. Bought credits never expire.
  * ``period_start`` / ``period_end``: UNTOUCHED. Apple owns that clock.
  * ``apple_subscriptions``: ``grandfathered_at=now``,
    ``grandfathered_product_id=<the product they hold right now>``.

That last column is what makes the grandfather survive 2026-10-01. On
``DID_RENEW`` the backend re-derives the plan from the PRODUCT id; without the
stamp, ``plan_for_subscription`` returns ``builder``/``starter`` and the
renewal silently reverts the grandfather. The command refuses to run at all
if the resolver deployed here does not honour the stamp — see
``_check_resolver``.

The receipt is written to stdout, to ``--receipt`` (default
``grandfather_unlimited_receipt.json``), and to ``platform_settings`` under
``billing.grandfather_unlimited_receipt``. Its ``original_transaction_id``
list is also the allowlist the reconciler's ``apple-legacy-crossgrade`` alert
reads: a Production row on a legacy product that is NOT in the receipt is a
post-cutover cross-grade down to $9.90, and pages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional

from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    BUCKET_INTEGRATION,
    BUCKET_MESSAGE,
    LEDGER_PLAN_CHANGE,
    AppleSubscription,
    CreditBalance,
    CreditLedger,
    PlatformSetting,
    SubscriptionPlan,
    User,
)
from app.db.plan_catalog import LEGACY_SUB_PRODUCT_IDS, UNLIMITED_PLAN_ID
from app.services import entitlement
from app.services.apple_iap_service import plan_for_subscription
from app.services.credit_service import PLAN_SOURCE_APPLE, credit_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grandfather_unlimited")

RECEIPT_SETTING_KEY = "billing.grandfather_unlimited_receipt"
GRANDFATHER_REASON = "grandfather:legacy_apple"
RECEIPT_VERSION = 1


# ── preconditions ────────────────────────────────────────────────────────


async def _check_plan_row(db) -> None:
    plan = await db.get(SubscriptionPlan, UNLIMITED_PLAN_ID)
    if plan is None:
        raise SystemExit(
            f"✗ subscription_plans has no '{UNLIMITED_PLAN_ID}' row — alembic 104 "
            f"has not been applied to this database. Refusing."
        )
    logger.info(
        "precondition ok: plan '%s' present (price=%s msg=%s int=%s)",
        plan.id, plan.price_cents, plan.message_credits_monthly,
        plan.integration_credits_monthly,
    )


def _check_resolver() -> None:
    """The deploy running here must resolve a grandfathered legacy product to
    Unlimited — otherwise the next DID_RENEW quietly reverts everything this
    command is about to do, and nothing would report it.
    """
    legacy = "ai.toup.app.sub.builder"
    marked = SimpleNamespace(grandfathered_at=datetime.utcnow())
    plain = SimpleNamespace(grandfathered_at=None)
    if plan_for_subscription(legacy, marked) != UNLIMITED_PLAN_ID:
        raise SystemExit(
            "✗ plan_for_subscription() does not honour grandfathered_at here — "
            "the resolver deploy is not live. Grandfathering now would be "
            "reverted at the next DID_RENEW. Refusing."
        )
    if plan_for_subscription(legacy, plain) == UNLIMITED_PLAN_ID:
        raise SystemExit(
            "✗ plan_for_subscription() returns 'unlimited' for a legacy product "
            "with NO grandfather stamp — the product map has been overloaded. "
            "That hands Unlimited to every future holder of a legacy id, "
            "including a cross-grade down to $9.90 Starter. Refusing."
        )
    crossgraded = SimpleNamespace(
        grandfathered_at=datetime.utcnow(),
        grandfathered_product_id="ai.toup.app.sub.builder",
    )
    if plan_for_subscription("ai.toup.app.sub.starter", crossgraded) == UNLIMITED_PLAN_ID:
        raise SystemExit(
            "✗ plan_for_subscription() keeps Unlimited across a PRODUCT CHANGE "
            "— the grandfather would follow a cross-grade down to $9.90 "
            "Starter from iOS Settings, permanently and unlogged. The four "
            "legacy products can never be taken off sale, so that route stays "
            "open forever. Refusing."
        )
    logger.info("precondition ok: resolver honours grandfathered_at per PRODUCT")


# ── the plan ─────────────────────────────────────────────────────────────


def _entry(cand) -> dict[str, Any]:
    """The BEFORE state, in the shape --revert restores from."""
    b, s, u = cand.balance, cand.sub, cand.user
    return {
        "user_id": u.id,
        "email": u.email,
        "original_transaction_id": s.original_transaction_id,
        "product_id": s.product_id,
        "expires_date": s.expires_date.isoformat() if s.expires_date else None,
        # Printed for the operator's read, and recorded so the receipt says
        # what Apple's mirror claimed at the moment of the grant.
        "status": s.status,
        "auto_renew_status": bool(s.auto_renew_status),
        "prior_plan_id": b.plan_id,
        "prior_plan_source": b.plan_source,
        "prior_message_credits_remaining": str(b.message_credits_remaining),
        "prior_integration_credits_remaining": str(b.integration_credits_remaining),
        "prior_message_credits_used_today": str(b.message_credits_used_today),
        "prior_message_credits_daily_cap": (
            None if b.message_credits_daily_cap is None
            else str(b.message_credits_daily_cap)
        ),
        "prior_purchased_credits_remaining": str(b.purchased_credits_remaining),
        "prior_grandfathered_at": (
            s.grandfathered_at.isoformat() if s.grandfathered_at else None
        ),
        "prior_grandfathered_product_id": s.grandfathered_product_id,
    }


def _print_plan(entries: list[dict[str, Any]], plan: SubscriptionPlan) -> None:
    print()
    print("=" * 78)
    print(f"GRANDFATHER PLAN — {len(entries)} account(s)")
    print("=" * 78)
    for e in entries:
        print(f"\n  {e['email']}   {e['user_id']}")
        print(f"    apple      {e['product_id']}  txn={e['original_transaction_id']}")
        print(f"    expires    {e['expires_date']}")
        # The three lines an operator has to read to spot a refund. A REFUND
        # keeps the ORIGINAL future expires_date, so "expires 2026-10-01" alone
        # looks like a healthy payer; `status` is what says otherwise.
        print(
            f"    status     {e['status']!r}   auto_renew="
            f"{e['auto_renew_status']}"
        )
        print(f"    plan       {e['prior_plan_id']!r}  ->  {UNLIMITED_PLAN_ID!r}")
        print(
            f"    message    {e['prior_message_credits_remaining']}  ->  "
            f"{plan.message_credits_monthly}"
        )
        print(
            f"    integration {e['prior_integration_credits_remaining']}  ->  "
            f"{plan.integration_credits_monthly}"
        )
        print(f"    used_today {e['prior_message_credits_used_today']}  ->  0")
        print(f"    daily_cap  {e['prior_message_credits_daily_cap']}  ->  None")
        print(
            f"    plan_source {e['prior_plan_source']!r}  ->  {PLAN_SOURCE_APPLE!r}"
        )
        print(
            f"    purchased  {e['prior_purchased_credits_remaining']}  ->  "
            f"UNTOUCHED (bought credits never expire)"
        )
        print("    period     UNTOUCHED (Apple owns the clock)")
        print(f"    apple_subscriptions.grandfathered_product_id -> {e['product_id']}")
    print()
    print("─" * 78)
    print("BEFORE YOU RUN THIS WITH --apply, check what these three people will")
    print("SEE. All three subscribed on App Store build 109, which renders")
    print("`message.remaining` verbatim:")
    print()
    print(f"    CreditsScreen  ->  '{int(plan.message_credits_monthly)} credits left'")
    print("                      under an 'Unlimited' pill")
    print("    UsageScreen    ->  'Unlimited / Monthly limit / 0% used'")
    print()
    print("and build 109 reads NONE of the billed_* fields, so the legacy")
    print("subscription they are still being charged for is invisible to them.")
    print("There is no server-side value that makes 'N credits left' read")
    print("correctly on that build — the fix is an app release, and the right")
    print("order is: ship the Unlimited-aware client, confirm these accounts")
    print("have it, THEN run this. The command is manual precisely so that")
    print("ordering is a decision somebody makes.")
    print()
    print("Running first is still defensible if somebody is locked out today —")
    print("a wrong number beats a 402 — but it is a trade, not a detail.")
    print("─" * 78)
    print()


async def _apply_one(db, cand, *, now: datetime) -> None:
    uid = cand.user.id
    await credit_service.apply_plan_change(
        db, uid, UNLIMITED_PLAN_ID, reason=GRANDFATHER_REASON,
    )
    balance = await credit_service._lock_balance(db, uid)
    # Re-stamped EXPLICITLY. plan_source 'apple' is what makes the shipped iOS
    # client render "Subscribed — manage in Settings"; NULL on a paid plan
    # reads as stripe server-side and 409-BLOCKS the account from ever buying
    # on iOS. period_start/period_end are deliberately NOT touched.
    balance.plan_source = PLAN_SOURCE_APPLE
    sub = await db.get(AppleSubscription, cand.sub.id)
    sub.grandfathered_at = now
    sub.grandfathered_product_id = sub.product_id
    await db.flush()


# ── revert ───────────────────────────────────────────────────────────────


async def _revert_one(db, entry: dict[str, Any], *, now: datetime) -> str:
    uid = entry["user_id"]
    balance = await db.get(CreditBalance, uid)
    if balance is None:
        return f"  ! {entry['email']}: no credit_balances row; skipped"
    if balance.plan_id != UNLIMITED_PLAN_ID:
        return (
            f"  · {entry['email']}: plan is {balance.plan_id!r}, not "
            f"{UNLIMITED_PLAN_ID!r} — already reverted or moved on; skipped"
        )

    old_msg = Decimal(balance.message_credits_remaining)
    old_int = Decimal(balance.integration_credits_remaining)
    new_msg = Decimal(entry["prior_message_credits_remaining"])
    new_int = Decimal(entry["prior_integration_credits_remaining"])
    cap = entry["prior_message_credits_daily_cap"]

    balance.plan_id = entry["prior_plan_id"]
    balance.message_credits_remaining = new_msg
    balance.integration_credits_remaining = new_int
    balance.message_credits_used_today = Decimal(
        entry["prior_message_credits_used_today"]
    )
    balance.message_credits_daily_cap = None if cap is None else Decimal(cap)
    balance.plan_source = entry["prior_plan_source"]
    # purchased_credits_remaining is not restored because it was never changed.

    # Compensating ledger rows — the history of the grandfather and of its
    # reversal both stay. Never delete a billing row.
    meta = {"reason": "grandfather:revert", "from": UNLIMITED_PLAN_ID,
            "to": entry["prior_plan_id"], "mode": "absolute"}
    for bucket, delta, after in (
        (BUCKET_MESSAGE, new_msg - old_msg, new_msg),
        (BUCKET_INTEGRATION, new_int - old_int, new_int),
    ):
        if delta != 0:
            db.add(CreditLedger(
                user_id=uid, event_type=LEDGER_PLAN_CHANGE, bucket=bucket,
                amount=delta, balance_after=after, metadata_json=meta,
            ))

    sub = (await db.execute(select(AppleSubscription).where(
        AppleSubscription.original_transaction_id == entry["original_transaction_id"]
    ))).scalar_one_or_none()
    if sub is not None:
        prior_at = entry.get("prior_grandfathered_at")
        sub.grandfathered_at = (
            datetime.fromisoformat(prior_at) if prior_at else None
        )
        sub.grandfathered_product_id = entry.get("prior_grandfathered_product_id")
    await db.flush()
    return f"  ✓ {entry['email']}: back to {entry['prior_plan_id']!r}"


# ── commands ─────────────────────────────────────────────────────────────


async def cmd_run(args) -> int:
    _check_resolver()
    now = datetime.utcnow()

    async with async_session_maker() as db:
        await _check_plan_row(db)
        plan = await db.get(SubscriptionPlan, UNLIMITED_PLAN_ID)

        candidates = await entitlement.select_legacy_apple_payers(db, now=now)
        entries = [_entry(c) for c in candidates]

        stripe_rows = []
        if args.include_stripe:
            stripe_rows = await entitlement.select_legacy_stripe_payers(db)

        if not entries and not stripe_rows:
            print("No legacy Apple payers inside a paid period. Nothing to do.")
            return 0

        _print_plan(entries, plan)

        if stripe_rows:
            # Never applied automatically: there is no Stripe subscription
            # mirror in this schema, so plan_id alone cannot tell a payer from
            # alembic 054's bundle grant. A human confirms these against the
            # Stripe dashboard and grants them individually.
            print("STRIPE CANDIDATES — NOT APPLIED, confirm each in the Stripe")
            print("dashboard and use `python -m app.scripts.unlimited_grant` or a")
            print("manual plan change:")
            for u, b in stripe_rows:
                print(f"  {u.email}  {u.id}  plan={b.plan_id} source={b.plan_source}")
            print()

        if len(entries) > args.max:
            raise SystemExit(
                f"✗ selection is {len(entries)} account(s), above --max "
                f"{args.max}. A predicate bug looks exactly like this. Read the "
                f"plan above; raise --max only if every line is a real payer."
            )

        already = [
            c for c in candidates
            if c.balance.plan_id == UNLIMITED_PLAN_ID
            and c.sub.grandfathered_at is not None
        ]
        for c in already:
            logger.info("skip %s: already grandfathered", c.user.email)
        todo = [c for c in candidates if c not in already]

        if not args.apply:
            print("DRY RUN — nothing written. Re-run with --apply.")
            print(f"({len(todo)} would be changed, {len(already)} already done)")
            return 0

        # ── the receipt is built from `todo`, BEFORE anything is written ──
        #
        # It used to be built from every candidate, and written on every
        # --apply run whether or not that run changed anything. A second run
        # — an operator confirming, a retry after a partial failure, a second
        # session — therefore re-derived the "prior" state from the
        # ALREADY-GRANDFATHERED rows and overwrote both durable copies with
        # prior_plan_id='unlimited', prior_message_credits_remaining='1000000',
        # prior_grandfathered_at=<t1>. The before-state was then gone from
        # everywhere, and `--revert` walked the entries, found each balance
        # already on 'unlimited', set it to 'unlimited' again, computed two
        # zero deltas, wrote no compensating ledger rows and printed
        # "✓ back to 'unlimited'". The one production mutation in this change
        # set, reported as reversed and not reversed, on the account of a
        # customer who is currently 402-locked.
        new_entries = [_entry(c) for c in todo]
        poisoned = [e for e in new_entries
                    if e["prior_plan_id"] == UNLIMITED_PLAN_ID]
        if poisoned:
            raise SystemExit(
                f"✗ refusing to record a receipt whose 'prior' state is already "
                f"{UNLIMITED_PLAN_ID!r}: {[e['email'] for e in poisoned]}. That "
                f"is a receipt that cannot revert anything."
            )

        for cand in todo:
            await _apply_one(db, cand, now=now)

        # MERGE, never replace. Each user's first recorded before-state is the
        # only one that can revert them, so an existing entry wins; a repeat
        # run adds nothing and destroys nothing.
        setting = await db.get(PlatformSetting, RECEIPT_SETTING_KEY)
        merged: dict[str, dict[str, Any]] = {}
        if setting is not None and setting.value:
            try:
                for e in (json.loads(setting.value).get("entries") or []):
                    merged[e["user_id"]] = e
            except (ValueError, KeyError, TypeError) as exc:
                raise SystemExit(
                    f"✗ the stored receipt at platform_settings"
                    f"['{RECEIPT_SETTING_KEY}'] is unreadable ({exc}). Export "
                    f"and move it aside before running again — overwriting it "
                    f"would destroy the only record of the prior state."
                )
        carried = len(merged)
        for e in new_entries:
            merged.setdefault(e["user_id"], e)

        if not merged:
            # Nothing changed and nothing was ever recorded. Writing an empty
            # receipt over args.receipt would be the only lossy thing this
            # branch could still do.
            print("Nothing to apply and no prior receipt — nothing written.")
            return 0

        receipt = {
            "version": RECEIPT_VERSION,
            "applied_at": now.isoformat(),
            "reason": GRANDFATHER_REASON,
            "entries": sorted(merged.values(), key=lambda e: e["email"]),
        }
        blob = json.dumps(receipt, indent=2, sort_keys=True)
        if setting is None:
            db.add(PlatformSetting(key=RECEIPT_SETTING_KEY, value=blob))
        else:
            setting.value = blob
        await db.commit()

    with open(args.receipt, "w") as fh:
        fh.write(blob)
    print(blob)
    print(f"\n✓ APPLIED to {len(todo)} account(s). Receipt: {args.receipt}")
    if carried:
        print(f"  ({carried} earlier entr(y|ies) carried through unchanged.)")
    print(f"  Also stored at platform_settings['{RECEIPT_SETTING_KEY}'].")
    print(f"  Revert with: python -m app.scripts.grandfather_unlimited "
          f"--revert {args.receipt}")
    return 0


async def cmd_revert(args) -> int:
    with open(args.revert) as fh:
        receipt = json.load(fh)
    if receipt.get("version") != RECEIPT_VERSION:
        raise SystemExit(
            f"✗ receipt version {receipt.get('version')!r} != {RECEIPT_VERSION}"
        )
    entries = receipt.get("entries") or []
    now = datetime.utcnow()
    print(f"REVERT — {len(entries)} account(s) from {receipt.get('applied_at')}")
    async with async_session_maker() as db:
        lines = [await _revert_one(db, e, now=now) for e in entries]
        for line in lines:
            print(line)
        if args.apply:
            await db.commit()
            print("✓ APPLIED")
        else:
            await db.rollback()
            print("DRY RUN — nothing written. Re-run with --apply.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.scripts.grandfather_unlimited",
        description="Move legacy Apple payers onto the Unlimited plan.",
    )
    p.add_argument("--apply", action="store_true",
                   help="write. Without it the command is a dry run.")
    p.add_argument("--include-stripe", action="store_true",
                   help="also PRINT Stripe candidates (never applied).")
    p.add_argument("--max", type=int, default=10,
                   help="refuse if the selection exceeds this many accounts.")
    p.add_argument("--receipt", default="grandfather_unlimited_receipt.json")
    p.add_argument("--revert", metavar="RECEIPT",
                   help="restore the prior values recorded in a receipt.")
    return p


async def run(argv: Optional[list[str]] = None) -> int:
    """The whole command, minus the event loop — see unlimited_grant.run."""
    args = build_parser().parse_args(argv)
    if args.revert:
        return await cmd_revert(args)
    return await cmd_run(args)


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    sys.exit(main())
