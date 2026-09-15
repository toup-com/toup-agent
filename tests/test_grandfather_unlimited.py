"""The grandfather command — who it selects, what it sets, and what it refuses.

The fixture is the production population as Phase 0 measured it on
2026-09-13, because every exclusion in this file is a real account that a
naive predicate sweeps in:

  * three real Apple payers (Production, legacy products, inside a paid
    period) — the ONLY rows that may be selected;
  * a Sandbox Elite that still says ``status='active'`` 2.7 months after it
    expired — excluded twice over, by ``environment`` and by ``expires_date``,
    which is why neither test relies on the other;
  * an alembic-054 Builder grant with ``plan_source`` NULL and no Apple row —
    the reason ``plan_id <> 'free'`` can never be the predicate;
  * two App Review comps on ``plan_id='free'`` with hand-written balances;
  * an admin, already unlimited via ``role``.

The other half of the file is the mutation itself: dry run writes nothing,
apply writes exactly the documented columns, a second run is a no-op, the
receipt round-trips through ``--revert``, and the 2026-10-01 renewal keeps
the grandfather rather than reverting it to Builder.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


LEGACY = {
    "starter": "ai.toup.app.sub.starter",
    "builder": "ai.toup.app.sub.builder",
    "pro": "ai.toup.app.sub.pro",
    "elite": "ai.toup.app.sub.elite",
}


async def _user(email: str, *, role: str = "beta_user",
                stripe_customer_id: str | None = None) -> str:
    from app.db import User, async_session_maker
    from app.services.auth_service import get_password_hash

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=email, name=email.split("@")[0],
            hashed_password=get_password_hash("test-password-1234"),
            role=role, stripe_customer_id=stripe_customer_id,
        ))
        await db.commit()
    return uid


async def _balance(uid: str, *, plan_id: str = "free",
                   plan_source: str | None = None, purchased: str = "0"):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        bal = await credit_service.get_or_create_balance(db, uid)
        if plan_id != "free":
            await credit_service.apply_plan_change(db, uid, plan_id)
            bal = await credit_service.get_or_create_balance(db, uid)
        bal.plan_source = plan_source
        bal.purchased_credits_remaining = Decimal(purchased)
        await db.commit()
    return bal


async def _sub(uid: str, product: str, *, environment="Production",
               days=18, status="active", plan_id="builder") -> str:
    from app.db import async_session_maker
    from app.db.models import AppleSubscription

    txn = f"txn-{uuid.uuid4().hex[:12]}"
    async with async_session_maker() as db:
        db.add(AppleSubscription(
            user_id=uid, original_transaction_id=txn, product_id=product,
            plan_id=plan_id, status=status,
            expires_date=datetime.utcnow() + timedelta(days=days),
            auto_renew_status=True, environment=environment,
        ))
        await db.commit()
    return txn


async def _get_balance(uid: str):
    from app.db import CreditBalance, async_session_maker
    async with async_session_maker() as db:
        return await db.get(CreditBalance, uid)


async def _get_sub(txn: str, *, db=None):
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import AppleSubscription

    stmt = select(AppleSubscription).where(
        AppleSubscription.original_transaction_id == txn)
    if db is not None:
        return (await db.execute(stmt)).scalar_one()
    async with async_session_maker() as own:
        return (await own.execute(stmt)).scalar_one()


@pytest_asyncio.fixture
async def population():
    """Production as Phase 0 measured it."""
    p = SimpleNamespace()

    # ── the three real payers ────────────────────────────────────────
    p.parmida = await _user("parmidaisazadeh@icloud.com")
    await _balance(p.parmida, plan_id="builder", plan_source="apple", purchased="50")
    p.parmida_txn = await _sub(p.parmida, LEGACY["builder"], days=18)

    p.azam = await _user("azam.yousefnejad@icloud.com")
    await _balance(p.azam, plan_id="starter", plan_source="apple")
    p.azam_txn = await _sub(p.azam, LEGACY["starter"], days=18, plan_id="starter")

    # Auto-renew OFF but still inside a period he paid for — SELECTED. Apple's
    # EXPIRED returns him to free through the ordinary path, no special case.
    p.alireza = await _user("5gvrjn45sf@privaterelay.appleid.com")
    await _balance(p.alireza, plan_id="starter", plan_source="apple")
    p.alireza_txn = await _sub(p.alireza, LEGACY["starter"], days=17, plan_id="starter")

    # ── everyone who must NOT be selected ────────────────────────────
    # The Sandbox Elite: environment='Sandbox', status STILL 'active',
    # expired 2.7 months ago.
    p.sandbox = await _user("b5fm6n85mt@privaterelay.appleid.com")
    await _balance(p.sandbox, plan_id="elite", plan_source="apple")
    p.sandbox_txn = await _sub(
        p.sandbox, LEGACY["elite"], environment="Sandbox", days=-81,
        status="active", plan_id="elite",
    )

    # alembic 054's LLM-bundle grandfather: builder, plan_source NULL, no
    # Apple row, no Stripe customer.
    p.bundle = await _user("nariman@toup.ai")
    await _balance(p.bundle, plan_id="builder", plan_source=None)

    # App Review comps: free plan, hand-written balances, no ledger rows.
    p.review_a = await _user("applereview@toup.ai")
    p.review_b = await _user("mrhx+google-review@toup.ai")
    for uid in (p.review_a, p.review_b):
        from app.db import async_session_maker
        await _balance(uid)
        async with async_session_maker() as db:
            bal = await _get_balance(uid)
            bal = await db.merge(bal)
            bal.message_credits_remaining = Decimal("9999")
            await db.commit()

    p.admin = await _user("canary@toup.ai", role="admin")
    await _balance(p.admin)

    p.payers = {p.parmida, p.azam, p.alireza}
    return p


# ── the predicate ────────────────────────────────────────────────────


async def test_selects_exactly_the_three_payers(population):
    from app.db import async_session_maker
    from app.services import entitlement

    async with async_session_maker() as db:
        cands = await entitlement.select_legacy_apple_payers(db)
    assert {c.user.id for c in cands} == population.payers


async def test_sandbox_row_is_excluded_by_environment_alone(population):
    """Independently of expiry: give the Sandbox row a FUTURE expires_date and
    it must still be excluded, because `environment` is the predicate that
    distinguishes a paid entitlement from a granted one."""
    from app.db import async_session_maker
    from app.services import entitlement

    sub = await _get_sub(population.sandbox_txn)
    async with async_session_maker() as db:
        row = await db.merge(sub)
        row.expires_date = datetime.utcnow() + timedelta(days=30)
        await db.commit()

    async with async_session_maker() as db:
        cands = await entitlement.select_legacy_apple_payers(db)
    assert population.sandbox not in {c.user.id for c in cands}


async def test_expired_row_is_excluded_by_date_alone(population):
    """Independently of environment: flip the Sandbox row to Production and it
    must still be excluded, because it lapsed 2026-06-24 — and `status` still
    reads 'active', which is exactly why status is never read."""
    from app.db import async_session_maker
    from app.services import entitlement

    sub = await _get_sub(population.sandbox_txn)
    assert sub.status == "active", "the row lies; that is the point"
    async with async_session_maker() as db:
        row = await db.merge(sub)
        row.environment = "Production"
        await db.commit()

    async with async_session_maker() as db:
        cands = await entitlement.select_legacy_apple_payers(db)
    assert population.sandbox not in {c.user.id for c in cands}


async def test_bundle_grandfather_and_review_comps_are_excluded(population):
    from app.db import async_session_maker
    from app.services import entitlement

    async with async_session_maker() as db:
        apple = {c.user.id for c in await entitlement.select_legacy_apple_payers(db)}
        stripe = {u.id for (u, _b) in await entitlement.select_legacy_stripe_payers(db)}

    for uid in (population.bundle, population.review_a, population.review_b,
                population.admin):
        assert uid not in apple
        assert uid not in stripe
    # No Stripe payer has ever existed. Zero rows is the expected output.
    assert stripe == set()


# ── the mutation ─────────────────────────────────────────────────────


async def test_dry_run_writes_nothing_and_names_the_users(population, capsys, tmp_path):
    from app.scripts.grandfather_unlimited import run

    assert await run(["--receipt", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    for uid in population.payers:
        assert uid in out
    assert "parmidaisazadeh@icloud.com" in out
    assert population.sandbox not in out
    assert population.bundle not in out

    for uid in population.payers:
        assert (await _get_balance(uid)).plan_id != "unlimited"
    assert not (tmp_path / "r.json").exists()


async def test_apply_sets_exactly_the_documented_columns(population, tmp_path, capsys):
    from app.scripts.grandfather_unlimited import run

    before = await _get_balance(population.parmida)
    period = (before.period_start, before.period_end)

    assert await run(["--apply", "--receipt", str(tmp_path / "r.json")]) == 0
    capsys.readouterr()

    bal = await _get_balance(population.parmida)
    assert bal.plan_id == "unlimited"
    assert Decimal(bal.message_credits_remaining) == Decimal("1000000")
    assert Decimal(bal.integration_credits_remaining) == Decimal("1000000")
    assert Decimal(bal.message_credits_used_today) == Decimal("0")
    assert bal.message_credits_daily_cap is None
    # plan_source stays 'apple': NULL on a paid plan reads as stripe
    # server-side and 409-BLOCKS the account from ever buying on iOS.
    assert bal.plan_source == "apple"
    # Bought credits never expire, in either direction.
    assert Decimal(bal.purchased_credits_remaining) == Decimal("50")
    # Apple owns the clock.
    assert (bal.period_start, bal.period_end) == period

    sub = await _get_sub(population.parmida_txn)
    assert sub.grandfathered_at is not None
    assert sub.grandfathered_product_id == LEGACY["builder"]
    # The MIRROR still records what the product was sold as — that is the only
    # in-DB record of what the customer is being charged for.
    assert sub.plan_id == "builder"
    assert sub.product_id == LEGACY["builder"]

    # Nobody else moved.
    assert (await _get_balance(population.sandbox)).plan_id == "elite"
    assert (await _get_balance(population.bundle)).plan_id == "builder"
    assert (await _get_balance(population.review_a)).plan_id == "free"
    assert (await _get_balance(population.admin)).plan_id == "free"


async def test_apply_is_idempotent(population, tmp_path, capsys):
    from app.scripts.grandfather_unlimited import run

    r = str(tmp_path / "r.json")
    assert await run(["--apply", "--receipt", r]) == 0
    first = await _get_sub(population.parmida_txn)
    stamp = first.grandfathered_at
    capsys.readouterr()

    assert await run(["--apply", "--receipt", r]) == 0
    out = capsys.readouterr().out
    assert "already grandfathered" in out or "0 account" in out
    again = await _get_sub(population.parmida_txn)
    assert again.grandfathered_at == stamp
    assert Decimal((await _get_balance(population.parmida)).message_credits_remaining) \
        == Decimal("1000000")


async def test_max_refuses_an_oversized_selection(population, tmp_path):
    from app.scripts.grandfather_unlimited import run

    with pytest.raises(SystemExit) as ei:
        await run(["--apply", "--max", "2", "--receipt", str(tmp_path / "r.json")])
    assert "above --max" in str(ei.value)
    for uid in population.payers:
        assert (await _get_balance(uid)).plan_id != "unlimited"


async def test_refuses_when_the_resolver_is_not_deployed(population, monkeypatch, tmp_path):
    """Grandfathering against a deploy whose resolver ignores grandfathered_at
    would be silently reverted at the next DID_RENEW."""
    import app.scripts.grandfather_unlimited as gf

    monkeypatch.setattr(gf, "plan_for_subscription", lambda pid, sub=None: "builder")
    with pytest.raises(SystemExit) as ei:
        await gf.run(["--apply", "--receipt", str(tmp_path / "r.json")])
    assert "does not honour grandfathered_at" in str(ei.value)


async def test_refuses_when_the_product_map_is_overloaded(population, monkeypatch, tmp_path):
    """The other direction: a map that returns 'unlimited' for an UNMARKED
    legacy product hands Unlimited to every future holder of a legacy id."""
    import app.scripts.grandfather_unlimited as gf

    monkeypatch.setattr(gf, "plan_for_subscription", lambda pid, sub=None: "unlimited")
    with pytest.raises(SystemExit) as ei:
        await gf.run(["--apply", "--receipt", str(tmp_path / "r.json")])
    assert "overloaded" in str(ei.value)


async def test_receipt_round_trips_through_revert(population, tmp_path, capsys):
    from app.scripts.grandfather_unlimited import RECEIPT_SETTING_KEY, run
    from app.db import async_session_maker
    from app.db.models import PlatformSetting

    r = tmp_path / "r.json"
    assert await run(["--apply", "--receipt", str(r)]) == 0
    capsys.readouterr()

    receipt = json.loads(r.read_text())
    assert {e["user_id"] for e in receipt["entries"]} == population.payers
    parmida = next(e for e in receipt["entries"] if e["user_id"] == population.parmida)
    assert parmida["prior_plan_id"] == "builder"
    assert parmida["original_transaction_id"] == population.parmida_txn

    async with async_session_maker() as db:
        stored = await db.get(PlatformSetting, RECEIPT_SETTING_KEY)
    assert stored is not None
    assert json.loads(stored.value)["entries"]

    # Dry-run revert changes nothing.
    assert await run(["--revert", str(r)]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert (await _get_balance(population.parmida)).plan_id == "unlimited"

    assert await run(["--revert", str(r), "--apply"]) == 0
    capsys.readouterr()
    bal = await _get_balance(population.parmida)
    assert bal.plan_id == parmida["prior_plan_id"]
    assert Decimal(bal.message_credits_remaining) == Decimal(
        parmida["prior_message_credits_remaining"])
    assert bal.plan_source == parmida["prior_plan_source"]
    assert Decimal(bal.purchased_credits_remaining) == Decimal("50")
    sub = await _get_sub(population.parmida_txn)
    assert sub.grandfathered_at is None
    assert sub.grandfathered_product_id is None


async def test_revert_keeps_the_ledger(population, tmp_path, capsys):
    """Billing history is never deleted. The grandfather and its reversal are
    both plan_change rows; the reversal is a compensating entry."""
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import CreditLedger
    from app.scripts.grandfather_unlimited import run

    r = tmp_path / "r.json"
    assert await run(["--apply", "--receipt", str(r)]) == 0
    assert await run(["--revert", str(r), "--apply"]) == 0
    capsys.readouterr()

    async with async_session_maker() as db:
        rows = (await db.execute(select(CreditLedger).where(
            CreditLedger.user_id == population.parmida,
            CreditLedger.event_type == "plan_change",
        ))).scalars().all()
    reasons = [(r_.metadata_json or {}).get("reason") for r_ in rows]
    assert "grandfather:legacy_apple" in reasons
    assert "grandfather:revert" in reasons


# ── the 2026-10-01 regression ────────────────────────────────────────


async def test_did_renew_on_a_legacy_product_keeps_unlimited(population, tmp_path, capsys):
    """The whole reason grandfathered_at exists. On DID_RENEW the backend
    re-derives the plan from the PRODUCT id; without the stamp, Parmida's
    2026-10-01 renewal silently reverts her to Builder."""
    from app.api.iap import _handle_subscription_notification
    from app.db import async_session_maker
    from app.scripts.grandfather_unlimited import run

    assert await run(["--apply", "--receipt", str(tmp_path / "r.json")]) == 0
    capsys.readouterr()

    new_expiry = datetime.utcnow() + timedelta(days=30)
    decoded = SimpleNamespace(
        notificationType="DID_RENEW", subtype=None,
        notificationUUID=str(uuid.uuid4()),
    )
    txn_payload = SimpleNamespace(
        productId=LEGACY["builder"],
        originalTransactionId=population.parmida_txn,
        # utcnow() is NAIVE, and .timestamp() on a naive datetime reads it as
        # LOCAL time — a 4h skew on this machine. ms_to_datetime returns naive
        # UTC, so the epoch has to be built as UTC explicitly.
        expiresDate=int(new_expiry.replace(tzinfo=timezone.utc).timestamp() * 1000),
        environment="Production",
    )
    async with async_session_maker() as db:
        await _handle_subscription_notification(
            db, decoded, txn_payload,
            SimpleNamespace(autoRenewStatus=1, autoRenewProductId=None,
                            gracePeriodExpiresDate=None),
        )
        await db.commit()

    bal = await _get_balance(population.parmida)
    assert bal.plan_id == "unlimited", "the renewal must not revert the grandfather"
    assert Decimal(bal.message_credits_remaining) == Decimal("1000000")
    assert bal.plan_source == "apple"
    assert abs((bal.period_end - new_expiry).total_seconds()) < 2


async def test_an_unmarked_legacy_subscription_still_gets_its_tier(population):
    """A subscription the command did NOT mark keeps the tier it was sold as —
    which is what stops a post-cutover cross-grade down to $9.90 Starter from
    quietly buying Unlimited."""
    from app.services.apple_iap_service import plan_for_subscription

    sub = await _get_sub(population.azam_txn)
    assert sub.grandfathered_at is None
    assert plan_for_subscription(LEGACY["starter"], sub) == "starter"


async def test_a_grandfathered_payer_who_crossgrades_down_loses_unlimited(
    population, tmp_path, capsys,
):
    """The money leak `grandfathered_product_id` exists to close, and did not.

    The four legacy products stay on sale in App Store Connect forever
    (removing one stops its renewals, and EXPIRED downgrades), so Apple keeps
    showing all of them in the customer's own Manage Subscriptions. A
    cross-grade inside the group keeps the SAME originalTransactionId, so
    Parmida switching Builder (CAD 19.90) → Starter (CAD 9.90) arrives at the
    next DID_RENEW on her existing row with `grandfathered_at` still set.
    Reading that stamp alone re-granted the 1,000,000-credit Unlimited
    allowance for half the price, permanently and unlogged.
    """
    from app.api.iap import _handle_subscription_notification
    from app.db import async_session_maker
    from app.scripts.grandfather_unlimited import run

    assert await run(["--apply", "--receipt", str(tmp_path / "r.json")]) == 0
    capsys.readouterr()
    assert (await _get_balance(population.parmida)).plan_id == "unlimited"

    new_expiry = datetime.utcnow() + timedelta(days=30)
    decoded = SimpleNamespace(
        notificationType="DID_RENEW", subtype=None,
        notificationUUID=str(uuid.uuid4()),
    )
    txn_payload = SimpleNamespace(
        productId=LEGACY["starter"],          # ← the cross-grade
        originalTransactionId=population.parmida_txn,
        expiresDate=int(new_expiry.replace(tzinfo=timezone.utc).timestamp() * 1000),
        environment="Production",
    )
    async with async_session_maker() as db:
        await _handle_subscription_notification(
            db, decoded, txn_payload,
            SimpleNamespace(autoRenewStatus=1, autoRenewProductId=None,
                            gracePeriodExpiresDate=None),
        )
        await db.commit()

    bal = await _get_balance(population.parmida)
    assert bal.plan_id == "starter", (
        "the grandfather followed a product change and bought Unlimited at "
        "the Starter price"
    )
    assert Decimal(bal.message_credits_remaining) == Decimal("130")

    # The stamp itself is NOT erased: it is the record of what happened, and
    # a cross-grade BACK to Builder must restore the entitlement.
    sub = await _get_sub(population.parmida_txn)
    assert sub.grandfathered_product_id == LEGACY["builder"]
    assert sub.grandfathered_at is not None


async def test_the_grandfather_still_holds_on_its_own_product(population, tmp_path, capsys):
    """Anti-vacuity for the test above: the forfeit must be caused by the
    product CHANGING, not by the product check existing."""
    from app.services.apple_iap_service import plan_for_subscription
    from app.scripts.grandfather_unlimited import run

    assert await run(["--apply", "--receipt", str(tmp_path / "r.json")]) == 0
    capsys.readouterr()

    sub = await _get_sub(population.parmida_txn)
    assert plan_for_subscription(LEGACY["builder"], sub) == "unlimited"
    assert plan_for_subscription(LEGACY["starter"], sub) == "starter"
    assert plan_for_subscription(LEGACY["pro"], sub) == "pro"


async def test_a_hand_stamped_row_with_no_product_snapshot_is_never_demoted(population):
    """NULL grandfathered_product_id counts as a match. A row stamped by hand,
    or before alembic 104 added the column, must keep its entitlement rather
    than be silently dropped by the new check."""
    from app.db import async_session_maker
    from app.services.apple_iap_service import plan_for_subscription

    async with async_session_maker() as db:
        sub = await _get_sub(population.parmida_txn, db=db)
        sub.grandfathered_at = datetime.utcnow()
        sub.grandfathered_product_id = None
        await db.commit()

    sub = await _get_sub(population.parmida_txn)
    assert plan_for_subscription(LEGACY["builder"], sub) == "unlimited"


# ── status may argue DEAD ────────────────────────────────────────────


async def test_a_refunded_subscription_is_not_a_payer(population, tmp_path, capsys):
    """The predicate read only the clock, and Apple's REFUND keeps the clock.

    On REFUND/REVOKE the handler writes status='expired'/'revoked' but only
    overwrites `expires_date` `if expires_date is not None` — and Apple's
    REFUND transaction carries the ORIGINAL expiresDate. So a refunded row
    keeps a FUTURE expires_date, and `environment='Production'` +
    `product_id IN (four)` + `expires_date > now` all still pass.

    The scenario is not hypothetical: Parmida has been locked out at 0.02
    credits since 2026-09-11, which is the single most likely trigger for a
    refund request. Apple refunds her, and the grandfather run that afternoon
    hands her 1,000,000 credits indefinitely with the money already returned.
    """
    from app.db import async_session_maker
    from app.scripts.grandfather_unlimited import run

    async with async_session_maker() as db:
        sub = await _get_sub(population.parmida_txn, db=db)
        sub.status = "expired"          # what REFUND writes
        assert sub.expires_date > datetime.utcnow(), (
            "the fixture must keep the future clock — that IS the trap"
        )
        await db.commit()

    assert await run(["--apply", "--receipt", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out

    assert (await _get_balance(population.parmida)).plan_id == "builder", (
        "a refunded subscription was grandfathered"
    )
    assert "parmidaisazadeh@icloud.com" not in out
    # anti-vacuity: the other two are still selected by the same run
    assert (await _get_balance(population.azam)).plan_id == "unlimited"
    assert (await _get_balance(population.alireza)).plan_id == "unlimited"


async def test_a_revoked_subscription_is_not_a_live_entitlement(population):
    """`has_live_apple_entitlement` gates the orphan-entitlement backstop and
    the admin grant's exclusivity check. Reading a revoked row as live means
    the backstop skips the very user whose downgrade was lost, and an admin
    comp is refused on the grounds of a subscription Apple has revoked."""
    from app.db import async_session_maker
    from app.services import entitlement

    async with async_session_maker() as db:
        sub = await _get_sub(population.parmida_txn, db=db)
        sub.status = "revoked"
        await db.commit()

    async with async_session_maker() as db:
        assert await entitlement.has_live_apple_entitlement(
            db, population.parmida) is None
        # anti-vacuity: an untouched payer is still live
        assert await entitlement.has_live_apple_entitlement(
            db, population.azam) is not None


async def test_the_operator_is_shown_the_status_they_have_to_judge(
    population, tmp_path, capsys,
):
    """`--dry-run` prints expiry, plan and credits. It did not print status,
    so the refund above was invisible in the one review the design relies on:
    "expires 2026-10-01" reads as a healthy payer either way."""
    from app.scripts.grandfather_unlimited import run

    assert await run(["--receipt", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "status" in out and "auto_renew" in out


# ── the receipt is the only thing that makes this reversible ─────────


async def test_a_second_apply_does_not_clobber_the_receipt(
    population, tmp_path, capsys,
):
    """The receipt was built from ALL candidates and written on every --apply.

    A second run — an operator confirming, a retry after a partial failure, a
    second session — therefore re-derived "prior" from the already-
    grandfathered rows and overwrote both durable copies with
    prior_plan_id='unlimited' / 1,000,000 credits. The before-state then
    existed nowhere, and --revert reported success while changing nothing.
    """
    import json as _json
    from app.db import async_session_maker
    from app.db.models import PlatformSetting
    from app.scripts.grandfather_unlimited import RECEIPT_SETTING_KEY, run

    path = tmp_path / "r.json"
    assert await run(["--apply", "--receipt", str(path)]) == 0
    capsys.readouterr()
    first = _json.loads(path.read_text())
    parmida_before = next(
        e for e in first["entries"] if e["email"] == "parmidaisazadeh@icloud.com")
    assert parmida_before["prior_plan_id"] == "builder"

    assert await run(["--apply", "--receipt", str(path)]) == 0
    capsys.readouterr()

    second = _json.loads(path.read_text())
    again = next(
        e for e in second["entries"] if e["email"] == "parmidaisazadeh@icloud.com")
    assert again["prior_plan_id"] == "builder", "the receipt was clobbered"
    assert again == parmida_before
    assert {e["email"] for e in second["entries"]} == {
        e["email"] for e in first["entries"]}

    async with async_session_maker() as db:
        stored = await db.get(PlatformSetting, RECEIPT_SETTING_KEY)
    kept = next(e for e in _json.loads(stored.value)["entries"]
                if e["email"] == "parmidaisazadeh@icloud.com")
    assert kept["prior_plan_id"] == "builder"


async def test_revert_after_a_second_apply_really_reverts(
    population, tmp_path, capsys,
):
    """The consequence, end to end: the receipt surviving is only interesting
    because --revert has to work off it."""
    from app.scripts.grandfather_unlimited import run

    path = tmp_path / "r.json"
    assert await run(["--apply", "--receipt", str(path)]) == 0
    assert await run(["--apply", "--receipt", str(path)]) == 0
    capsys.readouterr()

    # --revert is a dry run without --apply, like the forward direction.
    assert await run(["--revert", str(path)]) == 0
    capsys.readouterr()
    assert (await _get_balance(population.parmida)).plan_id == "unlimited"

    assert await run(["--revert", str(path), "--apply"]) == 0
    capsys.readouterr()

    assert (await _get_balance(population.parmida)).plan_id == "builder"
    assert (await _get_balance(population.azam)).plan_id == "starter"
    assert (await _get_sub(population.parmida_txn)).grandfathered_at is None


async def test_a_receipt_that_cannot_revert_is_refused(population, tmp_path, capsys):
    """Belt to the merge's braces: if a `todo` entry ever presents itself with
    a prior state of 'unlimited', that is a receipt which can revert nothing,
    and writing it is worse than failing."""
    from app.db import async_session_maker
    from app.scripts.grandfather_unlimited import run

    # A balance already on unlimited with NO grandfather stamp: not `already`
    # (which requires both), so it lands in `todo` with a useless prior state.
    async with async_session_maker() as db:
        bal = await db.merge(await _get_balance(population.parmida))
        bal.plan_id = "unlimited"
        await db.commit()

    with pytest.raises(SystemExit) as exc:
        await run(["--apply", "--receipt", str(tmp_path / "r.json")])
    assert "cannot revert" in str(exc.value)
    assert not (tmp_path / "r.json").exists()


async def test_the_command_refuses_a_deploy_without_the_crossgrade_fix(
    population, tmp_path, monkeypatch, capsys,
):
    """The preconditions exist so the operator cannot run this against a
    deploy that would undo it. A resolver that keeps Unlimited across a
    PRODUCT change is one of those: the four legacy products can never be
    taken off sale, so a grandfathered payer could cross-grade to Starter from
    iOS Settings and keep Unlimited at CAD 9.90 forever, unlogged."""
    from app.scripts import grandfather_unlimited as gf

    monkeypatch.setattr(
        gf, "plan_for_subscription",
        lambda product_id, sub=None: (
            "unlimited" if getattr(sub, "grandfathered_at", None) else "builder"
        ),
    )
    with pytest.raises(SystemExit) as exc:
        await gf.run(["--apply", "--receipt", str(tmp_path / "r.json")])
    assert "PRODUCT CHANGE" in str(exc.value)
    assert (await _get_balance(population.parmida)).plan_id == "builder"


async def test_the_dry_run_says_what_build_109_will_render(
    population, tmp_path, capsys,
):
    """All three payers subscribed on App Store build 109, which renders
    `message.remaining` verbatim — so the grandfather puts "1000000 credits
    left" under an "Unlimited" pill on their billing screen, and build 109
    reads none of the billed_* fields, so the legacy subscription they are
    still charged for is invisible. No server-side value fixes that; the
    ordering decision is the operator's, and they can only make it if the
    command says so."""
    from app.scripts.grandfather_unlimited import run

    assert await run(["--receipt", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "1000000 credits left" in out
    assert "build 109" in out
