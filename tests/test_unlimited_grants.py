"""The UNLIMITED admin override — grant, revoke, lapse, and the audit trail.

Covers design §4 and the founder's rule 3 ("a linked override lapses with its
sponsor, automatically, on every channel, with an audit trail"):

* a grant stamps the entitlement and the grantee is never denied;
* the sponsor's Apple EXPIRED lapses the grant IN THE SAME TRANSACTION and
  returns the grantee to free;
* revoke is a stamp, never a delete — the row survives with who/when/why;
* a revoked or lapsed grant grants nothing, and the entitlement it was
  holding up comes down with it;
* a deleted sponsor fails CLOSED (the CASCADE takes the subscription row, the
  grant row survives naming a transaction that no longer exists);
* one live grant per user, but a NEW grant is possible after a revoke —
  the index is partial for exactly that reason;
* the endpoints are admin-gated and share every refusal with the service.

The charge-path assertions matter as much as the table ones: this design is
only correct because ``credit_balances.plan_id`` is the single
materialisation, so every test that ends a grant also asserts the charge gate
agrees.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


LEGACY_PRODUCT = "ai.toup.app.sub.builder"


# ── helpers ──────────────────────────────────────────────────────────


async def _mk_user(email: str | None = None) -> str:
    from app.db import User, async_session_maker
    from app.services.auth_service import get_password_hash

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid,
            email=email or f"g-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("test-password-1234"),
            name="Grant Test",
        ))
        await db.commit()
    return uid


async def _mk_balance(user_id: str):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        bal = await credit_service.get_or_create_balance(db, user_id)
        await db.commit()
        return bal


async def _mk_apple_sub(
    user_id: str, *, txn: str | None = None, product: str = LEGACY_PRODUCT,
    environment: str = "Production", days: int = 20, status: str = "active",
) -> str:
    from app.db import async_session_maker
    from app.db.models import AppleSubscription

    txn = txn or f"txn-{uuid.uuid4().hex[:12]}"
    async with async_session_maker() as db:
        db.add(AppleSubscription(
            user_id=user_id,
            original_transaction_id=txn,
            product_id=product,
            plan_id="builder",
            status=status,
            expires_date=datetime.utcnow() + timedelta(days=days),
            auto_renew_status=True,
            environment=environment,
        ))
        await db.commit()
    return txn


async def _balance(user_id: str):
    from app.db import CreditBalance, async_session_maker
    async with async_session_maker() as db:
        return await db.get(CreditBalance, user_id)


async def _last_ledger(user_id: str):
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import CreditLedger
    async with async_session_maker() as db:
        return (await db.execute(
            select(CreditLedger)
            .where(CreditLedger.user_id == user_id,
                   CreditLedger.event_type == "chat_message")
            .order_by(CreditLedger.created_at.desc())
        )).scalars().first()


async def _grant_row(grant_id: str):
    from app.db import async_session_maker
    from app.db.models import UnlimitedGrant
    async with async_session_maker() as db:
        return await db.get(UnlimitedGrant, grant_id)


async def _charge(user_id: str, amount: str = "500"):
    """A real charge through the real gate, with enforcement ON."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    prev = (settings.credit_enforcement_enabled, settings.credit_cap_admission_control)
    settings.credit_enforcement_enabled = True
    settings.credit_cap_admission_control = True
    try:
        async with async_session_maker() as db:
            res = await credit_service.try_charge(
                db, user_id, bucket="message", amount=Decimal(amount),
                event_type="chat_message", underlying_cost_cents=Decimal(amount),
            )
            await db.commit()
            return res
    finally:
        settings.credit_enforcement_enabled = prev[0]
        settings.credit_cap_admission_control = prev[1]


async def _make_grant(grantee: str, sponsor: str, txn: str, admin_email="ops@toup.ai"):
    from app.db import async_session_maker
    from app.services import entitlement

    async with async_session_maker() as db:
        grant = await entitlement.create_grant(
            db,
            granted_to_user_id=grantee,
            sponsor_kind="apple",
            sponsor_original_txn_id=txn,
            reason="second account, rides the sponsor's Builder sub",
            granted_by_user_id=sponsor,
            granted_by_email=admin_email,
        )
        gid = grant.id
        await db.commit()
    return gid


@pytest_asyncio.fixture
async def pair():
    """A sponsor with a live Production Apple sub, and a free grantee."""
    sponsor = await _mk_user("sponsor@example.com")
    grantee = await _mk_user("second-account@example.com")
    await _mk_balance(sponsor)
    await _mk_balance(grantee)
    txn = await _mk_apple_sub(sponsor)
    return SimpleNamespace(sponsor=sponsor, grantee=grantee, txn=txn)


# ── the grant ────────────────────────────────────────────────────────


async def test_grant_stamps_the_entitlement(pair):
    from app.db.plan_catalog import UNLIMITED_MESSAGE_CREDITS_MONTHLY

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    bal = await _balance(pair.grantee)
    assert bal.plan_id == "unlimited"
    assert Decimal(bal.message_credits_remaining) == Decimal(
        UNLIMITED_MESSAGE_CREDITS_MONTHLY
    )
    # plan_source mirrors the SPONSOR's source. Not cosmetic: NULL/'stripe'
    # renders as 'web', and iOS + 'web' matches none of the shipped client's
    # three branches — the grantee would see no plan surface at all.
    assert bal.plan_source == "apple"
    assert gid


async def test_grantee_is_never_denied_and_nothing_is_debited(pair):
    """Rule 1 for a grantee. Served, metered, and charged exactly zero — the
    ledger's `amount` column stays the honest record of money moved."""
    await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    res = await _charge(pair.grantee, "500")
    assert res.success is True
    bal = await _balance(pair.grantee)
    assert Decimal(bal.message_credits_remaining) == Decimal("1000000")
    assert Decimal(bal.message_credits_used_today) == Decimal("0")

    row = await _last_ledger(pair.grantee)
    assert Decimal(row.amount) == Decimal("0")
    assert Decimal(row.underlying_cost_cents) == Decimal("500"), "still metered"
    assert row.metadata_json["unlimited"] is True
    assert row.metadata_json["unlimited_reason"] == "plan"


async def test_grant_audit_fields(pair):
    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn, admin_email="ops@toup.ai")
    g = await _grant_row(gid)
    assert g.granted_to_user_id == pair.grantee
    assert g.sponsor_kind == "apple"
    assert g.sponsor_original_txn_id == pair.txn
    # Denormalised from the subscription row, so the trail still names the
    # sponsor after their account (and their apple_subscriptions row) is gone.
    assert g.sponsor_user_id == pair.sponsor
    assert g.granted_by_email == "ops@toup.ai"
    assert g.granted_by_user_id == pair.sponsor
    assert g.reason
    assert g.granted_at is not None
    assert g.revoked_at is None and g.lapsed_at is None


async def test_second_live_grant_is_refused(pair):
    from app.db import async_session_maker
    from app.services import entitlement

    await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        with pytest.raises(entitlement.GrantError) as ei:
            await entitlement.create_grant(
                db, granted_to_user_id=pair.grantee, sponsor_kind="apple",
                sponsor_original_txn_id=pair.txn, reason="again",
                granted_by_user_id=pair.sponsor, granted_by_email="ops@toup.ai",
            )
        await db.rollback()
    assert ei.value.status == 409


async def test_grant_refused_when_sponsor_is_dead(pair):
    """An expired sponsor cannot start a grant — a comp cannot outlive a
    subscription it never overlapped."""
    from app.db import async_session_maker
    from app.db.models import AppleSubscription
    from sqlalchemy import select
    from app.services import entitlement

    async with async_session_maker() as db:
        sub = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == pair.txn))).scalar_one()
        sub.expires_date = datetime.utcnow() - timedelta(days=1)
        await db.commit()

    async with async_session_maker() as db:
        with pytest.raises(entitlement.GrantError) as ei:
            await entitlement.create_grant(
                db, granted_to_user_id=pair.grantee, sponsor_kind="apple",
                sponsor_original_txn_id=pair.txn, reason="x",
                granted_by_user_id=pair.sponsor, granted_by_email="ops@toup.ai",
            )
        await db.rollback()
    assert ei.value.status == 422
    assert (await _balance(pair.grantee)).plan_id == "free"


async def test_grant_refused_over_the_grantees_own_paid_plan(pair):
    """A grant ASSIGNS the wallet and revoke returns the account to FREE, so
    granting over someone's own paid plan would silently replace it and then
    hand them a free wallet while they are still being charged."""
    from app.db import async_session_maker
    from app.services import entitlement
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.apply_plan_change(db, pair.grantee, "starter")
        await db.commit()

    async with async_session_maker() as db:
        with pytest.raises(entitlement.GrantError) as ei:
            await entitlement.create_grant(
                db, granted_to_user_id=pair.grantee, sponsor_kind="apple",
                sponsor_original_txn_id=pair.txn, reason="x",
                granted_by_user_id=pair.sponsor, granted_by_email="ops@toup.ai",
            )
        await db.rollback()
    assert ei.value.status == 422
    assert (await _balance(pair.grantee)).plan_id == "starter"


async def test_self_sponsorship_is_refused(pair):
    from app.db import async_session_maker
    from app.services import entitlement

    async with async_session_maker() as db:
        with pytest.raises(entitlement.GrantError) as ei:
            await entitlement.create_grant(
                db, granted_to_user_id=pair.sponsor, sponsor_kind="apple",
                sponsor_original_txn_id=pair.txn, reason="x",
                granted_by_user_id=pair.sponsor, granted_by_email="ops@toup.ai",
            )
        await db.rollback()
    # The sponsor holds their own live Apple entitlement, so BOTH the
    # self-sponsor guard and the own-subscription guard refuse it.
    assert ei.value.status == 422


# ── revoke ───────────────────────────────────────────────────────────


async def test_revoke_keeps_the_row_and_downgrades(pair):
    from app.db import async_session_maker
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        await entitlement.revoke_grant(
            db, gid, revoked_by_user_id=pair.sponsor,
            revoked_by_email="ops@toup.ai", revoked_reason="no longer needed",
        )
        await db.commit()

    g = await _grant_row(gid)
    assert g is not None, "the audit row must never be deleted"
    assert g.revoked_at is not None
    assert g.revoked_by_email == "ops@toup.ai"
    assert g.revoked_reason == "no longer needed"
    assert g.lapsed_at is None, "exactly one of revoked_at / lapsed_at"

    bal = await _balance(pair.grantee)
    assert bal.plan_id == "free"
    assert Decimal(bal.message_credits_remaining) == Decimal("100")
    assert Decimal(bal.message_credits_used_today) == Decimal("0")
    assert bal.plan_source is None


async def test_a_revoked_grant_grants_nothing(pair):
    from app.db import async_session_maker
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        await entitlement.revoke_grant(
            db, gid, revoked_by_user_id=pair.sponsor,
            revoked_by_email="ops@toup.ai", revoked_reason="done",
        )
        await db.commit()

    async with async_session_maker() as db:
        assert await entitlement.live_grant_for(db, pair.grantee) is None
        ok, why = await entitlement.resolve_unlimited(db, pair.grantee)
    assert (ok, why) == (False, "none")
    res = await _charge(pair.grantee, "500")
    assert res.success is False


async def test_revoke_is_idempotent(pair):
    from app.db import async_session_maker
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        first = await entitlement.revoke_grant(
            db, gid, revoked_by_user_id=pair.sponsor,
            revoked_by_email="ops@toup.ai", revoked_reason="first",
        )
        stamp = first.revoked_at
        await db.commit()
    async with async_session_maker() as db:
        again = await entitlement.revoke_grant(
            db, gid, revoked_by_user_id=pair.sponsor,
            revoked_by_email="someone.else@toup.ai", revoked_reason="second",
        )
        assert again.revoked_at == stamp
        assert again.revoked_reason == "first", "how it died is not rewritten"
        await db.commit()


async def test_a_new_grant_is_possible_after_a_revoke(pair):
    """The live-grant uniqueness index is PARTIAL. A plain unique index would
    make every user un-re-grantable forever after their first revoke, because
    the row is never deleted."""
    from app.db import async_session_maker
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        await entitlement.revoke_grant(
            db, gid, revoked_by_user_id=pair.sponsor,
            revoked_by_email="ops@toup.ai", revoked_reason="oops",
        )
        await db.commit()
    second = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    assert second != gid
    assert (await _balance(pair.grantee)).plan_id == "unlimited"


# ── lapse ────────────────────────────────────────────────────────────


async def test_grant_lapses_with_its_sponsor_on_expired(pair):
    """Rule 3: no manual step. The Apple EXPIRED notification lapses the grant
    in the SAME transaction as the sponsor's own downgrade."""
    from app.api.iap import _handle_subscription_notification
    from app.db import async_session_maker
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    assert (await _balance(pair.grantee)).plan_id == "unlimited"

    decoded = SimpleNamespace(
        notificationType="EXPIRED", subtype="VOLUNTARY",
        notificationUUID=str(uuid.uuid4()),
    )
    txn_payload = SimpleNamespace(
        productId=LEGACY_PRODUCT, originalTransactionId=pair.txn,
        expiresDate=int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000),
        environment="Production",
    )
    async with async_session_maker() as db:
        await _handle_subscription_notification(db, decoded, txn_payload, None)
        await db.commit()

    g = await _grant_row(gid)
    assert g.lapsed_at is not None
    assert g.lapsed_reason == "apple:expired:voluntary"
    assert g.revoked_at is None

    for uid in (pair.sponsor, pair.grantee):
        bal = await _balance(uid)
        assert bal.plan_id == "free", uid
        assert bal.plan_source is None

    res = await _charge(pair.grantee, "500")
    assert res.success is False

    async with async_session_maker() as db:
        assert await entitlement.live_grant_for(db, pair.grantee) is None


async def test_auto_renew_off_does_not_lapse_the_grant(pair):
    """Only EXPIRED / GRACE_PERIOD_EXPIRED / REVOKE / REFUND end anything.
    Turning auto-renew off leaves the sponsor inside a period they paid for."""
    from app.api.iap import _handle_subscription_notification
    from app.db import async_session_maker

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    decoded = SimpleNamespace(
        notificationType="DID_CHANGE_RENEWAL_STATUS",
        subtype="AUTO_RENEW_DISABLED", notificationUUID=str(uuid.uuid4()),
    )
    txn_payload = SimpleNamespace(
        productId=LEGACY_PRODUCT, originalTransactionId=pair.txn,
        expiresDate=int((datetime.utcnow() + timedelta(days=20)).timestamp() * 1000),
        environment="Production",
    )
    async with async_session_maker() as db:
        await _handle_subscription_notification(
            db, decoded, txn_payload,
            SimpleNamespace(autoRenewStatus=0, autoRenewProductId=None,
                            gracePeriodExpiresDate=None),
        )
        await db.commit()

    g = await _grant_row(gid)
    assert g.lapsed_at is None
    assert (await _balance(pair.grantee)).plan_id == "unlimited"


async def test_deleted_sponsor_fails_closed(pair):
    """apple_subscriptions.user_id is ON DELETE CASCADE, so deleting the
    sponsor destroys the subscription row. The grant row must SURVIVE (no FK)
    while reading as dead (missing sponsor is not live)."""
    from app.db import User, async_session_maker
    from app.services import entitlement

    from sqlalchemy import delete
    from app.db.models import AppleSubscription

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        # The subscription row is deleted EXPLICITLY: SQLite does not enforce
        # ON DELETE CASCADE without `PRAGMA foreign_keys=ON`, and the state
        # under test is the post-cascade one Postgres produces — a grant whose
        # sponsor_original_txn_id names a row that is gone.
        #
        # Core DELETE, not session.delete(obj): the ORM cascade walks every
        # User relationship to nullify child FKs, and `User.memories` is an
        # AGENT_ONLY table that init_db does not create under RUN_MODE=platform
        # — the lane the sweep runs this file in. The row-level effect is
        # identical, and the sponsor being gone is the whole state under test.
        await db.execute(delete(AppleSubscription).where(
            AppleSubscription.original_transaction_id == pair.txn))
        await db.execute(delete(User).where(User.id == pair.sponsor))
        await db.commit()

    g = await _grant_row(gid)
    assert g is not None, "the audit row outlives the sponsor it names"
    assert g.sponsor_original_txn_id == pair.txn

    async with async_session_maker() as db:
        assert await entitlement.live_grant_for(db, pair.grantee) is None
        # The row is still OPEN — the reconciler is what closes it. That is the
        # difference open_grant_for exists to express.
        assert await entitlement.open_grant_for(db, pair.grantee) is not None
        lapsed = await entitlement.lapse_grants_for_sponsor(
            db, original_txn_id=pair.txn, reason="apple:sponsor_deleted",
        )
        await db.commit()
    assert len(lapsed) == 1
    assert (await _balance(pair.grantee)).plan_id == "free"


async def test_expires_at_backstop_ends_the_grant(pair):
    from app.db import async_session_maker
    from app.services import entitlement

    async with async_session_maker() as db:
        grant = await entitlement.create_grant(
            db, granted_to_user_id=pair.grantee, sponsor_kind="apple",
            sponsor_original_txn_id=pair.txn, reason="90-day trial comp",
            expires_at=datetime.utcnow() - timedelta(minutes=1),
            granted_by_user_id=pair.sponsor, granted_by_email="ops@toup.ai",
        )
        gid = grant.id
        await db.commit()
    async with async_session_maker() as db:
        assert await entitlement.live_grant_for(db, pair.grantee) is None
        assert (await db.get(type(grant), gid)).expires_at is not None


async def test_lapse_leaves_an_independently_entitled_grantee_alone(pair):
    """The worst available outcome is downgrading someone who is paying. If the
    grantee has since bought their own subscription, ending the comp must not
    touch their entitlement."""
    from app.db import async_session_maker
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    own_txn = await _mk_apple_sub(pair.grantee, product="ai.toup.app.sub.unlimited")
    async with async_session_maker() as db:
        await entitlement.lapse_grants_for_sponsor(
            db, original_txn_id=pair.txn, reason="apple:expired",
        )
        await db.commit()

    assert (await _grant_row(gid)).lapsed_at is not None
    assert (await _balance(pair.grantee)).plan_id == "unlimited", (
        "the grantee pays for their own now — ending the comp must not "
        "downgrade them"
    )
    assert own_txn


# ── the endpoints ────────────────────────────────────────────────────


async def _promote(user_id: str) -> None:
    from app.db import User, async_session_maker
    async with async_session_maker() as db:
        (await db.get(User, user_id)).role = "admin"
        await db.commit()


async def test_endpoints_are_admin_gated(client, auth_headers, test_user_id, pair):
    r = await client.post(
        "/api/admin/unlimited-grants", headers=auth_headers,
        json={"user_id": pair.grantee, "sponsor_kind": "apple",
              "sponsor_original_transaction_id": pair.txn, "reason": "x"},
    )
    assert r.status_code == 403
    assert (await client.get("/api/admin/unlimited-grants",
                             headers=auth_headers)).status_code == 403


async def test_endpoint_grant_list_revoke(client, auth_headers, test_user_id, pair):
    await _promote(test_user_id)

    r = await client.post(
        "/api/admin/unlimited-grants", headers=auth_headers,
        json={"email": "second-account@example.com", "sponsor_kind": "apple",
              "sponsor_original_transaction_id": pair.txn,
              "reason": "family plan"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["live"] is True
    assert body["granted_to_user_id"] == pair.grantee
    assert body["granted_by_email"]
    gid = body["id"]
    assert (await _balance(pair.grantee)).plan_id == "unlimited"

    # A second one is refused with the service's own 409, not an index 500.
    dup = await client.post(
        "/api/admin/unlimited-grants", headers=auth_headers,
        json={"user_id": pair.grantee, "sponsor_kind": "apple",
              "sponsor_original_transaction_id": pair.txn, "reason": "again"},
    )
    assert dup.status_code == 409

    listed = await client.get("/api/admin/unlimited-grants", headers=auth_headers)
    assert listed.status_code == 200
    assert [g["id"] for g in listed.json()["grants"]] == [gid]

    r = await client.request(
        "DELETE", f"/api/admin/unlimited-grants/{gid}", headers=auth_headers,
        json={"revoked_reason": "moved out"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["revoked_reason"] == "moved out"
    assert r.json()["live"] is False
    assert (await _balance(pair.grantee)).plan_id == "free"

    # Unfiltered by default: the revoked row is the record.
    listed = await client.get("/api/admin/unlimited-grants", headers=auth_headers)
    assert len(listed.json()["grants"]) == 1
    live_only = await client.get(
        "/api/admin/unlimited-grants?live=true", headers=auth_headers)
    assert live_only.json()["grants"] == []


async def test_refused_grant_leaves_no_row(client, auth_headers, test_user_id, pair):
    """The service flushes the grant row before checking sponsor liveness, so
    a refusal that lands after the flush must roll it back — otherwise the
    partial unique index blocks every later grant for that user."""
    from app.db import async_session_maker
    from app.db.models import AppleSubscription
    from sqlalchemy import select

    await _promote(test_user_id)
    async with async_session_maker() as db:
        sub = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == pair.txn))).scalar_one()
        sub.expires_date = datetime.utcnow() - timedelta(days=1)
        await db.commit()

    r = await client.post(
        "/api/admin/unlimited-grants", headers=auth_headers,
        json={"user_id": pair.grantee, "sponsor_kind": "apple",
              "sponsor_original_transaction_id": pair.txn, "reason": "x"},
    )
    assert r.status_code == 422
    listed = await client.get("/api/admin/unlimited-grants", headers=auth_headers)
    assert listed.json()["grants"] == []


# ── the CLI ──────────────────────────────────────────────────────────


async def test_cli_refuses_a_non_admin_actor(pair, capsys):
    from app.scripts.unlimited_grant import run

    with pytest.raises(SystemExit) as ei:
        await run([
            "grant", "--user", pair.grantee,
            "--sponsor-apple-txn", pair.txn,
            "--reason", "x", "--as-admin", "sponsor@example.com", "--apply",
        ])
    assert "not admin" in str(ei.value)
    assert (await _balance(pair.grantee)).plan_id == "free"


async def test_cli_dry_run_writes_nothing(pair, capsys):
    from app.scripts.unlimited_grant import run

    await _promote(pair.sponsor)
    assert await run([
        "grant", "--user", "second-account@example.com",
        "--sponsor-apple-txn", pair.txn,
        "--reason", "family", "--as-admin", "sponsor@example.com",
    ]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert (await _balance(pair.grantee)).plan_id == "free"


async def test_cli_grant_and_revoke(pair, capsys):
    from app.scripts.unlimited_grant import run

    await _promote(pair.sponsor)
    assert await run([
        "grant", "--user", "second-account@example.com",
        "--sponsor-apple-txn", pair.txn,
        "--reason", "family", "--as-admin", "sponsor@example.com", "--apply",
    ]) == 0
    assert (await _balance(pair.grantee)).plan_id == "unlimited"

    gid = None
    from app.db import async_session_maker
    from app.services import entitlement
    async with async_session_maker() as db:
        gid = (await entitlement.open_grant_for(db, pair.grantee)).id

    assert await run([
        "revoke", gid, "--reason", "done",
        "--as-admin", "sponsor@example.com", "--apply",
    ]) == 0
    assert (await _balance(pair.grantee)).plan_id == "free"
    g = await _grant_row(gid)
    assert g is not None and g.revoked_at is not None


# ── the controller-side resolver ─────────────────────────────────────


async def test_resolve_unlimited_names_the_grant(pair):
    """The reconciler's question. A grantee must resolve as 'grant', not as
    'plan' — 'plan' is the orphan-entitlement class (a stamp with nothing
    behind it), and confusing the two is how a comp becomes permanent."""
    from app.db import async_session_maker
    from app.services import entitlement

    await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        assert await entitlement.resolve_unlimited(db, pair.grantee) == (True, "grant")
        # The sponsor holds it on their own Apple row, which is a legacy
        # product with no grandfather stamp — so NOT unlimited.
        assert await entitlement.resolve_unlimited(db, pair.sponsor) == (False, "none")


async def test_missing_table_reads_as_no_override(pair):
    """A replica that boots before the deploy's migration finishes must answer
    "no override" and keep working — not 500, and not poison the session's
    transaction on the way (a swallowed ProgrammingError aborts a Postgres
    transaction, so every statement AFTER it fails, nowhere near the cause)."""
    from sqlalchemy import text
    from app.db import async_session_maker
    from app.services import entitlement

    entitlement._grants_table_present_cache = False
    async with async_session_maker() as db:
        await db.execute(text("DROP TABLE unlimited_grants"))
        await db.commit()

    try:
        async with async_session_maker() as db:
            assert await entitlement.live_grant_for(db, pair.grantee) is None
            assert await entitlement.open_grant_for(db, pair.grantee) is None
            assert await entitlement.grants_sponsored_by(
                db, original_txn_id=pair.txn) == []
            assert await entitlement.list_grants(db) == []
            assert await entitlement.lapse_grants_for_sponsor(
                db, original_txn_id=pair.txn, reason="x") == []
            # The session is still usable — that is the half a caught
            # ProgrammingError would have destroyed.
            assert (await db.execute(text("SELECT 1"))).scalar() == 1
            ok, why = await entitlement.resolve_unlimited(db, pair.grantee)
            assert (ok, why) == (False, "none")
    finally:
        entitlement._grants_table_present_cache = False


# ── the sweep (the backstop for a channel that has already failed) ───


async def test_sweep_closes_a_grant_whose_expires_at_passed(pair):
    """A backstop has no notification by definition — only the sweep can see
    it pass."""
    from app.db import async_session_maker
    from app.services import entitlement

    async with async_session_maker() as db:
        g = await entitlement.create_grant(
            db, granted_to_user_id=pair.grantee, sponsor_kind="apple",
            sponsor_original_txn_id=pair.txn, reason="90-day comp",
            expires_at=datetime.utcnow() + timedelta(days=90),
            granted_by_user_id=pair.sponsor, granted_by_email="ops@toup.ai",
        )
        gid = g.id
        await db.commit()
    assert (await _balance(pair.grantee)).plan_id == "unlimited"

    later = datetime.utcnow() + timedelta(days=91)
    async with async_session_maker() as db:
        would = await entitlement.sweep_dead_grants(db, now=later, apply=False)
        await db.commit()
    assert would == [(gid, "expires_at")]
    assert (await _grant_row(gid)).lapsed_at is None, "apply=False writes nothing"
    assert (await _balance(pair.grantee)).plan_id == "unlimited"

    async with async_session_maker() as db:
        done = await entitlement.sweep_dead_grants(db, now=later, apply=True)
        await db.commit()
    assert done == [(gid, "expires_at")]
    assert (await _grant_row(gid)).lapsed_reason == "expires_at"
    assert (await _balance(pair.grantee)).plan_id == "free"

    # Idempotent: the row is closed, so a second pass finds nothing.
    async with async_session_maker() as db:
        assert await entitlement.sweep_dead_grants(db, now=later, apply=True) == []


async def test_sweep_closes_a_grant_whose_sponsor_lapsed_unnoticed(pair):
    """The Sandbox Elite proves the notification channel drops messages. When
    it drops a lapse, the sweep is what ends the comp."""
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import AppleSubscription
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        sub = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == pair.txn))).scalar_one()
        sub.expires_date = datetime.utcnow() - timedelta(days=2)
        # The mirror still LIES about status — exactly the Sandbox Elite shape.
        sub.status = "active"
        await db.commit()

    async with async_session_maker() as db:
        done = await entitlement.sweep_dead_grants(db, apply=True)
        await db.commit()
    assert done == [(gid, "sponsor_dead:apple")]
    assert (await _balance(pair.grantee)).plan_id == "free"
    assert (await _charge(pair.grantee, "500")).success is False


async def test_sweep_leaves_a_live_grant_alone(pair):
    from app.db import async_session_maker
    from app.services import entitlement

    gid = await _make_grant(pair.grantee, pair.sponsor, pair.txn)
    async with async_session_maker() as db:
        assert await entitlement.sweep_dead_grants(db, apply=True) == []
        await db.commit()
    assert (await _grant_row(gid)).lapsed_at is None
    assert (await _balance(pair.grantee)).plan_id == "unlimited"
