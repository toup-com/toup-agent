"""Free-credit Sybil / multi-account abuse — proves the gaps are closed.

These tests exercise the real CreditService against the in-memory test DB
(conftest seeds the 'free' plan = 100 msg / 500 integration). They assert
the abuse vectors from SIGNUP_CREDIT_ABUSE_AUDIT.md are shut:

* Gmail dot/+alias variants → exactly ONE grant.
* delete → re-signup (same + aliased email) → NO new grant.
* two concurrent same-canonical signups → exactly ONE grant (race).
* OAuth/provider-verified → instant grant.
* password signup → grant deferred until verify; balance row still exists
  (product entry never blocked).
* monthly renewal on a LIVE account → still grants (regression guard).
* default flags OFF → legacy "grant at creation" preserved.
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select

pytestmark = pytest.mark.asyncio


# ── helpers ───────────────────────────────────────────────────────────

async def _mk_user(email: str, *, verified: bool = False) -> str:
    from datetime import datetime
    from app.db import async_session_maker, User
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=email.strip().lower(), hashed_password="x", name="t",
            email_verified_at=(datetime.utcnow() if verified else None),
        ))
        await db.commit()
    return uid


async def _balance_msg(uid: str) -> Decimal:
    from app.db import async_session_maker, CreditBalance
    async with async_session_maker() as db:
        b = await db.get(CreditBalance, uid)
        return Decimal(b.message_credits_remaining) if b else Decimal("-1")


async def _tomb_count() -> int:
    from app.db import async_session_maker
    from app.db.models import GrantEligibility
    async with async_session_maker() as db:
        return int((await db.execute(
            select(func.count()).select_from(GrantEligibility)
        )).scalar() or 0)


async def _grant_via_balance(uid: str) -> None:
    """Mimic signup: get_or_create_balance runs _maybe_grant_initial."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, uid)
        await db.commit()


# ── canonicalization unit pins ────────────────────────────────────────

def test_canonicalize_collapses_gmail_dots_and_tags():
    from app.services.email_canonical import canonicalize_email, canonical_email_hash
    assert canonicalize_email("U.S.E.R+promo@googlemail.com") == "user@gmail.com"
    assert canonicalize_email("a.b+x@fastmail.com") == "a.b@fastmail.com"  # listed: +tag stripped, dots kept
    assert canonical_email_hash("u.s.e.r@gmail.com") == canonical_email_hash("user+1@gmail.com")
    assert canonical_email_hash("user@gmail.com") != canonical_email_hash("other@gmail.com")


def test_canonicalizer_no_false_merge_on_unlisted_domain():
    """B3: +tag/dot collapsing must apply ONLY to known subaddressing
    providers. Two genuinely-distinct accounts at an unlisted domain must
    NEVER map to the same grant identity (a false-merge denies a real user
    their grant)."""
    from app.services.email_canonical import canonicalize_email, canonical_email_hash
    # Unlisted domain: +tag and dots are PRESERVED (not collapsed).
    assert canonicalize_email("a+promo@acme-corp.com") == "a+promo@acme-corp.com"
    assert canonical_email_hash("a+promo@acme-corp.com") != canonical_email_hash("a@acme-corp.com")
    assert canonical_email_hash("a.b@acme-corp.com") != canonical_email_hash("ab@acme-corp.com")
    # Two distinct humans at an unlisted domain never merge.
    assert canonical_email_hash("alice@acme-corp.com") != canonical_email_hash("alice+x@acme-corp.com")
    # But listed providers still collapse (coverage preserved where it's safe).
    assert canonical_email_hash("p+a@outlook.com") == canonical_email_hash("p@outlook.com")
    assert canonical_email_hash("q+a@icloud.com") == canonical_email_hash("q@icloud.com")


async def test_background_verification_email_completes(monkeypatch):
    """B1: the off-hot-path verification send must run to COMPLETION (not be
    GC'd as a stray create_task) so a password user actually gets the
    grant-unlock link."""
    import asyncio
    import app.api.email_verification as ev
    from app.db import async_session_maker, User

    sent = []
    async def _fake_send(**kw):
        sent.append(kw)
    monkeypatch.setattr(ev, "send_email", _fake_send)

    uid = await _mk_user("bguser@gmail.com", verified=False)
    ev.schedule_post_register_verification(uid)
    assert ev._PENDING_SENDS, "scheduled task must be pinned (strong ref), not GC-able"
    await asyncio.gather(*list(ev._PENDING_SENDS))   # drain to completion

    assert len(sent) == 1, "verification email must dispatch after the caller returned"
    async with async_session_maker() as db:
        u = await db.get(User, uid)
        assert u.email_verification_token is not None   # token persisted for /verify-email


async def test_abuse_metrics_events_and_pii_safety(monkeypatch, caplog):
    """C: controls emit structured events, fire SHADOW would-events while
    their flag is OFF (so false-positive rate is measurable pre-rollout),
    and never leak a raw email."""
    import logging
    from app.config import settings
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", False, raising=False)  # shadow mode
    with caplog.at_level(logging.INFO, logger="abuse_controls"):
        u1 = await _mk_user("metrics@gmail.com")
        await _grant_via_balance(u1)                       # → tombstone_claimed
        u2 = await _mk_user("m.e.t.r.i.c.s+x@gmail.com")   # same canonical inbox
        await _grant_via_balance(u2)                       # → grant_suppressed_would (flag OFF)
    msgs = " ".join(r.getMessage() for r in caplog.records if r.name == "abuse_controls")
    assert "event=tombstone_claimed" in msgs
    assert "event=grant_suppressed_would" in msgs
    # PII safety: never emit a raw email / local-part.
    assert "metrics@gmail.com" not in msgs and "m.e.t.r.i.c.s" not in msgs


async def test_reconciliation_grants_verified_deferred(monkeypatch):
    """B2: a user who became verified after a missed grant is swept up."""
    from datetime import datetime
    from app.config import settings
    from app.db import async_session_maker, User
    from app.services.credit_service import credit_service
    monkeypatch.setattr(settings, "require_verified_email_for_grant", True, raising=False)

    u = await _mk_user("deferred@gmail.com", verified=False)
    await _grant_via_balance(u)                       # deferred → balance 0, no grant
    assert await _balance_msg(u) == Decimal("0")
    # user verifies but (simulating a failed on-verify grant) is NOT granted yet
    async with async_session_maker() as db:
        usr = await db.get(User, u); usr.email_verified_at = datetime.utcnow(); await db.commit()
    assert await _balance_msg(u) == Decimal("0")
    # the sweep grants them
    async with async_session_maker() as db:
        n = await credit_service.reconcile_deferred_grants(db)
    assert n == 1
    assert await _balance_msg(u) == Decimal("100")


async def test_reconciliation_skips_unverified_and_suppressed(monkeypatch):
    """B2: the sweep never over-grants — unverified-and-gated users stay
    deferred, and aliases of an already-granted identity stay suppressed."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    monkeypatch.setattr(settings, "require_verified_email_for_grant", True, raising=False)
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", True, raising=False)

    # (a) verified user → granted at signup; (b) alias of them, verified → suppressed;
    # (c) unverified user → deferred.
    a = await _mk_user("base@gmail.com", verified=True)
    await _grant_via_balance(a)
    assert await _balance_msg(a) == Decimal("100")
    b = await _mk_user("b.a.s.e+x@gmail.com", verified=True)
    await _grant_via_balance(b)
    assert await _balance_msg(b) == Decimal("0")      # suppressed alias
    c = await _mk_user("unverified@gmail.com", verified=False)
    await _grant_via_balance(c)
    assert await _balance_msg(c) == Decimal("0")      # deferred

    async with async_session_maker() as db:
        n = await credit_service.reconcile_deferred_grants(db)
    assert n == 0, "no eligible deferred grants — nothing to do"
    assert await _balance_msg(b) == Decimal("0")      # still suppressed
    assert await _balance_msg(c) == Decimal("0")      # still deferred


def test_disposable_blocklist_core():
    from app.services.disposable_email import is_disposable_email
    assert is_disposable_email("x@mailinator.com") is True
    assert is_disposable_email("x@GUERRILLAMAIL.com") is True
    assert is_disposable_email("x@gmail.com") is False
    assert is_disposable_email("garbage") is False


# ── alias multiplication → one grant ──────────────────────────────────

async def test_alias_variants_get_one_grant(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", True, raising=False)

    u1 = await _mk_user("user@gmail.com")
    u2 = await _mk_user("u.s.e.r+promo@gmail.com")   # same canonical inbox
    u3 = await _mk_user("US.ER@googlemail.com")        # same canonical inbox
    await _grant_via_balance(u1)
    await _grant_via_balance(u2)
    await _grant_via_balance(u3)

    assert await _balance_msg(u1) == Decimal("100")   # first one granted
    assert await _balance_msg(u2) == Decimal("0")     # suppressed
    assert await _balance_msg(u3) == Decimal("0")     # suppressed
    assert await _tomb_count() == 1                     # one identity claimed


# ── delete → re-signup (same + aliased) → no new grant ────────────────

async def test_delete_resignup_no_new_grant(monkeypatch):
    from app.config import settings
    from app.db import async_session_maker, User, CreditBalance, CreditLedger
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", True, raising=False)

    u1 = await _mk_user("farmer@gmail.com")
    await _grant_via_balance(u1)
    assert await _balance_msg(u1) == Decimal("100")
    assert await _tomb_count() == 1

    # Simulate hard delete (users cascade wipes balance + ledger; the
    # grant_eligibility tombstone has NO FK and is NOT in the wipe list).
    async with async_session_maker() as db:
        await db.execute(delete(CreditLedger).where(CreditLedger.user_id == u1))
        await db.execute(delete(CreditBalance).where(CreditBalance.user_id == u1))
        await db.execute(delete(User).where(User.id == u1))
        await db.commit()
    assert await _tomb_count() == 1   # tombstone survived deletion

    # Re-register the SAME email → no fresh grant.
    u1b = await _mk_user("farmer@gmail.com")
    await _grant_via_balance(u1b)
    assert await _balance_msg(u1b) == Decimal("0")

    # And an ALIAS of it → also no fresh grant.
    u1c = await _mk_user("f.a.r.m.e.r+again@gmail.com")
    await _grant_via_balance(u1c)
    assert await _balance_msg(u1c) == Decimal("0")
    assert await _tomb_count() == 1


# ── race resolution: a claimed canonical identity suppresses the racer ─

async def test_race_loser_is_suppressed(monkeypatch):
    """The unique grant_eligibility PK is the race arbiter. We model the
    instant AFTER the winning transaction commits its tombstone: a second
    same-canonical signup arriving now must be suppressed (zero grant, no
    second tombstone). On Postgres the savepoint INSERT blocks-then-raises
    to reach this same state; sqlite's shared in-memory DB can't host two
    real concurrent writers, so we assert the resolved outcome directly —
    DB-agnostic and exactly what a real loser observes.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", True, raising=False)

    winner = await _mk_user("race@gmail.com")
    await _grant_via_balance(winner)          # winner commits its tombstone
    assert await _balance_msg(winner) == Decimal("100")
    assert await _tomb_count() == 1

    loser = await _mk_user("r.a.c.e+x@gmail.com")   # same canonical inbox
    await _grant_via_balance(loser)
    assert await _balance_msg(loser) == Decimal("0")   # suppressed, not double-granted
    assert await _tomb_count() == 1


# ── OAuth/provider-verified → instant grant ───────────────────────────

async def test_oauth_provider_verified_instant_grant(monkeypatch):
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    monkeypatch.setattr(settings, "require_verified_email_for_grant", True, raising=False)

    # OAuth path: user created unverified at create_user time → deferred.
    u = await _mk_user("oauth@outlook.com", verified=False)
    await _grant_via_balance(u)
    assert await _balance_msg(u) == Decimal("0")

    # Handler then stamps email_verified_at and fires the grant explicitly.
    from datetime import datetime
    from app.db import User
    async with async_session_maker() as db:
        usr = await db.get(User, u)
        usr.email_verified_at = datetime.utcnow()
        await db.commit()
        granted = await credit_service.grant_initial_free_credits(db, u, email_verified=True)
        await db.commit()
    assert granted is True
    assert await _balance_msg(u) == Decimal("100")


# ── password signup → grant deferred until verify, entry unblocked ────

async def test_password_grant_deferred_until_verify(monkeypatch):
    from app.config import settings
    from app.db import async_session_maker, User
    from app.services.credit_service import credit_service
    from datetime import datetime
    monkeypatch.setattr(settings, "require_verified_email_for_grant", True, raising=False)

    u = await _mk_user("pwuser@proton.me", verified=False)
    await _grant_via_balance(u)
    # Product entry NOT blocked: a balance row exists (just zero credits).
    assert await _balance_msg(u) == Decimal("0")

    # Calling grant while still unverified is a no-op.
    async with async_session_maker() as db:
        g = await credit_service.grant_initial_free_credits(db, u, email_verified=False)
        await db.commit()
    assert g is False
    assert await _balance_msg(u) == Decimal("0")

    # Clicking the verification link unlocks the grant.
    async with async_session_maker() as db:
        usr = await db.get(User, u)
        usr.email_verified_at = datetime.utcnow()
        await db.commit()
        g = await credit_service.grant_initial_free_credits(db, u, email_verified=True)
        await db.commit()
    assert g is True
    assert await _balance_msg(u) == Decimal("100")


# ── monthly renewal on a live account → still grants (regression) ─────

async def test_monthly_renewal_unaffected(monkeypatch):
    from datetime import datetime, timedelta
    from app.config import settings
    from app.db import async_session_maker, CreditBalance
    from app.services.credit_service import credit_service
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", True, raising=False)

    u = await _mk_user("renewer@gmail.com")
    await _grant_via_balance(u)
    assert await _balance_msg(u) == Decimal("100")

    # Spend down + push the period boundary into the past, then renew.
    async with async_session_maker() as db:
        b = await db.get(CreditBalance, u)
        b.message_credits_remaining = Decimal("4")
        b.period_end = datetime.utcnow() - timedelta(seconds=1)
        await db.commit()
        renewed = await credit_service.renew_period(db, u)
        await db.commit()
    assert renewed is True
    assert await _balance_msg(u) == Decimal("100")   # renewed, NOT suppressed by tombstone
    assert await _tomb_count() == 1


# ── default flags OFF → legacy grant-at-creation preserved ────────────

async def test_default_flags_preserve_legacy_grant(monkeypatch):
    from app.config import settings
    # Defaults are False, but pin them so the test is explicit + isolated.
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", False, raising=False)
    monkeypatch.setattr(settings, "require_verified_email_for_grant", False, raising=False)

    u = await _mk_user("legacy@gmail.com", verified=False)
    await _grant_via_balance(u)
    # Legacy behavior: granted at creation regardless of verification.
    assert await _balance_msg(u) == Decimal("100")
    # Tombstone still recorded (so enabling dedupe later is safe).
    assert await _tomb_count() == 1
