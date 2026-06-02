"""Unit + integration tests for the credit-system core.

Covers:
* tokens_to_credits: Decimal sub-cent precision (0.2 credit floor, etc.)
* try_charge: deduction, balance update, ledger row, idempotency,
  insufficient-balance + daily-cap + email-unverified denial paths,
  shadow-mode (enforcement=False) writes ledger without deducting.
* reserve + settle: two-phase commit with refund on overage.
* refund: returns full reservation on cancellation.
* grant + apply_plan_change + renew_period.

Uses the shared conftest test_user fixture so SQLite is in-memory and
each test gets a fresh database.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
import pytest_asyncio


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def credit_user(test_user_id: str):
    """Return the user_id along with a freshly-seeded CreditBalance.

    The credit_service.get_or_create_balance call lazy-creates the
    free-tier row with 100 message + 500 integration credits (post
    mig-059), so the fixture just primes it once.
    """
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, test_user_id)
        await db.commit()
    return test_user_id


async def _balance(user_id: str):
    from app.db import async_session_maker, CreditBalance
    async with async_session_maker() as db:
        return await db.get(CreditBalance, user_id)


async def _ledger_count(user_id: str) -> int:
    from sqlalchemy import select, func
    from app.db import async_session_maker, CreditLedger
    async with async_session_maker() as db:
        n = await db.execute(
            select(func.count(CreditLedger.id)).where(CreditLedger.user_id == user_id)
        )
        return int(n.scalar() or 0)


# ──────────────────────────────────────────────────────────────────────
# tokens_to_credits — Decimal precision
# ──────────────────────────────────────────────────────────────────────


def test_tokens_to_credits_sub_cent_floor():
    """A tiny call (100 input tokens of a cheap model) must NOT round to 1 credit."""
    from app.services.credit_service import tokens_to_credits
    # claude-haiku at ~$1/Mtok input; 100 tokens ≈ 0.01¢ underlying.
    # Floor is 0.1 credit (the display quantum), so result must be 0.1.
    credits = tokens_to_credits("claude-haiku-4-5-20251001", 100, 0)
    assert credits == Decimal("0.1")


def test_tokens_to_credits_large_call_scales():
    """A real-size call should produce >1 credit and quantize to 0.1."""
    from app.services.credit_service import tokens_to_credits
    credits = tokens_to_credits("claude-sonnet-4-6", 20_000, 5_000)
    # Must be > 0.1 and a multiple of 0.1
    assert credits > Decimal("0.1")
    assert credits % Decimal("0.1") == Decimal("0")


def test_tokens_to_credits_unknown_model_uses_fallback():
    """Unknown model uses the conservative fallback pricing — never 0."""
    from app.services.credit_service import tokens_to_credits
    credits = tokens_to_credits("gpt-9000-not-real", 1000, 1000)
    assert credits >= Decimal("0.1")


# ──────────────────────────────────────────────────────────────────────
# try_charge — happy path + denial paths
# ──────────────────────────────────────────────────────────────────────


async def test_try_charge_deducts_and_writes_ledger(credit_user, monkeypatch):
    """Shadow-mode (default) writes the ledger row AND deducts (because no deny)."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import (
        BUCKET_MESSAGE, credit_service,
    )
    monkeypatch.setattr(settings, "credit_enforcement_enabled", False, raising=False)

    before = await _balance(credit_user)
    before_remaining = Decimal(before.message_credits_remaining)

    async with async_session_maker() as db:
        result = await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("1.5"),
            idempotency_key="t1",
        )
        await db.commit()

    assert result.success is True
    after = await _balance(credit_user)
    assert Decimal(after.message_credits_remaining) == before_remaining - Decimal("1.5")
    assert await _ledger_count(credit_user) >= 1


async def test_try_charge_idempotent(credit_user, monkeypatch):
    """Same idempotency_key on a retry returns success with idempotent_hit=True."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import (
        BUCKET_MESSAGE, credit_service,
    )
    monkeypatch.setattr(settings, "credit_enforcement_enabled", False, raising=False)

    async with async_session_maker() as db:
        r1 = await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("2.0"),
            idempotency_key="dup",
        )
        await db.commit()
    async with async_session_maker() as db:
        r2 = await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("2.0"),
            idempotency_key="dup",
        )
        await db.commit()
    assert r1.success is True
    assert r2.success is True and r2.idempotent_hit is True


async def test_try_charge_denies_when_enforced_and_insufficient(credit_user, monkeypatch):
    """Enforcement ON + amount > remaining → denied with insufficient_message_credits.

    Free tier has a daily_cap (15 post-mig-059) lower than the monthly
    allotment (100), so draining the monthly bucket via repeated small
    charges would trip DAILY_CAP_EXCEEDED first, masking the INSUFFICIENT
    branch we're exercising. The check order in try_charge is
    insufficient → daily_cap (line ~376 → 378), so charging a single
    amount strictly greater than monthly_remaining fires INSUFFICIENT
    even when that amount also exceeds the daily cap.
    """
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import (
        BUCKET_MESSAGE, REASON_INSUFFICIENT_MESSAGE, credit_service,
    )
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)

    async with async_session_maker() as db:
        # Try to spend more than the entire monthly grant (100 credits
        # on the post-mig-059 free tier).
        result = await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("101"),
            idempotency_key="should-deny",
        )
        await db.commit()
    assert result.success is False
    assert result.reason == REASON_INSUFFICIENT_MESSAGE


async def test_try_charge_daily_cap_enforced(credit_user, monkeypatch):
    """Free tier daily_cap = 15 (post-mig-059) → spending 16 in one day is denied."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import (
        BUCKET_MESSAGE, REASON_DAILY_CAP_EXCEEDED, credit_service,
    )
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)

    async with async_session_maker() as db:
        await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("14.5"),
            idempotency_key="d-a",
        )
        await db.commit()
    async with async_session_maker() as db:
        # 14.5 already, plus 1.0 = 15.5 > daily cap of 15
        result = await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("1.0"),
            idempotency_key="d-b",
        )
        await db.commit()
    assert result.success is False
    assert result.reason == REASON_DAILY_CAP_EXCEEDED


# ──────────────────────────────────────────────────────────────────────
# reserve + settle + refund
# ──────────────────────────────────────────────────────────────────────


async def test_reserve_then_settle_refunds_overage(credit_user, monkeypatch):
    """Reserve 10, settle for 6 → 4 refunded; remaining unchanged net of 6."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import BUCKET_MESSAGE, credit_service
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)

    before = Decimal((await _balance(credit_user)).message_credits_remaining)

    async with async_session_maker() as db:
        r = await credit_service.reserve(
            db, credit_user, "auto_builder", BUCKET_MESSAGE, Decimal("10"),
            idempotency_key="res-1",
        )
        await db.commit()
    assert r.success
    held = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert held == before - Decimal("10")

    async with async_session_maker() as db:
        actual = await credit_service.settle(db, r.reservation_id, Decimal("6"))
        await db.commit()
    assert actual == Decimal("6")

    final = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert final == before - Decimal("6")  # 4 refunded


async def test_refund_returns_full_reservation(credit_user, monkeypatch):
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import BUCKET_INTEGRATION, credit_service
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)

    before = Decimal((await _balance(credit_user)).integration_credits_remaining)

    async with async_session_maker() as db:
        r = await credit_service.reserve(
            db, credit_user, "connector_op", BUCKET_INTEGRATION, Decimal("5"),
            idempotency_key="res-cancel",
        )
        await db.commit()
    async with async_session_maker() as db:
        await credit_service.refund(db, r.reservation_id, reason="user_cancelled")
        await db.commit()
    after = Decimal((await _balance(credit_user)).integration_credits_remaining)
    assert after == before


# ──────────────────────────────────────────────────────────────────────
# grant + plan change
# ──────────────────────────────────────────────────────────────────────


async def test_grant_increases_balance(credit_user):
    from app.db import async_session_maker
    from app.services.credit_service import BUCKET_MESSAGE, credit_service

    before = Decimal((await _balance(credit_user)).message_credits_remaining)
    async with async_session_maker() as db:
        await credit_service.grant(
            db, credit_user, BUCKET_MESSAGE, Decimal("50"),
            metadata={"reason": "promo"},
        )
        await db.commit()
    after = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert after == before + Decimal("50")


async def test_apply_plan_change_to_paid_tier(credit_user):
    """Upgrade free → builder mid-period prorates the delta into the bucket."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.apply_plan_change(
            db, credit_user, "builder", reason="test_upgrade",
        )
        await db.commit()
    b = await _balance(credit_user)
    assert b.plan_id == "builder"
    # Daily cap drops away on paid tiers (None or higher than the
    # free-tier 15 post mig-059).
    assert b.message_credits_daily_cap is None or Decimal(b.message_credits_daily_cap) > Decimal("15")


# ──────────────────────────────────────────────────────────────────────
# admin = unlimited (never denied, never deducted)
# ──────────────────────────────────────────────────────────────────────


async def _make_admin(user_id: str) -> None:
    from app.db import async_session_maker
    from app.db.models import User
    async with async_session_maker() as db:
        u = await db.get(User, user_id)
        u.role = "admin"
        await db.commit()


async def test_admin_try_charge_never_denied_never_deducted(credit_user, monkeypatch):
    """An admin charging MORE than their entire balance, with enforcement ON,
    must SUCCEED and leave the balance completely untouched (no limit)."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import BUCKET_MESSAGE, credit_service
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)
    await _make_admin(credit_user)

    before = Decimal((await _balance(credit_user)).message_credits_remaining)
    async with async_session_maker() as db:
        # 9999 ≫ the 100-credit free grant — a non-admin would be denied.
        result = await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("9999"),
            idempotency_key="admin-unlimited-1",
        )
        await db.commit()
    assert result.success is True, "admin must never be denied"
    after = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert after == before, "admin charge must NOT deduct — balance unchanged"


async def test_admin_daily_cap_does_not_apply(credit_user, monkeypatch):
    """The daily cap must not gate an admin either."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_CHAT_MESSAGE
    from app.services.credit_service import BUCKET_MESSAGE, credit_service
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)
    await _make_admin(credit_user)
    async with async_session_maker() as db:
        # Way over the 15/day free cap — non-admin would hit DAILY_CAP_EXCEEDED.
        result = await credit_service.try_charge(
            db, credit_user, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, Decimal("50"),
            idempotency_key="admin-daily-1",
        )
        await db.commit()
    assert result.success is True


async def test_admin_reserve_settle_is_balance_noop(credit_user, monkeypatch):
    """reserve + settle for an admin must not move the balance at all."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import BUCKET_MESSAGE, credit_service
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)
    await _make_admin(credit_user)

    before = Decimal((await _balance(credit_user)).message_credits_remaining)
    async with async_session_maker() as db:
        r = await credit_service.reserve(
            db, credit_user, "auto_builder", BUCKET_MESSAGE, Decimal("10"),
            idempotency_key="admin-res-1",
        )
        await db.commit()
    assert r.success
    held = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert held == before, "admin reserve must not hold/deduct credits"
    async with async_session_maker() as db:
        await credit_service.settle(db, r.reservation_id, Decimal("6"))
        await db.commit()
    final = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert final == before, "admin reserve+settle must be a balance no-op"
