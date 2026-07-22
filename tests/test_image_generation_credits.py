"""Tests for ChatGPT image-generation billing (gpt-image-2 primary, gpt-image-1 legacy).

Covers:
* image_generation_cost_cents: (size, quality, model) pricing — the primary
  gpt-image-2 table by default, the legacy gpt-image-1 table when the fallback
  model served, the fallback for unknown combos, and the "hd"/"standard"/"auto"
  quality mappings.
* try_charge with LEDGER_IMAGE_GEN + BUCKET_MESSAGE: deducts the correct
  credit amount, writes a ledger row stamped event_type="image_generation",
  and is idempotent on the same idempotency_key (the double-charge guard the
  proxy + agent-charge paths both rely on).

Uses the shared conftest in-memory-SQLite fixtures, mirroring
test_credit_service.py.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
import pytest_asyncio


pytestmark = pytest.mark.asyncio


# ──────────────────────────────────────────────────────────────────────
# image_generation_cost_cents — pricing table + fallbacks (no DB)
# ──────────────────────────────────────────────────────────────────────


def test_image_cost_high_quality_square():
    # Default model is gpt-image-2 → its published high 1024² price.
    from app.services.credit_service import image_generation_cost_cents
    assert image_generation_cost_cents("1024x1024", "high") == Decimal("21.1")


def test_image_cost_tiers_scale_with_quality():
    from app.services.credit_service import image_generation_cost_cents
    low = image_generation_cost_cents("1024x1024", "low")
    med = image_generation_cost_cents("1024x1024", "medium")
    high = image_generation_cost_cents("1024x1024", "high")
    assert low < med < high
    assert (low, med, high) == (Decimal("0.6"), Decimal("5.3"), Decimal("21.1"))


def test_image_cost_landscape_medium():
    from app.services.credit_service import image_generation_cost_cents
    assert image_generation_cost_cents("1536x1024", "medium") == Decimal("4.1")


def test_image_cost_legacy_model_uses_legacy_table():
    """The gpt-image-1 fallback must bill at v1's (lower) rates, not v2's — a
    fallback image should never be charged at the newer model's higher price."""
    from app.services.credit_service import image_generation_cost_cents
    assert image_generation_cost_cents("1024x1024", "high", "gpt-image-1") == Decimal("16.7")
    assert image_generation_cost_cents("1024x1024", "medium", "gpt-image-1") == Decimal("4.2")
    # dall-e also routes to the legacy table.
    assert image_generation_cost_cents("1024x1024", "high", "dall-e-3") == Decimal("16.7")


def test_image_cost_primary_model_explicit():
    from app.services.credit_service import image_generation_cost_cents
    assert image_generation_cost_cents("1024x1024", "high", "gpt-image-2") == Decimal("21.1")


def test_image_cost_unknown_combo_falls_back():
    from app.config import settings
    from app.services.credit_service import image_generation_cost_cents
    # Primary (gpt-image-2) unknown combo -> primary fallback.
    assert image_generation_cost_cents("999x999", "ultra") == Decimal(
        str(settings.image_gen_fallback_cents)
    )
    # Legacy model unknown combo -> legacy fallback.
    assert image_generation_cost_cents("999x999", "ultra", "gpt-image-1") == Decimal(
        str(settings.image_gen_fallback_cents_legacy)
    )


def test_image_cost_default_is_high_quality_square():
    """None/None must resolve to the configured defaults (high, 1024x1024) at the primary price."""
    from app.services.credit_service import image_generation_cost_cents
    assert image_generation_cost_cents(None, None) == Decimal("21.1")


def test_image_cost_dalle_quality_aliases():
    """'hd'/'standard' + 'auto' map onto our low/medium/high scale (primary prices)."""
    from app.services.credit_service import image_generation_cost_cents
    assert image_generation_cost_cents("1024x1024", "hd") == Decimal("21.1")       # -> high
    assert image_generation_cost_cents("1024x1024", "standard") == Decimal("5.3")  # -> medium
    assert image_generation_cost_cents("1024x1024", "auto") == Decimal("21.1")     # -> high


# ──────────────────────────────────────────────────────────────────────
# try_charge for an image event — deduction, ledger stamp, idempotency
# ──────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def credit_user(test_user_id: str):
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


async def _latest_ledger(user_id: str):
    from sqlalchemy import select
    from app.db import async_session_maker, CreditLedger
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(CreditLedger).where(CreditLedger.user_id == user_id)
        )).scalars().all()
        return rows


async def test_image_charge_deducts_and_stamps_event_type(credit_user, monkeypatch):
    """A medium-quality image (gpt-image-2 → 5.3cr, under the free daily cap)
    deducts cleanly and stamps event_type='image_generation' on the ledger row."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE
    from app.services.credit_service import (
        credit_service, image_generation_cost_cents, underlying_cost_to_credits,
    )
    monkeypatch.setattr(settings, "credit_enforcement_enabled", False, raising=False)

    before = Decimal((await _balance(credit_user)).message_credits_remaining)
    cents = image_generation_cost_cents("1024x1024", "medium")  # gpt-image-2 default → 5.3
    credits = underlying_cost_to_credits(cents)                 # 5.3

    async with async_session_maker() as db:
        result = await credit_service.try_charge(
            db, credit_user, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, credits,
            idempotency_key="img-1", underlying_cost_cents=cents,
            model="gpt-image-2", provider="openai",
        )
        await db.commit()

    assert result.success is True
    after = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert after == before - credits
    rows = await _latest_ledger(credit_user)
    assert any(r.event_type == LEDGER_IMAGE_GEN for r in rows)
    img_row = next(r for r in rows if r.event_type == LEDGER_IMAGE_GEN)
    # Ledger amounts are signed deltas — a charge is negative.
    assert Decimal(img_row.amount) == -credits


async def test_image_charge_idempotent(credit_user, monkeypatch):
    """Same idempotency_key must not double-charge — the guard the proxy and
    /credits/agent-charge both rely on to avoid billing an image twice."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE
    from app.services.credit_service import credit_service
    monkeypatch.setattr(settings, "credit_enforcement_enabled", False, raising=False)

    before = Decimal((await _balance(credit_user)).message_credits_remaining)
    async with async_session_maker() as db:
        r1 = await credit_service.try_charge(
            db, credit_user, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, Decimal("4.2"),
            idempotency_key="img-dup",
        )
        await db.commit()
    async with async_session_maker() as db:
        r2 = await credit_service.try_charge(
            db, credit_user, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, Decimal("4.2"),
            idempotency_key="img-dup",
        )
        await db.commit()

    assert r1.success and r2.success
    assert r2.idempotent_hit is True
    after = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert after == before - Decimal("4.2")  # charged once, not twice


async def test_high_quality_image_exceeds_free_daily_cap_when_enforced(credit_user, monkeypatch):
    """DOCUMENTS a real interaction: a HIGH-quality image (gpt-image-2 → 21.1cr)
    exceeds the free-tier 15-credit daily message cap, so with enforcement ON a
    fresh free user is denied with daily_cap_exceeded and nothing is deducted.

    This is why image pricing/quality is config-tunable (settings.image_gen_*)
    — operators can lower the default quality, raise the free daily cap, or
    rely on purchased (IAP) credits, which bypass the cap.
    """
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import LEDGER_IMAGE_GEN, BUCKET_MESSAGE
    from app.services.credit_service import (
        credit_service, REASON_DAILY_CAP_EXCEEDED,
    )
    monkeypatch.setattr(settings, "credit_enforcement_enabled", True, raising=False)

    before = Decimal((await _balance(credit_user)).message_credits_remaining)
    async with async_session_maker() as db:
        result = await credit_service.try_charge(
            db, credit_user, LEDGER_IMAGE_GEN, BUCKET_MESSAGE, Decimal("21.1"),
            idempotency_key="img-hq-cap",
        )
        await db.commit()

    assert result.success is False
    assert result.reason == REASON_DAILY_CAP_EXCEEDED
    after = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert after == before  # denied → nothing deducted
