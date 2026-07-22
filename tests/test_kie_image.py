"""Kie (Nano Banana) image engine — aspect mapping + free-tier monthly cap.

The pure kie_client helpers need no DB. The free_tier_image_quota tests use the
shared conftest fixtures (real User row + in-memory sqlite), mirroring
test_image_generation_credits.py.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        PYTHONPATH=$(pwd) pytest tests/test_kie_image.py -q
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

pytestmark = pytest.mark.asyncio


# ── kie_client pure helpers (no DB) ──────────────────────────────────────

def test_aspect_from_size_maps_openai_sizes():
    from app.services.kie_client import _aspect_from_size
    assert _aspect_from_size("1024x1024") == "1:1"
    assert _aspect_from_size("1024x1536") == "2:3"     # portrait
    assert _aspect_from_size("1536x1024") == "3:2"     # landscape
    assert _aspect_from_size("something-else") == "1:1"
    assert _aspect_from_size(None) == "1:1"


def test_nearest_aspect_preserves_source_framing():
    from app.services.kie_client import _nearest_aspect
    assert _nearest_aspect(1000, 1000) == "1:1"
    assert _nearest_aspect(1920, 1080) == "16:9"
    assert _nearest_aspect(1080, 1920) == "9:16"
    assert _nearest_aspect(1200, 800) == "3:2"
    assert _nearest_aspect(0, 0) == "1:1"              # guard against div-by-zero


# ── free-tier monthly image cap ──────────────────────────────────────────

async def _add_image_rows(uid: str, n: int, when: datetime | None = None) -> None:
    from app.db import async_session_maker
    from app.db.models import CreditLedger, LEDGER_IMAGE_GEN
    async with async_session_maker() as db:
        for _ in range(n):
            db.add(CreditLedger(
                user_id=uid, event_type=LEDGER_IMAGE_GEN, bucket="message",
                amount=Decimal("-9"), balance_after=Decimal("0"),
                created_at=when or datetime.utcnow(),
            ))
        await db.commit()


async def _quota(uid: str):
    from app.db import async_session_maker
    from app.services.credit_service import free_tier_image_quota
    async with async_session_maker() as db:
        return await free_tier_image_quota(db, uid)


async def test_free_user_under_limit_allowed(test_user_id, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "free_tier_monthly_image_limit", 10, raising=False)
    await _add_image_rows(test_user_id, 3)
    assert await _quota(test_user_id) == (False, 3, 10)


async def test_free_user_at_limit_blocked(test_user_id, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "free_tier_monthly_image_limit", 10, raising=False)
    await _add_image_rows(test_user_id, 10)
    exceeded, used, limit = await _quota(test_user_id)
    assert exceeded is True and used == 10 and limit == 10


async def test_admin_is_unlimited(test_user_id, monkeypatch):
    from app.config import settings
    from app.db import async_session_maker, User
    monkeypatch.setattr(settings, "free_tier_monthly_image_limit", 10, raising=False)
    async with async_session_maker() as db:
        u = await db.get(User, test_user_id)
        u.role = "admin"
        await db.commit()
    await _add_image_rows(test_user_id, 50)
    exceeded, _used, _limit = await _quota(test_user_id)
    assert exceeded is False


async def test_paid_plan_is_unlimited(test_user_id, monkeypatch):
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    monkeypatch.setattr(settings, "free_tier_monthly_image_limit", 10, raising=False)
    async with async_session_maker() as db:
        bal = await credit_service.get_or_create_balance(db, test_user_id)
        bal.plan_id = "builder"
        await db.commit()
    await _add_image_rows(test_user_id, 50)
    exceeded, _used, _limit = await _quota(test_user_id)
    assert exceeded is False


async def test_limit_zero_disables_cap(test_user_id, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "free_tier_monthly_image_limit", 0, raising=False)
    await _add_image_rows(test_user_id, 50)
    assert await _quota(test_user_id) == (False, 0, 0)


async def test_previous_month_rows_not_counted(test_user_id, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "free_tier_monthly_image_limit", 10, raising=False)
    # 20 images dated to before this calendar month must NOT count.
    last_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)
    await _add_image_rows(test_user_id, 20, when=last_month)
    await _add_image_rows(test_user_id, 2)   # this month
    exceeded, used, _limit = await _quota(test_user_id)
    assert exceeded is False and used == 2
