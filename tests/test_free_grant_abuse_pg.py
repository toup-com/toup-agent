"""Postgres-only verification for the free-grant abuse controls.

The default suite runs on sqlite, which cannot host concurrent writers and
does not run alembic migrations. These tests close that gap on a REAL
Postgres and are skipped unless ``TOUP_TEST_PG_URL`` points at a
throwaway database, e.g.:

    createdb toup_abuse_verify
    TOUP_TEST_PG_URL=postgresql+asyncpg://localhost/toup_abuse_verify \
        pytest tests/test_free_grant_abuse_pg.py -q

NEVER point this at production.
"""
from __future__ import annotations

import asyncio
import importlib.util as _il
import os
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

PG_URL = os.environ.get("TOUP_TEST_PG_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not PG_URL, reason="set TOUP_TEST_PG_URL to a throwaway Postgres to run"),
]


def _load(name, path):
    s = _il.spec_from_file_location(name, path)
    m = _il.module_from_spec(s)
    s.loader.exec_module(m)
    return m


@pytest_asyncio.fixture
async def pg_engine():
    eng = create_async_engine(PG_URL)
    yield eng
    await eng.dispose()


async def _fresh_tables(eng, models):
    tbls = [m.__table__ for m in models]
    async with eng.begin() as c:
        await c.run_sync(lambda s: __import__("app.db.models", fromlist=["Base"]).Base.metadata.drop_all(s, tables=tbls))
        await c.run_sync(lambda s: __import__("app.db.models", fromlist=["Base"]).Base.metadata.create_all(s, tables=tbls))


async def test_pg_migrations_apply_idempotent_and_backfill(pg_engine):
    """063/064/065 apply on Postgres: idempotent, down/up clean, backfill
    seeds the earliest account per canonical identity, unique apple_sub
    enforced."""
    from app.db.models import Base, User, SubscriptionPlan, CreditBalance
    from datetime import datetime
    from alembic.runtime.migration import MigrationContext
    from alembic.operations import Operations

    here = os.path.dirname(os.path.dirname(__file__))
    m063 = _load("m063_pg", f"{here}/alembic/versions/20260629_0067_067_grant_eligibility.py")
    m064 = _load("m064_pg", f"{here}/alembic/versions/20260629_0068_068_user_apple_sub.py")
    m065 = _load("m065_pg", f"{here}/alembic/versions/20260629_0069_069_signup_attempts.py")

    def run(sync_conn, module, fn):
        module.op = Operations(MigrationContext.configure(sync_conn))
        getattr(module, fn)()

    # clean slate + realistic prereq schema
    async with pg_engine.begin() as c:
        await c.execute(text("DROP TABLE IF EXISTS grant_eligibility, signup_attempts CASCADE"))
    await _fresh_tables(pg_engine, [User, SubscriptionPlan, CreditBalance])
    async with pg_engine.begin() as c:
        await c.execute(text("INSERT INTO subscription_plans (id,display_name,price_cents,message_credits_monthly,integration_credits_monthly,message_credits_daily_cap,rollover_message_credits,rollover_integration_credits,rollover_max_pct,active,sort_order,created_at) VALUES ('free','Free',0,100,500,15,false,false,0,true,0,now())"))
        seed = [("u1", "user@gmail.com", datetime(2026, 1, 1)),
                ("u2", "u.s.e.r+x@gmail.com", datetime(2026, 2, 1)),
                ("u3", "distinct@proton.me", datetime(2026, 1, 15)),
                ("u4", "nobalance@gmail.com", datetime(2026, 1, 20))]
        for uid, em, ca in seed:
            await c.execute(text("INSERT INTO users (id,email,hashed_password,role,created_at,updated_at,is_active,is_canary) VALUES (:i,:e,'x','beta_user',:c,:c,true,false)"), {"i": uid, "e": em, "c": ca})
        for uid in ("u1", "u2", "u3"):
            await c.execute(text("INSERT INTO credit_balances (user_id,plan_id,message_credits_remaining,integration_credits_remaining,message_credits_used_today,purchased_credits_remaining,day_anchor_local_date,period_start,period_end,updated_at) VALUES (:u,'free',100,500,0,0,'2026-01-01',now(),now(),now())"), {"u": uid})

    async with pg_engine.begin() as c: await c.run_sync(run, m063, "upgrade")
    async with pg_engine.begin() as c: await c.run_sync(run, m063, "upgrade")  # idempotent
    async with pg_engine.connect() as c:
        owners = sorted(r[0] for r in (await c.execute(text("SELECT first_user_id FROM grant_eligibility"))).fetchall())
    assert owners == ["u1", "u3"], owners  # u1 wins gmail collision, u3 distinct, u4 (no balance) excluded

    async with pg_engine.begin() as c: await c.run_sync(run, m063, "downgrade")
    async with pg_engine.connect() as c:
        assert (await c.execute(text("SELECT to_regclass('public.grant_eligibility')"))).scalar() is None
    async with pg_engine.begin() as c: await c.run_sync(run, m063, "upgrade")

    async with pg_engine.begin() as c: await c.run_sync(run, m064, "upgrade")
    async with pg_engine.begin() as c: await c.run_sync(run, m064, "upgrade")
    async with pg_engine.begin() as c: await c.execute(text("UPDATE users SET apple_sub='S1' WHERE id='u1'"))
    with pytest.raises(Exception):
        async with pg_engine.begin() as c:
            await c.execute(text("UPDATE users SET apple_sub='S1' WHERE id='u3'"))  # unique violation

    async with pg_engine.begin() as c: await c.run_sync(run, m065, "upgrade")
    async with pg_engine.begin() as c: await c.run_sync(run, m065, "upgrade")
    async with pg_engine.connect() as c:
        assert (await c.execute(text("SELECT to_regclass('public.signup_attempts')"))).scalar() is not None


async def test_pg_concurrent_signups_one_grant(pg_engine, monkeypatch):
    """8 concurrent same-canonical signups on Postgres → exactly ONE grant;
    losers suppressed, none rolled back. The savepoint must catch the real
    UniqueViolation. This is the guarantee sqlite cannot exercise."""
    from app.db.models import Base, User, SubscriptionPlan, CreditBalance, CreditLedger, GrantEligibility
    from app.config import settings
    from app.services.credit_service import CreditService
    monkeypatch.setattr(settings, "free_grant_dedupe_enabled", True, raising=False)

    async with pg_engine.begin() as c:
        await c.execute(text("DROP TABLE IF EXISTS grant_eligibility, credit_ledger, credit_balances CASCADE"))
    await _fresh_tables(pg_engine, [User, SubscriptionPlan, CreditBalance, CreditLedger, GrantEligibility])
    Session = async_sessionmaker(pg_engine, expire_on_commit=False)
    N = 8
    async with Session() as db:
        db.add(SubscriptionPlan(id="free", display_name="Free", price_cents=0, message_credits_monthly=Decimal("100"), integration_credits_monthly=Decimal("500"), message_credits_daily_cap=Decimal("15")))
        variants = ["racer@gmail.com", "r.acer@gmail.com", "ra.cer@gmail.com", "rac.er@gmail.com",
                    "race.r@gmail.com", "racer+1@gmail.com", "racer+two@gmail.com", "r.a.c.e.r@gmail.com"]
        for i, em in enumerate(variants):
            db.add(User(id=f"u{i}", email=em, hashed_password="x"))
        await db.commit()

    svc = CreditService()
    async def signup(uid):
        async with Session() as db:
            await svc.get_or_create_balance(db, uid)
            await db.commit()
    results = await asyncio.gather(*[signup(f"u{i}") for i in range(N)], return_exceptions=True)
    assert not [r for r in results if isinstance(r, Exception)], results

    async with Session() as db:
        bals = [Decimal(b.message_credits_remaining) for b in (await db.execute(select(CreditBalance))).scalars().all()]
        tombs = (await db.execute(select(func.count()).select_from(GrantEligibility))).scalar()
        grants = (await db.execute(select(func.count()).select_from(CreditLedger).where(CreditLedger.event_type == "plan_grant"))).scalar()
    assert len(bals) == N                      # every racer kept its balance row
    assert bals.count(Decimal("100")) == 1     # exactly one grant
    assert bals.count(Decimal("0")) == N - 1   # the rest suppressed
    assert tombs == 1
    assert grants == 2                          # the winner's msg + integration rows
