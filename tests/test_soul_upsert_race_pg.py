"""PUT /api/soul must be idempotent under concurrent first-time saves.

2026-09-06: a new user's onboarding fired two concurrent `PUT /api/soul` (the
Personality "Continue" save was still in flight when the "Wake up" tap fired a
retry, because the first save took 7857 ms). Both missed the SELECT, both
INSERTed, and the loser died at the far-away `await db.commit()` with
`UniqueViolationError: ix_soul_configs_user_id` -> HTTP 500.

Two layers here:

  * A SOURCE guard that always runs — it is what stops the read-modify-write
    coming back. The default suite is sqlite, which cannot host concurrent
    writers, so it cannot catch this any other way.
  * A REAL-Postgres reproduction, skipped unless ``TOUP_TEST_PG_URL`` points at
    a throwaway database:

        createdb toup_soul_race
        TOUP_TEST_PG_URL=postgresql+asyncpg://localhost/toup_soul_race \
            pytest tests/test_soul_upsert_race_pg.py -q

    NEVER point this at production.

The reproduction carries its own falsifier: `test_old_shape_still_races` asserts
the PRE-FIX shape still raises. Without it, the pass below would prove only that
the test cannot see the race.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import String, DateTime, Text, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker


def _soul_src() -> str:
    return (Path(__file__).resolve().parents[1] / "app" / "api" / "soul.py").read_text()


# ── source guards (always run) ────────────────────────────────────────────

def test_save_soul_uses_a_real_upsert():
    src = _soul_src()
    assert "on_conflict_do_update" in src, (
        "PUT /api/soul must upsert atomically. A SELECT-then-INSERT races with "
        "itself during onboarding and 500s on ix_soul_configs_user_id."
    )
    assert "from sqlalchemy.dialects.postgresql import insert as pg_insert" in src


def test_save_soul_does_not_read_modify_write_soulconfig():
    """The exact pre-fix shape must not come back."""
    src = _soul_src()
    assert "db.add(config)" not in src, (
        "db.add(config) on the save path is the read-modify-write that raced. "
        "Use the ON CONFLICT upsert instead."
    )


def test_upsert_does_not_rotate_identity_columns():
    """`id` and `created_at` must stay out of the update set, or a second save
    would rotate the primary key and re-stamp the row's birth."""
    src = _soul_src()
    assert "_mutable = {" in src and "await db.execute(_stmt)" in src, (
        "the upsert block is gone — see test_save_soul_uses_a_real_upsert"
    )
    block = src[src.index("_mutable = {"): src.index("await db.execute(_stmt)")]
    for forbidden in ('"id"', '"created_at"', '"vps_soul_synced_at"'):
        assert forbidden not in block, f"{forbidden} must not be in the upsert's update set"


# ── real-Postgres reproduction (opt-in) ───────────────────────────────────

PG_URL = os.environ.get("TOUP_TEST_PG_URL")
pg_only = pytest.mark.skipif(
    not PG_URL, reason="set TOUP_TEST_PG_URL to a throwaway Postgres to run"
)


class _Base(DeclarativeBase):
    pass


class _SoulConfig(_Base):
    """Mirrors app/db/models/soul_config.py: user_id is unique + indexed."""
    __tablename__ = "soul_configs_racetest"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(50))
    compiled_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@pytest_asyncio.fixture
async def sessionmaker_():
    eng = create_async_engine(PG_URL)
    async with eng.begin() as c:
        await c.run_sync(_Base.metadata.drop_all)
        await c.run_sync(_Base.metadata.create_all)
    yield async_sessionmaker(eng, expire_on_commit=False)
    await eng.dispose()


async def _race(Session, body):
    """Two savers that have BOTH missed the SELECT before either writes."""
    uid = str(uuid.uuid4())
    gate = asyncio.Event()

    async def one(label):
        async with Session() as db:
            await db.execute(select(_SoulConfig).where(_SoulConfig.user_id == uid))
            await gate.wait()          # both have now missed
            await body(db, uid, label)
            await asyncio.sleep(0.05)  # stands in for the 12s VPS sync
            await db.commit()

    tasks = [asyncio.create_task(one(f"w{i}")) for i in (1, 2)]
    await asyncio.sleep(0.15)
    gate.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    async with Session() as db:
        rows = (
            await db.execute(select(_SoulConfig).where(_SoulConfig.user_id == uid))
        ).scalars().all()
    return [r for r in results if isinstance(r, BaseException)], rows


async def _upsert(db, uid, label):
    """The shape shipped in soul.py."""
    now = datetime.utcnow()
    mutable = {"name": label, "compiled_text": "x", "updated_at": now}
    stmt = pg_insert(_SoulConfig.__table__).values(
        id=str(uuid.uuid4()), user_id=uid, created_at=now, **mutable
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[_SoulConfig.__table__.c.user_id],
        set_={k: stmt.excluded[k] for k in mutable},
    )
    await db.execute(stmt)


async def _read_modify_write(db, uid, label):
    """The PRE-FIX shape, kept only as this test's falsifier."""
    cfg = (
        await db.execute(select(_SoulConfig).where(_SoulConfig.user_id == uid))
    ).scalar_one_or_none()
    if cfg:
        cfg.name = label
    else:
        db.add(_SoulConfig(id=str(uuid.uuid4()), user_id=uid, name=label, compiled_text="x"))


@pg_only
@pytest.mark.asyncio
async def test_old_shape_still_races(sessionmaker_):
    """FALSIFIER. If this ever passes, the harness has stopped seeing the race
    and the assertion below proves nothing."""
    errs, rows = await _race(sessionmaker_, _read_modify_write)
    assert errs, "expected the pre-fix read-modify-write to raise"
    assert isinstance(errs[0], IntegrityError)
    assert len(rows) == 1


@pg_only
@pytest.mark.asyncio
@pytest.mark.parametrize("run", range(5))
async def test_upsert_survives_concurrent_first_saves(sessionmaker_, run):
    errs, rows = await _race(sessionmaker_, _upsert)
    assert not errs, f"concurrent saves must both succeed, got {errs!r}"
    assert len(rows) == 1, f"exactly one soul row per user, got {len(rows)}"
