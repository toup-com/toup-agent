"""The boot-DDL planner, proven against a real Postgres catalog.

tests/test_ddl_plan.py pins the parsing rules. This file proves the property
that actually matters on a live tenant: **executing everything the planner
skipped changes nothing**. If that holds, skipping loses nothing.

It also pins the reason this is catalog-derived rather than a stored schema
version: drop a column out of band and the very next plan puts the statement
back. A version marker cannot do that, and its failure mode is a silently
missing column weeks later.

Skipped unless DATABASE_URL points at Postgres (the pytest-postgres CI job).
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.ddl_plan import SNAPSHOT_SQL, is_satisfied, plan, snapshot_from_rows

DB_URL = os.environ.get("DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    "postgres" not in DB_URL, reason="needs the Postgres service"
)

TBL = "ddl_plan_probe"


async def _snapshot(engine):
    async with engine.connect() as conn:
        cols = (await conn.execute(text(SNAPSHOT_SQL["columns"]))).all()
        idxs = (await conn.execute(text(SNAPSHOT_SQL["indexes"]))).all()
        tbls = (await conn.execute(text(SNAPSHOT_SQL["tables"]))).all()
    return snapshot_from_rows(
        [(r[0], r[1]) for r in cols], [r[0] for r in idxs], [r[0] for r in tbls]
    )


async def _columns_of(engine, table):
    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = :t "
            "ORDER BY column_name"
        ), {"t": table})).all()
    return [(r[0], r[1]) for r in rows]


@pytest.mark.asyncio
async def test_skipped_statements_are_provably_no_ops():
    """The equivalence proof: run every statement the planner planned AWAY
    and show the schema is byte-identical afterwards."""
    engine = create_async_engine(DB_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
            await conn.execute(text(f"CREATE TABLE {TBL} (id serial primary key, a text)"))
            await conn.execute(text(f"CREATE INDEX ix_{TBL}_a ON {TBL} (a)"))

        statements = [
            f"ALTER TABLE {TBL} ADD COLUMN IF NOT EXISTS a text",
            f"ALTER TABLE {TBL} ADD COLUMN IF NOT EXISTS id integer",
            f"CREATE INDEX IF NOT EXISTS ix_{TBL}_a ON {TBL} (a)",
            f"ALTER TABLE {TBL} ADD COLUMN IF NOT EXISTS brand_new text",  # genuinely needed
        ]

        snap = await _snapshot(engine)
        to_run, skipped = plan(statements, snap)

        assert len(skipped) == 3, f"expected the 3 satisfied statements, got {skipped}"
        assert to_run == [statements[3]], f"the needed statement must survive: {to_run}"

        before = await _columns_of(engine, TBL)
        for stmt in skipped:
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        after = await _columns_of(engine, TBL)

        assert before == after, (
            "a statement the planner skipped DID change the schema — skipping "
            f"it would have lost something: {before} -> {after}"
        )
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_needed_statement_really_does_change_the_schema():
    """The other half: what the planner keeps is not busywork."""
    engine = create_async_engine(DB_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
            await conn.execute(text(f"CREATE TABLE {TBL} (id serial primary key)"))

        stmt = f"ALTER TABLE {TBL} ADD COLUMN IF NOT EXISTS brand_new text"
        assert not is_satisfied(stmt, await _snapshot(engine))

        before = await _columns_of(engine, TBL)
        async with engine.begin() as conn:
            await conn.execute(text(stmt))
        after = await _columns_of(engine, TBL)

        assert after != before and ("brand_new", "text") in after
        # And now it plans away.
        assert is_satisfied(stmt, await _snapshot(engine))
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
        await engine.dispose()


@pytest.mark.asyncio
async def test_out_of_band_drift_is_replanned_not_remembered():
    """Why this is catalog-derived and not a stored schema version.

    Drop the column behind the planner's back; the next plan must put the
    statement straight back. A version marker would say "already applied"
    forever and the column would stay missing.
    """
    engine = create_async_engine(DB_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
            await conn.execute(text(f"CREATE TABLE {TBL} (id serial primary key, drifty text)"))

        stmt = f"ALTER TABLE {TBL} ADD COLUMN IF NOT EXISTS drifty text"
        assert is_satisfied(stmt, await _snapshot(engine))

        async with engine.begin() as conn:
            await conn.execute(text(f"ALTER TABLE {TBL} DROP COLUMN drifty"))

        assert not is_satisfied(stmt, await _snapshot(engine)), (
            "the planner kept skipping a column that no longer exists — that "
            "is the version-marker failure mode this design exists to avoid"
        )
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_noop_alter_still_waits_for_an_exclusive_lock():
    """The premise of the whole change, pinned against a real server.

    `ADD COLUMN IF NOT EXISTS` acquires ACCESS EXCLUSIVE *before* it
    evaluates the condition, so an idempotent no-op still queues behind an
    ordinary reader. If a future Postgres ever stops doing this, the
    justification for planning weakens and this test should say so.
    """
    engine = create_async_engine(DB_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
            await conn.execute(text(f"CREATE TABLE {TBL} (id serial primary key, a text)"))

        reader = await engine.connect()
        await reader.execute(text(f"SELECT count(*) FROM {TBL}"))  # holds ACCESS SHARE

        try:
            async with engine.begin() as conn:
                await conn.execute(text("SET lock_timeout = '2s'"))
                with pytest.raises(Exception) as err:
                    await conn.execute(
                        text(f"ALTER TABLE {TBL} ADD COLUMN IF NOT EXISTS a text")
                    )
            assert "lock" in str(err.value).lower(), (
                f"expected a lock timeout, got: {str(err.value)[:200]}"
            )
        finally:
            await reader.close()

        # Control: with no reader open, the identical statement is instant.
        async with engine.begin() as conn:
            await conn.execute(text("SET lock_timeout = '2s'"))
            await conn.execute(text(f"ALTER TABLE {TBL} ADD COLUMN IF NOT EXISTS a text"))
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {TBL}"))
        await engine.dispose()
