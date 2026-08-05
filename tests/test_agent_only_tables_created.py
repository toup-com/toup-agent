"""The memory tables must land on tenants — and must NOT land on the platform.

`memory_capture_outbox` exists to hold facts a failed write would otherwise
lose. Both of its call sites are exception-guarded — deliberately, because a
failure to record a failure must not replace the original error — so a table
that never gets created raises nowhere. It degrades into exactly the silent data
loss it was built to prevent.

Tenant databases have no `alembic_version`, so `alembic upgrade head` restarts at
`001_initial` and dies; `init_db()` IS the tenant migrator. A *new* table needs
no migration and no `_alter_statements` mirror because `create_all` handles it —
but that is a claim about code someone can change.

**Which change, exactly, matters here — the first version of this file got that
wrong.** It asserted that dropping the table from `AGENT_ONLY_TABLES` would stop
it being created, and it passed anyway, because those sets are EXCLUSION lists:
under `RUN_MODE=agent` the excluded set is `PLATFORM_ONLY_TABLES`, so
`AGENT_ONLY_TABLES` membership has no bearing on whether a tenant gets the
table. The must-fail check is the only reason that was caught rather than
shipped as a guard that guards nothing.

The two changes that ARE silent:

  1. dropping the model's import from `app/db/models/__init__.py`, which takes
     it out of `Base.metadata` entirely so `create_all` never sees it;
  2. adding the name to `PLATFORM_ONLY_TABLES`, which excludes it on agents.

...and the mirror-image risk, which `AGENT_ONLY_TABLES` really does control:
the table leaking onto the PLATFORM database, where memory content does not
belong. `init_db` has a `_leaked` warning for that; this asserts it.

2026-08-03 added a third reason to pin all of it: #420 made `init_db()` plan its
DDL against the live catalog and skip already-satisfied statements. That
planning covers `_alter_statements` only and never touches `create_all` —
proven here rather than read.

The bootstrap tests need Postgres (the pytest-postgres CI job) — these tables
include pgvector columns that cannot compile on SQLite, which is also why the
sqlite job has never executed any of this. The registration invariants at the
bottom need no database at all and run on BOTH jobs, which is where a careless
import removal would most likely land.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

DB_URL = os.environ.get("DATABASE_URL", "")

# Applied per-test: the registration invariants below need no database and so
# run on the sqlite CI job too, which is where a careless import removal would
# most likely land.
needs_pg = pytest.mark.skipif(
    "postgres" not in DB_URL, reason="needs the Postgres service"
)

# Not the whole set — the ones whose absence is silent rather than loud.
MUST_EXIST_ON_TENANT = [
    "memories",
    "memory_capture_outbox",
    "brain_stats",
    "memory_events",
    "entities",
]

OUTBOX_COLUMNS = {
    "id", "user_id", "source_message_id", "payload_json",
    "created_at", "resolved_at", "attempts", "next_attempt_at", "last_error",
}


def _admin_engine():
    base = DB_URL.split("?")[0].rsplit("/", 1)[0]
    return create_async_engine(f"{base}/postgres", isolation_level="AUTOCOMMIT")


async def _bootstrap(run_mode: str):
    """Run the REAL tenant bootstrap against a scratch database in `run_mode`,
    and report what it actually created."""
    scratch = f"tblprobe_{run_mode}_{uuid.uuid4().hex[:8]}"

    admin = _admin_engine()
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{scratch}"'))
    finally:
        await admin.dispose()

    base = DB_URL.split("?")[0].rsplit("/", 1)[0]
    engine = create_async_engine(f"{base}/{scratch}")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

        import app.db.database as database
        from app.config import settings

        prev = (database.engine, database.async_session_maker, settings.run_mode)
        database.engine = engine
        database.async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
        settings.run_mode = run_mode
        try:
            await database.init_db()
        finally:
            (database.engine, database.async_session_maker, settings.run_mode) = prev

        async with engine.connect() as conn:
            tables = set((await conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public'"))).scalars().all())
            columns = set((await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='memory_capture_outbox'"))).scalars().all())
            indexes = set((await conn.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename='memory_capture_outbox'"))).scalars().all())
        return tables, columns, indexes
    finally:
        await engine.dispose()
        admin = _admin_engine()
        try:
            async with admin.connect() as conn:
                await conn.execute(text(f'DROP DATABASE IF EXISTS "{scratch}"'))
        finally:
            await admin.dispose()


@pytest.fixture(scope="module")
async def tenant():
    return await _bootstrap("agent")


@pytest.fixture(scope="module")
async def platform():
    return await _bootstrap("platform")


@needs_pg
async def test_the_memory_tables_are_created_on_a_tenant(tenant):
    tables, _, _ = tenant
    missing = [t for t in MUST_EXIST_ON_TENANT if t not in tables]
    assert not missing, (
        f"init_db() did not create {missing} on a tenant database. There is no "
        "alembic on tenants, so this IS the migration — a table missing here "
        "simply never exists in production, and the outbox fails silently."
    )


@needs_pg
async def test_the_capture_outbox_has_the_columns_the_service_writes(tenant):
    """A table with the right name and the wrong shape fails at the first
    INSERT — which, for this table, happens inside a failure handler."""
    _, columns, _ = tenant
    missing = OUTBOX_COLUMNS - columns
    assert not missing, f"memory_capture_outbox is missing {sorted(missing)}"


@needs_pg
async def test_the_capture_outbox_has_its_due_index(tenant):
    """`replay_pending` filters on (resolved_at, next_attempt_at) every turn."""
    _, _, indexes = tenant
    assert any("due" in i for i in indexes), (
        f"ix_memory_capture_outbox_due is absent; present: {sorted(indexes)}"
    )


class TestTheRegistrationThatDecidesAllOfThis:
    """The three facts `create_all` actually consults, asserted directly.

    Bootstrapping a virgin PLATFORM database to check the mirror-image case is
    not possible here: `init_db()` in platform mode fails on an unrelated table
    (`build_jobs`) against an empty database, because the real platform DB is
    long-established and has never been created from scratch. Rather than build
    an elaborate partial bootstrap to test a property that is decided by three
    set memberships, the memberships are asserted.

    Each of these was verified to FAIL when the corresponding registration is
    removed — which is more than could be said for the first version of this
    file.
    """

    TABLE = "memory_capture_outbox"

    def test_the_model_is_registered_in_the_metadata(self):
        """The silent one. `create_all` only creates what is in
        `Base.metadata`, so dropping the import from `app/db/models/__init__`
        stops the table being created and raises nowhere."""
        from app.db.models.base import Base
        import app.db.models  # noqa: F401 — the import under test

        assert self.TABLE in Base.metadata.tables, (
            f"{self.TABLE} is not in Base.metadata — create_all will never "
            "create it, on tenants or anywhere else"
        )

    def test_it_is_not_excluded_on_agents(self):
        """Under RUN_MODE=agent the excluded set is PLATFORM_ONLY_TABLES."""
        from app.db.models.base import PLATFORM_ONLY_TABLES

        assert self.TABLE not in PLATFORM_ONLY_TABLES, (
            f"{self.TABLE} is in PLATFORM_ONLY_TABLES, so tenants — the only "
            "databases that use it — would not get it"
        )

    def test_it_is_excluded_on_the_platform(self):
        """Under RUN_MODE=platform the excluded set is AGENT_ONLY_TABLES.

        This is the half AGENT_ONLY_TABLES really controls. Memory content
        lives in the per-tenant database; a memory table on the shared platform
        database is somewhere for it to end up where it does not belong."""
        from app.db.models.base import AGENT_ONLY_TABLES

        assert self.TABLE in AGENT_ONLY_TABLES, (
            f"{self.TABLE} is missing from AGENT_ONLY_TABLES — it would be "
            "created on the shared platform database as well as on tenants"
        )


# ── The case that actually happens on a tenant ───────────────────────────
#
# Everything above bootstraps a FRESH database. No live tenant is fresh: it has
# ~41 tables already and gains one. Those are different code paths through
# `create_all`, and only the second one ever runs in production.
#
# It is also the path where a mistake is silent. A new table that fails to
# appear on an existing database raises nothing — `create_all` simply emits no
# CREATE for it — and the outbox's callers are exception-guarded, so the first
# symptom would be facts quietly not being replayed.


@needs_pg
async def test_an_existing_tenant_schema_gains_the_new_table():
    """Build the schema as it was BEFORE this table existed, then run the real
    bootstrap and assert it arrives — adding exactly one table and dropping
    nothing.

    The old world is simulated by creating every table EXCEPT the outbox,
    rather than by mutating `Base.metadata`: this runs in the same process as
    every other test, and a global metadata mutation that leaked would be a
    far worse bug than the one being tested.
    """
    from app.db.models.base import Base, PLATFORM_ONLY_TABLES

    scratch = f"upgradepath_{uuid.uuid4().hex[:8]}"
    admin = _admin_engine()
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{scratch}"'))
    finally:
        await admin.dispose()

    base = DB_URL.split("?")[0].rsplit("/", 1)[0]
    engine = create_async_engine(f"{base}/{scratch}")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

        # ── the world as it was: every agent table except this one
        old_world = [
            t for t in Base.metadata.sorted_tables
            if t.name not in PLATFORM_ONLY_TABLES
            and t.name != "memory_capture_outbox"
        ]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, tables=old_world)

        async def tables():
            async with engine.connect() as conn:
                return set((await conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public'"))).scalars().all())

        before = await tables()
        assert "memory_capture_outbox" not in before, (
            "the simulation is meaningless if the table is already present"
        )
        assert "memories" in before, "the old world should still look like a tenant"

        # ── the upgrade: the real production bootstrap
        import app.db.database as database
        from app.config import settings

        prev = (database.engine, database.async_session_maker, settings.run_mode)
        database.engine = engine
        database.async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
        settings.run_mode = "agent"
        try:
            await database.init_db()
        finally:
            (database.engine, database.async_session_maker, settings.run_mode) = prev

        after = await tables()
        added = after - before

        # NOT "exactly one table appeared". `init_db` also creates a handful of
        # tables from raw DDL rather than from `Base.metadata` (e.g.
        # `redeemed_iap_transactions`, database.py:911), so they cannot be in a
        # metadata-built old world and legitimately show up here. Asserting
        # equality made this test fail on correct code the first time it ran.
        assert "memory_capture_outbox" in added, (
            f"the upgrade did not create the outbox; it added {sorted(added)}"
        )
        assert not (before - after), (
            f"the upgrade DROPPED pre-existing tables: {sorted(before - after)}"
        )

        async with engine.connect() as conn:
            cols = set((await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='memory_capture_outbox'"))).scalars().all())
            idx = set((await conn.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename='memory_capture_outbox'"))).scalars().all())
        assert not (OUTBOX_COLUMNS - cols), f"missing columns {sorted(OUTBOX_COLUMNS - cols)}"
        assert any("due" in i for i in idx), f"the due index is absent: {sorted(idx)}"
    finally:
        await engine.dispose()
        admin = _admin_engine()
        try:
            async with admin.connect() as conn:
                await conn.execute(text(f'DROP DATABASE IF EXISTS "{scratch}"'))
        finally:
            await admin.dispose()
