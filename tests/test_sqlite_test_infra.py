"""Test-infra regression guards for sqlite portability (P1).

Three things should hold on sqlite + in-memory DB so future contributors
don't have to re-discover the same paper cuts:

1. Two `User` rows with the default `is_canary=False` should both insert.
   Before P1, `Index(... postgresql_where=...)` got emitted as a TOTAL
   unique index on sqlite, breaking every test that needed two users.

2. The `Memory.search_vector` TSVECTOR column should compile under
   sqlite. The model now uses `with_variant(TSVECTOR(), "postgresql")`
   so the prod schema stays tsvector while sqlite gets Text.

3. `init_db()`'s `_alter_statements` should successfully add columns
   under sqlite. Before P1, `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
   raised a syntax error on every statement and the columns were
   silently missing. The rewriter inspects existing columns and emits
   the bare `ALTER TABLE ... ADD COLUMN` form.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import inspect as sa_inspect, select

from app.db.database import async_session_maker, engine
from app.db.models.user import User
from app.services.auth_service import get_password_hash


pytestmark = pytest.mark.asyncio


async def test_two_users_with_default_is_canary_can_coexist():
    async with async_session_maker() as db:
        u1 = User(
            id=str(uuid.uuid4()),
            email=f"a-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("x"),
            name="A",
        )
        u2 = User(
            id=str(uuid.uuid4()),
            email=f"b-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("y"),
            name="B",
        )
        db.add_all([u1, u2])
        await db.commit()

    async with async_session_maker() as db:
        rows = (await db.execute(select(User))).scalars().all()
        assert len(rows) >= 2
        canaries = [r for r in rows if r.is_canary]
        assert canaries == []  # default is False; no partial-index headache


async def test_memory_search_vector_column_compiles_under_sqlite():
    """search_vector is TSVECTOR on Postgres + Text on sqlite via with_variant.

    The `memories` table itself is not created in the conftest's sqlite DB
    when the pgvector extension isn't loaded (its `embedding` column uses
    VECTOR — see app/db/database.py:209 _vector_tables skip). The test
    here is narrower: load the Memory model class and verify the column
    definition compiles to a real type on sqlite, which would fail before
    P1 with `Compiler … can't render element of type TSVECTOR`.
    """
    from app.db.models.memory import Memory
    col = Memory.__table__.c.search_vector
    compiled = col.type.compile(dialect=engine.dialect)
    # On sqlite the variant resolves to TEXT; on Postgres it's TSVECTOR.
    assert compiled.upper() in {"TEXT", "TSVECTOR"}
    if engine.dialect.name == "sqlite":
        assert compiled.upper() == "TEXT", (
            f"Expected sqlite variant of TSVECTOR to compile to TEXT; got {compiled!r}"
        )


async def test_init_db_alter_statements_add_columns_on_sqlite():
    """`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` is rewritten on sqlite.

    Pick a column that exists ONLY because of an `_alter_statements` entry
    (not in the ORM model). `agent_configs.bundle_cancelled_at` is one
    such column that should land on every fresh init_db run.
    """
    if engine.dialect.name != "sqlite":
        pytest.skip("Test is sqlite-specific; on Postgres ALTER ... IF NOT EXISTS works natively")

    async with engine.connect() as conn:
        cols = await conn.run_sync(
            lambda sc: {c["name"] for c in sa_inspect(sc).get_columns("agent_configs")}
            if "agent_configs" in sa_inspect(sc).get_table_names()
            else None
        )
    if cols is None:
        pytest.skip("agent_configs not present in this run_mode partition")
    assert "bundle_cancelled_at" in cols, (
        "_alter_statements entry for bundle_cancelled_at didn't land — the "
        "sqlite rewrite of `ADD COLUMN IF NOT EXISTS` regressed"
    )


def test_default_database_url_is_shared_cache_sqlite():
    """The conftest defaults DATABASE_URL to the shared-cache form.

    If a contributor sets DATABASE_URL in their environment, the os
    .environ.setdefault is a no-op — that's fine. But on CI / fresh
    checkouts the default should be the shared-cache form so any new
    code that builds a second sqlite engine still sees the same DB.
    """
    import os
    url = os.environ["DATABASE_URL"]
    if url.startswith("sqlite"):
        assert "cache=shared" in url, (
            f"Expected shared-cache sqlite URL, got {url!r}. "
            f"Without cache=shared, a second create_async_engine in app code "
            f"would silently get its own in-memory DB."
        )
