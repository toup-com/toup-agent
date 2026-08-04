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


def test_conftest_defaults_database_url_to_the_shared_cache_form():
    """conftest must DEFAULT to shared-cache sqlite when nothing is set.

    This used to read `os.environ["DATABASE_URL"]` in-process and assert on
    it, which measured the AMBIENT ENVIRONMENT rather than our code. CI sets
    `DATABASE_URL=sqlite+aiosqlite:///:memory:` explicitly, so the conftest's
    `setdefault` is a no-op there and the assertion failed — while the thing
    it means to protect (the default) was perfectly fine.

    The fix is the same shape as the hermetic-settings lesson from #411:
    exercise the defaulting logic with a CLEAN environment, in a subprocess,
    so an ambient value can neither satisfy nor break it.

    Why shared-cache matters: with plain `sqlite:///:memory:`, a second
    `create_async_engine` anywhere in the app path gets its OWN empty
    in-memory DB, and the resulting schema-drift bug is painful to find.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    probe = (
        "import os, importlib.util as u, pathlib;"
        f"s=u.spec_from_file_location('cft', r'{Path(__file__).resolve().parent / 'conftest.py'}');"
        "m=u.module_from_spec(s);s.loader.exec_module(m);"
        "print(os.environ['DATABASE_URL'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, f"probe failed: {out.stderr[-800:]}"
    url = out.stdout.strip().splitlines()[-1]

    assert url.startswith("sqlite"), f"default should be sqlite, got {url!r}"
    assert "cache=shared" in url, (
        f"conftest defaulted DATABASE_URL to {url!r}, which is NOT the "
        "shared-cache form. Without cache=shared, a second "
        "create_async_engine in app code silently gets its own in-memory DB."
    )
