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


async def test_a_second_engine_sees_the_same_database():
    """The property the shared-cache URL exists for, asserted on BEHAVIOUR.

    The test above checks what conftest DEFAULTS to. This one checks what the
    suite is actually running against — which is not the same question, and the
    gap between them was live until 2026-08-04: the workflow set
    `DATABASE_URL=sqlite+aiosqlite:///:memory:` at job level, so conftest's
    `setdefault` never fired in the one environment that matters, and the
    defence existed only on developer laptops.

    With the plain `:memory:` form every engine gets its OWN private database.
    A second `create_async_engine` anywhere in an app path then sees an empty
    schema and fails with "no such table" — an error that points at the model
    rather than at the URL, which is what makes it expensive to diagnose.

    MUTATION: set the workflow's DATABASE_URL back to
    `sqlite+aiosqlite:///:memory:` and this goes red with exactly that
    "no such table" error. Measured both ways before this test was written.
    """
    if engine.dialect.name != "sqlite":
        pytest.skip("sqlite-specific: on Postgres every engine reaches the same server")

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    url = engine.url.render_as_string(hide_password=False)
    second = create_async_engine(url, poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE _two_engine_probe (id INTEGER PRIMARY KEY)"))
        try:
            async with second.begin() as conn:
                found = (await conn.execute(
                    text("SELECT COUNT(*) FROM _two_engine_probe")
                )).scalar_one()
        except Exception as exc:  # pragma: no cover — this IS the regression
            raise AssertionError(
                "A second engine on the SAME DATABASE_URL cannot see a table "
                f"the first one just created ({type(exc).__name__}: {exc}). "
                f"The URL in use is {url!r}. It needs the "
                "`file::memory:?cache=shared&uri=true` form — with plain "
                "`:memory:` each engine gets its own private database."
            ) from exc
        assert found == 0
    finally:
        await second.dispose()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS _two_engine_probe"))


def test_ci_and_conftest_agree_on_the_database_url():
    """CI's explicit value must not drift from conftest's default.

    Two places name this URL: conftest (with the reasoning) and the workflow
    (which overrides it). They disagreed for as long as the sweep has existed.
    Config values ARE the behaviour here, so comparing them is a real check —
    unlike grepping Python source for an implementation detail.

    A bare value is still required in the workflow rather than deleted: with no
    DATABASE_URL set at all, `app/config.py` falls back to a FILE-backed
    `sqlite+aiosqlite:///./toup.db`, so any step that imports app.config outside
    pytest would quietly start writing a real database file.
    """
    import re
    from pathlib import Path

    wf = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "test-backend.yml"
    values = re.findall(r'^\s*DATABASE_URL:\s*"?([^"\n]+)"?\s*$', wf.read_text(), re.M)
    sqlite_values = [v.strip().rstrip('"') for v in values if v.startswith("sqlite")]
    assert sqlite_values, "no sqlite DATABASE_URL found in the workflow"
    for v in sqlite_values:
        assert "cache=shared" in v, (
            f"the workflow sets DATABASE_URL={v!r}, which gives every engine "
            "its own private in-memory database and silently disables the "
            "defence conftest documents. Use the "
            "`sqlite+aiosqlite:///file::memory:?cache=shared&uri=true` form."
        )


def test_file_sqlite_pools_per_session_memory_keeps_the_shared_connection(tmp_path):
    """R28-D: StaticPool on a FILE-backed sqlite shares ONE connection
    process-wide, so a background loop's rollback lands on the same
    connection as a request handler mid-transaction and eats its
    flushed-but-uncommitted writes — a live agent lost its own `arm`
    that way. File DBs must pool per-session (NullPool). Memory DBs
    must KEEP the shared connection: every new connection to plain
    :memory: is a fresh empty database, and the CI shared-cache form
    dies with its last open connection."""
    from sqlalchemy.pool import NullPool, StaticPool

    from app.db.database import _build_engine_inner

    eng = _build_engine_inner(f"sqlite+aiosqlite:///{tmp_path}/x.db")
    try:
        assert isinstance(eng.sync_engine.pool, NullPool)
    finally:
        eng.sync_engine.dispose()

    for url in (
        "sqlite+aiosqlite:///:memory:",
        "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
    ):
        eng = _build_engine_inner(url)
        try:
            assert isinstance(eng.sync_engine.pool, StaticPool), url
        finally:
            eng.sync_engine.dispose()
