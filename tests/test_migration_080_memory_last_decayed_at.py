"""Migration 080 (`memories.last_decayed_at`) must reach BOTH schemas.

`memories` lives in two places: the platform DB (monolith leftover, served as
the read-fallback) and every tenant container DB. Only the platform DB has an
`alembic_version` row — tenant DBs have none, so `alembic upgrade head`
restarts at 001 there and dies. The ALTER list in
`app/db/database.py::init_db` IS the tenant migrator, which means a migration
without a mirror entry NEVER applies to tenants, and the moment the ORM
references the column those tenants 500 on every memory query.

So this file pins both halves:
  * the mirror entry exists, in the exact `ADD COLUMN IF NOT EXISTS` form the
    rest of the list uses (source-level; runs everywhere);
  * the revision itself is additive, reversible, and a no-op when the column
    is already present because init_db got there first (needs a live
    Postgres; skipped when none is reachable, as in the sqlite CI job).

Run: python3 -m pytest tests/test_migration_080_memory_last_decayed_at.py -q
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

_BACKEND = Path(__file__).resolve().parents[1]
_MIGRATION = (
    _BACKEND / "alembic" / "versions" / "20260806_0080_080_memory_last_decayed_at.py"
)

MIRROR_STATEMENT = (
    "ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMP"
)

# The pre-080 shape of memories, trimmed to what the revision touches.
_BASE_DDL = """
CREATE TABLE memories (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) NOT NULL,
  content TEXT,
  strength FLOAT DEFAULT 1.0,
  last_reinforced_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT now()
);
"""


# ── The tenant migrator (no DB needed) ────────────────────────────────

def test_init_db_alter_list_mirrors_the_column():
    """Tenant DBs migrate via this list and nothing else.

    Asserted on the source text rather than by booting init_db because the
    structural backstop (`_reconcile_missing_columns`) would ALSO add the
    column — self-healing it as a bare nullable and logging a warning. That
    safety net makes an end-to-end column check pass whether or not the
    explicit mirror was written, so only the source can pin the mirror.
    """
    src = (_BACKEND / "app" / "db" / "database.py").read_text()
    assert MIRROR_STATEMENT in src, (
        "migration 080 has no mirror entry in init_db's _alter_statements — "
        "tenant databases have no alembic_version, so the column would never "
        "reach them and every memory query would 500 once the ORM maps it"
    )


def test_model_maps_the_column():
    """The ORM side of the same contract."""
    from app.db.models.memory import Memory

    assert "last_decayed_at" in Memory.__table__.columns
    assert Memory.__table__.columns["last_decayed_at"].nullable, (
        "NULL must mean 'never decayed' — a NOT NULL/defaulted column would "
        "rewrite the decay reference of every pre-existing row"
    )


# ── The revision itself (needs Postgres) ──────────────────────────────

def _pg_url() -> str | None:
    url = os.environ.get(
        "TEST_PG_URL",
        f"postgresql+psycopg2://{os.environ.get('USER', 'postgres')}@localhost:5432/postgres",
    )
    try:
        sa.create_engine(url).connect().close()
    except Exception:
        return None
    return url


requires_pg = pytest.mark.skipif(
    _pg_url() is None, reason="no local Postgres for migration tests"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("mig080", _MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def scratch_db():
    """A throwaway database, dropped afterwards."""
    admin = sa.create_engine(_pg_url(), isolation_level="AUTOCOMMIT")
    name = "toup_mig080_test"
    with admin.connect() as c:
        c.execute(sa.text(f"DROP DATABASE IF EXISTS {name}"))
        c.execute(sa.text(f"CREATE DATABASE {name}"))
    url = _pg_url().rsplit("/", 1)[0] + f"/{name}"
    engine = sa.create_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as c:
            c.execute(sa.text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = '{name}'"
            ))
            c.execute(sa.text(f"DROP DATABASE IF EXISTS {name}"))
        admin.dispose()


def _columns(conn) -> set[str]:
    return {c["name"] for c in sa.inspect(conn).get_columns("memories")}


def _run(conn, fn):
    """Execute a migration function against a live connection, with the
    global `alembic.op` proxy installed exactly as `alembic upgrade` does."""
    with Operations.context(MigrationContext.configure(conn)):
        fn()


def test_revision_number_follows_the_head():
    mig = _load_migration()
    assert mig.revision == "080"
    assert mig.down_revision == "079"


@requires_pg
def test_upgrade_adds_the_column_and_leaves_rows_alone(scratch_db):
    mig = _load_migration()
    with scratch_db.begin() as conn:
        conn.execute(sa.text(_BASE_DDL))
        conn.execute(sa.text(
            "INSERT INTO memories (id, user_id, content, strength) "
            "VALUES ('m-legacy', 'u1', 'legacy row', 0.73)"
        ))
    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)
    with scratch_db.connect() as conn:
        assert "last_decayed_at" in _columns(conn)
        row = conn.execute(sa.text(
            "SELECT strength, last_decayed_at FROM memories WHERE id = 'm-legacy'"
        )).one()
    assert row[0] == 0.73, "the revision modified an existing row"
    assert row[1] is None, (
        "NULL = never decayed; a default here would silently reset the decay "
        "reference of every row already in the table"
    )


@requires_pg
def test_downgrade_fully_reverses(scratch_db):
    mig = _load_migration()
    with scratch_db.begin() as conn:
        conn.execute(sa.text(_BASE_DDL))
    with scratch_db.connect() as conn:
        before = _columns(conn)

    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)
    with scratch_db.begin() as conn:
        _run(conn, mig.downgrade)

    with scratch_db.connect() as conn:
        assert _columns(conn) == before


@requires_pg
def test_upgrade_is_a_noop_when_init_db_got_there_first(scratch_db):
    """Tenant/agent DBs are healed by the init_db mirror, and the platform DB
    boots init_db too — so the revision routinely finds its own work already
    done and must not explode."""
    mig = _load_migration()
    with scratch_db.begin() as conn:
        conn.execute(sa.text(_BASE_DDL))
        conn.execute(sa.text(MIRROR_STATEMENT))
    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)
    with scratch_db.connect() as conn:
        assert "last_decayed_at" in _columns(conn)
