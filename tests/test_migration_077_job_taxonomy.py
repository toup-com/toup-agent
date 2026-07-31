"""Migration 077 must be additive, reversible, and safe to run anywhere.

`build_jobs` is AGENT_ONLY, so this revision runs against platform DBs
that may or may not carry the table from the pre-split era. Both paths are
exercised here, plus a full up→down→up cycle, because an irreversible or
crashing migration is the one class of change that cannot be hot-fixed.

Postgres-only by necessity: the revision uses a partial index
(`WHERE archived_at IS NULL`) that SQLite cannot express. Skipped when no
local Postgres is reachable, so CI without a PG service still passes.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "alembic" / "versions" / "20260729_0077_077_job_error_taxonomy.py"
)

NEW_COLUMNS = {
    "error_class", "user_message", "technical_detail",
    "archived_at", "progress_step", "progress_total",
}

# Pre-migration shape of build_jobs, trimmed to what the revision touches.
_BASE_DDL = """
CREATE TABLE build_jobs (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) NOT NULL,
  title VARCHAR(200) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'queued',
  error_message TEXT,
  created_at TIMESTAMP DEFAULT now()
);
"""


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


pytestmark = pytest.mark.skipif(
    _pg_url() is None, reason="no local Postgres for migration tests"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("mig077", _MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def scratch_db():
    """A throwaway database, dropped afterwards."""
    admin = sa.create_engine(_pg_url(), isolation_level="AUTOCOMMIT")
    name = "toup_mig077_test"
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
    return {c["name"] for c in sa.inspect(conn).get_columns("build_jobs")}


def _indexes(conn) -> set[str]:
    return {i["name"] for i in sa.inspect(conn).get_indexes("build_jobs")}


def _run(conn, fn):
    """Execute a migration function against a live connection.

    `Operations.context()` installs the global `alembic.op` proxy the
    revision module imports, so `upgrade()`/`downgrade()` run exactly as
    they would under `alembic upgrade`.
    """
    with Operations.context(MigrationContext.configure(conn)):
        fn()


def test_upgrade_adds_columns_and_indexes(scratch_db):
    mig = _load_migration()
    with scratch_db.begin() as conn:
        conn.execute(sa.text(_BASE_DDL))
    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)
    with scratch_db.connect() as conn:
        assert NEW_COLUMNS <= _columns(conn)
        idx = _indexes(conn)
        assert "ix_build_jobs_archived_at" in idx
        assert "ix_build_jobs_user_active" in idx


def test_downgrade_fully_reverses(scratch_db):
    """Reversibility is the whole safety story for this revision."""
    mig = _load_migration()
    with scratch_db.begin() as conn:
        conn.execute(sa.text(_BASE_DDL))
    with scratch_db.connect() as conn:
        before_cols, before_idx = _columns(conn), _indexes(conn)

    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)
    with scratch_db.begin() as conn:
        _run(conn, mig.downgrade)

    with scratch_db.connect() as conn:
        assert _columns(conn) == before_cols
        assert _indexes(conn) == before_idx


def test_upgrade_is_rerunnable_after_partial_apply(scratch_db):
    """An agent DB may already carry these columns from `_alter_statements`
    (that is the authoritative path for agent DBs). The revision must not
    explode when it finds its own work already done."""
    mig = _load_migration()
    with scratch_db.begin() as conn:
        conn.execute(sa.text(_BASE_DDL))
        conn.execute(sa.text(
            "ALTER TABLE build_jobs ADD COLUMN archived_at TIMESTAMP"
        ))
    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)
    with scratch_db.connect() as conn:
        assert NEW_COLUMNS <= _columns(conn)


def test_noop_when_table_absent(scratch_db):
    """Cleanly-partitioned platform DBs have no build_jobs at all. The
    revision must be a silent no-op rather than a deploy-blocking crash."""
    mig = _load_migration()
    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)   # must not raise
    with scratch_db.begin() as conn:
        _run(conn, mig.downgrade)  # must not raise


def test_existing_rows_survive_and_are_nullable(scratch_db):
    """Additive means additive: no row is rewritten, no NOT NULL added."""
    mig = _load_migration()
    with scratch_db.begin() as conn:
        conn.execute(sa.text(_BASE_DDL))
        conn.execute(sa.text(
            "INSERT INTO build_jobs (id,user_id,title,status,error_message) "
            "VALUES ('j1','u1','legacy row','failed','Agent restarted during execution')"
        ))
    with scratch_db.begin() as conn:
        _run(conn, mig.upgrade)
    with scratch_db.connect() as conn:
        row = conn.execute(sa.text(
            "SELECT title, error_message, error_class, archived_at "
            "FROM build_jobs WHERE id='j1'"
        )).one()
        assert row.title == "legacy row"
        # error_message preserved verbatim — the down-migration depends on it.
        assert row.error_message == "Agent restarted during execution"
        assert row.error_class is None
        assert row.archived_at is None
