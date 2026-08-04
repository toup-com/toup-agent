"""
Smoke test for migration 021 (llm_proxy operation_type + day_chat archival columns).

Applies upgrade() then downgrade() against a fresh sqlite DB seeded with the
pre-021 schema for the two affected tables. Confirms:
  - Upgrade adds the columns (with NULL / server_default), non-blocking.
  - Existing rows retain their original values (no data loss).
  - Downgrade cleanly removes the columns.
  - Indexes come and go with the columns.

Until 2026-08-04 this file held all of that inside a `main()` with no `test_`
function, so pytest collected NOTHING and exited 5 — "no tests ran" is not a
pass, but on a per-file runner it looks like one. It also wrote to a FIXED
path, `/tmp/mig021_test.db`, which two concurrent runs would clobber; the sweep
runs several pytest processes at once, so that was a live collision. Both are
fixed: real tests, and `tmp_path`.

Split in two on purpose. As one blob, an upgrade failure and a downgrade
failure were indistinguishable in the summary line.

Run: pytest tests/test_migration_021_roundtrip.py   (or execute this file directly)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text, inspect
import importlib.util as _ilu

# Load just the migration module without alembic (avoids the broken 005 edge).
_spec = _ilu.spec_from_file_location(
    "mig021",
    str(Path(__file__).resolve().parent.parent / "alembic/versions/20260416_0021_021_llm_operation_type.py"),
)
_m = _ilu.module_from_spec(_spec)


def _mock_alembic_op(conn):
    """Minimal shim around alembic.op that applies DDL via a SQLAlchemy connection."""
    import sqlalchemy as sa_mod

    class _Op:
        @staticmethod
        def add_column(table, col):
            coltype = col.type.compile(conn.dialect)
            nullable = "NULL" if col.nullable else "NOT NULL"
            default = ""
            if col.server_default is not None:
                default_text = col.server_default.arg
                if hasattr(default_text, "text"):
                    default_text = default_text.text
                default = f" DEFAULT '{default_text}'"
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col.name} {coltype} {nullable}{default}"))

        @staticmethod
        def create_index(name, table, cols):
            conn.execute(text(f"CREATE INDEX {name} ON {table} ({', '.join(cols)})"))

        @staticmethod
        def drop_column(table, col):
            # SQLite supports DROP COLUMN on 3.35+
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {col}"))

        @staticmethod
        def drop_index(name, table_name=None):
            conn.execute(text(f"DROP INDEX {name}"))

    return _Op()


def _seed_pre_021_schema(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE llm_proxy_events (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL,
                provider VARCHAR(20) NOT NULL,
                model VARCHAR(100) NOT NULL,
                endpoint VARCHAR(20) NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_cents INTEGER DEFAULT 0,
                was_fallback BOOLEAN DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                status VARCHAR(10) DEFAULT 'ok',
                created_at TIMESTAMP NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE day_chats (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL,
                local_date DATE NOT NULL,
                timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP,
                last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT,
                summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP,
                summary_status VARCHAR(20) DEFAULT 'up_to_date'
            )
        """))
        # Seed data the migration must not lose.
        conn.execute(text(
            "INSERT INTO llm_proxy_events (id, user_id, provider, model, endpoint, cost_cents, created_at) "
            "VALUES ('pre1', 'u', 'anthropic', 'claude', 'chat', 123, '2026-04-01 12:00:00')"
        ))
        conn.execute(text(
            "INSERT INTO day_chats (id, user_id, local_date, rolling_summary, summary_status) "
            "VALUES ('dc1', 'u', '2026-04-10', 'existing summary', 'up_to_date')"
        ))


_EXECED = False


def _load_migration():
    """Exec the migration module once per process.

    Deliberately lazy — the module does `from alembic import op` at import,
    and the original file avoided doing that at collection time to sidestep a
    broken revision in the chain. Keep that property.
    """
    global _EXECED
    if not _EXECED:
        _spec.loader.exec_module(_m)
        _EXECED = True
    return _m


def _apply(engine, direction: str):
    """Run upgrade()/downgrade() with alembic.op swapped for the sqlite shim."""
    m = _load_migration()
    original_op = m.op
    with engine.begin() as conn:
        m.op = _mock_alembic_op(conn)
        try:
            getattr(m, direction)()
        finally:
            m.op = original_op


@pytest.fixture
def upgraded(tmp_path):
    """Pre-021 schema, seeded, with upgrade() applied. Own DB file per test."""
    engine = create_engine(f"sqlite:///{tmp_path / 'mig021.db'}")
    _seed_pre_021_schema(engine)
    _apply(engine, "upgrade")
    try:
        yield engine
    finally:
        engine.dispose()


def test_upgrade_adds_columns_and_preserves_existing_rows(upgraded):
    insp = inspect(upgraded)
    lpe_cols = {c["name"] for c in insp.get_columns("llm_proxy_events")}
    assert "operation_type" in lpe_cols, f"operation_type missing: {lpe_cols}"

    dc_cols = {c["name"] for c in insp.get_columns("day_chats")}
    for col in ("archival_summary", "archival_summary_generated_at",
                "archival_summary_status"):
        assert col in dc_cols, f"{col} missing: {dc_cols}"

    with upgraded.begin() as conn:
        row = conn.execute(text(
            "SELECT id, operation_type FROM llm_proxy_events WHERE id = 'pre1'"
        )).first()
        assert row[0] == "pre1"
        assert row[1] is None, f"pre-existing row should have NULL operation_type, got {row[1]!r}"

        row = conn.execute(text(
            "SELECT id, rolling_summary, archival_summary, archival_summary_status "
            "FROM day_chats WHERE id = 'dc1'"
        )).first()
        assert row[0] == "dc1"
        assert row[1] == "existing summary"
        assert row[2] is None, "archival_summary should be NULL for a pre-existing row"
        assert row[3] == "not_needed", (
            f"archival_summary_status should default to 'not_needed', got {row[3]!r}"
        )

    lpe_idx = {i["name"] for i in insp.get_indexes("llm_proxy_events")}
    assert "ix_llm_proxy_operation_type" in lpe_idx, lpe_idx


def test_downgrade_removes_columns_and_keeps_rows(upgraded):
    _apply(upgraded, "downgrade")

    insp = inspect(upgraded)
    lpe_cols = {c["name"] for c in insp.get_columns("llm_proxy_events")}
    assert "operation_type" not in lpe_cols, f"downgrade left operation_type: {lpe_cols}"

    dc_cols = {c["name"] for c in insp.get_columns("day_chats")}
    for col in ("archival_summary", "archival_summary_generated_at",
                "archival_summary_status"):
        assert col not in dc_cols, f"downgrade left {col}: {dc_cols}"

    lpe_idx = {i["name"] for i in insp.get_indexes("llm_proxy_events")}
    assert "ix_llm_proxy_operation_type" not in lpe_idx, lpe_idx

    # A downgrade drops COLUMNS, never rows.
    with upgraded.begin() as conn:
        assert conn.execute(text(
            "SELECT id FROM llm_proxy_events WHERE id='pre1'")).first() is not None
        assert conn.execute(text(
            "SELECT id FROM day_chats WHERE id='dc1'")).first() is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
