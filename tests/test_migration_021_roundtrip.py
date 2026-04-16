"""
Smoke test for migration 021 (llm_proxy operation_type + day_chat archival columns).

Applies upgrade() then downgrade() against a fresh sqlite DB seeded with the
pre-021 schema for the two affected tables. Confirms:
  - Upgrade adds the columns (with NULL / server_default), non-blocking.
  - Existing rows retain their original values (no data loss).
  - Downgrade cleanly removes the columns.
  - Indexes come and go with the columns.

Run: python tests/test_migration_021_roundtrip.py
"""

import sys
from pathlib import Path

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


def main():
    import os
    db_path = "/tmp/mig021_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = create_engine(f"sqlite:///{db_path}")

    # 1. Build the pre-021 schema.
    _seed_pre_021_schema(engine)

    # 2. Apply upgrade() by rewriting alembic.op to a local shim that uses
    #    our SQLAlchemy connection.
    _spec.loader.exec_module(_m)
    with engine.begin() as conn:
        shim = _mock_alembic_op(conn)
        _original_op = _m.op
        _m.op = shim
        try:
            _m.upgrade()
        finally:
            _m.op = _original_op

    # 3. Verify the columns exist and data is preserved.
    insp = inspect(engine)
    lpe_cols = {c["name"] for c in insp.get_columns("llm_proxy_events")}
    assert "operation_type" in lpe_cols, f"operation_type missing: {lpe_cols}"

    dc_cols = {c["name"] for c in insp.get_columns("day_chats")}
    assert "archival_summary" in dc_cols, dc_cols
    assert "archival_summary_generated_at" in dc_cols, dc_cols
    assert "archival_summary_status" in dc_cols, dc_cols

    with engine.begin() as conn:
        # Data preserved.
        row = conn.execute(text(
            "SELECT id, operation_type FROM llm_proxy_events WHERE id = 'pre1'"
        )).first()
        assert row[0] == "pre1"
        assert row[1] is None, f"Existing row should have NULL operation_type, got {row[1]!r}"

        row = conn.execute(text(
            "SELECT id, rolling_summary, archival_summary, archival_summary_status "
            "FROM day_chats WHERE id = 'dc1'"
        )).first()
        assert row[0] == "dc1"
        assert row[1] == "existing summary"
        assert row[2] is None, "archival_summary should be NULL for pre-existing row"
        assert row[3] == "not_needed", (
            f"archival_summary_status should default to 'not_needed', got {row[3]!r}"
        )

    # Indexes created?
    lpe_idx = {i["name"] for i in insp.get_indexes("llm_proxy_events")}
    assert "ix_llm_proxy_operation_type" in lpe_idx, lpe_idx

    print("✓ upgrade() — columns added, data preserved, index created")

    # 4. Now run downgrade() and verify columns/indexes disappear.
    with engine.begin() as conn:
        shim = _mock_alembic_op(conn)
        _m.op = shim
        try:
            _m.downgrade()
        finally:
            _m.op = _original_op

    insp = inspect(engine)
    lpe_cols_after = {c["name"] for c in insp.get_columns("llm_proxy_events")}
    assert "operation_type" not in lpe_cols_after, (
        f"downgrade did not drop operation_type: {lpe_cols_after}"
    )
    dc_cols_after = {c["name"] for c in insp.get_columns("day_chats")}
    assert "archival_summary" not in dc_cols_after
    assert "archival_summary_generated_at" not in dc_cols_after
    assert "archival_summary_status" not in dc_cols_after

    # Indexes gone.
    lpe_idx_after = {i["name"] for i in insp.get_indexes("llm_proxy_events")}
    assert "ix_llm_proxy_operation_type" not in lpe_idx_after, lpe_idx_after

    # Seed data still present — downgrade must not drop rows.
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id FROM llm_proxy_events WHERE id='pre1'")).first()
        assert row is not None
        row = conn.execute(text("SELECT id FROM day_chats WHERE id='dc1'")).first()
        assert row is not None

    print("✓ downgrade() — columns dropped, indexes dropped, data preserved")
    print("\nMigration 021 round-trip: PASS")

    engine.dispose()
    os.remove(db_path)


if __name__ == "__main__":
    main()
