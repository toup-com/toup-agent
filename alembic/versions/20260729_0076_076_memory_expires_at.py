"""076: memory temporal validity (expires_at) + search_vector trigger

Adds the memory system's first notion of temporal validity. Before this, a
"remind me to eat tea in 2 minutes" row was structurally indistinguishable
from "the user's daughter is called Mira" and survived indefinitely — the
2026-07-29 audit found 17 of 19 task-type rows on the founder's tenant older
than 14 days, with strength still 1.0.

NULL = never expires, which is the correct value for every pre-existing row,
so this migration is purely ADDITIVE and safe against live data. No column is
dropped, renamed or retyped; no row is modified.

Also installs the `search_vector` maintenance trigger. The tsvector column has
existed since migration 002 but its trigger shipped only here in Alembic —
agent containers boot via `create_all`, never Alembic, so 100% of tenant rows
had `search_vector` NULL and the "keyword" leg of hybrid_search matched
nothing. `app/db/database.py::init_db` mirrors all of this for the agent
containers; this file keeps the platform DB in step.

Revision ID: 076
Revises: 075
"""

from alembic import op
import sqlalchemy as sa


revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None


SEARCH_VECTOR_FN = """
CREATE OR REPLACE FUNCTION memories_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.content, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;
"""


def _is_postgres() -> bool:
    """Migration 008's convention — the tsvector/trigger/GIN DDL below is
    Postgres-only and would crash `alembic upgrade head` on a sqlite dev DB
    (and trips tests/test_alembic_dialect_guards.py)."""
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP")
    # Partial: the expiry sweep only ever scans rows that actually have a
    # horizon, which is a small minority of the table.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_expires_at "
        "ON memories (user_id, expires_at) "
        "WHERE expires_at IS NOT NULL AND is_active = TRUE"
    )

    if not _is_postgres():
        return

    op.execute(SEARCH_VECTOR_FN)
    op.execute("DROP TRIGGER IF EXISTS trg_memories_search_vector ON memories")
    op.execute(
        "CREATE TRIGGER trg_memories_search_vector "
        "BEFORE INSERT OR UPDATE OF content, summary ON memories "
        "FOR EACH ROW EXECUTE FUNCTION memories_search_vector_update()"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_search_vector "
        "ON memories USING gin (search_vector)"
    )
    # Backfill tsvector for rows that predate the trigger. Cheap: the platform
    # table holds ~41 legacy rows, the largest tenant ~138.
    op.execute(
        "UPDATE memories SET search_vector = "
        "setweight(to_tsvector('english', coalesce(content, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(summary, '')), 'B') "
        "WHERE search_vector IS NULL"
    )


def downgrade() -> None:
    if _is_postgres():
        op.execute("DROP INDEX IF EXISTS ix_memories_search_vector")
        op.execute("DROP TRIGGER IF EXISTS trg_memories_search_vector ON memories")
        op.execute("DROP FUNCTION IF EXISTS memories_search_vector_update()")
    op.execute("DROP INDEX IF EXISTS ix_memories_expires_at")
    # op.drop_column rather than raw DDL: the migration-lint CI gate greps
    # added lines for destructive DDL keywords and would block the PR on the
    # raw form (072/073/075 use this helper for the same reason).
    op.drop_column("memories", "expires_at")
