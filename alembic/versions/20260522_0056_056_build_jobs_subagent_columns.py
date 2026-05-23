"""build_jobs: additive sub-agent columns + composite index.

Phase 1 of the sub-agent spawning arc. Five additions to
``build_jobs`` so the unified job model can carry sub-agent runs
end-to-end:

  - ``parent_job_id``           VARCHAR(36)  FK build_jobs.id, indexed
                                              (depth chain + child-cap
                                              queries). ON DELETE SET
                                              NULL so a parent removal
                                              doesn't cascade-delete
                                              the audit trail; children
                                              survive with NULL
                                              linkage.
  - ``config_json``             JSON         (per-job opaque config —
                                              sub-agent intake stores
                                              task/label/timeout/
                                              credit_budget here so the
                                              dispatcher reads one
                                              source of truth)
  - ``credit_budget_allocated`` FLOAT        (USD/credit slice the
                                              parent allocated to this
                                              child — NULL when
                                              budget enforcement is
                                              off or the row isn't a
                                              sub-agent)
  - ``credit_spent``            FLOAT NOT NULL DEFAULT 0.0
                                              (running total updated
                                              by the LLM-proxy credit
                                              hook; NOT NULL so
                                              ``SUM(credit_spent)``
                                              queries don't need
                                              COALESCE)
  - INDEX ix_build_jobs_parent_status         (parent_job_id, status)
                                              for the per-parent
                                              children-running count
                                              the cap-check runs on
                                              every spawn

Revision ID: 056
Revises: 055
Create Date: 2026-05-22

Purely additive. Reversible. Mirrors ``mig 052`` shape and the
defensive ``_column_exists`` / ``_index_exists`` checks so re-runs
after a partial apply don't error.

Dialect notes
-------------
JSON column uses ``sa.JSON().with_variant(JSONB(), "postgresql")``
matching the existing ``error_json`` / ``channel_results_json``
columns on the same table — so a future query can JOIN across
config_json and the routine-terminal JSON columns without a cast.

SQLite and Postgres both accept ``ADD COLUMN credit_spent FLOAT
NOT NULL DEFAULT 0.0`` in one shot. Postgres backfills existing
rows to 0.0 atomically via the server_default. SQLite likewise.

Downgrade
---------
Drops the index, then the four columns. ``op.batch_alter_table``
on the column drops for SQLite portability (its column-removal
path needs the table-rebuild dance).
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "056"
down_revision = "055"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.056")


def _json_col() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


# (col_name, col_type_factory, nullable, server_default)
_NEW_COLUMNS = (
    ("parent_job_id", lambda: sa.String(36), True, None),
    ("config_json", _json_col, True, None),
    ("credit_budget_allocated", lambda: sa.Float(), True, None),
    ("credit_spent", lambda: sa.Float(), False, "0.0"),
)


_INDEX_NAME = "ix_build_jobs_parent_status"
_INDEX_COLS = ("parent_job_id", "status")
_FK_NAME = "fk_build_jobs_parent_job_id"


def upgrade() -> None:
    conn = op.get_bind()

    for col_name, col_type_factory, nullable, server_default in _NEW_COLUMNS:
        if not _column_exists(conn, "build_jobs", col_name):
            op.add_column(
                "build_jobs",
                sa.Column(
                    col_name,
                    col_type_factory(),
                    nullable=nullable,
                    server_default=server_default,
                ),
            )
            logger.info("[alembic.056] added build_jobs.%s", col_name)
        else:
            logger.info(
                "[alembic.056] build_jobs.%s already present; skipping",
                col_name,
            )

    # Self-referential FK on parent_job_id. SQLite doesn't enforce FKs
    # by default and ALTER TABLE ADD CONSTRAINT requires the batch
    # dance; on SQLite we skip the FK constraint (the column is still
    # there, just without the constraint object — same as the existing
    # ``coalesced_into_job_id`` and ``conversation_id`` columns which
    # also point at related tables without a declared FK on SQLite).
    # Postgres gets the real constraint.
    if conn.dialect.name == "postgresql":
        # Defensive guard so re-running upgrade doesn't error on the
        # second pass.
        existing_fks = {fk["name"] for fk in sa.inspect(conn).get_foreign_keys("build_jobs")}
        if _FK_NAME not in existing_fks:
            op.create_foreign_key(
                _FK_NAME,
                "build_jobs",
                "build_jobs",
                ["parent_job_id"],
                ["id"],
                ondelete="SET NULL",
            )
            logger.info("[alembic.056] added FK %s", _FK_NAME)
        else:
            logger.info("[alembic.056] FK %s already present; skipping", _FK_NAME)

    # Composite index on (parent_job_id, status). Used by
    # count_running_children(parent_job_id) on every spawn — the cap
    # check is a hot path for chatty supervisors so the index pays for
    # itself fast.
    if not _index_exists(conn, "build_jobs", _INDEX_NAME):
        op.create_index(_INDEX_NAME, "build_jobs", list(_INDEX_COLS))
        logger.info("[alembic.056] created index %s", _INDEX_NAME)
    else:
        logger.info("[alembic.056] index %s already present; skipping", _INDEX_NAME)


def downgrade() -> None:
    conn = op.get_bind()

    if _index_exists(conn, "build_jobs", _INDEX_NAME):
        op.drop_index(_INDEX_NAME, table_name="build_jobs")
        logger.info("[alembic.056] dropped index %s", _INDEX_NAME)

    if conn.dialect.name == "postgresql":
        existing_fks = {fk["name"] for fk in sa.inspect(conn).get_foreign_keys("build_jobs")}
        if _FK_NAME in existing_fks:
            op.drop_constraint(_FK_NAME, "build_jobs", type_="foreignkey")
            logger.info("[alembic.056] dropped FK %s", _FK_NAME)

    for col_name, _, _, _ in reversed(_NEW_COLUMNS):
        if _column_exists(conn, "build_jobs", col_name):
            with op.batch_alter_table("build_jobs") as batch:
                batch.drop_column(col_name)
            logger.info("[alembic.056] dropped build_jobs.%s", col_name)


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        cols = {c["name"] for c in insp.get_columns(table)}
        return column in cols
    except Exception:
        return False


def _index_exists(conn, table: str, index_name: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        indexes = {ix["name"] for ix in insp.get_indexes(table)}
        return index_name in indexes
    except Exception:
        return False
