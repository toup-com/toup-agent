"""routine_runs: add nullable job_id linkage column

Revision ID: 048
Revises: 047
Create Date: 2026-05-18

PR 4b of the unified-jobs arc. Mirror of migration 047
(trigger_events.job_id) — adds the same nullable ``job_id``
linkage column to ``routine_runs`` so every routine fire row
carries a soft pointer to its mirrored ``build_jobs`` row
(populated by PR 4b's runner change).

No FK on the column — same precedent as
``cron_jobs.migrated_to_routine_id`` and
``trigger_events.job_id``: the linkage survives a future Phase-D
drop of ``routine_runs`` without CASCADE firing on rows we want
to keep.

Purely additive. Reversible.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.048")


def upgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "routine_runs"):
        logger.info(
            "[alembic.048] routine_runs not present; skipping job_id add"
        )
        return
    if not _column_exists(conn, "routine_runs", "job_id"):
        op.add_column(
            "routine_runs",
            sa.Column("job_id", sa.String(36), nullable=True),
        )
        logger.info("[alembic.048] routine_runs.job_id added")
    else:
        logger.info(
            "[alembic.048] routine_runs.job_id already present; skipping"
        )


def downgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "routine_runs"):
        return
    if _column_exists(conn, "routine_runs", "job_id"):
        with op.batch_alter_table("routine_runs") as batch:
            batch.drop_column("job_id")
        logger.info("[alembic.048] routine_runs.job_id dropped")


def _table_exists(conn, table: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        return table in set(insp.get_table_names())
    except Exception:
        return False


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        cols = {c["name"] for c in insp.get_columns(table)}
        return column in cols
    except Exception:
        return False
