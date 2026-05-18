"""trigger_events: add nullable job_id linkage column

Revision ID: 047
Revises: 046
Create Date: 2026-05-18

PR 4a of the unified-jobs arc. Adds ``trigger_events.job_id`` so
every TriggerEvent row carries a soft pointer to its mirrored
``build_jobs`` row (PR 4a wires the dual-write in
``triggers_inbound``). No FK on the column for the same reason as
``cron_jobs.migrated_to_routine_id``: the linkage survives a future
Phase-D-style drop of the legacy ``trigger_events`` table without
the FK firing CASCADE on rows we want to retain.

Purely additive. Reversible.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.047")


def upgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "trigger_events"):
        # Platform DB doesn't have this table — no-op.
        logger.info(
            "[alembic.047] trigger_events not present; skipping job_id add"
        )
        return
    if not _column_exists(conn, "trigger_events", "job_id"):
        op.add_column(
            "trigger_events",
            sa.Column("job_id", sa.String(36), nullable=True),
        )
        logger.info("[alembic.047] trigger_events.job_id added")
    else:
        logger.info("[alembic.047] trigger_events.job_id already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "trigger_events"):
        return
    if _column_exists(conn, "trigger_events", "job_id"):
        with op.batch_alter_table("trigger_events") as batch:
            batch.drop_column("job_id")
        logger.info("[alembic.047] trigger_events.job_id dropped")


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
