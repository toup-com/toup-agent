"""build_jobs: additive runner-state columns so the runners can
read build_jobs as source of truth (prerequisite for the rest of
the cutover arc that retires trigger_events + routine_runs).

Revision ID: 051
Revises: 050
Create Date: 2026-05-20

Adds three runner-state columns to ``build_jobs``:

  - ``fire_instant``        DATETIME   (routines: APScheduler's fire
                                        moment, distinct from
                                        ``created_at`` which is the
                                        DB insert moment — Ticket 2.3
                                        in docs/routines/ for the
                                        observability rationale)
  - ``attempt``             INTEGER    (routines: retry counter
                                        stamped by ``_run_with_retry``
                                        so Mission Control can see
                                        "fired N times before
                                        succeeding/failing")
  - ``coalesced_into_job_id`` VARCHAR(36)
                                       (triggers: when an inbound
                                        event was folded into an
                                        already-running sibling
                                        handler, this points at the
                                        parent's BuildJob. Mirrors
                                        ``trigger_events.coalesced_
                                        into_event_id`` semantics in
                                        the build_jobs-as-source-of-
                                        truth world.)

Purely additive. Nullable, no server defaults. Reversible.

Rollout order (this is PR #45 of the cutover arc)
--------------------------------------------------
1. PR #45 (this): apply mig 051 — adds columns, NO-OP for callers.
2. PR #46: trigger runner cutover — every legacy read moves to
   build_jobs + new columns.
3. PR #47: routine runner cutover — same for routines + handler
   signature change.
4. PR #48: remove dual-writes (no reader left for legacy tables).
5. PR #49: flip ALLOW_LEGACY_JOB_TABLES_DROP=true → next deploy
   applies migration 050 (drops trigger_events + routine_runs +
   build_jobs.steps_json).

Dialect notes
-------------
Postgres + SQLite both accept ``ADD COLUMN ... NULL``. No partial
indexes here; the trigger queue scan will use the existing
``source_kind`` filtering. If query plans regress on Postgres
after the cutover, a follow-up migration can add an explicit
``(status, source_kind, created_at)`` composite index — defer
until measured.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "051"
down_revision = "050"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.051")


_NEW_COLUMNS = (
    ("fire_instant", sa.DateTime, True),
    ("attempt", sa.Integer, True),
    ("coalesced_into_job_id", sa.String(36), True),
)


def upgrade() -> None:
    conn = op.get_bind()
    for col_name, col_type, nullable in _NEW_COLUMNS:
        if not _column_exists(conn, "build_jobs", col_name):
            op.add_column(
                "build_jobs",
                sa.Column(col_name, col_type, nullable=nullable),
            )
            logger.info("[alembic.051] added build_jobs.%s", col_name)
        else:
            logger.info(
                "[alembic.051] build_jobs.%s already present; skipping",
                col_name,
            )


def downgrade() -> None:
    conn = op.get_bind()
    for col_name, _, _ in reversed(_NEW_COLUMNS):
        if _column_exists(conn, "build_jobs", col_name):
            with op.batch_alter_table("build_jobs") as batch:
                batch.drop_column(col_name)
            logger.info("[alembic.051] dropped build_jobs.%s", col_name)


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        cols = {c["name"] for c in insp.get_columns(table)}
        return column in cols
    except Exception:
        return False
