"""build_jobs: additive routine-terminal columns so the routine
runner can read build_jobs as source of truth for the terminal
shape (prerequisite for PR #48 which removes the legacy
``routine_runs`` writes, and PR #49 which drops the table).

Revision ID: 052
Revises: 051
Create Date: 2026-05-20

Adds five columns to ``build_jobs`` matching the legacy
``routine_runs`` terminal shape:

  - ``emails_fetched``       INTEGER   (routines: email_briefing
                                        handler emits the count of
                                        emails it pulled this fire —
                                        used by Mission Control to
                                        show "delivered N emails")
  - ``finished_local_at``    VARCHAR(40) (routines: finished_at
                                        rendered in the user's IANA
                                        tz, ISO8601 — render-time
                                        helper so dashboards don't
                                        have to join through
                                        User.timezone)
  - ``error_json``           JSON      (routines: structured error
                                        blob — superset of
                                        ``error_message``: class,
                                        detail, attempt, optional
                                        tool_name / tool_call_id /
                                        provider / http_status)
  - ``channel_results_json`` JSON      (routines: per-channel
                                        delivery results — shape
                                        ``{"website": {"status":
                                        "delivered", ...},
                                        "telegram": {...}, ...}``;
                                        empty on tool_error / failure
                                        outcomes)
  - ``tools_invoked_json``   JSON      (routines: list of MCP tool
                                        names the handler invoked
                                        during this run — helpful for
                                        "did this routine actually
                                        call Gmail?" forensics on a
                                        failure)

Purely additive. Nullable, no server defaults. Reversible. Uses
``JSON().with_variant(JSONB(), "postgresql")`` to match the existing
``routine_runs.{error_json, channel_results_json, tools_invoked_json}``
column types so a future re-target of readers from one table to the
other is type-compatible.

Rollout order (this is PR #47 of the cutover arc)
--------------------------------------------------
1. PR #45: mig 051 — additive runner-state columns (fire_instant,
   attempt, coalesced_into_job_id).
2. PR #46: trigger runner cutover — every legacy read moves to
   build_jobs + new columns.
3. PR #47 (this): routine runner cutover — same for routines. Adds
   mig 052 + ORM columns + cuts five sites in the routine runner.
4. PR #48: remove dual-writes (no reader left for legacy tables).
5. PR #49: flip ALLOW_LEGACY_JOB_TABLES_DROP=true → next deploy
   applies migration 050 (drops trigger_events + routine_runs +
   build_jobs.steps_json).

Dialect notes
-------------
SQLite stores JSON as TEXT; Postgres uses JSONB on the variant.
``op.batch_alter_table`` on downgrade is required for SQLite —
its column-removal path can't run a bare ``op.drop_column`` without
the table-rebuild dance.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision = "052"
down_revision = "051"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.052")


# JSON-typed columns use the same dialect-aware variant as RoutineRun
# in ``app/db/models/routine.py`` — SQLite stores as TEXT, Postgres
# stores as JSONB. Keeping the types in sync means a future PR can
# point readers at either column without a cast.
def _json_col() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


# (col_name, col_type_factory, nullable). col_type_factory is a
# zero-arg callable so we instantiate JSON variants once per call
# (Alembic's op.add_column expects a fresh Column each time on some
# dialects).
_NEW_COLUMNS = (
    ("emails_fetched", lambda: sa.Integer(), True),
    ("finished_local_at", lambda: sa.String(40), True),
    ("error_json", _json_col, True),
    ("channel_results_json", _json_col, True),
    ("tools_invoked_json", _json_col, True),
)


def upgrade() -> None:
    conn = op.get_bind()
    for col_name, col_type_factory, nullable in _NEW_COLUMNS:
        if not _column_exists(conn, "build_jobs", col_name):
            op.add_column(
                "build_jobs",
                sa.Column(col_name, col_type_factory(), nullable=nullable),
            )
            logger.info("[alembic.052] added build_jobs.%s", col_name)
        else:
            logger.info(
                "[alembic.052] build_jobs.%s already present; skipping",
                col_name,
            )


def downgrade() -> None:
    conn = op.get_bind()
    # batch_alter_table for SQLite portability — its column-removal
    # path can't take a bare ``op.drop_column`` without the table-
    # rebuild dance.
    for col_name, _, _ in reversed(_NEW_COLUMNS):
        if _column_exists(conn, "build_jobs", col_name):
            with op.batch_alter_table("build_jobs") as batch:
                batch.drop_column(col_name)
            logger.info("[alembic.052] dropped build_jobs.%s", col_name)


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        insp = sa.inspect(conn)
        insp.clear_cache()
        cols = {c["name"] for c in insp.get_columns(table)}
        return column in cols
    except Exception:
        return False
