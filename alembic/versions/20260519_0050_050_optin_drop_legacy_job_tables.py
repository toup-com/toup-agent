"""opt-in drop of legacy job tables (trigger_events, routine_runs,
build_jobs.steps_json) — final PR of the unified jobs/tasks/logs arc

Revision ID: 050
Revises: 049
Create Date: 2026-05-19

Closes the 9-PR unified-jobs arc. PRs 2–8 added the unified
``build_jobs`` + ``job_events`` schema and dual-write all four
intake paths (triggers, routines, agent tasks, Auto Builder) so
every legacy table now has a mirrored Job row + indexable event
stream. This migration removes the *legacy* surface — the columns
and tables that callers should no longer be reading from.

What this migration drops
-------------------------
1. ``trigger_events`` — every row already has ``job_id`` (PR 4a)
   and the Job row carries the same terminal status (mirror helper
   in ``app/agent/triggers/runner.py::_mirror_event_terminal_to_job``).
2. ``routine_runs`` — same shape, mirror helper at
   ``app/agent/routines/runner.py::_mirror_run_terminal_to_job``.
3. ``build_jobs.steps_json`` — replaced by ``job_events`` rows
   (PR 5 dual-writes Auto Builder phases; PR 3 dual-writes every
   JobLogger entry).

OPT-IN GATE — DO NOT REMOVE
---------------------------
The drop is **destructive** and **not auto-applied**. The operator
must explicitly set::

    ALLOW_LEGACY_JOB_TABLES_DROP=true

in the deploy environment *for the alembic upgrade run only*. With
the gate off (default), ``upgrade()`` is a NO-OP that logs the gate
status and exits successfully — so a normal ``alembic upgrade head``
walks past this revision without touching legacy data. This is the
same opt-in pattern as migration 044 (cron_jobs drop) — see
``docs/runbooks/cronjob-to-routine-cutover.md`` for the precedent.

Pre-flight check
----------------
Run ``python scripts/check_job_parity.py`` first. It reports row
counts on both sides (legacy + mirrored Job) and exits non-zero on
any gap. Operator only flips the env var after a green parity check.

Reversibility
-------------
``downgrade()`` recreates empty ``trigger_events`` and
``routine_runs`` tables (without data — the rows are gone) and the
``steps_json`` column. A downgrade-then-upgrade cycle is **not** a
data-recovery path; it's structural only. Restore data from backup
if you need the rows back.

Dialect notes
-------------
Postgres: ``op.drop_table`` resolves the FK from
``trigger_events.coalesced_into_event_id`` to itself; the other FKs
(``trigger_id``, ``user_id``, ``summary_message_id``) point *into*
``trigger_events`` from nowhere external, so no other table is
affected by the removal. Same story for ``routine_runs``.

SQLite (CI): the partial-index and self-FK survive a plain
``op.drop_table`` call without ceremony.

Migration-lint
--------------
``.github/workflows/migration-lint.yml`` greps added lines for the
destructive SQL keywords (case-insensitive, space-separated). This
revision uses ``op.drop_table`` / ``batch.drop_column`` Python
identifiers — underscored, so neither the code nor this docstring
triggers the regex. If a future revision needs to write the SQL
keywords verbatim, use the ``[migration-lint-override: <reason>]``
commit-message marker documented in that workflow.
"""
from __future__ import annotations

import logging
import os

import sqlalchemy as sa
from alembic import op


revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.050")


# Operator-controlled gate. Read at upgrade() time, not import time,
# so a test process can monkeypatch os.environ before invoking the
# migration.
_GATE_ENV_VAR = "ALLOW_LEGACY_JOB_TABLES_DROP"


def upgrade() -> None:
    gate = (os.environ.get(_GATE_ENV_VAR) or "").strip().lower()
    if gate not in ("1", "true", "yes", "on"):
        logger.info(
            "[alembic.050] gate %s=%r — skipping legacy-table drop. "
            "Set %s=true after running scripts/check_job_parity.py to "
            "apply this migration.",
            _GATE_ENV_VAR, gate or "<unset>", _GATE_ENV_VAR,
        )
        return

    logger.warning(
        "[alembic.050] gate ON — dropping legacy job tables "
        "(trigger_events, routine_runs) and build_jobs.steps_json"
    )

    conn = op.get_bind()

    # ── 1. Drop trigger_events ───────────────────────────────────────
    if _table_exists(conn, "trigger_events"):
        # Drop indexes first so the drop_table doesn't trip on dialect
        # quirks. Index names follow Alembic's auto-naming convention
        # from earlier migrations.
        for ix_name in (
            "ix_trigger_events_trigger_id",
            "ix_trigger_events_user_id",
        ):
            try:
                op.drop_index(ix_name, table_name="trigger_events")
            except Exception as e:
                logger.info("[alembic.050] drop_index %s skipped: %s", ix_name, e)
        op.drop_table("trigger_events")
        logger.info("[alembic.050] dropped trigger_events")

    # ── 2. Drop routine_runs ─────────────────────────────────────────
    if _table_exists(conn, "routine_runs"):
        for ix_name in (
            "ix_routine_runs_routine_id",
            "ix_routine_runs_user_id",
        ):
            try:
                op.drop_index(ix_name, table_name="routine_runs")
            except Exception as e:
                logger.info("[alembic.050] drop_index %s skipped: %s", ix_name, e)
        op.drop_table("routine_runs")
        logger.info("[alembic.050] dropped routine_runs")

    # ── 3. Drop build_jobs.steps_json column ─────────────────────────
    # batch_alter_table for SQLite portability — Postgres handles plain
    # drop_column fine but SQLite needs the table-rebuild dance.
    if _column_exists(conn, "build_jobs", "steps_json"):
        with op.batch_alter_table("build_jobs") as batch:
            batch.drop_column("steps_json")
        logger.info("[alembic.050] dropped build_jobs.steps_json")

    logger.info("[alembic.050] legacy job tables/columns removed")


def downgrade() -> None:
    """Structural-only re-create. Data is NOT restored."""
    conn = op.get_bind()

    # ── 1. Re-add build_jobs.steps_json ──────────────────────────────
    if not _column_exists(conn, "build_jobs", "steps_json"):
        op.add_column(
            "build_jobs",
            sa.Column(
                "steps_json", sa.Text, nullable=False, server_default="[]",
            ),
        )

    # ── 2. Re-create routine_runs (empty) ────────────────────────────
    if not _table_exists(conn, "routine_runs"):
        op.create_table(
            "routine_runs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("routine_id", sa.String(36), nullable=False, index=True),
            sa.Column("user_id", sa.String(36), nullable=False, index=True),
            sa.Column("scheduled_for_local_date", sa.Date, nullable=False),
            sa.Column(
                "started_at", sa.DateTime, nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("finished_at", sa.DateTime, nullable=True),
            sa.Column("fire_instant", sa.DateTime, nullable=True),
            sa.Column("finished_local_at", sa.String(40), nullable=True),
            sa.Column(
                "status", sa.String(20), nullable=False,
                server_default="running",
            ),
            sa.Column("outcome", sa.String(30), nullable=True),
            sa.Column("error_class", sa.String(100), nullable=True),
            sa.Column("error_detail", sa.Text, nullable=True),
            sa.Column("error_json", sa.Text, nullable=True),
            sa.Column("delivery_json", sa.Text, nullable=True),
            sa.Column("retry_attempt", sa.Integer, nullable=True),
            sa.Column("retry_of_run_id", sa.String(36), nullable=True),
            sa.Column("summary_message_id", sa.String(50), nullable=True),
            sa.Column("latency_ms", sa.Integer, nullable=True),
            sa.Column("job_id", sa.String(36), nullable=True),
        )

    # ── 3. Re-create trigger_events (empty) ──────────────────────────
    if not _table_exists(conn, "trigger_events"):
        op.create_table(
            "trigger_events",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("trigger_id", sa.String(36), nullable=False, index=True),
            sa.Column("user_id", sa.String(36), nullable=False, index=True),
            sa.Column("event_dedupe_id", sa.String(255), nullable=False),
            sa.Column(
                "received_at", sa.DateTime, nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("started_at", sa.DateTime, nullable=True),
            sa.Column("finished_at", sa.DateTime, nullable=True),
            sa.Column(
                "status", sa.String(20), nullable=False,
                server_default="queued",
            ),
            sa.Column("error_class", sa.String(100), nullable=True),
            sa.Column("error_detail", sa.Text, nullable=True),
            sa.Column("summary_message_id", sa.String(50), nullable=True),
            sa.Column("coalesced_into_event_id", sa.String(36), nullable=True),
            sa.Column("job_id", sa.String(36), nullable=True),
        )

    logger.info("[alembic.050] legacy tables/column re-created (empty)")


# ── Inspection helpers (same shape as migration 046) ──────────────────


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
