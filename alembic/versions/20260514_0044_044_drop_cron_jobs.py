"""Phase D — drop the cron_jobs table

Revision ID: 044
Revises: 043
Create Date: 2026-05-14

Final step of the CronJob → Routine consolidation. Drops the
`cron_jobs` table entirely.

Pre-flight guard
================
Before the DROP runs, we verify that every enabled row has been
migrated (mig 043 set `migrated_to_routine_id`). If any enabled row
remains un-migrated, the migration ABORTS with a clear error message
— this prevents accidental data loss for tenants where the Phase B
backfill skipped a row (e.g., malformed legacy schedule expressions).

Operator runbook if the guard trips:
  1. Query: SELECT id, name, schedule_kind FROM cron_jobs
            WHERE enabled=true AND migrated_to_routine_id IS NULL;
  2. Either:
     (a) Fix the schedule_spec and re-run mig 043, or
     (b) Manually create a Routine + set migrated_to_routine_id, or
     (c) Disable the cron_jobs row (enabled=false) — the guard
         only checks enabled rows.
  3. Re-run mig 044.

Idempotent — replays as a no-op when the table is already gone.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def _cron_table_exists(conn) -> bool:
    try:
        insp = sa.inspect(conn)
        return "cron_jobs" in set(insp.get_table_names())
    except Exception:
        return False


def upgrade() -> None:
    conn = op.get_bind()

    if not _cron_table_exists(conn):
        # Already dropped — replay no-op. Common case in CI when
        # rebuilding the DB after a fresh checkout.
        return

    # Pre-flight: confirm every enabled cron_job has a routine
    # backfill. Phase D MUST NOT silently drop user-scheduled work.
    leftover = conn.execute(sa.text("""
        SELECT COUNT(*) FROM cron_jobs
        WHERE enabled = true
          AND migrated_to_routine_id IS NULL
    """)).scalar() or 0

    if leftover > 0:
        # ABORT — keep the table around. Operator runbook is in the
        # docstring above. We raise instead of just warning so the
        # next migration step doesn't accidentally proceed.
        raise RuntimeError(
            f"Migration 044 (drop cron_jobs) aborted: {leftover} enabled "
            "cron_jobs rows still un-migrated. Run migration 043 first "
            "or disable the leftover rows manually. See the docstring "
            "in this migration file for the runbook."
        )

    print(f"[044] dropping cron_jobs table (all rows migrated to routines)")

    # Drop indexes first if present — be defensive against partial
    # past-state where the migrated_to_routine_id index from mig 042
    # is the only thing left.
    try:
        op.drop_index("ix_cron_jobs_migrated_to_routine_id", table_name="cron_jobs")
    except Exception:
        pass

    op.drop_table("cron_jobs")


def downgrade() -> None:
    """Recreate the cron_jobs table. We can only restore the schema,
    not the data — the rows have been migrated into routines and the
    user-facing experience now runs through that path."""
    conn = op.get_bind()
    if _cron_table_exists(conn):
        return

    op.create_table(
        "cron_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("schedule_kind", sa.String(length=20), nullable=False),
        sa.Column("schedule_spec", sa.String(length=200), nullable=False),
        sa.Column("schedule_at", sa.DateTime(), nullable=True),
        sa.Column("schedule_interval_seconds", sa.Integer(), nullable=True),
        sa.Column("schedule_cron_expr", sa.String(length=100), nullable=True),
        sa.Column("payload_text", sa.Text(), nullable=False),
        sa.Column("telegram_chat_id", sa.BigInteger(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("run_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("migrated_to_routine_id", sa.String(length=36), nullable=True),
    )
    op.create_index("ix_cron_jobs_user_id", "cron_jobs", ["user_id"])
    op.create_index(
        "ix_cron_jobs_migrated_to_routine_id",
        "cron_jobs",
        ["migrated_to_routine_id"],
    )
