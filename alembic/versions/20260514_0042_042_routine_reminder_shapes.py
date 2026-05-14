"""routines: schedule shapes (cron/at/every) + reminder text + cron_job link

Revision ID: 042
Revises: 041
Create Date: 2026-05-14

Phase A of the CronJob → Routine consolidation. Adds the columns
required for one-shot reminders, interval reminders, and
text-only reminder delivery, without touching the legacy CronJob
table or its scheduler. CronService keeps running normally.

Columns on `routines`:
  - `schedule_kind`               varchar(10) default 'cron'.
                                  One of: 'cron' | 'at' | 'every'.
  - `schedule_at`                 datetime nullable.
                                  Required when schedule_kind='at'. UTC.
  - `schedule_interval_seconds`   integer nullable.
                                  Required when schedule_kind='every'. Min 60.
  - `schedule_window_start_local` varchar(8) nullable. HH:MM:SS.
                                  Optional active-window for interval kinds.
  - `schedule_window_end_local`   varchar(8) nullable. HH:MM:SS.
  - `auto_disable_after_fire`     bool default false.
                                  When true, runner disables the routine
                                  in _post_terminal after a successful
                                  fire. Server-default for schedule_kind='at'.
  - `reminder_text`               text nullable.
                                  For kind='reminder': literal text to
                                  deliver (no LLM, no MCP).

Column on `cron_jobs`:
  - `migrated_to_routine_id` varchar(36) nullable.
                             Set in Phase B's data migration. Until then
                             always NULL; CronService treats NULL rows
                             as canonical and migrated rows as
                             "owned by Routine, skip me".

Backfill semantics for existing `routines` rows:
  - `schedule_kind` defaults to 'cron' on creation; existing rows pick
    up the default automatically via the server_default. Cron rows
    already have `schedule_cron_local` populated, so the CHECK
    constraint below is satisfied.

CHECK constraints (Postgres only; SQLite ignores via SQLAlchemy's
elision-on-unsupported). Skipping the CHECK on SQLite is fine — the
test env doesn't exercise the malformed-row case and the model-layer
Pydantic validators in app/api/routines.py catch it earlier anyway.

Idempotent — replays cleanly via column-exists guards.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    """Cross-dialect column-existence check. PG uses information_schema;
    SQLite falls back to the SQLAlchemy inspector (which works on both
    but is slower)."""
    try:
        return bool(conn.execute(
            sa.text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name=:t AND column_name=:c"
            ),
            {"t": table, "c": column},
        ).scalar())
    except Exception:
        try:
            insp = sa.inspect(conn)
            return column in {c["name"] for c in insp.get_columns(table)}
        except Exception:
            return False


def upgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name

    # ── routines: schedule shape columns ─────────────────────────────
    if not _column_exists(conn, "routines", "schedule_kind"):
        op.add_column(
            "routines",
            sa.Column(
                "schedule_kind",
                sa.String(length=10),
                nullable=False,
                server_default=sa.text("'cron'"),
            ),
        )
    if not _column_exists(conn, "routines", "schedule_at"):
        op.add_column(
            "routines",
            sa.Column("schedule_at", sa.DateTime(), nullable=True),
        )
    if not _column_exists(conn, "routines", "schedule_interval_seconds"):
        op.add_column(
            "routines",
            sa.Column("schedule_interval_seconds", sa.Integer(), nullable=True),
        )
    if not _column_exists(conn, "routines", "schedule_window_start_local"):
        op.add_column(
            "routines",
            sa.Column("schedule_window_start_local", sa.String(length=8), nullable=True),
        )
    if not _column_exists(conn, "routines", "schedule_window_end_local"):
        op.add_column(
            "routines",
            sa.Column("schedule_window_end_local", sa.String(length=8), nullable=True),
        )
    if not _column_exists(conn, "routines", "auto_disable_after_fire"):
        op.add_column(
            "routines",
            sa.Column(
                "auto_disable_after_fire",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )
    if not _column_exists(conn, "routines", "reminder_text"):
        op.add_column(
            "routines",
            sa.Column("reminder_text", sa.Text(), nullable=True),
        )

    # ── routines: schedule shape CHECK constraint (Postgres only) ────
    # NOTE: existing rows all have schedule_kind='cron' (server-default)
    # and a non-null schedule_cron_local, so the constraint is satisfied
    # on legacy data at upgrade time.
    if dialect == "postgresql":
        conn.execute(sa.text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'ck_routines_schedule_shape'
                ) THEN
                    ALTER TABLE routines ADD CONSTRAINT ck_routines_schedule_shape
                    CHECK (
                        (schedule_kind = 'cron'  AND schedule_cron_local IS NOT NULL)
                     OR (schedule_kind = 'at'    AND schedule_at IS NOT NULL)
                     OR (schedule_kind = 'every' AND schedule_interval_seconds IS NOT NULL
                                                  AND schedule_interval_seconds >= 60)
                    );
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'ck_routines_reminder_text_required'
                ) THEN
                    ALTER TABLE routines ADD CONSTRAINT ck_routines_reminder_text_required
                    CHECK (kind <> 'reminder' OR reminder_text IS NOT NULL);
                END IF;
            END $$;
        """))

    # ── cron_jobs: link column for Phase B backfill ──────────────────
    # On agent containers (where cron_jobs table lives). On the platform
    # container the table doesn't exist; the inspector check handles
    # both dialects.
    try:
        insp = sa.inspect(conn)
        cron_table_exists = "cron_jobs" in set(insp.get_table_names())
    except Exception:
        cron_table_exists = False
    if cron_table_exists and not _column_exists(conn, "cron_jobs", "migrated_to_routine_id"):
        op.add_column(
            "cron_jobs",
            sa.Column("migrated_to_routine_id", sa.String(length=36), nullable=True),
        )
        # Index so CronService.start()'s "WHERE migrated_to_routine_id IS
        # NULL" filter doesn't scan the whole table at boot.
        try:
            op.create_index(
                "ix_cron_jobs_migrated_to_routine_id",
                "cron_jobs",
                ["migrated_to_routine_id"],
            )
        except Exception:
            pass


def downgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect == "postgresql":
        conn.execute(sa.text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'ck_routines_reminder_text_required'
                ) THEN
                    ALTER TABLE routines DROP CONSTRAINT ck_routines_reminder_text_required;
                END IF;
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'ck_routines_schedule_shape'
                ) THEN
                    ALTER TABLE routines DROP CONSTRAINT ck_routines_schedule_shape;
                END IF;
            END $$;
        """))

    try:
        insp = sa.inspect(conn)
        cron_table_exists = "cron_jobs" in set(insp.get_table_names())
    except Exception:
        cron_table_exists = False
    if cron_table_exists and _column_exists(conn, "cron_jobs", "migrated_to_routine_id"):
        try:
            op.drop_index("ix_cron_jobs_migrated_to_routine_id", table_name="cron_jobs")
        except Exception:
            pass
        op.drop_column("cron_jobs", "migrated_to_routine_id")

    for col in (
        "reminder_text",
        "auto_disable_after_fire",
        "schedule_window_end_local",
        "schedule_window_start_local",
        "schedule_interval_seconds",
        "schedule_at",
        "schedule_kind",
    ):
        if _column_exists(conn, "routines", col):
            op.drop_column("routines", col)
