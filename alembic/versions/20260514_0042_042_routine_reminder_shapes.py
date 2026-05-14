"""routines: additive schedule shapes + reminder text (lock-friendly)

Revision ID: 042
Revises: 041
Create Date: 2026-05-14

Re-ship of the reminder feature with a Postgres-online migration pattern.
The first attempt added `NOT NULL DEFAULT` columns which require
`AccessExclusiveLock` and hung during Railway's blue-green deploy when
the old replica's idle SELECTs held `AccessShareLock` longer than
expected. This version is metadata-only:

  - ALL new columns are NULLABLE.
  - NO `server_default`.
  - NO `CHECK` constraints (those would also need exclusive lock; the
    Pydantic model_validator at the API layer enforces shape).
  - `SET LOCAL lock_timeout = '5s'` so if there's any contention, the
    migration FAILS fast instead of hanging — the Dockerfile's `||`
    catches that, uvicorn starts, app still serves because runtime
    code defaults NULL values.

After this lands, follow-up migrations CAN tighten constraints with
`ALTER … SET NOT NULL` once every running pod is on the new code
(expand-then-migrate-then-contract pattern).

Idempotent: each column add is guarded by `_column_exists`, so a
half-run replay completes cleanly.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    """Cross-dialect column existence check.

    Uses Postgres's information_schema when available; falls back to
    SQLAlchemy's inspector on other dialects (SQLite test env).
    """
    try:
        return bool(
            conn.execute(
                sa.text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name=:t AND column_name=:c"
                ),
                {"t": table, "c": column},
            ).scalar()
        )
    except Exception:
        try:
            insp = sa.inspect(conn)
            return column in {c["name"] for c in insp.get_columns(table)}
        except Exception:
            return False


def _table_exists(conn, table: str) -> bool:
    try:
        return table in set(sa.inspect(conn).get_table_names())
    except Exception:
        return False


def upgrade() -> None:
    conn = op.get_bind()

    # ── Critical: lock_timeout safety valve ──────────────────────────
    # During Railway blue-green deploy, the OLD container's connection
    # pool may hold AccessShareLock on `routines` from in-flight or
    # recently-finished SELECT statements. A NULLABLE ADD COLUMN is
    # metadata-only and shouldn't conflict, but if anything else
    # (autovacuum, an open long-running transaction) holds an
    # AccessExclusiveLock or even a stronger lock, our ALTER would
    # wait forever. Five seconds is long enough to clear normal
    # contention, short enough that the operator can re-run alembic
    # manually once the conflict clears.
    if conn.dialect.name == "postgresql":
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            conn.execute(sa.text("SET LOCAL statement_timeout = '30s'"))
        except Exception:
            pass

    # ── routines: new schedule shape columns (all NULLABLE) ──────────
    # Each ADD COLUMN is a metadata-only change in Postgres because
    # there is no default value being applied to existing rows. New
    # rows get their default in application code (Routine.__init__).
    if _table_exists(conn, "routines"):
        new_routine_columns: list[tuple[str, sa.types.TypeEngine]] = [
            ("schedule_kind", sa.String(length=10)),
            ("schedule_at", sa.DateTime()),
            ("schedule_interval_seconds", sa.Integer()),
            ("schedule_window_start_local", sa.String(length=8)),
            ("schedule_window_end_local", sa.String(length=8)),
            ("auto_disable_after_fire", sa.Boolean()),
            ("reminder_text", sa.Text()),
        ]
        for col_name, col_type in new_routine_columns:
            if not _column_exists(conn, "routines", col_name):
                op.add_column(
                    "routines",
                    sa.Column(col_name, col_type, nullable=True),
                )

    # ── cron_jobs.migrated_to_routine_id (deferred Phase B link) ─────
    # Reserved column for a future data-migration pass that backfills
    # cron_jobs → routines. Adding it here so the column is in place
    # without needing another migration later. Agent-DB-only (Platform
    # DB doesn't have cron_jobs); guarded by _table_exists.
    if _table_exists(conn, "cron_jobs"):
        if not _column_exists(conn, "cron_jobs", "migrated_to_routine_id"):
            op.add_column(
                "cron_jobs",
                sa.Column("migrated_to_routine_id", sa.String(length=36), nullable=True),
            )


def downgrade() -> None:
    """Reverse the column additions. Idempotent — checks each column
    exists before dropping. Routine rows with non-NULL values lose
    their schedule shape data; the runtime falls back to cron semantics
    (the only path that existed before this migration)."""
    conn = op.get_bind()

    if _table_exists(conn, "cron_jobs") and _column_exists(
        conn, "cron_jobs", "migrated_to_routine_id"
    ):
        op.drop_column("cron_jobs", "migrated_to_routine_id")

    if _table_exists(conn, "routines"):
        # Drop in reverse declaration order — defensive, not strictly
        # required since columns have no FKs.
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
