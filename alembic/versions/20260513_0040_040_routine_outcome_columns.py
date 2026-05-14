"""routine_runs outcome state machine + fire_instant + per-channel JSON

Revision ID: 040
Revises: 039
Create Date: 2026-05-13

Phase 2 / Ticket 2.1 — replaces the binary success/failure model on
routine_runs with a richer outcome state machine and adds structured
per-channel + per-tool data alongside the legacy `status` column.

New columns on `routine_runs`:
  - `fire_instant`         — APScheduler's scheduled trigger time (UTC).
                             Distinct from `started_at` which is
                             datetime.utcnow() at _fire entry. Empty for
                             pre-existing rows.
  - `finished_local_at`    — finished_at in the user's local tz, ISO8601.
                             Helps dashboards display "completed at
                             10:58:03 EDT" without a tz join.
  - `outcome`              — new state machine: success | success_empty |
                             partial | tool_error | failure. The legacy
                             `status` column is kept for backward
                             compatibility with dashboards / queries.
  - `channel_results_json` — per-channel delivery results
                             {"website": {...}, "telegram": {...}, ...}.
                             Set by the handler / dispatcher; consumed
                             by /api/routines/{id}/runs.
  - `tools_invoked_json`   — list of MCP tool names the handler called,
                             for forensics ("did it call gmail__get_message?").
  - `error_json`           — structured error info (replacing the loose
                             `error_class` + `error_detail` pair). Stays
                             null on success outcomes.

Additive change. Existing rows have nullable defaults; old code that
reads only `status`/`error_class`/`error_detail` continues to work.

Idempotent — guards against half-run prior migrations same way the 039
codex_token migration does.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def _column_exists(conn, table: str, column: str) -> bool:
    """Robust column existence check that works on Postgres + SQLite.

    Postgres exposes `information_schema.columns`. SQLite doesn't, so
    fall back to the SQLAlchemy inspector which works on both. Used both
    by the migration's idempotency guards AND by ops who replay this
    migration after a half-success.
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
        # SQLite or any dialect without information_schema — fall back to
        # the inspector. Cheap and correct on both Postgres and SQLite.
        try:
            inspector = sa.inspect(conn)
            cols = {c["name"] for c in inspector.get_columns(table)}
            return column in cols
        except Exception:
            return False


def _json_type() -> sa.types.TypeEngine:
    """Postgres prefers JSONB (richer GIN-indexable, faster reads on
    nested keys). Other dialects fall back to plain JSON.

    SQLAlchemy emits `JSON` on SQLite (stored as TEXT) and `JSONB` on
    Postgres when we use `.with_variant`. The model already uses this
    pattern; the migration matches it so existing rows + new rows have
    the same storage shape end-to-end.
    """
    return sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    conn = op.get_bind()

    if not _column_exists(conn, "routine_runs", "fire_instant"):
        op.add_column(
            "routine_runs",
            sa.Column("fire_instant", sa.DateTime(), nullable=True),
        )
    if not _column_exists(conn, "routine_runs", "finished_local_at"):
        op.add_column(
            "routine_runs",
            sa.Column("finished_local_at", sa.String(length=40), nullable=True),
        )
    if not _column_exists(conn, "routine_runs", "outcome"):
        op.add_column(
            "routine_runs",
            sa.Column("outcome", sa.String(length=30), nullable=True),
        )
    if not _column_exists(conn, "routine_runs", "channel_results_json"):
        # JSONB on Postgres, JSON on SQLite — matches the SQLAlchemy model
        # so reads come back as dicts on both ends without a marshalling
        # layer.
        op.add_column(
            "routine_runs",
            sa.Column("channel_results_json", _json_type(), nullable=True),
        )
    if not _column_exists(conn, "routine_runs", "tools_invoked_json"):
        op.add_column(
            "routine_runs",
            sa.Column("tools_invoked_json", _json_type(), nullable=True),
        )
    if not _column_exists(conn, "routine_runs", "error_json"):
        op.add_column(
            "routine_runs",
            sa.Column("error_json", _json_type(), nullable=True),
        )

    # An index on outcome accelerates the dashboard's "show me all
    # tool_error runs in the last 24h" query without forcing a full scan
    # of routine_runs.
    try:
        op.create_index(
            "ix_routine_runs_outcome_started",
            "routine_runs",
            ["outcome", "started_at"],
        )
    except Exception:
        # Idempotent — index may already exist from a half-run prior.
        pass

    # Ticket 2.2: dedupe table for reconnect / failure notices. The
    # UNIQUE constraint on (routine_id, kind, scope_date) is the
    # rate-limit gate — one notice of each kind per routine per local
    # day. Cron / dashboards don't read this directly; it's a write-once
    # store consumed exclusively by `_write_nudge`.
    inspector = sa.inspect(conn)
    if "routine_notification_dedupe" not in set(inspector.get_table_names()):
        op.create_table(
            "routine_notification_dedupe",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "routine_id",
                sa.String(length=36),
                sa.ForeignKey("routines.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("kind", sa.String(length=40), nullable=False),
            sa.Column("scope_date", sa.Date(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.UniqueConstraint(
                "routine_id", "kind", "scope_date",
                name="uq_routine_notification_dedupe_key",
            ),
        )
        op.create_index(
            "ix_routine_notification_dedupe_routine",
            "routine_notification_dedupe",
            ["routine_id"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "routine_notification_dedupe" in set(inspector.get_table_names()):
        try:
            op.drop_index(
                "ix_routine_notification_dedupe_routine",
                table_name="routine_notification_dedupe",
            )
        except Exception:
            pass
        op.drop_table("routine_notification_dedupe")
    try:
        op.drop_index("ix_routine_runs_outcome_started", table_name="routine_runs")
    except Exception:
        pass
    for col in (
        "error_json",
        "tools_invoked_json",
        "channel_results_json",
        "outcome",
        "finished_local_at",
        "fire_instant",
    ):
        if _column_exists(conn, "routine_runs", col):
            op.drop_column("routine_runs", col)
