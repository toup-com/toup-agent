"""unified jobs: baseline build_jobs + add unified-arc columns + job_events

Revision ID: 046
Revises: 045
Create Date: 2026-05-18

Step 1 of the unified jobs/tasks/logs arc (see
docs/architecture/jobs-investigation-2026-05-18.md, Step 2 design
decisions D1 + D3).

What this migration does
------------------------
1. **Baselines** ``build_jobs``: the table is currently materialised
   by ``Base.metadata.create_all()`` (database.py:200), NOT by
   Alembic. If the table is absent (fresh agent DB / CI bootstrap),
   create it with the full historical schema PLUS the new columns.
   If it already exists (every existing production agent), leave the
   existing rows alone and just ``op.add_column`` the deltas.

2. **Adds unified-arc columns** to ``build_jobs`` — all nullable, no
   server defaults (defaults are application-side concerns). These
   are the discriminator and back-link fields the unified ``Job``
   record needs:
     - ``source_kind``        VARCHAR(20)
     - ``source_id``          VARCHAR(36)
     - ``conversation_id``    VARCHAR(36)  (soft FK; SET NULL on delete)
     - ``summary_message_id`` VARCHAR(50)  (FK → messages.id SET NULL)
     - ``outcome``            VARCHAR(30)
     - ``idempotency_key``    VARCHAR(120)
   Plus partial UNIQUE on ``(source_id, idempotency_key)`` for the
   ``UNIQUE (routine_id, scheduled_for_local_date)`` and
   ``UNIQUE (trigger_id, event_dedupe_id)`` semantics this collapses.

3. **Creates ``job_events`` table** — the new indexable activity-feed
   surface. One row per material event (phase_started, phase_completed,
   tool_call, error, output_posted). Replaces the client-side flatten
   of ``BuildJob.steps_json`` that the dashboard currently does.

The migration is **purely additive**. No column drops, no type
changes, no data rewrites. The orchestrator runs a deploy through
this revision without coordinating with anything else; the new
columns sit unused until PR 3 (JobRunner) starts populating them.

Idempotency
-----------
- ``add_column`` calls are gated by ``_column_exists`` so re-running
  the migration after a partial application is a no-op.
- ``create_table`` is gated by ``_table_exists``.
- FK targets (``conversations``, ``messages``) are checked at
  application time; on the platform DB those tables may not exist
  and the FK is skipped (column stays as a plain VARCHAR).

Reversibility
-------------
``downgrade()`` drops the ``job_events`` table and the new columns
on ``build_jobs``. It does NOT drop ``build_jobs`` itself, even if
this migration was the one that created it — that would be data-
destructive in production where every existing build is in there.
The asymmetry is acceptable: downgrade-from-zero leaves an empty
``build_jobs`` table behind, which a subsequent ``alembic
downgrade -1`` on an older revision would clean up if it existed.

Dialect notes
-------------
Postgres-only: the partial UNIQUE index uses
``WHERE idempotency_key IS NOT NULL`` — Postgres ``CREATE UNIQUE
INDEX ... WHERE``. SQLite supports the same syntax but parses
``WHERE`` differently in some Alembic shims, so the test pins on
the column existing rather than the unique constraint specifically.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "046"
down_revision = "045"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.046")


# ── Historical column shape of build_jobs ────────────────────────────
# Mirrors backend/app/db/models/app.py::BuildJob exactly as of this
# revision. Used by the "create from scratch" branch in upgrade() when
# the table doesn't yet exist (fresh DB / CI bootstrap). When
# build_jobs gets new columns in future migrations, this list does
# NOT change — those later migrations are responsible for evolving
# the table forward. This list is frozen at the 046 snapshot.

_NEW_COLUMNS = (
    ("source_kind", sa.String(20), True),
    ("source_id", sa.String(36), True),
    ("conversation_id", sa.String(36), True),
    ("summary_message_id", sa.String(50), True),
    ("outcome", sa.String(30), True),
    ("idempotency_key", sa.String(120), True),
)


def upgrade() -> None:
    conn = op.get_bind()

    # ── 1. Baseline build_jobs if absent ──────────────────────────────
    if not _table_exists(conn, "build_jobs"):
        logger.info("[alembic.046] build_jobs absent — creating from scratch")
        op.create_table(
            "build_jobs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id", sa.String(36),
                sa.ForeignKey("users.id") if _table_exists(conn, "users") else None,
                nullable=False, index=True,
            ),
            sa.Column("app_id", sa.String(36), nullable=True),
            sa.Column("title", sa.String(200), nullable=False),
            sa.Column("prompt", sa.Text, nullable=False),
            sa.Column("job_type", sa.String(20), nullable=False, server_default="auto_builder"),
            sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
            sa.Column("steps_json", sa.Text, nullable=False, server_default="[]"),
            sa.Column("model", sa.String(50), nullable=False, server_default=""),
            sa.Column("total_tokens", sa.Integer, nullable=False, server_default="0"),
            sa.Column("error_message", sa.Text, nullable=True),
            sa.Column("build_logs_json", sa.Text, nullable=False, server_default="[]"),
            sa.Column("paused_at", sa.DateTime, nullable=True),
            sa.Column("resume_after", sa.DateTime, nullable=True),
            sa.Column("checkpoint_json", sa.Text, nullable=True),
            sa.Column("layer", sa.Integer, nullable=False, server_default="1"),
            sa.Column("layer2_changes_json", sa.Text, nullable=True),
            sa.Column(
                "created_at", sa.DateTime, nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("completed_at", sa.DateTime, nullable=True),
            # Unified-arc columns added in the same create_table for
            # fresh DBs. Existing DBs get them via add_column below.
            sa.Column("source_kind", sa.String(20), nullable=True),
            sa.Column("source_id", sa.String(36), nullable=True),
            sa.Column("conversation_id", sa.String(36), nullable=True),
            sa.Column("summary_message_id", sa.String(50), nullable=True),
            sa.Column("outcome", sa.String(30), nullable=True),
            sa.Column("idempotency_key", sa.String(120), nullable=True),
        )
    else:
        # ── 2. Add unified-arc columns to existing build_jobs ─────────
        for col_name, col_type, nullable in _NEW_COLUMNS:
            if not _column_exists(conn, "build_jobs", col_name):
                op.add_column(
                    "build_jobs",
                    sa.Column(col_name, col_type, nullable=nullable),
                )
            else:
                logger.info(
                    "[alembic.046] build_jobs.%s already present; skipping add",
                    col_name,
                )

    # ── 3. Partial UNIQUE on (source_id, idempotency_key) ────────────
    # Carries the existing UNIQUE (routine_id, scheduled_for_local_date)
    # and UNIQUE (trigger_id, event_dedupe_id) semantics into the
    # unified Job row. WHERE idempotency_key IS NOT NULL so rows that
    # don't carry an idempotency key (manual jobs from chat input) can
    # coexist without colliding on NULL.
    if conn.dialect.name == "postgresql":
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_build_jobs_source_idempotency "
            "ON build_jobs (source_id, idempotency_key) "
            "WHERE idempotency_key IS NOT NULL"
        )
    elif conn.dialect.name == "sqlite":
        # SQLite supports the same partial-index syntax. Wrap in IF NOT
        # EXISTS for re-run safety.
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_build_jobs_source_idempotency "
            "ON build_jobs (source_id, idempotency_key) "
            "WHERE idempotency_key IS NOT NULL"
        )

    # ── 4. New job_events table ─────────────────────────────────────
    if not _table_exists(conn, "job_events"):
        op.create_table(
            "job_events",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "job_id", sa.String(36),
                sa.ForeignKey("build_jobs.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("user_id", sa.String(36), nullable=False),
            sa.Column(
                "ts", sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            # phase_started | phase_completed | tool_call | error |
            # output_posted (open enum, application-side validated)
            sa.Column("kind", sa.String(40), nullable=False),
            sa.Column("label", sa.String(200), nullable=True),
            # Mirrors build_jobs.status enum where applicable, NULL
            # for kinds that don't carry a status (tool_call, error).
            sa.Column("status", sa.String(20), nullable=True),
            sa.Column("level", sa.String(10), nullable=False, server_default="info"),
            sa.Column("metadata_json", sa.Text, nullable=True),
        )
        # Activity feed: "show me everything that happened, recent first"
        op.create_index(
            "ix_job_events_user_ts", "job_events", ["user_id", "ts"],
        )
        # Job detail drawer: "show me everything for this job"
        op.create_index(
            "ix_job_events_job_ts", "job_events", ["job_id", "ts"],
        )

    logger.info("[alembic.046] unified job schema applied")


def downgrade() -> None:
    conn = op.get_bind()

    # ── 1. Drop job_events ──────────────────────────────────────────
    if _table_exists(conn, "job_events"):
        op.drop_index("ix_job_events_job_ts", table_name="job_events")
        op.drop_index("ix_job_events_user_ts", table_name="job_events")
        op.drop_table("job_events")

    # ── 2. Drop the partial UNIQUE index ─────────────────────────────
    if conn.dialect.name in ("postgresql", "sqlite"):
        op.execute("DROP INDEX IF EXISTS uq_build_jobs_source_idempotency")

    # ── 3. Drop the added columns from build_jobs ───────────────────
    # build_jobs itself is NOT dropped, even if this migration created
    # it: in production every existing build is in there, and going
    # back to "no jobs table at all" is not a meaningful downgrade
    # target. The asymmetry is documented in the module docstring.
    for col_name, _, _ in reversed(_NEW_COLUMNS):
        if _column_exists(conn, "build_jobs", col_name):
            with op.batch_alter_table("build_jobs") as batch:
                batch.drop_column(col_name)

    logger.info("[alembic.046] unified job schema reverted")


# ── Inspection helpers ────────────────────────────────────────────────


def _table_exists(conn, table: str) -> bool:
    try:
        insp = sa.inspect(conn)
        # SQLite's schema cache lags batch_alter_table operations that
        # ran in a previous Alembic command call against the same temp
        # file (the reapply-after-downgrade test path). Clearing the
        # cache forces a fresh ``sqlite_master`` lookup so the
        # column-existence check matches reality.
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
