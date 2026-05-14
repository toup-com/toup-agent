"""routines: opt-in DROP of legacy cron_jobs table (Phase D)

Revision ID: 044
Revises: 043
Create Date: 2026-05-14

Final step of the CronJob → Routine consolidation. **Destructive.**

Default behavior
----------------
**No-op.** Running ``alembic upgrade head`` after this lands does NOT
drop the ``cron_jobs`` table. The migration logs an instructive line
and exits cleanly. This preserves the rollback path — every deploy
through this revision is reversible until the operator explicitly
opts in.

Opting in
---------
Set the environment variable ``ALLOW_CRONJOB_TABLE_DROP=true`` on the
agent container ONCE, then run ``alembic upgrade head``. The
migration verifies:

  1. Every ``cron_jobs`` row has ``migrated_to_routine_id`` set OR
     ``enabled = false``. Any enabled+unmigrated rows abort the drop
     with a clear error — those reminders would silently stop firing.
  2. ``cron_service_enabled`` is False on the running app (operator
     has flipped Phase C already). Checked at migration runtime by
     looking at the env var, NOT the Python ``settings`` object, to
     keep migrations free of import-time app dependencies.

If both conditions hold the table is dropped (Postgres ``DROP TABLE
IF EXISTS … CASCADE``, lock-safe via ``SET LOCAL lock_timeout``).

After this migration succeeds
-----------------------------
A follow-up commit (Phase E, not in this push) removes the Python
``CronJob`` ORM model, ``cron_service.py``, the Telegram ``/cron``
handler, and any stale imports. Phase E is split off because the
Python deletion is irreversible-via-git-revert in a way the table
drop isn't (you can recreate a table from a backup but a force-pushed
deletion of code-while-services-depend-on-it is harder to recover
cleanly). Keeping the code intact while the table is gone means: the
ORM model still loads (it just maps to a non-existent table), no
imports break, no consumer crashes — they just stop being able to
read/write. Operator runs Phase E only after this revision has been
live for at least one tenant-week.
"""

from __future__ import annotations

import logging
import os

from alembic import op
import sqlalchemy as sa


revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.044")


_OPT_IN_ENV = "ALLOW_CRONJOB_TABLE_DROP"


def _table_exists(conn, table: str) -> bool:
    try:
        return table in set(sa.inspect(conn).get_table_names())
    except Exception:
        return False


def _opt_in_set() -> bool:
    raw = os.environ.get(_OPT_IN_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "cron_jobs"):
        logger.info(
            "[044] cron_jobs not present (platform DB or already-dropped tenant) — no-op."
        )
        return

    if not _opt_in_set():
        logger.info(
            "[044] cron_jobs DROP gated behind %s=true. Skipping. "
            "Set the env var on the agent container and re-run "
            "`alembic upgrade head` to drop the table once every active "
            "reminder has been migrated to a Routine.",
            _OPT_IN_ENV,
        )
        return

    # Safety check — refuse to drop while there are enabled rows the
    # CronService would otherwise still be firing.
    pending = conn.execute(sa.text(
        """
        SELECT COUNT(*) FROM cron_jobs
         WHERE enabled = true
           AND migrated_to_routine_id IS NULL
        """
    )).scalar() or 0
    if pending > 0:
        raise RuntimeError(
            f"[044] refusing to drop cron_jobs — {pending} enabled row(s) "
            f"have NULL migrated_to_routine_id. Run mig 043 first OR "
            f"manually disable them, then retry."
        )

    # Lock-safe DROP. Same pattern as mig 042 — fail fast under
    # contention rather than block alembic forever.
    if conn.dialect.name == "postgresql":
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '10s'"))
            conn.execute(sa.text("SET LOCAL statement_timeout = '30s'"))
        except Exception:
            pass

    logger.warning(
        "[044] DROPPING cron_jobs table (Phase D). This is destructive. "
        "All %d rows are either disabled or migrated.",
        conn.execute(sa.text("SELECT COUNT(*) FROM cron_jobs")).scalar() or 0,
    )
    op.execute("DROP TABLE IF EXISTS cron_jobs CASCADE")
    logger.warning("[044] cron_jobs dropped.")


def downgrade() -> None:
    """Recreate an EMPTY cron_jobs table.

    Cannot restore the row contents — those are gone. The table shape
    matches the ORM model at the time of the drop so any code still
    importing the model continues to load. If you actually need the
    data back, restore from your DB backup *before* downgrading the
    schema, then re-import the rows manually.
    """
    conn = op.get_bind()
    if _table_exists(conn, "cron_jobs"):
        return

    op.create_table(
        "cron_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), index=True),
        sa.Column("name", sa.String(200)),
        sa.Column("schedule_kind", sa.String(20)),
        sa.Column("schedule_spec", sa.String(200)),
        sa.Column("schedule_at", sa.DateTime, nullable=True),
        sa.Column("schedule_interval_seconds", sa.Integer, nullable=True),
        sa.Column("schedule_cron_expr", sa.String(100), nullable=True),
        sa.Column("payload_text", sa.Text),
        sa.Column("telegram_chat_id", sa.BigInteger, nullable=True),
        sa.Column("enabled", sa.Boolean, default=True),
        sa.Column("last_run_at", sa.DateTime, nullable=True),
        sa.Column("run_count", sa.Integer, default=0),
        sa.Column("migrated_to_routine_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime),
    )
