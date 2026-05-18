"""triggers: backfill stuck-failed trigger_events with explicit error class

Revision ID: 045
Revises: 044
Create Date: 2026-05-18

Companion data migration to the trigger-runner failure-handler fix
(``backend/app/agent/triggers/runner.py``: post-loop terminalisation in
``_handle_event_with_retry`` + new ``_finalise_exhausted`` helper).

Before the runner fix, if ``_dispatch_one`` crashed on all three retry
attempts the loop fell through without writing the ``TriggerEvent``
row. The row stayed in ``status='running'`` until the 10-minute orphan
sweep flipped it to ``status='failed'`` with
``error_class='agent_restarted'`` and NULL ``error_detail`` — or, in
some racing paths, stayed in ``status='failed'`` with both
``error_class`` and ``error_detail`` NULL. The live "Summarize every
new Gmail" trigger accumulated 11 such rows.

What this migration does
------------------------
Backfills any ``trigger_events`` row where:

  - ``status IN ('running', 'failed')``
  - ``error_class IS NULL``
  - ``received_at < NOW() - INTERVAL '5 minutes'`` (don't touch
    in-flight events the new code is currently working on)

with ``error_class='backfilled_unknown'`` + ``error_detail='Pre-fix
failure; logs not available — see commit fixing
_handle_event_with_retry for context.'``. Rows in ``'running'`` are
also flipped to ``'failed'`` and stamped with ``finished_at = NOW()``
so they don't keep appearing as orphans on subsequent sweeps.

Why a separate ``error_class`` value
------------------------------------
Distinguishing ``'backfilled_unknown'`` from the new
``'all_retries_exhausted'`` (set by the runner's post-loop write going
forward) lets operators tell ``"this row failed before the fix landed
and we have no logs"`` from ``"this row failed after the fix and the
error_detail is the actual repr"``. Don't merge the two — diagnosis
gets harder.

Idempotent — only matches rows with NULL ``error_class``, so re-running
the migration touches no rows.

Reversibility
-------------
``downgrade()`` undoes the backfill by clearing the marker on any row
where ``error_class='backfilled_unknown'``. Rows the migration flipped
from ``running → failed`` are NOT reverted to ``running`` — that would
just re-create the orphans the upgrade cleaned up.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import sqlalchemy as sa
from alembic import op


revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.045")


_BACKFILL_DETAIL = (
    "Pre-fix failure; logs not available — see commit fixing "
    "_handle_event_with_retry for context."
)


def upgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "trigger_events"):
        # Platform DB (vs agent DB) doesn't have this table. No-op.
        logger.info("[alembic.045] trigger_events not present; skipping backfill")
        return

    # 5-minute cutoff in Python so the SQL is dialect-agnostic. Avoids
    # Postgres ``INTERVAL`` literal vs. SQLite ``datetime('now', …)``
    # divergence and keeps the migration runnable in CI against either.
    cutoff = datetime.utcnow() - timedelta(minutes=5)

    # Match rows whose state proves they were stuck by the pre-fix
    # behaviour: NULL error_class AND status in {running, failed} AND
    # received_at older than the orphan threshold. The cutoff avoids
    # racing with the new runner code on rows actively being processed
    # at deploy time — those will land with explicit error_class values
    # ('all_retries_exhausted' or whatever the handler returned) and
    # therefore won't match this WHERE.
    result = conn.execute(
        sa.text(
            "UPDATE trigger_events "
            "SET status = 'failed', "
            "    error_class = 'backfilled_unknown', "
            "    error_detail = :detail, "
            "    finished_at = COALESCE(finished_at, :now) "
            "WHERE status IN ('running', 'failed') "
            "  AND error_class IS NULL "
            "  AND received_at < :cutoff"
        ),
        {"detail": _BACKFILL_DETAIL, "now": datetime.utcnow(), "cutoff": cutoff},
    )
    rc = getattr(result, "rowcount", None)
    logger.info(
        "[alembic.045] backfilled %s stuck trigger_events row(s) with "
        "error_class='backfilled_unknown'",
        rc if rc is not None else "?",
    )


def downgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "trigger_events"):
        return
    # Reverse only the marker — leave the status='failed' transition in
    # place. Flipping rows back to 'running' would re-create the orphan-
    # sweep candidates the upgrade cleaned up. The cost is asymmetric:
    # we lose the audit marker on downgrade, but rows don't get stuck
    # again.
    result = conn.execute(
        sa.text(
            "UPDATE trigger_events "
            "SET error_class = NULL, error_detail = NULL "
            "WHERE error_class = 'backfilled_unknown'"
        ),
    )
    rc = getattr(result, "rowcount", None)
    logger.info(
        "[alembic.045] cleared backfilled_unknown marker on %s row(s)",
        rc if rc is not None else "?",
    )


def _table_exists(conn, table: str) -> bool:
    try:
        return table in set(sa.inspect(conn).get_table_names())
    except Exception:
        return False
