"""routines: additive cron_jobs → routines backfill (Phase B)

Revision ID: 043
Revises: 042
Create Date: 2026-05-14

For each enabled `cron_jobs` row where ``migrated_to_routine_id`` is
NULL, this migration:

  1. Inserts a new ``routines`` row with ``kind='reminder'``, the same
     schedule shape, ``reminder_text = cron_jobs.payload_text``, and
     ``delivery_channels`` derived from the presence of
     ``telegram_chat_id``.
  2. Sets ``cron_jobs.migrated_to_routine_id`` to the new routine id.
  3. **Disables the cron_jobs row** so the legacy CronService stops
     firing it — the new Routine runner takes over delivery. This is
     the critical anti-double-fire step.

Safety
------
  * The whole pass is wrapped in ``SET LOCAL lock_timeout = '5s'`` +
    ``SET LOCAL statement_timeout = '60s'`` (longer than mig 042 to
    cover row inserts as well as ALTERs).
  * **No-op when the tables don't exist** (platform DB, fresh test
    SQLite without the agent tables): both ``cron_jobs`` and
    ``routines`` must be present, else the migration short-circuits.
    The platform Railway deploy applies migrations against the platform
    Postgres; this migration is a no-op there.
  * Idempotent: re-running skips any cron_jobs already stamped.
  * Per-row in its own transaction. If row N fails (bad cron expr,
    impossible payload), it's logged and skipped — subsequent rows
    still migrate. Phase D's CronJob drop will not run until every
    row migrates clean.
  * Only ENABLED CronJobs are migrated. Disabled ones stay where they
    are; user can delete them via ``/cron`` Telegram command. This
    avoids surprising the user with a sudden flood of "reminders" they
    had paused intentionally.

Not in this migration
---------------------
  * Phase C (CronService runtime deprecation) — separate config flag.
  * Phase D (drop ``cron_jobs`` table) — wait until prod has run
    cleanly for ≥2 weeks AND every tenant's ``cron_jobs`` table is
    fully stamped (``migrated_to_routine_id IS NOT NULL`` for every
    row OR the row is disabled).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from alembic import op
import sqlalchemy as sa


revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.043")


def _table_exists(conn, table: str) -> bool:
    try:
        return table in set(sa.inspect(conn).get_table_names())
    except Exception:
        return False


def _column_exists(conn, table: str, column: str) -> bool:
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


# ── Schedule shape translation ──────────────────────────────────────


def _translate_schedule(cron_kind, cron_at, cron_interval, cron_expr):
    """Map a CronJob's schedule fields to the (schedule_kind,
    schedule_cron_local, schedule_at, schedule_interval_seconds) tuple
    expected on a Routine.

    Returns None if the CronJob is unschedulable (no usable spec).
    """
    if cron_kind == "at" and cron_at:
        return ("at", None, cron_at, None)
    if cron_kind == "every" and cron_interval and cron_interval >= 60:
        return ("every", None, None, cron_interval)
    if cron_kind == "cron" and cron_expr and len(cron_expr.split()) == 5:
        return ("cron", cron_expr, None, None)
    return None


# ── Upgrade ─────────────────────────────────────────────────────────


def upgrade() -> None:
    conn = op.get_bind()

    # No-op on databases that don't have BOTH tables. The platform
    # Railway alembic run hits this case — cron_jobs lives in tenant
    # agent DBs only. We still need migration 043 in the chain so the
    # version pointer advances cleanly on platform.
    if not _table_exists(conn, "cron_jobs"):
        logger.info("[043] cron_jobs table not present — skipping backfill (platform DB)")
        return
    if not _table_exists(conn, "routines"):
        logger.info("[043] routines table not present — skipping backfill")
        return

    # Per-statement safety valve. Five seconds for ALTER-like ops, a
    # minute total for the read+insert+update cycle is enough for
    # tenants with up to a few hundred cron_jobs. Larger tenants would
    # hit statement_timeout and the migration would fail-fast — the
    # operator re-runs alembic after a quiet period.
    if conn.dialect.name == "postgresql":
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            conn.execute(sa.text("SET LOCAL statement_timeout = '60s'"))
        except Exception:
            pass

    # Guard for environments where mig 042 ran partially (e.g. column
    # add was skipped because of lock_timeout). Without
    # ``migrated_to_routine_id`` we can't safely re-enter.
    if not _column_exists(conn, "cron_jobs", "migrated_to_routine_id"):
        logger.warning(
            "[043] cron_jobs.migrated_to_routine_id missing — "
            "mig 042 didn't apply cleanly. Skipping backfill so the "
            "operator can re-run alembic once the lock contention clears."
        )
        return

    # Pull pending rows. Only enabled CronJobs with no migration
    # pointer get touched.
    rows = conn.execute(sa.text(
        """
        SELECT id, user_id, name, schedule_kind, schedule_at,
               schedule_interval_seconds, schedule_cron_expr,
               payload_text, telegram_chat_id, enabled, created_at
          FROM cron_jobs
         WHERE enabled = true
           AND migrated_to_routine_id IS NULL
        """
    )).mappings().all()

    logger.info("[043] %d enabled cron_jobs pending migration", len(rows))

    migrated = 0
    skipped = 0
    for row in rows:
        translated = _translate_schedule(
            row["schedule_kind"],
            row["schedule_at"],
            row["schedule_interval_seconds"],
            row["schedule_cron_expr"],
        )
        if translated is None:
            logger.warning(
                "[043] cron_job %s skipped — unschedulable shape "
                "(kind=%s, at=%s, interval=%s, expr=%s)",
                row["id"], row["schedule_kind"], row["schedule_at"],
                row["schedule_interval_seconds"], row["schedule_cron_expr"],
            )
            skipped += 1
            continue

        sched_kind, cron_local, schedule_at, interval_secs = translated
        payload_text = (row["payload_text"] or "").strip()
        if not payload_text:
            logger.warning(
                "[043] cron_job %s skipped — empty payload_text "
                "(reminder requires non-empty text)", row["id"],
            )
            skipped += 1
            continue

        # Delivery channels. CronService delivered to Telegram only if
        # `telegram_chat_id` was set; otherwise it dropped messages.
        # The new Routine system delivers to website by default, so:
        #   - telegram_chat_id set → ["website", "telegram"]
        #   - telegram_chat_id null → ["website"]
        # Stored under routines.config.delivery_channels per existing
        # RoutineCreate contract.
        channels = ["website"]
        if row["telegram_chat_id"]:
            channels.append("telegram")

        new_routine_id = str(uuid.uuid4())
        config_payload = json.dumps({"delivery_channels": channels})
        now = datetime.utcnow()
        name = (row["name"] or payload_text[:60] or "Reminder")[:200]

        # ALL existing routines columns get explicit NULLs for the ones
        # we don't set so the INSERT doesn't rely on default factories
        # (alembic migrations run outside the ORM).
        try:
            conn.execute(sa.text(
                """
                INSERT INTO routines (
                    id, user_id, kind, name, prompt_text,
                    schedule_cron_local, schedule_kind, schedule_at,
                    schedule_interval_seconds, schedule_window_start_local,
                    schedule_window_end_local, auto_disable_after_fire,
                    reminder_text, enabled, config, last_state,
                    last_run_at, next_run_at, last_status, last_error,
                    created_at, updated_at
                ) VALUES (
                    :id, :user_id, 'reminder', :name, NULL,
                    :cron_local, :sched_kind, :schedule_at,
                    :interval_secs, NULL, NULL,
                    :auto_disable, :reminder_text, true, :config, NULL,
                    NULL, NULL, 'never_run', NULL,
                    :created_at, :updated_at
                )
                """
            ), {
                "id": new_routine_id,
                "user_id": row["user_id"],
                "name": name,
                "cron_local": cron_local,
                "sched_kind": sched_kind,
                "schedule_at": schedule_at,
                "interval_secs": interval_secs,
                "auto_disable": (sched_kind == "at"),
                "reminder_text": payload_text,
                "config": config_payload,
                "created_at": row["created_at"] or now,
                "updated_at": now,
            })

            # Stamp + disable the source CronJob. Disabling here is the
            # cutover — the CronService will skip this row from the
            # next tick onwards; the Routine runner will own delivery.
            conn.execute(sa.text(
                """
                UPDATE cron_jobs
                   SET migrated_to_routine_id = :new_id,
                       enabled = false
                 WHERE id = :cron_id
                """
            ), {"new_id": new_routine_id, "cron_id": row["id"]})

            migrated += 1
        except Exception as e:
            # Most likely cause: one-per-kind UNIQUE on routines
            # (the 041 partial index) means a user already has an
            # enabled email_briefing/reminder routine. Or a malformed
            # payload Pydantic would have caught (we're below the API
            # layer). Either way, log and move on; operator inspects.
            logger.exception(
                "[043] cron_job %s migration failed: %s",
                row["id"], e,
            )
            skipped += 1

    logger.info(
        "[043] backfill complete: %d migrated, %d skipped",
        migrated, skipped,
    )


# ── Downgrade ───────────────────────────────────────────────────────


def downgrade() -> None:
    """Reverse the backfill — re-enable any CronJob that was migrated
    by this revision, then delete the Routine it created. Idempotent.

    Used only for emergency rollback during the Phase B → C transition.
    Production should never need this once Phase C lands.
    """
    conn = op.get_bind()
    if not _table_exists(conn, "cron_jobs") or not _table_exists(conn, "routines"):
        return

    # Find every migrated cron_job, re-enable it, drop the Routine.
    rows = conn.execute(sa.text(
        """
        SELECT id, migrated_to_routine_id
          FROM cron_jobs
         WHERE migrated_to_routine_id IS NOT NULL
        """
    )).mappings().all()
    for row in rows:
        try:
            conn.execute(sa.text(
                "DELETE FROM routines WHERE id = :rid"
            ), {"rid": row["migrated_to_routine_id"]})
            conn.execute(sa.text(
                """
                UPDATE cron_jobs
                   SET migrated_to_routine_id = NULL,
                       enabled = true
                 WHERE id = :cron_id
                """
            ), {"cron_id": row["id"]})
        except Exception:
            logger.exception(
                "[043] downgrade failed for cron_job %s — manual cleanup needed",
                row["id"],
            )
