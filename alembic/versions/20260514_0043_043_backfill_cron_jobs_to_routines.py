"""Phase B — backfill cron_jobs → routines (data migration)

Revision ID: 043
Revises: 042
Create Date: 2026-05-14

Migrates every enabled `cron_jobs` row into a sibling `routines` row
so the runtime can switch off `CronService` in Phase C without losing
user-scheduled work.

Mapping (one cron_job → one routine):
  - kind                 → 'agent_task' (the routine pipeline that runs
                           the prompt through the agent's full pipeline)
  - prompt_text          ← cron_jobs.payload_text
  - name                 ← cron_jobs.name
  - enabled              ← cron_jobs.enabled
  - schedule_kind        ← cron_jobs.schedule_kind  (matches 1:1: at/every/cron)
  - schedule_cron_local  ← cron_jobs.schedule_cron_expr  (when kind='cron')
  - schedule_at          ← cron_jobs.schedule_at        (when kind='at')
  - schedule_interval_seconds ← cron_jobs.schedule_interval_seconds
                                                        (when kind='every')
  - auto_disable_after_fire   ← True for 'at' kinds (one-shot semantic)
  - delivery_channels   → ["telegram"] in config_json.delivery_channels
                          since CronJob was Telegram-only by design.
                          Adding 'website' so Day-as-Chat ALSO gets a
                          copy — the modern unification.
  - config_json.telegram_chat_id_override
                         ← cron_jobs.telegram_chat_id (only used as a
                           fallback when the user's primary mapping
                           isn't usable)
  - config_json.migrated_from_cron_job_id ← source row id (audit trail)

Idempotency:
  - We mark each source row with `cron_jobs.migrated_to_routine_id`.
  - On replay, rows that already have a non-NULL migrated_to_routine_id
    are skipped. So an aborted-half-way migration resumes cleanly.

Reversibility:
  - downgrade() clears `migrated_to_routine_id` on the cron_jobs side
    AND deletes the routines that have
    `config_json.migrated_from_cron_job_id`. This makes the migration
    fully reversible until Phase D drops the `cron_jobs` table.

Tested-against (manual) shapes the existing CronService produces:
  - schedule_kind='at',    schedule_at=<dt>
  - schedule_kind='every', schedule_interval_seconds=N
  - schedule_kind='cron',  schedule_cron_expr=<5-part>

We DO NOT register the new routines with APScheduler here — that
happens when the runner boots / `_reload_all` is called. Phase C's
deploy will trigger reload_all post-startup.
"""

from __future__ import annotations

import json
import uuid

from alembic import op
import sqlalchemy as sa


revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


def _cron_table_exists(conn) -> bool:
    """The `cron_jobs` table lives only in agent DBs (per
    AGENT_ONLY_TABLES). Platform-side replays of this migration must
    be a no-op."""
    try:
        insp = sa.inspect(conn)
        return "cron_jobs" in set(insp.get_table_names())
    except Exception:
        return False


def _routines_table_exists(conn) -> bool:
    try:
        insp = sa.inspect(conn)
        return "routines" in set(insp.get_table_names())
    except Exception:
        return False


def _column_exists(conn, table: str, column: str) -> bool:
    """Same helper as 040/041 — cross-dialect column-existence check."""
    try:
        insp = sa.inspect(conn)
        return column in {c["name"] for c in insp.get_columns(table)}
    except Exception:
        return False


def upgrade() -> None:
    conn = op.get_bind()

    if not _cron_table_exists(conn) or not _routines_table_exists(conn):
        # Platform DB (no cron_jobs) or test DB that skipped routines
        # creation — nothing to do here.
        return

    # Required Phase A column. If 042 didn't land yet, this migration
    # MUST not run — fail loudly so the operator notices.
    if not _column_exists(conn, "cron_jobs", "migrated_to_routine_id"):
        raise RuntimeError(
            "Migration 043 requires `cron_jobs.migrated_to_routine_id` "
            "from migration 042. Run upgrade in order."
        )
    if not _column_exists(conn, "routines", "schedule_kind"):
        raise RuntimeError(
            "Migration 043 requires the schedule_* columns from "
            "migration 042. Run upgrade in order."
        )

    # Pull every un-migrated, enabled cron_jobs row. Disabled rows are
    # NOT migrated — if the user explicitly disabled them, they don't
    # want them firing on the new path either.
    rows = conn.execute(sa.text("""
        SELECT
            id, user_id, name, schedule_kind, schedule_spec,
            schedule_at, schedule_interval_seconds, schedule_cron_expr,
            payload_text, telegram_chat_id, enabled,
            last_run_at, run_count, created_at
        FROM cron_jobs
        WHERE migrated_to_routine_id IS NULL
          AND enabled = true
    """)).fetchall()

    if not rows:
        return

    print(f"[043] backfilling {len(rows)} cron_jobs row(s) → routines")

    inserted = 0
    skipped = 0
    for r in rows:
        rid = r[0]
        user_id = r[1]
        name = r[2]
        sk = (r[3] or "cron").strip()
        # schedule_spec = r[4]  # unused; we read the typed columns instead.
        schedule_at = r[5]
        schedule_interval_seconds = r[6]
        schedule_cron_expr = r[7] or "* * * * *"  # placeholder for non-cron shapes
        payload_text = (r[8] or "").strip()
        telegram_chat_id = r[9]
        enabled = bool(r[10])
        last_run_at = r[11]

        if sk not in ("at", "every", "cron"):
            # Defensive — old schedule_kinds we don't recognise.
            # Skip rather than blow up.
            print(f"[043] skipping cron_job id={rid} reason=unknown_schedule_kind={sk!r}")
            skipped += 1
            continue

        # `agent_task` is the right kind for free-form prompts. Reminders
        # carry literal text instead — CronJob doesn't distinguish, so
        # treat every migrated row as agent_task. Power users who want
        # text-only delivery can edit it to kind='reminder' later.
        new_routine_id = str(uuid.uuid4())
        config_json = {
            "delivery_channels": ["website", "telegram"],
            "migrated_from_cron_job_id": rid,
            # Capture the legacy telegram_chat_id so a deployment that
            # somehow loses the TelegramUserMapping row can still
            # deliver via the historical chat id.
            "telegram_chat_id_override": telegram_chat_id,
        }

        # Validate cron shape (5 parts) — silently downgrade to skipped
        # if not parseable; we don't want migration to crash on a
        # malformed legacy row.
        if sk == "cron":
            parts = schedule_cron_expr.strip().split()
            if len(parts) != 5:
                print(f"[043] skipping cron_job id={rid} reason=invalid_cron={schedule_cron_expr!r}")
                skipped += 1
                continue

        try:
            conn.execute(
                sa.text("""
                    INSERT INTO routines (
                        id, user_id, kind, name, prompt_text,
                        schedule_cron_local, schedule_kind, schedule_at,
                        schedule_interval_seconds, auto_disable_after_fire,
                        enabled, config_json, last_status, last_run_at,
                        created_at, updated_at
                    ) VALUES (
                        :id, :user_id, 'agent_task', :name, :prompt_text,
                        :schedule_cron_local, :schedule_kind, :schedule_at,
                        :schedule_interval_seconds, :auto_disable,
                        :enabled, CAST(:cfg AS jsonb), 'never_run', :last_run_at,
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                """),
                {
                    "id": new_routine_id,
                    "user_id": user_id,
                    "name": name,
                    "prompt_text": payload_text,
                    "schedule_cron_local": schedule_cron_expr,
                    "schedule_kind": sk,
                    "schedule_at": schedule_at,
                    "schedule_interval_seconds": schedule_interval_seconds,
                    "auto_disable": (sk == "at"),
                    "enabled": enabled,
                    "cfg": json.dumps(config_json),
                    "last_run_at": last_run_at,
                },
            )
        except Exception as e:
            # Most likely cause is the partial UNIQUE from 041
            # firing — the user already has a non-agent_task routine
            # of the same kind. Since agent_task is exempt from
            # one-per-kind, this shouldn't trigger; if it does, log
            # and move on.
            print(f"[043] insert failed for cron_job id={rid}: {e}")
            skipped += 1
            continue

        # Mark the source row migrated so subsequent replays + the
        # Phase C "skip already-migrated" filter in CronService both
        # behave correctly.
        conn.execute(
            sa.text("UPDATE cron_jobs SET migrated_to_routine_id = :rid WHERE id = :id"),
            {"rid": new_routine_id, "id": rid},
        )
        inserted += 1

    print(f"[043] done: inserted={inserted} skipped={skipped} total={len(rows)}")


def downgrade() -> None:
    """Reverse the backfill — clear the migrated_to_routine_id link AND
    delete the routines we created. Idempotent: if the linkage column
    is gone (042 already downgraded), this is a no-op."""
    conn = op.get_bind()
    if not _cron_table_exists(conn) or not _routines_table_exists(conn):
        return
    if not _column_exists(conn, "cron_jobs", "migrated_to_routine_id"):
        return

    # Find every migrated routine via the back-reference.
    rows = conn.execute(sa.text("""
        SELECT migrated_to_routine_id FROM cron_jobs
        WHERE migrated_to_routine_id IS NOT NULL
    """)).fetchall()
    routine_ids = [r[0] for r in rows if r[0]]

    if routine_ids:
        # Delete in a single statement — CASCADE handles routine_runs.
        conn.execute(
            sa.text("DELETE FROM routines WHERE id = ANY(:ids)"),
            {"ids": routine_ids},
        )

    conn.execute(sa.text(
        "UPDATE cron_jobs SET migrated_to_routine_id = NULL "
        "WHERE migrated_to_routine_id IS NOT NULL"
    ))
