#!/usr/bin/env python3
"""CronJob → Routine cutover status — operator-friendly readout.

Run inside a tenant agent container to see whether that tenant is
safe to advance through the Phases C/D cutover. Prints a structured
status block + a one-line verdict.

  docker exec toup-agent-<prefix> python backend/scripts/cronjob_cutover_status.py

Exit codes
----------
  0 — cutover-ready (Phase D safe to run)
  1 — work pending (run mig 043 first, or disable stragglers)
  2 — schema problem (cron_jobs / routines table missing, etc.)
"""
from __future__ import annotations

import asyncio
import os
import sys


async def _amain() -> int:
    try:
        from app.db.database import async_session_maker
        from sqlalchemy import inspect, select, text
    except Exception as e:
        print(f"FAIL: could not import app modules — are you inside the agent container? ({e})")
        return 2

    async with async_session_maker() as db:
        # Inspect the DB schema first — we'd rather print a clear
        # "table missing" message than a SQL error.
        def _has_table(sync_conn, name: str) -> bool:
            return name in set(inspect(sync_conn).get_table_names())

        bind = db.bind
        async with bind.connect() as raw:
            has_cron_jobs = await raw.run_sync(_has_table, "cron_jobs")
            has_routines = await raw.run_sync(_has_table, "routines")

        if not has_routines:
            print("FAIL: routines table missing — run `alembic upgrade head` first.")
            return 2

        # ── cron_jobs side ────────────────────────────────────────
        if has_cron_jobs:
            enabled_unmig = (await db.execute(text(
                """
                SELECT COUNT(*) FROM cron_jobs
                 WHERE enabled = true AND migrated_to_routine_id IS NULL
                """
            ))).scalar() or 0
            enabled_mig = (await db.execute(text(
                """
                SELECT COUNT(*) FROM cron_jobs
                 WHERE enabled = true AND migrated_to_routine_id IS NOT NULL
                """
            ))).scalar() or 0
            disabled = (await db.execute(text(
                "SELECT COUNT(*) FROM cron_jobs WHERE enabled = false"
            ))).scalar() or 0
        else:
            enabled_unmig = enabled_mig = disabled = 0

        # ── routines side ────────────────────────────────────────
        reminder_count = (await db.execute(text(
            "SELECT COUNT(*) FROM routines WHERE kind = 'reminder'"
        ))).scalar() or 0
        reminder_enabled = (await db.execute(text(
            "SELECT COUNT(*) FROM routines WHERE kind = 'reminder' AND enabled = true"
        ))).scalar() or 0

        # ── alembic head ─────────────────────────────────────────
        try:
            alembic_head = (await db.execute(text(
                "SELECT version_num FROM alembic_version"
            ))).scalar() or "(unknown)"
        except Exception:
            alembic_head = "(no alembic_version table)"

        # ── env flags ────────────────────────────────────────────
        cron_flag = os.environ.get("CRON_SERVICE_ENABLED", "(unset = default True)")
        drop_optin = os.environ.get("ALLOW_CRONJOB_TABLE_DROP", "(unset)")

    print("=" * 60)
    print(" CronJob → Routine cutover status")
    print("=" * 60)
    print(f"  alembic head                    : {alembic_head}")
    print(f"  cron_jobs table present         : {'yes' if has_cron_jobs else 'no (Phase D done)'}")
    print(f"  CRON_SERVICE_ENABLED env        : {cron_flag}")
    print(f"  ALLOW_CRONJOB_TABLE_DROP env    : {drop_optin}")
    print()
    print(" cron_jobs row counts")
    print(f"  enabled (unmigrated)            : {enabled_unmig}")
    print(f"  enabled (migrated)              : {enabled_mig}    "
          f"{'← anti-double-fire skip applies' if enabled_mig else ''}")
    print(f"  disabled                        : {disabled}")
    print()
    print(" routines (kind=reminder)")
    print(f"  total                           : {reminder_count}")
    print(f"  enabled                         : {reminder_enabled}")
    print()

    # ── Verdict ──────────────────────────────────────────────────
    if not has_cron_jobs:
        print("VERDICT: Phase D already ran — table dropped. Cutover complete.")
        print("         Next: Phase E (operator-coordinated code removal).")
        return 0
    if enabled_unmig == 0:
        if cron_flag.strip().lower() in {"false", "0", "no", "off"}:
            print("VERDICT: ready for Phase D. Run with ALLOW_CRONJOB_TABLE_DROP=true:")
            print("           ALLOW_CRONJOB_TABLE_DROP=true alembic upgrade head")
        else:
            print("VERDICT: ready for Phase C. Set CRON_SERVICE_ENABLED=false and restart agent.")
            print("         Wait ≥7 days before Phase D so any missed reminder surfaces.")
        return 0
    print("VERDICT: NOT ready — Phase B/Mig 043 hasn't migrated every enabled CronJob.")
    print("         Run `alembic upgrade head` (mig 043 is idempotent) OR disable the")
    print("         remaining rows: UPDATE cron_jobs SET enabled=false WHERE enabled=true")
    print("           AND migrated_to_routine_id IS NULL;")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(_amain()))
    except KeyboardInterrupt:
        sys.exit(130)
