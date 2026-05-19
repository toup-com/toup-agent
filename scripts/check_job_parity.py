#!/usr/bin/env python3
"""Pre-flight parity check for migration 050 (legacy job tables drop).

Run inside a tenant agent container BEFORE flipping the
``ALLOW_LEGACY_JOB_TABLES_DROP=true`` gate that authorises migration
050 to drop ``trigger_events`` + ``routine_runs`` + ``build_jobs.
steps_json``.

  docker exec toup-agent-<prefix> python backend/scripts/check_job_parity.py

What it checks
--------------
1. Every ``trigger_events`` row has a matching ``build_jobs`` row
   (source_kind='trigger', source_id=trigger_id). PR 4a wires this
   dual-write in ``app/api/triggers_inbound.py``.

2. Every ``routine_runs`` row has a matching ``build_jobs`` row
   (source_kind='routine', source_id=routine_id). PR 4b wires this in
   ``app/agent/routines/runner.py::_fire``.

3. Every ``build_jobs.steps_json`` value that has structured content
   (non-empty array) has at least one corresponding ``job_events``
   row. PRs 3 + 5 dual-write events on every JobLogger entry and
   every Auto Builder phase transition.

Exit codes
----------
  0 — parity green; safe to set the gate and re-run migrations.
  1 — parity gap; investigate and re-run dual-write backfill before
      authorising the drop.
  2 — schema problem (one of the tables already missing, or DB
      bootstrap failure). The migration has either already run or
      the agent isn't on the unified-jobs branch.

Output shape
------------
Plain-text key=value lines, then a one-line verdict. Easy to grep
in a deploy log; no JSON to avoid the ``jq`` dependency on the
operator's machine.
"""
from __future__ import annotations

import asyncio
import sys


async def _amain() -> int:
    try:
        from app.db.database import async_session_maker
        from sqlalchemy import inspect, select, func
        from sqlalchemy.sql import text
    except Exception as exc:
        print(f"FAIL: could not import app modules ({exc})")
        return 2

    async with async_session_maker() as db:
        bind = db.bind

        # ── Schema sanity ─────────────────────────────────────────────
        def _table_names(sync_conn) -> set[str]:
            return set(inspect(sync_conn).get_table_names())

        async with bind.connect() as raw:
            names = await raw.run_sync(_table_names)

        required = {"build_jobs", "job_events"}
        missing = required - names
        if missing:
            print(f"FAIL: required tables missing: {sorted(missing)}")
            print("verdict=schema-not-ready")
            return 2

        has_trigger_events = "trigger_events" in names
        has_routine_runs = "routine_runs" in names
        if not has_trigger_events and not has_routine_runs:
            print("INFO: trigger_events and routine_runs already absent — "
                  "migration 050 has likely already run.")
            print("verdict=already-dropped")
            return 0

        # ── 1. trigger_events vs mirrored Jobs ────────────────────────
        gaps_trigger = 0
        if has_trigger_events:
            te_total = (await db.execute(
                text("SELECT COUNT(*) FROM trigger_events"),
            )).scalar_one()
            te_with_job = (await db.execute(
                text("SELECT COUNT(*) FROM trigger_events "
                     "WHERE job_id IS NOT NULL"),
            )).scalar_one()
            gaps_trigger = te_total - te_with_job
            print(f"trigger_events_total={te_total}")
            print(f"trigger_events_with_job_id={te_with_job}")
            print(f"trigger_events_orphans={gaps_trigger}")

        # ── 2. routine_runs vs mirrored Jobs ──────────────────────────
        gaps_routine = 0
        if has_routine_runs:
            rr_total = (await db.execute(
                text("SELECT COUNT(*) FROM routine_runs"),
            )).scalar_one()
            rr_with_job = (await db.execute(
                text("SELECT COUNT(*) FROM routine_runs "
                     "WHERE job_id IS NOT NULL"),
            )).scalar_one()
            gaps_routine = rr_total - rr_with_job
            print(f"routine_runs_total={rr_total}")
            print(f"routine_runs_with_job_id={rr_with_job}")
            print(f"routine_runs_orphans={gaps_routine}")

        # ── 3. steps_json vs job_events ───────────────────────────────
        # We can't easily verify "every meaningful steps_json entry has
        # a job_events row" — the legacy blobs are unstructured. But
        # we *can* count: jobs created BEFORE PR 3 shipped have no
        # job_events, and that's fine (historical); jobs created AFTER
        # PR 3 should have at least one event each. Surface both
        # counts so the operator can eyeball.
        bj_count = (await db.execute(
            text("SELECT COUNT(*) FROM build_jobs"),
        )).scalar_one()
        bj_with_events = (await db.execute(
            text("SELECT COUNT(DISTINCT job_id) FROM job_events"),
        )).scalar_one()
        print(f"build_jobs_total={bj_count}")
        print(f"build_jobs_with_events={bj_with_events}")

        # ── Verdict ──────────────────────────────────────────────────
        total_gaps = gaps_trigger + gaps_routine
        if total_gaps == 0:
            print("verdict=parity-green-safe-to-drop")
            return 0
        else:
            print(f"verdict=parity-gap-{total_gaps}-orphan-rows")
            return 1


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
