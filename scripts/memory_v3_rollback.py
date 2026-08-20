"""Undo the round-8 → v3 memory migration, per tenant (operator runner).

The real rollback, not a paragraph. Read
`docs/memory/migration-v3-runbook.md` §Rollback first — in particular the
ORDER, which is the one thing that cannot be recovered from if it is wrong:

    1. run THIS script, while the v3 image is still live
    2. then redeploy the previous image tag

The route this script calls (`POST /api/memories/admin/migrate-v3/rollback`)
ships with v3. Redeploy first and the route is gone, and the only remaining
way to clear a tenant's v3 files is a database restore.

WHAT IT UNDOES, per tenant:
  * every `memory_files` row the migration CREATED is deleted,
  * every file the migration FILLED is restored to the exact body recorded
    in that tenant's own migration report,
  * every `memory_file_changes` row is deleted (the Memory log),
  * the `migration_status` marker is reset, so a later v3 deploy re-runs.

WHAT IT DOES NOT UNDO:
  * the legacy `memories` rows — because the migration never touched them.
    That is the whole reason this is small: round 8's content is exactly
    where it was, and the old product reads it again the moment the old
    image is back.
  * anything the v3 writer wrote AFTER the migration. Those bullets live in
    the same files and go with them. A tenant that has been on v3 for a
    while loses whatever it learned in that window.
  * `you/current-context`, whose body is rewritten by its own updater and
    is not restored from a pre-migration snapshot.
  * the Memory log itself. It is deleted, not rewound.

`--hard` is the escape hatch for a tenant whose report is gone (marker
wiped, or a rollback of a rollback): it drops every v3-sectioned file and
empties the three system files, without a per-file before-body to restore.
Prefer the report-driven path; `--hard` also removes `areas/work`, which is
a ROUND-8 system file that v3 adopts by slug.

Usage:
    python -m scripts.memory_v3_rollback --user <uuid>            # dry-run
    python -m scripts.memory_v3_rollback --user <uuid> --apply
    python -m scripts.memory_v3_rollback --all --apply --continue-on-error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import select

CONFIRM = "ROLLBACK MEMORY V3"
TIMEOUT_S = 120.0


async def _candidates(user_ids: Optional[List[str]]) -> List[Tuple[str, str, str, str]]:
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig, User

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(
                AgentConfig.user_id, User.email,
                AgentConfig.agent_url, AgentConfig.agent_api_key,
            )
            .join(User, User.id == AgentConfig.user_id)
            .where(AgentConfig.deploy_status == "active")
            .order_by(User.email)
        )).all()
    out = []
    for uid, email, url, key in rows:
        uid = str(uid)
        if user_ids and uid not in user_ids:
            continue
        if not url or not key:
            continue
        out.append((uid, email or "?", url.rstrip("/"), key))
    return out


async def _report(client: httpx.AsyncClient, url: str, key: str) -> Dict[str, Any]:
    try:
        r = await client.get(
            f"{url}/api/memories/admin/migrate-v3/report",
            headers={"X-Agent-Key": key}, timeout=TIMEOUT_S,
        )
        return r.json() if r.status_code == 200 else {"__error__": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}


async def _rollback(
    client: httpx.AsyncClient, url: str, key: str, *, hard: bool
) -> Dict[str, Any]:
    try:
        r = await client.post(
            f"{url}/api/memories/admin/migrate-v3/rollback",
            headers={"X-Agent-Key": key},
            params={"confirm": CONFIRM, "hard": str(hard).lower()},
            timeout=TIMEOUT_S,
        )
        if r.status_code != 200:
            return {"__error__": f"HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="roll back (default: report what would be undone)")
    ap.add_argument("--user", action="append", help="tenant user id(s)")
    ap.add_argument("--all", action="store_true", help="every active tenant")
    ap.add_argument("--hard", action="store_true",
                    help="ignore the report; drop every v3 file")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.user and not args.all:
        print("refusing to run with no scope — pass --user <uuid> or --all",
              flush=True)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or os.path.join(
        "artifacts", "memory_v3", f"rollback-{stamp}.json"
    )

    cands = await _candidates(args.user)
    print(f"candidates: {len(cands)} tenant(s) "
          f"({'APPLY' if args.apply else 'dry-run'}"
          f"{', HARD' if args.hard else ''})", flush=True)

    results: List[Dict[str, Any]] = []
    ok = failed = 0
    async with httpx.AsyncClient(verify=True) as client:
        for uid, email, url, key in cands:
            report = await _report(client, url, key)
            entry: Dict[str, Any] = {"user_id": uid, "email": email,
                                     "before": report}
            if "__error__" in report:
                print(f"  {uid[:8]} {email.split('@')[0][:14]:<14} "
                      f"UNREACHABLE {report['__error__']}", flush=True)
                results.append(entry)
                failed += 1
                if not args.continue_on_error:
                    break
                continue

            stored = report.get("report") or {}
            would_delete = stored.get("files_created") or []
            would_restore = [
                s for s in (stored.get("files_filled") or [])
                if s not in set(would_delete)
            ]
            print(
                f"  {uid[:8]} {email.split('@')[0][:14]:<14} "
                f"marker={report.get('status')} "
                f"delete={would_delete or '-'} restore={would_restore or '-'}",
                flush=True,
            )
            if not stored and not args.hard:
                print(f"      no stored report — rerun with --hard to force",
                      flush=True)

            if args.apply:
                result = await _rollback(client, url, key, hard=args.hard)
                entry["result"] = result
                print(f"      -> {json.dumps(result, default=str)[:300]}", flush=True)
                if "__error__" in result:
                    failed += 1
                    if not args.continue_on_error:
                        results.append(entry)
                        break
                else:
                    ok += 1
            results.append(entry)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"run": stamp, "applied": args.apply, "hard": args.hard,
                   "ok": ok, "failed": failed, "results": results},
                  fh, indent=2, default=str)
    print(f"artifact: {out_path}", flush=True)
    if not args.apply:
        print("dry-run — pass --apply to roll back, THEN redeploy the "
              "previous image tag (not the other way round)", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
