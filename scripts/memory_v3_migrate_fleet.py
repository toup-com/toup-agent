"""Round 8 → v3 memory migration, across the fleet (operator runner).

Drives the tenant-side one-shot (`POST /api/memories/admin/migrate-v3`) over
every active tenant, one at a time, and writes a consolidated JSON artifact
locally. The runbook is `docs/memory/migration-v3-runbook.md`; read it before
`--apply`.

DRY-RUN BY DEFAULT, and the dry run is complete rather than indicative: the
tenant builds the whole report — before counts by category and by round-8
file, the file bodies a real run would leave behind, and every source row id
mapped to a disposition with a reason — while writing nothing at all. Read
one tenant's dry run before you write anything to any of them.

BACKUP. The fleet snapshot is the nightly restic/B2 `pg_dump` of the `toup_%`
databases; this script does not replace it. What it adds is an IN-BAND,
per-tenant record: before every real run it calls the report endpoint, whose
`snapshot` is the tenant's live before-state, and stores it in the artifact
beside the result. That is what makes a single tenant's rollback checkable
without restoring a database.

ROLLBACK is `scripts/memory_v3_rollback.py`, and it must run BEFORE the
previous image tag is redeployed — the route it calls does not exist in that
image.

Environment: run with the PLATFORM service env injected (railway run /
exported env) with DATABASE_URL pointed at the external TCP proxy. Reads
`agent_configs` for each tenant's URL and `X-Agent-Key`; the key value never
leaves this process and is never printed.

Never run this during a rollout window: pushing platform main rebuilds the
agent fleet, and a tenant recreated mid-migration comes back on a marker that
says `in_progress`.

Usage:
    python -m scripts.memory_v3_migrate_fleet                  # dry-run, all
    python -m scripts.memory_v3_migrate_fleet --user <uuid>    # dry-run, one
    python -m scripts.memory_v3_migrate_fleet --apply --user <uuid>   # canary
    python -m scripts.memory_v3_migrate_fleet --apply --continue-on-error
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

#: The tenant runs one LLM call per batch of legacy rows. The corpora are
#: tiny (the 2026-08 audit found rows on 4 of 54 containers) but a cold
#: model call is slow, so this is generous on purpose.
MIGRATE_TIMEOUT_S = 600.0
REPORT_TIMEOUT_S = 60.0


async def _candidates(user_ids: Optional[List[str]]) -> List[Tuple[str, str, str, str]]:
    """Active tenants with a reachable agent URL and a key.

    Returns [(user_id, email, agent_url, agent_api_key)]. Same predicate the
    platform's own memories proxy uses (`deploy_status == "active"`), so a
    tenant this script skips is one no user request would have reached
    either.
    """
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

    out: List[Tuple[str, str, str, str]] = []
    for uid, email, url, key in rows:
        uid = str(uid)
        if user_ids and uid not in user_ids:
            continue
        if not url or not key:
            print(f"  {uid[:8]} SKIP no agent_url/key", flush=True)
            continue
        out.append((uid, email or "?", url.rstrip("/"), key))
    return out


async def _get_report(client: httpx.AsyncClient, url: str, key: str) -> Optional[Dict]:
    try:
        r = await client.get(
            f"{url}/api/memories/admin/migrate-v3/report",
            headers={"X-Agent-Key": key}, timeout=REPORT_TIMEOUT_S,
        )
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}
    if r.status_code != 200:
        return {"__error__": f"HTTP {r.status_code}"}
    try:
        return r.json()
    except Exception:
        return {"__error__": "report was not JSON"}


async def _post_migrate(
    client: httpx.AsyncClient, url: str, key: str, *, dry_run: bool
) -> Dict:
    try:
        r = await client.post(
            f"{url}/api/memories/admin/migrate-v3",
            headers={"X-Agent-Key": key},
            params={"dry_run": str(dry_run).lower()},
            timeout=MIGRATE_TIMEOUT_S,
        )
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}
    if r.status_code != 200:
        return {"__error__": f"HTTP {r.status_code}: {r.text[:200]}"}
    try:
        return r.json()
    except Exception:
        return {"__error__": "response was not JSON"}


def _one_line(uid: str, email: str, report: Dict) -> str:
    """One line per tenant, and it has to say enough to act on.

    Names the counts a reviewer would otherwise open the artifact for: how
    many rows there were, how many landed, how many were dropped, and how
    many the writer failed to account for — the last of which is a DEFECT
    and is why it is on the line at all.
    """
    if "__error__" in report:
        return f"  {uid[:8]} {email.split('@')[0][:14]:<14} UNREACHABLE {report['__error__']}"
    t = report.get("tallies") or {}
    kept = sum(t.get(k, 0) for k in ("kept", "merged", "rewritten"))
    dropped = t.get("dropped", 0)
    skipped = sum(v for k, v in t.items() if k.startswith("skipped_"))
    unaccounted = t.get("unaccounted", 0) + t.get("batch_failed", 0)
    files = ", ".join(report.get("files_filled") or []) or "-"
    flag = "  ⚠ UNACCOUNTED" if unaccounted else ""
    return (
        f"  {uid[:8]} {email.split('@')[0][:14]:<14} "
        f"{report.get('status', '?'):<22} "
        f"rows={sum(t.values())} kept={kept} dropped={dropped} "
        f"skipped={skipped} unaccounted={unaccounted} files=[{files}]{flag}"
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write (default: dry-run, which writes nothing)")
    ap.add_argument("--user", action="append", help="limit to user id(s)")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--out", default=None,
                    help="artifact path (default artifacts/memory_v3/<ts>.json)")
    args = ap.parse_args()

    dry_run = not args.apply
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or os.path.join(
        "artifacts", "memory_v3",
        f"fleet-{'dryrun' if dry_run else 'apply'}-{stamp}.json",
    )

    cands = await _candidates(args.user)
    print(f"candidates: {len(cands)} active tenant(s) "
          f"({'DRY RUN — nothing is written' if dry_run else 'APPLY'})",
          flush=True)
    if not cands:
        return 0

    results: List[Dict[str, Any]] = []
    ok = failed = unreachable = 0

    async with httpx.AsyncClient(verify=True) as client:
        for uid, email, url, key in cands:
            entry: Dict[str, Any] = {"user_id": uid, "email": email, "dry_run": dry_run}

            # The in-band backup: the tenant's live before-state, recorded
            # BEFORE anything is written to it. On a dry run it is also the
            # only "before" there is, because the tenant stores nothing.
            before = await _get_report(client, url, key)
            entry["before"] = before
            if before and "__error__" in before:
                print(_one_line(uid, email, before), flush=True)
                entry["result"] = before
                results.append(entry)
                unreachable += 1
                # Unreachable is REPORTED, not retried: a tenant that is
                # down stays down for the length of this pass, and a retry
                # loop here turns one bad container into a stalled fleet.
                if args.continue_on_error:
                    continue
                print("stopping (no --continue-on-error)", flush=True)
                break
            if (
                not dry_run
                and (before or {}).get("status") == "completed"
            ):
                print(f"  {uid[:8]} {email.split('@')[0][:14]:<14} "
                      f"already completed — skipping", flush=True)
                entry["result"] = {"status": "already_completed"}
                results.append(entry)
                ok += 1
                continue

            result = await _post_migrate(client, url, key, dry_run=dry_run)
            entry["result"] = result
            if not dry_run:
                entry["after"] = await _get_report(client, url, key)
            print(_one_line(uid, email, result), flush=True)
            results.append(entry)

            if "__error__" in result:
                failed += 1
                if not args.continue_on_error:
                    print("stopping on first failure (no --continue-on-error)",
                          flush=True)
                    break
            else:
                ok += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "run": stamp, "dry_run": dry_run,
            "tenants": len(cands), "ok": ok, "failed": failed,
            "unreachable": unreachable, "results": results,
        }, fh, indent=2, default=str)

    print(f"done: ok {ok}, failed {failed}, unreachable {unreachable}, "
          f"of {len(cands)}", flush=True)
    print(f"artifact: {out_path}", flush=True)
    if dry_run:
        print("this was a DRY RUN — pass --apply to write", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
