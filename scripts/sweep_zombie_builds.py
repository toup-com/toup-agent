#!/usr/bin/env python3
"""Fleet sweep: every app-build card still reading "In progress" after its
build stopped.

Round 27 (2026-08-23). A "Build: Habit Garden · In progress · 4/7 steps ·
57%" card sat in a founder's chat for hours: the build died inside the
looks-right loop, the delivered-turn reconciler's SELECT filtered the build
lane out (``job_type == 'agent_task'``), and the 30-minute stall reaper
wrote a status the clients cannot read — both build cards are rendered off
their STEPS, and the reaper never touches ``steps_json``.

``app/agent/build_watchdog.py`` fixes that going forward, and it fixes it
FLEET-WIDE by itself: the watchdog loop runs in every agent container
against its own tenant, so within a minute of the image rolling out every
stuck build anywhere in the fleet has been settled. **This script is not a
second implementation of that.** It is the operator's view of the same
sweep — what it would do, what it did, and what is left — for two jobs the
loop cannot do:

* answering "is the fleet clean?" with a number rather than a hope, and
* forcing the sweep on a tenant during an incident, at a tighter window than
  the loop's 15 minutes.

DRY RUN BY DEFAULT, and the dry run is complete: every tenant reports the
count it would settle, and nothing is written anywhere. Read the summary
before ``--apply``.

Environment: run with the PLATFORM service env injected (``railway run`` or
an exported env) with ``DATABASE_URL`` pointed at the external TCP proxy —
``agent_configs`` is a PLATFORM table and ``build_jobs`` is a TENANT one, so
this reads the first to reach the second. The ``X-Agent-Key`` value never
leaves this process and is never printed.

Usage::

    python -m scripts.sweep_zombie_builds                      # dry-run, all
    python -m scripts.sweep_zombie_builds --user <uuid>        # dry-run, one
    python -m scripts.sweep_zombie_builds --apply --user <uuid>   # canary
    python -m scripts.sweep_zombie_builds --apply                  # fleet
    python -m scripts.sweep_zombie_builds --apply --stale-minutes 5

Exit codes: 0 clean (or a dry run that found nothing), 1 rows still active
after an ``--apply`` — which is the "impossible by construction" case and
means a close path is not calling ``settle_build``, 2 no tenant was
reachable at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import select

SWEEP_TIMEOUT_S = 120.0


async def _candidates(user_ids: Optional[List[str]]) -> List[Tuple[str, str, str, str]]:
    """Active tenants with a reachable agent URL and a key.

    Returns ``[(user_id, email, agent_url, agent_api_key)]``. Same predicate
    the platform's own proxies use (``deploy_status == 'active'``), so a
    tenant this script skips is one no user request would have reached.
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


async def _sweep_one(
    client: httpx.AsyncClient, url: str, key: str, *,
    dry_run: bool, stale_minutes: Optional[int],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"dry_run": str(dry_run).lower()}
    if stale_minutes:
        params["stale_minutes"] = stale_minutes
    try:
        r = await client.post(
            f"{url}/api/apps/jobs/sweep-stuck-builds",
            headers={"X-Agent-Key": key}, params=params,
            timeout=SWEEP_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        return {"__error__": f"{type(e).__name__}: {e}"}
    if r.status_code == 404:
        # The tenant is on an image from before this round. Named rather
        # than counted as clean: "0 stuck builds" and "cannot answer" are
        # different facts and only one of them is good news.
        return {"__error__": "route absent — tenant predates round 27"}
    if r.status_code != 200:
        return {"__error__": f"HTTP {r.status_code}: {r.text[:200]}"}
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {"__error__": "response was not JSON"}


def _one_line(uid: str, email: str, res: Dict[str, Any]) -> str:
    who = email.split("@")[0][:14]
    if "__error__" in res:
        return f"  {uid[:8]} {who:<14} UNREACHABLE  {res['__error__']}"
    settled = res.get("settled", 0)
    left = res.get("still_active") or []
    verb = "would settle" if res.get("dry_run") else "settled"
    flag = f"  ⚠ {len(left)} STILL ACTIVE" if left else ""
    return f"  {uid[:8]} {who:<14} {verb} {settled}{flag}"


async def _amain(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually settle (default is a complete dry run)")
    ap.add_argument("--user", action="append", dest="users", default=None,
                    help="restrict to one user id (repeatable)")
    ap.add_argument("--stale-minutes", type=int, default=None,
                    help="override the 15-minute silence window")
    ap.add_argument("--artifact", default=None,
                    help="write the full per-tenant JSON here")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)

    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is not set — run this with the platform env "
              "injected (see the module docstring).")
        return 2

    tenants = await _candidates(args.users)
    if not tenants:
        print("No active tenants matched.")
        return 2

    mode = "APPLY" if args.apply else "DRY RUN"
    window = args.stale_minutes or 15
    print(f"{mode} — {len(tenants)} tenant(s), silence window {window}m\n")

    results: Dict[str, Dict[str, Any]] = {}
    sem = asyncio.Semaphore(max(1, args.concurrency))
    async with httpx.AsyncClient() as client:
        async def _run(uid: str, email: str, url: str, key: str) -> None:
            async with sem:
                res = await _sweep_one(
                    client, url, key, dry_run=not args.apply,
                    stale_minutes=args.stale_minutes,
                )
            results[uid] = {"email": email, **res}
            print(_one_line(uid, email, res), flush=True)

        await asyncio.gather(*(
            _run(uid, email, url, key) for uid, email, url, key in tenants
        ))

    total = sum(int(r.get("settled") or 0) for r in results.values()
                if "__error__" not in r)
    unreachable = [u for u, r in results.items() if "__error__" in r]
    still_active = {
        u: r.get("still_active") for u, r in results.items()
        if r.get("still_active")
    }

    print()
    print(f"{'would settle' if not args.apply else 'settled'}: {total}")
    if unreachable:
        print(f"unreachable: {len(unreachable)} tenant(s)")
    if still_active:
        print(f"STILL ACTIVE after the sweep: {still_active}")
        print("A build card older than its watchdog window is impossible by "
              "construction — a close path is not calling settle_build.")

    if args.artifact:
        with open(args.artifact, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print(f"artifact: {args.artifact}")

    if len(unreachable) == len(results):
        return 2
    return 1 if still_active else 0


def main() -> None:
    sys.exit(asyncio.run(_amain(sys.argv[1:])))


if __name__ == "__main__":
    main()
