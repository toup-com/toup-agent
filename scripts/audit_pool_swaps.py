#!/usr/bin/env python3
"""Find accounts that may have been silently moved off their pool database.

WHY THIS EXISTS
---------------
A warm-pool tenant's Postgres database is named from the SLOT
(`bridge/pool_addon.py::_pool_db_name` -> ``toup_agent_feed0017``). A named
tenant's is named from the USER (``docker_host_service.provision_container`` ->
``toup_agent_<user_prefix>``). Nothing migrates one to the other.

Until R40, four callers could re-provision a pool member through the NAMED
path — `update_container_env` on any transient bridge error, the 180 s
container reconciler, `restart_container`'s error bookkeeping feeding the
reclaimer, and agent key rotation. Each of those silently repointed the user's
agent at a brand-new, EMPTY database. Their chat history, automations and
memories were never deleted; they are still on the VPS, in a database the
agent no longer opens.

R40 refuses the swap. This script finds the accounts it may already have
happened to, so their data can be moved back.

WHAT IT DOES
------------
Read-only. For every managed container it:

  1. flags the row as NAMED or POOL,
  2. asks the tenant agent how much history it holds
     (``GET {agent_url}/api/day-chats?limit=1`` and ``/api/sessions?limit=1``,
     with the user's own ``X-Agent-Key``),
  3. reports every NAMED tenant whose account is OLDER than a day and whose
     agent reports NO history at all.

That last set is the candidate list. On its own it is a heuristic, not a
verdict: a user who signed up and never chatted looks the same.

WHEN THE BRIDGE IS REACHABLE, IT IS A VERDICT
---------------------------------------------
With ``BRIDGE_URL`` + the mTLS certs in the environment (the platform-api
service already has them), this also reads ``GET /v1/pool/list`` and compares
every ASSIGNED slot against the platform's own container rows. That comparison
is exact, because the bridge is the only authority on who holds a slot:

  * slot ASSIGNED, platform row is a POOL container  -> consistent
  * slot ASSIGNED, platform row is a NAMED container -> SWAPPED. The slot is
    leaked forever and the slot database may hold that user's data.
  * slot ASSIGNED, NO platform row at all            -> ORPHAN. The account was
    deleted and the slot was never released, so the reconciler never reaped it
    and its database was never dropped. That user's messages are still on the
    VPS after they asked for them to be erased.

The 2026-08-31 fleet run found one SWAPPED (slot 17) and three ORPHANs
(09, 27, 28), all four containers still running 23 hours later.

Confirm each finding on the VPS before moving anything --- an orphan
``toup_agent_feedNNNN`` holding rows, with no live account behind it, is the
other half of the pair:

    ssh <vps>
    sudo -u postgres psql -c "\\l+" | grep toup_agent_feed
    curl -s --cert … https://bridge…/v1/pool/list   # who holds each slot now
    sudo -u postgres psql -d toup_agent_feed0017 \\
      -c "select count(*), min(created_at), max(created_at) from messages"

Recovery is a database rename or a dump/restore into the named DB, and it is
NOT automated here on purpose: picking the wrong slot overwrites a live
tenant's data with a stranger's.

USAGE
-----
    DATABASE_URL=postgresql+asyncpg://…  python backend/scripts/audit_pool_swaps.py
    …                                    python backend/scripts/audit_pool_swaps.py --user 871bac24
    …                                    python backend/scripts/audit_pool_swaps.py --json

Runs against the PLATFORM database (it reads `managed_containers` and
`agent_configs`) and reaches each tenant over the public agent URL.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.models import AgentConfig, ManagedContainer, User  # noqa: E402

POOL_PREFIX = "toup-agent-pool-"
PROBE_TIMEOUT_S = 8.0
# Below this age an empty agent is just a new signup, not a casualty.
MIN_AGE = timedelta(days=1)


@dataclass
class Row:
    user_id: str
    email: str
    container_name: str
    kind: str            # "pool" | "named" | "unknown"
    db_name: str
    status: str
    deploy_status: str
    created_at: str
    day_chats: Optional[int]     # None = the probe did not answer
    sessions: Optional[int]
    slot: str                    # the bridge's ASSIGNED slot, "" if none/unknown
    verdict: str


async def _probe(client: httpx.AsyncClient, url: str, key: str, path: str) -> Optional[int]:
    """How many rows does the tenant report? None when it did not answer.

    `None` is deliberately NOT zero — the entire defect this script exists to
    find is a system that could not tell those apart.
    """
    try:
        r = await client.get(
            f"{url.rstrip('/')}{path}",
            headers={"X-Agent-Key": key},
            timeout=PROBE_TIMEOUT_S,
        )
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        body = r.json()
    except Exception:
        return None
    if isinstance(body, list):
        return len(body)
    if isinstance(body, dict) and isinstance(body.get("sessions"), list):
        return len(body["sessions"])
    return None


async def _pool_members() -> Optional[list[dict]]:
    """Every warm-pool slot the BRIDGE knows about, or None if we cannot ask.

    None is deliberately not an empty list: "the bridge did not answer" and
    "there are no pool members" are the two facts this whole incident was about
    confusing, and this script exists because of that confusion.
    """
    try:
        from app.services.docker_host_service import _bridge_client
        async with _bridge_client(timeout_s=30) as client:
            r = await client.get("/v1/pool/list")
            if r.status_code != 200:
                print(f"   (bridge /v1/pool/list -> {r.status_code}; "
                      f"falling back to the heuristic)", file=sys.stderr)
                return None
            return r.json().get("members") or []
    except Exception as e:
        print(f"   (bridge unreachable: {e!r}; falling back to the heuristic)",
              file=sys.stderr)
        return None


async def audit(db: AsyncSession, only_user: Optional[str]) -> list[Row]:
    q = (
        select(ManagedContainer, AgentConfig, User)
        .join(AgentConfig, AgentConfig.user_id == ManagedContainer.user_id, isouter=True)
        .join(User, User.id == ManagedContainer.user_id, isouter=True)
    )
    rows = (await db.execute(q)).all()
    now = datetime.now(timezone.utc)
    out: list[Row] = []

    members = await _pool_members()
    # user_id -> slot, for every slot the bridge says is ASSIGNED.
    assigned: dict[str, str] = {}
    if members is not None:
        for m in members:
            if m.get("state") == "ASSIGNED" and m.get("assigned_user_id"):
                assigned[str(m["assigned_user_id"])] = str(m.get("slot") or "?")

    async with httpx.AsyncClient(follow_redirects=False) as client:
        for mc, cfg, user in rows:
            uid = str(mc.user_id)
            if only_user and not uid.startswith(only_user):
                continue
            name = mc.container_name or ""
            kind = "pool" if name.startswith(POOL_PREFIX) else ("named" if name else "unknown")

            day_chats = sessions = None
            if cfg and cfg.agent_url and cfg.agent_api_key:
                day_chats = await _probe(client, cfg.agent_url, cfg.agent_api_key,
                                         "/api/day-chats?limit=1")
                sessions = await _probe(client, cfg.agent_url, cfg.agent_api_key,
                                        "/api/sessions?limit=1")

            created = getattr(user, "created_at", None) or getattr(mc, "created_at", None)
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            old_enough = created is not None and (now - created) > MIN_AGE

            slot = assigned.get(uid, "")

            # The bridge's answer outranks the heuristic wherever we have it:
            # it is a fact about who holds the slot, not an inference from how
            # empty an agent looks.
            if slot and kind == "named":
                verdict = (
                    f"SWAPPED — bridge still holds slot {slot} for this user "
                    f"while the platform runs them named; slot DB may hold their data"
                )
            elif day_chats is None and sessions is None:
                verdict = "unreachable — re-run"
            elif kind == "named" and old_enough and not (day_chats or sessions):
                verdict = "CANDIDATE — named container, account older than a day, agent reports no history"
            elif kind == "pool":
                verdict = "pool member — protected by the R40 guard"
            else:
                verdict = "ok"

            out.append(Row(
                user_id=uid,
                email=getattr(user, "email", "") or "",
                container_name=name,
                kind=kind,
                db_name=mc.db_name or "",
                status=mc.status or "",
                deploy_status=(cfg.deploy_status if cfg else "") or "",
                created_at=created.isoformat() if created else "",
                day_chats=day_chats,
                sessions=sessions,
                slot=slot,
                verdict=verdict,
            ))

    # Slots ASSIGNED to a user the platform has never heard of. These CANNOT
    # come out of the query above — the account is gone, so there is no
    # ManagedContainer row to join from — and they are the worst case: the
    # user asked for deletion, the receipt said the container was destroyed,
    # and the slot was never released so its database was never dropped.
    if members is not None:
        known = {str(mc.user_id) for mc, _, _ in rows}
        for m in members:
            if m.get("state") != "ASSIGNED" or not m.get("assigned_user_id"):
                continue
            uid = str(m["assigned_user_id"])
            if uid in known or (only_user and not uid.startswith(only_user)):
                continue
            out.append(Row(
                user_id=uid, email="(account deleted)",
                container_name=m.get("container_name", ""), kind="pool",
                db_name=m.get("db_name", ""), status="", deploy_status="",
                created_at="", day_chats=None, sessions=None,
                slot=str(m.get("slot") or "?"),
                verdict=(
                    "ORPHAN — slot still ASSIGNED to a deleted account; its "
                    "database was never dropped"
                ),
            ))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", help="only this user id / prefix")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL is required (the PLATFORM database)", file=sys.stderr)
        return 2
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with Session() as db:
            rows = await audit(db, args.user)
    finally:
        await engine.dispose()

    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0

    swapped = [r for r in rows if r.verdict.startswith("SWAPPED")]
    orphans = [r for r in rows if r.verdict.startswith("ORPHAN")]
    candidates = [r for r in rows if r.verdict.startswith("CANDIDATE")]
    unreachable = [r for r in rows if r.verdict.startswith("unreachable")]

    def _rank(r: Row) -> int:
        for i, p in enumerate(("SWAPPED", "ORPHAN", "CANDIDATE", "unreachable")):
            if r.verdict.startswith(p):
                return i
        return 9

    print(f"\n{len(rows)} row(s)\n")
    print(f"  {'user':10} {'kind':7} {'container':28} {'slot':>4} {'days':>5} {'sess':>5}  verdict")
    for r in sorted(rows, key=lambda x: (_rank(x), x.user_id)):
        d = "—" if r.day_chats is None else str(r.day_chats)
        s = "—" if r.sessions is None else str(r.sessions)
        print(f"  {r.user_id[:8]:10} {r.kind:7} {r.container_name[:28]:28} "
              f"{r.slot or '—':>4} {d:>5} {s:>5}  {r.verdict}")

    print()
    if swapped:
        print(f"⚠  {len(swapped)} SWAPPED — the bridge and the platform disagree about "
              f"who holds the slot:")
        for r in swapped:
            print(f"     slot {r.slot}  {r.user_id}  {r.email}")
            print(f"        platform runs {r.container_name} (db {r.db_name})")
        print("     The slot database is the one that may hold their older messages.")
        print()
    if orphans:
        print(f"⚠  {len(orphans)} ORPHAN — slot still ASSIGNED to a DELETED account, "
              f"database never dropped:")
        for r in orphans:
            print(f"     slot {r.slot}  {r.container_name}  db {r.db_name}  user {r.user_id}")
        print("     These users asked for deletion and the receipt said it was done.")
        print("     Releasing the slot is what finally drops the database:")
        print("        POST {BRIDGE_URL}/v1/pool/release  {\"user_id\": \"…\"}")
        print()
    if candidates:
        print(f"⚠  {len(candidates)} candidate(s) — confirm on the VPS before moving anything:")
        for r in candidates:
            print(f"     {r.user_id}  {r.email}  named DB = {r.db_name or f'toup_agent_{r.user_id[:8]}'}")
        print("\n   Their old data, if this happened to them, is in a toup_agent_feedNNNN")
        print("   database with no ASSIGNED pool member. See this file's docstring.")
    else:
        print("✓  no candidates")
    if unreachable:
        print(f"\n   {len(unreachable)} agent(s) did not answer — an unreachable agent is not an")
        print("   empty one, which is the whole point. Re-run before concluding anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
