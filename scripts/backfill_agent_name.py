"""One-time platform→tenant backfill of agent_configs.agent_name (L-1).

The write path is fixed (fail-closed write-through, PR #599); this script
converges the HISTORICAL drift: bound tenants whose platform row carries a
real agent name while the tenant copy — the one text chat and the W-6
voice assembler read — is NULL (the 30s post-start gate ate the seed for
years of provisions).

Rules (Last Mile L-1, verbatim from the run authorization):
  * REAL names only. NULL, empty, and the literal stub "Agent" are
    excluded — backfilling stubs would make agents say "your name is
    Agent" in voice.
  * platform→tenant direction (platform is the authority for this
    column; the OPPOSITE of the identities backfill).
  * Idempotent: the write sets the tenant copy to the platform value via
    the same PUT /api/soul/sync receiver every other carrier uses;
    re-running converges to the same state.
  * Dry-run by default; --execute to write. Never prints a key.

Usage (from the platform environment so agent_api_key stays in-process):
    python scripts/backfill_agent_name.py            # dry-run
    python scripts/backfill_agent_name.py --execute  # write
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

STUB_NAMES = {"Agent"}


def classify(name: str | None) -> str:
    if name is None:
        return "null"
    if not name.strip():
        return "empty"
    if name.strip() in STUB_NAMES:
        return "stub"
    return "real"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="write (default: dry-run)")
    args = ap.parse_args()

    dsn = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(
        dsn,
        connect_args={"statement_cache_size": 0},
        pool_pre_ping=True,
    )

    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT user_id::text, agent_name, agent_url, agent_api_key "
            "FROM agent_configs "
            "WHERE agent_url IS NOT NULL AND agent_url != '' "
            "AND agent_api_key IS NOT NULL AND agent_api_key != '' "
            "ORDER BY user_id"
        ))).fetchall()
    await engine.dispose()

    buckets: dict[str, list] = {"real": [], "stub": [], "null": [], "empty": []}
    for r in rows:
        buckets[classify(r[1])].append(r)

    print(f"bound tenants: {len(rows)}")
    for k in ("real", "stub", "null", "empty"):
        print(f"  {k}: {len(buckets[k])}")
    print("\nEXCLUDED (stubs — must stay untouched):")
    for r in buckets["stub"]:
        print(f"  {r[0][:8]}  name={r[1]!r}")
    print("EXCLUDED (null/empty — nothing to backfill):")
    for r in buckets["null"] + buckets["empty"]:
        print(f"  {r[0][:8]}")
    print("\nPLANNED WRITES (real names, platform→tenant):")
    for r in buckets["real"]:
        print(f"  {r[0][:8]}  name={r[1]!r}")

    if not args.execute:
        print("\nDRY-RUN — no writes. Re-run with --execute.")
        return 0

    print("\nEXECUTING:")
    ok = fail = 0
    async with httpx.AsyncClient(timeout=25.0) as client:
        for r in buckets["real"]:
            user_id, name, agent_url, agent_api_key = r[0], r[1], r[2], r[3]
            try:
                resp = await client.put(
                    f"{agent_url}/api/soul/sync",
                    json={
                        "user_id": user_id,
                        "agent_config_updates": {"agent_name": name},
                    },
                    headers={"X-Agent-Key": agent_api_key},
                )
                status = resp.status_code
            except Exception as e:
                status = f"EXC:{type(e).__name__}"
            if status == 200:
                ok += 1
                print(f"  {user_id[:8]}  OK")
            else:
                fail += 1
                print(f"  {user_id[:8]}  FAIL status={status}")
    print(f"\nexpected {len(buckets['real'])} / wrote {ok} / failed {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
