"""One-time tenant→platform backfill of users.first_media_played_at (mig 086).

WHY A SCRIPT AND NOT THE MIGRATION. Migration 086 backfills everything the
platform DB can see, which is only the monolith-era leftovers in `messages`
(1 user in prod on 2026-08-13). Every play since the databases were partitioned
lives in a per-user container DB that no platform migration can reach:
`messages`, `conversations` and `media_playlists` are all AGENT_ONLY, and the
platform's `media_playlists` mirror is empty by construction because the
proxy writes through to the tenant. So the remaining evidence has to be ASKED
for, one tenant at a time, over the same authenticated proxy the app uses.

WHAT COUNTS AS EVIDENCE. A saved playlist. Every station autosaves (the
creation hook and the first advance both INSERT), so a user who has played
music since 2026-08-03 has at least one row, and its `created_at` is a real
lower bound on when they first played something. This is deliberately a
CONSERVATIVE test:

  * it misses one-off plays that never built a station, and everyone who
    played before autosave existed. Those accounts re-latch on their next
    play, instantly and permanently — the cost of a miss is one song, not a
    lost feature.
  * it never over-claims: a playlist cannot exist without a station, and a
    station cannot exist without a play.

Rules:
  * Idempotent and MONOTONIC. Writes only where the column is NULL or later
    than the evidence, so re-running converges and can never move an instant
    forward. Safe to re-run after a partial failure.
  * Dry-run by default; --execute writes. Never prints an agent key.
  * Only tenants with an ACTIVE deploy and a URL+key are contacted, with
    bounded concurrency — this must not become a fleet-wide wake-up call.
  * An unreachable tenant is SKIPPED, never stamped and never cleared.

Usage (from the platform environment, where agent_api_key stays in-process):
    python scripts/backfill_first_media_played.py             # dry-run
    python scripts/backfill_first_media_played.py --execute   # write
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Tenants are contacted this many at a time. Low on purpose: a container that
# has been idle takes a moment to answer, and the point of this script is a
# quiet one-off sweep, not a thundering herd against the fleet.
CONCURRENCY = 4
TIMEOUT_S = 15.0


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        # The API serialises naive UTC; tolerate a trailing Z either way.
        return datetime.fromisoformat(raw.replace("Z", "").replace("+00:00", ""))
    except ValueError:
        return None


async def _earliest_playlist(client: httpx.AsyncClient, url: str, key: str) -> tuple[str, datetime | None]:
    """('ok'|'empty'|'unreachable', earliest created_at)."""
    try:
        resp = await client.get(
            f"{url}/api/media/playlists?media_type=music",
            headers={"X-Agent-Key": key},
            timeout=TIMEOUT_S,
        )
    except Exception:
        return ("unreachable", None)
    if resp.status_code != 200:
        return ("unreachable", None)
    try:
        rows = resp.json()
    except Exception:
        return ("unreachable", None)
    if not isinstance(rows, list) or not rows:
        return ("empty", None)
    stamps = [d for d in (_parse_dt(r.get("created_at")) for r in rows) if d]
    if not stamps:
        # Rows exist but carry no usable instant — they are still proof of a
        # play; fall back to "now" rather than dropping the evidence.
        return ("ok", datetime.utcnow())
    return ("ok", min(stamps))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="write (default: dry-run)")
    args = ap.parse_args()

    dsn = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(
        dsn, connect_args={"statement_cache_size": 0}, pool_pre_ping=True,
    )

    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT c.user_id::text, c.agent_url, c.agent_api_key, u.first_media_played_at "
            "FROM agent_configs c JOIN users u ON u.id = c.user_id "
            "WHERE c.deploy_status = 'active' "
            "  AND c.agent_url IS NOT NULL AND c.agent_url != '' "
            "  AND c.agent_api_key IS NOT NULL AND c.agent_api_key != '' "
            "ORDER BY c.user_id"
        ))).fetchall()

    print(f"bound, active tenants: {len(rows)}")
    already = [r for r in rows if r[3] is not None]
    todo = [r for r in rows if r[3] is None]
    print(f"  already stamped (nothing to do): {len(already)}")
    print(f"  to ask: {len(todo)}")

    sem = asyncio.Semaphore(CONCURRENCY)
    findings: list[tuple[str, str, datetime | None]] = []

    async with httpx.AsyncClient() as client:
        async def one(user_id: str, url: str, key: str) -> None:
            async with sem:
                state, when = await _earliest_playlist(client, url, key)
                findings.append((user_id, state, when))

        await asyncio.gather(*[one(r[0], r[1], r[2]) for r in todo])

    findings.sort()
    by_state: dict[str, list] = {"ok": [], "empty": [], "unreachable": []}
    for uid, state, when in findings:
        by_state[state].append((uid, when))

    print("\nEVIDENCE FOUND (will stamp):")
    for uid, when in by_state["ok"]:
        print(f"  {uid[:8]}  earliest playlist {when}")
    print(f"no playlists (correctly left NULL): {len(by_state['empty'])}")
    print(f"unreachable (SKIPPED, not stamped): {len(by_state['unreachable'])}")
    for uid, _ in by_state["unreachable"]:
        print(f"  {uid[:8]}")

    if not by_state["ok"]:
        print("\nnothing to write.")
        await engine.dispose()
        return 0

    if not args.execute:
        print(f"\nDRY-RUN — {len(by_state['ok'])} row(s) would be stamped. Re-run with --execute.")
        await engine.dispose()
        return 0

    print("\nEXECUTING:")
    wrote = 0
    async with engine.begin() as conn:
        for uid, when in by_state["ok"]:
            # Monotonic, same rule as the endpoint and the migration: only
            # NULL or a LATER value is replaced, so a re-run is a no-op and a
            # racing client report always wins if it is earlier.
            res = await conn.execute(
                text(
                    "UPDATE users SET first_media_played_at = :when "
                    "WHERE id = :uid AND (first_media_played_at IS NULL "
                    "                     OR first_media_played_at > :when)"
                ),
                {"uid": uid, "when": when},
            )
            wrote += res.rowcount or 0
            print(f"  {uid[:8]}  {'stamped' if res.rowcount else 'unchanged'}")
    await engine.dispose()
    print(f"\ncandidates {len(by_state['ok'])} / stamped {wrote}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
