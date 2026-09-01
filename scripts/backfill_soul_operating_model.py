"""Recompile every saved soul so existing agents get the default operating model.

WHY THIS EXISTS
    `compile_soul()` gained `OPERATING_MODEL` — the product's one default
    persona, Claude Cowork's working model — and `compiled_text` is CACHED, in
    two places: `soul_configs.compiled_text` on the platform and the tenant's
    `Identity(identity_type='soul').content`, which is what the agent actually
    reads on every turn. Nothing recompiles on read. So the compiler change
    reaches a user only when they next save their soul, which most people never
    will.

    Users with NO soul row need nothing from this script: they render
    `agent_runner.DEFAULT_SOUL_CONTENT`, which imports the same constant and
    lands with the next agent rollout.

WHAT IT DOES
    For every `soul_configs` row: recompile from the row's own stored fields
    (name, color, pronouns, style, traits, custom_instructions — the USER's
    answers, untouched), write it back, and push it to the tenant through the
    same `PUT /api/soul/sync` receiver every other carrier uses.

    It changes nobody's personality. The only difference between the old text
    and the new one is the block the compiler now adds to everyone.

RULES
    * Idempotent. Re-running converges: a row whose compiled text already equals
      the fresh compile is skipped, so a partial run can simply be re-run.
    * Dry-run by default. `--execute` writes. Never prints a key.
    * The tenant push is best-effort PER USER and reported per user. A tenant
      that is down does not fail the others, and the platform row is only marked
      synced when its push actually returned 200 — so the next run retries
      exactly the ones that failed.
    * `--limit N` to rehearse on a handful first.

USAGE (from the platform environment, so agent_api_key stays in-process)
    python scripts/backfill_soul_operating_model.py                # dry-run
    python scripts/backfill_soul_operating_model.py --limit 5 --execute
    python scripts/backfill_soul_operating_model.py --execute
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.soul_compiler import compile_soul  # noqa: E402


def _recompile(row) -> str:
    return compile_soul({
        "name": row.name or "Agent",
        "pronouns": row.pronouns or "they",
        "style": row.style or "casual",
        # JSON column: asyncpg hands back a list already; be tolerant of NULL.
        "traits": list(row.traits or []),
        "custom_instructions": row.custom_instructions or "",
    })


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="write (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows")
    args = ap.parse_args()

    dsn = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(
        dsn,
        connect_args={"statement_cache_size": 0},
        pool_pre_ping=True,
    )

    async with engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT sc.user_id::text AS user_id, sc.name, sc.pronouns, sc.style, "
            "       sc.traits, sc.custom_instructions, sc.compiled_text, "
            "       ac.agent_url, ac.agent_api_key "
            "FROM soul_configs sc "
            "LEFT JOIN agent_configs ac ON ac.user_id = sc.user_id "
            "ORDER BY sc.user_id"
        ))).fetchall()
    await engine.dispose()

    stale, fresh = [], 0
    for r in rows:
        if _recompile(r) == (r.compiled_text or ""):
            fresh += 1
        else:
            stale.append(r)
    if args.limit:
        stale = stale[: args.limit]

    bound = [r for r in stale if r.agent_url and r.agent_api_key]
    unbound = [r for r in stale if not (r.agent_url and r.agent_api_key)]

    print(f"soul rows: {len(rows)}")
    print(f"  already current: {fresh}")
    print(f"  to recompile:    {len(stale)}  (bound tenants {len(bound)}, unbound {len(unbound)})")
    print("\nUnbound rows get the platform write only — their tenant has no URL yet,")
    print("so the container picks the new text up when it is provisioned/bound.")
    for r in stale:
        where = "bound" if (r.agent_url and r.agent_api_key) else "unbound"
        print(f"  {r.user_id[:8]}  style={r.style!r}  {where}")

    if not args.execute:
        print("\nDRY-RUN — no writes. Re-run with --execute.")
        return 0

    print("\nEXECUTING:")
    ok = failed = 0
    engine = create_async_engine(
        dsn, connect_args={"statement_cache_size": 0}, pool_pre_ping=True,
    )
    async with httpx.AsyncClient(timeout=25.0) as client:
        for r in stale:
            compiled = _recompile(r)
            # Platform first: it is the authority for `compiled_text`, and a
            # tenant push that lands against an un-updated platform row would be
            # undone by the next ordinary save.
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "UPDATE soul_configs SET compiled_text = :c, updated_at = NOW() "
                        "WHERE user_id = CAST(:u AS uuid)"
                    ),
                    {"c": compiled, "u": r.user_id},
                )
                await conn.execute(
                    text(
                        "UPDATE identities SET content = :c, updated_at = NOW() "
                        "WHERE user_id = CAST(:u AS uuid) AND identity_type = 'soul'"
                    ),
                    {"c": compiled, "u": r.user_id},
                )

            if not (r.agent_url and r.agent_api_key):
                ok += 1
                print(f"  {r.user_id[:8]}  platform only (unbound)")
                continue

            try:
                resp = await client.put(
                    f"{r.agent_url}/api/soul/sync",
                    json={
                        "user_id": r.user_id,
                        "name": r.name,
                        "compiled_text": compiled,
                        # NOT a personality change and NOT an onboarding event:
                        # nothing else may ride along on a maintenance write.
                        "deactivate_agent_soul_memories": False,
                    },
                    headers={"X-Agent-Key": r.agent_api_key},
                )
                status = resp.status_code
            except Exception as e:  # noqa: BLE001 — reported, never raised
                status = f"EXC:{type(e).__name__}"

            if status == 200:
                async with engine.begin() as conn:
                    await conn.execute(
                        text(
                            "UPDATE soul_configs SET vps_soul_synced_at = NOW() "
                            "WHERE user_id = CAST(:u AS uuid)"
                        ),
                        {"u": r.user_id},
                    )
                ok += 1
                print(f"  {r.user_id[:8]}  OK")
            else:
                failed += 1
                print(f"  {r.user_id[:8]}  TENANT PUSH FAILED status={status} (re-run to retry)")
    await engine.dispose()

    print(f"\nplanned {len(stale)} / done {ok} / tenant-push failures {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
