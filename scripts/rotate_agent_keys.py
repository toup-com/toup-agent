"""B-4c — rolling per-tenant agent_api_key rotation (operator runner).

Drives the tested rotation primitive (`rotate_agent_api_key`: DB flush →
bridge recreate → health-verify with the new key → rollback on any
failure) across managed tenants, one at a time, and prints the
post-rotation proof the closeout run requires per secret:

    old key rejected live · new key works live

SECRET DISCIPLINE: this script never prints key material — not the new
key, not the old, not prefixes. (`RotationResult.key_prefix` exists for
the user-facing UI; the operator run does not use it.) Output is
user_id[:8] + timestamps + HTTP statuses only.

Environment: run with the PLATFORM service env injected (railway run /
exported env) with DATABASE_URL pointed at the external TCP proxy.
Needs BRIDGE_URL + BRIDGE_{CA,CLIENT}_* PEMs in env — the primitive
calls the bridge exactly like the platform process does.

The old-key/new-key proof probes GET /api/sessions (X-Agent-Key
authed) — NOT /agent/health, which answers unauthenticated and proves
nothing about keys (memory: "/agent/health green ≠ chat works").

Dry-run by default; --apply to rotate. Sequential (rolling) on purpose:
each tenant's container is recreated and verified before the next
starts, and any failure stops the run unless --continue-on-error.

Usage:
    python -m scripts.rotate_agent_keys                 # dry-run: list candidates
    python -m scripts.rotate_agent_keys --user <uuid>   # dry-run, one tenant
    python -m scripts.rotate_agent_keys --apply         # rotate all managed
    python -m scripts.rotate_agent_keys --apply --user <uuid> [--user <uuid> ...]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx
from sqlalchemy import select


async def _candidates(user_ids: list[str] | None):
    """Managed tenants with a running container — the only rows the
    primitive will accept. Returns [(user_id, email, container_status)]."""
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig, ManagedContainer, User

    async with async_session_maker() as db:
        q = (
            select(AgentConfig.user_id, User.email, ManagedContainer.status)
            .join(User, User.id == AgentConfig.user_id)
            .join(
                ManagedContainer,
                ManagedContainer.user_id == AgentConfig.user_id,
                isouter=True,
            )
            .where(AgentConfig.hosting_mode == "managed")
        )
        rows = (await db.execute(q)).all()

    out = []
    for uid, email, status in rows:
        uid = str(uid)
        if user_ids and uid not in user_ids:
            continue
        out.append((uid, email or "?", status))
    return out


async def _probe_key(agent_url: str, key: str) -> int:
    """Status of an AUTHED endpoint under the given key. 200 = key
    works; 401/403 = key rejected. The key value never leaves this
    process."""
    try:
        async with httpx.AsyncClient(timeout=15.0, verify=True) as client:
            r = await client.get(
                f"{agent_url.rstrip('/')}/api/sessions",
                headers={"X-Agent-Key": key},
            )
            return r.status_code
    except Exception as e:
        print(f"      probe error: {type(e).__name__}", flush=True)
        return -1


async def _rotate_one(user_id: str, email: str) -> bool:
    """One tenant, own session (a failure must not poison the next
    tenant's transaction). Returns True on proven success."""
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig
    from app.services.agent_key_rotation import (
        AgentKeyRotationError,
        rotate_agent_api_key,
    )

    async with async_session_maker() as db:
        cfg = (
            await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )
        ).scalar_one_or_none()
        if cfg is None:
            print(f"  {user_id[:8]} SKIP no agent_config", flush=True)
            return False
        old_key = cfg.agent_api_key  # in memory only, for the 401 proof

        try:
            result = await rotate_agent_api_key(db, user_id)
        except AgentKeyRotationError as e:
            # The primitive already rolled the DB back and attempted to
            # restore the old container. Verify the rollback held.
            restored = await _probe_key(cfg.agent_url, old_key) if cfg.agent_url else -1
            print(
                f"  {user_id[:8]} FAILED ({e}) — rollback probe with old "
                f"key: {restored} (200 = tenant restored)",
                flush=True,
            )
            return False

        # Post-rotation proof, per the closeout contract.
        agent_url = cfg.agent_url
        old_status = await _probe_key(agent_url, old_key)
        new_status = await _probe_key(agent_url, result.new_agent_api_key)
        ok = old_status in (401, 403) and new_status == 200
        print(
            f"  {user_id[:8]} {email.split('@')[0][:12]:<12} "
            f"rotated_at={result.rotated_at.isoformat()}Z "
            f"old_key={old_status} new_key={new_status} "
            f"{'PROVEN' if ok else 'UNPROVEN — investigate before continuing'}",
            flush=True,
        )
        return ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rotate (default: dry-run)")
    ap.add_argument("--user", action="append", help="limit to user id(s)")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    cands = await _candidates(args.user)
    print(f"candidates: {len(cands)} managed tenant(s)", flush=True)
    for uid, email, status in cands:
        print(f"  {uid[:8]} {email.split('@')[0][:12]:<12} container={status}", flush=True)

    if not args.apply:
        print("dry-run — pass --apply to rotate", flush=True)
        return 0

    rotated = failed = 0
    for uid, email, status in cands:
        if status not in ("running", "provisioning"):
            print(f"  {uid[:8]} SKIP container={status}", flush=True)
            continue
        if await _rotate_one(uid, email):
            rotated += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print("stopping on first failure (no --continue-on-error)", flush=True)
                break

    print(f"done: rotated+proven {rotated}, failed {failed}, of {len(cands)}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
