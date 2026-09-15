"""Carry the platform's known timezone down to pool tenants whose row is NULL.

Why this exists (incident 2026-09-14): the tenant `users.timezone` is only ever
written by the first web/mobile WS turn, so a tenant whose first-ever turn
arrives on a tz-less channel (WhatsApp, Telegram, voice, a routine) buckets
that turn into a UTC day chat — a FUTURE day for anyone west of Greenwich in
the evening — and the app can never show it. The bind payload now carries
`user_timezone` and /admin/bind NULL-fills the tenant row, but a bind only
happens at claim time or on a config push. This script is that push, once,
for every pool tenant the platform knows a zone for.

Platform-side (runs where the platform DB and the bridge client are reachable):

    python scripts/push_tenant_timezones.py            # dry run: prints the plan
    python scripts/push_tenant_timezones.py --apply    # pushes, one tenant at a time
    python scripts/push_tenant_timezones.py --user-id <uuid> --apply

Rules:
  * POOL tenants only. `update_container_env` routes a pool member through
    the bridge's /v1/pool/refresh-config → /admin/bind (warm, no restart);
    for a NAMED tenant the same call is `provision_container(recreate=True)`,
    i.e. a container rebuild — never an acceptable side effect of a timezone
    sync. Named tenants self-heal on their first WS turn and are listed as
    SKIP here.
  * Never overwrites: the agent side NULL-fills only. Re-running is a no-op
    for tenants already filled.
  * One push at a time with a pause between them: each push is a bridge
    round-trip plus an /admin/bind on a live agent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from zoneinfo import ZoneInfo


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="push (default: dry run)")
    ap.add_argument("--user-id", default=None, help="only this user")
    ap.add_argument("--pause-s", type=float, default=2.0, help="pause between pushes")
    ap.add_argument("--limit", type=int, default=0, help="stop after N pushes (0 = all)")
    args = ap.parse_args()

    from sqlalchemy import select

    from app.db.database import async_session_maker
    from app.db.models import AgentConfig, ManagedContainer, User
    from app.services.docker_host_service import update_container_env

    pushed = 0
    async with async_session_maker() as db:
        q = (
            select(User, AgentConfig, ManagedContainer)
            .join(AgentConfig, AgentConfig.user_id == User.id)
            .join(ManagedContainer, ManagedContainer.user_id == User.id)
            .where(AgentConfig.deploy_status == "active")
        )
        if args.user_id:
            q = q.where(User.id == args.user_id)
        rows = (await db.execute(q)).all()

        for user, cfg, container in rows:
            tz = (user.timezone or "").strip()
            uid = user.id[:8]
            if not tz:
                print(f"SKIP  {uid} platform timezone unknown")
                continue
            try:
                ZoneInfo(tz)
            except Exception:
                print(f"SKIP  {uid} platform timezone unresolvable {tz!r}")
                continue
            name = container.container_name or ""
            if not name.startswith("toup-agent-pool-"):
                print(f"SKIP  {uid} named tenant {name or '?'} — a push would rebuild it; heals on first WS turn")
                continue
            if container.status != "running":
                print(f"SKIP  {uid} {name} status={container.status}")
                continue
            if not args.apply:
                print(f"WOULD {uid} {name} tz={tz}")
                continue
            try:
                await update_container_env(db, user.id, cfg)
                pushed += 1
                print(f"PUSHED {uid} {name} tz={tz}")
            except Exception as exc:  # noqa: BLE001 — one tenant's failure must not end the sweep
                print(f"FAIL  {uid} {name} {type(exc).__name__}")
            if args.limit and pushed >= args.limit:
                print(f"limit {args.limit} reached")
                break
            await asyncio.sleep(args.pause_s)

    print(f"done: pushed={pushed} apply={args.apply}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
