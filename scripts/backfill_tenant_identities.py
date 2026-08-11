"""Backfill a tenant's `identities` from its platform-DB rows (W-6 / B-2).

Why this exists: `identities` is AGENT_ONLY by partition, but two writers
put rows in the PLATFORM copy anyway — `_seed_default_identities` at
signup (default soul + Behavioral Guidelines) and the voice-onboarding
finalize. The soul page dual-writes, so personalised souls exist on both
sides; the seeded defaults exist ONLY platform-side. Text chat and the
agent voice assembler read the TENANT copy, so every user whose tenant
row is missing speaks without their Behavioral Guidelines — and the W-6
shadow reports `differs=Behavioral Guidelines,Core Identity` on every
session, which withholds the flip.

Runs INSIDE the tenant container (docker exec), reading that ONE user's
platform rows as JSON on stdin:

    docker exec -i toup-agent-<id> python scripts/backfill_tenant_identities.py \
        --expect-user <full-user-uuid> [--apply] < rows.json

Rules, in order of importance:
  * `--expect-user` must equal the container's `settings.user_id` or the
    script aborts before touching anything — the failure mode this guards
    against is pushing user A's persona into user B's tenant.
  * The tenant is source of truth where it already has an ACTIVE row of a
    type: never overwritten, only compared (sha256 digests printed, 8
    hex chars, so a drifted soul is visible without printing anyone's
    persona).
  * Missing types are INSERTed verbatim (same name/content/priority,
    is_active=True) — but only with --apply; the default is a dry run.
  * Output is one line per type: INSERT/WOULD-INSERT/MATCH/DIFFER/SKIP.
    Never content, never secrets.

stdin JSON shape: [{"identity_type": ..., "name": ..., "content": ...,
"priority": ...}, ...] — the platform export for exactly one user.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:8]


async def _run(rows: list, expect_user: str, apply: bool) -> int:
    from sqlalchemy import select

    from app.db.database import async_session_maker
    from app.db.models import Identity
    from app.services.runtime_identity import get_user_id

    # NOT `settings.user_id`: pool-bound members carry their tenant in
    # /etc/toup-agent/runtime.json (restore_at_boot mutates the RUNNING
    # process's settings, which a docker-exec interpreter never sees).
    # runtime_identity reads env-or-file and is correct in both modes.
    bound_user = get_user_id()
    if not bound_user:
        print("ABORT: container has no bound user (unbound pool slot?)")
        return 2
    if bound_user != expect_user:
        print(
            f"ABORT: --expect-user {expect_user[:8]}… does not match this "
            f"container's user {bound_user[:8]}… — wrong tenant"
        )
        return 2

    async with async_session_maker() as db:
        existing = (await db.execute(
            select(Identity).where(
                Identity.user_id == expect_user,
                Identity.is_active.is_(True),
            )
        )).scalars().all()
        by_type: dict = {}
        for r in existing:
            by_type.setdefault(r.identity_type, []).append(r)

        changed = False
        for row in rows:
            itype = row.get("identity_type")
            content = row.get("content") or ""
            if not itype or not content:
                print(f"SKIP {itype or '?'}: empty row")
                continue
            have = by_type.get(itype)
            if have:
                for h in have:
                    verdict = "MATCH" if _digest(h.content) == _digest(content) else "DIFFER"
                    print(
                        f"{verdict} {itype}: tenant={_digest(h.content)} "
                        f"platform={_digest(content)} (tenant kept)"
                    )
                continue
            if apply:
                db.add(Identity(
                    id=str(uuid.uuid4()),
                    user_id=expect_user,
                    identity_type=itype,
                    name=row.get("name") or itype,
                    content=content,
                    priority=int(row.get("priority") or 0),
                    is_active=True,
                ))
                changed = True
                print(f"INSERT {itype}: {len(content)} chars, digest={_digest(content)}")
            else:
                print(f"WOULD-INSERT {itype}: {len(content)} chars, digest={_digest(content)}")
        if changed:
            await db.commit()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-user", required=True,
                    help="full user uuid this stdin payload belongs to")
    ap.add_argument("--apply", action="store_true",
                    help="write missing rows (default: dry run)")
    args = ap.parse_args()

    rows = json.load(sys.stdin)
    if not isinstance(rows, list):
        print("ABORT: stdin is not a JSON list")
        return 2
    return asyncio.run(_run(rows, args.expect_user, args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
