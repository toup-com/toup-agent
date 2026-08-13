"""Measure cross-database drift for a declared shared column (L-1b).

Every column in SHARED_COLUMN_AUTHORITY (app/db/models/base.py) exists in
BOTH the platform DB and a tenant DB. This probe connects to one platform
DB and ONE tenant DB and reports, per shared column, how many rows agree
and how many have split (the users.timezone / agent_configs.agent_name
defect class) — as COUNTS ONLY. It never prints a row value, a DSN, or
any user data: values are md5-hashed server-side and only compared.

DSNs come from env vars the operator passes; nothing is hardcoded:

    PLATFORM_DATABASE_URL=... TENANT_DATABASE_URL=... \
        python scripts/authority_drift_probe.py [--target users.timezone] \
        [--fail-on-drift]

Defaults probe the two columns that were live defects:
users.timezone and agent_configs.agent_name.

Read-only by construction (SELECT only — there is nothing to --apply).
Buckets per join key present in BOTH databases:
    match             — equal (including both NULL)
    platform-null     — platform NULL, tenant set
    tenant-null       — tenant NULL, platform set
    both-set-differ   — both set, values differ
Totals line: "expected N / compared N / drifted N" where expected is the
number of distinct join keys seen on either side, compared is keys on
both, drifted = platform-null + tenant-null + both-set-differ.

Exit codes: 0 ok, 1 drift found with --fail-on-drift, 2 usage/env error.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Runnable as `python scripts/authority_drift_probe.py` from backend/
# (same bootstrap as the sibling scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tables where the tenant row is keyed by user_id, not by the row's own
# uuid (agent_configs tenant rows are materialized with a FRESH id — see
# the SHARED_COLUMN_AUTHORITY note on agent_configs.id).
_JOIN_KEYS = {
    "users": "id",
    "agent_configs": "user_id",
    # extension_devices rows are surface-local (a device pairs against
    # exactly one surface); cross-DB comparison is structurally N/A.
}

DEFAULT_TARGETS = ["users.timezone", "agent_configs.agent_name"]


def _clean_dsn(raw: str) -> str:
    """Normalize a SQLAlchemy-style DSN for asyncpg. Never printed."""
    return raw.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres+asyncpg://", "postgresql://"
    )


async def _fetch_side(dsn: str, table: str, column: str, join_key: str) -> dict:
    """Return {join_key_text: (is_null, md5_or_None)} for one database.

    The value itself never crosses the wire: md5(col::text) is computed
    server-side, so this process only ever holds hashes and counts.
    """
    import asyncpg  # imported here so --help works without the dep

    conn = await asyncpg.connect(
        dsn,
        statement_cache_size=0,  # pgbouncer txn-mode safe
        server_settings={"timezone": "UTC"},  # canonical ::text rendering
    )
    try:
        rows = await conn.fetch(
            f'SELECT "{join_key}"::text AS k, '
            f'("{column}" IS NULL) AS is_null, '
            f'md5("{column}"::text) AS h '
            f'FROM "{table}"'
        )
    finally:
        await conn.close()

    out: dict = {}
    dupes = 0
    for r in rows:
        k = r["k"]
        if k in out:
            dupes += 1
            continue
        out[k] = (r["is_null"], r["h"])
    if dupes:
        print(f"  WARNING: {dupes} duplicate join-key row(s) skipped on one side")
    return out


async def probe_target(target: str, platform_dsn: str, tenant_dsn: str) -> int:
    """Probe one '<table>.<column>'. Returns the drifted-row count."""
    from app.db.models.base import SHARED_COLUMN_AUTHORITY  # validates target

    if target not in SHARED_COLUMN_AUTHORITY:
        print(f"ABORT: {target} has no SHARED_COLUMN_AUTHORITY entry — declare it first")
        raise SystemExit(2)
    table, column = target.split(".", 1)
    join_key = _JOIN_KEYS.get(table)
    if join_key is None:
        print(f"{target}: rows are surface-local (see base.py) — cross-DB drift is N/A, skipping")
        return 0

    entry = SHARED_COLUMN_AUTHORITY[target]
    print(f"{target}  authority={entry['authority']}  sync={entry['sync']}")

    platform_rows, tenant_rows = await asyncio.gather(
        _fetch_side(platform_dsn, table, column, join_key),
        _fetch_side(tenant_dsn, table, column, join_key),
    )

    all_keys = set(platform_rows) | set(tenant_rows)
    both = set(platform_rows) & set(tenant_rows)
    match = p_null = t_null = differ = 0
    for k in both:
        p_is_null, p_hash = platform_rows[k]
        t_is_null, t_hash = tenant_rows[k]
        if p_is_null and t_is_null:
            match += 1
        elif p_is_null:
            p_null += 1
        elif t_is_null:
            t_null += 1
        elif p_hash == t_hash:
            match += 1
        else:
            differ += 1

    drifted = p_null + t_null + differ
    print(
        f"  platform rows: {len(platform_rows)}   tenant rows: {len(tenant_rows)}   "
        f"platform-only keys: {len(platform_rows) - len(both)}   "
        f"tenant-only keys: {len(tenant_rows) - len(both)}"
    )
    print(
        f"  match: {match}   platform-null: {p_null}   tenant-null: {t_null}   "
        f"both-set-differ: {differ}"
    )
    print(f"  expected {len(all_keys)} / compared {len(both)} / drifted {drifted}")
    return drifted


async def _run(targets: list[str], fail_on_drift: bool) -> int:
    platform_dsn = os.environ.get("PLATFORM_DATABASE_URL", "").strip()
    tenant_dsn = os.environ.get("TENANT_DATABASE_URL", "").strip()
    if not platform_dsn or not tenant_dsn:
        print(
            "ABORT: set PLATFORM_DATABASE_URL and TENANT_DATABASE_URL in the "
            "environment (values are never printed)"
        )
        return 2

    total_drift = 0
    for target in targets:
        total_drift += await probe_target(
            target, _clean_dsn(platform_dsn), _clean_dsn(tenant_dsn)
        )
        print()

    print(f"TOTAL drifted rows across {len(targets)} target(s): {total_drift}")
    if total_drift and fail_on_drift:
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--target",
        action="append",
        dest="targets",
        metavar="TABLE.COLUMN",
        help=f"shared column to probe (repeatable; default: {', '.join(DEFAULT_TARGETS)})",
    )
    ap.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit 1 when any drifted row is found (for orchestrator gating)",
    )
    args = ap.parse_args()
    targets = args.targets or DEFAULT_TARGETS
    return asyncio.run(_run(targets, args.fail_on_drift))


if __name__ == "__main__":
    sys.exit(main())
