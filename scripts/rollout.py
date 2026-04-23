"""
Manual rollout CLI — break-glass fallback for when platform-api is down.

Normal path: CI pushes to main → build-agent.yml → POST /api/admin/rollout/start.
If Railway is down and we need to roll out (or roll BACK), this script
talks directly to the bridge using the stored mTLS certs.

Usage:
    # Normal: dispatch via platform webhook (takes most of the rollout burden)
    python -m scripts.rollout --sha abc1234

    # Break-glass: bypass platform, drive rollout from the local shell
    python -m scripts.rollout --sha abc1234 --bypass-platform

    # Rollback to a specific SHA (arbitrary tag, no canary wait)
    python -m scripts.rollout --sha abc1234 --bypass-platform --skip-canary-wait

    # Dry run — print what would happen
    python -m scripts.rollout --sha abc1234 --dry-run

Reads creds from Railway env (same Settings object as platform) via
`railway run python -m scripts.rollout ...`. No new secrets to manage.

Fires logs to stderr so stdout is reserved for the rollout summary.

See docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md §9 for the break-glass
rationale.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

# Ensure repo root is importable before app.* imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402


def _log(msg: str) -> None:
    print(f"[rollout-cli] {msg}", file=sys.stderr, flush=True)


async def _via_platform(image_tag: str, canary_wait_minutes: Optional[int], dry_run: bool) -> int:
    """Happy path: POST to /api/admin/rollout/start. Same code path as CI."""
    import httpx

    if not settings.rollout_secret:
        _log("ERROR: ROLLOUT_SECRET not set in Railway env — cannot dispatch via platform")
        return 2
    if not settings.platform_api_url:
        _log("ERROR: PLATFORM_API_URL not set")
        return 2

    url = settings.platform_api_url.rstrip("/") + "/admin/rollout/start"
    body: dict = {"image_tag": image_tag, "trigger": "manual"}
    if canary_wait_minutes is not None:
        body["canary_wait_minutes"] = canary_wait_minutes

    _log(f"POST {url} — trigger=manual, image_tag={image_tag}")
    if dry_run:
        _log("(dry-run — no request sent)")
        print(f"DRY RUN: would POST {url} with {body}")
        return 0

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            url,
            headers={"X-Rollout-Secret": settings.rollout_secret, "Content-Type": "application/json"},
            json=body,
        )
    _log(f"HTTP {r.status_code}")
    print(r.text)
    return 0 if r.status_code == 202 else 1


async def _bypass_platform(
    image_tag: str,
    canary_wait_minutes: int,
    skip_canary_wait: bool,
    dry_run: bool,
) -> int:
    """Break-glass path: drive the rollout from this shell using the bridge
    directly. Replicates rollout_service.py but without the Rollout / RolloutAttempt
    audit rows (platform is assumed down, DB is inaccessible).

    Compromise: no audit log if platform is down. On operator.
    """
    if not (settings.bridge_url and settings.bridge_ca_cert
            and settings.bridge_client_cert and settings.bridge_client_key):
        _log("ERROR: bridge mTLS settings not populated — cannot reach bridge")
        return 2

    # We can't enumerate tenants without platform DB. So bypass mode needs
    # the operator to supply the tenant list explicitly, OR we query the
    # bridge for running containers.
    _log("querying bridge for running tenants (platform DB may be unreachable)")

    import httpx
    import ssl
    import tempfile

    # Write certs to tmp (mirrors docker_host_service._write_cert_tmpfile)
    def _tmpfile(pem: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".pem")
        with os.fdopen(fd, "w") as f:
            f.write(pem)
        os.chmod(path, 0o600)
        return path

    ca_path = _tmpfile(settings.bridge_ca_cert)
    cert_path = _tmpfile(settings.bridge_client_cert)
    key_path = _tmpfile(settings.bridge_client_key)
    ctx = ssl.create_default_context(cafile=ca_path)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)

    base_url = settings.bridge_url.rstrip("/")

    # The bridge has GET /v1/tenants/<prefix>/status but no "list all" endpoint.
    # For break-glass, we rely on the operator-supplied --prefix-list OR
    # shell-out to docker ps via a bridge admin endpoint (not implemented).
    # For Phase 3 MVP: require the operator to pass a comma-separated prefix list.
    prefix_list_env = os.environ.get("ROLLOUT_PREFIXES", "").strip()
    if not prefix_list_env:
        _log("ERROR: bypass mode requires ROLLOUT_PREFIXES env var "
             "(comma-separated 8-char prefixes to upgrade)")
        _log("  example: ROLLOUT_PREFIXES=3134fece,6381be69 python -m scripts.rollout --sha ... --bypass-platform")
        return 2
    prefixes = [p.strip() for p in prefix_list_env.split(",") if p.strip()]
    _log(f"planning rollout to {len(prefixes)} tenant(s): {prefixes}")

    if dry_run:
        print(f"DRY RUN — would upgrade prefixes {prefixes} to {image_tag}")
        return 0

    # Sequential (no canary, no batching) for simplicity in break-glass.
    # Operator is doing this manually — parallelism isn't the point.
    async with httpx.AsyncClient(base_url=base_url, verify=ctx, timeout=240) as client:
        results: list[tuple[str, int, str]] = []
        for i, prefix in enumerate(prefixes, 1):
            _log(f"[{i}/{len(prefixes)}] upgrading {prefix} → {image_tag}")
            try:
                r = await client.post(
                    f"/v1/tenants/{prefix}/upgrade",
                    json={"image_tag": image_tag, "rollout_id": "manual-cli"},
                )
                status_str = r.text[:200] if r.status_code != 200 else "ok"
                results.append((prefix, r.status_code, status_str))
                _log(f"  → HTTP {r.status_code}")
            except Exception as e:
                results.append((prefix, 0, str(e)[:200]))
                _log(f"  → EXCEPTION: {e}")

    # Summary on stdout
    print("\n=== Rollout summary ===")
    ok_count = sum(1 for _, code, _ in results if code == 200)
    fail_count = len(results) - ok_count
    for prefix, code, detail in results:
        marker = "✓" if code == 200 else "✗"
        print(f"  {marker} {prefix}  HTTP {code}  {detail}")
    print(f"\n{ok_count}/{len(results)} succeeded, {fail_count} failed")

    # Cleanup cert tmp files
    for p in (ca_path, cert_path, key_path):
        try: os.unlink(p)
        except Exception: pass

    return 0 if fail_count == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(
        prog="rollout",
        description="Manual rollout trigger. Default path: POST to platform webhook. "
                    "Use --bypass-platform when platform-api is down.",
    )
    p.add_argument(
        "--sha", required=True,
        help="Short SHA (12-char) or full SHA. Will be prefixed with "
             "ghcr.io/toup-com/toup-agent: if not already.",
    )
    p.add_argument(
        "--canary-wait-minutes", type=int, default=None,
        help="Override canary observation window (1-60). Default: platform default (10).",
    )
    p.add_argument(
        "--bypass-platform", action="store_true",
        help="Drive the rollout directly against the bridge, bypassing the platform webhook. "
             "Requires ROLLOUT_PREFIXES env var with a comma-separated tenant list.",
    )
    p.add_argument(
        "--skip-canary-wait", action="store_true",
        help="(Bypass mode only) Upgrade all tenants sequentially with no canary observation. "
             "USE ONLY FOR EMERGENCY ROLLBACK.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen; make no requests.",
    )
    args = p.parse_args()

    # Normalize the image tag
    tag = args.sha
    if not tag.startswith("ghcr.io/toup-com/toup-agent:"):
        tag = f"ghcr.io/toup-com/toup-agent:{tag}"

    if args.bypass_platform:
        return asyncio.run(_bypass_platform(
            tag,
            canary_wait_minutes=args.canary_wait_minutes or 10,
            skip_canary_wait=args.skip_canary_wait,
            dry_run=args.dry_run,
        ))
    else:
        return asyncio.run(_via_platform(
            tag,
            canary_wait_minutes=args.canary_wait_minutes,
            dry_run=args.dry_run,
        ))


if __name__ == "__main__":
    sys.exit(main())
