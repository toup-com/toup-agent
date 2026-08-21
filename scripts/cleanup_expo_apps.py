#!/usr/bin/env python3
"""Reclaim the Expo pipeline's disk inside one agent container.

Measured baseline (2026-08-20, this repo's own dependency set — see
``MIGRATION_INVENTORY.md`` §3): **462,972 KiB of ``node_modules`` across
27,133 files per app**, against 1.25 MiB of actual authored source. 99.7 % of
an app directory is machinery that the single-file HTML pipeline does not
have.

**Dry-run by default.** It prints exactly what it would remove and how many
bytes that is, and exits without touching anything. ``--apply`` is required
to delete, and even then it only ever removes regenerable build machinery:

    <app>/node_modules/            reinstallable from package.json
    <app>/.expo/                   Metro/Expo cache
    <app>/node_modules/.cache/     (inside node_modules)
    $TMPDIR/metro-*                Metro transform cache, process-level

It **never** removes an app's source, its ``storage/``, its SQLite file, its
``.git``, its lockfile — nor anything at all under the HTML app root. So a
container that is later rolled back to the Expo pipeline recovers with one
``npm install`` per app, and a container that is not rolled back never
notices.

Safety rails, because this runs as root inside a tenant's container:

  * every deletion target must resolve under the configured apps root, after
    ``realpath`` — symlink escapes are refused, not followed;
  * a running app is skipped unless ``--include-running``, so we do not pull
    ``node_modules`` out from under a live Metro process;
  * ``--keep-newest N`` protects the N most recently updated apps.

Usage::

    python3 backend/scripts/cleanup_expo_apps.py                 # measure
    python3 backend/scripts/cleanup_expo_apps.py --json          # machine-readable
    python3 backend/scripts/cleanup_expo_apps.py --apply         # reclaim
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Regenerable machinery only. Every name here is reproducible from the app's
# own package.json; nothing authored appears in this list.
REGENERABLE = ("node_modules", ".expo", ".expo-shared", "web-build", "dist")

# Ports the Expo pipeline allocates (app_manager.METRO_PORT_RANGE /
# WEB_PORT_RANGE). A listening port in these ranges means a live dev server.
METRO_PORT_RANGE = (3001, 3050)
WEB_PORT_RANGE = (4001, 4050)


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024.0
    return f"{n:.1f} TiB"


def dir_size(path: str) -> tuple:
    """(bytes, files) actually on disk under ``path``. Hardlinks counted once."""
    total = 0
    files = 0
    seen: set = set()
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if st.st_nlink > 1:
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            # st_blocks is 512-byte units — real disk usage, not apparent
            # size. A node_modules full of tiny files costs far more on disk
            # than its bytes suggest, and that is what the volume sees.
            total += getattr(st, "st_blocks", 0) * 512 or st.st_size
            files += 1
    return total, files


def _listening_ports() -> set:
    """Ports currently accepting connections on loopback.

    Probed rather than read from ``AppManager._running``: this script runs as
    a separate process and has no access to the agent's in-memory state, and
    a stale in-memory record is exactly the case where we would delete under
    a live server.
    """
    live = set()
    for lo, hi in (METRO_PORT_RANGE, WEB_PORT_RANGE):
        for port in range(lo, hi + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.02)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    live.add(port)
    return live


@dataclass
class AppReport:
    name: str
    path: str
    total_bytes: int = 0
    total_files: int = 0
    reclaimable_bytes: int = 0
    reclaimable_files: int = 0
    targets: List[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "app": self.name,
            "path": self.path,
            "total_bytes": self.total_bytes,
            "total_files": self.total_files,
            "reclaimable_bytes": self.reclaimable_bytes,
            "reclaimable_files": self.reclaimable_files,
            "targets": self.targets,
            "skipped": self.skipped_reason,
        }


def _jailed(root: str, candidate: str) -> bool:
    real_root = os.path.realpath(root)
    real = os.path.realpath(candidate)
    return real != real_root and real.startswith(real_root + os.sep)


def survey(apps_root: str, *, include_running: bool, keep_newest: int) -> List[AppReport]:
    if not os.path.isdir(apps_root):
        return []

    entries = [
        e for e in sorted(os.listdir(apps_root))
        if os.path.isdir(os.path.join(apps_root, e)) and not e.startswith(".")
    ]
    protected = set()
    if keep_newest > 0:
        by_mtime = sorted(
            entries,
            key=lambda e: os.path.getmtime(os.path.join(apps_root, e)),
            reverse=True,
        )
        protected = set(by_mtime[:keep_newest])

    running_ports = set() if include_running else _listening_ports()
    reports: List[AppReport] = []

    for name in entries:
        path = os.path.join(apps_root, name)
        rep = AppReport(name=name, path=path)
        rep.total_bytes, rep.total_files = dir_size(path)

        if name in protected:
            rep.skipped_reason = f"protected by --keep-newest {keep_newest}"
            reports.append(rep)
            continue

        if running_ports and _app_looks_running(path, running_ports):
            rep.skipped_reason = "a dev server is listening for this app"
            reports.append(rep)
            continue

        for target_name in REGENERABLE:
            target = os.path.join(path, target_name)
            if not os.path.isdir(target) or os.path.islink(target):
                continue
            if not _jailed(apps_root, target):
                rep.skipped_reason = f"{target_name} escapes the apps root"
                continue
            b, f = dir_size(target)
            rep.reclaimable_bytes += b
            rep.reclaimable_files += f
            rep.targets.append(target)
        reports.append(rep)
    return reports


def _app_looks_running(app_path: str, running_ports: set) -> bool:
    """Conservative: if we cannot tell, assume it IS running.

    There is no per-app port record on disk, so any live dev-server port
    makes every app a candidate. Deleting `node_modules` under a running
    Metro is a crash the user sees; keeping 452 MiB one more cycle is not.
    """
    return bool(running_ports)


def metro_tmp_caches() -> List[str]:
    base = tempfile.gettempdir()
    return [p for p in glob.glob(os.path.join(base, "metro-*")) if os.path.isdir(p)]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps-root",
                    default=os.environ.get("TOUP_APPS_DIR", "/opt/toup-agent/apps"),
                    help="Expo apps root (default: $TOUP_APPS_DIR or /opt/toup-agent/apps)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete. Without this, nothing is removed.")
    ap.add_argument("--include-running", action="store_true",
                    help="do not skip apps while a dev server is listening")
    ap.add_argument("--keep-newest", type=int, default=0,
                    help="protect the N most recently modified apps")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.apps_root))
    reports = survey(root, include_running=args.include_running,
                     keep_newest=args.keep_newest)
    caches = metro_tmp_caches()
    cache_bytes = sum(dir_size(c)[0] for c in caches)
    cache_files = sum(dir_size(c)[1] for c in caches)

    total_before = sum(r.total_bytes for r in reports)
    reclaimable = sum(r.reclaimable_bytes for r in reports) + cache_bytes
    reclaim_files = sum(r.reclaimable_files for r in reports) + cache_files

    removed_bytes = 0
    removed_files = 0
    errors: List[str] = []
    if args.apply:
        for rep in reports:
            for target in rep.targets:
                b, f = dir_size(target)
                try:
                    shutil.rmtree(target)
                    removed_bytes += b
                    removed_files += f
                except OSError as exc:
                    errors.append(f"{target}: {exc}")
        for cache in caches:
            b, f = dir_size(cache)
            try:
                shutil.rmtree(cache)
                removed_bytes += b
                removed_files += f
            except OSError as exc:
                errors.append(f"{cache}: {exc}")

    payload = {
        "apps_root": root,
        "apps": [r.to_dict() for r in reports],
        "metro_tmp_caches": caches,
        "metro_tmp_cache_bytes": cache_bytes,
        "total_bytes_before": total_before,
        "reclaimable_bytes": reclaimable,
        "reclaimable_files": reclaim_files,
        "applied": bool(args.apply),
        "removed_bytes": removed_bytes,
        "removed_files": removed_files,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 1 if errors else 0

    print(f"Expo apps root: {root}")
    if not reports:
        print("  no Expo app directories found — nothing to reclaim")
    for rep in reports:
        line = (f"  {rep.name:<32} {human(rep.total_bytes):>12} total, "
                f"{rep.total_files:>7,} files")
        if rep.skipped_reason:
            print(line + f"  — SKIPPED ({rep.skipped_reason})")
        else:
            print(line + f"  → reclaimable {human(rep.reclaimable_bytes)} "
                         f"({rep.reclaimable_files:,} files)")
    if caches:
        print(f"  Metro tmp caches: {len(caches)} dir(s), {human(cache_bytes)}")

    print()
    print(f"Total on disk now : {human(total_before)}")
    print(f"Reclaimable       : {human(reclaimable)}  ({reclaim_files:,} files)")
    if args.apply:
        print(f"REMOVED           : {human(removed_bytes)}  ({removed_files:,} files)")
        for e in errors:
            print(f"  error: {e}", file=sys.stderr)
    else:
        print("Dry run — nothing was deleted. Re-run with --apply to reclaim.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
