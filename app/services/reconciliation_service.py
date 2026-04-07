"""
Workspace reconciliation — keeps the DB (apps table) in sync with the VPS filesystem.

Three trigger modes:
  - lazy:  called on GET /apps/ (page load) — fast, cached per user
  - sweep: background task every 5-10 min for active users
  - manual: explicit API call

Scans two directories on the VPS:
  - TOUP_APPS_DIR  (/app/workspace/apps/)     — App Builder output
  - WORKSPACE/vibecoding/                       — Vibecoding output

For each directory found on disk without a matching DB row, creates an App
record with source="filesystem_discovered" and status="untracked".

For each DB row whose directory has disappeared, marks status="orphaned"
(never hard-deletes — transient SSH failures must not wipe user data).
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from fnmatch import fnmatch
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import App, BuildJob, ReconciliationLog

logger = logging.getLogger(__name__)

# Configurable ignore patterns — dirs matching any of these are skipped.
# Checked against the directory *name* (not the full path).
WORKSPACE_IGNORE_PATTERNS = [
    "dashboard",     # has its own dedicated page
    ".*",            # hidden folders (.git, .cache, etc.)
    "node_modules",
    "__pycache__",
]

# Per-user cooldown: don't re-reconcile within this many seconds (lazy mode)
_LAZY_COOLDOWN_SECONDS = 30
_last_reconcile: Dict[str, float] = {}  # user_id -> timestamp


def _should_ignore(dirname: str) -> bool:
    """Check if a directory name matches any ignore pattern."""
    for pattern in WORKSPACE_IGNORE_PATTERNS:
        if fnmatch(dirname, pattern):
            return True
    return False


def _display_name_from_slug(slug: str) -> str:
    """Convert a slug or folder name into a human-readable display name."""
    # UUID-like names get a generic label
    uuid_pattern = re.compile(r'^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$', re.I)
    if uuid_pattern.match(slug):
        return f"Project {slug[:8]}"
    # Short hex prefixes (e.g., 8e3521f6)
    if re.match(r'^[0-9a-f]{6,}$', slug, re.I) and len(slug) <= 36:
        return f"Project {slug[:8]}"
    # Normal slugs: replace hyphens/underscores with spaces, title-case
    return slug.replace("-", " ").replace("_", " ").title()


async def _list_dirs_on_vps(base_path: str, ssh_cmd_fn) -> List[str]:
    """List immediate subdirectories at base_path on VPS via SSH.

    Returns list of directory names (not full paths).
    ssh_cmd_fn should be an async function that takes a shell command string
    and returns stdout as a string.
    """
    raw = await ssh_cmd_fn(
        f"find {base_path} -maxdepth 1 -mindepth 1 -type d -printf '%f\\n' 2>/dev/null | head -500"
    )
    if not raw or not raw.strip():
        return []
    return [d.strip() for d in raw.strip().split("\n") if d.strip()]


async def reconcile_apps(
    user_id: str,
    db: AsyncSession,
    ssh_cmd_fn,
    *,
    trigger: str = "lazy",
    apps_dir: str = "/app/workspace/apps",
    workspace_dir: str = "/app/workspace",
    force: bool = False,
) -> Dict:
    """Run a reconciliation pass for a single user.

    Args:
        user_id: The user whose apps to reconcile.
        db: Active async DB session.
        ssh_cmd_fn: Async callable(cmd: str) -> str for running VPS commands.
        trigger: "lazy", "sweep", or "manual".
        apps_dir: Path to the App Builder output directory on the VPS.
        workspace_dir: Path to the workspace root on the VPS.
        force: Skip cooldown check.

    Returns:
        Dict with summary: created, orphaned, excluded counts.
    """
    # Cooldown check for lazy mode
    if not force and trigger == "lazy":
        last = _last_reconcile.get(user_id, 0)
        if time.time() - last < _LAZY_COOLDOWN_SECONDS:
            return {"skipped": True, "reason": "cooldown"}

    start_time = time.time()

    # 1. List directories on disk in both scan targets
    vibecoding_dir = f"{workspace_dir}/vibecoding"
    scan_targets = {
        apps_dir: "app_builder",
        vibecoding_dir: "vibecoding",
    }

    disk_dirs: Dict[str, str] = {}  # full_path -> source_hint
    excluded_dirs: List[str] = []

    for scan_dir, source_hint in scan_targets.items():
        dir_names = await _list_dirs_on_vps(scan_dir, ssh_cmd_fn)
        for dname in dir_names:
            if _should_ignore(dname):
                excluded_dirs.append(f"{scan_dir}/{dname}")
                logger.debug("[RECONCILE] Excluded by ignore pattern: %s/%s", scan_dir, dname)
                continue
            full_path = f"{scan_dir}/{dname}"
            disk_dirs[full_path] = source_hint

    # 2. Load all App records for this user
    result = await db.execute(
        select(App).where(App.user_id == user_id)
    )
    db_apps = result.scalars().all()
    db_app_dirs: Dict[str, App] = {app.app_dir: app for app in db_apps}
    db_slugs: Set[str] = {app.slug for app in db_apps}

    # 3. Find orphaned filesystem dirs → create new App records
    created_apps: List[Dict] = []
    for full_path, source_hint in disk_dirs.items():
        if full_path in db_app_dirs:
            continue  # already tracked

        dirname = full_path.split("/")[-1]

        # Deduplicate slug
        slug = re.sub(r'[^a-zA-Z0-9-]', '-', dirname.lower()).strip('-')[:60] or "app"
        base_slug = slug
        counter = 1
        while slug in db_slugs:
            slug = f"{base_slug}-{counter}"
            counter += 1
        db_slugs.add(slug)

        display_name = _display_name_from_slug(dirname)

        app = App(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=display_name,
            slug=slug,
            status="untracked",
            source="filesystem_discovered",
            app_dir=full_path,
            platforms="web,ios",
        )
        db.add(app)
        created_apps.append({"id": app.id, "name": display_name, "slug": slug, "path": full_path})
        logger.info("[RECONCILE] Auto-created app '%s' from disk path %s", display_name, full_path)

    # 4. Find DB apps whose directories disappeared → mark orphaned
    #    Only check apps that aren't already orphaned/untracked and whose dir
    #    would be under one of our scan targets.
    orphaned_apps: List[Dict] = []
    all_disk_paths = set(disk_dirs.keys())

    for app in db_apps:
        if app.status in ("orphaned", "untracked"):
            continue
        # Only check apps whose app_dir is under one of the scanned directories
        is_in_scan_scope = any(app.app_dir.startswith(sd) for sd in scan_targets)
        if not is_in_scan_scope:
            continue
        if app.app_dir not in all_disk_paths:
            app.status = "orphaned"
            orphaned_apps.append({"id": app.id, "name": app.name, "path": app.app_dir})
            logger.info("[RECONCILE] Marked app '%s' as orphaned (dir missing: %s)", app.name, app.app_dir)

    # 5. Commit changes
    if created_apps or orphaned_apps:
        await db.commit()

    # 6. Write audit log
    duration_ms = int((time.time() - start_time) * 1000)
    log_entry = ReconciliationLog(
        id=str(uuid.uuid4()),
        user_id=user_id,
        trigger=trigger,
        dirs_scanned=json.dumps(list(scan_targets.keys())),
        dirs_on_disk=json.dumps(list(disk_dirs.keys())),
        dirs_in_db=json.dumps(list(db_app_dirs.keys())),
        created_apps=json.dumps(created_apps),
        orphaned_apps=json.dumps(orphaned_apps),
        excluded_dirs=json.dumps(excluded_dirs),
        duration_ms=duration_ms,
    )
    db.add(log_entry)
    await db.commit()

    _last_reconcile[user_id] = time.time()

    summary = {
        "skipped": False,
        "trigger": trigger,
        "created": len(created_apps),
        "orphaned": len(orphaned_apps),
        "excluded": len(excluded_dirs),
        "total_on_disk": len(disk_dirs),
        "total_in_db": len(db_app_dirs),
        "duration_ms": duration_ms,
    }
    logger.info("[RECONCILE] %s pass for user %s: %s", trigger, user_id[:8], summary)
    return summary
