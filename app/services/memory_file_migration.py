"""Memory-file maintenance: the one-time migration AND the nightly upkeep.

One code path on purpose (docs/memory/rebuild-2026-08.md §3.7). Per user:

1. organize   — deterministic, LLM-free: every active row gets a file_slug,
                the system file rows exist (MemoryFileService.organize_user).
2. normalize  — legacy working rows repaired: standing arrangements lose
                their lease, unleased one-offs gain one anchored at their
                last sign of life (active_task_service.normalize_working_leases),
                then the expiry sweep archives what has already run out —
                with DECAYED events, nothing deleted.
3. curate     — per-file LLM consolidation under the strict ops contract
                (memory_file_ops.curate_file): duplicates merge, voice
                normalizes to second person, noise archives. Sources keep
                superseded_by/merged_from pointers — fully reversible.

A file is curated when it has never been curated and holds ≥2 entries, or
when it changed and its last curation is over a week old. So the first run
after this ships IS the migration, and every nightly run after it is the
upkeep — there is no separate flag-day script to get wrong. An LLM failure
skips that file and the next run retries it; steps 1–2 still leave the
tenant organized.

Replaces the additive ConsolidationService job on the agent scheduler: that
one wrote a NEW semantic row and retired nothing (the 2026-08-10 dry run
recorded "0 rows rewritten or retired"), which is half of how 89 unmerged
factlets happened.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)

# Curation policy.
CURATE_MIN_ENTRIES = 2
RECURATE_AFTER_DAYS = 7

# The whole pass is single-flighted: the boot one-shot and the nightly cron
# can overlap when curation is slow (it holds LLM calls), and two concurrent
# passes would race the same rows.
_maintenance_lock = asyncio.Lock()


async def _user_api_key(db, user_id: str) -> Optional[str]:
    try:
        from app.db import AgentConfig
        result = await db.execute(
            select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
        )
        return result.scalar_one_or_none()
    except Exception:
        return None


def _needs_curation(f: Dict, now: datetime) -> bool:
    if f["entry_count"] < CURATE_MIN_ENTRIES:
        return False
    if not f.get("consolidated_at"):
        return True
    try:
        consolidated = datetime.fromisoformat(f["consolidated_at"])
        updated = datetime.fromisoformat(f["updated_at"]) if f.get("updated_at") else consolidated
    except Exception:
        return True
    return updated > consolidated and now - consolidated > timedelta(days=RECURATE_AFTER_DAYS)


async def migrate_user_memory_files(
    db, user_id: str, *, consolidate: bool = True
) -> Dict:
    """Organize + normalize + (optionally) curate one user. Own report."""
    from app.services.active_task_service import normalize_working_leases
    from app.services.memory_expiry import expire_stale_memories
    from app.services.memory_file_service import MemoryFileService

    service = MemoryFileService(db)
    report: Dict = {"user_id": user_id}

    organize = await service.organize_user(user_id)
    report["organized"] = organize["assigned"]

    leases = await normalize_working_leases(db, user_id)
    await db.commit()
    report["leases"] = leases

    # Archive what the repaired leases say has already run out — the same
    # sweep the per-turn path runs, so events and semantics are identical.
    try:
        expired_rows = await expire_stale_memories(db, user_id)
        report["expired"] = len(expired_rows)
        await db.commit()
    except Exception as e:
        logger.warning("[memory_files] expiry sweep failed for %s: %s", user_id, e)
        report["expired"] = 0

    report["curated"] = []
    if consolidate:
        from app.services.memory_file_ops import curate_file

        api_key = await _user_api_key(db, user_id)
        listing = await service.list_files(user_id)
        now = datetime.utcnow()
        todo = [
            f for section in listing["sections"] for f in section["files"]
            if _needs_curation(f, now)
        ]
        for f in todo:
            try:
                result = await curate_file(
                    db, user_id, f["slug"], api_key=api_key, mark_consolidated=True,
                )
                report["curated"].append({
                    "slug": f["slug"],
                    "applied": len(result["applied"]),
                    "rejected": len(result["rejected"]),
                })
            except Exception as e:
                # Next run retries this file; the tenant is already organized.
                logger.warning(
                    "[memory_files] curation of %s failed for %s: %s",
                    f["slug"], user_id, e,
                )
                try:
                    await db.rollback()
                except Exception:
                    pass

    logger.info("[memory_files] maintenance user=%s %s", user_id, report)
    return report


async def run_memory_file_maintenance(consolidate: bool = True) -> Dict:
    """All-users pass, one session per user (same shape as
    run_decay_for_all_users). Registered on the agent scheduler as the
    nightly `memory_consolidation` job and fired once shortly after boot so
    an updated tenant migrates without waiting for 03:00."""
    from app.db import User, async_session_maker

    if _maintenance_lock.locked():
        logger.info("[memory_files] maintenance already running — skipped")
        return {"skipped": "already_running"}

    async with _maintenance_lock:
        async with async_session_maker() as db:
            result = await db.execute(select(User.id).where(User.is_active == True))  # noqa: E712
            user_ids = [row[0] for row in result.fetchall()]

        totals = {"users": len(user_ids), "organized": 0, "expired": 0, "curated_files": 0}
        for user_id in user_ids:
            try:
                async with async_session_maker() as user_db:
                    report = await migrate_user_memory_files(
                        user_db, user_id, consolidate=consolidate
                    )
                    totals["organized"] += report.get("organized", 0)
                    totals["expired"] += report.get("expired", 0)
                    totals["curated_files"] += len(report.get("curated", []))
            except Exception as e:
                logger.error("[memory_files] maintenance failed for %s: %s", user_id, e)

        logger.info("[memory_files] maintenance complete: %s", totals)
        return totals
