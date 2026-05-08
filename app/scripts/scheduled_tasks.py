"""
Scheduled Tasks for Memory System

This module sets up periodic background tasks for:
1. Memory Decay - Apply Ebbinghaus forgetting curve
2. Memory Consolidation - Promote episodic to semantic

Uses APScheduler for in-process scheduling. For production deployments
with multiple workers, consider using Celery or a dedicated job queue.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_maker, User
from app.services.decay_service import get_decay_service
from app.services.consolidation_service import get_consolidation_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("toup.scheduler")

# Global scheduler instance
scheduler: Optional[AsyncIOScheduler] = None


async def run_decay_for_all_users():
    """
    Run memory decay for all active users.
    This task applies the Ebbinghaus forgetting curve to reduce memory strength over time.
    """
    logger.info("Starting scheduled decay task...")
    
    async with async_session_maker() as db:
        # Get all active users
        result = await db.execute(
            select(User.id).where(User.is_active == True)
        )
        user_ids = [row[0] for row in result.fetchall()]
        
        total_processed = 0
        total_updated = 0
        
        for user_id in user_ids:
            try:
                # Create a new session for each user to avoid long transactions
                async with async_session_maker() as user_db:
                    decay_service = get_decay_service(user_db)
                    processed, updated = await decay_service.apply_decay_to_user(user_id)
                    total_processed += processed
                    total_updated += updated
            except Exception as e:
                logger.error(f"Error running decay for user {user_id}: {e}")
        
        logger.info(
            f"Decay task complete: {total_updated} of {total_processed} memories updated "
            f"across {len(user_ids)} users"
        )


async def run_consolidation_for_all_users():
    """
    Run memory consolidation for all active users.
    This task groups similar episodic memories into semantic memories.
    """
    logger.info("Starting scheduled consolidation task...")
    
    async with async_session_maker() as db:
        # Get all active users
        result = await db.execute(
            select(User.id).where(User.is_active == True)
        )
        user_ids = [row[0] for row in result.fetchall()]
        
        total_considered = 0
        total_groups = 0
        total_consolidated = 0
        
        for user_id in user_ids:
            try:
                async with async_session_maker() as user_db:
                    consolidation_service = get_consolidation_service(user_db)
                    considered, groups, consolidated = await consolidation_service.run_consolidation(user_id)
                    total_considered += considered
                    total_groups += groups
                    total_consolidated += consolidated
            except Exception as e:
                logger.error(f"Error running consolidation for user {user_id}: {e}")
        
        logger.info(
            f"Consolidation task complete: {total_consolidated} memories consolidated "
            f"into {total_groups} groups from {total_considered} candidates "
            f"across {len(user_ids)} users"
        )


async def run_health_check():
    """
    Periodic health check and logging of memory system stats.
    """
    logger.info("Running memory system health check...")
    
    async with async_session_maker() as db:
        from sqlalchemy import func, case
        from app.db import Memory
        
        result = await db.execute(
            select(
                func.count(Memory.id).label("total"),
                func.avg(Memory.strength).label("avg_strength"),
                func.sum(case((Memory.strength < 0.3, 1), else_=0)).label("weak_count"),
            ).where(Memory.is_deleted == False)
        )
        row = result.first()
        
        logger.info(
            f"Memory Health: {row.total} total memories, "
            f"avg strength: {row.avg_strength:.2f}, "
            f"weak memories (<0.3): {row.weak_count}"
        )


async def run_retrieval_feedback_analysis():
    """
    Weekly retrieval quality analysis for the self-improvement feedback loop (Phase 5).
    Analyzes retrieval events, generates quality reports, and logs improvement suggestions.
    """
    logger.info("Starting weekly retrieval feedback analysis...")
    
    async with async_session_maker() as db:
        result = await db.execute(
            select(User.id).where(User.is_active == True)
        )
        user_ids = [row[0] for row in result.fetchall()]
    
    for user_id in user_ids:
        try:
            async with async_session_maker() as user_db:
                from app.services.retrieval_feedback import get_retrieval_feedback
                feedback = get_retrieval_feedback(user_db)
                report = await feedback.generate_weekly_report(user_id)
                
                grade = report.get("health_grade", "N/A")
                metrics = report.get("metrics", {})
                suggestions = report.get("suggestions", [])
                
                logger.info(
                    f"Feedback report for user {user_id}: "
                    f"grade={grade}, events={metrics.get('total_events', 0)}, "
                    f"recall={metrics.get('recall_estimate', 0):.1%}, "
                    f"precision={metrics.get('precision_estimate', 0):.1%}, "
                    f"suggestions={len(suggestions)}"
                )
        except Exception as e:
            logger.error(f"Error running feedback analysis for user {user_id}: {e}")
    
    logger.info(f"Retrieval feedback analysis complete for {len(user_ids)} users")


async def run_retrieval_feedback_analysis():
    """
    Weekly retrieval quality analysis for the self-improvement feedback loop (Phase 5).
    Analyzes retrieval events, generates quality reports, and logs improvement suggestions.
    """
    logger.info("Starting weekly retrieval feedback analysis...")
    
    async with async_session_maker() as db:
        result = await db.execute(
            select(User.id).where(User.is_active == True)
        )
        user_ids = [row[0] for row in result.fetchall()]
    
    for user_id in user_ids:
        try:
            async with async_session_maker() as user_db:
                from app.services.retrieval_feedback import get_retrieval_feedback
                feedback = get_retrieval_feedback(user_db)
                report = await feedback.generate_weekly_report(user_id)
                
                grade = report.get("health_grade", "N/A")
                metrics = report.get("metrics", {})
                suggestions = report.get("suggestions", [])
                
                logger.info(
                    f"Feedback report for user {user_id}: "
                    f"grade={grade}, events={metrics.get('total_events', 0)}, "
                    f"recall={metrics.get('recall_estimate', 0):.1%}, "
                    f"precision={metrics.get('precision_estimate', 0):.1%}, "
                    f"suggestions={len(suggestions)}"
                )
        except Exception as e:
            logger.error(f"Error running feedback analysis for user {user_id}: {e}")
    
    logger.info(f"Retrieval feedback analysis complete for {len(user_ids)} users")


# Process-level lock guarding the archival pass. Prevents a second invocation
# from starting while the previous one is still running (e.g. if an hourly run
# is slow). Multi-worker deployments would need a DB advisory lock; a single
# AsyncIOScheduler in a single process is the only call site today.
_ARCHIVAL_LOCK: Optional["asyncio.Lock"] = None


def _archival_lock() -> "asyncio.Lock":
    """Return a singleton asyncio.Lock for the archival pass.

    Created lazily on first call because module-load time may precede the
    event loop in some deployment contexts.
    """
    import asyncio as _asyncio
    global _ARCHIVAL_LOCK
    if _ARCHIVAL_LOCK is None:
        _ARCHIVAL_LOCK = _asyncio.Lock()
    return _ARCHIVAL_LOCK


# How long a DayChat may sit in archival_summary_status='pending' before the
# next run considers it stuck and re-attempts. Set large enough that slow Haiku
# calls aren't clobbered, small enough that a crashed worker's zombie row
# recovers within the next job cycle.
_STUCK_PENDING_MAX_AGE_HOURS = 2


async def run_end_of_day_archival():
    """Generate archival summaries for days that have rolled over in each user's local timezone.

    Runs hourly. Single-instance via a process-level asyncio lock; a second
    invocation while the previous is still running returns immediately.

    For each user:
      1. Compute the user's local today (IANA timezone-aware; DST-safe).
      2. Find their DayChat rows where local_date < local_today AND
         archival_summary_status IN ('not_needed', 'pending').
         - 'pending' rows older than _STUCK_PENDING_MAX_AGE_HOURS are treated
           as stuck (crashed worker, API timeout) and re-attempted.
         - 'failed' rows are NEVER auto-retried (requires admin reset to avoid
           tight-loop cost on a systematically-bad day).
      3. Skip if the DayChat has zero user/assistant messages.
      4. Atomically claim the row (UPDATE ... WHERE current_status = ...) so
         concurrent workers can't double-summarize the same day.
      5. Call generate_archival_summary (system.day_archival-tagged).
      6. Commit archival_summary + archival_summary_generated_at + status.

    Idempotent: on re-run, already-up_to_date days are skipped. The per-call cost
    is tagged operation_type="system.day_archival" by internal_llm so it lands in
    cost dashboards but never touches user budget caps.

    Feature-flagged on settings.enable_day_recall; the scheduler skips scheduling
    this job entirely when the flag is off.
    """
    import zoneinfo
    from datetime import datetime as dt, timezone as tz, timedelta as td
    from sqlalchemy import and_, or_, update

    from app.db import async_session_maker, User
    from app.db.models.day_chat import DayChat
    from app.db.models import Message
    from app.services.day_summarizer import generate_archival_summary

    lock = _archival_lock()
    if lock.locked():
        logger.info("[archival] Skipping run — previous invocation still in flight.")
        return

    async with lock:
        started_at = dt.now(tz.utc)
        logger.info("[archival] Starting end-of-day archival pass at %s", started_at.isoformat())

        # Enumerate active users once.
        try:
            async with async_session_maker() as db:
                result = await db.execute(
                    select(User.id, User.timezone).where(User.is_active == True)  # noqa: E712
                )
                users = [(row[0], row[1]) for row in result.fetchall()]
        except Exception as e:
            logger.error("[archival] Failed to enumerate users: %s", e, exc_info=True)
            return

        # Naive UTC to match the `archival_summary_generated_at` column which
        # is DateTime (no tz) per the day_chats model. asyncpg refuses to
        # compare tz-aware vs tz-naive — a real production bug caught here.
        stuck_cutoff = (dt.now(tz.utc) - td(hours=_STUCK_PENDING_MAX_AGE_HOURS)).replace(tzinfo=None)

        processed = 0
        claimed = 0
        succeeded = 0
        failed = 0
        skipped_empty = 0
        skipped_raced = 0

        for user_id, user_tz_name in users:
            try:
                # Compute local today for this user with explicit fallback log.
                if user_tz_name:
                    try:
                        local_today = dt.now(zoneinfo.ZoneInfo(user_tz_name)).date()
                    except Exception as tz_err:
                        logger.warning(
                            "[archival] user=%s invalid timezone %r (%s), falling back to UTC",
                            user_id[:8], user_tz_name, tz_err,
                        )
                        local_today = dt.now(tz.utc).date()
                else:
                    local_today = dt.now(tz.utc).date()

                async with async_session_maker() as udb:
                    # Find candidates. 'pending' only counts as a candidate if stale
                    # (older than _STUCK_PENDING_MAX_AGE_HOURS) — otherwise another
                    # worker is actively processing it.
                    rows = (await udb.execute(
                        select(DayChat).where(
                            and_(
                                DayChat.user_id == user_id,
                                DayChat.local_date < local_today,
                                or_(
                                    DayChat.archival_summary_status == "not_needed",
                                    and_(
                                        DayChat.archival_summary_status == "pending",
                                        or_(
                                            DayChat.archival_summary_generated_at.is_(None),
                                            DayChat.archival_summary_generated_at < stuck_cutoff,
                                        ),
                                    ),
                                ),
                            )
                        ).order_by(DayChat.local_date.asc())
                    )).scalars().all()

                    for dc in rows:
                        processed += 1

                        # Skip days with no user/assistant content.
                        msg_count = (await udb.execute(
                            select(func.count()).select_from(Message).where(
                                and_(
                                    Message.day_chat_id == dc.id,
                                    Message.role.in_(["user", "assistant"]),
                                )
                            )
                        )).scalar() or 0
                        if msg_count == 0:
                            skipped_empty += 1
                            continue

                        # Atomic claim: only one worker succeeds in flipping the
                        # status from its current value to 'pending'. If the row
                        # already changed (another worker got there first, or it
                        # was healed), skip without double-processing.
                        expected_status = dc.archival_summary_status
                        # Naive UTC — column has no tz. See comment on stuck_cutoff.
                        claim_now = dt.now(tz.utc).replace(tzinfo=None)
                        claim = await udb.execute(
                            update(DayChat)
                            .where(
                                and_(
                                    DayChat.id == dc.id,
                                    DayChat.archival_summary_status == expected_status,
                                )
                            )
                            .values(
                                archival_summary_status="pending",
                                archival_summary_generated_at=claim_now,
                            )
                        )
                        await udb.commit()
                        if (claim.rowcount or 0) == 0:
                            skipped_raced += 1
                            logger.info(
                                "[archival] user=%s date=%s lost claim race, skipping",
                                user_id[:8], dc.local_date.isoformat(),
                            )
                            continue
                        claimed += 1

                        # Generate. Runs in a separate session so a rollback here
                        # doesn't blow up the outer transaction.
                        try:
                            async with async_session_maker() as gen_db:
                                summary = await generate_archival_summary(gen_db, dc.id)
                        except Exception as gen_err:
                            logger.warning(
                                "[archival] generate failed user=%s date=%s: %s",
                                user_id[:8], dc.local_date.isoformat(), gen_err,
                            )
                            summary = None

                        # Commit the outcome.
                        fresh = (await udb.execute(
                            select(DayChat).where(DayChat.id == dc.id)
                        )).scalar_one_or_none()
                        if not fresh:
                            continue
                        if summary:
                            fresh.archival_summary = summary
                            fresh.archival_summary_generated_at = dt.now(tz.utc).replace(tzinfo=None)
                            fresh.archival_summary_status = "up_to_date"
                            succeeded += 1
                        else:
                            fresh.archival_summary_status = "failed"
                            fresh.archival_summary_generated_at = dt.now(tz.utc).replace(tzinfo=None)
                            failed += 1
                        await udb.commit()
            except Exception as e:
                logger.error(
                    "[archival] user=%s failed: %s",
                    (user_id or "-")[:8], e, exc_info=True,
                )
                failed += 1

        duration_s = (dt.now(tz.utc) - started_at).total_seconds()
        logger.info(
            "[archival] complete duration=%.1fs users=%d processed=%d claimed=%d "
            "succeeded=%d failed=%d empty=%d raced=%d",
            duration_s, len(users), processed, claimed,
            succeeded, failed, skipped_empty, skipped_raced,
        )

    logger.info(
        "End-of-day archival complete: processed=%d succeeded=%d failed=%d empty=%d users=%d",
        processed, succeeded, failed, skipped_empty, len(users),
    )


async def run_account_purge():
    """Hard-delete accounts that were anonymized for deletion more than 30 days ago.

    Our account-deletion endpoint (POST /api/auth/delete-account) immediately
    anonymizes PII and deactivates the account — email becomes
    `deleted-<id>@deleted.toup.ai`, is_active is set to False, and
    password_changed_at marks the moment of deletion. This job runs daily and
    removes the underlying user row plus all owned data after 30 days, honoring
    the retention promise made to users in the Privacy Policy.

    Safe by construction: only accounts that match BOTH the deleted-email
    pattern AND is_active=False AND password_changed_at older than 30 days
    are considered, so an active user cannot be caught accidentally.
    """
    from datetime import datetime, timedelta
    from sqlalchemy import inspect, delete as sa_delete
    from app.db import Base

    cutoff = datetime.utcnow() - timedelta(days=30)

    async with async_session_maker() as db:
        result = await db.execute(
            select(User).where(
                User.is_active == False,  # noqa: E712
                User.email.like("deleted-%@deleted.toup.ai"),
                User.password_changed_at.is_not(None),
                User.password_changed_at < cutoff,
            )
        )
        to_purge: list[User] = list(result.scalars().all())

    if not to_purge:
        logger.info("[account_purge] nothing to purge")
        return

    # Discover every table whose rows reference users.id so we stay correct as
    # new models are added. Child tables without a direct user_id (e.g.
    # Message → day_chat_id → DayChat) cascade from their parent below.
    user_owned_tables: list[str] = []
    for table_name, table in Base.metadata.tables.items():
        if "user_id" in table.c:
            user_owned_tables.append(table_name)

    logger.info(
        "[account_purge] purging %d account(s) across %d user-owned table(s)",
        len(to_purge), len(user_owned_tables),
    )

    purged = 0
    failed = 0
    for user in to_purge:
        try:
            async with async_session_maker() as udb:
                # Delete day_chat message rows via the FK from messages →
                # day_chats, then delete all user_owned rows, then the user.
                from app.db.models.day_chat import DayChat
                from app.db.models import Message

                day_chat_ids = [
                    row[0] for row in (
                        await udb.execute(
                            select(DayChat.id).where(DayChat.user_id == user.id)
                        )
                    ).fetchall()
                ]
                if day_chat_ids:
                    await udb.execute(
                        sa_delete(Message).where(Message.day_chat_id.in_(day_chat_ids))
                    )

                for table_name in user_owned_tables:
                    table = Base.metadata.tables[table_name]
                    await udb.execute(
                        sa_delete(table).where(table.c.user_id == user.id)
                    )

                # Finally the user row itself.
                await udb.execute(
                    sa_delete(User.__table__).where(User.__table__.c.id == user.id)
                )
                await udb.commit()
                purged += 1
                logger.info("[account_purge] removed account %s…", user.id[:8])
        except Exception as e:
            failed += 1
            logger.error(
                "[account_purge] failed to purge %s: %s", user.id[:8], e, exc_info=True,
            )

    logger.info(
        "[account_purge] complete — purged=%d failed=%d", purged, failed,
    )


async def run_managed_container_reaper():
    """Destroy managed-tenant containers whose bundle was cancelled
    AND whose user account has been deactivated.

    Phase C of the never-sleep plan softened this from a pure
    "X days after bundle cancellation" reaper into one that ALSO
    requires the user to have flagged their account inactive (or had
    it deactivated by an operator). Reasoning: a user who cancelled
    but might come back next week deserves to skip the cold-start —
    their container costs ~150MB of RAM, cheap insurance against the
    "agent isn't there when I log back in" UX hit.

    Filtering (all four conditions must hold):
      * `User.is_active == False` — the new gate. Without this, any
        cancelled-but-still-logging-in user has their container
        reaped and faces a 15s cold start on next chat.
      * `hosting_mode == 'managed'` — never touches self-hosted
      * `bundle_status == 'cancelled'` — `cancelling` (period not
        over) and `past_due` (could recover) are explicitly skipped
      * `bundle_cancelled_at` is set AND older than the grace window
        (30 days by default — Phase C bumped from 7d) — rows missing
        the timestamp are skipped (will be backfilled on next webhook)

    The `ManagedContainer.status` already-deleted check is preserved.

    Failures of `destroy_container` are logged and counted; the loop
    continues so one stuck tenant doesn't block the rest. Set
    `settings.bundle_cancel_grace_days = 0` to disable the job.

    Phase C-specific telemetry: counts the `kept_alive_cancelled_users`
    population (cancelled bundles whose user is still active) so we
    can alert if that group exceeds 100 — at which point the cost
    ceiling argues for revisiting the policy.
    """
    from datetime import timedelta
    from app.config import settings as _s
    from app.db.models import AgentConfig, User
    from app.db.models.managed_container import ManagedContainer
    from app.services import docker_host_service

    grace_days = int(getattr(_s, "bundle_cancel_grace_days", 30) or 0)
    if grace_days <= 0:
        logger.info("[bundle_reaper] disabled (bundle_cancel_grace_days=%s)", grace_days)
        return

    cutoff = datetime.utcnow() - timedelta(days=grace_days)

    async with async_session_maker() as db:
        # Telemetry pass: count cancelled-but-still-active users.
        # This is the population we INTENTIONALLY don't reap; alert
        # if it grows past 100 (~15GB RAM ceiling on the host).
        kept_alive_q = await db.execute(
            select(AgentConfig.user_id)
            .join(User, AgentConfig.user_id == User.id)
            .where(
                AgentConfig.hosting_mode == "managed",
                AgentConfig.bundle_status == "cancelled",
                User.is_active.is_(True),
            )
        )
        kept_alive_count = len(list(kept_alive_q))
        logger.info(
            "[bundle_reaper] kept_alive_cancelled_users=%d (no-op gate; alert threshold=100)",
            kept_alive_count,
        )
        if kept_alive_count > 100:
            logger.warning(
                "[bundle_reaper] kept_alive_cancelled_users=%d exceeds 100 — "
                "revisit Phase C policy (RAM ceiling).",
                kept_alive_count,
            )

        # Reap pass — BOTH bundle-cancelled AND user-inactive.
        result = await db.execute(
            select(AgentConfig.user_id, AgentConfig.bundle_cancelled_at)
            .join(User, AgentConfig.user_id == User.id)
            .where(
                AgentConfig.hosting_mode == "managed",
                AgentConfig.bundle_status == "cancelled",
                AgentConfig.bundle_cancelled_at.is_not(None),
                AgentConfig.bundle_cancelled_at < cutoff,
                User.is_active.is_(False),
            )
        )
        candidates = list(result.all())

    if not candidates:
        logger.info("[bundle_reaper] no candidates (grace=%dd, cutoff=%s)", grace_days, cutoff.isoformat())
        return

    logger.info("[bundle_reaper] %d candidate(s) past %dd grace", len(candidates), grace_days)

    reaped = 0
    skipped = 0
    failed = 0
    for user_id, cancelled_at in candidates:
        try:
            async with async_session_maker() as cdb:
                cresult = await cdb.execute(
                    select(ManagedContainer).where(ManagedContainer.user_id == user_id)
                )
                container = cresult.scalar_one_or_none()
                if not container:
                    skipped += 1
                    continue
                if container.status in ("deleted", "destroyed"):
                    skipped += 1
                    continue

                ok = await docker_host_service.destroy_container(cdb, user_id)
                if ok:
                    reaped += 1
                    logger.info(
                        "[bundle_reaper] destroyed container user=%s cancelled_at=%s",
                        user_id[:8], cancelled_at.isoformat() if cancelled_at else "?",
                    )
                else:
                    failed += 1
        except Exception as e:
            failed += 1
            logger.error(
                "[bundle_reaper] destroy failed user=%s err=%s",
                user_id[:8], e, exc_info=True,
            )

    logger.info(
        "[bundle_reaper] complete reaped=%d skipped=%d failed=%d",
        reaped, skipped, failed,
    )


async def run_abandoned_onboarding_reaper():
    """Destroy managed-tenant containers belonging to users who created
    an account, kicked off the prewarm-on-Soul.save trigger, and then
    never finished onboarding.

    Per docs/onboarding/prewarm-phase0.md (Phase 2). Closes the
    abandoned-onboarding cost leak that signup-time / Soul.save-time
    pre-warming creates: every user who fills Soul gets a container,
    but a user who never reaches Install or Chat leaves that container
    running indefinitely until manual cleanup.

    Filter (all conditions must hold):
      * `agent_configs.created_at < now() - abandoned_onboarding_grace_days`
      * `agent_configs.onboarding_completed = false`
      * `bundle_status` in ('none', null) — paying users are never reaped here
        (the bundle_reaper above owns the post-cancellation path)
      * `ManagedContainer` row exists AND status not in ('deleted','destroyed')

    NOTE on the day_chats engagement signal:
      The original Phase-2 spec also gated on "no day_chats rows for
      this user." day_chats lives in the per-tenant agent DB
      (AGENT_ONLY_TABLES at backend/app/db/models/base.py:56), not the
      platform DB this reaper queries. Cross-DB lookup from the reaper
      would require per-tenant connection issuance (architecturally
      heavy, no precedent in scheduled_tasks.py). We rely instead on
      `onboarding_completed=False` which is strictly stronger for the
      abandoned-user case — a user who finished onboarding has by
      definition reached past Install, regardless of message count.

    Behavior on reap:
      * Destroy the container via existing teardown path (bridge DELETE)
      * KEEP the agent_configs row so a returning user re-onboards
        cleanly with a fresh container provision
      * Never touches the User row; account survives

    Set `settings.abandoned_onboarding_grace_days = 0` to disable.
    """
    from datetime import timedelta
    from app.config import settings as _s
    from app.db.models import AgentConfig
    from app.db.models.platform import ManagedContainer
    from app.services import docker_host_service

    grace_days = int(getattr(_s, "abandoned_onboarding_grace_days", 14) or 0)
    if grace_days <= 0:
        logger.info(
            "[abandoned_onboarding_reaper] disabled (grace_days=%s)", grace_days,
        )
        return

    cutoff = datetime.utcnow() - timedelta(days=grace_days)

    async with async_session_maker() as db:
        result = await db.execute(
            select(
                AgentConfig.user_id,
                AgentConfig.created_at,
                AgentConfig.bundle_status,
            ).where(
                AgentConfig.hosting_mode == "managed",
                AgentConfig.onboarding_completed == False,  # noqa: E712 — sqlalchemy
                AgentConfig.created_at < cutoff,
                # Bundle gate: explicitly skip any paying state. The
                # bundle_reaper owns post-cancellation cleanup; here we
                # only touch users who never engaged.
                AgentConfig.bundle_status.in_(("none", None)),
            )
        )
        candidates = list(result.all())

    if not candidates:
        logger.info(
            "[abandoned_onboarding_reaper] no candidates (grace=%dd, cutoff=%s)",
            grace_days, cutoff.isoformat(),
        )
        return

    logger.info(
        "[abandoned_onboarding_reaper] %d candidate(s) past %dd grace",
        len(candidates), grace_days,
    )

    reaped = 0
    skipped = 0
    failed = 0
    for user_id, created_at, bundle_status in candidates:
        try:
            async with async_session_maker() as cdb:
                container = (
                    await cdb.execute(
                        select(ManagedContainer).where(
                            ManagedContainer.user_id == user_id
                        )
                    )
                ).scalar_one_or_none()
                if not container:
                    skipped += 1
                    continue
                if container.status in ("deleted", "destroyed"):
                    skipped += 1
                    continue

                ok = await docker_host_service.destroy_container(cdb, user_id)
                if ok:
                    reaped += 1
                    logger.info(
                        "[abandoned_onboarding_reaper] destroyed container "
                        "user=%s created_at=%s bundle_status=%s",
                        str(user_id)[:8],
                        created_at.isoformat() if created_at else "?",
                        bundle_status,
                    )
                else:
                    failed += 1
        except Exception as e:
            failed += 1
            logger.error(
                "[abandoned_onboarding_reaper] destroy failed user=%s err=%s",
                str(user_id)[:8], e, exc_info=True,
            )

    logger.info(
        "[abandoned_onboarding_reaper] complete reaped=%d skipped=%d failed=%d",
        reaped, skipped, failed,
    )


def setup_scheduler(
    decay_interval_hours: int = 6,
    consolidation_interval_hours: int = 24,
    health_check_interval_minutes: int = 60
) -> AsyncIOScheduler:
    """
    Set up the APScheduler with memory maintenance tasks.
    
    Args:
        decay_interval_hours: How often to run decay (default: every 6 hours)
        consolidation_interval_hours: How often to run consolidation (default: daily)
        health_check_interval_minutes: How often to run health check (default: hourly)
    
    Returns:
        Configured scheduler instance
    """
    global scheduler
    
    scheduler = AsyncIOScheduler()
    
    # Memory Decay - runs every N hours
    scheduler.add_job(
        run_decay_for_all_users,
        trigger=IntervalTrigger(hours=decay_interval_hours),
        id="memory_decay",
        name="Memory Decay (Ebbinghaus Curve)",
        replace_existing=True,
    )
    
    # Memory Consolidation - runs daily at 3 AM
    scheduler.add_job(
        run_consolidation_for_all_users,
        trigger=CronTrigger(hour=3, minute=0),  # 3:00 AM
        id="memory_consolidation",
        name="Memory Consolidation (Episodic→Semantic)",
        replace_existing=True,
    )
    
    # Health Check - runs every hour
    scheduler.add_job(
        run_health_check,
        trigger=IntervalTrigger(minutes=health_check_interval_minutes),
        id="health_check",
        name="Memory Health Check",
        replace_existing=True,
    )
    
    # Phase 5: Retrieval Feedback Analysis — runs weekly (Sundays at 4 AM)
    scheduler.add_job(
        run_retrieval_feedback_analysis,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=0),
        id="retrieval_feedback_analysis",
        name="Retrieval Quality Analysis (Weekly)",
        replace_existing=True,
    )

    # Account Purge — runs daily at 4:30 AM, hard-deletes accounts that were
    # anonymized >30 days ago. Honors the Privacy Policy retention promise.
    scheduler.add_job(
        run_account_purge,
        trigger=CronTrigger(hour=4, minute=30),
        id="account_purge",
        name="Account Purge (>30d after deletion)",
        replace_existing=True,
    )

    # Managed-tenant container reaper — daily at 5:00 AM. Destroys
    # containers whose bundle was cancelled longer than
    # settings.bundle_cancel_grace_days (default 7) ago. Self-disables
    # when the grace setting is 0; safe to run on every deployment.
    scheduler.add_job(
        run_managed_container_reaper,
        trigger=CronTrigger(hour=5, minute=0),
        id="bundle_container_reaper",
        name="Managed Container Reaper (post-cancel grace)",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # Phase-2 abandoned-onboarding reaper — destroys containers belonging
    # to users who created an account, fired the Soul.save prewarm, and
    # never finished onboarding. 14d grace by default. Runs at 5:30 UTC
    # so it doesn't collide with the bundle reaper at 5:00 UTC; both
    # tap the bridge DELETE path so staggering minimises rate-limit
    # risk. Self-disables when grace setting is 0.
    scheduler.add_job(
        run_abandoned_onboarding_reaper,
        trigger=CronTrigger(hour=5, minute=30),
        id="abandoned_onboarding_reaper",
        name="Abandoned Onboarding Reaper (Phase-2 prewarm cost cap)",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # Day-as-Chat archival — runs hourly, gated on feature flag. Not timezone-bound:
    # each run iterates every user and compares THEIR local today to their DayChat
    # local_date values. A day rolling over in Asia/Tokyo will be caught within the
    # hour; same for America/Los_Angeles. Cheaper than per-timezone cron entries.
    try:
        from app.config import settings as _s
        if getattr(_s, "enable_day_recall", False):
            scheduler.add_job(
                run_end_of_day_archival,
                trigger=IntervalTrigger(hours=1),
                id="day_archival",
                name="End-of-Day Archival Summaries",
                replace_existing=True,
            )
    except Exception as e:
        logger.warning("Could not register day_archival job: %s", e)

    logger.info(
        f"Scheduler configured: decay every {decay_interval_hours}h, "
        f"consolidation daily at 3AM, "
        f"health check every {health_check_interval_minutes}min, "
        f"feedback analysis weekly (Sun 4AM)"
    )

    return scheduler


def start_scheduler():
    """Start the scheduler if not already running."""
    global scheduler
    
    if scheduler is None:
        scheduler = setup_scheduler()
    
    if not scheduler.running:
        scheduler.start()
        logger.info("Memory maintenance scheduler started")


def stop_scheduler():
    """Stop the scheduler if running."""
    global scheduler

    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("Memory maintenance scheduler stopped")


def schedule_one_shot(func, kwargs: dict, job_id: str, delay_s: int = 0):
    """Fire a one-shot async job on the shared scheduler.

    Used by the rollout service to kick off background rollout execution
    without holding the HTTP handler's request context. The job runs once
    then is auto-removed; we tag it with job_id so callers can query/cancel.

    Starts the scheduler on first use if it's not yet running (e.g. in test
    or single-worker contexts where start_scheduler() hasn't been invoked).
    """
    from datetime import datetime, timedelta
    from apscheduler.triggers.date import DateTrigger
    global scheduler
    if scheduler is None:
        scheduler = setup_scheduler()
    if not scheduler.running:
        scheduler.start()
    run_at = datetime.utcnow() + timedelta(seconds=delay_s)
    scheduler.add_job(
        func,
        trigger=DateTrigger(run_date=run_at),
        kwargs=kwargs,
        id=job_id,
        replace_existing=True,
        misfire_grace_time=300,  # if the executor is busy, still run up to 5 min late
    )
    logger.info(f"[SCHED] one-shot job {job_id} scheduled for {run_at.isoformat()}Z")


# Optional: Run scheduled tasks manually for testing
if __name__ == "__main__":
    async def main():
        print("Running scheduled tasks manually...")
        
        print("\n1. Running decay...")
        await run_decay_for_all_users()
        
        print("\n2. Running consolidation...")
        await run_consolidation_for_all_users()
        
        print("\n3. Running health check...")
        await run_health_check()
        
        print("\n4. Running retrieval feedback analysis...")
        await run_retrieval_feedback_analysis()
        
        print("\nDone!")
    
    asyncio.run(main())
