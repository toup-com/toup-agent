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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_maker, User
from app.services.decay_service import get_decay_service
from app.services.consolidation_service import get_consolidation_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hexbrain.scheduler")

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
    
    # Phase 5: Retrieval Feedback Analysis — runs weekly (Sundays at 4 AM)
    scheduler.add_job(
        run_retrieval_feedback_analysis,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=0),
        id="retrieval_feedback_analysis",
        name="Retrieval Quality Analysis (Weekly)",
        replace_existing=True,
    )
    
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
