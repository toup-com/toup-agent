"""
Cron Service — Scheduled tasks for the HexBrain Agent Runtime.

Supports:
- "at" jobs: one-shot at a specific datetime
- "every" jobs: recurring at a fixed interval  
- "cron" jobs: recurring via cron expression

Jobs are stored in the DB (cron_jobs table) and loaded on startup.
Uses APScheduler for scheduling.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.config import settings

logger = logging.getLogger(__name__)

# Interval parsing: "30s", "5m", "2h", "1d"
INTERVAL_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def parse_interval(spec: str) -> Optional[Dict[str, int]]:
    """Parse an interval like '30m', '2h', '1d' into scheduler kwargs."""
    spec = spec.strip().lower()
    for suffix, unit in INTERVAL_UNITS.items():
        if spec.endswith(suffix):
            try:
                value = int(spec[: -len(suffix)])
                return {unit: value}
            except ValueError:
                return None
    return None


def parse_schedule(schedule: str) -> tuple[str, Any]:
    """
    Parse a schedule string and return (kind, trigger).
    
    Accepts:
    - ISO datetime → "at" job
    - Interval like "30m", "2h" → "every" job
    - Cron expression like "*/30 * * * *" → "cron" job
    """
    schedule = schedule.strip()

    # Try ISO datetime first
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(schedule, fmt)
            return "at", DateTrigger(run_date=dt)
        except ValueError:
            continue

    # Try relative time: "in 5m", "in 2h", "in 30s"
    if schedule.startswith("in "):
        interval_kwargs = parse_interval(schedule[3:])
        if interval_kwargs:
            run_at = datetime.utcnow() + timedelta(**interval_kwargs)
            return "at", DateTrigger(run_date=run_at)

    # Try interval: "30m", "2h", "1d"
    interval_kwargs = parse_interval(schedule)
    if interval_kwargs:
        return "every", IntervalTrigger(**interval_kwargs)

    # Try cron expression (5-part: min hour day month weekday)
    parts = schedule.split()
    if len(parts) == 5:
        try:
            trigger = CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
            )
            return "cron", trigger
        except Exception:
            pass

    return "unknown", None


class CronService:
    """
    Manages scheduled jobs for the HexBrain agent.
    
    Jobs are stored in DB and scheduled via APScheduler.
    On startup, loads all enabled jobs from DB.
    The agent can add/remove/list jobs via the cron tool.
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._telegram_bot = None
        self._agent_runner = None
        # In-memory job registry: job_id → {user_id, chat_id, name, ...}
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def set_bot(self, telegram_bot):
        """Set the Telegram bot reference for sending messages."""
        self._telegram_bot = telegram_bot

    def set_agent_runner(self, agent_runner):
        """Set the agent runner for running agent turns on schedule."""
        self._agent_runner = agent_runner

    async def start(self):
        """Start the scheduler and load jobs from DB."""
        self.scheduler.start()
        await self._load_jobs_from_db()
        logger.info("⏰ Cron service started")

    async def stop(self):
        """Shut down the scheduler."""
        self.scheduler.shutdown(wait=False)
        logger.info("⏰ Cron service stopped")

    # ------------------------------------------------------------------
    # Public API (called by tool_executor)
    # ------------------------------------------------------------------

    async def add_job(
        self,
        user_id: str,
        chat_id: int,
        name: str,
        schedule: str,
        message: str,
        wake_event: Optional[str] = None,
        delivery_mode: str = "gateway",
    ) -> Dict[str, Any]:
        """Add a new scheduled job. Returns job info dict."""
        kind, trigger = parse_schedule(schedule)
        if trigger is None:
            return {"error": f"Invalid schedule: '{schedule}'. Use cron expr, interval (30m/2h), ISO datetime, or 'in 5m'."}

        job_id = str(uuid.uuid4())

        # Save to DB
        try:
            from app.db.database import async_session_maker
            from app.db.models import CronJob

            cron_expr = schedule if kind == "cron" else None
            interval_seconds = None
            at_time = None

            if kind == "every":
                interval_kwargs = parse_interval(schedule)
                if interval_kwargs:
                    td = timedelta(**interval_kwargs)
                    interval_seconds = int(td.total_seconds())
            elif kind == "at":
                if isinstance(trigger, DateTrigger):
                    at_time = trigger.run_date

            async with async_session_maker() as db:
                job = CronJob(
                    id=job_id,
                    user_id=user_id,
                    name=name,
                    schedule_kind=kind,
                    schedule_spec=schedule,
                    schedule_cron_expr=cron_expr,
                    schedule_interval_seconds=interval_seconds,
                    schedule_at=at_time,
                    payload_text=message,
                    telegram_chat_id=chat_id,
                    enabled=True,
                )
                db.add(job)
                await db.commit()
        except Exception as e:
            logger.warning(f"Failed to save cron job to DB: {e}")

        # Schedule with APScheduler
        self.scheduler.add_job(
            self._execute_job,
            trigger=trigger,
            id=job_id,
            args=[job_id, user_id, chat_id, name, message],
            replace_existing=True,
        )

        self._jobs[job_id] = {
            "id": job_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "name": name,
            "schedule": schedule,
            "kind": kind,
            "message": message,
            "enabled": True,
            "run_count": 0,
            "wake_event": wake_event,
            "delivery_mode": delivery_mode,
        }

        logger.info(f"⏰ Cron job added: {name} ({kind}: {schedule}) for user {user_id}")
        return {
            "id": job_id,
            "name": name,
            "schedule": schedule,
            "kind": kind,
            "status": "scheduled",
        }

    async def remove_job(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Remove a scheduled job."""
        if job_id not in self._jobs:
            return {"error": f"Job not found: {job_id}"}

        job_info = self._jobs[job_id]
        if job_info["user_id"] != user_id:
            return {"error": "Not your job"}

        # Remove from scheduler
        try:
            self.scheduler.remove_job(job_id)
        except Exception:
            pass

        # Remove from DB
        try:
            from app.db.database import async_session_maker
            from app.db.models import CronJob
            from sqlalchemy import select

            async with async_session_maker() as db:
                result = await db.execute(select(CronJob).where(CronJob.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    await db.delete(job)
                    await db.commit()
        except Exception as e:
            logger.warning(f"Failed to delete cron job from DB: {e}")

        del self._jobs[job_id]
        logger.info(f"⏰ Cron job removed: {job_info['name']}")
        return {"id": job_id, "name": job_info["name"], "status": "removed"}

    async def list_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """List all jobs for a user."""
        return [
            {
                "id": j["id"],
                "name": j["name"],
                "schedule": j["schedule"],
                "kind": j["kind"],
                "enabled": j["enabled"],
                "run_count": j["run_count"],
            }
            for j in self._jobs.values()
            if j["user_id"] == user_id
        ]

    async def run_job_now(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Trigger a job immediately."""
        if job_id not in self._jobs:
            return {"error": f"Job not found: {job_id}"}
        job_info = self._jobs[job_id]
        if job_info["user_id"] != user_id:
            return {"error": "Not your job"}

        asyncio.create_task(
            self._execute_job(
                job_id,
                job_info["user_id"],
                job_info["chat_id"],
                job_info["name"],
                job_info["message"],
            )
        )
        return {"id": job_id, "name": job_info["name"], "status": "triggered"}

    # ------------------------------------------------------------------
    # Wake Events — fire cron jobs based on external triggers
    # ------------------------------------------------------------------

    async def fire_wake_event(self, event_name: str, payload: Optional[Dict] = None):
        """Trigger all cron jobs registered for a given wake event."""
        job_ids = []
        for jid, jinfo in self._jobs.items():
            if jinfo.get("wake_event") == event_name and jinfo.get("enabled", True):
                job_ids.append(jid)

        results = []
        for jid in job_ids:
            jinfo = self._jobs.get(jid)
            if jinfo:
                extra_context = ""
                if payload:
                    import json as _json
                    extra_context = f" [Event payload: {_json.dumps(payload)[:500]}]"
                await self._execute_job(
                    jid, jinfo["user_id"], jinfo["chat_id"],
                    jinfo["name"], jinfo["message"] + extra_context,
                    delivery_mode=jinfo.get("delivery_mode", "gateway"),
                )
                results.append({"job_id": jid, "name": jinfo["name"], "status": "triggered"})
        return results

    VALID_DELIVERY_MODES = ("gateway", "direct", "announce", "silent")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute_job(
        self,
        job_id: str,
        user_id: str,
        chat_id: int,
        name: str,
        message: str,
        delivery_mode: str = "gateway",
    ):
        """Execute a scheduled job — run the agent and send result to user."""
        logger.info(f"⏰ Executing cron job: {name} (id={job_id[:8]})")

        if job_id in self._jobs:
            self._jobs[job_id]["run_count"] += 1

        # Update last_run in DB
        try:
            from app.db.database import async_session_maker
            from app.db.models import CronJob
            from sqlalchemy import select

            async with async_session_maker() as db:
                result = await db.execute(select(CronJob).where(CronJob.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.last_run_at = datetime.utcnow()
                    job.run_count = (job.run_count or 0) + 1
                    await db.commit()
        except Exception:
            pass

        if not self._agent_runner or not self._telegram_bot:
            logger.warning("⏰ Cannot execute cron job: bot or agent runner not set")
            return

        try:
            # Import here to get streaming handler + extraction helpers
            from app.agent.streaming import (
                TelegramStreamHandler,
                extract_reaction,
                extract_buttons,
            )
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            bot = self._telegram_bot.app.bot

            # Create stream handler
            handler = TelegramStreamHandler(chat_id, bot=bot)
            await handler.send_initial(f"⏰ <i>{name}</i>")

            # Run the agent with the job's prompt
            response = await self._agent_runner.run(
                user_message=f"[Scheduled task: {name}] {message}",
                user_id=user_id,
                telegram_chat_id=chat_id,
                on_text_chunk=handler.on_text_chunk,
                on_tool_start=handler.on_tool_start,
                on_tool_end=handler.on_tool_end,
            )

            # Extract reactions and buttons before finalizing
            final_text = response.text
            final_text, _reaction_emoji = extract_reaction(final_text)
            final_text, buttons = extract_buttons(final_text)

            # Build inline keyboard if agent included buttons
            reply_markup = None
            if buttons:
                keyboard = [
                    [InlineKeyboardButton(label, callback_data=cb)]
                    for label, cb in buttons
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

            # Finalize response with buttons attached
            await handler.finalize(final_text, reply_markup=reply_markup)

            logger.info(f"⏰ Cron job completed: {name} ({response.tokens_total} tokens)")

        except Exception as e:
            logger.exception(f"⏰ Cron job failed: {name}")
            try:
                bot = self._telegram_bot.app.bot
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"⏰ Scheduled task <b>{name}</b> failed:\n<code>{str(e)[:200]}</code>",
                    parse_mode="HTML",
                )
            except Exception:
                pass

        # Clean up one-shot jobs
        if job_id in self._jobs and self._jobs[job_id].get("kind") == "at":
            del self._jobs[job_id]
            try:
                from app.db.database import async_session_maker
                from app.db.models import CronJob
                from sqlalchemy import select

                async with async_session_maker() as db:
                    result = await db.execute(select(CronJob).where(CronJob.id == job_id))
                    job = result.scalar_one_or_none()
                    if job:
                        job.enabled = False
                        await db.commit()
            except Exception:
                pass

    async def _load_jobs_from_db(self):
        """Load all enabled cron jobs from DB and schedule them."""
        try:
            from app.db.database import async_session_maker
            from app.db.models import CronJob
            from sqlalchemy import select

            async with async_session_maker() as db:
                result = await db.execute(
                    select(CronJob).where(CronJob.enabled == True)
                )
                jobs = result.scalars().all()

            loaded = 0
            for job in jobs:
                kind, trigger = parse_schedule(job.schedule_spec or "")
                if trigger is None:
                    logger.warning(f"⏰ Skipping job with invalid schedule: {job.name} ({job.schedule_spec})")
                    continue

                # For "at" jobs, skip if already in the past
                if kind == "at" and job.schedule_at and job.schedule_at < datetime.utcnow():
                    continue

                self.scheduler.add_job(
                    self._execute_job,
                    trigger=trigger,
                    id=job.id,
                    args=[job.id, job.user_id, job.telegram_chat_id, job.name, job.payload_text],
                    replace_existing=True,
                )

                self._jobs[job.id] = {
                    "id": job.id,
                    "user_id": job.user_id,
                    "chat_id": job.telegram_chat_id,
                    "name": job.name,
                    "schedule": job.schedule_spec,
                    "kind": kind,
                    "message": job.payload_text,
                    "enabled": True,
                    "run_count": job.run_count or 0,
                }
                loaded += 1

            if loaded:
                logger.info(f"⏰ Loaded {loaded} cron jobs from DB")
        except Exception as e:
            logger.warning(f"⏰ Failed to load cron jobs from DB: {e}")
