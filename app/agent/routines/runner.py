"""Routine scheduler — sibling of CronService.

Owns its own `AsyncIOScheduler`. Loads `Routine` rows on `start()`, sweeps
orphaned `running` runs from a previous boot, registers a `CronTrigger`
per enabled routine using the user's IANA timezone, and dispatches per
fire to the kind-specific handler.

Gate 1 (this file): full lifecycle + idempotency gate + restart sweep
WITHOUT calling any handler. Per-fire path logs `would_execute` and
exits. Gate 2 wires `KIND_HANDLERS` into the dispatch loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


def _resolve_tz(tz_str: Optional[str], user_id: str):
    """Return a ZoneInfo for tz_str, falling back to UTC with a structured
    log on missing/invalid input. Log shape matches `agent_runner.py:2103`
    (`[agent] tz_fallback source=…`) so existing log-grep tooling sees the
    routine path too.

    Returns (zoneinfo, fellback_to_utc_bool).
    """
    try:
        from zoneinfo import ZoneInfo  # py3.9+
    except ImportError:  # pragma: no cover
        from backports.zoneinfo import ZoneInfo  # type: ignore

    if not tz_str:
        logger.warning(
            "[routine_runner] tz_fallback source=utc_default user=%s — presenting UTC as local",
            user_id,
        )
        return ZoneInfo("UTC"), True
    try:
        return ZoneInfo(tz_str), False
    except Exception:
        logger.warning(
            "[routine_runner] tz_fallback source=invalid_tz user=%s invalid=%r — using UTC",
            user_id, tz_str,
        )
        try:
            return ZoneInfo("UTC"), True
        except Exception:  # pragma: no cover
            raise


def _parse_cron(expr: str, tz) -> Optional[CronTrigger]:
    """Parse a 5-part cron string into a tz-aware CronTrigger. Returns
    None on malformed input (caller logs + skips)."""
    parts = (expr or "").split()
    if len(parts) != 5:
        return None
    try:
        return CronTrigger(
            minute=parts[0], hour=parts[1], day=parts[2],
            month=parts[3], day_of_week=parts[4], timezone=tz,
        )
    except Exception:
        return None


class RoutineRunner:
    """Scheduler + dispatcher for `Routine` rows.

    One instance per agent container. Distinct from `CronService` and
    `HeartbeatService` so a bug in routines can't blow up user-authored
    cron jobs or proactive heartbeats.
    """

    # Heuristic: any `routine_runs` row in 'running' state older than this
    # at boot is treated as orphaned (agent crashed/restarted mid-run).
    ORPHAN_THRESHOLD = timedelta(minutes=10)
    # Bounded wait for in-flight jobs to drain on stop().
    SHUTDOWN_TIMEOUT_SECONDS = 30
    # Retry delays between handler attempts (seconds). Three attempts,
    # exponential backoff. Tests pass tiny values to keep CI fast.
    DEFAULT_RETRY_DELAYS = (10.0, 30.0, 90.0)

    def __init__(
        self,
        session_maker: Optional[async_sessionmaker] = None,
        mcp_client: Any = None,
        retry_delays: Optional[tuple[float, ...]] = None,
    ):
        # Module proxy by default — resolved lazily so a test can swap
        # the engine without re-importing this module.
        if session_maker is None:
            from app.db.database import async_session_maker as _proxy
            session_maker = _proxy  # type: ignore[assignment]
        self._session_maker = session_maker
        # Injected at agent_main boot — handlers grab it for MCP calls.
        # None is acceptable; handlers that need MCP will return a
        # `failed` result with error_class="no_mcp_client".
        self._mcp_client = mcp_client
        # Agent runner — wired post-construction (lifespan order). The
        # generic `agent_task` handler uses it for tool-using prompts;
        # if absent, that handler falls back to a no-tools internal_llm
        # call so pure-text routines still work.
        self._agent_runner: Any = None
        self._retry_delays = retry_delays or self.DEFAULT_RETRY_DELAYS
        self.scheduler = AsyncIOScheduler()
        # routine_id → APScheduler job_id (1:1; the job_id IS the routine_id
        # so reload is just `add_job(..., id=routine_id, replace_existing=True)`)
        self._jobs: dict[str, str] = {}

    def set_mcp_client(self, mcp_client: Any) -> None:
        """Late-bind the MCP client. agent_main constructs the runner
        BEFORE the MCP client (lifespan order), so wire-up uses this
        rather than the constructor."""
        self._mcp_client = mcp_client

    def set_agent_runner(self, agent_runner: Any) -> None:
        """Late-bind the AgentRunner so `agent_task` handlers can call
        `agent_runner.run(prompt, ...)` with full tool access. agent_main
        wires this after the runner is constructed."""
        self._agent_runner = agent_runner

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Load enabled routines, sweep orphans, register triggers, start.

        Order matters: sweep BEFORE registering so a routine that orphaned
        last boot doesn't get a fresh fire while its old run row is still
        'running'."""
        swept = await self._restart_sweep()
        routines = await self._load_enabled_routines()
        tz_fallbacks = 0
        for r in routines:
            if await self._register_trigger_for(r) == "utc_fallback":
                tz_fallbacks += 1
        if not self.scheduler.running:
            self.scheduler.start()
        logger.info(
            "[routine_runner] started routines_loaded=%d tz_fallbacks=%d restart_sweep_marked=%d",
            len(routines), tz_fallbacks, swept,
        )

    async def stop(self) -> None:
        """Graceful shutdown. Blocks up to SHUTDOWN_TIMEOUT_SECONDS for
        in-flight jobs to drain. Beyond that, abandon and log so a hung
        handler can't keep the container alive."""
        if not self.scheduler.running:
            return
        # APScheduler.shutdown(wait=True) blocks the calling thread, so
        # offload to a thread executor and apply our own deadline.
        loop = asyncio.get_event_loop()
        completed_evt = asyncio.Event()

        def _shutdown_blocking():
            try:
                self.scheduler.shutdown(wait=True)
            finally:
                loop.call_soon_threadsafe(completed_evt.set)

        loop.run_in_executor(None, _shutdown_blocking)
        try:
            await asyncio.wait_for(completed_evt.wait(), timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
            logger.info("[routine_runner] stopped jobs_completed=clean jobs_abandoned=0")
        except asyncio.TimeoutError:
            # Force-shutdown what's left — APScheduler may have stuck jobs
            # we can't await any longer.
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:
                pass
            logger.warning(
                "[routine_runner] stopped jobs_completed=partial jobs_abandoned=%d timeout_seconds=%d",
                len(self._jobs), self.SHUTDOWN_TIMEOUT_SECONDS,
            )

    # ------------------------------------------------------------------ reload
    async def reload_routine(self, routine_id: str) -> None:
        """Re-read one routine from DB and re-register its trigger.
        Idempotent — used by the API on toggle/edit. If the routine no
        longer exists or is disabled, removes the trigger."""
        from app.db.models import Routine

        async with self._session_maker() as db:
            routine = await db.get(Routine, routine_id)

        # Remove first so the path is the same whether routine vanished,
        # got disabled, or just got rescheduled.
        self._unregister(routine_id)

        if routine and routine.enabled:
            await self._register_trigger_for(routine)

    async def reload_all(self) -> None:
        """Drop every registered trigger and rebuild from DB. Exposed for
        ops — `await rr.reload_all()` recovers from "I edited the DB
        directly and the runner is stale" situations."""
        for rid in list(self._jobs.keys()):
            self._unregister(rid)
        for routine in await self._load_enabled_routines():
            await self._register_trigger_for(routine)

    # ------------------------------------------------------------------ internals
    async def _load_enabled_routines(self):
        """Read enabled routines and gate by feature flag(s). A routine
        whose kind isn't enabled by the per-tenant flag stays in DB but
        does NOT get a scheduler trigger — flipping the flag at runtime
        + reload_all() is enough to activate it."""
        from app.config import settings
        from app.db.models import Routine

        async with self._session_maker() as db:
            result = await db.execute(select(Routine).where(Routine.enabled == True))  # noqa: E712
            rows = list(result.scalars().all())

        out = []
        for r in rows:
            if not self._kind_enabled(r.kind, settings):
                logger.info(
                    "[routine_runner] skipped kind=%s routine_id=%s reason=feature_flag_off",
                    r.kind, r.id,
                )
                continue
            out.append(r)
        return out

    @staticmethod
    def _kind_enabled(kind: str, settings) -> bool:
        """Map a routine kind to its feature flag. Unknown kinds default
        off so a typo or future-kind row doesn't surprise-activate.

        v2 generalisation: a single master flag
        (`routines_email_briefing_enabled`, retained for backward compat
        with the env var the canary tenant already has set) enables
        BOTH `email_briefing` (preset) AND `agent_task` (generic). One
        knob, two real kinds available. Add a per-kind flag later if a
        future kind needs separate gating."""
        master = bool(getattr(settings, "routines_email_briefing_enabled", False))
        # Test-only kinds bypass the flag so the smoke test doesn't need
        # to flip a real production setting on.
        if kind.startswith("_test_") or kind == "_smoke":
            return True
        if kind in ("email_briefing", "agent_task"):
            return master
        return False

    async def _register_trigger_for(self, routine) -> str:
        """Register one routine's trigger. Returns a tag string for
        observability: 'ok' | 'utc_fallback' | 'invalid_cron' | 'invalid_kind'.

        Async because we must resolve the user's IANA timezone from the
        DB before building the CronTrigger — a `0 7 * * *` schedule for a
        Toronto user has to fire at 07:00 local, not 07:00 UTC.
        """
        tz_str = await self._user_tz_async(routine.user_id)
        tz, fellback = _resolve_tz(tz_str, routine.user_id)
        trigger = _parse_cron(routine.schedule_cron_local, tz)
        if trigger is None:
            logger.warning(
                "[routine_runner] invalid_cron routine_id=%s kind=%s expr=%r — skipped",
                routine.id, routine.kind, routine.schedule_cron_local,
            )
            return "invalid_cron"

        # job_id == routine_id keeps replace_existing idempotent and makes
        # reload_routine() a single add_job call.
        self.scheduler.add_job(
            self._fire,
            trigger=trigger,
            id=routine.id,
            args=[routine.id],
            replace_existing=True,
            misfire_grace_time=300,  # if container was down ≤5min, still fire
        )
        self._jobs[routine.id] = routine.id
        # APScheduler computes next_run_time at register-time. Sync to
        # the DB so the dashboard + `/api/routines/{id}` response know
        # when the next fire is — pre-2026-05-12 the `next_run_at`
        # column was added to the schema but never written, surfacing
        # in the UI as a permanently "null" next-run badge even for
        # healthy daily routines.
        await self._sync_next_run(routine.id)
        return "utc_fallback" if fellback else "ok"

    async def _sync_next_run(self, routine_id: str) -> None:
        """Query APScheduler for the job's next fire time and persist
        it to `Routine.next_run_at`. Idempotent — safe to call from
        any code path that mutates the schedule (register, reload,
        post-terminal).

        Stored as naive UTC because the column is a tz-less DateTime.
        APScheduler hands back a tz-aware datetime, so we convert
        before stripping tzinfo (otherwise we'd land a 8-hours-off
        timestamp for Toronto users)."""
        from app.db.models import Routine

        try:
            if not self.scheduler.running:
                return
            job = self.scheduler.get_job(routine_id)
            next_run = getattr(job, "next_run_time", None) if job else None
            if next_run is None:
                next_run_utc: Optional[datetime] = None
            else:
                next_run_utc = next_run.astimezone(timezone.utc).replace(tzinfo=None)
            async with self._session_maker() as db:
                await db.execute(
                    update(Routine)
                    .where(Routine.id == routine_id)
                    .values(next_run_at=next_run_utc)
                )
                await db.commit()
        except Exception as e:
            # Never let next_run_at sync take down the scheduler. The
            # dashboard's "next fire" badge being null is a visual
            # bug; APScheduler still fires on schedule from its
            # in-memory state regardless of the DB column.
            logger.warning(
                "[routine_runner] next_run_at sync failed routine_id=%s err=%s",
                routine_id, e,
            )

    def _unregister(self, routine_id: str) -> None:
        try:
            self.scheduler.remove_job(routine_id)
        except Exception:
            pass
        self._jobs.pop(routine_id, None)

    async def _user_tz_async(self, user_id: str) -> Optional[str]:
        """Resolve a user's IANA timezone string, or None if missing.
        Used at trigger-registration AND fire time so the local-date
        computation matches the schedule's tz semantics."""
        from app.db.models import User
        async with self._session_maker() as db:
            user = await db.get(User, user_id)
        return getattr(user, "timezone", None) if user else None

    # ------------------------------------------------------------------ restart sweep
    async def _restart_sweep(self) -> int:
        """Mark orphaned 'running' rows from a prior boot as failed.

        A `routine_runs` row in 'running' state older than ORPHAN_THRESHOLD
        means: the agent crashed/restarted mid-execute. The next fire on
        the same local_date would otherwise hit the idempotency UNIQUE
        and silently exit, leaving the row stuck forever. We flip them
        to failed/agent_restarted so the next fire can claim a fresh row.

        NOTE: this does NOT change the (routine_id, scheduled_for_local_date)
        unique key. The orphaned row keeps that day's slot. That's intentional
        — we don't want to silently retry a half-completed briefing. The
        user sees the failure status and can force-run via the API.
        """
        from app.db.models import RoutineRun

        threshold = datetime.utcnow() - self.ORPHAN_THRESHOLD
        async with self._session_maker() as db:
            stmt = (
                update(RoutineRun)
                .where(
                    RoutineRun.status == "running",
                    RoutineRun.started_at < threshold,
                )
                .values(
                    status="failed",
                    error_class="agent_restarted",
                    error_detail="Agent restarted before run completed",
                    finished_at=datetime.utcnow(),
                )
            )
            result = await db.execute(stmt)
            await db.commit()
            return int(result.rowcount or 0)

    # ------------------------------------------------------------------ fire
    async def _fire(self, routine_id: str) -> None:
        """Per-trigger dispatch. Full flow:
          1. Reload routine fresh
          2. Compute scheduled_for_local_date in user tz
          3. Idempotency claim (INSERT … UNIQUE collision → silent exit)
          4. Look up handler in KIND_HANDLERS
          5. Retry loop: 3 attempts with backoff (10s/30s/90s default)
          6. On success → finalize run, advance watermark
          7. On skipped_reauth → write reauth nudge Message, finalize
          8. On terminal failed → write failure nudge Message, finalize
        """
        from app.db.models import Routine, RoutineRun
        from .registry import KIND_HANDLERS

        # Reload routine fresh — a toggle/edit between trigger registration
        # and this fire must be honored.
        async with self._session_maker() as db:
            routine = await db.get(Routine, routine_id)
        if routine is None or not routine.enabled:
            return

        tz_str = await self._user_tz_async(routine.user_id)
        tz, _ = _resolve_tz(tz_str, routine.user_id)
        local_date = datetime.now(tz).date()

        # Idempotency claim. SQLAlchemy raises IntegrityError on the UNIQUE
        # violation under both Postgres and SQLite — that's our gate. A
        # collision means another fire (or a force-run) already claimed
        # this user-local day; exit silently.
        run_id = str(uuid.uuid4())
        try:
            async with self._session_maker() as db:
                run = RoutineRun(
                    id=run_id,
                    routine_id=routine_id,
                    user_id=routine.user_id,
                    scheduled_for_local_date=local_date,
                    status="running",
                )
                db.add(run)
                await db.commit()
        except IntegrityError:
            logger.info(
                "[routine_runner] idempotency_collision kind=%s routine_id=%s scheduled_local_date=%s",
                routine.kind, routine_id, local_date,
            )
            return

        # Find handler. Unknown kind → log "would_execute" + finalize as
        # success (Gate 1 no-op behavior preserved for _smoke / _test_*
        # kinds that have no real handler).
        handler = KIND_HANDLERS.get(routine.kind)
        if handler is None:
            logger.info(
                "[routine_runner] would_execute kind=%s user_id=%s routine_id=%s scheduled_local_date=%s run_id=%s",
                routine.kind, routine.user_id, routine_id, local_date, run_id,
            )
            await self._finalize_run(run_id, status="success")
            return

        # Wire the MCP client + agent runner into the handler if the
        # handler declares either dep via attribute. Stateless: handlers
        # may share an instance across tenants — per-call X-Agent-Key
        # already scopes the MCP client; agent_runner is per-tenant by
        # virtue of being constructed in this tenant's lifespan.
        if self._mcp_client is not None and hasattr(handler, "_mcp_client"):
            if handler._mcp_client is None:
                handler._mcp_client = self._mcp_client
        if getattr(self, "_agent_runner", None) is not None and hasattr(handler, "_agent_runner"):
            if handler._agent_runner is None:
                handler._agent_runner = self._agent_runner

        result = await self._run_with_retry(handler, routine, run_id)
        await self._post_terminal(routine, run_id, result)

    # ------------------------------------------------------------------ retry loop
    async def _run_with_retry(self, handler, routine, run_id: str):
        """Up to len(retry_delays) attempts. Backoff between attempts
        (not before attempt 1). `skipped_reauth` short-circuits — no
        retry, it's a clean intentional exit. `success` short-circuits
        for obvious reasons. `failed` retries up to the budget."""
        from app.db.models import RoutineRun

        from .base_handler import RoutineResult

        attempts = len(self._retry_delays)
        last_result: Optional[RoutineResult] = None
        start_ts = datetime.utcnow()
        for attempt_idx in range(attempts):
            if attempt_idx > 0:
                delay = self._retry_delays[attempt_idx - 1]
                logger.info(
                    "[routine_run] retry kind=%s run_id=%s attempt=%d delay_ms=%d error_class=%s",
                    routine.kind, run_id, attempt_idx + 1, int(delay * 1000),
                    (last_result.error_class if last_result else "unknown"),
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise

            # Update attempt counter so observers see progress.
            try:
                async with self._session_maker() as db:
                    await db.execute(
                        update(RoutineRun)
                        .where(RoutineRun.id == run_id)
                        .values(attempt=attempt_idx + 1)
                    )
                    await db.commit()
            except Exception:
                pass  # Best-effort bookkeeping; don't fail the run on it.

            try:
                async with self._session_maker() as db:
                    run_obj = await db.get(RoutineRun, run_id)
                    last_result = await handler.execute(routine, run_obj, db)
            except Exception as e:
                logger.exception(
                    "[routine_run] handler_raised kind=%s run_id=%s attempt=%d err=%s",
                    routine.kind, run_id, attempt_idx + 1, e,
                )
                last_result = RoutineResult(
                    status="failed",
                    error_class=type(e).__name__,
                    error_detail=str(e)[:300],
                )

            latency_ms = int((datetime.utcnow() - start_ts).total_seconds() * 1000)
            logger.info(
                "[routine_run] kind=%s user_id=%s routine_id=%s run_id=%s "
                "status=%s emails_fetched=%s summary_chars=%s latency_ms=%d attempt=%d",
                routine.kind, routine.user_id, routine.id, run_id,
                last_result.status,
                last_result.emails_fetched,
                (last_result.metrics or {}).get("summary_chars", 0),
                latency_ms, attempt_idx + 1,
            )

            if last_result.status in ("success", "partial", "skipped_reauth"):
                # Emit metric hook (string log for now; real collector wired later).
                logger.info("[metric] routine.run.%s kind=%s", last_result.status, routine.kind)
                return last_result
            # status == "failed" → retry until the budget is exhausted.

        logger.info("[metric] routine.run.failed kind=%s", routine.kind)
        return last_result  # final failed result after exhausting retries

    # ------------------------------------------------------------------ terminal post + finalize
    async def _post_terminal(self, routine, run_id: str, result) -> None:
        """Handle the terminal RoutineResult: write nudge Message for
        non-success outcomes, advance the routine's watermark on
        success, finalize the run row.

        The idempotency gate is the `(routine_id, scheduled_for_local_date)`
        UNIQUE constraint that already claimed this run row — so
        skipped_reauth and failed nudges can post at most once per
        local day per the spec, because only one run row exists."""
        from app.db.models import Routine, RoutineRun

        message_id_for_run: Optional[str] = result.summary_message_id

        if result.status == "skipped_reauth":
            message_id_for_run = await self._write_nudge(
                routine, run_id,
                content=(
                    "⚠ Reconnect Gmail to resume morning briefings. "
                    "Open Mission Control → Routines and click Reconnect."
                ),
                source=routine.kind,
            )
        elif result.status == "failed":
            message_id_for_run = await self._write_nudge(
                routine, run_id,
                content=(
                    "Couldn't reach Gmail this morning — I'll try again tomorrow."
                ),
                source=routine.kind,
            )

        # Advance watermark on success only.
        if result.status == "success" and result.new_watermark is not None:
            try:
                async with self._session_maker() as db:
                    await db.execute(
                        update(Routine)
                        .where(Routine.id == routine.id)
                        .values(
                            last_state_json=result.new_watermark,
                            last_run_at=datetime.utcnow(),
                            last_status="success",
                            last_error=None,
                        )
                    )
                    await db.commit()
            except Exception as e:
                logger.warning(
                    "[routine_runner] watermark_advance_failed routine_id=%s err=%s",
                    routine.id, e,
                )
        else:
            # Update routine-level last_status without touching the watermark.
            try:
                async with self._session_maker() as db:
                    await db.execute(
                        update(Routine)
                        .where(Routine.id == routine.id)
                        .values(
                            last_run_at=datetime.utcnow(),
                            last_status=result.status,
                            last_error=result.error_detail,
                        )
                    )
                    await db.commit()
            except Exception:
                pass

        # After EVERY terminal outcome, refresh next_run_at from
        # APScheduler. The job's next_run_time has already advanced
        # past today's fire — without this sync, the dashboard shows
        # yesterday's planned fire time (or null for first-run-since-
        # boot) until the next scheduler restart.
        await self._sync_next_run(routine.id)

        await self._finalize_run(
            run_id,
            status=result.status,
            error_class=result.error_class,
            error_detail=result.error_detail,
            emails_fetched=result.emails_fetched,
            summary_message_id=message_id_for_run,
        )

    async def _write_nudge(self, routine, run_id: str, *, content: str, source: str) -> Optional[str]:
        """Post a routine-channel Message for the reauth or failure path.
        Returns the new message id (or None if the writer raised)."""
        from .message_writer import write_routine_message, broadcast_routine_message

        try:
            async with self._session_maker() as db:
                msg_id, day_chat_id = await write_routine_message(
                    db,
                    user_id=routine.user_id,
                    content=content,
                    source=source,
                    routine_id=routine.id,
                    title=f"Morning briefing — {datetime.utcnow().date().isoformat()}",
                    model_used=None,
                )
            await broadcast_routine_message(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=source,
                content=content,
                model_used=None,
            )
            return msg_id
        except Exception as e:
            logger.warning(
                "[routine_runner] nudge_write_failed routine_id=%s err=%s",
                routine.id, e,
            )
            return None

    async def _finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        error_class: Optional[str] = None,
        error_detail: Optional[str] = None,
        emails_fetched: int = 0,
        summary_message_id: Optional[str] = None,
    ) -> None:
        """Close out the routine_runs row. Best-effort — a failure here
        leaves the row in 'running' which the next-boot restart sweep
        will clean up."""
        from app.db.models import RoutineRun

        try:
            async with self._session_maker() as db:
                await db.execute(
                    update(RoutineRun)
                    .where(RoutineRun.id == run_id)
                    .values(
                        status=status,
                        error_class=error_class,
                        error_detail=error_detail,
                        emails_fetched=emails_fetched,
                        summary_message_id=summary_message_id,
                        finished_at=datetime.utcnow(),
                    )
                )
                await db.commit()
        except Exception as e:
            logger.warning("[routine_runner] finalize_failed run_id=%s err=%s", run_id, e)

    # ------------------------------------------------------------------ status (for /_runner_status)
    def status_snapshot(self) -> dict[str, Any]:
        """Return a small dict for the /_runner_status endpoint."""
        next_fire: Optional[datetime] = None
        try:
            jobs = self.scheduler.get_jobs() if self.scheduler.running else []
            fire_times = [j.next_run_time for j in jobs if j.next_run_time is not None]
            next_fire = min(fire_times) if fire_times else None
        except Exception:
            pass
        return {
            "running": bool(self.scheduler.running),
            "routines_registered": len(self._jobs),
            "next_fire_at": next_fire.isoformat() if next_fire else None,
        }
