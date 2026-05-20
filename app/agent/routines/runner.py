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
import types
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
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


def _build_trigger_for_routine(routine, tz):
    """Mig 042 — dispatch on schedule_kind.

    Treats NULL schedule_kind as 'cron' (legacy default) so rows that
    pre-date the migration keep firing as cron triggers — no operator
    intervention needed.

    Returns (trigger, tag) where tag is one of:
      'cron' | 'at' | 'every' | 'invalid' | 'past_at'
    """
    # NULL-tolerant read — DB rows older than mig 042 don't have a value.
    raw_kind = getattr(routine, "schedule_kind", None)
    kind = (raw_kind or "cron").strip() or "cron"

    if kind == "at":
        when = getattr(routine, "schedule_at", None)
        if when is None:
            return None, "invalid"
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when <= datetime.now(timezone.utc):
            return None, "past_at"
        return DateTrigger(run_date=when), "at"

    if kind == "every":
        interval = getattr(routine, "schedule_interval_seconds", None) or 0
        if interval < 60:
            return None, "invalid"
        return IntervalTrigger(seconds=int(interval), timezone=tz), "every"

    # Default path: cron (legacy, plus any row whose schedule_kind is NULL).
    trigger = _parse_cron(routine.schedule_cron_local, tz)
    if trigger is None:
        return None, "invalid"
    return trigger, "cron"


def _parse_window_time(s: Optional[str]):
    """Parse an HH:MM or HH:MM:SS string into a datetime.time.
    Returns None on missing/invalid input."""
    if not s:
        return None
    from datetime import time as dtime
    try:
        parts = s.split(":")
        if len(parts) == 2:
            return dtime(int(parts[0]), int(parts[1]))
        if len(parts) == 3:
            return dtime(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError):
        return None
    return None


def _in_active_window(now_local_time, start_local, end_local) -> bool:
    """True if `now_local_time` falls inside the daily active window.

    Handles overnight windows (22:00–06:00) by wrapping past midnight.
    None on either bound means that side is unbounded; both None means
    "always active" and the caller shouldn't be calling this.
    """
    if start_local is None and end_local is None:
        return True
    if start_local is None:
        return now_local_time < end_local
    if end_local is None:
        return now_local_time >= start_local
    if start_local <= end_local:
        return start_local <= now_local_time < end_local
    # Overnight wrap.
    return now_local_time >= start_local or now_local_time < end_local


def _independent_next_fire(expr: str, tz) -> Optional[datetime]:
    """Compute the next fire instant via croniter (independent of APScheduler).

    Used by the DST watchdog at trigger registration. If croniter is not
    available the watchdog is a no-op (returns None) — we don't fail
    registration on a missing optional dependency.

    Returns a tz-aware datetime in the SAME tz, or None on parse error /
    missing croniter.
    """
    try:
        from croniter import croniter
    except ImportError:
        return None
    parts = (expr or "").split()
    if len(parts) != 5:
        return None
    try:
        now = datetime.now(tz)
        nxt = croniter(expr, now).get_next(datetime)
        # croniter returns naive datetimes when base is naive, tz-aware
        # when base is aware. We pass an aware base; verify the result
        # carries the right tz.
        if nxt.tzinfo is None:
            return nxt.replace(tzinfo=tz)
        return nxt
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

        # Observability counters (Ticket 3 / invariant #4). Exposed via
        # status_snapshot() → /_runner_status. Don't reset across reloads;
        # they are process-lifetime totals so operators can spot drift
        # between successive observations.
        self._missed_fires_total = 0
        self._reload_failures_total = 0
        self._reconcile_runs_total = 0
        # Ticket 2.3 — DST watchdog drift counter (CRITICAL when bumps,
        # surfaced via /_runner_status for Mission Control).
        self._watchdog_drift_total = 0
        # In-memory background task for the periodic reconciler. Started
        # alongside the scheduler in `start()`, cancelled in `stop()`.
        self._reconcile_task: Optional[asyncio.Task] = None
        # Reconciler cadence — every 10 minutes is enough to catch drift
        # within one user-visible window without thrashing the DB.
        self.RECONCILE_INTERVAL_SECONDS = 10 * 60

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
        'running'.

        Per-routine registration failures are caught and logged so one
        bad row (missing tz, malformed schedule, APScheduler refusing
        the trigger expression) does NOT prevent the runner from
        starting and registering the rest. set_runner_ref must run for
        /api/routines GET to render — a fail-fast here was producing
        runner_not_started + every /api/routines call 500ing on the
        dashboard for the affected tenant. Caught 2026-05-18.
        """
        swept = await self._restart_sweep()
        try:
            routines = await self._load_enabled_routines()
        except Exception as e:
            logger.exception(
                "[routine_runner] load_enabled_routines failed: %s — "
                "starting with empty schedule, /_reload_all can recover",
                e,
            )
            routines = []
        tz_fallbacks = 0
        register_errors = 0
        for r in routines:
            try:
                if await self._register_trigger_for(r) == "utc_fallback":
                    tz_fallbacks += 1
            except Exception as e:
                register_errors += 1
                logger.exception(
                    "[routine_runner] register_trigger crash routine_id=%s "
                    "kind=%s — skipping this routine, continuing with rest: %s",
                    getattr(r, "id", "?"),
                    getattr(r, "kind", "?"),
                    e,
                )
        if not self.scheduler.running:
            self.scheduler.start()
        # Periodic reconciler — backstop for any in-memory↔DB drift
        # (silent reload failures, dropped APScheduler jobs, etc.). Idem-
        # potent: if state is already clean, each cycle is a no-op.
        if self._reconcile_task is None or self._reconcile_task.done():
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        logger.info(
            "[routine_runner] started routines_loaded=%d tz_fallbacks=%d "
            "restart_sweep_marked=%d reconcile_interval=%ds",
            len(routines), tz_fallbacks, swept,
            self.RECONCILE_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        """Graceful shutdown. Blocks up to SHUTDOWN_TIMEOUT_SECONDS for
        in-flight jobs to drain. Beyond that, abandon and log so a hung
        handler can't keep the container alive."""
        # Cancel the reconciler first so it can't trigger a reload mid-
        # shutdown. Await briefly so the cancellation completes cleanly.
        if self._reconcile_task is not None and not self._reconcile_task.done():
            self._reconcile_task.cancel()
            try:
                await asyncio.wait_for(self._reconcile_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._reconcile_task = None
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

    async def _reconcile_loop(self) -> None:
        """Periodic reconciler — every RECONCILE_INTERVAL_SECONDS, re-read
        every enabled routine from DB and re-register it. Catches the
        failure mode where `update_routine` swallows a reload exception
        (Ticket 3 / Bug B) or APScheduler internal state diverges from
        DB silently.

        Idempotent: `_register_trigger_for` uses `replace_existing=True`
        so a no-op cycle just refreshes `next_run_at` to whatever the
        in-memory scheduler thinks.

        Self-cancelling on `asyncio.CancelledError` (the normal
        shutdown path). Other exceptions are logged and the loop
        continues — never let a transient DB blip stop the backstop.
        """
        try:
            while True:
                await asyncio.sleep(self.RECONCILE_INTERVAL_SECONDS)
                try:
                    await self.reload_all()
                    self._reconcile_runs_total += 1
                    logger.info(
                        "[routine_runner] reconcile_ok runs_total=%d routines=%d",
                        self._reconcile_runs_total, len(self._jobs),
                    )
                except Exception as e:  # pragma: no cover — defensive
                    logger.warning(
                        "[routine_runner] reconcile_failed err=%s — will retry next cycle",
                        e,
                    )
        except asyncio.CancelledError:
            logger.debug("[routine_runner] reconcile_loop cancelled")
            raise

    async def reload_all(self) -> None:
        """Drop every registered trigger and rebuild from DB. Exposed for
        ops — `await rr.reload_all()` recovers from "I edited the DB
        directly and the runner is stale" situations.

        Ticket 2.3(d): post-deploy operators run this once via the
        `/api/routines/_reload_all` endpoint after a release that
        changes scheduler semantics (DST audit, fire-time logic, cron
        parser). Logs before/after delta so the on-call sees what
        actually moved."""
        from app.db.models import Routine

        before_ids = set(self._jobs.keys())
        for rid in list(before_ids):
            self._unregister(rid)
        loaded = await self._load_enabled_routines()
        for routine in loaded:
            await self._register_trigger_for(routine)
        after_ids = set(self._jobs.keys())
        added = after_ids - before_ids
        removed = before_ids - after_ids
        logger.info(
            "[routine_runner] reload_all_complete before=%d after=%d "
            "added=%d removed=%d watchdog_drift_total=%d",
            len(before_ids), len(after_ids),
            len(added), len(removed), self._watchdog_drift_total,
        )

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
        # Mig 042 — reminder kind has its own dedicated flag so operators
        # can roll out reminders independently of email briefing. Falls
        # back to the master flag for canary tenants that already have
        # the master one set. `getattr` with default False means tenants
        # whose Settings haven't picked up the new field stay safe.
        if kind == "reminder":
            return bool(getattr(settings, "routines_reminders_enabled", master))
        if kind in ("email_briefing", "agent_task"):
            return master
        return False

    async def _register_trigger_for(self, routine) -> str:
        """Register one routine's trigger. Returns a tag string for
        observability: 'ok' | 'utc_fallback' | 'invalid_cron' |
        'invalid_kind' | 'missing_tz'.

        Async because we must resolve the user's IANA timezone from the
        DB before building the CronTrigger — a `0 7 * * *` schedule for a
        Toronto user has to fire at 07:00 local, not 07:00 UTC.

        When the user has no stored timezone we log loudly and SKIP
        registration. Falling back to UTC silently was the 2026-05-12
        production bug: a Toronto user's "fire at 1:21 PM" landed as
        1:21 PM UTC = 9:21 AM Toronto. A missed registration is a
        better failure mode than a misfire at the wrong hour — the
        operator sees the log, the routine reactivates the moment the
        user's tz is captured + the runner reloads.
        """
        tz_str = await self._user_tz_async(routine.user_id)
        if not tz_str:
            logger.error(
                "[routine_runner] missing_tz routine_id=%s user=%s — "
                "skipping registration. User must set timezone before "
                "routines can fire at the right local time.",
                routine.id, routine.user_id,
            )
            return "missing_tz"
        tz, fellback = _resolve_tz(tz_str, routine.user_id)
        # Mig 042: dispatch on schedule_kind (NULL → cron). Old rows with
        # NULL schedule_kind keep behaving as cron triggers.
        trigger, trigger_tag = _build_trigger_for_routine(routine, tz)
        if trigger is None:
            if trigger_tag == "past_at":
                logger.info(
                    "[routine_runner] past_at routine_id=%s kind=%s "
                    "schedule_at=%s — already delivered or expired, skipping",
                    routine.id, routine.kind,
                    getattr(routine, "schedule_at", None),
                )
                return "past_at"
            logger.warning(
                "[routine_runner] invalid_schedule routine_id=%s kind=%s "
                "schedule_kind=%s — skipped",
                routine.id, routine.kind,
                getattr(routine, "schedule_kind", None) or "cron",
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

        # Ticket 2.3 — DST watchdog. Independently recompute the next
        # fire instant via croniter+ZoneInfo and compare against
        # APScheduler's projection. >60s drift triggers a CRITICAL log
        # AND we count it on `_watchdog_drift_total` (Mission Control
        # observability). The watchdog is a *passive* safety net — we
        # do NOT refuse to register, because Phase 1.3 proved APScheduler
        # 3.10.4 is currently correct. Future regressions (apscheduler
        # bump, tzdata churn) will trip the watchdog with a SEV-2 alert
        # before they ship to users.
        try:
            independent = _independent_next_fire(routine.schedule_cron_local, tz)
            job = self.scheduler.get_job(routine.id)
            aps_next = getattr(job, "next_run_time", None) if job else None
            if independent is not None and aps_next is not None:
                drift = abs((aps_next - independent).total_seconds())
                if drift > 60:
                    self._watchdog_drift_total += 1
                    logger.critical(
                        "[routine_runner] dst_watchdog_drift routine_id=%s "
                        "cron=%r tz=%s aps_next=%s croniter_next=%s "
                        "drift_seconds=%.1f counter_total=%d — possible "
                        "APScheduler regression vs croniter, review needed",
                        routine.id, routine.schedule_cron_local, tz_str,
                        aps_next.isoformat(), independent.isoformat(),
                        drift, self._watchdog_drift_total,
                    )
        except Exception as e:  # pragma: no cover — watchdog must never block
            logger.debug(
                "[routine_runner] dst_watchdog_skipped routine_id=%s err=%s",
                routine.id, e,
            )

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

        A row in 'running' state older than ORPHAN_THRESHOLD means: the
        agent crashed/restarted mid-execute. The next fire on the same
        local_date would otherwise hit the idempotency UNIQUE and
        silently exit, leaving the row stuck forever. We flip them to
        failed/agent_restarted so the next fire can claim a fresh row.

        PR #49 of the unified-jobs cutover arc — the legacy
        ``routine_runs`` UPDATE was removed once the read path was
        repointed at ``build_jobs``. We use ``created_at`` as the
        orphan-cutoff proxy on the Job side since BuildJob has no
        ``started_at`` column — a row stuck in ``status='running'``
        with a ``created_at`` older than the threshold is, by
        definition, an orphan from a previous boot.

        NOTE: the orphaned row keeps its (source_id, idempotency_key)
        slot. That's intentional — we don't want to silently retry a
        half-completed briefing. The user sees the failure status and
        can force-run via the API.
        """
        from app.db.models import BuildJob

        threshold = datetime.utcnow() - self.ORPHAN_THRESHOLD
        now = datetime.utcnow()
        async with self._session_maker() as db:
            res_job = await db.execute(
                update(BuildJob)
                .where(
                    BuildJob.status == "running",
                    BuildJob.source_kind == "routine",
                    BuildJob.created_at < threshold,
                )
                .values(
                    status="failed",
                    error_message="agent_restarted: orphaned by agent restart",
                    completed_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            await db.commit()
            return int(getattr(res_job, "rowcount", 0) or 0)

    # ------------------------------------------------------------------ fire
    async def _fire(self, routine_id: str) -> None:
        """Per-trigger dispatch. Full flow:
          1. Reload routine fresh
          2. Capture fire_instant from APScheduler (Ticket 2.3) and
             compute scheduled_for_local_date in user tz
          3. Idempotency claim on build_jobs (SELECT + create_job)
          4. Look up handler in KIND_HANDLERS
          5. Retry loop: 3 attempts with backoff (10s/30s/90s default)
          6. On success → finalize run, advance watermark
          7. On skipped_reauth → write reauth nudge Message, finalize
          8. On terminal failed → write failure nudge Message, finalize

        PR #49 of the cutover arc — the legacy ``routine_runs``
        dual-write was removed; the BuildJob is now the only row
        carrying per-fire state. Downstream callers (``_run_with_retry``,
        ``_finalize_run``, ``_post_terminal``) operate on the
        ``BuildJob.id`` returned by ``JobRunner.create_job`` — the
        variable is still named ``run_id`` to keep the internal call
        graph readable.
        """
        from app.db.models import Routine
        from .registry import KIND_HANDLERS

        # Reload routine fresh — a toggle/edit between trigger registration
        # and this fire must be honored.
        async with self._session_maker() as db:
            routine = await db.get(Routine, routine_id)
        if routine is None or not routine.enabled:
            return

        tz_str = await self._user_tz_async(routine.user_id)
        tz, _ = _resolve_tz(tz_str, routine.user_id)
        now_local = datetime.now(tz)
        local_date = now_local.date()

        # Mig 042: daily active-window gate for interval reminders.
        # When schedule_kind='every' and a window is configured, fires
        # outside the window are silently dropped (no run row, no
        # broadcast). Cron + at kinds fire at literal scheduled times;
        # they don't use the window.
        if (getattr(routine, "schedule_kind", None) == "every"
                and (getattr(routine, "schedule_window_start_local", None)
                     or getattr(routine, "schedule_window_end_local", None))):
            start = _parse_window_time(getattr(routine, "schedule_window_start_local", None))
            end = _parse_window_time(getattr(routine, "schedule_window_end_local", None))
            if not _in_active_window(now_local.time(), start, end):
                logger.debug(
                    "[routine_runner] window_skip routine_id=%s now_local=%s "
                    "window=[%s, %s] — outside active hours",
                    routine_id, now_local.time().isoformat(),
                    getattr(routine, "schedule_window_start_local", None),
                    getattr(routine, "schedule_window_end_local", None),
                )
                return

        # Ticket 2.3: capture the scheduled fire instant BEFORE doing any
        # other work. APScheduler hands us this via `job.next_run_time`
        # at fire-callback entry — it's the wall-clock the trigger was
        # supposed to fire at, not "now". Persisted into `fire_instant`
        # so `last_run_at` can be stamped from it (instead of from
        # datetime.utcnow() at end of _post_terminal, which lies about
        # the fire time by `handler_latency` seconds/minutes).
        fire_instant: Optional[datetime] = None
        try:
            _job = self.scheduler.get_job(routine_id)
            _scheduled_for = getattr(_job, "next_run_time", None) if _job else None
            if _scheduled_for is not None:
                fire_instant = _scheduled_for.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:  # pragma: no cover — never block on this
            pass

        # Missed-fire detection (Ticket 3 / Bug D). APScheduler invokes
        # _fire whenever it decides the trigger is due — including a
        # "catch up" fire when misfire_grace_time is satisfied. If the
        # job's recorded next_run_time is more than grace_time behind
        # now-wall-clock, something went wrong (scheduler was paused, OS
        # clock jumped, in-process delay). Count it so operators have a
        # signal that "the routine fired but late."
        try:
            _job = self.scheduler.get_job(routine_id)
            _scheduled_for = getattr(_job, "next_run_time", None) if _job else None
            if _scheduled_for is not None:
                _now = datetime.now(_scheduled_for.tzinfo)
                _drift = (_now - _scheduled_for).total_seconds()
                if _drift > 300:  # misfire_grace_time
                    self._missed_fires_total += 1
                    logger.warning(
                        "[routine_runner] missed_fire routine_id=%s "
                        "drift_seconds=%.0f counter_total=%d",
                        routine_id, _drift, self._missed_fires_total,
                    )
        except Exception:  # pragma: no cover — counter must never block
            pass

        # ── PR #47 cutover: Job-first intake ─────────────────────────
        # The composite UNIQUE on ``(build_jobs.source_id,
        # build_jobs.idempotency_key)`` is the new dedupe gate,
        # replacing the legacy ``UNIQUE (routine_id,
        # scheduled_for_local_date)`` on ``routine_runs`` as source of
        # truth. We do an explicit SELECT for the pair BEFORE calling
        # ``JobRunner.create_job`` — matches the PR #46 fix in
        # ``triggers_inbound._idempotent_insert``: the proximity
        # heuristic (now - job.created_at < 5s) is unreliable when two
        # near-simultaneous calls land within microseconds (tests, and
        # real overlapping fires on a slow DB). One indexed SELECT per
        # fire — negligible cost on the hot path.
        from app.agent.job_runner import JobRunner, TaskSpec
        from app.db.models import BuildJob

        idempotency_key = str(local_date)
        async with self._session_maker() as db:
            existing_job_id = (await db.execute(
                select(BuildJob.id).where(
                    BuildJob.source_id == routine_id,
                    BuildJob.idempotency_key == idempotency_key,
                )
            )).scalar_one_or_none()
        if existing_job_id is not None:
            logger.info(
                "[routine_runner] idempotency_collision kind=%s "
                "routine_id=%s scheduled_local_date=%s",
                routine.kind, routine_id, local_date,
            )
            return

        # Fresh-mint the Job. ``JobRunner.create_job`` is idempotent
        # under concurrent calls (a race past the SELECT above
        # collapses on the partial UNIQUE INSERT), so a loser thread
        # gets the existing Job back. PR #49 removed the legacy
        # ``routine_runs`` dual-write that used to follow this mint.
        _runner = JobRunner()
        _spec = TaskSpec(
            user_id=routine.user_id,
            channel="routine",
            source_kind="routine",
            source_id=routine_id,
        )
        try:
            _job = await _runner.create_job(
                job_type="routine_run",
                spec=_spec,
                title=f"Routine fire: {routine.kind} {local_date}",
                idempotency_key=idempotency_key,
            )
        except Exception as _e:
            # Contract change in PR #47 (matches PR #46): a Job mint
            # failure aborts the fire. Job is the source of truth now;
            # without it we have no row to read in the runner's
            # queue-scan, no terminal mirror target, and no anchor for
            # observability. APScheduler will fire again on the next
            # interval; the operator gets a structured log.
            logger.error(
                "[routine_runner] job_mint_failed routine_id=%s "
                "kind=%s local_date=%s err=%s — aborting fire "
                "(APScheduler will retry on next interval)",
                routine_id, routine.kind, local_date, _e,
            )
            return

        # Stamp fire_instant + attempt=1 on the new BuildJob. attempt
        # bumps in ``_run_with_retry`` per attempt; we set the floor
        # value here so the activity feed sees attempt=1 from the
        # moment the row is visible. fire_instant uses the APScheduler
        # value computed above — the user-visible "when the routine
        # fired" wall-clock.
        #
        # PR #49 cutover: the legacy ``routine_runs`` dual-write
        # (preserved across PR #47 + #48) is gone. The BuildJob row
        # IS the per-fire state. We keep the ``run_id`` name in the
        # downstream call chain (``_run_with_retry`` / ``_finalize_run``
        # / ``_post_terminal``) — it now holds ``BuildJob.id``.
        run_id = _job.id
        async with self._session_maker() as db:
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == _job.id)
                .values(
                    status="running",
                    fire_instant=fire_instant,
                    attempt=1,
                )
            )
            await db.commit()

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
        for obvious reasons. `failed` retries up to the budget.

        PR #49 cutover: ``run_id`` is now a ``BuildJob.id`` (the
        legacy RoutineRun dual-write was removed in ``_fire``). The
        handler signature ``execute(routine, run, db)`` stays
        unchanged — we build a ``types.SimpleNamespace`` shim
        from the BuildJob fields so handlers see a row-shaped
        object. Audit (PR #47) confirmed no handler reads any
        attribute off ``run`` today; the shim keeps the contract
        sane for future handler authors.
        """
        from app.db.models import BuildJob

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

            # PR #49: single UPDATE on BuildJob.attempt — the legacy
            # RoutineRun.attempt write was removed alongside the rest
            # of the dual-write surface. Best-effort bookkeeping;
            # don't fail the run on it.
            try:
                async with self._session_maker() as db:
                    await db.execute(
                        update(BuildJob)
                        .where(BuildJob.id == run_id)
                        .values(attempt=attempt_idx + 1)
                    )
                    await db.commit()
            except Exception:
                pass

            try:
                async with self._session_maker() as db:
                    job_row = await db.get(BuildJob, run_id)
                    run_shim = self._build_run_shim(job_row, routine)
                    last_result = await handler.execute(routine, run_shim, db)
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

    @staticmethod
    def _build_run_shim(job_row: Any, routine: Any) -> types.SimpleNamespace:
        """Build a row-shaped object that handlers receive as the
        ``run`` parameter of ``handler.execute(routine, run, db)``.

        PR #49 of the cutover arc: with the legacy RoutineRun gone,
        nothing on ``run`` is read by any handler today (audit
        confirmed pre-cutover). We still populate the fields a
        future handler might want — sourced from the BuildJob row
        — so the contract reads sanely. Specifically:

          - ``id``: BuildJob.id (also the value of ``run_id``).
          - ``routine_id``: BuildJob.source_id (routine.id).
          - ``user_id``: BuildJob.user_id.
          - ``scheduled_for_local_date``: derived from
            BuildJob.idempotency_key, which ``_fire`` sets to
            ``str(local_date)``. We parse leniently — if a future
            kind uses a different idempotency_key shape, the field
            falls back to None rather than crashing.
          - ``fire_instant``: BuildJob.fire_instant (mig 051).
        """
        scheduled_local_date: Optional[date] = None
        try:
            raw_key = getattr(job_row, "idempotency_key", None)
            if raw_key:
                scheduled_local_date = date.fromisoformat(raw_key)
        except Exception:
            scheduled_local_date = None
        return types.SimpleNamespace(
            id=getattr(job_row, "id", None),
            routine_id=getattr(job_row, "source_id", None) or routine.id,
            user_id=getattr(job_row, "user_id", None) or routine.user_id,
            scheduled_for_local_date=scheduled_local_date,
            fire_instant=getattr(job_row, "fire_instant", None),
        )

    # ------------------------------------------------------------------ outcome derivation
    @staticmethod
    def _derive_outcome(result) -> str:
        """Map a `RoutineResult` into the Ticket 2.1 outcome state machine.

        Precedence:
          1. Explicit `result.outcome` (when a handler knows better).
          2. Status-driven mapping:
               status="skipped_reauth" → "tool_error"
               status="failed"         → "failure"
               status="partial"        → "partial"
          3. Channel-results-driven downgrade on `status="success"`:
               - any channel_results value with status != "delivered" → "partial"
               - empty inbox (emails_fetched==0 AND new_watermark is None) → "success_empty"
               - otherwise → "success"
        """
        if getattr(result, "outcome", None):
            return result.outcome  # type: ignore[return-value]
        s = result.status
        if s == "skipped_reauth":
            return "tool_error"
        if s == "failed":
            return "failure"
        if s == "partial":
            return "partial"
        # status == "success" — branch on channel_results then emails_fetched.
        channel_results = getattr(result, "channel_results", None) or {}
        if channel_results:
            statuses = {
                (entry or {}).get("status", "unknown")
                for entry in channel_results.values()
            }
            if statuses and any(s != "delivered" for s in statuses):
                # At least one channel failed to deliver. The Day-as-Chat
                # row may still exist (handler wrote it). Outcome is
                # "partial" so the operator and the user can see which.
                return "partial"
        # Empty-inbox signal: handler returned success with no new
        # watermark AND no emails fetched. `_post_empty` path.
        if (result.emails_fetched or 0) == 0 and result.new_watermark is None:
            return "success_empty"
        return "success"

    @staticmethod
    def _build_error_json(result, attempt: Optional[int] = None) -> Optional[dict]:
        """Build the structured error blob persisted into routine_runs.error_json.
        Returns None for success outcomes so we don't pollute the column."""
        if result.status == "success" or not (result.error_class or result.error_detail):
            return None
        out: dict[str, Any] = {
            "class": result.error_class,
            "detail": result.error_detail,
        }
        if attempt is not None:
            out["attempt"] = attempt
        # Optional fields handlers may have set on metrics.
        m = result.metrics or {}
        for k in ("tool_name", "tool_call_id", "provider", "http_status"):
            if k in m:
                out[k] = m[k]
        return out

    # ------------------------------------------------------------------ terminal post + finalize
    async def _post_terminal(self, routine, run_id: str, result) -> None:
        """Handle the terminal RoutineResult: write nudge Message for
        non-success outcomes, advance the routine's watermark on
        success, finalize the run row.

        The idempotency gate is the composite UNIQUE on
        ``(build_jobs.source_id, build_jobs.idempotency_key)`` (where
        ``idempotency_key=str(local_date)``) that already claimed this
        Job row — so skipped_reauth and failed nudges can post at most
        once per local day per the spec, because only one Job row
        exists."""
        from app.db.models import Routine

        message_id_for_run: Optional[str] = result.summary_message_id

        # Ticket 2.1 — derive outcome from result + channel_results map.
        outcome = self._derive_outcome(result)

        if result.status == "skipped_reauth":
            # Ticket 2.2: build the reconnect notice + dedupe + fan-out
            # across `delivery_channels` so Telegram/WhatsApp are no longer
            # silent on tool failures.
            nudge_msg_id, nudge_channel_results = await self._write_nudge(
                routine, run_id,
                content=self._build_reconnect_message(routine, kind="reauth"),
                source=routine.kind,
                notice_kind="reauth",
            )
            if nudge_msg_id:
                message_id_for_run = nudge_msg_id
            # Per-channel reconnect-notice results live alongside any
            # handler-emitted channel_results — merge so the run row
            # reflects everything that actually went out.
            if nudge_channel_results:
                cr = dict(getattr(result, "channel_results", None) or {})
                cr.update(nudge_channel_results)
                result.channel_results = cr  # type: ignore[attr-defined]
        elif result.status == "failed":
            nudge_msg_id, nudge_channel_results = await self._write_nudge(
                routine, run_id,
                content=self._build_reconnect_message(routine, kind="failure"),
                source=routine.kind,
                notice_kind="failure",
            )
            if nudge_msg_id:
                message_id_for_run = nudge_msg_id
            if nudge_channel_results:
                cr = dict(getattr(result, "channel_results", None) or {})
                cr.update(nudge_channel_results)
                result.channel_results = cr  # type: ignore[attr-defined]

        # Ticket 2.3: use fire_instant for last_run_at instead of
        # datetime.utcnow() at end of _post_terminal. The fire instant
        # is what "when did the routine run?" actually means to a user
        # — the post-terminal stamp lies by handler_latency seconds, and
        # for a slow run with retries can lie by minutes.
        # Fall back to utcnow when fire_instant isn't recoverable (e.g.,
        # a force-run via /api/routines/{id}/run skipped through _fire).
        #
        # PR #49 of the cutover arc: read directly from
        # ``BuildJob.fire_instant`` (mig 051 column, populated by the
        # Job-first intake in ``_fire``). The legacy
        # ``RoutineRun.fire_instant`` read was removed alongside the
        # rest of the dual-write surface.
        from app.db.models import BuildJob
        fire_at: Optional[datetime] = None
        try:
            async with self._session_maker() as db:
                job_row = await db.get(BuildJob, run_id)
                fire_at = getattr(job_row, "fire_instant", None) if job_row else None
        except Exception:
            fire_at = None
        if fire_at is None:
            fire_at = datetime.utcnow()

        # Advance watermark on success only.
        if result.status == "success" and result.new_watermark is not None:
            try:
                async with self._session_maker() as db:
                    await db.execute(
                        update(Routine)
                        .where(Routine.id == routine.id)
                        .values(
                            last_state_json=result.new_watermark,
                            last_run_at=fire_at,
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
                            last_run_at=fire_at,
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

        # Structured logging at the outcome boundary. Operators read this
        # to spot drift between status (legacy) and outcome (new) — for
        # example status=success + outcome=partial means "handler said
        # success but a channel skipped" — exactly the Defect A symptom.
        logger.info(
            "[routine_run] terminal kind=%s routine_id=%s run_id=%s "
            "status=%s outcome=%s emails_fetched=%d channels=%d tools=%d",
            routine.kind, routine.id, run_id,
            result.status, outcome,
            int(result.emails_fetched or 0),
            len(getattr(result, "channel_results", None) or {}),
            len(getattr(result, "tools_invoked", None) or []),
        )

        await self._finalize_run(
            run_id,
            status=result.status,
            outcome=outcome,
            error_class=result.error_class,
            error_detail=result.error_detail,
            error_json=self._build_error_json(result),
            emails_fetched=result.emails_fetched,
            summary_message_id=message_id_for_run,
            channel_results=dict(getattr(result, "channel_results", None) or {}) or None,
            tools_invoked=list(getattr(result, "tools_invoked", None) or []) or None,
        )

        # Mig 042 — auto-disable after a successful fire for one-shot
        # reminders (and any future kind that opts in). Only triggers
        # on delivered outcomes (success / success_empty / partial) so
        # transient failures still get a retry on the next tick.
        # `getattr` is NULL-tolerant — rows pre-dating the migration
        # have auto_disable_after_fire=None, treated as False.
        if (getattr(routine, "auto_disable_after_fire", False)
                and outcome in ("success", "success_empty", "partial")):
            try:
                async with self._session_maker() as db:
                    await db.execute(
                        update(Routine)
                        .where(Routine.id == routine.id)
                        .values(enabled=False, updated_at=datetime.utcnow())
                    )
                    await db.commit()
                self._unregister(routine.id)
                logger.info(
                    "[routine_runner] auto_disabled routine_id=%s kind=%s "
                    "outcome=%s — one-shot delivered, removing trigger",
                    routine.id, routine.kind, outcome,
                )
            except Exception as e:  # pragma: no cover — best-effort cleanup
                logger.warning(
                    "[routine_runner] auto_disable_failed routine_id=%s err=%s",
                    routine.id, e,
                )

    @staticmethod
    def _build_reconnect_message(routine, *, kind: str) -> str:
        """Per-channel content for the reauth/failure notice (Ticket 2.2).

        Body carries both plain text + the inline connector-card marker
        `[[connector_card:<slug>|Reconnect]]`. The chat UI's
        `parseMessageSections` (frontend/src/modules/chat/parseMessageBlocks.ts)
        recognises the marker and renders a one-tap Reconnect card.
        Telegram/WhatsApp clients render the URL as a clickable link
        (the bracket marker degrades to plain text — the URL still works).

        Connector slug defaults to `gmail` for `email_briefing` kinds.
        A future kind can override via `config_json.reconnect_connector`.
        """
        connector = "gmail"
        try:
            cfg = routine.config_json or {}
            connector = (cfg.get("reconnect_connector") or "gmail").strip().lower()
        except Exception:
            pass

        if kind == "reauth":
            return (
                f"⚠ Reconnect {connector.title()} to resume your routine. "
                f"Tap the card below, or open "
                f"https://toup.ai/agent/integrations?reconnect={connector}\n\n"
                f"[[connector_card:{connector}|Reconnect]]"
            )
        return (
            f"Couldn't reach {connector.title()} this morning — I'll try again "
            f"tomorrow. If this keeps happening, reconnect:\n\n"
            f"[[connector_card:{connector}|Reconnect]]"
        )

    async def _notice_dedupe_taken(
        self,
        *,
        routine_id: str,
        kind: str,
        scope_date: date,
    ) -> bool:
        """Ticket 2.2 — 24h rate limit on reconnect/failure notices.

        Returns True if a notice of this kind was already sent for this
        (routine, scope_date). The dedupe key is `(routine_id, kind,
        scope_date)`; one notice per kind per local day per routine.
        Misses the first time → claim the slot and return False.
        """
        from app.db.models import RoutineNotificationDedupe

        try:
            async with self._session_maker() as db:
                row = RoutineNotificationDedupe(
                    routine_id=routine_id, kind=kind, scope_date=scope_date,
                )
                db.add(row)
                await db.commit()
                return False
        except IntegrityError:
            return True
        except Exception as e:
            logger.warning(
                "[routine_runner] notice_dedupe_check_failed routine_id=%s err=%s",
                routine_id, e,
            )
            # Fail-open: we'd rather post a duplicate than swallow the
            # first one. The 24h window is a UX nicety, not a hard
            # guarantee.
            return False

    async def _write_nudge(
        self,
        routine,
        run_id: str,
        *,
        content: str,
        source: str,
        notice_kind: str,
    ) -> tuple[Optional[str], dict[str, dict[str, Any]]]:
        """Post a routine-channel reconnect/failure notice.

        Ticket 2.2 — broadcasts to EVERY configured `delivery_channels`
        (not just website). Also rate-limits via `routine_notification_dedupe`
        so the same notice can't post twice for the same routine on the
        same local day.

        Returns (message_id, channel_results) — both consumed by the
        caller to fold into the run row.
        """
        from .message_writer import write_routine_message, broadcast_routine_message
        from .channel_dispatcher import parse_delivery_channels

        # Dedupe — at most one notice of each kind per routine per local day.
        try:
            tz_str = await self._user_tz_async(routine.user_id)
            tz, _ = _resolve_tz(tz_str, routine.user_id)
            local_date = datetime.now(tz).date()
        except Exception:
            local_date = datetime.utcnow().date()

        if await self._notice_dedupe_taken(
            routine_id=routine.id, kind=notice_kind, scope_date=local_date,
        ):
            logger.info(
                "[routine_runner] nudge_deduped routine_id=%s kind=%s scope_date=%s "
                "— suppressing duplicate notice (24h window)",
                routine.id, notice_kind, local_date,
            )
            return None, {}

        delivery_channels = parse_delivery_channels(routine.config_json)
        routine_name = (routine.name or "").strip() or "Morning email briefing"

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
                    extra_metadata={
                        "notice_kind": notice_kind,
                        "routine_message": True,
                    },
                )
            broadcast_out = await broadcast_routine_message(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=source,
                content=content,
                model_used=None,
                delivery_channels=delivery_channels,
                routine_name=routine_name,
            )
            # Broadcast helper returns either int (legacy) or a
            # ChannelResults dict (Ticket 2.5 — after upgrade). Support both.
            channel_results: dict[str, dict[str, Any]] = {}
            if isinstance(broadcast_out, dict):
                channel_results = broadcast_out.get("channel_results", {}) or {}
            return msg_id, channel_results
        except Exception as e:
            logger.warning(
                "[routine_runner] nudge_write_failed routine_id=%s err=%s",
                routine.id, e,
            )
            return None, {}

    async def _finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        outcome: Optional[str] = None,
        error_class: Optional[str] = None,
        error_detail: Optional[str] = None,
        error_json: Optional[dict] = None,
        emails_fetched: int = 0,
        summary_message_id: Optional[str] = None,
        channel_results: Optional[dict] = None,
        tools_invoked: Optional[list] = None,
    ) -> None:
        """Close out the BuildJob row. Best-effort — a failure here
        leaves the row in 'running' which the next-boot restart sweep
        will clean up.

        PR #49 of the cutover arc: this used to write to both
        ``routine_runs`` AND ``build_jobs`` (via
        ``_mirror_run_terminal_to_job``). The legacy UPDATE was
        removed once every reader was repointed at the Job; the
        mirror helper is gone and the BuildJob write is the only
        terminal stamp. All five mig 052 terminal columns
        (``emails_fetched``, ``finished_local_at``, ``error_json``,
        ``channel_results_json``, ``tools_invoked_json``) land on
        ``build_jobs`` directly.

        Status → (job_status, job_outcome) mapping (inlined from
        the deleted mirror helper, same table):

          status='success'        → completed, outcome=derived or 'success'
          status='partial'        → completed, outcome=derived or 'partial'
          status='skipped_reauth' → completed, outcome=derived or 'skipped_reauth'
          status='failed'         → failed,    outcome=NULL
          (unknown)               → completed, outcome=derived or <raw>
        """
        from app.db.models import BuildJob

        finished_at = datetime.utcnow()
        # finished_local_at: ISO8601 in the user's tz. Best-effort — falls
        # back to UTC if we can't resolve the tz cheaply.
        finished_local_iso: Optional[str] = None
        try:
            user_id: Optional[str] = None
            async with self._session_maker() as db:
                job_row = await db.get(BuildJob, run_id)
                if job_row is not None:
                    user_id = job_row.user_id
            if user_id is not None:
                tz_str = await self._user_tz_async(user_id)
                tz, _ = _resolve_tz(tz_str, user_id)
                finished_local_iso = finished_at.replace(tzinfo=timezone.utc).astimezone(tz).isoformat()
        except Exception:
            try:
                finished_local_iso = finished_at.isoformat() + "Z"
            except Exception:
                finished_local_iso = None

        # Status → (job_status, job_outcome) mapping, inlined from the
        # deleted ``_mirror_run_terminal_to_job`` helper.
        if status == "failed":
            job_status = "failed"
            job_outcome: Optional[str] = None
        elif status in ("success", "partial", "skipped_reauth"):
            job_status = "completed"
            # Prefer the derived `outcome` value when present
            # (Ticket 2.1 — success_empty, partial, etc.). Fall back
            # to the raw status as a discriminator.
            job_outcome = outcome or status
        else:
            # Unknown status — be conservative and mark completed
            # with the raw value as outcome so we can audit later.
            job_status = "completed"
            job_outcome = outcome or status

        values: dict[str, Any] = {
            "status": job_status,
            "outcome": job_outcome,
            "completed_at": finished_at,
            # Mig 052 terminal columns — written verbatim from the
            # handler-supplied terminal payload.
            "emails_fetched": emails_fetched,
            "finished_local_at": finished_local_iso,
            "error_json": error_json,
            "channel_results_json": channel_results,
            "tools_invoked_json": tools_invoked,
        }
        if error_detail and job_status == "failed":
            # Unified error column on BuildJob — carries the legacy
            # error_detail forward for the dashboard. Truncate to
            # match Postgres column size.
            values["error_message"] = (error_detail or "")[:1000]
        if summary_message_id and job_status == "completed":
            values["summary_message_id"] = summary_message_id

        try:
            async with self._session_maker() as db:
                await db.execute(
                    update(BuildJob)
                    .where(BuildJob.id == run_id)
                    .values(**values)
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
            # Ticket 3 observability — process-lifetime counters for the
            # invariants the missed-fire bug surfaced. Operators read these
            # off the /_runner_status probe to spot drift.
            "missed_fires_total": self._missed_fires_total,
            "reload_failures_total": self._reload_failures_total,
            "reconcile_runs_total": self._reconcile_runs_total,
            "reconcile_active": (
                self._reconcile_task is not None
                and not self._reconcile_task.done()
            ),
            "watchdog_drift_total": self._watchdog_drift_total,
        }
