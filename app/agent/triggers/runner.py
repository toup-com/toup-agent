"""TriggerRunner — event-driven dispatch loop for trigger events.

Routines own an APScheduler instance and fire on a cron. Triggers do
not — events arrive via the platform's webhook → agent inbound
endpoint, which INSERTs a `trigger_events` row with `status='queued'`.
This module's job: claim queued rows, run them through the rate
limiter, dispatch to the kind handler, finalise.

The runner exposes two entry points:

  - **`handle_event(event_id)`** — called by the inbound endpoint
    immediately after a successful INSERT, via `asyncio.create_task`.
    Hot-path; non-blocking from the endpoint's perspective. The
    endpoint already returned 200 to Pub/Sub by the time this fires.
  - **`drain_loop`** — periodic background sweep (every 15 s) that
    picks up `status='queued'` rows the inline path missed (race
    window during container restart, or events INSERTed during a
    handler crash). The dedupe gate prevents double-fire.

Restart sweep (`_restart_sweep`) runs once on `start()`. Any row
stuck in `status='running'` from a previous boot older than
ORPHAN_THRESHOLD is flipped to `failed` with
`error_class='agent_restarted'` so it doesn't block forever.

The runner does NOT poll for events on a hot loop — that'd be
wasteful at trigger volumes. The inline `handle_event` path is the
hot path; the periodic drain is the safety net.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from .base_handler import TriggerResult
from .rate_limiter import (
    TriggerRateLimiter,
    parse_rate_limit_config,
)
from .registry import KIND_HANDLERS

logger = logging.getLogger(__name__)


class TriggerRunner:
    """One instance per agent container.

    Construction order at agent_main lifespan:
      1. Build the runner (no args; defaults).
      2. `runner.set_mcp_client(tool_executor.mcp_client)`.
      3. `runner.set_session_maker(async_session_maker)` — optional;
         the runner uses the module proxy by default.
      4. `await runner.start()`.

    The handler modules auto-register at import time. Just importing
    `app.agent.triggers.email_received_handler` populates
    `KIND_HANDLERS['email_received']`.
    """

    ORPHAN_THRESHOLD = timedelta(minutes=10)
    DRAIN_LOOP_INTERVAL_SECONDS = 15.0
    SHUTDOWN_TIMEOUT_SECONDS = 30
    # Retry delays between handler attempts (seconds). Three attempts,
    # exponential backoff. Tests pass tiny values.
    DEFAULT_RETRY_DELAYS = (5.0, 30.0, 120.0)
    # Maximum events to drain in one tick — bounds the latency
    # impact of a backlog catch-up.
    DRAIN_BATCH_LIMIT = 25

    def __init__(
        self,
        session_maker: Optional[async_sessionmaker] = None,
        mcp_client: Any = None,
        retry_delays: Optional[tuple[float, ...]] = None,
    ):
        if session_maker is None:
            from app.db.database import async_session_maker as _proxy
            session_maker = _proxy  # type: ignore[assignment]
        self._session_maker = session_maker
        self._mcp_client = mcp_client
        self._retry_delays = retry_delays or self.DEFAULT_RETRY_DELAYS
        self._limiter = TriggerRateLimiter()
        self._running = False
        self._drain_task: Optional[asyncio.Task] = None
        self._inflight_tasks: set[asyncio.Task] = set()

    # ── Lifecycle ────────────────────────────────────────────────

    def set_mcp_client(self, mcp_client: Any) -> None:
        """Wire the MCP client late — agent_main constructs the runner
        before the MCP client exists. Handlers grab it through us."""
        self._mcp_client = mcp_client
        # Push into the auto-registered handlers — they hold their own
        # reference for the duration of the runner's life.
        for handler in KIND_HANDLERS.values():
            if hasattr(handler, "_mcp_client"):
                handler._mcp_client = mcp_client

    def set_session_maker(self, session_maker: async_sessionmaker) -> None:
        self._session_maker = session_maker

    async def start(self) -> None:
        if self._running:
            return
        swept = await self._restart_sweep()
        await self._warm_rate_buckets()
        self._running = True
        self._drain_task = asyncio.create_task(self._drain_loop())
        logger.info(
            "[trigger_runner] started swept_orphans=%d kinds_registered=%d",
            swept, len(KIND_HANDLERS),
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._inflight_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._inflight_tasks, return_exceptions=True),
                    timeout=self.SHUTDOWN_TIMEOUT_SECONDS,
                )
                logger.info("[trigger_runner] stopped drained=clean")
            except asyncio.TimeoutError:
                logger.warning(
                    "[trigger_runner] stopped drained=partial inflight=%d",
                    len(self._inflight_tasks),
                )
        else:
            logger.info("[trigger_runner] stopped drained=empty")

    # ── Inbound-endpoint hook ────────────────────────────────────

    def handle_event_background(self, job_id: str) -> None:
        """Fire-and-forget entry point used by the inbound endpoint
        and the drain loop.

        PR #46 cutover: the argument is now a ``BuildJob.id`` (the
        new source-of-truth row), NOT a ``TriggerEvent.id`` — the
        intake and queue-scan paths both work in BuildJob ids now.
        The dispatch pipeline below (``_handle_event_with_retry`` →
        ``_dispatch_one``) still operates on ``TriggerEvent.id`` to
        avoid churning the rest of the runner in this PR; we adapt
        by looking up the TriggerEvent.id from ``trigger_events.
        job_id`` (column added in mig 047, populated by
        ``triggers_inbound._idempotent_insert``).
        """
        if not self._running:
            # Pre-start arrival (boot race). The drain loop will pick it
            # up on its first tick.
            return
        task = asyncio.create_task(self._dispatch_for_job(job_id))
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)

    async def _dispatch_for_job(self, job_id: str) -> None:
        """Resolve a BuildJob id to its TriggerEvent id and hand off
        to the retry-aware dispatch loop. Single-row indexed lookup
        on ``trigger_events.job_id``. If no TriggerEvent row exists
        for this Job (mid-PR-#48 transition window when the legacy
        write is gone), we silently bail — the Job-only dispatch
        path lives in PR #48."""
        from app.db.models import TriggerEvent

        async with self._session_maker() as db:
            row = (await db.execute(
                select(TriggerEvent.id).where(TriggerEvent.job_id == job_id)
            )).first()
        if row is None:
            logger.warning(
                "[trigger_runner] no TriggerEvent for job_id=%s — "
                "skipping dispatch (legacy table likely already dropped)",
                job_id,
            )
            return
        await self._handle_event_with_retry(row[0])

    # ── Restart sweep ────────────────────────────────────────────

    async def _restart_sweep(self) -> int:
        """Flip rows stuck in 'running' from a previous boot to
        'failed'. Returns the count for the structured log.

        PR #46 of the unified-jobs arc — this sweep now reads/writes
        ``build_jobs`` (the new source of truth) instead of
        ``trigger_events``. The legacy table is still dual-written
        for one more PR (#48 removes it). We use ``created_at`` as
        the orphan-cutoff proxy on the Job side since BuildJob has
        no ``started_at`` column — a row stuck in ``status='running'``
        with a ``created_at`` older than the threshold is, by
        definition, an orphan from a previous boot.
        """
        from app.db.models import BuildJob, TriggerEvent

        cutoff = datetime.utcnow() - self.ORPHAN_THRESHOLD
        now = datetime.utcnow()
        async with self._session_maker() as db:
            # New source of truth: build_jobs. Return its rowcount.
            res_job = await db.execute(
                update(BuildJob)
                .where(
                    BuildJob.status == "running",
                    BuildJob.source_kind == "trigger",
                    BuildJob.created_at < cutoff,
                )
                .values(
                    status="failed",
                    error_message="agent_restarted: orphaned by agent restart",
                    completed_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            # Legacy dual-write (PR #48 removes this) so the
            # trigger_events table stays consistent for any reader
            # that hasn't migrated yet.
            await db.execute(
                update(TriggerEvent)
                .where(
                    TriggerEvent.status == "running",
                    TriggerEvent.started_at < cutoff,
                )
                .values(
                    status="failed",
                    error_class="agent_restarted",
                    error_detail="orphaned by agent restart",
                    finished_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            await db.commit()
            return getattr(res_job, "rowcount", 0) or 0

    async def _warm_rate_buckets(self) -> None:
        """Seed the per-trigger fire history so a hot rate-limited
        trigger doesn't reset its budget when the container recycles.
        Reads the last hour's worth of fire timestamps from
        ``build_jobs`` (the new source of truth — PR #46 cutover).

        BuildJob has no ``started_at`` column; we use ``created_at``
        as the fire-moment proxy. The limiter is timestamp-relative;
        the small (sub-second) skew between row-insert and
        handler-start is below the resolution that affects rate-limit
        decisions.

        We include status='running' rows alongside terminal ones so
        that an in-flight fire (claimed but not yet completed when the
        container died and restarted) still counts toward the budget.
        """
        from app.db.models import BuildJob

        cutoff = datetime.utcnow() - timedelta(hours=1)
        async with self._session_maker() as db:
            rows = (await db.execute(
                select(BuildJob.source_id, BuildJob.created_at)
                .where(
                    BuildJob.source_kind == "trigger",
                    BuildJob.created_at >= cutoff,
                    BuildJob.status.in_(("completed", "failed", "running")),
                )
            )).all()
        by_trigger: dict[str, list[float]] = {}
        for tid, created_at in rows:
            if tid is None or created_at is None:
                continue
            by_trigger.setdefault(tid, []).append(created_at.timestamp())
        for tid, fires in by_trigger.items():
            self._limiter.warmup(tid, fires)

    # ── Drain loop (background) ─────────────────────────────────

    async def _drain_loop(self) -> None:
        """Periodic safety-net sweep. The hot path is the inline
        `handle_event_background` call from the inbound endpoint; this
        loop only matters for events that arrived during a runner
        crash / container restart."""
        try:
            while self._running:
                try:
                    await asyncio.sleep(self.DRAIN_LOOP_INTERVAL_SECONDS)
                    if not self._running:
                        break
                    queued = await self._fetch_queued_ids(self.DRAIN_BATCH_LIMIT)
                    for job_id in queued:
                        if not self._running:
                            break
                        # PR #46: queued ids are now BuildJob ids.
                        # handle_event_background resolves to a
                        # TriggerEvent.id before dispatching.
                        self.handle_event_background(job_id)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception("[trigger_runner] drain_loop error: %s", e)
        except asyncio.CancelledError:
            return

    async def _fetch_queued_ids(self, limit: int) -> list[str]:
        """Return BuildJob ids of queued trigger-sourced rows, oldest
        first. PR #46 — reads from ``build_jobs`` (new source of
        truth) instead of ``trigger_events``. The returned ids are
        ``BuildJob.id`` values; ``handle_event_background`` resolves
        each to a TriggerEvent.id before dispatching."""
        from app.db.models import BuildJob

        async with self._session_maker() as db:
            rows = (await db.execute(
                select(BuildJob.id)
                .where(
                    BuildJob.status == "queued",
                    BuildJob.source_kind == "trigger",
                )
                .order_by(BuildJob.created_at.asc())
                .limit(limit)
            )).all()
        return [r[0] for r in rows]

    # ── Per-event dispatch ──────────────────────────────────────

    async def _handle_event_with_retry(self, event_id: str) -> None:
        """Top-level dispatch for one event. Three attempts with
        exponential backoff. On terminal failure (all retries exhausted),
        the event row is flipped to ``status='failed'`` with
        ``error_class='all_retries_exhausted'`` and
        ``error_detail=<repr of last exception>``, and the parent trigger
        is stamped with ``last_status='failed'`` + ``last_error`` +
        ``fire_count`` bump. The row is never left in ``status='running'``
        for the 10-minute orphan sweep to find — that path was the cause
        of the "N fires, status=failed, error fields NULL" symptom
        on the live Gmail trigger before this fix.
        """
        last_error_text: Optional[str] = None
        for attempt_idx, delay in enumerate(self._retry_delays):
            try:
                done = await self._dispatch_one(event_id, attempt_idx=attempt_idx)
                if done:
                    return
            except Exception as e:
                logger.exception(
                    "[trigger_runner] dispatch_one crash event_id=%s attempt=%d err=%s",
                    event_id, attempt_idx, e,
                )
                last_error_text = repr(e)[:1000]
            if attempt_idx < len(self._retry_delays) - 1:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
        # All retries exhausted. Terminalise the row + parent trigger so
        # the orphan sweep doesn't have to clean up 10 minutes later
        # with no error context. Idempotent / race-safe inside.
        await self._finalise_exhausted(event_id, last_error_text or "unknown")

    async def _mirror_dispatch_state_to_job(
        self,
        db,
        *,
        event_id: str,
        job_status: Optional[str] = None,
        outcome: Optional[str] = None,
        error_message: Optional[str] = None,
        completed_at: Optional[datetime] = None,
        coalesced_into_job_id_event_id: Optional[str] = None,
        summary_message_id: Optional[str] = None,
    ) -> None:
        """Mirror a dispatch-time state change onto the linked
        ``build_jobs`` row.

        PR #46 of the unified-jobs arc — every UPDATE on
        ``trigger_events`` made by the runner gets a sibling UPDATE
        on the linked ``build_jobs`` row so that ``build_jobs``
        becomes the readable source of truth ahead of PR #48 (which
        removes the legacy writes entirely).

        Distinct from ``_mirror_event_terminal_to_job``:
          - terminal helper maps a TriggerEvent.status terminal value
            (success/failed/skipped_*/coalesced) to (Job.status,
            Job.outcome) and writes the closed set.
          - this helper writes whatever subset of Job fields the
            caller supplies — used for ``running`` transitions and
            for the coalesce/rate-limit/no-handler paths where the
            caller has already done the mapping inline.

        ``coalesced_into_job_id_event_id`` is the *TriggerEvent.id*
        of the coalesce parent — we look it up to BuildJob.id via
        ``trigger_events.job_id``. Naming is awkward but it makes
        the call-site read naturally (the caller knows the parent's
        event_id, not its job_id).

        Best-effort: any exception is logged and swallowed so the
        mirror failure can't fail the handler.
        """
        from app.db.models import BuildJob, TriggerEvent
        try:
            ev = await db.get(TriggerEvent, event_id)
            if ev is None or ev.job_id is None:
                return  # no Job to mirror to — silent no-op
            values: dict[str, Any] = {}
            if job_status is not None:
                values["status"] = job_status
            if outcome is not None:
                values["outcome"] = outcome
            if error_message is not None:
                values["error_message"] = (error_message or "")[:1000]
            if completed_at is not None:
                values["completed_at"] = completed_at
            if summary_message_id is not None:
                values["summary_message_id"] = summary_message_id
            if coalesced_into_job_id_event_id is not None:
                parent_ev = await db.get(
                    TriggerEvent, coalesced_into_job_id_event_id,
                )
                if parent_ev is not None and parent_ev.job_id is not None:
                    values["coalesced_into_job_id"] = parent_ev.job_id
            if not values:
                return
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == ev.job_id)
                .values(**values)
            )
        except Exception as e:
            logger.warning(
                "[trigger_runner] mirror_dispatch_to_job failed "
                "event_id=%s err=%s (legacy fire path unaffected)",
                event_id, e,
            )

    async def _mirror_event_terminal_to_job(
        self,
        db,
        *,
        event_id: str,
        terminal_status: str,
        error_class: Optional[str] = None,
        error_detail: Optional[str] = None,
        summary_message_id: Optional[str] = None,
        finished_at: Optional[datetime] = None,
    ) -> None:
        """Mirror a TriggerEvent's terminal state onto its linked
        ``build_jobs`` row (PR 4a of the unified-jobs arc).

        TriggerEvent has more granular statuses than BuildJob — the
        BuildJob enum is the closed ``queued/running/completed/failed``
        set, with the per-source granularity living on
        ``BuildJob.outcome``. Mapping:

          success          → status='completed', outcome='success'
          failed           → status='failed',    outcome=NULL
          skipped_rate_limit → status='completed', outcome='skipped_rate_limit'
          skipped_filter   → status='completed', outcome='skipped_filter'
          coalesced        → status='completed', outcome='coalesced'

        If the TriggerEvent row has no ``job_id`` (legacy events from
        before PR 4a, or a Job-mint failure at intake), this is a
        no-op — the legacy fire path still works without the mirror.
        Best-effort: any exception is logged and swallowed so a mirror
        failure can't fail the handler.
        """
        from app.db.models import BuildJob, TriggerEvent
        try:
            ev = await db.get(TriggerEvent, event_id)
            if ev is None or ev.job_id is None:
                return  # no Job to mirror to — silent no-op
            if terminal_status == "success":
                job_status, job_outcome = "completed", "success"
            elif terminal_status == "failed":
                job_status, job_outcome = "failed", None
            elif terminal_status in (
                "skipped_rate_limit", "skipped_filter", "coalesced",
            ):
                job_status, job_outcome = "completed", terminal_status
            else:
                # Unknown enum value — be conservative: mark completed
                # with the raw value as outcome so we can audit later.
                job_status, job_outcome = "completed", terminal_status
            values: dict[str, Any] = {
                "status": job_status,
                "outcome": job_outcome,
                "completed_at": finished_at or datetime.utcnow(),
            }
            if error_detail and job_status == "failed":
                # ``error_message`` is the BuildJob-side equivalent of
                # TriggerEvent.error_detail. Mirror it so the dashboard
                # can render a single attribution either way.
                values["error_message"] = (error_detail or "")[:1000]
            if summary_message_id and job_status == "completed":
                values["summary_message_id"] = summary_message_id
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == ev.job_id)
                .values(**values)
            )
        except Exception as e:
            logger.warning(
                "[trigger_runner] mirror_to_job failed event_id=%s err=%s "
                "(legacy fire path unaffected)",
                event_id, e,
            )

    async def _finalise_exhausted(self, event_id: str, error_detail: str) -> None:
        """Mark a ``TriggerEvent`` as failed after the retry loop in
        ``_handle_event_with_retry`` exhausts. Idempotent and race-safe:
        if the row already reached a terminal state (success/failed/
        coalesced/skipped — e.g. a sibling drain claimed it), preserve
        existing state and return without writing. Same for a vanished
        row. The parent ``Trigger`` is only stamped when we actually
        write the event row, so we don't double-bump ``fire_count``."""
        from app.db.models import Trigger, TriggerEvent
        now = datetime.utcnow()
        async with self._session_maker() as db:
            ev = await db.get(TriggerEvent, event_id)
            if ev is None:
                return  # row vanished — nothing to do
            if ev.status not in ("queued", "running"):
                # Already terminal — preserve whatever the racing writer
                # set. This is the same race-safe pattern _dispatch_one
                # uses on step 1 (claim).
                return
            await db.execute(
                update(TriggerEvent)
                .where(TriggerEvent.id == event_id)
                .values(
                    status="failed",
                    error_class="all_retries_exhausted",
                    error_detail=error_detail[:1000],
                    finished_at=now,
                )
            )
            await self._mirror_event_terminal_to_job(
                db,
                event_id=event_id,
                terminal_status="failed",
                error_class="all_retries_exhausted",
                error_detail=error_detail,
                finished_at=now,
            )
            trigger = await db.get(Trigger, ev.trigger_id)
            if trigger is not None:
                await db.execute(
                    update(Trigger)
                    .where(Trigger.id == trigger.id)
                    .values(
                        last_fired_at=now,
                        fire_count=(trigger.fire_count or 0) + 1,
                        last_status="failed",
                        last_error=error_detail[:1000],
                    )
                )
            await db.commit()
        logger.warning(
            "[trigger_runner] terminalised_after_retries event_id=%s detail=%s",
            event_id, error_detail[:200],
        )

    async def _dispatch_one(self, event_id: str, *, attempt_idx: int) -> bool:
        """One attempt at handling an event. Returns True when the
        event reached a terminal state (success/failed/skipped) so the
        retry loop can stop; False when a transient failure should
        retry.
        """
        from app.db.models import Trigger, TriggerEvent

        # ── 1. Claim the row by flipping queued → running ──
        async with self._session_maker() as db:
            ev = await db.get(TriggerEvent, event_id)
            if ev is None:
                return True  # row vanished — nothing to do
            if ev.status not in ("queued", "running"):
                # Already terminal (success/failed/coalesced/skipped).
                # Race: the drain loop and the inline call both fired.
                return True
            trigger = await db.get(Trigger, ev.trigger_id)
            if trigger is None or not trigger.enabled:
                ev.status = "skipped_filter"
                ev.error_class = "trigger_missing_or_disabled"
                ev.finished_at = datetime.utcnow()
                await self._mirror_event_terminal_to_job(
                    db,
                    event_id=event_id,
                    terminal_status="skipped_filter",
                    error_class="trigger_missing_or_disabled",
                    finished_at=ev.finished_at,
                )
                await db.commit()
                return True

        # ── 2. Rate / coalesce gate ──
        rate_cfg = parse_rate_limit_config(
            (trigger.config_json or {}).get("rate_limit")
        )
        decision = self._limiter.gate(trigger.id, event_id, rate_cfg)

        if decision.action == "coalesce":
            # Mark this event coalesced; the in-flight parent will
            # pick it up via limiter.drain_coalesced when it runs.
            _now = datetime.utcnow()
            async with self._session_maker() as db:
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == event_id)
                    .values(
                        status="coalesced",
                        coalesced_into_event_id=decision.parent_event_id,
                        started_at=_now,
                        finished_at=_now,
                    )
                )
                await self._mirror_event_terminal_to_job(
                    db, event_id=event_id, terminal_status="coalesced",
                    finished_at=_now,
                )
                # PR #46 — write the BuildJob-side coalesce pointer
                # alongside the TriggerEvent one. Replaces
                # ``trigger_events.coalesced_into_event_id`` semantics
                # on the new source-of-truth row.
                await self._mirror_dispatch_state_to_job(
                    db, event_id=event_id,
                    coalesced_into_job_id_event_id=decision.parent_event_id,
                )
                await db.commit()
            logger.info(
                "[trigger_runner] coalesced event_id=%s parent=%s trigger=%s",
                event_id, decision.parent_event_id, trigger.id,
            )
            return True

        if decision.action == "rate_limit":
            _now = datetime.utcnow()
            _detail = f"fires_in_window={decision.fires_in_window}"
            async with self._session_maker() as db:
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == event_id)
                    .values(
                        status="skipped_rate_limit",
                        error_class=decision.reason,
                        error_detail=_detail,
                        started_at=_now,
                        finished_at=_now,
                    )
                )
                await self._mirror_event_terminal_to_job(
                    db, event_id=event_id,
                    terminal_status="skipped_rate_limit",
                    error_class=decision.reason,
                    error_detail=_detail,
                    finished_at=_now,
                )
                await db.commit()
            logger.info(
                "[trigger_runner] rate_limit event_id=%s reason=%s fires=%d",
                event_id, decision.reason, decision.fires_in_window,
            )
            return True

        # action == "fire" — proceed to handler.
        # Move the row + any siblings already in the limiter's coalesce
        # bucket into 'running'.
        async with self._session_maker() as db:
            _now_claim = datetime.utcnow()
            await db.execute(
                update(TriggerEvent)
                .where(TriggerEvent.id == event_id)
                .values(status="running", started_at=_now_claim)
            )
            # PR #46 — mirror the queued→running claim onto the
            # BuildJob so the new source-of-truth row reflects the
            # in-flight state. PR #48 removes the TriggerEvent write
            # above; this mirror stays.
            await self._mirror_dispatch_state_to_job(
                db, event_id=event_id, job_status="running",
            )
            await db.commit()

        handler = KIND_HANDLERS.get(trigger.kind)
        if handler is None:
            _now = datetime.utcnow()
            _detail = f"no handler for kind={trigger.kind!r}"
            async with self._session_maker() as db:
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == event_id)
                    .values(
                        status="failed",
                        error_class="no_handler",
                        error_detail=_detail,
                        finished_at=_now,
                    )
                )
                await self._mirror_event_terminal_to_job(
                    db, event_id=event_id, terminal_status="failed",
                    error_class="no_handler",
                    error_detail=_detail,
                    finished_at=_now,
                )
                await db.commit()
            self._limiter.release(trigger.id)
            return True  # terminal — no retry will help

        # Give the handler a small grace to let siblings arrive
        # (the bridge dispatches each gmail message as its own event in
        # rapid succession; we want them to coalesce). 750 ms is enough
        # to capture the back-to-back HTTP calls without adding visible
        # latency.
        await asyncio.sleep(0.75)
        sibling_ids = self._limiter.drain_coalesced(trigger.id)

        # ── 3. Load the event batch ──
        async with self._session_maker() as db:
            ev = await db.get(TriggerEvent, event_id)
            siblings: list[Any] = []
            if sibling_ids:
                sibs = (await db.execute(
                    select(TriggerEvent).where(
                        TriggerEvent.id.in_(sibling_ids)
                    )
                )).scalars().all()
                siblings.extend(sibs)
            events_batch = [ev] + siblings

            # ── 4. Run the handler ──
            try:
                result: TriggerResult = await handler.execute(
                    trigger, events_batch, db,
                )
            except Exception as e:
                logger.exception(
                    "[trigger_runner] handler_crash trigger_id=%s event_id=%s err=%s",
                    trigger.id, event_id, e,
                )
                # Release the limiter and let the retry loop decide. We
                # re-raise (rather than `return False`) so the outer
                # `_handle_event_with_retry` can capture the exception
                # text and, if all retries exhaust, write it into
                # `TriggerEvent.error_detail` instead of leaving the row
                # stuck in `status='running'` with NULL error fields for
                # the 10-minute orphan sweep to find. Behaviour is
                # otherwise unchanged — the outer except already treats
                # both raise and `return False` as "retry me".
                self._limiter.release(trigger.id)
                raise

            # ── 5. Persist outcomes ──
            now = datetime.utcnow()
            for batch_ev in events_batch:
                final = result.per_event_status.get(batch_ev.id, "failed")
                values: dict[str, Any] = {
                    "status": final,
                    "finished_at": now,
                }
                _summary_msg_for_mirror: Optional[str] = None
                _err_class_for_mirror: Optional[str] = None
                _err_detail_for_mirror: Optional[str] = None
                if result.summary_message_id and final == "success":
                    values["summary_message_id"] = result.summary_message_id
                    _summary_msg_for_mirror = result.summary_message_id
                if final != "success" and result.error_class:
                    values["error_class"] = result.error_class
                    values["error_detail"] = (result.error_detail or "")[:1000]
                    _err_class_for_mirror = result.error_class
                    _err_detail_for_mirror = result.error_detail or ""
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == batch_ev.id)
                    .values(**values)
                )
                # Mirror this event's terminal state onto its
                # linked BuildJob (PR 4a unified-jobs arc).
                await self._mirror_event_terminal_to_job(
                    db,
                    event_id=batch_ev.id,
                    terminal_status=final,
                    error_class=_err_class_for_mirror,
                    error_detail=_err_detail_for_mirror,
                    summary_message_id=_summary_msg_for_mirror,
                    finished_at=now,
                )

            # ── 6. Update parent trigger.last_* fields ──
            # The handler reports one of five statuses:
            #   success         — a real event delivered a summary
            #   test_success    — a synthetic Test-button fire wrote a
            #                     wiring-check message (no real delivery)
            #   success_empty   — handler ran clean but produced no
            #                     output (filters dropped all events,
            #                     or every fetch failed). NOT proof the
            #                     trigger works end-to-end.
            #   skipped_reauth  — connector lost auth
            #   failed          — anything else
            #
            # last_status='active' is the user-facing "this trigger is
            # delivering real value" signal. We promote ONLY on real
            # success — test fires and empty batches stay in whatever
            # previous state the trigger was in. The dedicated test
            # fields (last_test_at) and empty-batch counters live in
            # provider_state_json so the UI can show "Test passed —
            # awaiting first real event" without a schema change.
            new_state_patch: dict[str, Any] = dict(trigger.provider_state_json or {})
            if result.new_provider_state:
                new_state_patch.update(result.new_provider_state)

            res_status = result.status
            promote_active = res_status == "success"
            if promote_active:
                next_last_status = "active"
                next_last_error: Optional[str] = None
                new_state_patch["last_real_fired_at"] = now.isoformat() + "Z"
            elif res_status == "test_success":
                # Wiring check passed. Keep last_status untouched (don't
                # downgrade or false-promote). Stamp last_test_at so the
                # UI can show "Test passed at <ts>" honestly.
                next_last_status = trigger.last_status or "never_fired"
                next_last_error = trigger.last_error
                new_state_patch["last_test_at"] = now.isoformat() + "Z"
            elif res_status == "skipped_reauth":
                next_last_status = "skipped_reauth"
                next_last_error = (result.error_detail or "")[:1000]
            elif res_status == "success_empty":
                # Don't claim "active" — nothing was actually delivered.
                # Don't claim "failed" either — the handler ran clean,
                # there just wasn't anything to do. Reflect the run
                # outcome on the event rows and leave last_status as
                # whatever it was before this batch.
                next_last_status = trigger.last_status or "never_fired"
                next_last_error = trigger.last_error
                new_state_patch["last_empty_batch_at"] = now.isoformat() + "Z"
            else:
                next_last_status = "failed"
                next_last_error = (result.error_detail or "")[:1000]

            # fire_count counts every batch the runner dispatched, REAL
            # or synthetic. For an "events that mattered" count, sum
            # real_fire_count + test_fire_count from provider_state_json.
            if res_status == "test_success":
                new_state_patch["test_fire_count"] = (
                    int(new_state_patch.get("test_fire_count") or 0) + 1
                )
            elif res_status == "success":
                new_state_patch["real_fire_count"] = (
                    int(new_state_patch.get("real_fire_count") or 0) + 1
                )

            trigger_values: dict[str, Any] = {
                "last_fired_at": now,
                "fire_count": (trigger.fire_count or 0) + 1,
                "last_status": next_last_status,
                "last_error": next_last_error,
                "provider_state_json": new_state_patch,
            }
            await db.execute(
                update(Trigger)
                .where(Trigger.id == trigger.id)
                .values(**trigger_values)
            )
            await db.commit()

        self._limiter.release(trigger.id)
        logger.info(
            "[trigger_runner] dispatched trigger_id=%s event_id=%s "
            "batch=%d status=%s metrics=%s",
            trigger.id, event_id, len(events_batch), result.status,
            result.metrics,
        )
        return True

    # ── Snapshot / introspection ─────────────────────────────────

    def status_snapshot(self) -> dict:
        return {
            "running": self._running,
            "kinds_registered": list(KIND_HANDLERS.keys()),
            "inflight": len(self._inflight_tasks),
            "tracked_triggers": len(self._limiter._buckets),
        }
