"""TriggerRunner — event-driven dispatch loop for trigger events.

Routines own an APScheduler instance and fire on a cron. Triggers do
not — events arrive via the platform's webhook → agent inbound
endpoint, which mints a ``build_jobs`` row with ``status='queued'``.
This module's job: claim queued rows, run them through the rate
limiter, dispatch to the kind handler, finalise.

The runner exposes two entry points:

  - **`handle_event(job_id)`** — called by the inbound endpoint
    immediately after a successful Job mint, via
    ``asyncio.create_task``. Hot-path; non-blocking from the
    endpoint's perspective. The endpoint already returned 200 to
    Pub/Sub by the time this fires.
  - **`drain_loop`** — periodic background sweep (every 15 s) that
    picks up ``status='queued'`` rows the inline path missed (race
    window during container restart, or events minted during a
    handler crash). The dedupe gate prevents double-fire.

Restart sweep (``_restart_sweep``) runs once on ``start()``. Any row
stuck in ``status='running'`` from a previous boot older than
ORPHAN_THRESHOLD is flipped to ``failed`` with a clear
``error_message`` so it doesn't block forever.

The runner does NOT poll for events on a hot loop — that'd be
wasteful at trigger volumes. The inline ``handle_event`` path is the
hot path; the periodic drain is the safety net.

PR #49 of the unified-jobs cutover arc — every read and write
operates directly on ``build_jobs``. The legacy ``trigger_events``
dual-write was removed in this PR; the ORM class still exists but
no production code path touches it. PR #50 drops the table.
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

        PR #49 cutover: the argument is a ``BuildJob.id``. The whole
        dispatch pipeline operates on BuildJob ids natively now — the
        old TriggerEvent.id resolver is gone.
        """
        if not self._running:
            # Pre-start arrival (boot race). The drain loop will pick it
            # up on its first tick.
            return
        task = asyncio.create_task(self._dispatch_for_job(job_id))
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)

    async def _dispatch_for_job(self, job_id: str) -> None:
        """Hand off to the retry-aware dispatch loop. After the PR #49
        cutover the pipeline is BuildJob-native, so this is a thin
        shim retained for the call-site naming symmetry with the
        routine runner."""
        await self._handle_event_with_retry(job_id)

    # ── Restart sweep ────────────────────────────────────────────

    async def _restart_sweep(self) -> int:
        """Flip rows stuck in 'running' from a previous boot to
        'failed'. Returns the count for the structured log.

        PR #49 of the unified-jobs cutover arc — this used to also
        UPDATE the legacy ``trigger_events`` table; that dual-write
        was removed when the runner finished cutting over to
        ``build_jobs`` as the sole source of truth.
        """
        from app.db.models import BuildJob

        cutoff = datetime.utcnow() - self.ORPHAN_THRESHOLD
        now = datetime.utcnow()
        async with self._session_maker() as db:
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
            await db.commit()
            return getattr(res_job, "rowcount", 0) or 0

    async def _warm_rate_buckets(self) -> None:
        """Seed the per-trigger fire history so a hot rate-limited
        trigger doesn't reset its budget when the container recycles.
        Reads the last hour's worth of fire timestamps from
        ``build_jobs`` (the sole source of truth — PR #49 finished
        the cutover).

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
                        self.handle_event_background(job_id)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception("[trigger_runner] drain_loop error: %s", e)
        except asyncio.CancelledError:
            return

    async def _fetch_queued_ids(self, limit: int) -> list[str]:
        """Return BuildJob ids of queued trigger-sourced rows, oldest
        first. PR #46 — reads from ``build_jobs`` (the sole source of
        truth after PR #49)."""
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

    async def _handle_event_with_retry(self, job_id: str) -> None:
        """Top-level dispatch for one job. Three attempts with
        exponential backoff. On terminal failure (all retries exhausted),
        the BuildJob row is flipped to ``status='failed'`` with
        ``error_message`` set, and the parent trigger is stamped with
        ``last_status='failed'`` + ``last_error`` + ``fire_count``
        bump. The row is never left in ``status='running'`` for the
        10-minute orphan sweep to find — that path was the cause of
        the "N fires, status=failed, error fields NULL" symptom on
        the live Gmail trigger before this fix.

        PR #49: parameter and downstream operations are
        ``BuildJob.id``-native.
        """
        last_error_text: Optional[str] = None
        for attempt_idx, delay in enumerate(self._retry_delays):
            try:
                done = await self._dispatch_one(job_id, attempt_idx=attempt_idx)
                if done:
                    return
            except Exception as e:
                logger.exception(
                    "[trigger_runner] dispatch_one crash job_id=%s attempt=%d err=%s",
                    job_id, attempt_idx, e,
                )
                last_error_text = repr(e)[:1000]
                # Only retry genuinely transient classes. This loop used
                # to retry ANY exception, so a 402 out_of_credits — which
                # no amount of retrying can fix — burned every attempt and
                # re-billed tokens each time. Needs-user and terminal
                # classes break out immediately and are finalised below.
                from app.agent.job_status import DISPOSITION_RETRY, classify

                _verdict = classify(e)
                if _verdict.disposition != DISPOSITION_RETRY:
                    logger.info(
                        "[trigger_runner] non-retryable job_id=%s class=%s "
                        "disposition=%s — skipping remaining attempts",
                        job_id, _verdict.error_class, _verdict.disposition,
                    )
                    break
            if attempt_idx < len(self._retry_delays) - 1:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
        # All retries exhausted. Terminalise the row + parent trigger so
        # the orphan sweep doesn't have to clean up 10 minutes later
        # with no error context. Idempotent / race-safe inside.
        await self._finalise_exhausted(job_id, last_error_text or "unknown")

    async def _finalise_exhausted(self, job_id: str, error_detail: str) -> None:
        """Mark a ``BuildJob`` as failed after the retry loop in
        ``_handle_event_with_retry`` exhausts. Idempotent and race-safe:
        if the row already reached a terminal state (completed/failed
        — e.g. a sibling drain claimed it), preserve existing state
        and return without writing. Same for a vanished row. The
        parent ``Trigger`` is only stamped when we actually write the
        Job row, so we don't double-bump ``fire_count``.

        PR #49: legacy ``trigger_events`` UPDATE removed — this used
        to dual-write the same terminal state to TriggerEvent +
        BuildJob; now only the BuildJob path remains.
        """
        from app.db.models import BuildJob, Trigger
        now = datetime.utcnow()
        async with self._session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if job is None:
                return  # row vanished — nothing to do
            if job.status not in ("queued", "running"):
                # Already terminal — preserve whatever the racing writer
                # set. This is the same race-safe pattern _dispatch_one
                # uses on step 1 (claim).
                return
            # Classify before writing. Pre-2026-07-29 this wrote
            # `"all_retries_exhausted: " + repr(exc)` straight into
            # error_message, which the app rendered verbatim — that is
            # how `all_retries_exhausted: AttributeError("'BuildJob'
            # object has no attribute 'event_dedupe_id'")` reached the
            # founder's phone. error_message is still written so legacy
            # readers and internal debugging keep working; user_message
            # is what the client renders.
            from app.agent.job_status import (
                DISPOSITION_NEEDS_USER, STATUS_WAITING_ON_USER, classify,
            )

            _verdict = classify(error_detail)
            _terminal_status = (
                STATUS_WAITING_ON_USER
                if _verdict.disposition == DISPOSITION_NEEDS_USER
                else "failed"
            )
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id)
                .values(
                    status=_terminal_status,
                    error_message=("all_retries_exhausted: " + error_detail)[:1000],
                    error_class=_verdict.error_class,
                    user_message=_verdict.user_message,
                    technical_detail=("all_retries_exhausted: " + error_detail)[:2000],
                    # A job parked on the user is NOT finished — leaving
                    # completed_at NULL keeps it out of "Recent".
                    completed_at=(
                        None if _terminal_status == STATUS_WAITING_ON_USER else now
                    ),
                )
            )

            # A job parked on the user must announce itself, or the only
            # trace is a row nobody is looking at. `needs_input` keeps the
            # Live Activity card ALIVE (unlike mission_failed, which ends
            # it) — correct, because this job resumes when the user acts.
            if _terminal_status == STATUS_WAITING_ON_USER and _verdict.required_action:
                try:
                    from app.agent.subagent_orchestrator import notify_job_needs_user

                    await notify_job_needs_user(
                        job_id=job_id,
                        label=job.title,
                        summary=_verdict.user_message or "Your agent needs your input.",
                        action_type=_verdict.required_action,
                    )
                except Exception as _ne:  # noqa: BLE001
                    logger.debug("[trigger_runner] waiting notify skipped: %s", _ne)

            trigger = await db.get(Trigger, job.source_id)
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
            "[trigger_runner] terminalised_after_retries job_id=%s detail=%s",
            job_id, error_detail[:200],
        )

    async def _dispatch_one(self, job_id: str, *, attempt_idx: int) -> bool:
        """One attempt at handling a queued trigger Job. Returns True
        when the row reached a terminal state (completed/failed) so
        the retry loop can stop; False when a transient failure
        should retry.

        PR #49 cutover: every UPDATE here targets ``build_jobs``
        directly. The legacy TriggerEvent dual-write and the
        ``_mirror_event_terminal_to_job`` /
        ``_mirror_dispatch_state_to_job`` helpers were removed once
        the read path no longer needs them.

        Status mapping (from handler result → BuildJob):
          success            → status='completed', outcome='success'
          failed             → status='failed',    outcome=NULL
          skipped_rate_limit → status='completed', outcome='skipped_rate_limit'
          skipped_filter     → status='completed', outcome='skipped_filter'
          coalesced          → status='completed', outcome='coalesced',
                               coalesced_into_job_id=<parent>
        """
        from app.db.models import BuildJob, Trigger

        # ── 1. Claim the row by flipping queued → running ──
        async with self._session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if job is None:
                return True  # row vanished — nothing to do
            if job.status not in ("queued", "running"):
                # Already terminal (completed/failed). Race: the drain
                # loop and the inline call both fired.
                return True
            trigger = await db.get(Trigger, job.source_id)
            if trigger is None or not trigger.enabled:
                _now = datetime.utcnow()
                await db.execute(
                    update(BuildJob)
                    .where(BuildJob.id == job_id)
                    .values(
                        status="completed",
                        outcome="skipped_filter",
                        error_message="trigger_missing_or_disabled",
                        completed_at=_now,
                    )
                )
                await db.commit()
                return True

        # ── 2. Rate / coalesce gate ──
        rate_cfg = parse_rate_limit_config(
            (trigger.config_json or {}).get("rate_limit")
        )
        decision = self._limiter.gate(trigger.id, job_id, rate_cfg)

        if decision.action == "coalesce":
            # Mark this job coalesced; the in-flight parent will pick
            # it up via limiter.drain_coalesced when it runs. The
            # rate-limiter still returns the *parent's job_id* in
            # ``parent_event_id`` (legacy field name on the limiter
            # decision — internal to that module; PR #49 leaves the
            # field name untouched to keep the diff focused).
            _now = datetime.utcnow()
            async with self._session_maker() as db:
                await db.execute(
                    update(BuildJob)
                    .where(BuildJob.id == job_id)
                    .values(
                        status="completed",
                        outcome="coalesced",
                        coalesced_into_job_id=decision.parent_event_id,
                        completed_at=_now,
                    )
                )
                await db.commit()
            logger.info(
                "[trigger_runner] coalesced job_id=%s parent=%s trigger=%s",
                job_id, decision.parent_event_id, trigger.id,
            )
            return True

        if decision.action == "rate_limit":
            _now = datetime.utcnow()
            _detail = (
                f"{decision.reason}: fires_in_window={decision.fires_in_window}"
            )
            async with self._session_maker() as db:
                await db.execute(
                    update(BuildJob)
                    .where(BuildJob.id == job_id)
                    .values(
                        status="completed",
                        outcome="skipped_rate_limit",
                        error_message=_detail[:1000],
                        completed_at=_now,
                    )
                )
                await db.commit()
            logger.info(
                "[trigger_runner] rate_limit job_id=%s reason=%s fires=%d",
                job_id, decision.reason, decision.fires_in_window,
            )
            return True

        # action == "fire" — proceed to handler.
        # Move the row + any siblings already in the limiter's coalesce
        # bucket into 'running'.
        async with self._session_maker() as db:
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id)
                .values(status="running")
            )
            await db.commit()

        handler = KIND_HANDLERS.get(trigger.kind)
        if handler is None:
            _now = datetime.utcnow()
            _detail = f"no handler for kind={trigger.kind!r}"
            async with self._session_maker() as db:
                await db.execute(
                    update(BuildJob)
                    .where(BuildJob.id == job_id)
                    .values(
                        status="failed",
                        error_message=_detail[:1000],
                        completed_at=_now,
                    )
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

        # ── 3. Load the job batch ──
        # The handler signature is ``execute(trigger, events_batch,
        # db)`` — ``events_batch`` is a list of row objects. PR #49
        # passes BuildJob rows directly; the legacy TriggerEvent
        # shape is gone. Handlers read row.id (used to key
        # ``per_event_status``) and don't depend on the legacy
        # TriggerEvent-specific columns (audit-verified during the
        # PR #46 cutover; PR #49 just renames the binding).
        async with self._session_maker() as db:
            job = await db.get(BuildJob, job_id)
            siblings: list[Any] = []
            if sibling_ids:
                sibs = (await db.execute(
                    select(BuildJob).where(
                        BuildJob.id.in_(sibling_ids)
                    )
                )).scalars().all()
                siblings.extend(sibs)
            events_batch = [job] + siblings

            # ── 4. Run the handler ──
            try:
                result: TriggerResult = await handler.execute(
                    trigger, events_batch, db,
                )
            except Exception as e:
                logger.exception(
                    "[trigger_runner] handler_crash trigger_id=%s job_id=%s err=%s",
                    trigger.id, job_id, e,
                )
                # Release the limiter and let the retry loop decide. We
                # re-raise (rather than `return False`) so the outer
                # `_handle_event_with_retry` can capture the exception
                # text and, if all retries exhaust, write it into
                # ``BuildJob.error_message`` instead of leaving the row
                # stuck in ``status='running'`` with NULL error fields
                # for the 10-minute orphan sweep to find. Behaviour is
                # otherwise unchanged — the outer except already treats
                # both raise and `return False` as "retry me".
                self._limiter.release(trigger.id)
                raise

            # ── 5. Persist outcomes ──
            # PR #49: inline the status → (status, outcome) mapping
            # that the old ``_mirror_event_terminal_to_job`` helper
            # used to do. Same mapping table — see module docstring.
            now = datetime.utcnow()
            for batch_job in events_batch:
                final = result.per_event_status.get(batch_job.id, "failed")
                if final == "success":
                    job_status, job_outcome = "completed", "success"
                elif final == "failed":
                    job_status, job_outcome = "failed", None
                elif final in (
                    "skipped_rate_limit", "skipped_filter", "coalesced",
                ):
                    job_status, job_outcome = "completed", final
                else:
                    # Unknown enum value — be conservative: mark
                    # completed with the raw value as outcome so we
                    # can audit later. Matches the legacy mirror's
                    # fallback branch.
                    job_status, job_outcome = "completed", final

                values: dict[str, Any] = {
                    "status": job_status,
                    "outcome": job_outcome,
                    "completed_at": now,
                }
                if result.summary_message_id and final == "success":
                    values["summary_message_id"] = result.summary_message_id
                if final != "success" and result.error_class:
                    # ``error_message`` on BuildJob is the unified
                    # error column; carry the class + detail forward.
                    _err_detail = (result.error_detail or "")[:1000]
                    values["error_message"] = _err_detail or result.error_class
                await db.execute(
                    update(BuildJob)
                    .where(BuildJob.id == batch_job.id)
                    .values(**values)
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
            "[trigger_runner] dispatched trigger_id=%s job_id=%s "
            "batch=%d status=%s metrics=%s",
            trigger.id, job_id, len(events_batch), result.status,
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
