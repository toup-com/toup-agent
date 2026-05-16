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

    def handle_event_background(self, event_id: str) -> None:
        """Fire-and-forget entry point used by the inbound endpoint.

        Schedules `_handle_event_with_retry` on the loop without
        blocking the caller. The endpoint already INSERTed the row;
        we just trigger dispatch.
        """
        if not self._running:
            # Pre-start arrival (boot race). The drain loop will pick it
            # up on its first tick.
            return
        task = asyncio.create_task(self._handle_event_with_retry(event_id))
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)

    # ── Restart sweep ────────────────────────────────────────────

    async def _restart_sweep(self) -> int:
        """Flip rows stuck in 'running' from a previous boot to
        'failed'. Returns the count for the structured log."""
        from app.db.models import TriggerEvent

        cutoff = datetime.utcnow() - self.ORPHAN_THRESHOLD
        async with self._session_maker() as db:
            res = await db.execute(
                update(TriggerEvent)
                .where(
                    TriggerEvent.status == "running",
                    TriggerEvent.started_at < cutoff,
                )
                .values(
                    status="failed",
                    error_class="agent_restarted",
                    error_detail="orphaned by agent restart",
                    finished_at=datetime.utcnow(),
                )
                .execution_options(synchronize_session=False)
            )
            await db.commit()
            return getattr(res, "rowcount", 0) or 0

    async def _warm_rate_buckets(self) -> None:
        """Seed the per-trigger fire history so a hot rate-limited
        trigger doesn't reset its budget when the container recycles.
        Reads the last hour's worth of started_at timestamps for every
        trigger that's seen a fire."""
        from app.db.models import TriggerEvent

        cutoff = datetime.utcnow() - timedelta(hours=1)
        async with self._session_maker() as db:
            rows = (await db.execute(
                select(TriggerEvent.trigger_id, TriggerEvent.started_at)
                .where(
                    TriggerEvent.started_at >= cutoff,
                    TriggerEvent.status.in_(("success", "failed", "running")),
                )
            )).all()
        by_trigger: dict[str, list[float]] = {}
        for tid, started_at in rows:
            if started_at is None:
                continue
            by_trigger.setdefault(tid, []).append(started_at.timestamp())
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
                    for event_id in queued:
                        if not self._running:
                            break
                        self.handle_event_background(event_id)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception("[trigger_runner] drain_loop error: %s", e)
        except asyncio.CancelledError:
            return

    async def _fetch_queued_ids(self, limit: int) -> list[str]:
        from app.db.models import TriggerEvent

        async with self._session_maker() as db:
            rows = (await db.execute(
                select(TriggerEvent.id)
                .where(TriggerEvent.status == "queued")
                .order_by(TriggerEvent.received_at.asc())
                .limit(limit)
            )).all()
        return [r[0] for r in rows]

    # ── Per-event dispatch ──────────────────────────────────────

    async def _handle_event_with_retry(self, event_id: str) -> None:
        """Top-level dispatch for one event. Implements the runner's
        retry policy (three attempts, exponential backoff) around the
        handler.execute call. On terminal failure, the event row is
        flipped to status='failed' and the parent trigger's last_error
        is set."""
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
            if attempt_idx < len(self._retry_delays) - 1:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return

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
            async with self._session_maker() as db:
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == event_id)
                    .values(
                        status="coalesced",
                        coalesced_into_event_id=decision.parent_event_id,
                        started_at=datetime.utcnow(),
                        finished_at=datetime.utcnow(),
                    )
                )
                await db.commit()
            logger.info(
                "[trigger_runner] coalesced event_id=%s parent=%s trigger=%s",
                event_id, decision.parent_event_id, trigger.id,
            )
            return True

        if decision.action == "rate_limit":
            async with self._session_maker() as db:
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == event_id)
                    .values(
                        status="skipped_rate_limit",
                        error_class=decision.reason,
                        error_detail=f"fires_in_window={decision.fires_in_window}",
                        started_at=datetime.utcnow(),
                        finished_at=datetime.utcnow(),
                    )
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
            await db.execute(
                update(TriggerEvent)
                .where(TriggerEvent.id == event_id)
                .values(status="running", started_at=datetime.utcnow())
            )
            await db.commit()

        handler = KIND_HANDLERS.get(trigger.kind)
        if handler is None:
            async with self._session_maker() as db:
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == event_id)
                    .values(
                        status="failed",
                        error_class="no_handler",
                        error_detail=f"no handler for kind={trigger.kind!r}",
                        finished_at=datetime.utcnow(),
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
                # Transient — let the retry loop handle.
                self._limiter.release(trigger.id)
                return False  # retry

            # ── 5. Persist outcomes ──
            now = datetime.utcnow()
            for batch_ev in events_batch:
                final = result.per_event_status.get(batch_ev.id, "failed")
                values: dict[str, Any] = {
                    "status": final,
                    "finished_at": now,
                }
                if result.summary_message_id and final == "success":
                    values["summary_message_id"] = result.summary_message_id
                if final != "success" and result.error_class:
                    values["error_class"] = result.error_class
                    values["error_detail"] = (result.error_detail or "")[:1000]
                await db.execute(
                    update(TriggerEvent)
                    .where(TriggerEvent.id == batch_ev.id)
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
