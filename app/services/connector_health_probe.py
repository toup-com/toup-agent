"""
T5c — Connector health-probe scheduler.

Periodic background loop that walks every active `ConnectorIdentity`,
calls the connector's `provider.health_probe(ctx)`, and:

  - On `HealthResult(ok=True)`: bump `last_health_ok_at` and emit
    `connector_health_probes_total{outcome="ok"}` metric.
  - On `HealthResult(ok=False)`: emit
    `connector_health_probes_total{outcome="fail",detail=...}` and
    bump a per-identity failure counter. After N consecutive
    failures (default 3), flag the identity status as
    `provider_down` so the card UI flips and the dispatcher
    surfaces the right state.

Cadence: 5-minute sweep, randomized stagger so 1000 users don't
all probe at the same instant. Per-connector concurrency cap
(default 10 in flight) so we don't DoS the provider during recovery.

Lifespan: started from platform_main lifespan, owns its own
asyncio.Task. Cancel-safe — `await task.cancel()` cleanly exits.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update as sa_update

from app.connectors.base import ConnectorContext, ConnectorReauthRequired
from app.db.database import async_session_maker
from app.db.models import ConnectorIdentity
from app.services import connector_metrics as _m
from app.services import connector_vault as vault
from app.services.connector_dispatcher import (
    needs_refresh,
    refresh_with_coalescing,
)
from app.services.connector_registry import get_registry

logger = logging.getLogger(__name__)


_DEFAULT_INTERVAL_S = 300            # 5 min between full sweeps
_DEFAULT_CONNECTOR_CONCURRENCY = 10  # per-connector parallel probes
_FAILURE_THRESHOLD = 3               # flips status after this many in a row


class HealthProbeScheduler:
    def __init__(
        self,
        interval_s: int = _DEFAULT_INTERVAL_S,
        connector_concurrency: int = _DEFAULT_CONNECTOR_CONCURRENCY,
        failure_threshold: int = _FAILURE_THRESHOLD,
    ):
        self._interval_s = interval_s
        self._concurrency = connector_concurrency
        self._failure_threshold = failure_threshold
        # connector_id → identity_id → consecutive_failures.
        # Bounded by (connectors × users) which is small.
        self._fail_counts: dict[str, dict[str, int]] = {}
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def run_once(self) -> dict:
        """One full sweep. Returns a summary dict for logging."""
        registry = get_registry()
        connectors = {e.manifest.id: e for e in registry.list_all()}
        if not connectors:
            return {"sweeps": 0, "skipped_reason": "no connectors"}

        async with async_session_maker() as db:
            rows = (
                await db.execute(
                    select(ConnectorIdentity).where(
                        # Probe both active identities and previously-
                        # flipped provider_down ones so the latter
                        # can auto-recover when the upstream comes back.
                        ConnectorIdentity.status.in_(
                            ("active", "provider_down")
                        ),
                    )
                )
            ).scalars().all()

        # Group by connector_id so each provider's concurrency is
        # bounded independently.
        by_connector: dict[str, list[ConnectorIdentity]] = {}
        for r in rows:
            by_connector.setdefault(r.connector_id, []).append(r)

        total_ok = 0
        total_fail = 0
        for connector_id, identities in by_connector.items():
            entry = connectors.get(connector_id)
            if entry is None:
                continue
            sem = asyncio.Semaphore(self._concurrency)
            tasks = [
                self._probe_one(connector_id, ident, entry, sem)
                for ident in identities
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if r is True:
                    total_ok += 1
                else:
                    total_fail += 1
        return {
            "ok": total_ok, "fail": total_fail,
            "connectors": list(by_connector.keys()),
        }

    async def _probe_one(
        self,
        connector_id: str,
        ident: ConnectorIdentity,
        entry,
        sem: asyncio.Semaphore,
    ) -> bool:
        async with sem:
            ctx = ConnectorContext(
                user_id=ident.user_id,
                channel="health_probe",
                request_id=f"health_{int(time.time())}",
            )
            # Refresh-on-expiring BEFORE the probe, using the same
            # helper the dispatcher uses on tool calls. Google access
            # tokens are 1 h-lived; without this step every overnight
            # gap (no user activity → no refresh-on-call) hands an
            # expired token to the provider, the probe 401s, and three
            # consecutive sweeps flip the identity to provider_down.
            # The per-identity lock is module-global in the dispatcher
            # so a concurrent tool call + probe can't double-refresh.
            async with async_session_maker() as db:
                fresh = await vault.get(db, ident.user_id, connector_id)
                if fresh is None:
                    # Identity vanished mid-sweep (user disconnected).
                    # Nothing to probe; don't count against the fail
                    # threshold either.
                    self._fail_counts.get(connector_id, {}).pop(ident.id, None)
                    return True
                if needs_refresh(fresh):
                    fresh, refresh_outcome = await refresh_with_coalescing(
                        db, entry, fresh,
                    )
                    if isinstance(refresh_outcome, ConnectorReauthRequired):
                        # Refresh failed — the helper already moved the
                        # identity to reauth_required (the correct
                        # terminal state). Don't count this as a
                        # transient probe failure or we'd race ourselves
                        # toward provider_down on top of reauth_required.
                        _m.inc(_m.M_HEALTH_PROBES, labels={
                            "connector": connector_id,
                            "outcome": "refresh_failed",
                        })
                        self._fail_counts.get(connector_id, {}).pop(ident.id, None)
                        return False

            provider = entry.provider
            try:
                result = await provider.health_probe(ctx)
            except Exception as e:
                logger.warning(
                    "[health_probe] %s/%s raised: %s",
                    connector_id, ident.user_id[:8], e,
                )
                await self._record_fail(connector_id, ident)
                _m.inc(_m.M_HEALTH_PROBES, labels={
                    "connector": connector_id, "outcome": "exception",
                })
                return False

            if result.ok:
                self._fail_counts.get(connector_id, {}).pop(ident.id, None)
                _m.inc(_m.M_HEALTH_PROBES, labels={
                    "connector": connector_id, "outcome": "ok",
                })
                async with async_session_maker() as db:
                    # Auto-recover from provider_down on a clean probe.
                    # active stays active; revoked/reauth_required don't
                    # revive — those need user action.
                    new_values: dict = {"last_used_at": datetime.utcnow()}
                    if ident.status == "provider_down":
                        new_values["status"] = "active"
                    await db.execute(
                        sa_update(ConnectorIdentity)
                        .where(ConnectorIdentity.id == ident.id)
                        .values(**new_values)
                    )
                    await db.commit()
                return True

            await self._record_fail(connector_id, ident)
            _m.inc(_m.M_HEALTH_PROBES, labels={
                "connector": connector_id, "outcome": "fail",
            })
            return False

    async def _record_fail(self, connector_id: str, ident: ConnectorIdentity) -> None:
        bucket = self._fail_counts.setdefault(connector_id, {})
        bucket[ident.id] = bucket.get(ident.id, 0) + 1
        # Threshold — flip identity to provider_down so the card UI
        # reflects the issue. Awaited inline so the next caller (and
        # test assertions) sees the new status without races.
        if bucket[ident.id] >= self._failure_threshold:
            await self._flip_identity_down(ident.id)

    async def _flip_identity_down(self, identity_id: str) -> None:
        async with async_session_maker() as db:
            await db.execute(
                sa_update(ConnectorIdentity)
                .where(ConnectorIdentity.id == identity_id)
                .values(status="provider_down")
            )
            await db.commit()
        logger.warning(
            "[health_probe] identity %s flipped to provider_down "
            "after %d consecutive failures",
            identity_id, self._failure_threshold,
        )

    async def loop(self) -> None:
        """Long-running coroutine. Stagger first run by [0, interval/2)
        so cold boots don't all probe at t=0."""
        await asyncio.sleep(random.random() * (self._interval_s / 2))
        while not self._stop_event.is_set():
            try:
                summary = await self.run_once()
                logger.info("[health_probe] sweep: %s", summary)
            except Exception as e:
                logger.exception("[health_probe] sweep raised: %s", e)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._interval_s,
                )
                return  # stop_event set during wait
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def start(self) -> asyncio.Task:
        self._stop_event.clear()
        self._task = asyncio.create_task(self.loop(), name="connector_health_probe")
        return self._task


# Module-level singleton for the platform process.
_scheduler: Optional[HealthProbeScheduler] = None


def get_scheduler() -> HealthProbeScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = HealthProbeScheduler()
    return _scheduler


def reset_for_tests() -> None:
    global _scheduler
    _scheduler = None
