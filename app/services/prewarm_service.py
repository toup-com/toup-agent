"""Prewarm service — fire managed-container provisioning ahead of the
Install step so the user's agent is fully booted by the time they
finish onboarding.

Two callers:
  - `POST /agent-setup/prewarm` HTTP endpoint (frontend trigger)
  - `Soul.save` handler in-process (backend trigger, gated by
    `settings.prewarm_on_soul_save`)

Contract:
  - Idempotent on user_id. Safe to call repeatedly.
  - Marks `managed_containers.status='provisioning'` BEFORE the bridge
    call. The 30s reconciler tick (see rollout_service `_reconcile_once`)
    re-fires this if the asyncio task dies (Railway redeploy mid-flight).
  - Fire-and-forget: the caller is not blocked on bridge latency.
  - Self-hosted users: no-op. Bundle-without-active-subscription users:
    no-op (the bundle gate at /managed-agent/provision still applies for
    the actual user-initiated provision).
  - Errors logged, never raised — prewarm is a perf optimisation; if it
    fails, the existing Install-step provision call is still the
    contract-bearing path.

Background-task durability rule (from docs/security/audit-backlog.md
architectural notes):

  asyncio.create_task dies on Railway redeploy. We mark
  managed_containers.status='provisioning' BEFORE the bridge call so
  the durable signal is in the DB. The reconciler picks up rows with
  status='provisioning' AND updated_at < now() - 5min and re-fires.

This file only schedules and reconciles. Actual bridge work lives in
docker_host_service.provision_container which is already idempotent.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig
from app.db.models.platform import ManagedContainer

logger = logging.getLogger(__name__)


# How stale a 'provisioning' row must be before the reconciler retries
# its bridge call. Five minutes covers a Railway redeploy + warm-start
# (typically <2 min) without piling reprovision attempts on a tenant
# whose bridge call is still legitimately in-flight (~25-40s).
_STUCK_PROVISIONING_MIN = 5


# How long to poll the agent's /agent/health for boot_ready=true after
# provision_container returns. Container start typically takes 13-20s;
# we cap at 60s so a stuck boot still emits a "[BOOT-READY-TIMEOUT]"
# marker the aggregator can flag rather than silently leaving prewarm
# without a paired BOOT-READY line.
_BOOT_READY_POLL_TIMEOUT_S = 60
_BOOT_READY_POLL_INTERVAL_S = 1.5


def _iso_now() -> str:
    """ISO-8601 UTC timestamp suitable for log-line aggregation. The
    aggregator parses ts=<...> tokens; using a single helper keeps the
    format identical across PREWARM-START / BOOT-READY / WAKE-CLICK."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


PrewarmStatus = Literal["queued", "already_running", "skipped"]


def _should_skip_prewarm(agent_config: AgentConfig | None) -> str | None:
    """Return a short skip-reason string, or None if prewarm should fire."""
    if not agent_config:
        return "no_agent_config"
    if agent_config.hosting_mode != "managed":
        return "self_hosted"
    if (
        agent_config.llm_mode == "bundle"
        and agent_config.bundle_status not in ("active", "cancelling")
    ):
        # Bundle gate would 402 at /managed-agent/provision. Don't pre-warm
        # a 402 — the user is mid-payment-flow and we'd just churn the
        # bridge for nothing. The existing Welcome-mount prewarm has the
        # same gate. (See OnboardingShell.tsx prewarmContainerOnce.)
        return "bundle_pending_payment"
    return None


async def schedule_prewarm(user_id: str) -> PrewarmStatus:
    """Schedule a fire-and-forget container prewarm for `user_id`.

    Returns immediately. The actual bridge call runs on a background
    asyncio task. If `user_id` already has a running container, returns
    "already_running" without scheduling anything.

    Caller MUST have committed any AgentConfig mutations they want the
    prewarm to see — this function opens its own session, so an
    uncommitted mutation in the caller's session would not be visible.
    """
    async with async_session_maker() as db:
        agent_config = (
            await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )
        ).scalar_one_or_none()

        skip_reason = _should_skip_prewarm(agent_config)
        if skip_reason:
            logger.info(
                "[PREWARM] user=%s skipped reason=%s",
                str(user_id)[:8], skip_reason,
            )
            return "skipped"

        # Check current container state. provision_container is idempotent
        # but we want a clear status hint for the caller's response.
        existing = (
            await db.execute(
                select(ManagedContainer).where(ManagedContainer.user_id == user_id)
            )
        ).scalar_one_or_none()
        if existing and existing.status == "running":
            return "already_running"

        # Mark provisioning BEFORE the bridge call so the reconciler can
        # detect a stuck/aborted task. provision_container itself also
        # writes status, but the durable marker must exist before the
        # task is even scheduled — otherwise a Railway redeploy in the
        # ~10ms gap between task creation and the bridge call leaves no
        # signal for recovery.
        if existing:
            existing.status = "provisioning"
            existing.error_message = None
            # updated_at auto-bumps via the model's onupdate hook
            await db.commit()
        # If `existing` is None, provision_container will INSERT the row
        # itself with status='provisioning'. No pre-INSERT here; doing it
        # would race the bridge port allocation.

    # asyncio.create_task fires AFTER the session is closed and the
    # commit is durable. The task opens its own session.
    asyncio.create_task(_run_prewarm(user_id))
    # Telemetry marker: prewarm trigger time. Pair with the matching
    # [BOOT-READY] line (same user prefix) to compute boot duration,
    # and with [WAKE-CLICK] to compute the user-facing gap. Format
    # parsed by scripts/prewarm_aggregator.py — keep stable.
    logger.info(
        "[PREWARM-START] user=%s ts=%s",
        str(user_id)[:8], _iso_now(),
    )
    return "queued"


async def _run_prewarm(user_id: str) -> None:
    """Background task entry point. Calls provision_container in its own
    session and swallows errors — prewarm is a perf optimisation, not a
    contract-bearing path. The Install-step handler still calls
    provision_container synchronously and is the source of truth for any
    user-visible failure.

    After provision_container returns, polls the agent's `/agent/health`
    until `boot_progress.ready=true`, then emits `[BOOT-READY]`. The
    aggregator pairs this with the prior `[PREWARM-START]` (boot
    duration) and the later `[WAKE-CLICK]` (user-visible gap)."""
    try:
        async with async_session_maker() as db:
            agent_config = (
                await db.execute(
                    select(AgentConfig).where(AgentConfig.user_id == user_id)
                )
            ).scalar_one_or_none()
            if not agent_config:
                logger.warning("[PREWARM] user=%s task aborted: no agent_config", str(user_id)[:8])
                return
            # Late-import to break the docker_host_service ↔ scheduled_tasks
            # import cycle that otherwise surfaces at platform-api boot.
            from app.services import docker_host_service
            await docker_host_service.provision_container(db, user_id, agent_config)
            # Refresh to pick up the agent_url the bridge wrote.
            await db.refresh(agent_config)
            agent_url = agent_config.agent_url
            agent_api_key = agent_config.agent_api_key
        # Poll /agent/health for boot_ready outside the session — keeps
        # the DB connection idle while we wait on network I/O.
        await _await_boot_ready(user_id, agent_url, agent_api_key)
    except Exception as e:
        logger.error(
            "[PREWARM] user=%s task failed: %s",
            str(user_id)[:8], e, exc_info=True,
        )


async def _await_boot_ready(
    user_id: str, agent_url: str | None, agent_api_key: str | None,
) -> None:
    """Poll the agent's /agent/health until boot_progress.ready=true,
    then emit the [BOOT-READY] telemetry marker. Bounded by
    _BOOT_READY_POLL_TIMEOUT_S; emits [BOOT-READY-TIMEOUT] on miss so
    the aggregator can distinguish "boot took too long" from "prewarm
    crashed before reaching boot."""
    if not agent_url:
        logger.warning(
            "[BOOT-READY-SKIPPED] user=%s reason=no_agent_url",
            str(user_id)[:8],
        )
        return
    import httpx as _httpx
    headers = {}
    if agent_api_key:
        headers["X-Agent-Key"] = agent_api_key
    deadline = asyncio.get_event_loop().time() + _BOOT_READY_POLL_TIMEOUT_S
    last_phase: str | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with _httpx.AsyncClient(timeout=4) as client:
                resp = await client.get(
                    f"{agent_url}/agent/health", headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    boot = data.get("boot_progress", {}) or {}
                    last_phase = boot.get("phase") or last_phase
                    if boot.get("ready", False):
                        logger.info(
                            "[BOOT-READY] user=%s ts=%s phase=%s",
                            str(user_id)[:8], _iso_now(),
                            boot.get("phase", "ready"),
                        )
                        return
        except Exception:
            # Container still starting / Caddy route still propagating.
            # Loop until the deadline; don't spam the log on transient
            # connection errors.
            pass
        await asyncio.sleep(_BOOT_READY_POLL_INTERVAL_S)
    logger.warning(
        "[BOOT-READY-TIMEOUT] user=%s ts=%s last_phase=%s timeout_s=%d",
        str(user_id)[:8], _iso_now(), last_phase, _BOOT_READY_POLL_TIMEOUT_S,
    )


async def _is_agent_actually_healthy(user_id: str) -> bool:
    """Return True iff the agent's `/agent/health` reports
    `boot_progress.ready=true`. Fail-soft on any error — caller should
    treat False as "couldn't verify, fall through to bridge re-fire."
    Uses the agent_url + agent_api_key already written by an earlier
    successful provision (we only get here after status was once set
    to 'provisioning', meaning provision_container at least started)."""
    try:
        async with async_session_maker() as db:
            agent_config = (
                await db.execute(
                    select(AgentConfig).where(AgentConfig.user_id == user_id)
                )
            ).scalar_one_or_none()
            if not agent_config or not agent_config.agent_url:
                return False
            agent_url = agent_config.agent_url
            agent_api_key = agent_config.agent_api_key
        import httpx as _httpx
        headers = {"X-Agent-Key": agent_api_key} if agent_api_key else {}
        async with _httpx.AsyncClient(timeout=4) as client:
            resp = await client.get(f"{agent_url}/agent/health", headers=headers)
            if resp.status_code != 200:
                return False
            data = resp.json()
            boot = data.get("boot_progress", {}) or {}
            if not bool(boot.get("ready", False)):
                return False
            # Bind-aware readiness (agent /agent/health Change 3). A GENERIC,
            # not-yet-bound pool container reports boot.ready=true but is NOT
            # ready for THIS user — its first message would 401 on a stale
            # token. Require it to be bound to the claiming user. Backward-
            # compatible: an OLD agent image omits these fields, so we keep
            # trusting `ready` alone (no false re-provisions mid-rollout).
            if "is_bound" in data:
                if not data.get("is_bound"):
                    return False
                _bound_uid = data.get("bound_user_id")
                if _bound_uid not in (None, user_id):
                    return False
            return True
    except Exception as e:
        logger.info(
            "[PREWARM-RECONCILER] user=%s health-check failed (will re-fire): %s",
            str(user_id)[:8], e,
        )
        return False


# NOTE (bulletproof plan M, 2026-07-04): reconcile_stuck_provisioning was
# deleted. pool_service.reclaim_stranded_users (180s container-reconciler
# tick) owns stuck-provisioning recovery now — two reconciliation systems
# racing the same rows meant one could re-fire a cold provision while the
# other was mid pool-claim. One reconciliation system only.
