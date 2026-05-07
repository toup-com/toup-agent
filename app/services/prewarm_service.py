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
from datetime import datetime, timedelta
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
    logger.info("[PREWARM] user=%s queued", str(user_id)[:8])
    return "queued"


async def _run_prewarm(user_id: str) -> None:
    """Background task entry point. Calls provision_container in its own
    session and swallows errors — prewarm is a perf optimisation, not a
    contract-bearing path. The Install-step handler still calls
    provision_container synchronously and is the source of truth for any
    user-visible failure."""
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
            logger.info("[PREWARM] user=%s container provisioned", str(user_id)[:8])
    except Exception as e:
        logger.error(
            "[PREWARM] user=%s task failed: %s",
            str(user_id)[:8], e, exc_info=True,
        )


async def reconcile_stuck_provisioning() -> int:
    """Find managed_containers in status='provisioning' for longer than
    `_STUCK_PROVISIONING_MIN` and re-fire the bridge call. Returns the
    number of rows requeued for telemetry / tests.

    Called every reconciler tick from the rollout reconciler loop
    (mirrors the same lifecycle: in-process loop, dies on redeploy,
    fresh process re-runs on next tick).

    Idempotent: re-firing provision_container on a row that has actually
    finished provisioning since the query is harmless — provision_container
    short-circuits on `status in ('running', 'provisioning')` without
    `recreate=True`.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=_STUCK_PROVISIONING_MIN)
    requeued = 0
    async with async_session_maker() as db:
        result = await db.execute(
            select(ManagedContainer.user_id).where(
                ManagedContainer.status == "provisioning",
                ManagedContainer.updated_at.is_not(None),
                ManagedContainer.updated_at < cutoff,
            )
        )
        stuck_user_ids = [row[0] for row in result.all()]

    if not stuck_user_ids:
        return 0

    for user_id in stuck_user_ids:
        logger.warning(
            "[PREWARM-RECONCILER] user=%s stuck provisioning > %dmin, re-firing",
            str(user_id)[:8], _STUCK_PROVISIONING_MIN,
        )
        # Bump updated_at so we don't re-fire on the very next tick
        # if this re-fire is also still in-flight.
        async with async_session_maker() as db:
            mc = (await db.execute(
                select(ManagedContainer).where(ManagedContainer.user_id == user_id)
            )).scalar_one_or_none()
            if mc:
                mc.error_message = None
                # updated_at bumps via onupdate
                await db.commit()
        asyncio.create_task(_run_prewarm(user_id))
        requeued += 1

    return requeued
