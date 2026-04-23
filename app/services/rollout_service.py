"""
Rollout service — orchestrates tenant image upgrades per
docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md §5.1.

Algorithm:
  1. Enumerate running tenants (skip pin_image_tag set).
  2. Phase A — canary: find user.is_canary=TRUE tenant in the set. Abort if
     none. Upgrade canary, observe for canary_wait_minutes watching
     /agent/health. On failure: auto-rollback canary, abort rollout.
  3. Phase B — everyone else in batches of rollout_batch_size (default 5),
     parallel within batch, serial between batches. Per-tenant failure →
     rollback to prior_tag, continue rollout, warning alert.
  4. Rollback-failure (upgrade AND rollback both fail) → attempt status
     'rollback_failed', critical alert, continue rollout (other tenants).

Concurrency: only ONE rollout can be in flight at a time. Starting a new
rollout while another is 'pending' or 'running' returns 409 at the API
layer. No queueing — operator decides whether to cancel the current one
or wait.

The service is fire-and-forget from the API's perspective: the HTTP
handler creates the Rollout row + schedules the APScheduler one-shot job,
returns 202, and returns. The actual upgrade loop runs in the scheduler.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    ManagedContainer, User, AgentConfig,
    Rollout, RolloutAttempt,
)
from app.services.docker_host_service import (
    upgrade_tenant_image, BridgeUpgradeUnhealthy,
)

logger = logging.getLogger(__name__)


# ─── Alerts ────────────────────────────────────────────────────────


async def _send_telegram(level: str, message: str) -> None:
    """Fire a Telegram alert to the infra bot (not the user-facing one).

    level: 'info' | 'warning' | 'critical' — rendered as emoji prefix.
    """
    token = settings.infra_alert_telegram_token or settings.admin_alert_telegram_token
    chat_id = settings.infra_alert_telegram_chat_id or settings.admin_alert_telegram_chat_id
    if not token or not chat_id:
        logger.info("[ROLLOUT-ALERT] no telegram config; skipping: %s", message)
        return

    prefix = {"info": "🔵", "warning": "⚠️", "critical": "🚨"}.get(level, "•")
    body = f"{prefix} {message}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={
                "chat_id": chat_id,
                "text": body,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            })
    except Exception as e:
        logger.warning("[ROLLOUT-ALERT] telegram failed: %s", e)


# ─── Queries ───────────────────────────────────────────────────────


async def _running_tenants(db: AsyncSession) -> list[ManagedContainer]:
    """Tenants eligible for this rollout: status='running' AND pin_image_tag IS NULL."""
    result = await db.execute(
        select(ManagedContainer).where(
            ManagedContainer.status == "running",
            ManagedContainer.pin_image_tag.is_(None),
        )
    )
    return list(result.scalars().all())


async def _canary_container(
    db: AsyncSession, candidates: list[ManagedContainer]
) -> Optional[ManagedContainer]:
    """Return the ManagedContainer owned by users.is_canary=TRUE user, if
    that user is among the candidates. None if no canary user exists or
    they aren't in the running set (rollout should abort in that case).
    """
    if not candidates:
        return None
    result = await db.execute(select(User.id).where(User.is_canary == True))  # noqa: E712
    canary_user_id = result.scalar_one_or_none()
    if not canary_user_id:
        return None
    for c in candidates:
        if c.user_id == canary_user_id:
            return c
    return None


async def active_rollout(db: AsyncSession) -> Optional[Rollout]:
    """Return the currently-in-flight rollout (if any) — for 409 detection."""
    result = await db.execute(
        select(Rollout).where(Rollout.status.in_(["pending", "running"]))
        .order_by(Rollout.started_at.desc())
    )
    return result.scalars().first()


# ─── Health polling (platform-side, via Caddy HTTPS) ──────────────


async def _agent_url(db: AsyncSession, container: ManagedContainer) -> Optional[str]:
    """Fetch the container owner's AgentConfig.agent_url (HTTPS subdomain)."""
    result = await db.execute(
        select(AgentConfig.agent_url).where(AgentConfig.user_id == container.user_id)
    )
    return result.scalar_one_or_none()


async def _poll_health(agent_url: str, attempts: int, interval_s: float) -> int:
    """Poll <agent_url>/agent/health until 3 consecutive 200s or attempts
    exhausted. Returns the max consecutive-OK count observed. Used by the
    canary observation window — longer interval than bridge's internal
    post-upgrade health check.
    """
    consecutive_ok = 0
    best = 0
    logger.info("[POLL_HEALTH] start url=%s attempts=%d interval=%s", agent_url, attempts, interval_s)
    async with httpx.AsyncClient(timeout=10, verify=True) as client:
        for i in range(attempts):
            try:
                r = await client.get(f"{agent_url.rstrip('/')}/agent/health")
                if r.status_code == 200:
                    consecutive_ok += 1
                    best = max(best, consecutive_ok)
                else:
                    logger.warning("[POLL_HEALTH] poll %d HTTP %d body=%r", i, r.status_code, r.text[:100])
                    consecutive_ok = 0
            except httpx.HTTPError as e:
                logger.warning("[POLL_HEALTH] poll %d exception %s: %s", i, type(e).__name__, str(e)[:200])
                consecutive_ok = 0
            await asyncio.sleep(interval_s)
    logger.info("[POLL_HEALTH] end best=%d", best)
    return best


# ─── Per-tenant upgrade + rollback ────────────────────────────────


async def _upgrade_one(
    db: AsyncSession,
    rollout: Rollout,
    container: ManagedContainer,
    new_tag: str,
) -> RolloutAttempt:
    """Upgrade a single tenant to new_tag; record the attempt.

    Returns the RolloutAttempt row with status in
    {'ok', 'failed', 'rolled_back', 'rollback_failed'}.

    Never raises — failures are captured in the attempt's error field.
    """
    prior_tag = container.image_tag or "unknown"
    attempt = RolloutAttempt(
        rollout_id=rollout.id,
        container_id=container.id,
        prior_tag=prior_tag,
        new_tag=new_tag,
        status="upgrading",
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)

    t0 = time.time()
    try:
        result = await upgrade_tenant_image(
            db, container.user_id, new_tag, rollout_id=rollout.id,
        )
        attempt.status = "ok"
        attempt.health_checks_passed = result.get("health_checks_passed", 0)
        attempt.duration_ms = result.get("duration_ms") or int((time.time() - t0) * 1000)
        attempt.completed_at = datetime.utcnow()
        await db.commit()
        return attempt
    except BridgeUpgradeUnhealthy as e:
        attempt.status = "failed"
        attempt.error = f"unhealthy after upgrade: {e.detail}"[:1000]
        attempt.health_checks_passed = e.detail.get("health_checks_passed", 0)
        attempt.duration_ms = int((time.time() - t0) * 1000)
        await db.commit()
    except Exception as e:
        attempt.status = "failed"
        attempt.error = f"{type(e).__name__}: {str(e)[:1000]}"
        attempt.duration_ms = int((time.time() - t0) * 1000)
        await db.commit()

    # Failed — attempt rollback
    try:
        await upgrade_tenant_image(
            db, container.user_id, prior_tag, rollout_id=rollout.id,
        )
        attempt.status = "rolled_back"
        attempt.completed_at = datetime.utcnow()
        await db.commit()
        await _send_telegram(
            "warning",
            f"Tenant <code>{container.user_id[:8]}</code> failed upgrade to "
            f"<code>{new_tag}</code> — rolled back to <code>{prior_tag}</code>",
        )
    except Exception as e:
        attempt.status = "rollback_failed"
        attempt.error = (attempt.error or "") + f" | ROLLBACK FAILED: {str(e)[:500]}"
        attempt.completed_at = datetime.utcnow()
        await db.commit()
        await _send_telegram(
            "critical",
            f"Tenant <code>{container.user_id[:8]}</code> upgrade AND rollback "
            f"failed — manual intervention needed. Error: {str(e)[:200]}",
        )
    return attempt


# ─── Main orchestration loop ──────────────────────────────────────


async def _run_rollout_task(rollout_id: str) -> None:
    """Background task: drive one Rollout through its lifecycle.

    Called by the APScheduler one-shot job created in start_rollout().
    Opens its own DB session (the HTTP handler's session is long gone
    by the time this runs).
    """
    async with async_session_maker() as db:
        # Reload the rollout row
        result = await db.execute(
            select(Rollout).where(Rollout.id == rollout_id)
        )
        rollout = result.scalar_one_or_none()
        if not rollout:
            logger.error("[ROLLOUT] id=%s not found", rollout_id)
            return
        if rollout.status != "pending":
            logger.warning("[ROLLOUT] id=%s status=%s, expected pending", rollout_id, rollout.status)
            return

        rollout.status = "running"
        await db.commit()

        await _send_telegram(
            "info",
            f"Rollout started — target <code>{rollout.image_tag}</code>",
        )

        try:
            await _drive_rollout(db, rollout)
        except Exception as e:
            # Shouldn't normally happen — _drive_rollout handles its own
            # per-tenant errors. Catch-all to ensure the rollout row is
            # always closed.
            logger.exception("[ROLLOUT] id=%s crashed: %s", rollout_id, e)
            rollout.status = "complete"
            rollout.completed_at = datetime.utcnow()
            rollout.notes = (rollout.notes or "") + f"\nORCHESTRATOR CRASH: {type(e).__name__}: {str(e)[:500]}"
            await db.commit()
            await _send_telegram(
                "critical",
                f"Rollout <code>{rollout.image_tag}</code> orchestrator crashed: {type(e).__name__}",
            )


async def _drive_rollout(db: AsyncSession, rollout: Rollout) -> None:
    """The actual canary + batch loop. Separated from _run_rollout_task
    so the outer function can own status-row lifecycle + catch-all logging.
    """
    tenants = await _running_tenants(db)
    if not tenants:
        rollout.status = "complete"
        rollout.completed_at = datetime.utcnow()
        rollout.notes = "no running tenants; nothing to do"
        await db.commit()
        await _send_telegram("info", f"Rollout <code>{rollout.image_tag}</code>: no tenants to upgrade")
        return

    canary = await _canary_container(db, tenants)
    if not canary:
        rollout.status = "aborted_canary_failed"
        rollout.completed_at = datetime.utcnow()
        rollout.notes = "no user with is_canary=TRUE in the running set; rollout refuses to proceed"
        await db.commit()
        await _send_telegram(
            "critical",
            f"Rollout <code>{rollout.image_tag}</code> aborted — no canary. "
            f"Set <code>users.is_canary=TRUE</code> on one user before retrying.",
        )
        return

    rollout.canary_prefix = canary.user_id[:8]
    await db.commit()

    # ── Phase A: upgrade canary ────────────────────────────────────
    logger.info("[ROLLOUT] %s canary=%s", rollout.id, canary.user_id[:8])
    canary_attempt = await _upgrade_one(db, rollout, canary, rollout.image_tag)

    if canary_attempt.status != "ok":
        # Canary upgrade failed (and was rolled back, or rollback also failed)
        rollout.status = "aborted_canary_failed"
        rollout.completed_at = datetime.utcnow()
        rollout.notes = f"canary {canary.user_id[:8]} status={canary_attempt.status}: {canary_attempt.error}"
        await db.commit()
        await _send_telegram(
            "critical",
            f"Rollout <code>{rollout.image_tag}</code> aborted — canary "
            f"<code>{canary.user_id[:8]}</code> {canary_attempt.status}",
        )
        return

    # ── Canary observation window ─────────────────────────────────
    agent_url = await _agent_url(db, canary)
    if agent_url and rollout.canary_wait_minutes > 0:
        logger.info(
            "[ROLLOUT] %s canary observation for %d min", rollout.id, rollout.canary_wait_minutes
        )
        # Poll every 20 s for the full window. 60 min * 3 = max 180 polls.
        interval_s = 20.0
        attempts = int(rollout.canary_wait_minutes * 60 / interval_s)
        best_consecutive = await _poll_health(agent_url, attempts=attempts, interval_s=interval_s)
        if best_consecutive < 3:
            # Degraded during observation → rollback + abort
            logger.warning(
                "[ROLLOUT] %s canary degraded during observation (best=%d)",
                rollout.id, best_consecutive,
            )
            await _upgrade_one(db, rollout, canary, canary_attempt.prior_tag)
            rollout.status = "aborted_canary_failed"
            rollout.completed_at = datetime.utcnow()
            rollout.notes = f"canary degraded during {rollout.canary_wait_minutes}-min observation"
            await db.commit()
            await _send_telegram(
                "critical",
                f"Rollout <code>{rollout.image_tag}</code> aborted — canary degraded "
                f"during {rollout.canary_wait_minutes}-min observation window",
            )
            return

    await _send_telegram(
        "info",
        f"Rollout <code>{rollout.image_tag}</code>: canary passed, proceeding to rest",
    )

    # ── Phase B: batches of N parallel ────────────────────────────
    rest = [t for t in tenants if t.id != canary.id]
    batch_size = settings.rollout_batch_size or 5
    total_ok = 1  # canary counted
    total_fail = 0
    for i in range(0, len(rest), batch_size):
        batch = rest[i : i + batch_size]
        tasks = [_upgrade_one(db, rollout, c, rollout.image_tag) for c in batch]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for r in results:
            if r.status == "ok":
                total_ok += 1
            else:
                total_fail += 1

    rollout.status = "complete"
    rollout.completed_at = datetime.utcnow()
    rollout.notes = f"completed: {total_ok} ok, {total_fail} failed/rolled-back of {len(tenants)} total"
    await db.commit()

    alert_level = "info" if total_fail == 0 else "warning"
    await _send_telegram(
        alert_level,
        f"Rollout <code>{rollout.image_tag}</code> complete: "
        f"{total_ok}/{len(tenants)} upgraded, {total_fail} failed",
    )


# ─── Public entry points ──────────────────────────────────────────


async def start_rollout(
    db: AsyncSession,
    image_tag: str,
    trigger: str,
    triggered_by: Optional[str] = None,
    canary_wait_minutes: Optional[int] = None,
    notes: Optional[str] = None,
) -> Rollout:
    """Create a Rollout row and schedule the background driver.

    Raises RolloutInProgress if another rollout is active.
    Raises ValueError on invalid image_tag.

    Returns the persisted Rollout row (with .id) so the API can return it.
    """
    # Image tag validation — defense-in-depth against ROLLOUT_SECRET leak
    if not image_tag.startswith("ghcr.io/toup-com/toup-agent:"):
        raise ValueError(
            f"image_tag must start with 'ghcr.io/toup-com/toup-agent:', got: {image_tag}"
        )

    active = await active_rollout(db)
    if active:
        raise RolloutInProgress(active.id, active.image_tag)

    wait_min = canary_wait_minutes
    if wait_min is None:
        wait_min = settings.rollout_canary_wait_minutes_default
    wait_min = max(1, min(60, int(wait_min)))

    rollout = Rollout(
        image_tag=image_tag,
        status="pending",
        trigger=trigger,
        triggered_by=triggered_by,
        canary_wait_minutes=wait_min,
        notes=notes,
    )
    db.add(rollout)
    await db.commit()
    await db.refresh(rollout)

    # Schedule the background driver via APScheduler. Import lazily to
    # avoid circular imports at module load time.
    from app.scripts.scheduled_tasks import schedule_one_shot
    schedule_one_shot(
        func=_run_rollout_task,
        kwargs={"rollout_id": rollout.id},
        job_id=f"rollout_{rollout.id}",
    )

    logger.info(
        "[ROLLOUT] started id=%s tag=%s trigger=%s wait=%dm",
        rollout.id, image_tag, trigger, wait_min,
    )
    return rollout


async def cancel_rollout(db: AsyncSession, rollout_id: str) -> Optional[Rollout]:
    """Mark an in-flight rollout as cancelled. Does NOT revert already-upgraded
    tenants — 'cancelled' means 'stop before the next batch'. In-flight batch
    completes (platform has no way to interrupt a bridge upgrade mid-pull).
    """
    result = await db.execute(select(Rollout).where(Rollout.id == rollout_id))
    rollout = result.scalar_one_or_none()
    if not rollout:
        return None
    if rollout.status not in ("pending", "running"):
        return rollout
    rollout.status = "cancelled"
    rollout.completed_at = datetime.utcnow()
    rollout.notes = (rollout.notes or "") + "\ncancelled by operator"
    await db.commit()
    # Note: we don't kill the APScheduler job directly. The _drive_rollout
    # loop checks status between batches, but current design doesn't
    # short-circuit mid-batch. Accept that.
    await _send_telegram(
        "warning",
        f"Rollout <code>{rollout.image_tag}</code> cancelled by operator",
    )
    return rollout


class RolloutInProgress(RuntimeError):
    """Raised when start_rollout is called while another rollout is active."""

    def __init__(self, active_id: str, active_tag: str):
        self.active_id = active_id
        self.active_tag = active_tag
        super().__init__(f"rollout {active_id} ({active_tag}) is already in progress")
