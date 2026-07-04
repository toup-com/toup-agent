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
from datetime import datetime, timedelta
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
    upgrade_tenant_image, BridgeUpgradeUnhealthy, BridgeContainerNotFound,
)

logger = logging.getLogger(__name__)


# ─── Alerts ────────────────────────────────────────────────────────


async def _send_telegram(level: str, message: str) -> None:
    """Fire a Telegram alert to the infra bot (not the user-facing one).

    level: 'info' | 'warning' | 'critical' — rendered as emoji prefix.
    Delegates to the canonical alerting module; rollout alerts are
    deliberate one-shot events, so no rate limiting (min_interval_s=0).
    """
    from app.services.alerting import send_infra_alert
    await send_infra_alert("rollout", level, message, min_interval_s=0)


# ─── Queries ───────────────────────────────────────────────────────


async def _running_tenants(db: AsyncSession) -> list[ManagedContainer]:
    """Tenants eligible for this rollout: status='running' AND pin_image_tag IS NULL.

    Pool-bound containers (`toup-agent-pool-NN`) are EXCLUDED. The bridge's
    per-tenant `/upgrade` + `/whois` endpoints resolve `toup-agent-<prefix>`
    by container NAME, so every pool-bound tenant 404s both — and the
    orphan-quarantine in `_upgrade_one` then flips healthy users to
    status='orphan' on every backend rollout (2026-07-01: f4f52f6b,
    a5774ff4, b60f7255 quarantined minutes after b60f7255 signed up).
    Pool members take image updates via the pool refresh/drain cycle
    (`/v1/pool/refresh-image`), not per-tenant blue-green. NULL names are
    kept (legacy rows predating the name column backfill).
    """
    from sqlalchemy import or_
    result = await db.execute(
        select(ManagedContainer).where(
            ManagedContainer.status == "running",
            ManagedContainer.pin_image_tag.is_(None),
            or_(
                ManagedContainer.container_name.is_(None),
                ~ManagedContainer.container_name.like("toup-agent-pool-%"),
            ),
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
    """Legacy elapsed-time poll. Kept for any external caller; new code uses
    `_observe_canary_signal` which short-circuits on real signal instead of
    burning the full window."""
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


# Signal-based canary observation — replaces the elapsed-time wait.
#
# Rationale: the prior algorithm polled /agent/health for the full
# canary_wait_minutes (default 10), then promoted if `best_consecutive >= 3`
# was seen anywhere in the window. After ~30s of healthy boot, the
# remaining ~9.5 minutes contributed zero signal — they were pure
# wall-clock buffer hoping a regression manifested. With ~10 tenants, no
# real traffic hit the canary during that window, so the buffer was
# theatre.
#
# The signal-based approach has two phases:
#   1. Boot gate: require 3 consecutive 200s within the first 30s.
#      Failure here = rollback. ~10-15s typical on healthy code.
#   2. Stability hold: after boot gate passes, watch /agent/health for
#      another `_CANARY_STABILITY_HOLD_S` seconds. Any non-200 or
#      exception triggers abort+rollback. ~60s default — long enough
#      to catch slow crashes, fast enough to not wait pointlessly.
#
# Operator-set canary_wait_minutes is treated as a HARD CAP (not target):
# the observe never runs longer than that. So setting `canary_wait_minutes=30`
# for a high-risk schema migration extends the cap; default of 5 is fine
# for routine deploys.

_CANARY_BOOT_GATE_S = 75.0          # consecutive-OK gate must clear in this long
_CANARY_BOOT_INTERVAL_S = 3.0       # poll cadence during boot gate
_CANARY_REQUIRED_OK = 3             # consecutive 200s to pass boot gate
_CANARY_STABILITY_HOLD_S = 60.0     # additional sustained-healthy window
_CANARY_STABILITY_INTERVAL_S = 10.0 # poll cadence during stability hold

# Why 75s/3s: the post-upgrade /agent/health observation needs to
# survive a Caddy reconfigure race the bridge does after returning
# from /upgrade. Even though the bridge confirms its OWN 3-pass
# health check before returning 200, the platform's separate observer
# polls through Caddy which can briefly route to the draining old
# slot. 4 production rollouts on 2026-05-24 (74d636db, ece2c2e2,
# 540cf51e + the originating PR's rollout) aborted with "1/3
# consecutive 200s within 30s" while the agent itself responded to
# direct /agent/health curls with sub-100ms 200s right before and
# after the window. The fix is to widen the gate enough that any
# brief mid-swap drop doesn't kill the run:
#   - boot gate window 30 → 75s (caddy reconfigure + first 3 polls)
#   - poll cadence 5 → 3s (so 3 consecutive 200s lands in ~9s of
#     steady state, not ~15s)
# Total worst case: 75s of boot-gate polling before fail (was 30s).
# Stability hold (60s) and hard cap unchanged.


async def _observe_canary_signal(
    agent_url: str,
    *,
    cap_seconds: float,
    boot_gate_s: float = _CANARY_BOOT_GATE_S,
    boot_interval_s: float = _CANARY_BOOT_INTERVAL_S,
    required_ok: int = _CANARY_REQUIRED_OK,
    stability_hold_s: float = _CANARY_STABILITY_HOLD_S,
    stability_interval_s: float = _CANARY_STABILITY_INTERVAL_S,
) -> tuple[bool, str]:
    """Signal-based canary observation. Returns (passed, reason).

    `cap_seconds` is a hard timeout — the function never runs longer than
    this regardless of boot/stability progress. It's the caller's
    operator-set safety budget. Boot gate + stability hold defaults sum
    to ~90s; setting cap_seconds < 90 effectively disables stability hold
    (acceptable trade-off for ultra-fast deploys).

    The timing parameters default to module constants for production but
    are injectable so tests can run the full algorithm with tiny windows
    without burning real wall-clock time.
    """
    deadline = time.time() + cap_seconds
    boot_deadline = min(time.time() + boot_gate_s, deadline)
    consecutive_ok = 0

    async with httpx.AsyncClient(timeout=10, verify=True) as client:
        # Phase 1: boot gate — need `required_ok` consecutive 200s.
        logger.info(
            "[CANARY-OBSERVE] boot gate: need %d consecutive 200s within %.0fs",
            required_ok, boot_gate_s,
        )
        while time.time() < boot_deadline and consecutive_ok < required_ok:
            try:
                r = await client.get(f"{agent_url.rstrip('/')}/agent/health")
                if r.status_code == 200:
                    consecutive_ok += 1
                    logger.info("[CANARY-OBSERVE] boot ok %d/%d", consecutive_ok, required_ok)
                else:
                    logger.warning("[CANARY-OBSERVE] boot HTTP %d body=%r", r.status_code, r.text[:100])
                    consecutive_ok = 0
            except httpx.HTTPError as e:
                logger.warning("[CANARY-OBSERVE] boot exception %s: %s", type(e).__name__, str(e)[:200])
                consecutive_ok = 0
            if consecutive_ok < required_ok:
                await asyncio.sleep(boot_interval_s)

        if consecutive_ok < required_ok:
            return (False, f"boot gate failed: only {consecutive_ok}/{required_ok} consecutive 200s within {boot_gate_s:.0f}s")

        # Phase 2: stability hold — sustained healthy window. Any non-200
        # or exception during this phase aborts the canary.
        stability_deadline = min(time.time() + stability_hold_s, deadline)
        stability_window_s = stability_deadline - time.time()
        logger.info(
            "[CANARY-OBSERVE] boot ok; entering %.0fs stability hold",
            stability_window_s,
        )
        while time.time() < stability_deadline:
            try:
                r = await client.get(f"{agent_url.rstrip('/')}/agent/health")
                if r.status_code != 200:
                    return (False, f"stability hold failed: HTTP {r.status_code} after boot")
            except httpx.HTTPError as e:
                return (False, f"stability hold failed: {type(e).__name__}: {str(e)[:200]}")
            await asyncio.sleep(stability_interval_s)

    elapsed = cap_seconds - max(0.0, deadline - time.time())
    return (True, f"healthy (boot + stability passed in {elapsed:.0f}s)")


# ─── Per-tenant upgrade + rollback ────────────────────────────────


async def _upgrade_one(
    _shared_db: Optional[AsyncSession],  # signature compat; ignored
    rollout: Rollout,
    container: ManagedContainer,
    new_tag: str,
) -> RolloutAttempt:
    """Upgrade a single tenant to new_tag; record the attempt.

    Each DB write happens in its OWN narrow session — opened, used for
    one statement-group, committed, closed. NO DB session is held
    across the bridge call. This eliminates the held-session-across-
    network anti-pattern that contributed to the 547-min stall on
    rollout ed53f0d89a11 (2026-05-25): when the bridge stopped
    responding, each hung attempt held a DB connection, and the
    reconciler's own pool was starved on the Supabase pooler so it
    couldn't mark the rollout orphan.

    The session lifecycle now looks like:
      1. Open → write attempt='upgrading' → close.
      2. Bridge call (no session held), wrapped in asyncio.wait_for.
      3. Open → write outcome → close.
      4. Telegram alert (no session).
      5. If rollback needed: bridge call again (no session), then
         open → write rollback outcome → close.

    Returns the RolloutAttempt row with status in
    {'ok', 'failed', 'rolled_back', 'rollback_failed'}.

    Never raises — failures are captured in the attempt's error field.
    """
    prior_tag = container.image_tag or "unknown"
    t0 = time.time()

    # Hard per-attempt timeout — defense in depth above httpx's own
    # timeout. See PR #127 for the 547-min stall context.
    hard_timeout_s = (settings.bridge_upgrade_timeout_s or 180) + 30

    # ── Step 1: narrow session — write attempt='upgrading' ──
    async with async_session_maker() as db:
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
        attempt_id = attempt.id

    # ── Step 2: bridge call WITHOUT holding any DB session ──
    bridge_outcome: str
    bridge_result: Optional[dict] = None
    bridge_error: Optional[str] = None
    orphan_exc: Optional[BridgeContainerNotFound] = None
    unhealthy_exc: Optional[BridgeUpgradeUnhealthy] = None
    try:
        bridge_result = await asyncio.wait_for(
            upgrade_tenant_image(
                None, container.user_id, new_tag, rollout_id=rollout.id,
            ),
            timeout=hard_timeout_s,
        )
        bridge_outcome = "ok"
    except asyncio.TimeoutError:
        bridge_outcome = "timeout"
        bridge_error = (
            f"hard timeout exceeded ({hard_timeout_s}s) — bridge or "
            f"platform unresponsive; tenant state unknown"
        )
    except BridgeContainerNotFound as e:
        bridge_outcome = "container_not_found"
        orphan_exc = e
    except BridgeUpgradeUnhealthy as e:
        bridge_outcome = "unhealthy"
        unhealthy_exc = e
    except Exception as e:
        bridge_outcome = "other_error"
        bridge_error = f"{type(e).__name__}: {str(e)[:1000]}"

    duration_ms = int((time.time() - t0) * 1000)

    # ── Step 3: narrow session — record bridge outcome ──
    quarantined = False
    needs_rollback = False
    async with async_session_maker() as db:
        attempt = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == attempt_id)
        )).scalar_one()
        attempt.duration_ms = duration_ms

        if bridge_outcome == "ok":
            attempt.status = "ok"
            attempt.health_checks_passed = (bridge_result or {}).get("health_checks_passed", 0)
            attempt.duration_ms = (bridge_result or {}).get("duration_ms") or duration_ms
            attempt.completed_at = datetime.utcnow()
            await db.commit()
            return attempt

        if bridge_outcome == "container_not_found":
            attempt.status = "failed"
            attempt.error = "bridge: container not found (orphan tenant)"
            attempt.completed_at = datetime.utcnow()
            # Auto-quarantine when whois corroborates the 404
            # (`confirmed_orphan=True`). See PR #123.
            if orphan_exc is not None and orphan_exc.confirmed_orphan:
                mc = (await db.execute(
                    select(ManagedContainer).where(
                        ManagedContainer.id == container.id
                    )
                )).scalar_one_or_none()
                if mc is not None and mc.status == "running":
                    mc.status = "orphan"
                    mc.error_message = (
                        f"auto-quarantined by rollout {rollout.id} at "
                        f"{datetime.utcnow().isoformat()}Z — bridge 404 on "
                        f"/upgrade AND /whois (prefix={container.user_id[:8]})"
                    )[:500]
                    quarantined = True
            await db.commit()

        elif bridge_outcome == "timeout":
            attempt.status = "failed"
            attempt.error = bridge_error
            attempt.completed_at = datetime.utcnow()
            await db.commit()

        elif bridge_outcome == "unhealthy":
            attempt.status = "failed"
            assert unhealthy_exc is not None
            attempt.error = f"unhealthy after upgrade: {unhealthy_exc.detail}"[:1000]
            attempt.health_checks_passed = unhealthy_exc.detail.get("health_checks_passed", 0)
            await db.commit()
            needs_rollback = True

        else:  # other_error
            attempt.status = "failed"
            attempt.error = bridge_error
            await db.commit()
            needs_rollback = True

    # ── Step 4: alerts for terminal outcomes (no DB session held) ──
    if bridge_outcome == "container_not_found":
        if quarantined:
            await _send_telegram(
                "warning",
                f"Tenant <code>{container.user_id[:8]}</code> auto-quarantined "
                f"— bridge has no container for this prefix (confirmed via "
                f"/whois). managed_containers row flipped "
                f"<code>running</code>→<code>orphan</code>; future rollouts "
                f"will skip until re-provision. No in-flight users were "
                f"affected.",
            )
        else:
            await _send_telegram(
                "warning",
                f"Tenant <code>{container.user_id[:8]}</code> skipped — no "
                f"container on bridge (orphan, unconfirmed by /whois). "
                f"Manual cleanup may be needed but no in-flight users are "
                f"affected.",
            )
        return attempt

    if bridge_outcome == "timeout":
        await _send_telegram(
            "warning",
            f"Tenant <code>{container.user_id[:8]}</code> upgrade "
            f"<b>hard-timed out</b> after {hard_timeout_s}s — bridge "
            f"unresponsive. No rollback attempted (state unknown). "
            f"Rollout continues with remaining tenants.",
        )
        return attempt

    if not needs_rollback:
        return attempt

    # ── Step 5: refuse rollback if prior_tag is invalid ──
    if prior_tag == "unknown" or not prior_tag.startswith("ghcr.io/toup-com/toup-agent:"):
        async with async_session_maker() as db:
            attempt = (await db.execute(
                select(RolloutAttempt).where(RolloutAttempt.id == attempt_id)
            )).scalar_one()
            attempt.error = (attempt.error or "") + (
                f" | rollback skipped: prior_tag={prior_tag!r} not a valid"
                f" GHCR tag — DB record missing or malformed"
            )
            attempt.completed_at = datetime.utcnow()
            await db.commit()
        await _send_telegram(
            "warning",
            f"Tenant <code>{container.user_id[:8]}</code> failed upgrade to "
            f"<code>{new_tag}</code> — rollback skipped (no valid prior tag).",
        )
        return attempt

    # ── Step 6: rollback bridge call WITHOUT holding any DB session ──
    rollback_outcome: str
    rollback_error: Optional[str] = None
    try:
        await asyncio.wait_for(
            upgrade_tenant_image(
                None, container.user_id, prior_tag, rollout_id=rollout.id,
            ),
            timeout=hard_timeout_s,
        )
        rollback_outcome = "rolled_back"
    except asyncio.TimeoutError:
        rollback_outcome = "rollback_failed"
        rollback_error = f"rollback hard-timed out after {hard_timeout_s}s"
    except Exception as e:
        rollback_outcome = "rollback_failed"
        rollback_error = f"{type(e).__name__}: {str(e)[:500]}"

    # ── Step 7: narrow session — record rollback outcome ──
    async with async_session_maker() as db:
        attempt = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == attempt_id)
        )).scalar_one()
        if rollback_outcome == "rolled_back":
            attempt.status = "rolled_back"
        else:
            attempt.status = "rollback_failed"
            attempt.error = (attempt.error or "") + f" | ROLLBACK FAILED: {rollback_error}"
        attempt.completed_at = datetime.utcnow()
        await db.commit()

    if rollback_outcome == "rolled_back":
        await _send_telegram(
            "warning",
            f"Tenant <code>{container.user_id[:8]}</code> failed upgrade to "
            f"<code>{new_tag}</code> — rolled back to <code>{prior_tag}</code>",
        )
    else:
        await _send_telegram(
            "critical",
            f"Tenant <code>{container.user_id[:8]}</code> upgrade AND rollback "
            f"failed — manual intervention needed. Error: {(rollback_error or '')[:200]}",
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
        rollout.last_progress_at = datetime.utcnow()
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
            rollout.last_progress_at = datetime.utcnow()
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
        rollout.last_progress_at = datetime.utcnow()
        rollout.notes = "no running tenants; nothing to do"
        await db.commit()
        await _send_telegram("info", f"Rollout <code>{rollout.image_tag}</code>: no tenants to upgrade")
        return

    canary = await _canary_container(db, tenants)
    if not canary:
        rollout.status = "aborted_canary_failed"
        rollout.completed_at = datetime.utcnow()
        rollout.last_progress_at = datetime.utcnow()
        rollout.notes = "no user with is_canary=TRUE in the running set; rollout refuses to proceed"
        await db.commit()
        await _send_telegram(
            "critical",
            f"Rollout <code>{rollout.image_tag}</code> aborted — no canary. "
            f"Set <code>users.is_canary=TRUE</code> on one user before retrying.",
        )
        return

    rollout.canary_prefix = canary.user_id[:8]
    rollout.phase = "canary_upgrading"
    rollout.last_progress_at = datetime.utcnow()
    await db.commit()

    # ── Phase A: upgrade canary ────────────────────────────────────
    logger.info("[ROLLOUT] %s canary=%s", rollout.id, canary.user_id[:8])
    canary_attempt = await _upgrade_one(db, rollout, canary, rollout.image_tag)

    if canary_attempt.status != "ok":
        # Canary upgrade failed (and was rolled back, or rollback also failed)
        rollout.status = "aborted_canary_failed"
        rollout.completed_at = datetime.utcnow()
        rollout.last_progress_at = datetime.utcnow()
        rollout.phase = ""
        rollout.notes = f"canary {canary.user_id[:8]} status={canary_attempt.status}: {canary_attempt.error}"
        await db.commit()
        await _send_telegram(
            "critical",
            f"Rollout <code>{rollout.image_tag}</code> aborted — canary "
            f"<code>{canary.user_id[:8]}</code> {canary_attempt.status}",
        )
        return

    # ── Canary observation window ─────────────────────────────────
    # Persist the deadline so resume_orphaned_rollouts() can pick this up
    # if the platform redeploys mid-observation. canary_observe_loop()
    # below is idempotent — safe to call from either start_rollout's task
    # or the resumer.
    if rollout.canary_wait_minutes > 0:
        rollout.phase = "canary_observing"
        rollout.resume_after = datetime.utcnow() + timedelta(minutes=rollout.canary_wait_minutes)
        rollout.last_progress_at = datetime.utcnow()
        await db.commit()

    agent_url = await _agent_url(db, canary)
    proceed = await _canary_observe_loop(db, rollout, canary, canary_attempt.prior_tag, agent_url)
    if not proceed:
        return

    rollout.phase = "batching"
    rollout.resume_after = None
    rollout.last_progress_at = datetime.utcnow()
    await db.commit()

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
        # Mid-batch heartbeat: with `bridge_upgrade_timeout_s=180` and the
        # hard_timeout_s=210 cap in `_upgrade_one`, a single batch can
        # legitimately take up to ~3.5 min. Without a mid-batch heartbeat,
        # the reconciler's 3-min idle threshold false-positives every slow
        # batch and aborts healthy rollouts. Stamp every 60s while
        # `asyncio.gather` is in flight; on batch finish, cancel the
        # heartbeat and stamp once more for the natural per-batch beat.
        # Uses its OWN session so a starved orchestrator pool doesn't
        # silence the heartbeat that's meant to detect that very state.
        async def _batch_heartbeat(rollout_id: str) -> None:
            while True:
                await asyncio.sleep(60)
                try:
                    async with async_session_maker() as hb_db:
                        await hb_db.execute(
                            Rollout.__table__.update()
                            .where(Rollout.id == rollout_id)
                            .values(last_progress_at=datetime.utcnow())
                        )
                        await hb_db.commit()
                except Exception as hb_err:
                    logger.warning(
                        "[ROLLOUT] mid-batch heartbeat failed (non-fatal): %s",
                        hb_err,
                    )

        hb_task = asyncio.create_task(_batch_heartbeat(rollout.id))
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except (asyncio.CancelledError, Exception):
                pass
        for r in results:
            if r.status == "ok":
                total_ok += 1
            else:
                total_fail += 1
        # Heartbeat after each batch so a stuck mid-batch sequence is
        # detected by the reconciler within minutes, not at the 30-min
        # total-age fallback.
        rollout.last_progress_at = datetime.utcnow()
        await db.commit()

    rollout.status = "complete"
    rollout.completed_at = datetime.utcnow()
    rollout.last_progress_at = datetime.utcnow()
    rollout.phase = ""
    rollout.resume_after = None
    rollout.notes = f"completed: {total_ok} ok, {total_fail} failed/rolled-back of {len(tenants)} total"
    await db.commit()

    # Tell the bridge so the warm pool flips to the new image. Reconciler
    # drains stale-image GENERIC members on its next tick (≤30s) and
    # respawns on `rollout.image_tag`. Non-fatal: rollout success is
    # decoupled from pool refresh notification.
    try:
        from app.services.pool_service import notify_pool_image_refresh
        await notify_pool_image_refresh(rollout.image_tag)
    except Exception as e:
        logger.warning("[rollout] pool image refresh notify failed: %s", e)

    alert_level = "info" if total_fail == 0 else "warning"
    await _send_telegram(
        alert_level,
        f"Rollout <code>{rollout.image_tag}</code> complete: "
        f"{total_ok}/{len(tenants)} upgraded, {total_fail} failed",
    )


async def _canary_observe_loop(
    db: AsyncSession,
    rollout: Rollout,
    canary: ManagedContainer,
    prior_tag: Optional[str],
    agent_url: Optional[str],
) -> bool:
    """Signal-based canary observation: boot gate (3 consecutive 200s in 30s)
    + stability hold (sustained healthy for 60s). Operator-set
    `canary_wait_minutes` is the HARD CAP. Returns True on pass, False on
    fail (in which case canary has already been rolled back and rollout
    row marked aborted_canary_failed).

    Typical cap-bounded happy path: ~90 seconds. Was previously always
    `canary_wait_minutes` (default 10 min) of pure wall-clock burn after
    the first 3 OKs. The observation now exits as soon as it has the
    signal it needs.

    Idempotent across resumer invocations: reads `resume_after` from the
    rollout row, computes the remaining cap from there. If `resume_after`
    is already past or `canary_wait_minutes=0`, short-circuits success.
    """
    if not agent_url or rollout.canary_wait_minutes <= 0:
        return True

    deadline = rollout.resume_after or datetime.utcnow()
    remaining_s = max(0.0, (deadline - datetime.utcnow()).total_seconds())
    if remaining_s <= 0:
        logger.info(
            "[ROLLOUT] %s canary observation deadline already passed — proceeding to batch",
            rollout.id,
        )
        return True

    logger.info(
        "[ROLLOUT] %s canary signal-based observe (cap=%.0fs)",
        rollout.id, remaining_s,
    )
    passed, reason = await _observe_canary_signal(agent_url, cap_seconds=remaining_s)
    if passed:
        logger.info("[ROLLOUT] %s canary %s", rollout.id, reason)
        return True

    logger.warning("[ROLLOUT] %s canary failed observation: %s", rollout.id, reason)
    await _upgrade_one(db, rollout, canary, prior_tag)
    rollout.status = "aborted_canary_failed"
    rollout.completed_at = datetime.utcnow()
    rollout.last_progress_at = datetime.utcnow()
    rollout.phase = ""
    rollout.resume_after = None
    rollout.notes = f"canary observation failed: {reason}"
    await db.commit()
    await _send_telegram(
        "critical",
        f"Rollout <code>{rollout.image_tag}</code> aborted — canary failed: {reason}",
    )
    return False


# In-flight resume tracking so the reconciler tick doesn't double-fire when
# a previous resume is still running. Process-local; not shared across
# replicas — but we only ever run one platform-api instance, and the
# database `phase` advance is the real source of truth either way.
_resume_inflight: set[str] = set()


# Stuck-rollout threshold. With signal-based canary observation
# (typical ~90s) + ~30s canary upgrade + <1 min batch, worst-case happy
# path is well under 5 min. The 30-min threshold absorbs slow bridge
# calls and operator-extended canary windows (canary_wait_minutes can
# be set up to 60), and won't false-positive any legitimate rollout.
_STUCK_ROLLOUT_THRESHOLD_MIN = 30

# Pending-rollout threshold — much shorter than the running threshold. A
# row sitting in 'pending' means the APScheduler one-shot job never fired
# (process died before pickup, scheduler crashed, jobstore stale). The
# scheduler is supposed to fire within seconds; 5 min is a generous SLA
# that won't false-positive a healthy startup.
_STUCK_PENDING_THRESHOLD_MIN = 5

# Heartbeat threshold — orphan if `last_progress_at` is older than this,
# regardless of total age. The orchestrator stamps `last_progress_at` at
# every state transition; if no stamp lands for 3 min, the orchestrator
# is dead (typically: Railway redeploy killed it). This is the primary
# stuck-rollout detection; the 30-min total-age threshold remains as a
# fallback for the case where heartbeat itself somehow isn't updating
# (e.g. orchestrator hung between state changes during a long single
# operation).
#
# 3 min is sized against the actual operations:
#   - Boot gate: 30s
#   - Stability hold: 60s (heartbeat stamps once at start, once at end)
#   - Per-batch: <30s (heartbeat stamps after each batch)
# So normal operation never has a >2-min gap between heartbeats. 3 min
# adds a buffer for slow bridge calls without false-positiving.
_STUCK_HEARTBEAT_MIN = 3


async def _stamp_progress(db: AsyncSession, rollout: Rollout) -> None:
    """Update the rollout's heartbeat. Call after every state transition.

    Use ALONGSIDE existing field updates: we set last_progress_at on
    `rollout`, and the caller's `await db.commit()` flushes both fields
    in one transaction. We only commit here ourselves when the caller
    has nothing else to commit (use `_stamp_progress_and_commit` for
    that path).
    """
    rollout.last_progress_at = datetime.utcnow()


async def _stamp_progress_and_commit(db: AsyncSession, rollout: Rollout) -> None:
    """Stamp + commit. Use when there are no other field changes to flush
    (i.e. the orchestrator wants to refresh the heartbeat without
    advancing state)."""
    rollout.last_progress_at = datetime.utcnow()
    await db.commit()


async def rollout_reconciler_loop() -> None:
    """Long-running background task — wakes every 30 s, advances rollouts
    that need it, and orphans the ones stuck way past their budget.

    Why this exists: previously a Railway redeploy during the 10-min canary
    observation window killed the in-memory `asyncio.sleep` driving the
    rollout. The startup hook tried to recover via APScheduler, which
    sometimes fails to start (logged `name 'asyncio' is not defined` was the
    most recent symptom). Result: rollouts wedged in `running/canary_observing`
    forever, blocking every subsequent CI push with HTTP 409 and leaving
    tenants on stale images.

    This reconciler is self-contained — no APScheduler, no external clock
    source. It re-derives intent from the DB on every tick, so any process
    death is recovered within 30 s of the next start. Idempotent: if a
    resume is already in-flight in this process, the next tick skips it.

    Two responsibilities, kept in one loop because they share the same
    "list rollouts in 'running'" query:
      1. Resume rollouts in phase='canary_observing' once `resume_after`
         has passed. This is the recovery path for redeploys mid-canary.
      2. Auto-orphan rollouts that have been `running` for longer than
         `_STUCK_ROLLOUT_THRESHOLD_MIN`, regardless of phase. Catches
         double failures (e.g. resume crashes too) and rare phase corruptions.
    """
    logger.info("[ROLLOUT-RECONCILER] started (tick=30s)")
    while True:
        try:
            await _reconcile_once()
        except Exception:
            # Never let a tick exception kill the loop. Log and keep going.
            logger.exception("[ROLLOUT-RECONCILER] tick failed; will retry")
        # NOTE (bulletproof plan M): the legacy prewarm reconciler
        # (`reconcile_stuck_provisioning`) used to piggy-back here. It was
        # removed — `pool_service.reclaim_stranded_users` (180s tick in the
        # container reconciler) now owns stuck-provisioning recovery, and
        # running two reconciliation systems against the same rows meant one
        # could re-fire a cold provision while the other was mid pool-claim.
        # One reconciliation system only.
        await asyncio.sleep(30)


async def _reconcile_once(db: Optional[AsyncSession] = None) -> None:
    """One reconciler tick — split out so tests can call it deterministically.

    Accepts an optional pre-existing session so callers like `start_rollout`
    can run the reconcile pass inside their own transaction (lock self-heal
    before the active-rollout check). When called with no argument the loop
    creates its own session.
    """
    if db is None:
        async with async_session_maker() as own_db:
            await _reconcile_once_in_session(own_db)
    else:
        await _reconcile_once_in_session(db)


async def _reconcile_once_in_session(db: AsyncSession) -> None:
    """The body of one reconciler tick. Operates on the provided session.

    Watches BOTH `pending` and `running` statuses:
      - `pending` past `_STUCK_PENDING_THRESHOLD_MIN`: APScheduler never
        fired. Orphan and free the lock.
      - `running` past `_STUCK_ROLLOUT_THRESHOLD_MIN`: orchestrator died
        mid-flight. Orphan, free the lock.
      - `running` AND `phase='canary_observing'` AND `resume_after` passed:
        resume the canary observation in this process.
    """
    now = datetime.utcnow()
    result = await db.execute(
        select(Rollout).where(Rollout.status.in_(["pending", "running"]))
    )
    inflight = result.scalars().all()
    if not inflight:
        return

    for rollout in inflight:
        age_min = (now - rollout.started_at).total_seconds() / 60 if rollout.started_at else 0
        # Heartbeat — fall back to started_at for rows created before the
        # last_progress_at column existed (or rolled over from a backfill).
        last_progress = rollout.last_progress_at or rollout.started_at
        idle_min = (now - last_progress).total_seconds() / 60 if last_progress else 0

        # (0) Heartbeat-stale orphan — orchestrator hasn't stamped progress
        # in `_STUCK_HEARTBEAT_MIN` minutes. Catches the rapid-redeploy case
        # where each new platform-api boot kills the previous orchestrator
        # before it can complete. Only applies once a `running` rollout has
        # had a chance to make at least one heartbeat — pending rollouts
        # are handled by their own threshold below.
        if rollout.status == "running" and idle_min > _STUCK_HEARTBEAT_MIN:
            logger.warning(
                "[ROLLOUT-RECONCILER] auto-orphaning %s (idle=%.1fmin, age=%.1fmin, phase=%r)",
                rollout.id, idle_min, age_min, rollout.phase,
            )
            rollout.status = "aborted_orphan"
            rollout.completed_at = now
            rollout.notes = (rollout.notes or "") + (
                f"\nAuto-orphaned by reconciler — heartbeat stale "
                f"({idle_min:.1f}min idle, age={age_min:.1f}min, "
                f"phase={rollout.phase!r}, threshold={_STUCK_HEARTBEAT_MIN}min)"
            )
            await db.commit()
            await _send_telegram(
                "warning",
                f"Rollout <code>{rollout.image_tag}</code> auto-orphaned "
                f"(no progress in {idle_min:.0f}min, phase={rollout.phase!r}). "
                f"Re-trigger if still needed.",
            )
            continue

        # (1) `pending` orphan path — the row was created but the
        # APScheduler one-shot never fired. Short threshold (5 min): the
        # scheduler is supposed to pick this up within seconds.
        if rollout.status == "pending" and age_min > _STUCK_PENDING_THRESHOLD_MIN:
            logger.warning(
                "[ROLLOUT-RECONCILER] auto-orphaning pending %s (age=%.1fmin)",
                rollout.id, age_min,
            )
            rollout.status = "aborted_orphan"
            rollout.completed_at = now
            rollout.notes = (rollout.notes or "") + (
                f"\nAuto-orphaned by reconciler at age={age_min:.1f}min "
                f"(status=pending — APScheduler never fired; "
                f"threshold={_STUCK_PENDING_THRESHOLD_MIN}min)"
            )
            await db.commit()
            await _send_telegram(
                "warning",
                f"Rollout <code>{rollout.image_tag}</code> auto-orphaned in "
                f"<b>pending</b> after {age_min:.0f} min (scheduler never fired). "
                f"Re-trigger if still needed.",
            )
            continue

        # (2) Auto-orphan rollouts past the budget — runs FIRST so a
        # truly-stuck rollout doesn't keep getting "resumed" forever.
        if rollout.status == "running" and age_min > _STUCK_ROLLOUT_THRESHOLD_MIN:
            logger.warning(
                "[ROLLOUT-RECONCILER] auto-orphaning %s (age=%.1fmin, phase=%r)",
                rollout.id, age_min, rollout.phase,
            )
            rollout.status = "aborted_orphan"
            rollout.completed_at = now
            rollout.notes = (rollout.notes or "") + (
                f"\nAuto-orphaned by reconciler at age={age_min:.1f}min "
                f"(phase={rollout.phase!r}, threshold={_STUCK_ROLLOUT_THRESHOLD_MIN}min)"
            )
            await db.commit()
            await _send_telegram(
                "warning",
                f"Rollout <code>{rollout.image_tag}</code> auto-orphaned "
                f"after {age_min:.0f} min (phase={rollout.phase!r}). "
                f"Re-trigger if still needed.",
            )
            continue

        # Pending rows under the threshold are healthy in-flight work; skip
        # them entirely (no resume logic applies). Only `running` rows
        # continue past this point.
        if rollout.status != "running":
            continue

        # (1) Resume canary_observing rollouts whose deadline passed.
        # Other phases (canary_upgrading, batching) make bridge calls
        # whose results we lost — replaying would risk double-deploys,
        # so we leave them for the orphan threshold above.
        if rollout.phase != "canary_observing":
            continue
        if not rollout.resume_after or rollout.resume_after > now:
            continue
        if rollout.id in _resume_inflight:
            logger.debug("[ROLLOUT-RECONCILER] %s already resuming, skip", rollout.id)
            continue

        logger.info(
            "[ROLLOUT-RECONCILER] resuming %s (deadline %s passed at age %.1fmin)",
            rollout.id, rollout.resume_after, age_min,
        )
        _resume_inflight.add(rollout.id)
        asyncio.create_task(_resume_with_cleanup(rollout.id))


async def _resume_with_cleanup(rollout_id: str) -> None:
    """Wrap _resume_rollout_task to always release the in-flight slot."""
    try:
        await _resume_rollout_task(rollout_id)
    except Exception:
        logger.exception("[ROLLOUT-RECONCILER] resume of %s crashed", rollout_id)
    finally:
        _resume_inflight.discard(rollout_id)


async def resume_orphaned_rollouts() -> None:
    """Startup hook (kept for fast recovery on redeploy).

    Triggers one reconcile cycle immediately so a redeploy doesn't have to
    wait the full 30 s reconciler tick to pick up an orphan. The reconciler
    loop is the durable mechanism — this is just the warm-start.

    Called from app startup (FastAPI lifespan).
    """
    try:
        await _reconcile_once()
    except Exception:
        # Must not block app boot. The reconciler loop will retry shortly.
        logger.exception("[ROLLOUT-RESUMER] startup reconcile failed")


async def _resume_rollout_task(rollout_id: str) -> None:
    """Resumer entry point — re-enters _drive_rollout from the canary
    observation phase. Same lifecycle as _run_rollout_task but skips the
    initial "pending → running" transition since we're already running.
    """
    async with async_session_maker() as db:
        result = await db.execute(select(Rollout).where(Rollout.id == rollout_id))
        rollout = result.scalar_one_or_none()
        if not rollout or rollout.status != "running":
            logger.warning(
                "[ROLLOUT-RESUMER] %s no longer eligible (status=%s)",
                rollout_id, rollout.status if rollout else "missing",
            )
            return
        if rollout.phase != "canary_observing":
            logger.warning(
                "[ROLLOUT-RESUMER] %s phase changed to %r before resume",
                rollout_id, rollout.phase,
            )
            return

        # Find the canary container from the prefix we stored at start
        canary_result = await db.execute(
            select(ManagedContainer).where(
                ManagedContainer.user_id.like(f"{rollout.canary_prefix}%")
            )
        )
        canary = canary_result.scalar_one_or_none()
        if not canary:
            rollout.status = "aborted_orphan"
            rollout.completed_at = datetime.utcnow()
            rollout.last_progress_at = datetime.utcnow()
            rollout.notes = (rollout.notes or "") + "\nResumer: canary container not found"
            await db.commit()
            return

        # Look up the canary's prior tag from its successful attempt row
        prior = None
        for a in (rollout.attempts or []):
            if a.status == "ok":
                prior = a.prior_tag
                break

        agent_url = await _agent_url(db, canary)
        proceed = await _canary_observe_loop(db, rollout, canary, prior, agent_url)
        if not proceed:
            return

        rollout.phase = "batching"
        rollout.resume_after = None
        rollout.last_progress_at = datetime.utcnow()
        await db.commit()
        await _send_telegram(
            "info",
            f"Rollout <code>{rollout.image_tag}</code>: canary passed (resumed), "
            f"proceeding to rest",
        )

        # Continue with the batch phase. Need to re-enumerate tenants.
        tenants = await _running_tenants(db)
        rest = [t for t in tenants if t.id != canary.id]
        batch_size = settings.rollout_batch_size or 5
        total_ok = 1
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
            # Heartbeat after each batch — same rationale as _drive_rollout.
            rollout.last_progress_at = datetime.utcnow()
            await db.commit()

        rollout.status = "complete"
        rollout.completed_at = datetime.utcnow()
        rollout.last_progress_at = datetime.utcnow()
        rollout.phase = ""
        rollout.resume_after = None
        rollout.notes = (
            f"completed (resumed from orphan): {total_ok} ok, "
            f"{total_fail} failed/rolled-back of {len(tenants)} total"
        )
        await db.commit()
        try:
            from app.services.pool_service import notify_pool_image_refresh
            await notify_pool_image_refresh(rollout.image_tag)
        except Exception as e:
            logger.warning("[rollout-resume] pool image refresh notify failed: %s", e)
        await _send_telegram(
            "info" if total_fail == 0 else "warning",
            f"Rollout <code>{rollout.image_tag}</code> complete (resumed): "
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

    # Self-heal: run a reconcile pass BEFORE checking the lock. If a prior
    # rollout died and is sitting past its threshold (running >30 min, or
    # pending >5 min), this orphans it and frees the lock so the new one
    # can proceed. Without this, every CI push hit 409 indefinitely after
    # the very first stuck rollout — operators had to manually cancel.
    await _reconcile_once(db)

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
        last_progress_at=datetime.utcnow(),
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


async def force_orphan_active(db: AsyncSession, reason: str = "operator force-orphan") -> Optional[Rollout]:
    """Operator escape hatch — force any current pending/running rollout to
    `aborted_orphan`, freeing the lock immediately.

    Returns the orphaned Rollout (or None if no lock was held). Useful when
    the reconciler hasn't yet aged a stuck rollout past the threshold but
    the operator knows the orchestrator is dead and wants to push a new
    deploy now.

    Idempotent: if no rollout is active this is a no-op returning None.
    """
    active = await active_rollout(db)
    if not active:
        return None
    age_min = (datetime.utcnow() - active.started_at).total_seconds() / 60 if active.started_at else 0
    active.status = "aborted_orphan"
    active.completed_at = datetime.utcnow()
    active.last_progress_at = datetime.utcnow()
    active.notes = (active.notes or "") + f"\nForce-orphaned by operator: {reason} (age={age_min:.1f}min)"
    await db.commit()
    await _send_telegram(
        "warning",
        f"Rollout <code>{active.image_tag}</code> force-orphaned by operator "
        f"(age {age_min:.0f}min, was status={active.status!r}, phase={active.phase!r}).",
    )
    logger.warning(
        "[ROLLOUT] force-orphan id=%s tag=%s age=%.1fmin reason=%r",
        active.id, active.image_tag, age_min, reason,
    )
    return active


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
    rollout.last_progress_at = datetime.utcnow()
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
