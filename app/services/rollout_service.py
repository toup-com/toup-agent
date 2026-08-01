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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import httpx
from sqlalchemy import func, select, update
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
            # Clear any stale error: a reconciler false-positive (heartbeat-
            # stale orphan on a still-alive driver, observed 2026-07-25) may
            # have stamped this row 'orphaned' + "driver died..." while we
            # were awaiting the bridge. Our write wins on status — it must
            # win on error too, or the permanent record reads
            # status='ok' error='driver died...'.
            attempt.error = None
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
            #
            # Status MUST NOT be 'complete': a crash here can predate the
            # canary observation (e.g. a transient DB error in the first
            # statements of _drive_rollout), and 'complete' is load-bearing
            # downstream — the convergence sweep picks the newest complete/
            # complete_with_failures rollout as the fleet's target tag, and
            # _latest_known_good_image_tag pins new provisions to it. A
            # crash-stamped 'complete' would have the sweep batch-drive the
            # whole fleet onto a tag whose canary never passed.
            # 'aborted_orphan' is equally terminal (frees the lock) and the
            # CI gate already handles it by re-firing once.
            logger.exception("[ROLLOUT] id=%s crashed: %s", rollout_id, e)
            rollout.status = "aborted_orphan"
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

    # ── Phase A-0: let the bridge finish recycling the pool ────────
    # The PREVIOUS rollout's completion told the bridge to refresh the pool
    # image, and the bridge recycles every member from there — 49 of 50 inside
    # 30 minutes, measured 2026-08-01. Starting the canary upgrade inside that
    # window is what killed five consecutive rollouts (ConnectError / stale
    # heartbeat -> aborted_orphan / 0 health checks in 259s), on diffs that
    # could not fail a boot. See wait_for_pool_quiescence for the timeline.
    #
    # Heartbeat via `_heartbeating`, NOT an inline beat on `db`. The first cut
    # of this used a callback that wrote `last_progress_at` through the caller's
    # own session; it beat for ~5.5 min and then went silent, and the rollout
    # was orphaned at 3.1 min idle with no redeploy anywhere near it
    # (2026-08-01 19:22:35 -> 19:31:47). That is precisely what `_heartbeating`
    # documents itself as existing to prevent — it uses its OWN session per
    # beat, so a starved orchestrator pool cannot silence the heartbeat meant to
    # detect that state, and its docstring names the inline copy as the bug.
    #
    # On timeout we PROCEED and record it — the canary gate still catches a
    # genuinely bad image.
    rollout.phase = "pool_quiesce"
    rollout.last_progress_at = datetime.utcnow()
    await db.commit()

    from app.services.pool_service import wait_for_pool_quiescence
    async with _heartbeating(rollout.id, "pool-quiesce"):
        _quiet, _why = await wait_for_pool_quiescence(
            settings.rollout_pool_quiesce_timeout_s,
        )
    logger.info("[ROLLOUT] %s pool quiescence: quiet=%s (%s)", rollout.id, _quiet, _why)
    if not _quiet:
        rollout.notes = ((rollout.notes or "") + f"\npool NOT quiescent at canary start: {_why}").strip()

    rollout.canary_prefix = canary.user_id[:8]
    rollout.phase = "canary_upgrading"
    rollout.last_progress_at = datetime.utcnow()
    await db.commit()

    # ── Phase A: upgrade canary ────────────────────────────────────
    # Heartbeat REQUIRED: this single call is allowed 210s (hard_timeout_s) but
    # the reconciler orphans at 180s, so a healthy-but-slow canary upgrade used
    # to abort itself with the driver still running.
    logger.info("[ROLLOUT] %s canary=%s", rollout.id, canary.user_id[:8])
    async with _heartbeating(rollout.id, "canary-upgrade"):
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
        # Shared helper (see `_heartbeating`) rather than an inline copy — the
        # duplicate was how `_resume_rollout_task`'s identical loop ended up
        # with no heartbeat at all.
        async with _heartbeating(rollout.id, "mid-batch"):
            results = await asyncio.gather(*tasks, return_exceptions=False)
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

    # Honest terminal status (2026-07-28 incident: re-drive d79584ea hit the
    # bridge mid-restart on one tenant, reported 'complete', and left a real
    # user silently on the old image for ~40 min). A rollout that failed any
    # tenant is NOT 'complete' — 'complete_with_failures' is equally terminal
    # (active_rollout only counts pending/running) but visible to operators
    # and to the convergence sweep.
    rollout.status = "complete" if total_fail == 0 else "complete_with_failures"
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


@asynccontextmanager
async def _heartbeating(rollout_id: str, what: str):
    """Keep `last_progress_at` fresh across a long uninstrumented await.

    The reconciler orphans a running rollout after `_STUCK_HEARTBEAT_MIN`
    minutes (180s) without a beat, but a SINGLE bridge call is deliberately
    allowed to take `hard_timeout_s` = bridge_upgrade_timeout_s + 30 = 210s.
    Any stretch that awaits the bridge without beating is therefore a standing
    race that aborts healthy rollouts — it does not need a redeploy to fire.
    Observed 2026-07-25: three aborts in 20 minutes, two of them
    `aborted_orphan` with the driver still alive and working.

    Uses its OWN session per beat, so a starved orchestrator pool cannot
    silence the very heartbeat meant to detect that state. Non-fatal by
    contract: a failed beat must never fail a rollout.

    This exists as ONE helper because the copy-paste version was the bug —
    `_drive_rollout`'s batch loop had an inline heartbeat and
    `_resume_rollout_task`'s otherwise-identical loop was missing it entirely.
    """
    async def _beat() -> None:
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
                    "[ROLLOUT] %s heartbeat failed (non-fatal): %s", what, hb_err,
                )

    task = asyncio.create_task(_beat())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


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
        # Convergence sweep — separate try/except so a sweep failure can't
        # mask (or be masked by) a reconcile failure. Lives in the LOOP, not
        # in _reconcile_once: start_rollout runs _reconcile_once as its lock
        # self-heal pass, and a sweep firing there would grab the lock the
        # caller is about to check and 409 every CI push with divergence.
        # Note the tick pauses while a sweep drives its batches (bounded by
        # hard_timeout_s per batch) — acceptable: the sweep holds the rollout
        # lock anyway, and its heartbeat keeps the row visibly alive.
        try:
            await _convergence_sweep_once()
        except Exception:
            logger.exception("[ROLLOUT-SWEEP] tick failed; will retry")
        # NOTE (bulletproof plan M): the legacy prewarm reconciler
        # (`reconcile_stuck_provisioning`) used to piggy-back here. It was
        # removed — `pool_service.reclaim_stranded_users` (180s tick in the
        # container reconciler) now owns stuck-provisioning recovery, and
        # running two reconciliation systems against the same rows meant one
        # could re-fire a cold provision while the other was mid pool-claim.
        # One reconciliation system only.
        await asyncio.sleep(30)


# ─── Convergence sweep (2026-07-28 incident) ──────────────────────
#
# Rollout 01a945e2's driver was killed by its own merge's Railway
# redeploy; the reconciler orphaned it, and the re-drive d79584ea then
# hit the bridge mid-restart (502 — the same merge touched
# bridge/pool_addon.py, which restarts the bridge) on tenant 2739b5c6.
# The re-drive reported 'complete' while a real beta user sat on the old
# image until an operator noticed ~40 min later. The sweep closes that
# class: after every rollout settles, compare what the fleet SHOULD be
# on against what managed_containers says it IS on, and re-drive only
# the divergent tenants. The sweep IS the retry for transient per-tenant
# failures (bridge restarts, 502s) — _upgrade_one itself never retries.

# Max sweep rollouts per (image_tag, 24h). A tenant that still diverges
# after 3 sweeps has a real problem that retries won't fix — alert and
# stop instead of hammering the bridge every 30s tick.
_SWEEP_MAX_PER_TAG_24H = 3

# Minimum spacing between consecutive sweeps of the same tag. The sweep
# exists to retry transient bridge outages (deploy-bridge.yml restarts
# span minutes), but the reconciler ticks every 30s and a failed sweep
# completes in seconds — without spacing, all 3 budget slots burned in
# ~90s against a bridge that was still restarting, and the tag was then
# capped for 24h (the exact incident outcome, minus automation). Three
# attempts spread ≥10 min apart outlast any bridge restart window.
_SWEEP_MIN_GAP_MIN = 10

# Tags whose exhausted-cap alert has already fired this process — the
# sweep skips capped tags on every subsequent tick, and re-alerting
# every 30s would flood Telegram. Process-local, same trade-off as
# _resume_inflight; a new tag has a different key so it alerts again.
_sweep_exhausted_alerted: set[str] = set()

# Serializes the check-then-insert window of start_rollout and the sweep.
# Both do `active_rollout()` → (several SELECTs) → INSERT with awaits in
# between; without exclusion, a CI webhook landing inside the sweep's
# window (build-agent.yml retries every 60s for 40 min, clustering in
# exactly the divergence-heavy periods the sweep wakes for) passes its
# own active-check and commits a second concurrent rollout — two drivers
# then blue-green the same tenants to different tags. Process-local lock
# is sufficient: only one platform-api instance runs (same assumption as
# _resume_inflight), and it is held only across quick DB statements —
# never across bridge/Telegram I/O.
_rollout_creation_lock = asyncio.Lock()


async def _convergence_sweep_once() -> None:
    """One convergence-sweep pass. Runs only when no rollout is active.

    Target = image_tag of the most recent rollout whose status is
    'complete' or 'complete_with_failures' (by completed_at). Any tenant
    eligible for rollouts (same `_running_tenants` filter — running, not
    pinned, not pool-bound) whose managed_containers.image_tag differs is
    driven through the existing `_upgrade_one` path under a fresh
    Rollout(trigger='sweep'), so attempt rows, quarantine handling and
    Telegram alerts all work unchanged.

    Guard rails:
      - Kill switch: settings.rollout_convergence_sweep (default ON).
      - Newest-row guard: if the newest rollout row overall (any status,
        by started_at) is NOT the target and carries a DIFFERENT tag, the
        sweep stands down. Replaying the 2026-07-28 incident without this:
        the reconciler orphans the NEW-tag rollout and the sweep fires in
        the same loop iteration, sees target = the PREVIOUS tag, reads the
        just-upgraded tenants as 'divergent', and blue-green DOWNGRADES
        them while holding the lock the CI gate's re-fire needs (its 409
        was silently swallowed). Same guard keeps cancel_rollout's
        documented "does NOT revert already-upgraded tenants" true — the
        newest row being 'cancelled' also stands the sweep down. The
        newer tag's re-drive (CI gate re-fire, or the operator) owns the
        fleet's intent; the sweep only converges toward settled history.
      - Spacing: ≥ `_SWEEP_MIN_GAP_MIN` minutes between sweeps of the
        same tag, so transient bridge outages get outlasted instead of
        burning the whole budget in ~90s of consecutive ticks.
      - Budget: at most `_SWEEP_MAX_PER_TAG_24H` sweep rollouts per tag
        per 24h — after that, alert once and skip until a new tag ships.

    Session discipline (pgbouncer txn-mode pooler): all DB checks + the
    sweep-row insert happen under `_rollout_creation_lock` in one short
    session; Telegram calls and the batch drive run with NO session open
    and NO transaction pending — heartbeats/stamps use narrow sessions,
    same contract as `_upgrade_one`.
    """
    if not settings.rollout_convergence_sweep:
        return

    sweep_id: Optional[str] = None
    sweep_rollout: Optional[Rollout] = None
    exhausted_prefixes: Optional[str] = None
    target = ""
    divergent: list[ManagedContainer] = []
    prefixes = ""

    async with async_session_maker() as db:
        async with _rollout_creation_lock:
            # Never sweep under an active rollout — the driver owns the fleet.
            if await active_rollout(db) is not None:
                return

            target_rollout = (await db.execute(
                select(Rollout)
                .where(Rollout.status.in_(["complete", "complete_with_failures"]))
                .order_by(Rollout.completed_at.desc())
            )).scalars().first()
            if target_rollout is None:
                return  # fresh install; no rollout history to converge on
            target = target_rollout.image_tag

            # Newest-row guard (see docstring): a newer rollout row on a
            # DIFFERENT tag that never completed (aborted_orphan, cancelled,
            # aborted_canary_failed) means the fleet's intent is that newer
            # tag — sweeping now would converge BACKWARD onto old history.
            newest = (await db.execute(
                select(Rollout).order_by(Rollout.started_at.desc()).limit(1)
            )).scalars().first()
            if (
                newest is not None
                and newest.id != target_rollout.id
                and newest.image_tag != target
            ):
                logger.info(
                    "[ROLLOUT-SWEEP] standing down: newest rollout %s "
                    "(status=%s, tag=%s) is not the convergence target %s "
                    "(tag=%s) — the newer tag's re-drive owns the fleet",
                    newest.id, newest.status, newest.image_tag,
                    target_rollout.id, target,
                )
                return

            tenants = await _running_tenants(db)
            divergent = [t for t in tenants if (t.image_tag or "") != target]
            if not divergent:
                return
            prefixes = ", ".join(sorted(t.user_id[:8] for t in divergent))

            # Budget: count prior sweeps for this tag in the last 24h BEFORE
            # starting another. Uses started_at (creation time) — completed_at
            # is NULL if a sweep itself died.
            cutoff = datetime.utcnow() - timedelta(hours=24)
            prior_sweeps = (await db.execute(
                select(func.count()).select_from(Rollout).where(
                    Rollout.trigger == "sweep",
                    Rollout.image_tag == target,
                    Rollout.started_at >= cutoff,
                )
            )).scalar() or 0
            if prior_sweeps >= _SWEEP_MAX_PER_TAG_24H:
                if target not in _sweep_exhausted_alerted:
                    _sweep_exhausted_alerted.add(target)
                    exhausted_prefixes = prefixes
                # Close the read transaction before falling out to the
                # (external HTTP) alert below — never hold a pooled
                # connection idle-in-transaction across network I/O.
                await db.rollback()
            else:
                # Spacing: skip this tick if the last sweep for this tag is
                # too recent — a failed sweep completes in seconds while the
                # bridge outage it hit spans minutes.
                last_sweep_started = (await db.execute(
                    select(Rollout.started_at).where(
                        Rollout.trigger == "sweep",
                        Rollout.image_tag == target,
                        Rollout.started_at >= cutoff,
                    ).order_by(Rollout.started_at.desc()).limit(1)
                )).scalar_one_or_none()
                if (
                    last_sweep_started is not None
                    and datetime.utcnow() - last_sweep_started
                    < timedelta(minutes=_SWEEP_MIN_GAP_MIN)
                ):
                    logger.info(
                        "[ROLLOUT-SWEEP] standing down: last sweep for %s "
                        "started %s — waiting out the %d-min gap",
                        target, last_sweep_started, _SWEEP_MIN_GAP_MIN,
                    )
                    return

                # Created directly in 'running' — the sweep drives inline,
                # there is no scheduler hop that a 'pending' state would
                # represent. No canary phase: the target tag already passed
                # canary in the rollout that made it the target (the crash
                # catch-all no longer stamps 'complete'), and _upgrade_one's
                # per-tenant health gate + rollback still applies.
                rollout = Rollout(
                    image_tag=target,
                    status="running",
                    trigger="sweep",
                    canary_wait_minutes=0,
                    phase="batching",
                    notes=f"convergence sweep: {len(divergent)} divergent tenant(s): {prefixes}",
                    last_progress_at=datetime.utcnow(),
                )
                db.add(rollout)
                await db.commit()
                # NOTE: no db.refresh() here — the id is client-generated
                # (uuid4 default) and the sessionmaker runs
                # expire_on_commit=False, so nothing needs reloading. The
                # refresh this replaced opened a NEW transaction (SELECT
                # autobegin) that then sat idle-in-transaction across the
                # Telegram call and the entire first batch of bridge
                # upgrades (~210s) — the exact pgbouncer anti-pattern
                # purged in commit 6d173563.
                sweep_id = rollout.id
                sweep_rollout = rollout

    # ── Session closed, lock released; everything below is I/O-heavy and
    # runs with no DB transaction pending. ──

    if exhausted_prefixes is not None:
        await _send_telegram(
            "critical",
            f"convergence sweep exhausted for <code>{target}</code>: "
            f"tenant(s) <code>{exhausted_prefixes}</code> still divergent",
        )
        return
    if sweep_id is None:
        return

    logger.warning(
        "[ROLLOUT-SWEEP] %d divergent tenant(s) [%s] vs target %s — starting sweep %s",
        len(divergent), prefixes, target, sweep_id,
    )
    await _send_telegram(
        "warning",
        f"Convergence sweep started — {len(divergent)} tenant(s) "
        f"<code>{prefixes}</code> diverged from <code>{target}</code>, re-driving",
    )

    # Same batch loop + heartbeat contract as _drive_rollout Phase B. The
    # `divergent` containers and `sweep_rollout` are detached rows (their
    # session is closed); `_upgrade_one` only reads loaded scalar attrs
    # and opens its own narrow sessions. Progress stamps use narrow
    # sessions too.
    batch_size = settings.rollout_batch_size or 5
    total_ok = 0
    total_fail = 0
    for i in range(0, len(divergent), batch_size):
        batch = divergent[i : i + batch_size]
        tasks = [_upgrade_one(None, sweep_rollout, c, target) for c in batch]
        async with _heartbeating(sweep_id, "sweep-batch"):
            results = await asyncio.gather(*tasks, return_exceptions=False)
        for r in results:
            if r.status == "ok":
                total_ok += 1
            else:
                total_fail += 1
        async with async_session_maker() as db:
            await db.execute(
                update(Rollout).where(Rollout.id == sweep_id)
                .values(last_progress_at=datetime.utcnow())
            )
            await db.commit()

    async with async_session_maker() as db:
        await db.execute(
            update(Rollout).where(Rollout.id == sweep_id).values(
                status="complete" if total_fail == 0 else "complete_with_failures",
                completed_at=datetime.utcnow(),
                last_progress_at=datetime.utcnow(),
                phase="",
                notes=(
                    f"convergence sweep completed: {total_ok} ok, {total_fail} "
                    f"failed/rolled-back of {len(divergent)} divergent"
                ),
            )
        )
        await db.commit()
    await _send_telegram(
        "info" if total_fail == 0 else "warning",
        f"Convergence sweep <code>{target}</code> complete: "
        f"{total_ok}/{len(divergent)} converged, {total_fail} failed",
    )


# Error stamped on attempt rows whose driver died mid-'upgrading'. The
# bridge finishes a POSTed blue-green swap independently of the platform
# driver, so the tenant may well be on the new tag (3 of the 4 stuck rows
# from rollout 01a945e2 were) — the convergence sweep settles it either way.
_ORPHANED_ATTEMPT_ERROR = (
    "driver died; terminal state unknown (bridge may have completed the swap)"
)


async def _orphan_stuck_attempts(db: AsyncSession, rollout_id: str) -> int:
    """Mark a dying rollout's in-flight attempts terminal.

    Called when the reconciler orphans a rollout: any attempt row still in
    status='upgrading' belongs to a driver presumed dead — left alone it
    reads as in-flight forever (rollout 01a945e2 left FOUR such rows).

    Single guarded UPDATE (`WHERE status='upgrading'`) executed in the
    caller's transaction — the caller's commit flushes it alongside the
    rollout row. The guard makes the interleaving with a still-alive
    driver (heartbeat-stale false positive, observed 2026-07-25) safe in
    both orders: if the driver's narrow-session write already landed a
    real terminal outcome, this matches zero rows; if this lands first,
    the driver's own unconditional write overwrites it with the truth
    (and the 'ok' branch clears the stale error). The previous ORM
    read-modify-write could clobber a driver outcome committed between
    its SELECT and its flush.
    """
    result = await db.execute(
        update(RolloutAttempt)
        .where(
            RolloutAttempt.rollout_id == rollout_id,
            RolloutAttempt.status == "upgrading",
        )
        .values(
            status="orphaned",
            error=_ORPHANED_ATTEMPT_ERROR,
            completed_at=datetime.utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount or 0


async def _close_stuck_attempts_of_terminal_rollouts(db: AsyncSession) -> None:
    """Heal 'upgrading' attempt rows whose parent rollout is already terminal.

    Backfill companion to `_orphan_stuck_attempts`: rows orphaned BEFORE
    this cleanup shipped (e.g. rollout 01a945e2's four), or whose rollout
    was closed by a path that didn't run the in-line cleanup. Commits only
    when something changed.

    Age gate: only attempts older than the driver's hard per-attempt
    timeout are touched. A terminal PARENT does not imply a dead DRIVER —
    `cancel_rollout` deliberately lets the in-flight batch finish (up to
    ~210s of live 'upgrading' rows under a status='cancelled' parent),
    and a heartbeat-stale orphan can false-positive on a live driver.
    Past hard_timeout_s(+30s margin) every driver has resolved its
    attempt by construction, so anything still 'upgrading' is debris.
    Same guarded-UPDATE shape as `_orphan_stuck_attempts` so a late
    driver write is never clobbered.
    """
    hard_timeout_s = (settings.bridge_upgrade_timeout_s or 180) + 30
    age_cutoff = datetime.utcnow() - timedelta(seconds=hard_timeout_s + 30)
    result = await db.execute(
        update(RolloutAttempt)
        .where(
            RolloutAttempt.status == "upgrading",
            RolloutAttempt.started_at < age_cutoff,
            RolloutAttempt.rollout_id.in_(
                select(Rollout.id).where(
                    Rollout.status.not_in(["pending", "running"])
                )
            ),
        )
        .values(
            status="orphaned",
            error=_ORPHANED_ATTEMPT_ERROR,
            completed_at=datetime.utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    n = result.rowcount or 0
    if not n:
        return
    await db.commit()
    logger.warning(
        "[ROLLOUT-RECONCILER] closed %d stuck 'upgrading' attempt(s) of terminal rollouts",
        n,
    )


async def _reconcile_once(db: Optional[AsyncSession] = None) -> None:
    """One reconciler tick — split out so tests can call it deterministically.

    Accepts an optional pre-existing session so callers like `start_rollout`
    can run the reconcile pass inside their own transaction (lock self-heal
    before the active-rollout check). When called with no argument the loop
    creates its own session.

    The stuck-attempt backfill runs here (not in `_reconcile_once_in_session`)
    so it fires even when NO rollout is inflight — the rows it heals belong
    to rollouts that are already terminal.
    """
    if db is None:
        async with async_session_maker() as own_db:
            await _close_stuck_attempts_of_terminal_rollouts(own_db)
            await _reconcile_once_in_session(own_db)
    else:
        await _close_stuck_attempts_of_terminal_rollouts(db)
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
            # Close the dead driver's in-flight attempt rows in the same
            # commit — 01a945e2 left four 'upgrading' rows behind forever.
            await _orphan_stuck_attempts(db, rollout.id)
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
            # Same in-flight-attempt cleanup as the heartbeat path above.
            await _orphan_stuck_attempts(db, rollout.id)
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

        # Look up the canary's prior tag from its successful attempt row.
        # Explicit query — NOT `rollout.attempts`: the relationship is
        # lazy='select', and touching it on an async session raises
        # MissingGreenlet, which crashed the resume task on the one path
        # that runs after a redeploy kills the driver mid-observation
        # (caught by _resume_with_cleanup, so it only surfaced as the
        # rollout falling through to the heartbeat orphan). Exposed by
        # test_resume_rollout_task_stamps_complete_with_failures_on_partial_failure.
        prior = (await db.execute(
            select(RolloutAttempt.prior_tag).where(
                RolloutAttempt.rollout_id == rollout.id,
                RolloutAttempt.status == "ok",
            ).order_by(RolloutAttempt.started_at).limit(1)
        )).scalar_one_or_none()

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
            # Beat DURING the batch, not just between batches: one batch can
            # legitimately consume 210s (hard_timeout_s), which exceeds the
            # reconciler's 180s orphan threshold. This loop previously had no
            # in-flight heartbeat at all, so a resumed rollout could re-abort
            # itself on its first slow batch — the very abort resume exists to
            # recover from.
            async with _heartbeating(rollout.id, "resume-batch"):
                results = await asyncio.gather(*tasks, return_exceptions=False)
            for r in results:
                if r.status == "ok":
                    total_ok += 1
                else:
                    total_fail += 1
            # Heartbeat after each batch — same rationale as _drive_rollout.
            rollout.last_progress_at = datetime.utcnow()
            await db.commit()

        # Honest terminal status — same rule as _drive_rollout's stamp.
        rollout.status = "complete" if total_fail == 0 else "complete_with_failures"
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

    # Serialize the check-then-insert window with the convergence sweep
    # (and concurrent webhook retries): both do active_rollout() → INSERT
    # with awaits in between, and without exclusion two 'active' rollouts
    # can be committed concurrently — two drivers then upgrade the same
    # tenants to different tags. The lock covers only quick DB statements.
    async with _rollout_creation_lock:
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
