"""Bridge supervisor — watches the Contabo provisioning bridge from the
platform side.

The bridge is a single-instance Python service on the VPS (systemd, no
HA). Every per-tenant operation — create, upgrade, restart, pool claim,
tenant healthchecks — funnels through it. If the bridge dies and noone
notices, the warm pool stops refilling, dead tenants stop auto-restarting,
new signups stall waiting for `_bridge_client()` calls that fail in
unhelpful ways.

This module pings `/v1/health` every BRIDGE_PROBE_INTERVAL_S and:
  - tracks consecutive failures
  - on 3 consecutive failures (~3 min): logs CRITICAL once
  - on recovery: logs INFO once
  - exposes last state via `last_status()` for the admin dashboard

Detection only — never auto-restarts the bridge. Restarting a remote
systemd service from a Railway worker is a footgun: we'd need root SSH
back into the VPS, which is exactly the surface area the mTLS-only
bridge architecture eliminated. The right escalation path is paging an
operator (currently: log scrape; future: PagerDuty/Slack webhook).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Tunables. 60s probe + 3-fail trigger = ~3 min mean detection time.
# Tighter than this and we false-positive on any transient bridge GC
# pause; looser and a real outage stays hidden too long.
BRIDGE_PROBE_INTERVAL_S = 60
BRIDGE_PROBE_TIMEOUT_S = 5
BRIDGE_FAILS_BEFORE_ALARM = 3

_state: Dict[str, Any] = {
    "consecutive_failures": 0,
    "last_probe_ts": 0.0,
    "last_success_ts": 0.0,
    "last_status": "unknown",  # unknown | healthy | unhealthy | unreachable
    "last_error": None,
    "alarm_active": False,     # True while we're in the >=3-fail state
}
_state_lock = asyncio.Lock()


async def _probe_once() -> tuple[bool, Optional[str]]:
    """One health probe. Returns (ok, error_msg)."""
    # Lazy import: docker_host_service pulls in settings, which pulls in
    # heavy dependencies. Keeping this lazy means importing this module
    # is cheap during platform startup before settings are validated.
    try:
        from app.services.docker_host_service import _bridge_client
    except Exception as e:
        return False, f"import error: {e}"

    try:
        async with _bridge_client(timeout_s=BRIDGE_PROBE_TIMEOUT_S) as client:
            r = await client.get("/v1/health")
            if r.status_code == 200:
                return True, None
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        # httpx.ConnectError, ssl errors, timeout, etc.
        return False, f"{type(e).__name__}: {str(e)[:200]}"


async def _supervisor_tick() -> None:
    """One probe + state update. Logs only on transitions, not every tick.

    The transition-only logging is deliberate — without it, a dead bridge
    would generate 1 log/min indefinitely, drowning real signal.
    """
    ok, err = await _probe_once()
    now = time.time()

    async with _state_lock:
        _state["last_probe_ts"] = now

        if ok:
            _state["last_success_ts"] = now
            prev_failures = _state["consecutive_failures"]
            _state["consecutive_failures"] = 0
            _state["last_status"] = "healthy"
            _state["last_error"] = None

            # Recovery transition.
            if _state["alarm_active"]:
                logger.info(
                    "[bridge-supervisor] RECOVERED — bridge reachable again "
                    "after %d consecutive failures",
                    prev_failures,
                )
                _state["alarm_active"] = False
                try:
                    from app.services.alerting import send_infra_alert
                    await send_infra_alert(
                        "bridge-down", "info",
                        f"Bridge RECOVERED — reachable again after {prev_failures} "
                        "consecutive failures.",
                        min_interval_s=0,
                    )
                except Exception:
                    logger.exception("[bridge-supervisor] recovery alert failed")
            return

        _state["consecutive_failures"] += 1
        _state["last_status"] = "unreachable"
        _state["last_error"] = err
        n = _state["consecutive_failures"]

        if n < BRIDGE_FAILS_BEFORE_ALARM:
            logger.warning(
                "[bridge-supervisor] probe %d/%d failed: %s",
                n, BRIDGE_FAILS_BEFORE_ALARM, err,
            )
            return

        # Alarm transition — log only on the first crossing into the
        # alarmed state. Subsequent failures stay quiet (one warn per
        # tick) until recovery.
        if not _state["alarm_active"]:
            logger.critical(
                "[bridge-supervisor] BRIDGE UNREACHABLE — %d consecutive failures "
                "(~%ds), last_error=%s. Pool reconciler, tenant healthchecks, "
                "and new-signup provisioning are paused until bridge recovers.",
                n, n * BRIDGE_PROBE_INTERVAL_S, err,
            )
            _state["alarm_active"] = True
            # PAGE the operator. The bridge is the single funnel for every pool
            # claim, tenant restart, upgrade and route change — and its death
            # DISABLES every self-heal loop (they all call the bridge), so
            # nothing recovers automatically. Detection must reach a human
            # (law 4). Before 2026-07-04 this only logged, so a 02:00 bridge
            # OOM stayed silent until a user complained.
            try:
                from app.services.alerting import send_infra_alert
                await send_infra_alert(
                    "bridge-down", "critical",
                    f"BRIDGE UNREACHABLE — {n} consecutive /v1/health failures "
                    f"(~{n * BRIDGE_PROBE_INTERVAL_S}s), last_error={err}. Pool "
                    "claims, tenant restarts, upgrades and route changes are ALL "
                    "paused and no self-heal can run until the bridge is back. "
                    "Check `systemctl status toup-bridge` on the Contabo box.",
                    min_interval_s=0,
                )
            except Exception:
                logger.exception("[bridge-supervisor] alert dispatch failed")


async def supervisor_loop() -> None:
    """Forever loop. Sleeps BRIDGE_PROBE_INTERVAL_S between ticks.

    Tick exceptions are logged and swallowed. The whole point of this
    module is to be the *thing that notices* problems, so it can never
    silently die from one.
    """
    logger.info(
        "[bridge-supervisor] started: interval=%ds timeout=%ds fails_before_alarm=%d",
        BRIDGE_PROBE_INTERVAL_S, BRIDGE_PROBE_TIMEOUT_S, BRIDGE_FAILS_BEFORE_ALARM,
    )
    # Stagger first probe so platform startup isn't blocked.
    await asyncio.sleep(10)
    while True:
        try:
            await _supervisor_tick()
        except asyncio.CancelledError:
            logger.info("[bridge-supervisor] stopped")
            raise
        except Exception:
            logger.exception("[bridge-supervisor] tick failed")
        await asyncio.sleep(BRIDGE_PROBE_INTERVAL_S)


def last_status() -> Dict[str, Any]:
    """Snapshot of current state — for the admin dashboard."""
    return {
        "consecutive_failures": _state["consecutive_failures"],
        "last_probe_ts": _state["last_probe_ts"],
        "last_success_ts": _state["last_success_ts"],
        "last_status": _state["last_status"],
        "last_error": _state["last_error"],
        "alarm_active": _state["alarm_active"],
        "probe_interval_s": BRIDGE_PROBE_INTERVAL_S,
        "fails_before_alarm": BRIDGE_FAILS_BEFORE_ALARM,
    }
