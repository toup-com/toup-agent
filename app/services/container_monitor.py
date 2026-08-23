"""
Container Health Monitor — periodically checks managed containers
and alerts admin via Telegram + in-app notification if any go down.

Runs on the platform (Railway), not on the VPS.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ManagedContainer

logger = logging.getLogger(__name__)

# Track consecutive failures per container
_failure_counts: Dict[str, int] = {}
_last_alert: Dict[str, datetime] = {}
ALERT_AFTER_FAILURES = 3  # Alert after 3 consecutive failures (15 min)
ALERT_COOLDOWN = timedelta(hours=1)  # Don't spam — max 1 alert per container per hour

# ── Tenant DB path ────────────────────────────────────────────────
# 2026-08-01: pgbouncer died and every tenant lost its database. Chat 500'd
# fleet-wide for six minutes and NOTHING alerted, because the containers were
# up, Postgres was up, and /agent/health kept returning 200 with
# `status: "healthy"`. The agent's own db_watchdog had already detected it and
# published `db_ok: false` in that same payload — this monitor just never read
# the field. That is the "/agent/health green != chat works" trap, and it is
# the second time it has cost a fleet-wide outage.
#
# db_ok is deliberately NOT folded into `healthy` below. `healthy` drives the
# auto-restart path, and restarting 55 containers because a SHARED component
# (pgbouncer/Postgres) is down is both useless and harmful — it would turn one
# outage into a restart storm. So this is a separate, aggregated signal that
# alerts and never acts.
_db_down_counts: Dict[str, int] = {}
_last_db_alert: Optional[datetime] = None
# One tenant reporting db_ok=false could be its own database; wait a cycle.
# Two or more at once is a shared component — say so immediately, because the
# operator's next move differs completely.
DB_ALERT_AFTER_FAILURES = 2
DB_ALERT_MIN_TENANTS_IMMEDIATE = 2
DB_ALERT_COOLDOWN = timedelta(minutes=30)

# ── Turn path (round N P0, 2026-08-23) ────────────────────────────
# Same contract as the DB path: an agent can be UP and healthy while unable
# to serve a single chat turn (`turn_ready: false` — its pipeline init threw
# and was swallowed). Tracked and alerted, never acted on: for a bad IMAGE
# this hits every recreated tenant at once and a restart storm fixes
# nothing. One consecutive miss alerts — a dead turn path is a total outage
# for that user, and the founder finding it in his own chat at 11:28 PM is
# the incident this block exists to end.
_turn_down_counts: Dict[str, int] = {}
_last_turn_alert: Optional[datetime] = None
TURN_ALERT_AFTER_FAILURES = 1
TURN_ALERT_COOLDOWN = timedelta(minutes=30)


def verdict_from_health_body(data: dict) -> "tuple[bool, Optional[bool], Optional[bool]]":
    """Read `(healthy, db_ok, turn_ready)` out of an /agent/health body.

    Pure, so the decision that mattered on 2026-08-01 is testable without a
    database or an HTTP stack. `db_ok` is a SIBLING of `status`, not nested
    under it — a tenant whose database is unreachable still answers
    `{"status": "healthy", "db_ok": false}`, which is exactly why reading
    only `status` missed a fleet-wide outage.

    `turn_ready` (round N P0, 2026-08-23 — the THIRD "green health while
    every chat fails" arc): the agent-pipeline init threw and was swallowed,
    `_agent_runner` stayed None, and every chat turn answered "Agent not
    available" behind a 200. Same sibling contract as db_ok: tracked and
    alerted, never acted on.

    A missing or non-boolean field is None, meaning "the agent did not
    say" — never "down". Older images and pool-generic boots omit them, and
    treating absence as failure would page forever.
    """
    healthy = data.get("status") in ("healthy", "ok")
    db_ok = data.get("db_ok")
    turn_ready = data.get("turn_ready")
    return (
        healthy,
        (db_ok if isinstance(db_ok, bool) else None),
        (turn_ready if isinstance(turn_ready, bool) else None),
    )


async def _probe_agent_health(
    container: ManagedContainer,
) -> "tuple[bool, Optional[bool], Optional[bool]]":
    """Probe a container's agent health endpoint.

    Returns `(healthy, db_ok)`. `healthy` is the liveness verdict that drives
    the alert + auto-restart path below. `db_ok` is the agent's own
    db_watchdog verdict on whether its tenant database is reachable, and is
    reported separately — see the note by `_db_down_counts`.
    `db_ok` is None when the agent did not report the field at all.

    Phase 3: URL is the HTTPS subdomain per AgentConfig.agent_url, fronted
    by Caddy on 443. The platform doesn't need bridge mTLS for this check —
    /agent/health is reachable without client cert (the tenant's own agent
    responds publicly; X-Agent-Key only gates authenticated endpoints).
    """
    from app.db.models import AgentConfig
    from app.db.database import async_session_maker
    from sqlalchemy import select

    # Prefer AgentConfig.agent_url (HTTPS subdomain) over the legacy
    # http://{docker_host_ip}:{port} form.
    url: str | None = None
    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.agent_url).where(AgentConfig.user_id == container.user_id)
        )
        agent_url = result.scalar_one_or_none()
    if agent_url:
        url = f"{agent_url.rstrip('/')}/agent/health"
    elif container.host_port and settings.docker_host_ip:
        # Legacy fallback — only hit during the Phase 3 transition window
        # when AgentConfig rows haven't been populated with HTTPS URLs yet.
        url = f"http://{settings.docker_host_ip}:{container.host_port}/agent/health"
    else:
        return False, None, None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return verdict_from_health_body(resp.json())
    except Exception:
        pass
    return False, None, None


async def _send_telegram_alert(message: str):
    """Send alert to admin via Telegram."""
    bot_token = settings.admin_alert_telegram_token
    chat_id = settings.admin_alert_telegram_chat_id
    if not bot_token or not chat_id:
        logger.warning("[MONITOR] No Telegram alert config — skipping alert")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
            })
        logger.info("[MONITOR] Telegram alert sent")
    except Exception as e:
        logger.warning(f"[MONITOR] Telegram alert failed: {e}")


async def _store_alert(db: AsyncSession, container: ManagedContainer, message: str):
    """Store alert in container's error_message field for admin panel visibility."""
    await db.execute(
        update(ManagedContainer)
        .where(ManagedContainer.id == container.id)
        .values(error_message=message)
    )
    await db.commit()


async def check_all_containers():
    """Check health of all managed containers. Called periodically."""
    from app.db.database import async_session_maker

    if not settings.docker_host_ip:
        return

    async with async_session_maker() as db:
        result = await db.execute(
            select(ManagedContainer)
            .where(ManagedContainer.status.in_(["running", "provisioning"]))
        )
        containers = result.scalars().all()

        if not containers:
            return

        db_down: list[ManagedContainer] = []
        turn_down: list[ManagedContainer] = []

        for container in containers:
            healthy, db_ok, turn_ready = await _probe_agent_health(container)
            key = container.id

            # Turn path — tracked and alerted, never acted on (see the
            # `_turn_down_counts` note). Only an ANSWERING container can
            # report it; a dead one belongs to the liveness path below.
            if turn_ready is False:
                _turn_down_counts[key] = _turn_down_counts.get(key, 0) + 1
                turn_down.append(container)
                logger.error(
                    "[MONITOR] %s reports turn_ready=false (%d consecutive) — "
                    "agent is up but cannot serve a chat turn",
                    container.container_name, _turn_down_counts[key],
                )
            elif turn_ready is True:
                _turn_down_counts.pop(key, None)

            # Tenant DB path — tracked and alerted, never acted on. Only a
            # container that is otherwise ANSWERING can report db_ok; a dead
            # container returns (False, None) and belongs to the liveness
            # path below, not here.
            if db_ok is False:
                _db_down_counts[key] = _db_down_counts.get(key, 0) + 1
                db_down.append(container)
                logger.warning(
                    "[MONITOR] %s reports db_ok=false (%d consecutive) — "
                    "agent is up but its database is unreachable",
                    container.container_name, _db_down_counts[key],
                )
            elif db_ok is True:
                _db_down_counts.pop(key, None)

            if healthy:
                # Reset failure count and clear error
                if key in _failure_counts:
                    if _failure_counts[key] >= ALERT_AFTER_FAILURES:
                        # Was down, now recovered — send recovery alert
                        msg = (
                            f"✅ <b>Container Recovered</b>\n"
                            f"Container: <code>{container.container_name}</code>\n"
                            f"Port: {container.host_port}\n"
                            f"Time: {datetime.utcnow().strftime('%H:%M UTC')}"
                        )
                        await _send_telegram_alert(msg)
                        await _store_alert(db, container, None)
                    del _failure_counts[key]
                # Update status to running if it was something else
                if container.status != "running":
                    container.status = "running"
                    container.error_message = None
                    await db.commit()
            else:
                # Increment failure count
                _failure_counts[key] = _failure_counts.get(key, 0) + 1
                count = _failure_counts[key]

                logger.warning(
                    f"[MONITOR] {container.container_name} unhealthy "
                    f"(failure {count}/{ALERT_AFTER_FAILURES})"
                )

                if count >= ALERT_AFTER_FAILURES:
                    # Check cooldown
                    last = _last_alert.get(key)
                    now = datetime.utcnow()

                    if not last or (now - last) > ALERT_COOLDOWN:
                        _last_alert[key] = now

                        msg = (
                            f"🚨 <b>Container Down</b>\n"
                            f"Container: <code>{container.container_name}</code>\n"
                            f"Port: {container.host_port}\n"
                            f"Failures: {count} consecutive\n"
                            f"Time: {now.strftime('%H:%M UTC')}\n\n"
                            f"Auto-restart attempting..."
                        )
                        await _send_telegram_alert(msg)
                        await _store_alert(db, container, f"Unhealthy since {now.strftime('%H:%M UTC')}")

                        # Attempt auto-restart via SSH
                        try:
                            from app.services.docker_host_service import restart_container
                            await restart_container(db, container.user_id)
                            logger.info(f"[MONITOR] Auto-restarted {container.container_name}")

                            restart_msg = (
                                f"🔄 <b>Auto-Restart Triggered</b>\n"
                                f"Container: <code>{container.container_name}</code>\n"
                                f"Waiting for health check..."
                            )
                            await _send_telegram_alert(restart_msg)
                        except Exception as e:
                            logger.error(f"[MONITOR] Auto-restart failed: {e}")
                            fail_msg = (
                                f"❌ <b>Auto-Restart Failed</b>\n"
                                f"Container: <code>{container.container_name}</code>\n"
                                f"Error: {str(e)[:200]}\n\n"
                                f"Manual intervention required."
                            )
                            await _send_telegram_alert(fail_msg)

                    # Update status
                    container.status = "error"
                    container.error_message = f"Unhealthy for {count * 5} minutes"
                    await db.commit()

        await _alert_on_db_path(db_down)
        await _alert_on_turn_path(turn_down)


async def _alert_on_turn_path(turn_down: "list[ManagedContainer]") -> None:
    """One aggregated alert for agents that cannot serve a chat turn.

    Mirrors `_alert_on_db_path`: aggregated because the failure this exists
    for (a bad agent image whose init throws) hits every recreated tenant at
    once. Alerts on the FIRST observation — turn_ready=false is a total chat
    outage for that user, already at least one monitor cycle old.
    """
    global _last_turn_alert

    if not turn_down:
        return
    if len(turn_down) < 2:
        only = turn_down[0]
        if _turn_down_counts.get(only.id, 0) < TURN_ALERT_AFTER_FAILURES:
            return

    now = datetime.utcnow()
    if _last_turn_alert and (now - _last_turn_alert) < TURN_ALERT_COOLDOWN:
        return
    _last_turn_alert = now

    names = ", ".join(f"<code>{c.container_name}</code>" for c in turn_down[:3])
    if len(turn_down) > 3:
        names += f" +{len(turn_down) - 3} more"
    many = len(turn_down) >= 2
    verdict = (
        f"<b>{len(turn_down)} tenants at once → almost certainly the current "
        f"agent image.</b> Check the latest rollout; roll back the image. "
        f"Restarting containers will NOT help."
        if many else
        "<b>Single tenant</b> — check its /agent/diagnose (agent_runner "
        "check has the full init traceback)."
    )
    await _send_telegram_alert(
        f"💬 <b>Agent cannot serve chat turns</b>\n"
        f"{names}\n"
        f"Time: {now.strftime('%H:%M UTC')}\n\n"
        f"{verdict}\n\n"
        f"<i>Agents are UP and /agent/health returns 200 — they report "
        f"turn_ready=false. Every chat message answers 'Agent not "
        f"available'.</i>"
    )


async def _alert_on_db_path(db_down: "list[ManagedContainer]") -> None:
    """One aggregated alert for tenants whose database is unreachable.

    Aggregated on purpose: the failure this exists for (pgbouncer down) hits
    every tenant at once, and 55 separate messages would bury the one fact
    that matters. The count IS the diagnosis — several tenants at once means
    a shared component, and the alert says so, because "restart the
    container" is the wrong move there and is what an operator would
    otherwise reach for.
    """
    global _last_db_alert

    if not db_down:
        return

    fleet_wide = len(db_down) >= DB_ALERT_MIN_TENANTS_IMMEDIATE
    # A single tenant may just be its own database; give it one more cycle.
    # Several at once is shared infrastructure — say it now.
    if not fleet_wide:
        only = db_down[0]
        if _db_down_counts.get(only.id, 0) < DB_ALERT_AFTER_FAILURES:
            return

    now = datetime.utcnow()
    if _last_db_alert and (now - _last_db_alert) < DB_ALERT_COOLDOWN:
        return
    _last_db_alert = now

    names = ", ".join(f"<code>{c.container_name}</code>" for c in db_down[:3])
    if len(db_down) > 3:
        names += f" +{len(db_down) - 3} more"

    if fleet_wide:
        verdict = (
            f"<b>{len(db_down)} tenants at once → shared component.</b>\n"
            f"Check pgbouncer first (<code>systemctl is-active pgbouncer</code>, "
            f"port 6432), then Postgres.\n"
            f"<b>Restarting containers will NOT help</b> and risks a restart storm."
        )
    else:
        verdict = (
            "<b>Single tenant</b> — likely its own database, not shared "
            "infrastructure. Check that tenant's DB before touching the fleet."
        )

    await _send_telegram_alert(
        f"🗄️ <b>Tenant DB path unreachable</b>\n"
        f"{names}\n"
        f"Time: {now.strftime('%H:%M UTC')}\n\n"
        f"{verdict}\n\n"
        f"<i>Agents are UP and /agent/health returns 200 — they report "
        f"db_ok=false. Chat is failing for these users.</i>"
    )


async def monitor_loop():
    """Run health checks every 5 minutes forever."""
    logger.info("[MONITOR] Container health monitor started (every 5 min)")
    while True:
        try:
            await check_all_containers()
        except Exception as e:
            logger.error(f"[MONITOR] Check failed: {e}")
        await asyncio.sleep(300)  # 5 minutes
