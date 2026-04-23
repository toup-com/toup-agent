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


async def _check_container_health(container: ManagedContainer) -> bool:
    """Check if a container's agent is healthy by hitting its health endpoint.

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
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") in ("healthy", "ok")
    except Exception:
        pass
    return False


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

        for container in containers:
            healthy = await _check_container_health(container)
            key = container.id

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


async def monitor_loop():
    """Run health checks every 5 minutes forever."""
    logger.info("[MONITOR] Container health monitor started (every 5 min)")
    while True:
        try:
            await check_all_containers()
        except Exception as e:
            logger.error(f"[MONITOR] Check failed: {e}")
        await asyncio.sleep(300)  # 5 minutes
