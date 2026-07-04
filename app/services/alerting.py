"""Infra alerting — one canonical path to the operator's Telegram.

Law 4 of the bulletproof plan: users are never the monitoring system.
Every self-heal loop that FIRES (reclaimed a stranded user, re-bound a
keyless agent, restarted a sick container) must tell the operator —
healing is good, repeated healing means something upstream broke.

Uses the dedicated infra bot (`INFRA_ALERT_TELEGRAM_TOKEN`), falling
back to the admin bot. Per-category rate limiting keeps a flapping
loop from flooding the chat: the first alert in a window goes out,
repeats within `min_interval_s` are counted and folded into the next
one.
"""

import logging
import time
from typing import Dict

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_LEVEL_PREFIX = {"info": "🔵", "warning": "⚠️", "critical": "🚨"}

# category -> (last_sent_epoch, suppressed_count)
_last_sent: Dict[str, list] = {}


async def send_infra_alert(
    category: str,
    level: str,
    message: str,
    *,
    min_interval_s: int = 600,
) -> bool:
    """Send a Telegram alert, rate-limited per category.

    Returns True if the alert was actually sent, False if suppressed
    (rate limit) or unconfigured. Never raises.
    """
    token = settings.infra_alert_telegram_token or settings.admin_alert_telegram_token
    chat_id = settings.infra_alert_telegram_chat_id or settings.admin_alert_telegram_chat_id
    if not token or not chat_id:
        logger.info("[infra-alert] no telegram config; skipping: %s", message)
        return False

    now = time.time()
    entry = _last_sent.setdefault(category, [0.0, 0])
    if now - entry[0] < min_interval_s:
        entry[1] += 1
        logger.info(
            "[infra-alert] suppressed (%s, %d in window): %s",
            category, entry[1], message,
        )
        return False

    suppressed = entry[1]
    entry[0] = now
    entry[1] = 0

    prefix = _LEVEL_PREFIX.get(level, "•")
    body = f"{prefix} [{category}] {message}"
    if suppressed:
        body += f"\n(+{suppressed} similar suppressed in the last window)"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": body,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
        return True
    except Exception as e:
        logger.warning("[infra-alert] telegram send failed: %s", e)
        return False
