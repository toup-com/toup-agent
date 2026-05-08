"""
Lazy channel-adapter initialization (Phase A — never-sleep plan).

A pool-bound agent has no channel tokens at startup — it sits in lobby
mode until `POST /admin/bind` writes the tenant payload. Once bound,
any channel whose token is now populated needs to start. This module
is the single hook the bind handler calls; it delegates to the
existing hot-restart functions in `agent_main`.

This file kept intentionally small + dependency-light. The real
adapter-start logic lives in `agent_main.restart_telegram_bot()` and
`agent_main.restart_whatsapp_channel()` (already exist for
configuration-changes-mid-run support); we just trigger them with a
populated settings.

Discord and Slack adapters don't have hot-restart equivalents yet —
their startup runs only in `agent_main.lifespan`. For Phase A those
channels will only work in legacy env-mode tenants (existing behavior
unchanged). Adding hot-restart for Discord/Slack is a Phase A.0
follow-up; flagged in the never-sleep plan."""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def wake_lazy_channels() -> None:
    """Best-effort kick — fire-and-forget the restart paths for any
    channel adapter whose token is now populated. Returns immediately;
    actual adapter init runs as background tasks.

    Safe to call multiple times: each restart helper checks current
    state before doing work."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Called outside an event loop (e.g., during tests). Skip
        # silently — the bind handler always runs inside FastAPI which
        # has a running loop.
        logger.warning("[channel_init] No running event loop; skipping wake")
        return

    # Telegram
    try:
        import agent_main as _am
        loop.create_task(_am.restart_telegram_bot())
        logger.info("[channel_init] Triggered Telegram bot wake")
    except Exception as e:
        logger.warning("[channel_init] Telegram wake failed: %s", e)

    # WhatsApp
    try:
        import agent_main as _am
        loop.create_task(_am.restart_whatsapp_channel())
        logger.info("[channel_init] Triggered WhatsApp wake")
    except Exception as e:
        logger.warning("[channel_init] WhatsApp wake failed: %s", e)
