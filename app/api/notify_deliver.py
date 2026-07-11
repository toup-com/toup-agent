"""Agent-side out-of-band delivery endpoint (Autopilot PR4).

``POST /notify/deliver`` — called BY THE PLATFORM dispatcher (PR3)
when a notification couldn't reach any push device. Channel
connections (Telegram polling bot, WhatsApp Baileys sidecar) live in
THIS container, never on the platform, so the platform can only ask
us to deliver.

Auth: X-Agent-Key vs the container's own settings.agent_api_key — the
triggers_inbound.py pattern (401 on mismatch; the platform is the only
caller holding the per-tenant key).

Delivery reuses the routine fan-out (channel_dispatcher), so the
Telegram markdown-retry and no-recipient semantics are the proven
ones. Response tells the dispatcher whether ANYTHING was delivered:
{"delivered": bool, "channels": {...}} — a 200 with delivered=false
means "agent reachable, no channel had a recipient", which the
dispatcher treats as not-delivered.

Mounted in agent_main (agent entrypoint — NOT platform_main). The
/notify prefix has no param catch-alls, so route order is safe.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notify", tags=["Notify Deliver"])


class NotifyDeliverRequest(BaseModel):
    event_kind: str = Field(max_length=32)
    title: str = Field(min_length=1, max_length=200)
    body: Optional[str] = Field(default=None, max_length=2000)
    data: Optional[Dict[str, Any]] = None


class NotifyDeliverResponse(BaseModel):
    delivered: bool
    channels: Dict[str, Any]


@router.post("/deliver", response_model=NotifyDeliverResponse)
async def notify_deliver(
    body: NotifyDeliverRequest,
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> NotifyDeliverResponse:
    expected_key = (getattr(settings, "agent_api_key", "") or "").strip()
    user_id = (getattr(settings, "user_id", "") or "").strip()
    if not expected_key or not user_id:
        # Pool-lobby container with no bound tenant — nothing to
        # deliver to. 401 (not 5xx) so the dispatcher doesn't retry
        # against a lobby.
        raise HTTPException(401, "agent not bound")
    if not x_agent_key or x_agent_key.strip() != expected_key:
        raise HTTPException(401, "invalid agent key")

    from app.agent.routines.channel_dispatcher import (
        deliver_to_extra_channels_detailed,
    )
    from app.db.database import async_session_maker

    text = body.body or body.title
    results = await deliver_to_extra_channels_detailed(
        user_id=user_id,
        delivery_channels=["telegram", "whatsapp"],
        routine_name=body.title,
        content=text,
        db_session_maker=async_session_maker,
    )
    delivered = any(
        isinstance(entry, dict) and entry.get("status") == "delivered"
        for entry in results.values()
    )
    logger.info(
        "notify/deliver kind=%s delivered=%s channels=%s",
        body.event_kind, delivered,
        {ch: (e or {}).get("status") for ch, e in results.items()},
    )
    return NotifyDeliverResponse(delivered=delivered, channels=results)
