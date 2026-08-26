"""Connector + grant cards — server-driven chat payloads (Phase 4).

The contract (shared with the Round-26 app session, 2026-08-23):
payload built ONCE, persisted verbatim under a named metadata_json key
on its own assistant Message, echoed verbatim in a WS frame of the
same name, NO channel key. In-place updates re-broadcast the same
frame type with the same `id` — clients upsert by id — and re-persist
onto the SAME message row so a reload agrees with the live view.

  automation_connector_card  — AutomationAuthSession (10-min TTL)
  automation_grant_card      — platform AutomationGrant (1-h TTL)

Writers here follow write_routine_message's Conversation/DayChat
bookkeeping so the cards land in the day chat like any routine post.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AutomationAuthSession

logger = logging.getLogger(__name__)

CONNECTOR_CARD_KEY = "automation_connector_card"
GRANT_CARD_KEY = "automation_grant_card"


def connector_card_payload(
    session: AutomationAuthSession,
    *,
    name: str,
    icon: Optional[str],
    scopes: list[dict],
) -> dict:
    return {
        "id": session.id,
        "connector_id": session.connector_id,
        "name": name,
        "icon": icon,
        "mode": session.mode,
        "scopes": scopes,
        "status": session.status,
        "retry_used": bool(session.retry_used),
        "connect_url": f"/api/oauth/connect/{session.connector_id}",
        "created_at": session.created_at.isoformat() + "Z",
        "expires_at": session.expires_at.isoformat() + "Z",
    }


# CONTRACTS-R31 §4.1: the ONLY metadata key an automation may write
# into the day chat. Everything else belongs in the thread.
#
# The three setup cards are the documented exception and they are not
# an automation NARRATING — they are the connector/grant consent flow
# and the setup card, which happen in the main chat because that is
# where the user typed the request (§2.1's `automation_setup` clause).
DAY_CHAT_METADATA_KEYS: frozenset = frozenset({
    "automation_notification",
    "automation_connector_card",
    "automation_grant_card",
})


class AutomationDayChatWrite(RuntimeError):
    """An automation tried to put something else in the day chat."""


def assert_day_chat_clean(metadata_key: str) -> None:
    """Refuse a day-chat write outside the sanctioned key set.

    §4.1's storage invariant, at the one remaining writer. R31 retired
    `session.write_session_message` — the path that put run cards,
    memory chips, auto-pause notices, pending cards and draft cards in
    the user's main chat — and this is the door that stays open, so it
    is the door that needs a check.

    Raises in dev/test so a new card fails at its first call; logs in
    production, because losing one card is survivable and losing a run
    is not.
    """
    if metadata_key in DAY_CHAT_METADATA_KEYS:
        return
    msg = (
        f"day-chat write refused (CONTRACTS-R31 §4.1): "
        f"{metadata_key!r} is not one of {sorted(DAY_CHAT_METADATA_KEYS)}. "
        "Write a turn with ledger.append_turn instead."
    )
    logger.error("[automations] %s", msg)
    try:
        from app.config import settings
        if (getattr(settings, "environment", "") or "").lower() != "production":
            raise AutomationDayChatWrite(msg)
    except AutomationDayChatWrite:
        raise
    except Exception:  # noqa: BLE001 — settings unavailable ⇒ fail soft
        pass


async def write_card_message(
    db: AsyncSession,
    *,
    user_id: str,
    content: str,
    metadata_key: str,
    payload: dict,
    title: str,
) -> tuple[str, Optional[str]]:
    """Persist one assistant message carrying the card. Returns
    (message_id, day_chat_id)."""
    assert_day_chat_clean(metadata_key)
    from app.agent.conversation_resolver import (
        resolve_or_create_day_conversation,
    )
    from app.db.message_helpers import resolve_day_chat_id_for_now
    from app.db.models import DayChat, Message

    day_chat_id = await resolve_day_chat_id_for_now(db, user_id)
    conv = await resolve_or_create_day_conversation(
        db,
        user_id=user_id,
        day_chat_id=day_chat_id,
        channel="routine",
        title=title,
        metadata={"routine_source": "automation"},
    )
    msg_id = str(uuid.uuid4())
    db.add(Message(
        id=msg_id,
        conversation_id=conv.id,
        day_chat_id=day_chat_id,
        role="assistant",
        content=content,
        channel="routine",
        source="automation",
        metadata_json=json.dumps({metadata_key: payload}, default=str),
    ))
    await db.flush()
    if day_chat_id:
        dc = await db.get(DayChat, day_chat_id)
        if dc:
            dc.message_count = (dc.message_count or 0) + 1
            dc.last_message_at = datetime.utcnow()
    await db.commit()
    return msg_id, day_chat_id


async def update_card_message(
    db: AsyncSession,
    *,
    message_id: Optional[str],
    metadata_key: str,
    payload: dict,
) -> None:
    """Re-persist the updated payload onto the card's message row so
    reloads agree with the live frame. Missing row is fine (the card
    may predate its message on the staging path)."""
    if not message_id:
        return
    from app.db.models import Message
    msg = await db.get(Message, message_id)
    if msg is None:
        return
    try:
        meta = json.loads(msg.metadata_json) if msg.metadata_json else {}
    except (ValueError, TypeError):
        meta = {}
    meta[metadata_key] = payload
    msg.metadata_json = json.dumps(meta, default=str)
    await db.commit()


async def broadcast_card(user_id: str, frame_type: str, payload: dict) -> None:
    """Best-effort live push; the persisted copy is the durable record."""
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, {"type": frame_type, **payload})
    except Exception as e:  # noqa: BLE001 — no live socket is normal
        logger.debug("[automations] card broadcast skipped: %s", e)
