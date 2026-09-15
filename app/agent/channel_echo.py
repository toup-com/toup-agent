"""Live publication of an OFF-APP channel turn to the user's own sockets.

A WhatsApp / Telegram / Discord / Slack turn persists both rows and then
tells nobody: `make_channel_handler` runs the agent and calls `send_text`,
and `AgentRunner._save_messages` broadcasts nothing. So a user with the app
or the web tab open watched their WhatsApp conversation happen invisibly
and only saw it after a day refetch.

One module, four channels, because Telegram bypasses `BaseChannel`
entirely — two copies of this frame would drift.

The `channel` key is MANDATORY on every frame here. Mobile's filter
(`api.ts:3189`) admits a `type:'message'` frame whatever its channel, but
its settle predicate (`api.ts:2703`) reads a CHANNEL-LESS assistant row as
a candidate answer to the phone's OWN in-flight turn — so an unstamped
frame would stand that turn's working surface down and withhold its error
bubble. `ws_chat`'s late-answer frame omits the key deliberately, because
it IS the phone's own answer; do not copy its shape here.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Channels whose turns are invisible to the app/web socket. `web`, `mobile`,
# `app` and `voice` are excluded on purpose: they already stream to their own
# socket, and echoing them would double every bubble.
OFF_APP_ORIGIN_CHANNELS = frozenset({"whatsapp", "telegram", "discord", "slack"})


class _PersistedRow:
    """Just enough of a `Message` for the day route's serializers."""

    __slots__ = ("id", "attachments", "metadata_json", "__weakref__")

    def __init__(self, msg_id: str, attachments: Any, tool_events: Any) -> None:
        import json as _json

        self.id = msg_id
        self.attachments = attachments
        self.metadata_json = _json.dumps({"tool_events": tool_events or []})


def _public_parts(msg_id: str, attachments: Any, tool_events: Any) -> Dict[str, Any]:
    """`attachments` / `tool_events` exactly as GET /api/day-chats/{date}/
    messages would serialize them for this row — `storage_path` stripped,
    `download_url`/`preview_url` added, malformed tool records dropped, the
    public copy applied. Falls back to stripping the internal path when the
    route module is unavailable; never raises."""
    out: Dict[str, Any] = {}
    if not attachments and not tool_events:
        return out
    row = _PersistedRow(msg_id, attachments, tool_events)
    try:
        from app.api.day_chats import _serialize_attachments, _serialize_tool_events

        atts = _serialize_attachments(row) if attachments else None
        events = _serialize_tool_events(row) if tool_events else None
    except Exception:  # noqa: BLE001
        logger.debug("[channel-echo] REST serializers unavailable", exc_info=True)
        atts = [
            {k: v for k, v in a.items() if k != "storage_path"}
            for a in (attachments or []) if isinstance(a, dict)
        ] or None
        events = [e for e in (tool_events or []) if isinstance(e, dict)] or None
    if atts:
        out["attachments"] = atts
    if events:
        out["tool_events"] = events
    return out


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    if isinstance(value, str) and value:
        return value
    return datetime.utcnow().isoformat() + "Z"


async def broadcast_channel_turn(
    user_id: str,
    *,
    origin_channel: str,
    session_id: str,
    persisted: Dict[str, Any],
    user_text: str,
    assistant_text: str,
) -> int:
    """Publish the user row then the assistant row of a channel turn.

    Returns the number of sockets the ASSISTANT frame reached (0 when the
    turn was not eligible, nothing persisted, or delivery failed). Never
    raises: live delivery may not fail a turn that already happened.
    """
    if origin_channel not in OFF_APP_ORIGIN_CHANNELS or not persisted:
        return 0

    try:
        # Lazy: every other `{type:'message'}` producer imports ws_chat
        # inside the function to avoid the module cycle.
        from app.api.ws_chat import broadcast_to_user
        from app.api.message_cards import public_text

        day_chat_id = persisted.get("day_chat_id")
        sent = 0

        def _frame(role: str, msg_id: Optional[str], created_at: Any, text: str) -> Dict[str, Any]:
            return {
                "type": "message",
                "id": msg_id,
                "role": role,
                "content": public_text(role, text),
                "created_at": _iso(created_at),
                "channel": origin_channel,
                "day_chat_id": day_chat_id,
                "conversation_id": session_id,
            }

        # User row first: the app must render the conversation in the order
        # it happened, and the two frames are independent sends.
        user_msg_id = persisted.get("user_message_id")
        if user_msg_id:
            await broadcast_to_user(
                user_id,
                _frame("user", user_msg_id, persisted.get("user_created_at"), user_text or ""),
            )

        asst_msg_id = persisted.get("asst_message_id")
        if asst_msg_id:
            frame = _frame(
                "assistant",
                asst_msg_id,
                persisted.get("asst_created_at"),
                assistant_text or "",
            )
            if persisted.get("media"):
                frame["media"] = persisted["media"]
            # Attachments and tool records go through the SAME serializers
            # the day route uses. The raw persisted values carry the agent's
            # internal `storage_path` and no `download_url`, and a tool
            # record without its public copy renders a different pill live
            # than after a refetch.
            frame.update(_public_parts(
                asst_msg_id, persisted.get("attachments"), persisted.get("tool_events"),
            ))
            sent = await broadcast_to_user(user_id, frame)

        return sent
    except Exception:
        logger.warning(
            "[channel-echo] broadcast failed channel=%s user=%s",
            origin_channel, (user_id or "")[:8], exc_info=True,
        )
        return 0
