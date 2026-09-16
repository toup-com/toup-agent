"""One builder for the `{"type": "message"}` chat-socket frame.

Two writers commit rows into a user's day thread outside `AgentRunner`:
`sessions.create_session_message` (the platform voice relay's transcript) and
`agent.voice_tasks._persist_message` (a durable voice task's result). Before
this module only the first of them broadcast anything, and it built the frame
inline — so a task the user started by voice reached an already-open ChatScreen
only on the next history fetch, and any field added to one writer's frame was
silently missing from the other's.

The frame's shape is the client's contract (`ChatScreen.handleServerMessage`):
`id`/`role`/`content`/`created_at`/`channel` always, and the renderable parts
(`media`, `tool_events`, `attachments`, `app_artifact`) only when present — a
row with no text and no renderable part is dropped by the client, which is
correct and must stay possible to reason about from one place.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from app.api.message_cards import public_text

__all__ = ["message_frame", "iso_z"]


def iso_z(value: Any) -> Optional[str]:
    """Serialize a naive-UTC datetime the way every other chat frame does."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat() + "Z" if value.tzinfo is None else value.isoformat()
    return str(value)


def message_frame(
    msg: Any,
    *,
    channel: Optional[str] = None,
    day_chat_id: Optional[str] = None,
    media: Optional[Mapping[str, Any]] = None,
    tool_events: Optional[Sequence[Mapping[str, Any]]] = None,
    attachments: Optional[Sequence[Mapping[str, Any]]] = None,
    app_artifact: Optional[Mapping[str, Any]] = None,
    revision: Optional[int] = None,
) -> dict:
    """Build the live frame for one committed `Message` row.

    `msg` is the ORM row (or anything with the same attributes). Everything the
    row cannot answer for itself — the conversation's channel, the wire form of
    its attachments (whose URLs are message-scoped) — is passed in.
    """
    created = getattr(msg, "created_at", None)
    frame: dict = {
        "type": "message",
        "id": getattr(msg, "id", None),
        "role": getattr(msg, "role", None),
        # The one message surface that bypasses every serializer gets the same
        # guard the history readers get (see message_cards.py).
        "content": public_text(getattr(msg, "role", None), getattr(msg, "content", "") or ""),
        "created_at": iso_z(created) or (datetime.utcnow().isoformat() + "Z"),
        "channel": channel if channel is not None else getattr(msg, "channel", None),
    }
    _day = day_chat_id or getattr(msg, "day_chat_id", None)
    if _day:
        frame["day_chat_id"] = _day
    # Message identity (R46 C1). Both are nullable columns and an older agent
    # image has neither, so absence must stay indistinguishable from today.
    _cmid = getattr(msg, "client_msg_id", None)
    if _cmid:
        frame["client_msg_id"] = _cmid
    _occurred = iso_z(getattr(msg, "occurred_at", None))
    if _occurred:
        frame["occurred_at"] = _occurred
    # How many times this row has been REWRITTEN. The deterministic row ids
    # (`uuid5(result:<job id>)`, and the voice transcript's UPSERT key) mean the
    # same id is legitimately broadcast more than once with different content —
    # a durable task corrected by `steer` finishes a second time under the id it
    # already used. The client de-dupes by id with no update path, so without a
    # monotonic marker the CORRECTED answer is the one that is discarded and the
    # stale one is what the on-device day cache keeps. Additive: a client that
    # does not read it behaves exactly as today.
    if revision is not None:
        try:
            frame["revision"] = int(revision)
        except (TypeError, ValueError):
            pass
    if media:
        frame["media"] = dict(media)
    if tool_events:
        # Live delivery carries the run too, or an open thread shows the turn as
        # a bare bubble until the next resync and then silently grows a run card
        # under the reader.
        frame["tool_events"] = [dict(e) for e in tool_events]
    if attachments:
        # `storage_path` is the internal object key ("{user_id}/{att_id}_{name}")
        # that `files.py` authorizes against. Both REST serializers delete it on
        # purpose (day_chats._serialize_attachments, sessions._message_to_response);
        # this builder copied caller dicts wholesale, so the one surface that
        # bypasses both put it — and the raw tenant id inside it — on the wire.
        # Stripped HERE so both writers inherit the guarantee.
        frame["attachments"] = [
            {k: v for k, v in dict(a).items() if k != "storage_path"}
            for a in attachments
        ]
    if app_artifact:
        frame["app_artifact"] = dict(app_artifact)
    return frame
