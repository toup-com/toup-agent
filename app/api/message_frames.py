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

`voice` rides the same only-when-present rule but is derived here from the
committed row rather than passed in; see `_committed_voice`. It is also what
decides the frame's BODY: a transcript row the caller heard nothing of is
projected to empty content here exactly as the three REST readers project it
(`schemas.public_heard_text`), so a row cannot say one thing to an open thread
and another to the same thread after a refetch. Such a frame is a row with no
text and no renderable part — dropped by the client, which is precisely the
"renders nothing" half of the nothing-heard rule.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from app.api.message_cards import public_text
# The nothing-heard projection (Blocker B) — see `schemas.public_heard_text`.
# `app.schemas` is a leaf (pydantic only), so this builder keeps its freedom
# from the FastAPI/DB half of `app.api`.
from app.schemas import public_heard_text

__all__ = ["message_frame", "iso_z"]


def iso_z(value: Any) -> Optional[str]:
    """Serialize a naive-UTC datetime the way every other chat frame does."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat() + "Z" if value.tzinfo is None else value.isoformat()
    return str(value)


def _committed_voice(msg: Any) -> Optional[dict]:
    """The row's own `metadata_json["voice"]`, or None.

    Read from the COMMITTED row, never handed in by a caller — unlike
    `media`/`tool_events`/`attachments`, whose wire form the caller holds and
    this builder cannot reconstruct. `voice` has no wire form: it is exactly
    the allowlisted object `sessions._clean_voice` wrote and `_merge_metadata`
    reconciled with whatever was already on the row, so taking it from the
    request body would hand an open thread a different object than the one the
    history readers return for the same id — and on a rewrite (the normal case
    for a Live utterance) the body is a REVISION, not the row.

    Deriving it here rather than adding a parameter also means both writers
    inherit it: the relay transcript (`sessions.create_session_message`) and a
    durable voice task (`agent.voice_tasks._persist_message`), which is the
    whole reason this builder exists.

    Defensive by the same rule as `day_chats._metadata`: a hand-edited or
    half-written row degrades to "no provenance" instead of raising on a path
    that is fire-and-forget live delivery.
    """
    raw = getattr(msg, "metadata_json", None)
    if not raw:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    voice = parsed.get("voice")
    return voice if isinstance(voice, dict) and voice else None


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
    # Read once: the body's projection and the `voice` key below must describe
    # the same committed object (see the `voice` note at the end of this
    # builder).
    _voice = _committed_voice(msg)
    frame: dict = {
        "type": "message",
        "id": getattr(msg, "id", None),
        "role": getattr(msg, "role", None),
        # The one message surface that bypasses every serializer gets the same
        # TWO guards the history readers get: `public_text` for the ROLE (see
        # message_cards.py) and `public_heard_text` for a voice transcript row
        # the caller heard nothing of (see schemas.py). A live frame that
        # carried the generated text while the three REST readers project it
        # away would put the divergence back inside a single session — the
        # open thread showing words that the same thread loses on refetch.
        "content": public_heard_text(
            public_text(getattr(msg, "role", None), getattr(msg, "content", "") or ""),
            _voice,
        ),
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
    # Voice provenance (R48 §8, contract v0.2 `message.voice`) — present only
    # when the row has it, like every renderable part above, so a client that
    # does not read it cannot tell this build from the previous one. Live
    # delivery carries it for the same reason it carries `tool_events`: the
    # three REST readers now return it, and a frame that omitted it would make
    # an open thread disagree with the same thread after a refetch — here that
    # would mean a row silently changing from "this is what you heard" to a
    # complete answer, or back, under the reader.
    if _voice:
        frame["voice"] = dict(_voice)
    return frame
