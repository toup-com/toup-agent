"""Shared write path for routine-generated Messages.

Both the success path (handler) and the nudge path (runner) post a
Message via this helper so the row shape stays consistent — same
`channel="routine"`, same Conversation creation, same DayChat counter
bumps, same live WS broadcast.

Day-as-Chat rule: `Message.day_chat_id` is canonical. Always resolved
via `resolve_day_chat_id_for_now()` — never trust `Conversation.day_chat_id`,
which is a loose hint for long-lived sessions.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


async def write_routine_message(
    db: AsyncSession,
    *,
    user_id: str,
    content: str,
    source: str,
    routine_id: Optional[str] = None,
    title: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    tz_override: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Tuple[str, Optional[str]]:
    """Create a Conversation + Message pair representing one routine
    post, bump DayChat counters, commit. Returns (message_id, day_chat_id).

    Does NOT broadcast — the caller does that AFTER receiving the id so
    the broadcast event can carry it. Broadcast is also explicitly
    isolated so a test environment without ws_chat doesn't error here.
    """
    from app.db.message_helpers import resolve_day_chat_id_for_now
    from app.db.models import Conversation, DayChat, Message

    day_chat_id = await resolve_day_chat_id_for_now(db, user_id, tz_override=tz_override)

    convo_meta = {"routine_source": source}
    if routine_id:
        convo_meta["routine_id"] = routine_id
    if extra_metadata:
        convo_meta.update(extra_metadata)

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_id,
        channel="routine",
        is_active=True,
        day_chat_id=day_chat_id,
        title=title or f"Routine: {source}",
        metadata_json=json.dumps(convo_meta),
    )
    db.add(conv)
    await db.flush()

    msg_id = str(uuid.uuid4())
    msg = Message(
        id=msg_id,
        conversation_id=conv.id,
        day_chat_id=day_chat_id,
        role="assistant",
        content=content,
        channel="routine",
        source=source,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        model_used=model_used,
    )
    db.add(msg)
    await db.flush()

    # DayChat counters — same bookkeeping as agent_runner._save_messages
    # (lines 2596-2606). Without these, the per-day budget logs drift
    # and the UI count gets stuck.
    if day_chat_id:
        dc = await db.get(DayChat, day_chat_id)
        if dc:
            dc.message_count = (dc.message_count or 0) + 1
            tt = (tokens_prompt or 0) + (tokens_completion or 0)
            if tt:
                dc.total_tokens = (dc.total_tokens or 0) + tt
            dc.last_message_at = datetime.utcnow()

    await db.commit()
    logger.info(
        "[routine_writer] wrote source=%s user_id=%s msg_id=%s day_chat_id=%s tokens=%d",
        source, user_id, msg_id, day_chat_id,
        (tokens_prompt or 0) + (tokens_completion or 0),
    )
    return msg_id, day_chat_id


async def broadcast_routine_message(
    user_id: str,
    *,
    message_id: str,
    day_chat_id: Optional[str],
    source: str,
    content: str,
    model_used: Optional[str] = None,
) -> int:
    """Push a routine Message to any live WS clients. Returns the number
    of queues that accepted the event.

    Lazy-imports `broadcast_to_user` so this module is importable in test
    environments where the full `app.api` package can't load (the
    FastAPI/Starlette version skew the test env exhibits). Broadcast in
    such envs degrades silently to a no-op."""
    try:
        from app.api.ws_chat import broadcast_to_user
    except Exception as e:
        logger.debug("[routine_writer] broadcast skipped — ws_chat unavailable: %s", e)
        return 0

    event = {
        "type": "message",
        "id": message_id,
        "day_chat_id": day_chat_id,
        "role": "assistant",
        "channel": "routine",
        "source": source,
        "content": content,
        "model_used": model_used,
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        return await broadcast_to_user(user_id, event)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("[routine_writer] broadcast failed: %s", e)
        return 0
