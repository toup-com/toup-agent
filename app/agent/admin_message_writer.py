"""Shared write path for operator-authored (Admin Dispatch) Messages.

Mirrors ``app/agent/routines/message_writer.py`` — same Conversation
resolution, same write/broadcast split (the writer commits and returns the
id; the CALLER broadcasts, because the frame has to carry that id) — with
three deliberate differences that ARE the feature:

- ``Message.day_chat_id`` is NULL. ``load_day_context`` selects
  ``WHERE Message.day_chat_id = :id``, so a NULL row is structurally
  invisible to the agent's context. An admin notice comes FROM THE
  OPERATOR, not from the agent; the agent must never read it and must
  never start answering for it. The Conversation still carries the day —
  see ``write_admin_notice``.
- ``Message.id`` is a uuid5 of (dispatch, user) rather than a fresh uuid4,
  which is what makes a re-POST idempotent (see
  ``admin_notice_message_id``).
- No DayChat counter bump. The row is not in a day chat, so bumping
  ``message_count`` would drift the per-day budget logs against a row
  nothing in the day can see.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Mapping, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.admin_dispatch import ORIGIN_ADMIN


logger = logging.getLogger(__name__)


def admin_notice_message_id(dispatch_id: str, user_id: str) -> str:
    """The Message id for one (dispatch, user) pair — deterministic BECAUSE
    it is the idempotency key.

    The platform's fan-out worker retries a target on any transport failure
    (and a broadcast re-run replays every target), so a second POST must
    collide on the PRIMARY KEY instead of writing the user a second copy of
    the same announcement. It is also what the retract deletes: the platform
    holds the dispatch id, never the message id, and must be able to name
    the row without asking for it back.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"admin-notice:{dispatch_id}:{user_id}"))


def build_admin_notice_payload(
    *,
    dispatch_id: str,
    mode: str,
    title: str,
    sender_name: str,
    sent_at: Optional[str] = None,
) -> dict[str, Any]:
    """The ``AdminNoticePayload`` both clients read.

    ONE construction site on purpose: this dict is persisted at
    ``metadata_json.admin_notice`` AND echoed verbatim in the
    ``admin_notice`` WS frame, so a client that hydrates from history and a
    client that took the live frame render the same card. Two builders
    would drift.

    The prose lives in ``Message.content`` and is deliberately not
    duplicated here.
    """
    return {
        "dispatch_id": dispatch_id,
        "mode": mode,
        "title": title,
        "sender_name": sender_name,
        "sent_at": sent_at or datetime.utcnow().isoformat(),
    }


async def write_admin_notice(
    db: AsyncSession,
    *,
    user_id: str,
    content: str,
    notice: Mapping[str, Any],
) -> Tuple[str, Optional[str]]:
    """Insert the notice Message into the day's admin Conversation, commit.

    Returns ``(message_id, day_chat_id)``; the second element is ALWAYS
    None and is kept so the shape matches the routine/subagent writers —
    the endpoint echoes it so the platform can see the NULL is intentional.

    Does NOT broadcast — the caller does that AFTER receiving the id so the
    frame can carry it. Broadcast is also explicitly isolated so a test
    environment without ws_chat doesn't error here.
    """
    from app.agent.conversation_resolver import resolve_or_create_day_conversation
    from app.db.message_helpers import resolve_day_chat_id_for_now
    from app.db.models import Message

    dispatch_id = str(notice.get("dispatch_id") or "").strip()
    if not dispatch_id:
        # Without it there is no idempotency key and no retract handle, so
        # this is a caller bug, not a degraded input to work around.
        raise ValueError("admin notice payload carries no dispatch_id")

    msg_id = admin_notice_message_id(dispatch_id, user_id)

    # Explicit dedupe check before anything is created, so a retry costs one
    # indexed PK read and leaves no debris — the IntegrityError branch below
    # then only ever covers a genuine concurrent double-POST.
    existing_id = (
        await db.execute(select(Message.id).where(Message.id == msg_id))
    ).scalar_one_or_none()
    if existing_id is not None:
        logger.info(
            "[admin_writer] dedupe hit dispatch=%s user=%s msg=%s — not rewriting",
            dispatch_id, user_id[:8], msg_id,
        )
        return existing_id, None

    # The CONVERSATION gets today's day chat; the MESSAGE below does not.
    # Reading A (Day-as-Chat) dedupes on (user, day_chat, channel), so
    # without a day here every dispatch would spawn its own thread and the
    # "admin" entry in SYSTEM_CHANNELS plus the partial unique index behind
    # it would buy nothing. This is the only reason a day is resolved at all
    # on this path.
    day_chat_id = await resolve_day_chat_id_for_now(db, user_id)
    conv = await resolve_or_create_day_conversation(
        db,
        user_id=user_id,
        day_chat_id=day_chat_id,
        channel="admin",
        title="Toup",
        metadata={"admin": True},
    )

    msg = Message(
        id=msg_id,
        conversation_id=conv.id,
        # NULL ON PURPOSE — do NOT "fix" this by calling
        # resolve_day_chat_id_for_now. `load_day_context` selects
        # WHERE Message.day_chat_id = :id, so this is what keeps the notice
        # out of the agent's context: the operator is talking to the user,
        # and an agent that reads the message starts answering for the
        # operator. The Conversation above carries the day instead.
        day_chat_id=None,
        role="assistant",
        content=content,
        channel="admin",
        source="admin_dispatch",
        # The party this is FROM, and the predicate `load_day_context` now
        # excludes on. Belt-and-braces with the NULL day above, on purpose:
        # that one is structural but cannot survive agent-initiated contact
        # (which needs a real day), and this one is the guard that will still
        # be standing afterwards. Neither is redundant with the other.
        origin=ORIGIN_ADMIN,
        metadata_json=json.dumps({"admin_notice": dict(notice)}),
    )
    try:
        # Real SAVEPOINT around the insert — a bare rollback here would
        # discard the Conversation resolved above too (same reasoning as
        # conversation_resolver's optimistic insert).
        async with db.begin_nested():
            db.add(msg)
            await db.flush()
    except IntegrityError:
        # Two workers POSTed the same target at once; the deterministic PK
        # is exactly what turned that into a collision instead of a second
        # copy. Read the winner back.
        winner_id = (
            await db.execute(select(Message.id).where(Message.id == msg_id))
        ).scalar_one_or_none()
        if winner_id is None:
            # The IntegrityError said a row exists — if it doesn't, the
            # constraint that fired was something else. Don't guess.
            raise
        await db.commit()
        logger.info(
            "[admin_writer] concurrent insert collision dispatch=%s user=%s — "
            "returning winner msg=%s",
            dispatch_id, user_id[:8], winner_id,
        )
        return winner_id, None

    await db.commit()
    logger.info(
        "[admin_writer] wrote dispatch=%s user=%s msg=%s conv=%s mode=%s",
        dispatch_id, user_id[:8], msg_id, conv.id, notice.get("mode"),
    )
    return msg_id, None


async def delete_admin_notice(
    db: AsyncSession, *, user_id: str, dispatch_id: str,
) -> int:
    """Delete the notice row for (dispatch, user), commit, return rowcount.

    Zero rows is SUCCESS, not an error: the retract is driven by a read
    receipt the platform may replay, and a `once` notice is meant to end up
    gone. The PK already encodes the user (see ``admin_notice_message_id``),
    so deleting by id alone cannot reach another tenant's row.
    """
    from app.db.models import Message

    msg_id = admin_notice_message_id(dispatch_id, user_id)
    result = await db.execute(delete(Message).where(Message.id == msg_id))
    await db.commit()
    deleted = int(result.rowcount or 0)
    logger.info(
        "[admin_writer] retract dispatch=%s user=%s msg=%s deleted=%d",
        dispatch_id, user_id[:8], msg_id, deleted,
    )
    return deleted


async def broadcast_admin_notice(
    user_id: str,
    *,
    message_id: str,
    created_at: str,
    content: str,
    notice: Mapping[str, Any],
) -> int:
    """Push the notice to any live WS clients. Returns the connection count.

    ``ws_count == 0`` is the NORMAL case — nobody has a socket open — and
    still counts as delivered: the row is in the DB for the next history
    fetch, and the push notification is the other half of the delivery.
    """
    try:
        from app.api.ws_chat import broadcast_to_user
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("[admin_writer] broadcast skipped — ws_chat unavailable: %s", e)
        return 0

    # NO "channel" key, and that is not an omission. Mobile accepts a frame
    # when `!channel || channel === 'app' || type === 'message'`, the web
    # when `!channel || channel === 'web'` — so a frame stamped
    # channel:'admin' is silently DROPPED by the browser. Absent is the one
    # value both filters pass.
    event = {
        "type": "admin_notice",
        "id": message_id,
        "created_at": created_at,
        "content": content,
        "notice": dict(notice),
    }
    try:
        return await broadcast_to_user(user_id, event)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("[admin_writer] broadcast failed: %s", e)
        return 0


async def broadcast_admin_notice_retract(user_id: str, *, dispatch_id: str) -> int:
    """Tell every live client to drop the card for this dispatch.

    Same no-`channel` rule as ``broadcast_admin_notice`` above — the retract
    has to reach the browser, which is the client that drops a stamped
    frame.
    """
    try:
        from app.api.ws_chat import broadcast_to_user
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("[admin_writer] retract broadcast skipped — ws_chat unavailable: %s", e)
        return 0

    try:
        return await broadcast_to_user(
            user_id, {"type": "admin_notice_retract", "dispatch_id": dispatch_id},
        )
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("[admin_writer] retract broadcast failed: %s", e)
        return 0
