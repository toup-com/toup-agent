"""Per-automation session conversations (Round 28).

The automation "session" is the automation's own chat thread. In every
client-facing contract the session id IS the automation id; physically
the thread rides Day-as-Chat like everything else — one `Conversation`
per (user, local day, automation), `channel="automation"`, keyed by
`metadata_json {"automation_id": …}`.

This is the app-channel pattern (agent_runner's DayChat `FOR UPDATE`
lock + metadata scan), NOT the partial-unique-index pattern: N
automations share one day, and `ix_conversations_system_channel_per_day`
is `(user_id, day_chat_id, channel)` — it knows nothing about metadata,
so admitting "automation" to `INDEXED_SYSTEM_CHANNELS` would make the
second automation of the day an IntegrityError. Keep it out (the
`subagent` precedent in conversation_resolver).

Everything automation-shaped lands in the session: connector/grant
cards once the automation exists, run cards (`role="job"` markers),
and the auto-pause notice. Messages get a real `day_chat_id`, so the
thread is part of the agent's day context by design — the agent should
know what its automations did today.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

SESSION_CHANNEL = "automation"


def automation_id_of(conversation) -> Optional[str]:
    """The automation this conversation belongs to, or None. The one
    parse for `metadata_json {"automation_id": …}` — the runner's R28
    branch, both scans below, and the R29 turn path all read it."""
    try:
        meta = json.loads(conversation.metadata_json or "{}")
    except (ValueError, TypeError):
        return None
    value = meta.get("automation_id")
    return value if isinstance(value, str) and value else None


async def resolve_session_conversation(
    db: AsyncSession,
    *,
    user_id: str,
    automation_id: str,
    title: Optional[str] = None,
    tz_override: Optional[str] = None,
):
    """Return `(conversation, day_chat_id)` for this automation's thread
    today, creating the row if the automation hasn't spoken today.

    The DayChat row is locked (`FOR UPDATE`; a no-op on sqlite, which
    single-writes anyway) so two concurrent fires of the same automation
    converge on one row instead of racing to two.
    """
    from app.db.message_helpers import resolve_day_chat_id_for_now
    from app.db.models import Conversation
    from app.db.models.day_chat import DayChat

    day_chat_id = await resolve_day_chat_id_for_now(
        db, user_id, tz_override=tz_override,
    )
    if day_chat_id:
        await db.execute(
            select(DayChat).where(DayChat.id == day_chat_id).with_for_update()
        )
        candidates = (
            await db.execute(
                select(Conversation).where(
                    Conversation.user_id == user_id,
                    Conversation.day_chat_id == day_chat_id,
                    Conversation.channel == SESSION_CHANNEL,
                    Conversation.is_active.is_(True),
                )
            )
        ).scalars().all()
        for conv in candidates:
            if automation_id_of(conv) == automation_id:
                return conv, day_chat_id
    else:
        # Degraded path — same policy as conversation_resolver: create
        # unconstrained and say so, rather than dropping the write.
        logger.warning(
            "[automations] session resolve with no day_chat_id user=%s "
            "automation=%s — creating unconstrained row",
            user_id[:8], automation_id[:8],
        )

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_id,
        channel=SESSION_CHANNEL,
        is_active=True,
        day_chat_id=day_chat_id,
        title=title,
        metadata_json=json.dumps({"automation_id": automation_id}),
    )
    db.add(conv)
    await db.flush()
    return conv, day_chat_id


async def list_session_conversation_ids(
    db: AsyncSession,
    *,
    user_id: str,
    automation_id: str,
    max_days: int = 30,
) -> list[str]:
    """All of this automation's conversation ids, newest day first.

    The key lives inside `metadata_json`, so this is a bounded LIKE
    pre-filter + exact JSON check — the `_grant_decided` precedent, but
    scoped to the automation channel.
    """
    from app.db.models import Conversation

    rows = (
        await db.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.channel == SESSION_CHANNEL,
                Conversation.metadata_json.contains(automation_id),
            )
            .order_by(Conversation.started_at.desc())
            .limit(max_days)
        )
    ).scalars().all()
    out = []
    for conv in rows:
        if automation_id_of(conv) == automation_id:
            out.append(conv.id)
    return out


async def write_session_message(
    db: AsyncSession,
    *,
    user_id: str,
    automation_id: str,
    content: str,
    role: str = "assistant",
    metadata: Optional[dict] = None,
    title: Optional[str] = None,
    commit: bool = True,
) -> tuple[Optional[str], Optional[str]]:
    """Persist one message into the automation's session thread.

    Returns `(message_id, day_chat_id)`, or `(None, None)` if the write
    failed — session writes are companions to engine state, never a veto
    on it (the R27 settle lesson), so callers treat None as "no card,
    carry on".
    """
    from app.db.models import Message
    from app.db.models.day_chat import DayChat

    try:
        conv, day_chat_id = await resolve_session_conversation(
            db, user_id=user_id, automation_id=automation_id, title=title,
        )
        msg_id = str(uuid.uuid4())
        db.add(Message(
            id=msg_id,
            conversation_id=conv.id,
            day_chat_id=day_chat_id,
            role=role,
            content=content,
            channel=SESSION_CHANNEL,
            source="automation",
            metadata_json=json.dumps(metadata, default=str) if metadata else None,
        ))
        conv.message_count = (conv.message_count or 0) + 1
        await db.flush()
        if day_chat_id:
            dc = await db.get(DayChat, day_chat_id)
            if dc:
                dc.message_count = (dc.message_count or 0) + 1
                dc.last_message_at = datetime.utcnow()
        if commit:
            await db.commit()
        return msg_id, day_chat_id
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.warning(
            "[automations] session write failed user=%s automation=%s: %s",
            user_id[:8], automation_id[:8], e,
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None, None


async def on_run_created(db: AsyncSession, *, job, automation) -> None:
    """The one seam every run-minting site calls (both executors, all
    six sites): persist the run's card in the automation's session and
    back-link it from the job (`summary_message_id` + `conversation_id`)
    so web deep links (`/chat/<date>#m=<id>`) and push data can address
    it. Best-effort end to end — the run never depends on its card.

    The card write uses its OWN session (R27 settle lesson); only the
    two stamp columns ride the caller's `db`, in a short transaction.
    """
    from app.db.models import BuildJob, Message

    if getattr(job, "summary_message_id", None):
        return  # idempotency replay — this run's card already exists
    msg_id, _day = await write_run_card(
        user_id=automation.user_id,
        automation_id=automation.id,
        automation_name=automation.name,
        job_id=job.id,
    )
    if not msg_id:
        return
    try:
        msg = await db.get(Message, msg_id)
        row = await db.get(BuildJob, job.id)
        if row is not None:
            row.summary_message_id = msg_id
            if msg is not None and msg.conversation_id:
                row.conversation_id = msg.conversation_id
            await db.commit()
    except Exception as e:  # noqa: BLE001 — stamps are conveniences
        logger.warning(
            "[automations] run card stamp failed job=%s: %s", job.id[:8], e,
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def write_run_card(
    *,
    user_id: str,
    automation_id: str,
    automation_name: str,
    job_id: str,
) -> tuple[Optional[str], Optional[str]]:
    """Persist the run's card — a `role="job"` marker in the session —
    in its OWN session, best-effort. Call AFTER the run row is
    committed; a failure here never touches the run.

    Returns `(message_id, day_chat_id)`; the caller stamps
    `summary_message_id` / config extras onto the job itself.
    """
    from app.api.message_cards import job_marker_content
    from app.db.database import async_session_maker

    try:
        async with async_session_maker() as db:
            return await write_session_message(
                db,
                user_id=user_id,
                automation_id=automation_id,
                role="job",
                content=job_marker_content(
                    job_id, automation_name, "automation_run",
                ),
                title=automation_name,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[automations] run card write failed job=%s: %s", job_id[:8], e,
        )
        return None, None


async def emit_memory_update(
    db: AsyncSession,
    *,
    user_id: str,
    automation_id: str,
    count: int,
    title: Optional[str] = None,
) -> Optional[str]:
    """The "Memory updated · N facts" chip (CONTRACTS-R29 §4): one
    session marker message carrying `memory_update` in its metadata,
    plus the live `automation_memory_update` frame — NO channel key
    (the app's frame filter drops channeled frames). Best-effort like
    every session write; the facts themselves are already committed by
    the write seam before this is called."""
    if count <= 0:
        return None
    at = datetime.utcnow().isoformat() + "Z"
    noun = "fact" if count == 1 else "facts"
    msg_id, _day = await write_session_message(
        db,
        user_id=user_id,
        automation_id=automation_id,
        content=f"Memory updated · {count} {noun}",
        metadata={"memory_update": {"count": count, "at": at}},
        title=title,
    )
    if msg_id:
        try:
            from app.api.ws_chat import broadcast_to_user
            await broadcast_to_user(user_id, {
                "type": "automation_memory_update",
                "automation_id": automation_id,
                "count": count,
                "message_id": msg_id,
            })
        except Exception as e:  # noqa: BLE001 — no live socket is normal
            logger.debug("[automations] memory chip broadcast skipped: %s", e)
    return msg_id
