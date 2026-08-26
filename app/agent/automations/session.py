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

**R31: this module no longer WRITES.** The paragraph that used to stand
here said the session's messages get a real `day_chat_id` "so the thread
is part of the agent's day context by design". That sentence is the
whole of F1. It meant the automation's run cards, memory chips,
auto-pause notices, pending cards and draft cards all rendered in the
user's main chat — and, because `load_day_context` selected by
`day_chat_id` with no channel predicate, the agent then read the
thread back as day context and answered it a second time there.

What survives here is the READ side and the row resolver: `GET /thread`
still serves the legacy `{session_id, messages}` keys off these rows
until B flips (CONTRACTS-R30 §9), and the clean-up migration needs the
resolver to find what it has to move. Every write is now a turn in
`automation_turns` (`ledger.append_turn`); the one sanctioned day-chat
row per run is the notification card in `run_v3._write_chat_card`.
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


class AutomationDayChatWrite(RuntimeError):
    """An automation tried to write a row into the day chat.

    CONTRACTS-R31 §4.1 allows exactly one: the notification card. Raised
    in dev/test so a resurrected writer fails at its first call;
    downgraded to an ERROR log in production, because losing a card is
    survivable and losing a run is not.
    """


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
    """RETIRED WRITER — refuses, and says so (CONTRACTS-R31 §4.1).

    This wrote a real `Message` with a real `day_chat_id` on
    `channel="automation"`. The module docstring above used to call that
    a feature ("part of the agent's day context by design"); the 26
    August recordings are what it costs. Five callers used it — the run
    card, the memory chip, the auto-pause notice, the pending card and
    the draft card — and every one of them put an automation's private
    conversation into the user's main chat, where the agent then read
    it back as context and answered it a second time.

    Each of those five now writes a TURN in the automation's thread.
    Nothing may write a day-chat row on an automation's behalf except
    `run_v3._write_chat_card` (the one notification card per run).

    Kept as a refusing stub rather than deleted so that a caller
    resurrected from an older branch fails LOUDLY in dev and merely
    logs in production — a lost card is survivable, a leaked thread is
    the defect this round exists to close. `test_thread_isolation_
    both_directions` drives it.
    """
    del content, role, metadata, title, commit
    msg = (
        "write_session_message is retired (CONTRACTS-R31 §4.1): an "
        f"automation may not write into the day chat "
        f"(automation={automation_id[:8]}, user={user_id[:8]}). "
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
    return None, None


async def _retired_write_session_message(
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
    """The pre-R31 body, kept only for the clean-up migration's tests to
    manufacture the rows it has to move. Never called in production."""
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
    """RETIRED as a writer (CONTRACTS-R31 §4.1; R31-02).

    This seam is still called from all six run-minting sites, and it
    still writes NOTHING. Until R31 it minted a `role="job"` marker
    message in the automation's session — a day-chat row — for every
    scheduled, push, poll, manual **and test** run. That is where the
    founder's main chat got `[test] Morning work brief — 5/7 steps —
    71%`, `[automation] Jira → Slack · Done` and `Morning work brief ·
    Done in 7 steps` on 26 August: a run wearing a chat job's clothes,
    with a sheet that read `Nothing recorded for this task yet.`
    because the real record was in the thread all along.

    A run is not a chat job. The one object a run may put in the main
    chat is the `automation_notification` card (`run_v3._write_chat_card`),
    which is minted once per run and updated in place.

    The function is kept rather than deleted so the six call sites stay
    honest about the seam they are on, and so a future card (of any
    kind) has one place to be added instead of six. `check-automation
    -cards.js` and `test_thread_isolation_both_directions` both assert
    that a run writes no day-chat row but its notification.
    """
    del db, job, automation  # retired — see docstring


async def emit_memory_update(
    db: AsyncSession,
    *,
    user_id: str,
    automation_id: str,
    count: int,
    title: Optional[str] = None,
) -> Optional[str]:
    """The "Memory updated · N facts" chip — now a THREAD turn.

    CONTRACTS-R31 §4.1: the chip belongs to the conversation that
    learned the facts. It used to be a day-chat Message, which is why
    `Memory updated · 5 facts` appeared in the founder's main chat at
    11:17 on 26 August, directly under a thread answer that had leaked
    there too — two rows, one cause.

    The `automation_memory_update` frame is unchanged and still goes
    out: it is an automation frame (carries `automation_id`, no
    `channel` key), not a chat frame, and B's bridge routes it to the
    thread. `message_id` is gone from it — there is no message any
    more; `turn_id` takes its place, which is also what a deep link
    now needs.

    Best-effort, as it always was: the facts are already committed by
    the write seam before this is called, and a chip that fails to
    render must never look like a fact that failed to save.
    """
    del title  # the thread already knows whose it is
    if count <= 0:
        return None
    from . import ledger as _ledger

    turn_id: Optional[str] = None
    try:
        thread = await _ledger.ensure_thread(
            db, user_id=user_id, automation_id=automation_id,
        )
        turn = await _ledger.append_turn(
            db, user_id=user_id, thread=thread, run_id=None,
            kind="memory",
            payload={"count": int(count), "sheet": "memory"},
        )
        turn_id = turn.get("id") if isinstance(turn, dict) else None
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.warning(
            "[automations] memory chip turn failed automation=%s: %s",
            automation_id[:8], e,
        )
        return None
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, {
            "type": "automation_memory_update",
            "automation_id": automation_id,
            "count": count,
            "turn_id": turn_id,
        })
    except Exception as e:  # noqa: BLE001 — no live socket is normal
        logger.debug("[automations] memory chip broadcast skipped: %s", e)
    return turn_id
