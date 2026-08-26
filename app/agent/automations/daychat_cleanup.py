"""Move the leak out of the day chat — CONTRACTS-R31 §4.1.

R31 stops the WRITERS. This moves what they already wrote.

The founder's 26 August day chat holds an automation's whole
conversation: run job cards frozen at 71%, `[automation] Jira → Slack ·
Done` whose sheet says `Nothing recorded for this task yet.`, a thread
answer about "everything in all channels", `Memory updated · 5 facts`,
and the turns of a thread that should never have been visible there.
Every one of those rows renders on every reload, so a user who opens
26 August after the fix still sees the defect.

**Identified by PRODUCER, never by title.** The dispatch is explicit
about that and it is not a style preference: a title match would catch
a user's own message that happens to name their automation, and miss
every row whose title the agent phrased differently. The producer is a
real column — `Message.source` — written at row creation:

    source="automation"  the engine and the R28 session writers
    source="reminder" / "email_briefing" / "agent_task"  routines
    source IS NULL       ordinary chat turns, INCLUDING skill output

That last line is the limit of what this can do, and it is why the
skill's own chat-job rows are handled by their JOB, not their message:
a skill runs inside an ordinary turn and sets no source, so
`source IS NULL` covers both "the automations skill produced this" and
"the agent was talking". The `build_jobs` row is what separates them.

Three passes, and each says what it did:

  1. **MOVE** — a `channel="automation"` message becomes a legacy turn
     in that automation's thread, keeping its `created_at` so the
     thread's order is the order things happened.
  2. **DELETE** — a `role="job"` marker whose job is an
     `automation_run`. Its replacement is the notification card, which
     already exists for those runs; keeping both would show the same
     run twice with different numbers.
  3. **KEEP** — the notification card itself (`channel="routine"`,
     metadata `automation_notification`). §2.1 says the main chat gets
     exactly one object per run, and this is it. A filter written from
     §4.1's paragraph alone would delete it, because that paragraph
     talks about `automation_id` and the card is the one row that
     legitimately carries one.

Idempotent: a moved message is deleted in the same transaction, so a
second run finds nothing. Reports counts per user so D can diff what it
missed rather than guessing.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation, BuildJob, Conversation, Message

logger = logging.getLogger(__name__)

# The R28 session channel. Its rows ARE the thread, written into the day.
SESSION_CHANNEL = "automation"
# Where the one sanctioned card lives. Never touched.
CARD_CHANNEL = "routine"
NOTIFICATION_KEY = "automation_notification"


def _metadata(msg: Message) -> dict:
    try:
        raw = msg.metadata_json
        if not raw:
            return {}
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _turn_for(msg: Message) -> Optional[tuple[str, dict]]:
    """`(kind, payload)` for one legacy day-chat row, or None to delete.

    Deliberately narrow. A row whose shape we cannot render as a turn is
    DROPPED rather than forced into an `agent` turn: a half-understood
    card rendered as prose is worse in the thread than absent, because
    the thread is where the user goes to find out what really happened.
    """
    meta = _metadata(msg)
    role = (msg.role or "").lower()
    content = (msg.content or "").strip()

    if meta.get("memory_update"):
        count = int((meta["memory_update"] or {}).get("count") or 0)
        if count > 0:
            return "memory", {"count": count, "sheet": "memory"}
        return None
    if role == "job":
        return None                    # pass 2 owns these
    if role == "user" and content:
        return "user", {"text": content[:4000]}
    if content:
        return "agent", {"text": content[:4000]}
    return None


async def cleanup_day_chat(
    db: AsyncSession, *, user_id: str, dry_run: bool = False,
) -> dict:
    """Run the three passes for one user. Returns `{moved, deleted,
    skipped, details}`."""
    from . import ledger

    moved = deleted = skipped = 0
    details: list[dict] = []

    # ── pass 1: the R28 session rows ─────────────────────────────────
    convs = list((await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .where(Conversation.channel == SESSION_CHANNEL)
    )).scalars())
    for conv in convs:
        meta = {}
        try:
            meta = json.loads(conv.metadata_json or "{}")
        except (ValueError, TypeError):
            meta = {}
        automation_id = meta.get("automation_id")
        if not automation_id:
            skipped += 1
            continue
        automation = await db.get(Automation, automation_id)
        if automation is None:
            # The automation is gone; so should its leaked rows be.
            rows = list((await db.execute(
                select(Message).where(Message.conversation_id == conv.id)
            )).scalars())
            for m in rows:
                if not dry_run:
                    await db.delete(m)
                deleted += 1
            continue

        thread = await ledger.ensure_thread(
            db, user_id=user_id, automation_id=automation_id,
        )
        rows = list((await db.execute(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.created_at.asc())
        )).scalars())
        for m in rows:
            if (m.role or "").lower() == "job":
                # Pass 2 owns these, and only pass 2 may count them.
                # Counting here as well double-reported the same row on
                # a dry run and, on a live one, queued a second delete
                # for a row pass 1 had already removed.
                continue
            shape = _turn_for(m)
            if shape is None:
                if not dry_run:
                    await db.delete(m)
                deleted += 1
                continue
            kind, payload = shape
            if not dry_run:
                try:
                    await ledger.append_turn(
                        db, user_id=user_id, thread=thread, run_id=None,
                        kind=kind, payload=payload, broadcast=False,
                    )
                except Exception as e:  # noqa: BLE001
                    # A row we cannot render is not worth failing the
                    # whole migration for — it is left where it is and
                    # reported, so D can look at it.
                    logger.warning(
                        "[daychat_cleanup] turn skipped msg=%s: %s",
                        m.id[:8], e,
                    )
                    skipped += 1
                    details.append({"message_id": m.id, "reason": str(e)[:120]})
                    continue
                await db.delete(m)
            moved += 1

    # ── pass 2: run job cards, keyed on the JOB ──────────────────────
    #
    # `role="job"` markers carry a job id in their content. The job row
    # is what says whether it is an automation run — the title is not,
    # and `[automation] Jira → Slack` is exactly the kind of title a
    # user could also write.
    job_rows = list((await db.execute(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
        .where(Message.role == "job")
    )).scalars())
    for m in job_rows:
        job_id = _job_id_of(m)
        if not job_id:
            continue
        job = await db.get(BuildJob, job_id)
        if job is None or job.job_type != "automation_run":
            continue
        if not dry_run:
            await db.delete(m)
        deleted += 1

    if not dry_run:
        await db.commit()
    out = {"moved": moved, "deleted": deleted, "skipped": skipped,
           "dry_run": dry_run}
    if details:
        out["details"] = details[:50]
    logger.info("[daychat_cleanup] user=%s %s", user_id[:8], out)
    return out


def _job_id_of(msg: Message) -> Optional[str]:
    try:
        from app.api.message_cards import parse_job_marker
        marker = parse_job_marker(msg.content or "")
        return (marker or {}).get("job_id")
    except Exception:  # noqa: BLE001
        return None
