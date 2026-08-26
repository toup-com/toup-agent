# agent-mode: automation_threads/_turns are AGENT_ONLY tables.
"""R31 §4.1 — an automation's conversation is its own, in both directions.

The 26 August recordings show one conversation where there should be
two. `Run all of them again` is in the thread AND in the main chat at
10:15 and 10:29; the thread's 11:17 answer about "everything in all
channels" is duplicated there with `Memory updated · 5 facts` under it;
run cards wear chat-job clothes and freeze at 71%.

The contract names three tests. They are here, plus the two guards that
make the first one hard to un-fix.
"""

import json
import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationTurn, BuildJob, Conversation, Message, User,
)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Isolation"))
        await db.commit()
    return uid


async def _mk_automation(uid: str, name: str) -> str:
    aid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Automation(
            id=aid, user_id=uid, name=name, status="armed",
            spec_json=json.dumps({"version": 2, "name": name,
                                  "mode": "auto",
                                  "trigger": {"sources": []},
                                  "steps": []}),
            trigger_mode="schedule",
        ))
        await db.commit()
    return aid


async def _day_rows(uid: str) -> list[Message]:
    async with async_session_maker() as db:
        return list((await db.execute(
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == uid)
        )).scalars())


# ── the storage invariant ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_isolation_both_directions():
    """A thread turn writes no day-chat row, and the reverse.

    CONTRACTS-R30 §4.10 claimed this was "pinned by guard tests" and no
    such test existed — while `test_automation_sessions.py` asserted the
    OPPOSITE ("in the agent's day"). That is how a documented invariant
    survived a whole round with a live writer against it.
    """
    from app.agent.automations import ledger

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Morning work brief")

    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(
            db, user_id=uid, automation_id=aid,
        )
        for kind, payload in (
            ("user", {"text": "Run all of them again"}),
            ("agent", {"text": "I ran it. Two accounts need you."}),
            ("memory", {"count": 5, "sheet": "memory"}),
            ("note", {"stamp": "ran", "at": datetime.utcnow().isoformat()}),
        ):
            await ledger.append_turn(
                db, user_id=uid, thread=thread, kind=kind, payload=payload,
            )

    # Direction 1: the thread wrote nothing into the day.
    assert await _day_rows(uid) == [], (
        "a thread turn wrote a day-chat row"
    )

    # Direction 2: a day-chat message writes nothing into the thread.
    async with async_session_maker() as db:
        conv = Conversation(id=str(uuid.uuid4()), user_id=uid,
                            channel="web", is_active=True)
        db.add(conv)
        await db.flush()
        db.add(Message(id=str(uuid.uuid4()), conversation_id=conv.id,
                       role="user", content="what did the work brief find?"))
        await db.commit()

    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, aid)
        turns = list((await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread.id)
        )).scalars())
    assert len(turns) == 4, (
        "a day-chat message reached the thread"
    )


@pytest.mark.asyncio
async def test_thread_isolation_between_automations():
    """One automation's thread never reads or writes another's.

    "Automations are isolated from each other: a thread never reads
    another thread" (§2.1). The two threads are separate rows by
    construction — this drives it rather than assuming the schema.
    """
    from app.agent.automations import ledger

    uid = await _mk_user()
    a1 = await _mk_automation(uid, "Morning work brief")
    a2 = await _mk_automation(uid, "Jira to Slack")

    async with async_session_maker() as db:
        t1 = await ledger.ensure_thread(db, user_id=uid, automation_id=a1)
        t2 = await ledger.ensure_thread(db, user_id=uid, automation_id=a2)
        assert t1.id != t2.id
        await ledger.append_turn(
            db, user_id=uid, thread=t1, kind="agent",
            payload={"text": "Only the work brief knows this."},
        )

    async with async_session_maker() as db:
        turns2, _ = await ledger.list_turns(db, thread_id=t2.id)
    assert turns2 == [], "one automation's turn appeared in another's thread"


@pytest.mark.asyncio
async def test_the_day_chat_writer_is_refused():
    """The R28 writer is retired, and says so.

    Kept as a refusing stub rather than deleted so a caller resurrected
    from an older branch fails LOUDLY in dev — a lost card is
    survivable, a leaked thread is the defect this round exists to
    close.
    """
    from app.agent.automations.session import (
        AutomationDayChatWrite, write_session_message,
    )

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Refused")
    async with async_session_maker() as db:
        with pytest.raises(AutomationDayChatWrite):
            await write_session_message(
                db, user_id=uid, automation_id=aid, content="a notice",
            )
    assert await _day_rows(uid) == []


@pytest.mark.asyncio
async def test_every_day_reader_hides_the_automation_channel():
    """Fixing one reader leaves the other doors open.

    Four readers select day-scoped messages, and each had its own copy
    of a channel predicate. They share one tuple now; this asserts the
    tuple covers the automation channel AND still lets the notification
    card's channel through, which is the one row §2.1 requires to STAY.
    """
    from app.db.models.conversation import HIDDEN_DAY_CHANNELS

    assert "automation" in HIDDEN_DAY_CHANNELS
    assert "autopilot" in HIDDEN_DAY_CHANNELS
    # The notification card is written on `routine`. A filter derived
    # from §4.1's paragraph alone would have hidden it, because that
    # paragraph talks about `automation_id` and the card is the one row
    # that legitimately carries one.
    assert "routine" not in HIDDEN_DAY_CHANNELS


@pytest.mark.asyncio
async def test_the_day_context_loader_excludes_the_thread():
    """The "and back" direction, at the reader that caused it.

    `load_day_context` selected by `day_chat_id` with no channel
    predicate, so the R28 session rows — which carry a real
    `day_chat_id` — entered the agent's main-chat context. That is how
    a thread answer was re-asked and re-answered in the main chat.
    """
    import inspect
    from app.agent import day_context_loader

    src = inspect.getsource(day_context_loader.load_day_context)
    assert "HIDDEN_DAY_CHANNELS" in src, (
        "the day context loader lost its channel predicate"
    )


# ── the clean-up migration ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleanup_moves_the_thread_and_deletes_the_run_cards():
    """R31's migration, on rows shaped like the founder's.

    Stopping the writers does not repair a day that is already written:
    every leaked row renders on every reload, so 26 August would still
    show the defect after the fix.
    """
    from app.agent.automations import ledger
    from app.agent.automations.daychat_cleanup import cleanup_day_chat
    from app.agent.automations.session import (
        _retired_write_session_message as legacy_write,
    )
    from app.api.message_cards import job_marker_content

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Morning work brief")

    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id, user_id=uid, title="Morning work brief",
            prompt="(automation)", job_type="automation_run",
            status="completed", outcome="sent",
            source_kind="automation", source_id=aid,
        ))
        await db.commit()

    async with async_session_maker() as db:
        await legacy_write(db, user_id=uid, automation_id=aid,
                           role="user", content="Run all of them again")
        await legacy_write(db, user_id=uid, automation_id=aid,
                           content="I ran the work brief again.")
        await legacy_write(db, user_id=uid, automation_id=aid,
                           content="Memory updated · 5 facts",
                           metadata={"memory_update": {"count": 5}})
        await legacy_write(
            db, user_id=uid, automation_id=aid, role="job",
            content=job_marker_content(job_id, "Morning work brief",
                                       "automation_run"),
        )
    assert len(await _day_rows(uid)) == 4

    # Dry run touches nothing — a migration that MOVES history should
    # have to be asked twice.
    async with async_session_maker() as db:
        plan = await cleanup_day_chat(db, user_id=uid, dry_run=True)
    assert plan["moved"] == 3 and plan["deleted"] == 1
    assert len(await _day_rows(uid)) == 4

    async with async_session_maker() as db:
        out = await cleanup_day_chat(db, user_id=uid, dry_run=False)
    assert out["moved"] == 3
    assert out["deleted"] == 1
    assert await _day_rows(uid) == [], "the day chat still holds the thread"

    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, aid)
        turns, _ = await ledger.list_turns(db, thread_id=thread.id)
    kinds = [t["kind"] for t in turns]
    assert "user" in kinds and "agent" in kinds and "memory" in kinds
    assert "job" not in kinds

    # Idempotent: a second run finds nothing left.
    async with async_session_maker() as db:
        again = await cleanup_day_chat(db, user_id=uid, dry_run=False)
    assert again["moved"] == 0 and again["deleted"] == 0


@pytest.mark.asyncio
async def test_cleanup_keeps_the_notification_card():
    """§2.1: the main chat gets exactly ONE object per run, and this is
    it. A migration that deleted it would take the automation out of the
    main chat entirely — the opposite mistake, and one nobody would
    notice until they wondered why runs stopped announcing themselves."""
    from app.agent.automations.cards import write_card_message
    from app.agent.automations.daychat_cleanup import cleanup_day_chat

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Kept")
    async with async_session_maker() as db:
        msg_id, _day = await write_card_message(
            db, user_id=uid, content="",
            metadata_key="automation_notification",
            payload={"automation_id": aid, "status": "completed"},
            title="Kept",
        )
    assert msg_id

    async with async_session_maker() as db:
        out = await cleanup_day_chat(db, user_id=uid, dry_run=False)
    assert out["deleted"] == 0
    rows = await _day_rows(uid)
    assert len(rows) == 1
    assert rows[0].id == msg_id


@pytest.mark.asyncio
async def test_a_day_chat_write_outside_the_sanctioned_keys_is_refused():
    """§4.1's storage invariant, at the one remaining writer.

    R31-D searched the founder's day and found the shape that makes
    this necessary: eight rows whose CONTENT is a thread turn — both
    `Run all of them again` pairs and the whole 15:16/15:17 Q&A — with
    `metadata_json` empty on every one. No `automation_id`, no
    `thread_id`, nothing. Their only marker is `channel='automation'`.

    So a tag-based filter cannot catch this shape, on either side: the
    migration finds them by channel (the only thing they have), and
    B's `mapDayMsg` clause is specified against metadata these rows do
    not carry. The net has to be at the WRITE path, where a missing tag
    is loud once instead of silent in every consumer downstream.
    """
    from app.agent.automations.cards import (
        AutomationDayChatWrite, DAY_CHAT_METADATA_KEYS,
        assert_day_chat_clean, write_card_message,
    )

    # The three sanctioned keys pass.
    for key in DAY_CHAT_METADATA_KEYS:
        assert_day_chat_clean(key)

    # Anything an automation might narrate does not.
    for key in ("memory_update", "draft_card", "pending_action",
                "fix_chip", "automation_turn"):
        with pytest.raises(AutomationDayChatWrite):
            assert_day_chat_clean(key)

    # And it is wired at the writer, not only exported.
    uid = await _mk_user()
    aid = await _mk_automation(uid, "Guarded")
    async with async_session_maker() as db:
        with pytest.raises(AutomationDayChatWrite):
            await write_card_message(
                db, user_id=uid, content="Memory updated · 2 facts",
                metadata_key="memory_update",
                payload={"count": 2}, title="Guarded",
            )
    assert await _day_rows(uid) == []
    del aid


# ── R31-18: the two back-fills ───────────────────────────────────────


@pytest.mark.asyncio
async def test_rules_extracted_and_backfilled():
    """`LINES IT WILL NOT CROSS 0` on an automation that states three.

    `rules_json` had three writers and all three were the user typing
    into the Rules sheet. Nothing ever read a constraint out of the
    description the automation was created from, or out of the setup
    conversation where the user said it — so the founder's Morning work
    brief, whose own description says "post ONE line, no thread", opened
    its Workflow with an empty Rules tab.
    """
    from app.agent.automations import workflow
    from app.agent.automations.backfill_r31 import backfill_rules

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Morning work brief")
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        a.description = ("Read unread Gmail and post one line in "
                         "#all-toup — no thread, and only unread.")
        await db.commit()

    async with async_session_maker() as db:
        plan = await backfill_rules(db, user_id=uid, dry_run=True)
    assert plan["scanned"] == 1
    assert plan["added"] >= 1, plan
    # A dry run writes nothing.
    async with async_session_maker() as db:
        assert workflow.rules_list(await db.get(Automation, aid)) == []

    async with async_session_maker() as db:
        out = await backfill_rules(db, user_id=uid, dry_run=False)
    assert out["added"] == plan["added"]

    async with async_session_maker() as db:
        rules = workflow.rules_list(await db.get(Automation, aid))
    assert rules, "the back-fill recorded nothing"
    assert all(r.get("text") for r in rules)

    # Idempotent — running it twice adds nothing the second time, which
    # is what makes it safe to re-run over a tenant whose first pass
    # half-finished.
    async with async_session_maker() as db:
        again = await backfill_rules(db, user_id=uid, dry_run=False)
    assert again["added"] == 0
    async with async_session_maker() as db:
        assert len(workflow.rules_list(
            await db.get(Automation, aid))) == len(rules)


@pytest.mark.asyncio
async def test_thread_facts_rescoped():
    """Five memory groups at `0 things` on an automation that has run
    for days.

    `curator_v2` files a thread-learned fact as `global` unless the
    extraction model literally emits the word "automation", while the
    Memory tab reads `scope == automation_id` exactly. So the facts an
    automation learned in its own thread were invisible in its own
    Memory tab.

    The move is one-directional on purpose: `global → automation_id`
    only, and only when the fact NAMES the automation. C's ambiguity
    note is why — deleting an automation hard-deletes
    `MemoryFact WHERE scope == automation_id`, so a fact wrongly
    narrowed is destroyed rather than misfiled.
    """
    from app.db.models import MemoryFact
    from app.agent.automations.backfill_r31 import rescope_thread_facts

    uid = await _mk_user()
    aid = await _mk_automation(uid, "Morning work brief")

    async with async_session_maker() as db:
        db.add(MemoryFact(
            user_id=uid, domain="work", category="work_you_own",
            scope="global", source="agent",
            text="The Morning work brief should skip Dependabot noise.",
            why="Said in setup",
        ))
        db.add(MemoryFact(
            user_id=uid, domain="work", category="people",
            scope="global", source="agent",
            text="Marcus gets same-day answers.",
            why="Said in setup",
        ))
        await db.commit()

    async with async_session_maker() as db:
        out = await rescope_thread_facts(db, user_id=uid, dry_run=False)
    assert out["moved"] == 1, out

    async with async_session_maker() as db:
        rows = list((await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars())
    by_scope = {r.scope for r in rows}
    assert aid in by_scope, "the automation's own fact was not scoped to it"
    assert "global" in by_scope, (
        "a fact about the PERSON was narrowed — it would be destroyed "
        "the day they delete this automation"
    )
    named = next(r for r in rows if r.scope == aid)
    assert "Morning work brief" in named.text
