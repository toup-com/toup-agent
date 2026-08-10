"""Turn counters increment atomically — no ORM read-modify-write.

G-21's investigation verified every durable cross-turn write path is locked
(pg advisory lock on memory writes, FOR UPDATE on balances, upsert on
day-row creation) EXCEPT the per-turn counter updates in
`AgentRunner._save_messages`: Conversation and DayChat `message_count` /
`total_tokens` were plain `x = (x or 0) + n` on a loaded ORM object, so two
concurrent turns for one user (a voice think beside a chat turn, a second
device) each held a stale value and the last writer erased the other's
increment. The fix is an atomic `UPDATE ... SET c = COALESCE(c,0) + :n`.

The source pin fails on the pre-fix code. The behavioural test below drives
the REAL `AgentRunner._save_messages` twice — it used to build its own
UPDATE and assert that SQLAlchemy arithmetic works, which passed happily on
the buggy code and even survived pointing the UPDATE at a row that cannot
exist. A test that never calls the function it is named after is not a
detector; an adversarial review proved that one vacuous by mutation.
"""

import re
from pathlib import Path

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

RUNNER_SRC = Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_runner.py"


def test_no_read_modify_write_on_turn_counters():
    src = RUNNER_SRC.read_text()
    rmw = re.findall(
        r"\.(message_count|total_tokens)\s*=\s*\(\s*\w+\.(?:message_count|total_tokens)\s*or\s*0\s*\)\s*\+",
        src,
    )
    assert not rmw, (
        f"read-modify-write counter update(s) back in agent_runner.py: {rmw} — "
        "concurrent turns lose increments; use the atomic "
        "UPDATE ... SET c = COALESCE(c,0) + n form"
    )
    assert "coalesce(_Conv.message_count, 0)" in src, (
        "the Conversation counter no longer uses the atomic coalesce increment"
    )
    assert "coalesce(DayChat.message_count, 0)" in src, (
        "the DayChat counter no longer uses the atomic coalesce increment"
    )


async def test_save_messages_does_not_erase_a_concurrent_increment(tmp_path):
    """Drive the real `_save_messages` against a STALE identity-mapped row.

    This is the interleaving two concurrent turns produce, made
    deterministic. Session A loads the Conversation (count=0). Session B
    commits a turn, so the row is now 2. Session A then runs
    `_save_messages` — and because SQLAlchemy's identity map returns the
    instance A already loaded, `session.message_count` is still the stale
    0. The pre-fix `x = (x or 0) + n` therefore writes 2, erasing B's turn.
    The atomic `UPDATE ... SET c = COALESCE(c,0) + n` computes against the
    row's CURRENT value and lands 4.

    The previous version of this test opened a fresh session per turn, so
    nothing was ever stale and it passed on the buggy code — an adversarial
    review proved it vacuous by reverting the fix and by pointing the
    UPDATE at a non-existent row. Both mutations now fail here.
    """
    from unittest.mock import AsyncMock
    from app.agent.agent_runner import AgentRunner
    from app.db.models import Conversation, Message, User

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    async with engine.begin() as conn:
        for table in (User.__table__, Conversation.__table__, Message.__table__):
            await conn.run_sync(lambda sc, t=table: t.create(sc, checkfirst=True))
    maker = async_sessionmaker(engine, expire_on_commit=False)

    user_id = "u-counters"
    async with maker() as db:
        db.add(User(id=user_id, email="c@t.local", hashed_password="x", name="C"))
        conv = Conversation(user_id=user_id, title="t", message_count=0, total_tokens=0)
        db.add(conv)
        await db.commit()
        conv_id = conv.id

    tools = AsyncMock()
    tools._last_media = None
    tools._last_pending_action = None
    runner = AgentRunner(llm_service=AsyncMock(), tool_executor=tools)

    async def save(db):
        await runner._save_messages(
            db=db, session_id=conv_id, user_id=user_id,
            user_message="hi", assistant_response="hello",
            tokens_input=100, tokens_output=20,
            model="gpt-4o-mini", processing_time_ms=5,
        )
        await db.commit()

    async with maker() as session_a:
        # A loads the row FIRST and holds it in its identity map.
        stale = (await session_a.execute(
            select(Conversation).where(Conversation.id == conv_id)
        )).scalar_one()
        assert (stale.message_count or 0) == 0

        # B commits a whole turn underneath A.
        async with maker() as session_b:
            await save(session_b)

        # A now writes. Its re-query returns the STALE instance above.
        await save(session_a)

    async with maker() as db:
        row = (await db.execute(
            select(Conversation).where(Conversation.id == conv_id)
        )).scalar_one()

    assert row.message_count == 4, (
        f"expected 4 messages from two turns, got {row.message_count} — a "
        "concurrent turn's increment was erased (last-writer-wins)"
    )
    assert row.total_tokens == 240, (
        f"expected 240 tokens from two turns, got {row.total_tokens}"
    )

    await engine.dispose()
