"""Turn counters increment atomically — no ORM read-modify-write.

G-21's investigation verified every durable cross-turn write path is locked
(pg advisory lock on memory writes, FOR UPDATE on balances, upsert on
day-row creation) EXCEPT the per-turn counter updates in
`AgentRunner._save_messages`: Conversation and DayChat `message_count` /
`total_tokens` were plain `x = (x or 0) + n` on a loaded ORM object, so two
concurrent turns for one user (a voice think beside a chat turn, a second
device) each held a stale value and the last writer erased the other's
increment. The fix is an atomic `UPDATE ... SET c = COALESCE(c,0) + :n`.

The source pin fails on the pre-fix code; the semantics test pins that the
atomic form actually accumulates across two sessions that both start from a
stale read (the exact interleaving that lost increments before).
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


async def test_atomic_increment_survives_the_stale_read_interleaving(tmp_path):
    """Two sessions, both holding a stale read, both increment: nothing lost.

    This is the interleaving that dropped increments pre-fix: A loads
    count=0, B loads count=0, A writes, B writes. With the ORM assignment B
    overwrote A (final 1). The atomic UPDATE computes against the row's
    CURRENT value at execution time (final 2). NULL start pins the COALESCE.
    """
    from app.db.models import Conversation
    from app.db.database import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/c.db")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sc: Conversation.__table__.create(sc, checkfirst=True)
        )
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with maker() as s:
        conv = Conversation(user_id="u1", title="t", message_count=None, total_tokens=None)
        s.add(conv)
        await s.commit()
        cid = conv.id

    async def bump(n_msgs: int, n_toks: int):
        async with maker() as s:
            # Stale read first — pre-fix code based its write on this value.
            await s.execute(select(Conversation).where(Conversation.id == cid))
            await s.execute(
                update(Conversation)
                .where(Conversation.id == cid)
                .values(
                    message_count=func.coalesce(Conversation.message_count, 0) + n_msgs,
                    total_tokens=func.coalesce(Conversation.total_tokens, 0) + n_toks,
                )
            )
            await s.commit()

    await bump(2, 100)
    await bump(2, 150)

    async with maker() as s:
        row = (
            await s.execute(select(Conversation).where(Conversation.id == cid))
        ).scalar_one()
        assert row.message_count == 4, f"lost increment: {row.message_count}"
        assert row.total_tokens == 250, f"lost tokens: {row.total_tokens}"

    await engine.dispose()
