"""I. Concurrency and races."""

from __future__ import annotations

import asyncio

import pytest

from .conftest import record_metric
from .pipeline import active_contents, active_memories, drive_turn, store_direct


async def test_web_and_mobile_writing_simultaneously_both_land(db, user_a):
    """The same user active on two surfaces at once. Both facts must survive —
    a lost update here is a fact the user stated and the agent will never know."""
    from app.db.database import async_session_maker

    async def _write(text):
        async with async_session_maker() as s:
            await store_direct(s, user_a, text)

    await asyncio.gather(
        _write("The user's dog is named Kesh, a border collie."),
        _write("The user's office is on Adelaide Street West."),
    )
    contents = " | ".join(await active_contents(db, user_a)).lower()
    assert "kesh" in contents, contents
    assert "adelaide" in contents, contents


async def test_rapid_fire_writes_lose_nothing(db, user_a):
    from app.db.database import async_session_maker

    facts = [f"The user's project number {i:02d} is codenamed vireo-{i:02d}." for i in range(20)]

    async def _write(text):
        async with async_session_maker() as s:
            await store_direct(s, user_a, text)

    # return_exceptions so that ALL 20 writes are attempted and the report
    # names every failure, rather than gather() cancelling the rest on the
    # first one and leaving a bare PendingRollbackError with no context.
    #
    # This does not soften the test: a write that raised also did not store
    # its fact, so the `missing` assertion below still fails. It only changes
    # what the failure TELLS you — which mattered once. On 2026-08-03 a single
    # run of this test died with PendingRollbackError raised on the first
    # statement of a fresh session, and the bare traceback said nothing about
    # which of the 20 died or why. Ruled out at the time: connection
    # exhaustion (peak 18 of max_connections=100) and the brain_stats race
    # (that is an ON CONFLICT upsert since BUG-3). Not reproduced in 3
    # isolated runs or the 4 full runs since. See report §5.8.
    results = await asyncio.gather(*(_write(f) for f in facts),
                                   return_exceptions=True)
    raised = [r for r in results if isinstance(r, BaseException)]

    contents = " | ".join(await active_contents(db, user_a))
    missing = [f"vireo-{i:02d}" for i in range(20) if f"vireo-{i:02d}" not in contents]
    record_metric("concurrent_writes_lost", len(missing))
    record_metric("concurrent_writes_raised", len(raised))
    assert not missing, (
        f"{len(missing)} of 20 concurrent writes lost: {missing}\n"
        + (f"{len(raised)} write(s) raised: "
           + "; ".join(f"{type(e).__name__}: {e}" for e in raised[:5])
           if raised else "no write raised — the rows are simply absent")
    )
    assert not raised, (
        f"{len(raised)} of 20 concurrent writes raised even though every fact "
        "landed: " + "; ".join(f"{type(e).__name__}: {e}" for e in raised[:5])
    )


async def test_concurrent_turns_do_not_corrupt_rows(db, user_a):
    """Full capture pipeline concurrently on one user."""
    from app.db.database import async_session_maker

    turns = [
        ("My dog is called Kesh.", "Noted."),
        ("I am allergic to peanuts.", "Noted."),
        ("I live in Toronto.", "Noted."),
        ("My co-founder is Zbigniew.", "Noted."),
    ]

    async def _turn(msg, reply):
        async with async_session_maker() as s:
            await drive_turn(s, user_a, msg, reply)

    await asyncio.gather(*(_turn(m, r) for m, r in turns))

    rows = await active_memories(db, user_a)
    for r in rows:
        assert r.content and r.content.strip(), f"empty content on {r.id}"
        assert r.user_id == user_a
        assert r.is_active and not r.is_deleted
        assert 0.0 <= (r.importance or 0) <= 1.0
        assert 0.0 <= (r.confidence or 0) <= 1.0
    record_metric("concurrent_turn_rows", len(rows))
    assert rows, "four concurrent turns produced no memories at all"


async def test_concurrent_duplicate_writes_do_not_both_land(db, user_a):
    """Two surfaces stating the same fact at the same instant. First-writer-wins
    is acceptable; TWO rows for one fact is not."""
    from app.db.database import async_session_maker

    async def _write():
        async with async_session_maker() as s:
            await store_direct(s, user_a, "The user is a vegetarian.", category="preferences")

    await asyncio.gather(*(_write() for _ in range(4)))
    rows = [m for m in await active_memories(db, user_a) if "vegetarian" in m.content.lower()]
    record_metric("concurrent_duplicate_rows", len(rows))
    assert len(rows) <= 1, (
        f"{len(rows)} rows for one fact written concurrently:\n"
        + "\n".join(f"  - {r.id} {r.content}" for r in rows)
    )


async def test_concurrent_delete_and_read_is_consistent(db, user_a):
    from app.db.database import async_session_maker
    from app.services.memory_service import MemoryService
    from .pipeline import recall

    await store_direct(db, user_a, "The user's safe code is nightjar-4417.")
    target = (await active_memories(db, user_a))[0]

    async def _delete():
        async with async_session_maker() as s:
            await MemoryService(s).delete_memory(target.id, user_a)

    async def _read():
        async with async_session_maker() as s:
            return await recall(s, user_a, "safe code", min_similarity=0.0, limit=20)

    results = await asyncio.gather(_delete(), *( _read() for _ in range(5)))
    async with async_session_maker() as s:
        after = await recall(s, user_a, "safe code", min_similarity=0.0, limit=20)
    assert not any("nightjar" in h["content"] for h in after), after
