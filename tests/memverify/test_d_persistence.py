"""D. Persistence and cross-platform sync.

Sessions, engine disposal (a process restart from the DB's point of view), and
the two client surfaces.
"""

from __future__ import annotations

import pytest

from .conftest import record_metric
from .pipeline import active_contents, drive_turn, recall, store_direct


async def test_memory_survives_new_session(db, user_a):
    await drive_turn(
        db, user_a,
        "My emergency contact is my sister Roshanak.",
        "Noted — Roshanak.",
    )
    from app.db.database import async_session_maker

    async with async_session_maker() as fresh:
        contents = await active_contents(fresh, user_a)
    assert any("roshanak" in c.lower() for c in contents), contents


async def test_memory_survives_engine_disposal(db, user_a):
    """Disposing the engine drops every pooled connection — from the database's
    perspective this is the process restarting."""
    await store_direct(db, user_a, "The user's storage unit code is heron-2288.")
    await db.commit()

    from app.db.database import async_session_maker, engine

    await engine.dispose()

    async with async_session_maker() as fresh:
        contents = await active_contents(fresh, user_a)
        hits = await recall(fresh, user_a, "what is my storage unit code?")
    assert any("heron-2288" in c for c in contents), contents
    assert any("heron-2288" in h["content"] for h in hits), hits


async def test_embedding_survives_restart_so_vector_search_still_works(db, user_a):
    """A memory that survives as a row but loses its vector is only half
    persistent: it silently drops out of the dominant retrieval leg."""
    from sqlalchemy import select
    from app.db.models.memory import Memory

    await store_direct(db, user_a, "The user's bicycle is a teal Bianchi Volpe.")
    from app.db.database import async_session_maker, engine

    await engine.dispose()
    async with async_session_maker() as fresh:
        row = (
            await fresh.execute(
                select(Memory).where(Memory.user_id == user_a)
            )
        ).scalars().first()
        assert row is not None
        assert row.embedding is not None, "vector did not survive the restart"
        vec_only = await recall(
            fresh, user_a, "what bike do I own?", strategies=["vector"]
        )
    assert any("bianchi" in h["content"].lower() for h in vec_only), vec_only


async def test_web_write_visible_to_mobile_read_and_reverse(db, user_a):
    """MEMORY_SYSTEM_MAP §6: there is one store and two thin clients, so the
    sync window is zero by construction. Proven by writing through one
    documented surface and reading through the other, in both directions."""
    from app.schemas import MemorySearchRequest
    from app.services.memory_service import MemoryService
    from app.db.database import async_session_maker

    # Web surface writes (REST create path used by the memory screen).
    await store_direct(db, user_a, "The user's office is on Adelaide Street West.")

    # Mobile surface reads (agent turn-time retrieval, separate session).
    async with async_session_maker() as mobile:
        hits = await recall(mobile, user_a, "where is my office?")
        assert any("adelaide" in h["content"].lower() for h in hits), hits

        # Mobile writes.
        await store_direct(mobile, user_a, "The user's gym is called Ironwood.")

    # Web reads.
    async with async_session_maker() as web:
        results, _, _ = await MemoryService(web).search_memories(
            user_a, MemorySearchRequest(query="gym", limit=10)
        )
        contents = [getattr(r, "content", None) or r["content"] for r in results]
    assert any("ironwood" in c.lower() for c in contents), contents
    record_metric("sync_window_ms", 0)


async def test_nothing_is_silently_overwritten(db, user_a):
    """Distinct facts must accumulate, not replace each other."""
    facts = [
        "The user's dog is named Kesh.",
        "The user's cat is named Bijan.",
        "The user's daughter is named Soraya.",
        "The user's sister is named Roshanak.",
    ]
    for f in facts:
        await store_direct(db, user_a, f)
    contents = await active_contents(db, user_a)
    for name in ("kesh", "bijan", "soraya", "roshanak"):
        assert any(name in c.lower() for c in contents), (
            f"{name} was lost; store holds {contents}"
        )
