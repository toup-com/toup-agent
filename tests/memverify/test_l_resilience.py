"""L. Failure injection and resilience.

The contract (MEMORY_SYSTEM_MAP §7): capture is best-effort and must never take
the conversation down; retrieval degrades leg-by-leg; a malformed row must not
poison context or crash anything.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text as sql_text

from .conftest import record_metric
from .pipeline import active_contents, active_memories, drive_turn, recall, store_direct


async def test_llm_extraction_failure_does_not_break_the_turn(db, user_a, monkeypatch):
    """The LLM is down. The turn must still complete and write nothing —
    never a regex-fallback fragment (that fallback is disabled on purpose)."""
    from app.services import memory_extractor as me

    async def _boom(*a, **k):
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(me, "_complete_json_with_retry", _boom)

    n = await drive_turn(db, user_a, "My dog is called Kesh.", "Noted.")
    assert n == 0
    assert await active_contents(db, user_a) == []


async def test_embedding_outage_still_stores_the_fact(db, user_a, monkeypatch):
    """A vector is an optimisation; the fact is the product. An embedding
    outage must degrade to an unembedded write, and keyword retrieval must
    still find it."""
    from app.services import embedding_service as es

    real = es.EmbeddingService.embed_async

    async def _boom(self, *a, **k):
        raise RuntimeError("simulated embedding outage")

    monkeypatch.setattr(es.EmbeddingService, "embed_async", _boom)
    await store_direct(db, user_a, "The user's storage unit code is heron-2288.")
    monkeypatch.setattr(es.EmbeddingService, "embed_async", real)

    rows = await active_memories(db, user_a)
    assert len(rows) == 1, rows
    assert rows[0].embedding is None, "expected an unembedded row"

    hits = await recall(
        db, user_a, "heron-2288", strategies=["keyword"], min_similarity=0.0, limit=20
    )
    assert any("heron-2288" in h["content"] for h in hits), (
        f"unembedded row is unreachable — the fact is stored but lost: {hits}"
    )


async def test_database_outage_does_not_crash_the_turn(db, user_a, monkeypatch):
    """DB down mid-conversation. The turn survives and nothing is corrupted.

    Capture reports 0 for the turn — correctly, since no row was written — but
    the facts are no longer LOST: they are owed, and a later turn replays them.
    This test measures both halves, because "the turn survived" on its own was
    what made the old data loss look acceptable."""
    from app.services import memory_capture_outbox_service as outbox
    from app.services import memory_dedup_service as mds

    async def _boom(self, *a, **k):
        raise ConnectionRefusedError("simulated pgbouncer outage")

    monkeypatch.setattr(mds.MemoryDedupService, "smart_create_memories", _boom)

    n = await drive_turn(db, user_a, "My dog is called Kesh.", "Noted.")
    assert n == 0, "capture reported success during a DB outage"

    owed = await outbox.pending_count(db, user_a)
    assert owed >= 1, "the turn's facts were lost with nothing owed"
    record_metric("db_outage_facts_owed_not_lost", owed)


async def test_partial_write_leaves_no_half_row(db, user_a, monkeypatch):
    """If entity linking blows up after the memory row is created, the row must
    either be complete or absent — never a content-less shell."""
    from app.services import memory_service as ms

    real = ms.MemoryService._upsert_entity

    async def _boom(self, *a, **k):
        raise RuntimeError("simulated entity upsert failure")

    monkeypatch.setattr(ms.MemoryService, "_upsert_entity", _boom)
    try:
        await drive_turn(
            db, user_a, "My co-founder is Zbigniew Wojciechowski.", "Noted."
        )
    finally:
        monkeypatch.setattr(ms.MemoryService, "_upsert_entity", real)

    for row in await active_memories(db, user_a):
        assert row.content and row.content.strip(), f"content-less row {row.id}"
        assert row.user_id == user_a


async def test_malformed_row_does_not_poison_retrieval(db, user_a):
    """Rows written by older code, by a failed migration, or by a partial write.
    Retrieval must survive all of them and still return the good row."""
    await store_direct(db, user_a, "The user's dog is named Kesh, a border collie.")

    # Hand-crafted damage, written straight past the ORM. Every shape here is
    # one the schema actually permits, so each could genuinely exist in a
    # tenant DB: written by older code, left by a half-finished migration, or
    # produced by a partial write. NOT-NULL columns are deliberately excluded —
    # the database already refuses those, which is itself worth knowing and is
    # asserted separately below.
    rows = [
        ("mal-empty", "", "other", "user", "fact", None, None),
        ("mal-long", "x" * 40000, "other", "user", "fact", None, None),
        ("mal-enum", "row with values from no enum", "", "wat", "bogus", None, None),
        ("mal-ctrl", "escapes, emoji, tabs\tand 'quotes'", "other", "user", "fact", None, None),
        ("mal-json", "row with corrupt json columns", "other", "user", "fact", "{not valid json", "[[["),
    ]
    await db.execute(
        sql_text(
            "INSERT INTO memories (id, user_id, brain_type, content, category, "
            "memory_type, memory_level, source_type, emotional_salience, importance, "
            "confidence, strength, decay_rate, access_count, consolidation_count, "
            "metadata_json, tags_json, is_active, is_deleted, created_at, updated_at) "
            "VALUES (:id, :u, :brain, :content, :cat, :mtype, 'episodic', 'seed', "
            "0.5, 0.5, 0.5, 1.0, 0.01, 0, 0, :meta, :tags, true, false, now(), now())"
        ),
        [
            {"id": rid, "u": user_a, "content": content, "cat": cat,
             "brain": brain, "mtype": mtype, "meta": meta, "tags": tags}
            for rid, content, cat, brain, mtype, meta, tags in rows
        ],
    )
    await db.commit()

    hits = await recall(db, user_a, "what is my dog's name?", min_similarity=0.0, limit=20)
    assert any("kesh" in h["content"].lower() for h in hits), (
        f"malformed rows displaced the good one: {[h['content'][:40] for h in hits]}"
    )
    for strategy in ("vector", "keyword", "graph", "temporal"):
        await recall(db, user_a, "dog", strategies=[strategy], min_similarity=0.0, limit=20)

    from app.services.memory_service import MemoryService
    from app.schemas import MemorySearchRequest

    await MemoryService(db).search_memories(user_a, MemorySearchRequest(query="dog", limit=10))
    await MemoryService(db).list_memories(user_id=user_a, limit=100)


async def test_retrieval_survives_a_dead_vector_index(db, user_a, monkeypatch):
    """One leg failing must not take retrieval down — the other legs still fuse."""
    from app.services import memory_service as ms

    await store_direct(db, user_a, "The user's dog is named Kesh, a border collie.")

    async def _boom(self, *a, **k):
        raise RuntimeError("simulated pgvector failure")

    monkeypatch.setattr(ms.MemoryService, "_vector_search", _boom)
    hits = await recall(db, user_a, "Kesh", min_similarity=0.0, limit=20)
    assert any("kesh" in h["content"].lower() for h in hits), (
        f"losing the vector leg lost the memory entirely: {hits}"
    )


# ── The write failed after extraction succeeded ──────────────────────────
#
# The gap this closes: capture is fire-and-forget in post-processing, so when
# the WRITE fails there is no request to return an error to and no user to retry
# it. The LLM call already retries once (A6-2) and an embedding outage already
# degrades to an unembedded write (W1.5) — but a connection blip during the
# store dropped every fact the user stated that turn, silently and permanently.

async def test_a_failed_write_parks_the_facts_instead_of_losing_them(
    db, user_a, monkeypatch
):
    from app.services import memory_capture_outbox_service as outbox
    from app.services.memory_dedup_service import MemoryDedupService

    async def _boom(*a, **k):
        raise RuntimeError("simulated storage outage")

    monkeypatch.setattr(MemoryDedupService, "smart_create_memories", _boom)

    n = await drive_turn(db, user_a, "My dog is called Kesh, a border collie.", "Noted.")
    assert n == 0, "the write failed, so nothing should be reported stored"
    assert await active_contents(db, user_a) == []

    # ...but the extracted facts are owed, not gone.
    assert await outbox.pending_count(db, user_a) >= 1


async def test_parked_facts_are_replayed_and_become_real_memories(
    db, user_a, monkeypatch
):
    from app.services import memory_capture_outbox_service as outbox
    from app.services.memory_dedup_service import MemoryDedupService

    real = MemoryDedupService.smart_create_memories

    async def _boom(*a, **k):
        raise RuntimeError("simulated storage outage")

    monkeypatch.setattr(MemoryDedupService, "smart_create_memories", _boom)
    await drive_turn(db, user_a, "My dog is called Vantabrook, a whippet.", "Noted.")
    assert await active_contents(db, user_a) == []
    monkeypatch.setattr(MemoryDedupService, "smart_create_memories", real)

    # The storage outage is over. The next turn's maintenance replays it.
    resolved = await outbox.replay_pending(db, user_a)
    await db.commit()

    assert resolved >= 1
    contents = " ".join(await active_contents(db, user_a)).lower()
    assert "vantabrook" in contents, "the parked fact never made it into memory"
    assert await outbox.pending_count(db, user_a) == 0


async def test_a_poison_payload_is_abandoned_not_retried_forever(
    db, user_a, monkeypatch
):
    """A row that fails every time must not become a permanent per-turn tax."""
    from app.services import memory_capture_outbox_service as outbox
    from app.db.models.memory_capture_outbox import MemoryCaptureOutbox
    from app.services.memory_dedup_service import MemoryDedupService
    from sqlalchemy import select

    async def _boom(*a, **k):
        raise RuntimeError("simulated storage outage")

    monkeypatch.setattr(MemoryDedupService, "smart_create_memories", _boom)
    await drive_turn(db, user_a, "My cat is called Sundermere.", "Noted.")
    assert await outbox.pending_count(db, user_a) >= 1

    # Replay repeatedly; the write keeps failing. Backoff is cleared each round
    # so the attempt counter, not the clock, is what is under test.
    for _ in range(outbox.MAX_ATTEMPTS + 2):
        rows = (await db.execute(
            select(MemoryCaptureOutbox).where(
                MemoryCaptureOutbox.user_id == user_a,
                MemoryCaptureOutbox.resolved_at.is_(None),
            )
        )).scalars().all()
        for r in rows:
            r.next_attempt_at = None
        await db.flush()
        await outbox.replay_pending(db, user_a)

    assert await outbox.pending_count(db, user_a) == 0, "poison row retried forever"

    row = (await db.execute(
        select(MemoryCaptureOutbox).where(MemoryCaptureOutbox.user_id == user_a)
    )).scalars().first()
    assert row.attempts <= outbox.MAX_ATTEMPTS
    assert row.last_error, "abandoned without recording why"
    record_metric("outbox_max_attempts", outbox.MAX_ATTEMPTS)


def test_the_outbox_table_is_agent_only():
    """memories live in the tenant database; its outbox must not be created in
    the platform database, where nothing would ever drain it."""
    from app.db.models.base import AGENT_ONLY_TABLES

    assert "memory_capture_outbox" in AGENT_ONLY_TABLES


async def test_parking_survives_a_poisoned_transaction(db, user_a, monkeypatch):
    """The case the outbox exists for, which the test above does NOT cover.

    `test_a_failed_write_parks_the_facts` raises a RuntimeError, which leaves the
    session perfectly usable. A real database error does not: it poisons the
    transaction, so every later statement on that session raises — including the
    INSERT that parks the row. The safety net would fail in exactly the
    circumstance it was built for.

    Simulated by breaking flush on THIS session object only, so the rescue
    session (a different object) still works.
    """
    from app.services.memory_dedup_service import MemoryDedupService
    from app.db.database import async_session_maker
    from app.db.models.memory_capture_outbox import MemoryCaptureOutbox
    from sqlalchemy import select

    async def _boom(*a, **k):
        raise RuntimeError("simulated storage outage")

    monkeypatch.setattr(MemoryDedupService, "smart_create_memories", _boom)

    async def _poisoned_flush(*a, **k):
        raise RuntimeError("current transaction is aborted")

    monkeypatch.setattr(db, "flush", _poisoned_flush)

    await drive_turn(db, user_a, "My dog is called Emberline, a lurcher.", "Noted.")

    # Counted on a SEPARATE session: the caller's is the one we just broke.
    async with async_session_maker() as check:
        rows = (await check.execute(
            select(MemoryCaptureOutbox).where(MemoryCaptureOutbox.user_id == user_a)
        )).scalars().all()

    assert len(rows) == 1, "a poisoned transaction lost the facts entirely"
    stored = " ".join(str(r.payload_json) for r in rows).lower()
    assert "emberline" in stored, "parked, but without the fact it was meant to save"
