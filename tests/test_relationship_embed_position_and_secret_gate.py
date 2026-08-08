"""`store_entity_relationship`: one embedding, hoisted out of the transaction,
and a never-store screen on the one Memory INSERT that had none.

Three properties, each of which was false in production:

  1. The function embedded the IDENTICAL string twice — once inside
     `find_similar_memories`, once directly before the INSERT.
  2. Both calls were sync `embed()` inside an `async def`, and both ran AFTER
     `_upsert_entity` had already flushed an `entities` INSERT — a network
     round-trip while holding row locks and a pooled connection. That is the
     #407/#408 shape (network call inside an open transaction -> pinned
     pgbouncer connection -> PendingRollbackError -> HTTP 500 on chat turns).
  3. It was the only Memory INSERT in the codebase with no content screen:
     `relationship_gate_reason` judges the SHAPE of a triple, never its
     values, so `("Nariman", "has_api_key", "sk-…")` was stored verbatim —
     content, entity name and graph edge label. `entity_relationship_create`
     is a model-callable MCP tool, so that payload is reachable from a prompt.

Everything here runs on the file's own sqlite engine (same pattern as
test_memory_taxonomy_and_ttl.py), so it also runs in the sqlite CI sweep.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# An obviously-fake, non-functional key that still matches the gate's
# provider-key pattern. Never use a real value here.
FAKE_API_KEY = "sk-live-FAKEFAKEFAKEFAKE1234567890"


def _embed_dim() -> int:
    """Match Memory.embedding's configured width — pgvector rejects a mismatch."""
    from app.config import settings

    return settings.embedding_dimension


async def _graph_session():
    """sqlite engine carrying exactly the tables this path writes.

    `memories`, `entities`, `entity_links` and `entity_relationships` are all
    AGENT_ONLY, so conftest's platform-profile init_db() does not create them.
    Schema comes from the live ORM so it cannot drift.
    """
    from app.db.models.base import Base
    from app.db.models.entity import Entity, EntityLink, EntityRelationship
    from app.db.models.memory import Memory, MemoryEvent
    from app.db.models.user import User

    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[
                User.__table__,
                Memory.__table__,
                MemoryEvent.__table__,
                Entity.__table__,
                EntityLink.__table__,
                EntityRelationship.__table__,
            ],
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


class RecordingEmbedder:
    """Stands in for EmbeddingService and records WHEN each call happened.

    Implements BOTH `embed` and `embed_async` deliberately: the mutation check
    reverts the production code to the sync call, and this stub must keep
    working there so the test fails on the ASSERTION (call count / ordering)
    rather than on a missing attribute. A stub that only had `embed_async`
    would go red under the mutation for the wrong reason.
    """

    def __init__(self, db, events):
        self._db = db
        self.events = events
        self.calls: list[dict] = []

    def _record(self, text: str, how: str):
        self.calls.append(
            {
                "text": text,
                "how": how,
                # Snapshot of the session AT THE MOMENT of the round-trip.
                "in_transaction": self._db.in_transaction(),
                "pending_new": len(self._db.new),
                "pending_dirty": len(self._db.dirty),
            }
        )
        self.events.append(f"embed:{how}")
        return [0.05] * _embed_dim()

    def embed(self, text, api_key=None):
        return self._record(text, "sync")

    async def embed_async(self, text, api_key=None):
        return self._record(text, "async")


def _instrument(svc, db):
    """Attach the recorder and trace the first statement-issuing step."""
    events: list[str] = []
    embedder = RecordingEmbedder(db, events)
    # Replace the whole service, not just `.embed` — get_embedding_service()
    # hands back a process-wide singleton and must not be mutated.
    svc.embedding_service = embedder

    original_upsert = svc._upsert_entity

    async def traced_upsert(*args, **kwargs):
        events.append("upsert_entity")
        return await original_upsert(*args, **kwargs)

    svc._upsert_entity = traced_upsert
    return embedder, events


async def _seed_user(db, user_id: str, name: str = "Nariman"):
    from app.db.models.user import User

    db.add(
        User(
            id=user_id,
            email=f"rel-{user_id[:8]}@example.com",
            hashed_password="x",
            name=name,
        )
    )
    await db.commit()


async def _count(db, model) -> int:
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


# ── 1. One embedding per relationship, not two ────────────────────────


@pytest.mark.asyncio
async def test_relationship_embeds_the_string_exactly_once():
    """It used to embed the identical sentence twice, per stored relationship.

    Once inside find_similar_memories (memory_service.py:1967 before this
    change) and once immediately before the INSERT (:2309). Same string, same
    provider, two round-trips.
    """
    from app.db.models.memory import Memory
    from app.services.memory_service import MemoryService

    user_id = str(uuid.uuid4())
    engine, Session = await _graph_session()
    try:
        async with Session() as db:
            await _seed_user(db, user_id)
            svc = MemoryService(db)
            embedder, _ = _instrument(svc, db)

            await svc.store_entity_relationship(
                user_id=user_id,
                source_name="Nariman",
                source_type="person",
                target_name="Toup",
                target_type="project",
                relationship="owns",
            )

            texts = [c["text"] for c in embedder.calls]
            assert len(embedder.calls) == 1, (
                f"expected ONE embedding round-trip, got {len(embedder.calls)}: {texts}"
            )
            assert texts == ["Nariman owns Toup"]

            # ...and the vector actually reached the row, so "one call" did not
            # become "one call and an unembedded memory".
            stored = (await db.execute(select(Memory))).scalars().all()
            assert len(stored) == 1
            assert stored[0].embedding_json is not None
    finally:
        await engine.dispose()


# ── 2. The round-trip happens outside the transaction ─────────────────


@pytest.mark.asyncio
async def test_relationship_embedding_happens_before_the_session_is_touched():
    """The embedding must not run while the session holds write state.

    WHAT THIS ASSERTS: at the instant the embedding call is made, the session
    has no open transaction, no pending INSERTs (`db.new`), no pending UPDATEs
    (`db.dirty`), and `_upsert_entity` — the first thing in the function that
    executes SQL, and which FLUSHES a new `entities` row — has not run yet.

    WHAT IT DOES NOT ASSERT: that the event loop is unblocked during the call.
    That is `embed_async`'s contract (a run_in_executor wrapper); this stub is
    a plain coroutine so the snapshot above is read on the loop thread. Nor
    does it prove anything about a caller that hands the service a session
    already inside a transaction — it proves this function does not open one
    before embedding.
    """
    from app.services.memory_service import MemoryService

    user_id = str(uuid.uuid4())
    engine, Session = await _graph_session()
    try:
        async with Session() as db:
            await _seed_user(db, user_id)
            svc = MemoryService(db)
            embedder, events = _instrument(svc, db)

            # Session is clean when the call starts — otherwise the snapshot
            # below would be measuring the fixture, not the function.
            assert db.in_transaction() is False
            assert not db.new and not db.dirty

            await svc.store_entity_relationship(
                user_id=user_id,
                source_name="Nariman",
                source_type="person",
                target_name="Toup",
                target_type="project",
                relationship="owns",
            )

            # Snapshot assertions FIRST, deliberately: under the mutation that
            # puts the embedding back inside the transaction there are two
            # calls, and this test should report the position failure rather
            # than shadow it with a call-count mismatch (test 1 owns that).
            assert embedder.calls, "no embedding happened at all"
            snap = embedder.calls[0]
            assert snap["in_transaction"] is False, (
                "embedded with a transaction already open — the #407/#408 shape"
            )
            assert snap["pending_new"] == 0, (
                f"embedded with {snap['pending_new']} pending INSERT(s) staged"
            )
            assert snap["pending_dirty"] == 0, (
                f"embedded with {snap['pending_dirty']} pending UPDATE(s) staged"
            )
            assert len(embedder.calls) == 1

            # Ordering, independent of the session-state snapshot: the embed
            # precedes the first entity upsert (which flushes an INSERT and so
            # takes row locks for the rest of the transaction).
            assert events, "nothing was recorded"
            assert events[0].startswith("embed:"), events
            assert "upsert_entity" in events, events
            assert events.index("embed:async") < events.index("upsert_entity"), events
    finally:
        await engine.dispose()


# ── 3. A secret cannot enter through the entity graph ─────────────────


@pytest.mark.asyncio
async def test_relationship_carrying_an_api_key_is_refused_everywhere():
    """No Memory row, no entity, no graph edge — and no embedding of it.

    `relationship_gate_reason` returns None for this triple (verified: it is
    well-shaped and user-endpointed), which is exactly why the shape gate
    alone was not a content screen.
    """
    from app.db.models.entity import Entity, EntityLink, EntityRelationship
    from app.db.models.memory import Memory
    from app.services.memory_gate import relationship_gate_reason
    from app.services.memory_service import MemoryService

    # Pin the premise: the pre-existing gate has no objection to this payload.
    assert (
        relationship_gate_reason(
            "Nariman",
            "has_api_key",
            FAKE_API_KEY,
            user_aliases=["Nariman"],
            rendered=f"Nariman has api key {FAKE_API_KEY}",
        )
        is None
    ), "premise broken: the shape gate now rejects this for an unrelated reason"

    user_id = str(uuid.uuid4())
    engine, Session = await _graph_session()
    try:
        async with Session() as db:
            await _seed_user(db, user_id)
            svc = MemoryService(db)
            embedder, _ = _instrument(svc, db)

            await svc.store_entity_relationship(
                user_id=user_id,
                source_name="Nariman",
                source_type="person",
                target_name=FAKE_API_KEY,
                target_type="credential",
                relationship="has_api_key",
            )

            assert await _count(db, Memory) == 0, "the secret was mirrored to memories"
            assert await _count(db, Entity) == 0, "the secret was stored as an entity name"
            assert await _count(db, EntityRelationship) == 0, "the secret reached the graph"
            assert await _count(db, EntityLink) == 0
            assert embedder.calls == [], "the secret was sent to the embedding provider"

            # Nothing of the value survives anywhere in the store.
            for table in (Memory.content, Entity.name, EntityRelationship.relationship_label):
                rows = (await db.execute(select(table))).scalars().all()
                assert not any(FAKE_API_KEY in (r or "") for r in rows), table
    finally:
        await engine.dispose()


# ── 4. The knowledge graph stays ungated ──────────────────────────────


@pytest.mark.asyncio
async def test_a_mirror_gated_relationship_still_writes_the_graph_edge():
    """CONTROL for the reordering. The mirror gate must not become a graph gate.

    The mirror gate now has an alias-independent half that runs BEFORE the
    entity writes (that is what lets the embedding be hoisted out of the
    transaction). If that early verdict were allowed to `return`, every
    scaffolding/tautology/agent-self edge would vanish from
    `entity_relationships` too — traversal and the entity map are deliberately
    ungated and would silently lose fidelity.

    This is also an anti-vacuity control and must stay GREEN under both
    mutations in the PR body. It deliberately does NOT assert how many
    embeddings a gated edge costs: the alias-aware half rejects AFTER the
    embedding, so an edge dropped by it pays for one it never uses. That is a
    stated trade-off, not an invariant — pinning it would make the test fail
    if someone later found a way to reclaim it.
    """
    from app.db.models.entity import Entity, EntityRelationship
    from app.db.models.memory import Memory
    from app.services.memory_service import MemoryService

    user_id = str(uuid.uuid4())
    engine, Session = await _graph_session()
    try:
        async with Session() as db:
            await _seed_user(db, user_id)
            svc = MemoryService(db)
            embedder, _ = _instrument(svc, db)

            # (a) Rejected by the alias-independent half -> no embedding at all.
            await svc.store_entity_relationship(
                user_id=user_id,
                source_name="the assistant",
                source_type="agent",
                target_name="Nariman",
                target_type="person",
                relationship="helps",
            )
            assert await _count(db, Memory) == 0, "agent-self edge was mirrored"
            assert await _count(db, EntityRelationship) == 1, "graph edge was dropped"
            assert await _count(db, Entity) == 2, "entities were dropped"
            assert embedder.calls == [], "junk the pure gate already caught was embedded"

            # (b) Rejected only by the alias-AWARE half, which runs after the
            #     entity writes. Graph edge still written, still no mirror row.
            await svc.store_entity_relationship(
                user_id=user_id,
                source_name="Better Call Saul",
                source_type="show",
                target_name="Netflix",
                target_type="company",
                relationship="available_on",
            )
            assert await _count(db, Memory) == 0, "world knowledge was mirrored"
            assert await _count(db, EntityRelationship) == 2, "graph edge was dropped"
            assert await _count(db, Entity) == 4, "entities were dropped"
    finally:
        await engine.dispose()


# ── 5. ANTI-VACUITY CONTROL ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_benign_relationship_still_stores_end_to_end():
    """CONTROL. Must stay GREEN under every mutation in the PR body.

    Without it, "the secret test passes" and "the embed test passes" are both
    satisfiable by a function that stores nothing at all.
    """
    from app.db.models.entity import Entity, EntityLink, EntityRelationship
    from app.db.models.memory import Memory
    from app.services.memory_service import MemoryService

    user_id = str(uuid.uuid4())
    engine, Session = await _graph_session()
    try:
        async with Session() as db:
            await _seed_user(db, user_id)
            svc = MemoryService(db)
            # Stubbed only to keep the provider off the network; the assertions
            # below are about the rows, not about the embedder.
            _instrument(svc, db)

            await svc.store_entity_relationship(
                user_id=user_id,
                source_name="Nariman",
                source_type="person",
                target_name="Toup",
                target_type="project",
                relationship="owns",
                confidence=0.9,
            )

            memories = (await db.execute(select(Memory))).scalars().all()
            assert len(memories) == 1, "the benign relationship was not mirrored"
            mem = memories[0]
            assert mem.content == "Nariman owns Toup"
            assert mem.source_type == "entity_extraction"
            assert mem.embedding_json is not None, "stored without a vector"

            assert await _count(db, Entity) == 2
            assert await _count(db, EntityRelationship) == 1
            assert await _count(db, EntityLink) == 2

            edge = (await db.execute(select(EntityRelationship))).scalars().one()
            assert edge.relationship_type == "owns"
            assert edge.relationship_label == "Nariman owns Toup"
    finally:
        await engine.dispose()
