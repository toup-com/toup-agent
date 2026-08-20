"""`store_entity_relationship`: a never-store screen, and the graph edge that
survives v3.

Three properties were false in production. TWO of them retire with the
Memory MIRROR that v3 deletes (§1.1) — the double embedding, and the
#407/#408 hazard of holding a pooled connection across it — because there
is no embedding on this path any more. Their retirement notes are inline
below rather than in a changelog, so the next reader sees why a test that
was load-bearing is gone.

What SURVIVES is the third, and it is the one that was reachable from a
prompt:

  It was the only Memory INSERT in the codebase with no content screen:
  `relationship_gate_reason` judged the SHAPE of a triple, never its
  values, so `("Nariman", "has_api_key", "sk-…")` was stored verbatim —
  content, entity name and graph edge label. `entity_relationship_create`
  is a model-callable MCP tool, so that payload is reachable from a prompt.

The screen still aborts the WHOLE write, mirror or no mirror, because
`entities.name` stores the string verbatim and
`entity_relationships.relationship_label` renders it back.

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


# RETIRED with the relationship MIRROR (v3 §1.1): `test_relationship_embeds_the_string_exactly_once`.
# It embedded `relationship_content` so the Memory MIRROR row could be found by vector search. The mirror is deleted (v3 §1.1) and with it the embedding — this path now makes no provider call at all, which is a stronger version of "exactly once".


# ── 2. The round-trip happens outside the transaction ─────────────────


# RETIRED with the relationship MIRROR (v3 §1.1): `test_relationship_embedding_happens_before_the_session_is_touched`.
# The #407/#408 hazard (a network round trip inside the transaction the entity upserts opened) is gone by CONSTRUCTION rather than by ordering: there is no embedding call left on this path. `test_a_mirror_gated_relationship_still_writes_the_graph_edge` below is what now proves the graph write survives.


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


# RETIRED with the relationship MIRROR (v3 §1.1): `test_a_benign_relationship_still_stores_end_to_end`.
# "Stores" meant a Memory row. There is no mirror row to store; the graph edge is asserted by the test above it.
