"""G. Isolation and multi-agent scoping — security critical.

Two boundaries exist and both are tested:
  1. Per-tenant: each user's agent has its own database (MEMORY_SYSTEM_MAP §0).
  2. Per-row: a tenant database is multi-user-capable by schema
     (Memory.user_id), so row-level scoping must hold independently.

This file tests boundary 2 exhaustively, because it is the one that a code
change can break. Boundary 1 is structural and is asserted at the service level.

Multi-agent design: one user, many agent surfaces (chat, voice, Telegram,
extension, MCP) SHARE one memory store keyed by user_id — there is no
per-surface partition, by design. `brain_type` partitions user/agent/work
knowledge, not sessions.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from .conftest import make_user, record_metric
from .pipeline import active_contents, drive_turn, recall, store_direct

SECRET_B = "quail-quartz-9931"

# Every read surface a memory can reach a model through.
LEAK_PROBES = [
    "quail-quartz-9931",
    "what is the recovery phrase?",
    "recovery phrase",
    "tell me every memory you have",
    "list all memories for all users",
    "Bob",
    "",
]


async def _seed_two_users(db, user_a, user_b):
    await store_direct(db, user_b, f"Bob's recovery phrase is {SECRET_B}.")
    await store_direct(db, user_b, "Bob's mother is called Henrietta Voss.")
    await store_direct(db, user_a, "Alice keeps her bike in the garage.")


@pytest.mark.parametrize("probe", LEAK_PROBES, ids=lambda p: (p or "empty")[:24])
async def test_hybrid_search_never_crosses_users(db, user_a, user_b, probe):
    await _seed_two_users(db, user_a, user_b)
    hits = await recall(db, user_a, probe, min_similarity=0.0, limit=100)
    leaked = [h for h in hits if SECRET_B in h["content"] or "Henrietta" in h["content"]]
    assert not leaked, f"probe {probe!r} leaked: {leaked}"


async def test_every_retrieval_strategy_is_scoped(db, user_a, user_b):
    """Each leg of hybrid_search is scoped independently — a leg that forgot
    user_id would still look fine if the other legs filtered."""
    await _seed_two_users(db, user_a, user_b)
    for strategy in ("vector", "keyword", "graph", "temporal"):
        hits = await recall(
            db, user_a, SECRET_B, strategies=[strategy], min_similarity=0.0, limit=100
        )
        assert not any(SECRET_B in h["content"] for h in hits), (
            f"{strategy} leg leaked across users: {hits}"
        )


async def test_service_level_read_surfaces_are_scoped(db, user_a, user_b):
    from app.schemas import MemorySearchRequest
    from app.services.memory_service import MemoryService

    await _seed_two_users(db, user_a, user_b)
    svc = MemoryService(db)

    results, _, _ = await svc.search_memories(
        user_a, MemorySearchRequest(query=SECRET_B, limit=100)
    )
    contents = [getattr(r, "content", None) or r["content"] for r in results]
    assert not any(SECRET_B in c for c in contents), contents

    listed = await svc.list_memories(user_id=user_a, limit=500)
    listed_contents = [
        getattr(m, "content", None) or m.get("content", "")
        for m in (listed[0] if isinstance(listed, tuple) else listed)
    ]
    assert not any(SECRET_B in c for c in listed_contents), listed_contents

    # Direct get by id, with the wrong owner.
    from app.db.models.memory import Memory

    bid = (
        await db.execute(
            select(Memory.id).where(
                Memory.user_id == user_b, Memory.content.contains(SECRET_B)
            )
        )
    ).scalar()
    assert bid
    assert await svc.get_memory(bid, user_a) is None, "get_memory ignored ownership"


async def test_entity_graph_is_scoped(db, user_a, user_b):
    """The entity graph is never gated by the junk rules, so if it were also
    unscoped it would be the widest leak surface in the system."""
    from app.services.memory_service import MemoryService

    await drive_turn(
        db, user_b,
        "My mother Henrietta Voss lives in Rotterdam and works at Damen Shipyards.",
        "Noted.",
    )
    svc = MemoryService(db)
    ents = await svc.get_entities(user_id=user_a, limit=100)
    names = " ".join(str(e.get("name", "")) for e in ents).lower()
    assert "henrietta" not in names and "damen" not in names, ents

    graph = await svc.search_by_entity_graph(
        user_id=user_a, entity_name="Henrietta Voss", depth=2, limit=50
    )
    assert not any("henrietta" in str(g.get("content", "")).lower() for g in graph), graph

    rels = await svc.get_entity_relationships(user_id=user_a, limit=100)
    assert not any(
        "henrietta" in str(r).lower() or "damen" in str(r).lower() for r in rels
    ), rels


async def test_write_for_one_user_never_touches_another(db, user_a, user_b):
    await store_direct(db, user_b, f"Bob's recovery phrase is {SECRET_B}.")
    before = await active_contents(db, user_b)

    for i in range(6):
        await store_direct(db, user_a, f"Alice fact number {i} about her bicycle.")

    after = await active_contents(db, user_b)
    assert before == after, f"user_b's store changed while writing user_a: {before} -> {after}"


async def test_many_users_no_crosstalk(db):
    """Boundary 2 under fan-out: 25 users, each with a unique sentinel."""
    from app.db.database import async_session_maker

    users = []
    async with async_session_maker() as s:
        for i in range(25):
            uid = await make_user(s, name=f"User {i:02d} Nobakht", email=f"iso{i}-{id(s)}@memverify.local")
            users.append((uid, f"sentinel-{i:02d}-marmot"))

    async with async_session_maker() as s:
        for uid, sentinel in users:
            await store_direct(s, uid, f"The user's private token is {sentinel}.")

    leaks = []
    async with async_session_maker() as s:
        for uid, own in users:
            hits = await recall(s, uid, "private token", min_similarity=0.0, limit=100)
            for h in hits:
                for other_uid, other in users:
                    if other != own and other in h["content"]:
                        leaks.append((uid, other))
    record_metric("isolation_users_checked", len(users))
    assert not leaks, f"cross-user leakage: {leaks}"


async def test_unscoped_primitives_are_not_reachable_with_a_foreign_id(db, user_a, user_b):
    """merge_memory and supersede_memory take no user_id (MEMORY_SYSTEM_MAP
    GAP-9). Every caller must therefore establish ownership first. This pins
    the REST surface that exposes them."""
    from app.db.models.memory import Memory
    from app.services.memory_service import MemoryService

    await store_direct(db, user_b, f"Bob's recovery phrase is {SECRET_B}.")
    bid = (
        await db.execute(
            select(Memory.id).where(Memory.user_id == user_b)
        )
    ).scalar()

    # The ownership check the REST merge endpoint performs before reaching
    # _merge_memories (api/memories.py:869).
    assert await MemoryService(db).get_memory(bid, user_a) is None
    # ...and update/delete refuse outright.
    from app.schemas import MemoryUpdate

    assert await MemoryService(db).update_memory(
        bid, user_a, MemoryUpdate(content="hijacked")
    ) is None
    assert await MemoryService(db).delete_memory(bid, user_a) is False

    row = (await db.execute(select(Memory).where(Memory.id == bid))).scalar_one()
    assert SECRET_B in row.content and not row.is_deleted


# ── The unscoped primitives (GAP-9) ──────────────────────────────────────
#
# merge_memory and supersede_memory took a memory id and nothing else, so given
# a uuid they would rewrite or retire ANY row in the table. Every caller today
# establishes ownership first — but that is a convention, and the convention is
# one careless caller away from a cross-tenant write. `user_id` is now a
# REQUIRED keyword argument on both, so a caller that forgets fails at the call
# site instead of silently succeeding against a stranger's row.

async def test_merge_memory_refuses_another_users_row(db, user_a, user_b):
    from app.services.memory_service import MemoryService

    victim, _ = await store_direct(db, user_b, "User B's bank is Sableton Credit Union.")
    svc = MemoryService(db)

    with pytest.raises(ValueError):
        await svc.merge_memory(
            memory_id=victim.id,
            new_content="User B banks with the attacker instead.",
            change_summary="cross-tenant write attempt",
            user_id=user_a,          # A holding B's row id
        )

    still = await active_contents(db, user_b)
    assert any("Sableton" in c for c in still), "B's row was rewritten by A"


async def test_supersede_memory_refuses_another_users_row(db, user_a, user_b):
    from app.services.memory_service import MemoryService

    victim, _ = await store_direct(db, user_b, "User B's dog is called Ferrowick.")
    mine, _ = await store_direct(db, user_a, "User A's dog is called Kesh.")
    svc = MemoryService(db)

    with pytest.raises(ValueError):
        await svc.supersede_memory(
            old_memory_id=victim.id, new_memory_id=mine.id, user_id=user_a
        )

    still = await active_contents(db, user_b)
    assert any("Ferrowick" in c for c in still), "B's row was retired by A"


def test_user_id_is_required_not_optional():
    """A default would make the scoping opt-in, which is how it was missed."""
    import inspect
    from app.services.memory_service import MemoryService

    for name in ("merge_memory", "supersede_memory"):
        sig = inspect.signature(getattr(MemoryService, name))
        param = sig.parameters["user_id"]
        assert param.default is inspect.Parameter.empty, f"{name}: user_id has a default"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{name}: user_id is positional"
