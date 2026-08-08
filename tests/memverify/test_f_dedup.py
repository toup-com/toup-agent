"""F. Deduplication and consolidation."""

from __future__ import annotations

import uuid

import pytest

from .conftest import record_metric
from .pipeline import active_contents, active_memories, drive_turn, store_direct

VEGETARIAN_PHRASINGS = [
    "The user is a vegetarian.",
    "The user does not eat meat.",
    "The user follows a vegetarian diet.",
    "The user has been vegetarian for eight years.",
    "The user avoids all meat products.",
]


async def test_same_fact_five_ways_yields_one_entry(db, user_a):
    for phrasing in VEGETARIAN_PHRASINGS:
        await store_direct(db, user_a, phrasing, category="preferences")

    rows = await active_memories(db, user_a)
    record_metric("dedup_rows_for_5_phrasings", len(rows))
    assert len(rows) == 1, (
        f"expected 1 consolidated row, got {len(rows)}:\n"
        + "\n".join(f"  - {r.content}" for r in rows)
    )


async def test_dedup_compares_against_the_nearest_row_not_the_hottest(db, user_a):
    """Regression guard for the candidate-ranking defect.

    `search_memories_by_embedding` returns candidates ordered by a BLENDED
    score (similarity is 40%, against strength/importance/salience/recency).
    If the write path takes candidates[0] without re-sorting by similarity, a
    hot, frequently-retrieved row outranks the genuine nearest duplicate, gets
    adjudicated in its place, and the real duplicate is never compared — so it
    is stored again. That is self-reinforcing: the more a row is retrieved, the
    more reliably it hijacks the comparison.

    Setup: one hot decoy with maximal strength/importance/salience/recency, plus
    the true near-duplicate. Then write the duplicate again.
    """
    from datetime import datetime

    from app.db.models.memory import Memory

    await store_direct(
        db, user_a, "The user is a vegetarian.", category="preferences"
    )
    await store_direct(
        db, user_a,
        "The user enjoys cooking elaborate meals at home on weekends.",
        category="preferences",
    )

    # Make the unrelated row hot: max strength/importance/salience + just-accessed.
    decoy = [
        m for m in await active_memories(db, user_a) if "elaborate" in m.content
    ][0]
    decoy.strength = 1.0
    decoy.importance = 1.0
    decoy.emotional_salience = 1.0
    decoy.access_count = 40
    decoy.last_accessed_at = datetime.utcnow()
    await db.commit()

    before = len(await active_memories(db, user_a))
    await store_direct(
        db, user_a, "The user follows a vegetarian diet.", category="preferences"
    )
    rows = await active_memories(db, user_a)
    assert len(rows) == before, (
        "a duplicate was stored as a new row while a hotter, less similar row "
        "was adjudicated in its place:\n"
        + "\n".join(f"  - s={r.strength:.2f} {r.content}" for r in rows)
    )


async def test_batch_write_path_dedups_like_the_single_path(db, user_a):
    """agent_runner uses smart_create_memories (batch); the memory_store tool
    uses smart_create_memory (single). They must agree — the production path is
    the batch one, and a fix applied to only one of them is not applied."""
    from app.schemas import BrainType, MemoryCreate, MemoryLevel, MemoryType
    from app.services.memory_dedup_service import MemoryDedupService
    from datetime import datetime

    await store_direct(db, user_a, "The user is a vegetarian.", category="preferences")
    await store_direct(
        db, user_a,
        "The user enjoys cooking elaborate meals at home on weekends.",
        category="preferences",
    )
    decoy = [m for m in await active_memories(db, user_a) if "elaborate" in m.content][0]
    decoy.strength = 1.0
    decoy.importance = 1.0
    decoy.emotional_salience = 1.0
    decoy.last_accessed_at = datetime.utcnow()
    await db.commit()

    before = len(await active_memories(db, user_a))
    dedup = MemoryDedupService(db)
    await dedup.smart_create_memories(
        new_memories=[
            MemoryCreate(
                content="The user follows a vegetarian diet.",
                brain_type=BrainType.USER,
                category="preferences",
                memory_type=MemoryType("preference"),
                importance=0.7,
                confidence=0.9,
                memory_level=MemoryLevel.EPISODIC,
                emotional_salience=0.5,
                source_type="conversation",
            )
        ],
        user_id=user_a,
    )
    await db.commit()
    rows = await active_memories(db, user_a)
    assert len(rows) == before, (
        "the BATCH path stored a duplicate the single path would have caught:\n"
        + "\n".join(f"  - {r.content}" for r in rows)
    )


async def test_restating_a_fact_across_turns_does_not_pile_up(db, user_a):
    for msg, reply in [
        ("I'm a vegetarian.", "Noted."),
        ("Just so you know, I don't eat meat.", "Understood."),
        ("I should mention I'm vegetarian.", "Already noted."),
        ("Remember I follow a vegetarian diet.", "I have that."),
    ]:
        await drive_turn(db, user_a, msg, reply)

    veg = [
        c for c in await active_contents(db, user_a)
        if "vegetarian" in c.lower() or "meat" in c.lower()
    ]
    record_metric("dedup_rows_after_4_restatements", len(veg))
    assert len(veg) <= 1, (
        f"{len(veg)} rows for one fact:\n" + "\n".join(f"  - {c}" for c in veg)
    )


async def test_distinct_facts_are_not_collapsed(db, user_a):
    """Dedup must not be over-eager: two facts that share a sentence shape but
    differ in subject are different facts (the Priya/Marco case)."""
    await store_direct(
        db, user_a,
        "The user's colleague Priya's desk door code is 4471.",
        category="people",
    )
    await store_direct(
        db, user_a,
        "The user's colleague Marco uses door code 8823 for his desk.",
        category="people",
    )
    contents = await active_contents(db, user_a)
    joined = " | ".join(contents)
    assert "4471" in joined and "8823" in joined, (
        f"two distinct people's codes collapsed into one row: {contents}"
    )


async def test_a_fact_arriving_between_two_like_it_destroys_neither(db):
    """A fact whose neighbours STRADDLE it must not retire either of them.

    This is the deterministic form of the silent loss that
    test_i_concurrency.py::test_rapid_fire_writes_lose_nothing catches by
    accident, and it took measuring to find the right shape — writing the same
    facts in order does NOT reproduce it. What matters is not how many
    same-shaped facts exist but WHERE the incoming one sits among the
    candidates it is adjudicated against (gpt-4o-mini, 40 trials per cell,
    2026-08-08; "destructive" = the applied verdict removed a fact):

        incoming 04, candidates {05, 03}  straddled   -> 70%
        incoming 04, candidates {03, 02}  both below  ->  0%
        incoming 02, candidates {00, 01}  sequential  ->  0%
        incoming 03, candidates {00,01,02}            ->  0%

    Straddled, the model reads "project 04 … vireo-04" as project 05's value
    having changed and answers contradiction_update, which RETIRES the
    existing row (is_active=False). Nothing raises; the fact is simply gone.
    Widen the batch and the misfire fades (5 candidates -> 2%), so the
    exposure is greatest when only a couple of rows exist — which is exactly
    what racing writes produce, and why this needed concurrency to show up at
    all.

    Each triple gets its OWN user: sharing a store would let earlier triples
    appear as extra candidates, widening the batch out of the cell under test
    and quietly making this vacuous. Four triples because one reproduces the
    pre-fix failure about half the time — four make a regression ~94% likely
    to be caught rather than a coin toss.
    """
    from .conftest import make_user

    lost = []
    for stem in ("vireo", "kestrel", "marlin", "ibis"):
        uid = await make_user(db, email=f"{stem}-{uuid.uuid4().hex[:8]}@memverify.local")
        fact = lambda n: f"The user's project number {n} is codenamed {stem}-{n}."

        # Order matters: the two outer facts must be stored BEFORE the middle
        # one, or there is no straddle to adjudicate against.
        for n in ("03", "05", "04"):
            await store_direct(db, uid, fact(n), category="work")

        contents = " | ".join(await active_contents(db, uid))
        lost += [f"{stem}-{n}" for n in ("03", "04", "05")
                 if f"{stem}-{n}" not in contents]

    record_metric("straddled_facts_lost", len(lost))
    assert not lost, (
        f"{len(lost)} of 12 facts were destroyed by a neighbour arriving "
        f"between them: {lost}"
    )


async def test_store_stays_bounded_over_many_sessions(db, user_a):
    """50 sessions restating a small set of facts must not produce 50 rows."""
    from app.db.database import async_session_maker

    facts = [
        "The user is a vegetarian.",
        "The user's dog is named Kesh.",
        "The user lives in Toronto.",
    ]
    for i in range(50):
        async with async_session_maker() as s:
            await store_direct(s, user_a, facts[i % len(facts)], category="preferences")

    rows = await active_memories(db, user_a)
    record_metric("rows_after_50_sessions", len(rows))
    assert len(rows) <= len(facts) + 1, (
        f"{len(rows)} rows after 50 sessions of {len(facts)} facts:\n"
        + "\n".join(f"  - {r.content}" for r in rows)
    )
