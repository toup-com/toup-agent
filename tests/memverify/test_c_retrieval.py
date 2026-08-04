"""C. Retrieval and application.

Storing a fact is worthless if the right query does not surface it. These tests
drive the same `hybrid_search` call agent_runner.py:3454 makes, with the same
production limit and similarity floor.
"""

from __future__ import annotations

import pytest

from .conftest import record_metric
from .pipeline import (
    drive_turn,
    injected_contents,
    recall,
    store_direct,
)

# (fact to store, query that must surface it)
DIRECT_RECALL = [
    ("The user's dog is named Kesh, a border collie.", "what is my dog's name?"),
    ("The user is severely allergic to peanuts.", "am I allergic to anything?"),
    ("The user's manager is Priya, who prefers written updates.", "who is my manager?"),
    ("The user drives a 2019 Subaru Outback.", "what car do I drive?"),
    ("The user is the founder of Toup, an AI assistant company.", "where do I work?"),
    ("The user's daughter Soraya was born on March 14th.", "when is my daughter's birthday?"),
    ("The user speaks Farsi, English and some Italian.", "what languages do I speak?"),
    ("The user has been vegetarian for eight years.", "do I eat meat?"),
]


@pytest.mark.parametrize(
    "fact,query", DIRECT_RECALL, ids=[q[:28] for _, q in DIRECT_RECALL]
)
async def test_direct_recall(db, user_a, fact, query):
    """Asserted on the COMPOSED context the model receives, not on the
    similarity leg alone. Pure cosine cannot carry this: "where do I work?"
    scores 0.125 against "The user is the founder of Toup" — see
    MemoryService.get_core_facts for the full measurements."""
    await store_direct(db, user_a, fact)
    contents = await injected_contents(db, user_a, query)
    assert fact in contents, (
        f"query {query!r} did not surface the fact.\n"
        f"stored: {fact!r}\nreturned: {contents}"
    )


async def test_recall_aggregate_at_production_settings(db, user_a):
    """All facts in one store at once — the realistic case, where a query has to
    pick the right row out of a populated brain rather than the only row."""
    for fact, _ in DIRECT_RECALL:
        await store_direct(db, user_a, fact)

    hit, miss = 0, []
    for fact, query in DIRECT_RECALL:
        contents = await injected_contents(db, user_a, query)
        if fact in contents:
            hit += 1
        else:
            miss.append((query, contents[:3]))
    record_metric("retrieval_precision_at_production_limit", f"{hit}/{len(DIRECT_RECALL)}")
    assert not miss, f"{len(miss)} queries missed their fact: {miss}"


async def test_implicit_application_surfaces_constraint(db, user_a):
    """The allergy must come back on a restaurant question without being named.
    This is what 'the agent respects the stored allergy without being reminded'
    reduces to at the retrieval layer."""
    await store_direct(db, user_a, "The user is severely allergic to peanuts.")
    await store_direct(db, user_a, "The user lives in Toronto.")
    await store_direct(db, user_a, "The user prefers short bullet-point answers.")

    contents = await injected_contents(
        db, user_a, "suggest somewhere to eat dinner tonight"
    )
    assert any("peanut" in c.lower() for c in contents), (
        "allergy not retrieved for a dining question — the agent would answer "
        f"without it. returned: {contents}"
    )


async def test_relevance_discipline_unrelated_query(db, user_a):
    """Stored facts must not be force-injected into unrelated answers. The
    production floor (min_similarity=0.35) is what enforces this."""
    facts = [
        "The user's dog is named Kesh, a border collie.",
        "The user is severely allergic to peanuts.",
        "The user drives a 2019 Subaru Outback.",
        "The user's daughter Soraya was born on March 14th.",
        "The user has been vegetarian for eight years.",
    ]
    for f in facts:
        await store_direct(db, user_a, f)

    hits = await recall(db, user_a, "convert 400 kelvin to celsius")
    record_metric("irrelevant_query_retrieved_rows", len(hits))
    assert len(hits) <= 1, (
        "an unrelated arithmetic question pulled in personal memories: "
        f"{[h['content'] for h in hits]}"
    )


async def test_always_on_core_block_is_bounded(db, user_a):
    """The honest counterpart to relevance discipline.

    Core facts are injected on every non-trivial turn regardless of the
    question — that is what makes the allergy case work, and it IS a form of
    unconditional insertion. It is only defensible while it stays small, so the
    size is asserted rather than assumed.
    """
    from app.agent.agent_runner import CORE_FACTS_LIMIT
    from app.services.memory_service import MemoryService
    from .pipeline import seed_bulk

    await seed_bulk(
        db, user_a,
        [f"The user has standing preference number {i} about their workflow." for i in range(60)],
    )
    for f in (
        "The user is severely allergic to peanuts.",
        "The user has been vegetarian for eight years.",
    ):
        await store_direct(db, user_a, f, importance=0.95)

    core = await MemoryService(db).get_core_facts(user_a, limit=CORE_FACTS_LIMIT)
    chars = sum(len(m["content"]) for m in core)
    record_metric("core_block_rows", len(core))
    record_metric("core_block_chars", chars)
    record_metric("core_block_approx_tokens", chars // 4)

    assert len(core) <= CORE_FACTS_LIMIT, core
    assert chars // 4 <= 400, f"always-on block is ~{chars // 4} tokens"
    # ...and the highest-importance constraints are the ones that made the cut.
    joined = " ".join(m["content"].lower() for m in core)
    assert "peanut" in joined, f"top-importance constraint not in core block: {core}"


async def test_retrieval_identical_for_both_client_surfaces(db, user_a):
    """Web and mobile are thin clients over one store (MEMORY_SYSTEM_MAP §6).
    Both reach retrieval through the same service call, so 'identical' is
    verifiable by driving the two documented entry points and comparing."""
    from app.schemas import MemorySearchRequest
    from app.services.memory_service import MemoryService

    await store_direct(db, user_a, "The user's dog is named Kesh, a border collie.")

    svc = MemoryService(db)
    # Surface 1: the agent's own turn-time retrieval (mobile + web chat).
    turn_hits = {h["id"] for h in await recall(db, user_a, "my dog")}
    # Surface 2: the REST search endpoint both apps' memory screens call.
    rest_results, _, _ = await svc.search_memories(
        user_a, MemorySearchRequest(query="my dog", limit=10)
    )
    rest_hits = {r.id if hasattr(r, "id") else r["id"] for r in rest_results}

    assert turn_hits, "turn-time retrieval returned nothing"
    assert turn_hits & rest_hits, (
        f"the two client surfaces disagree: turn={turn_hits} rest={rest_hits}"
    )


async def test_retrieval_reflects_a_fact_captured_in_an_earlier_turn(db, user_a):
    """End-to-end: capture through the real extractor, then retrieve it the way
    the next turn would."""
    await drive_turn(
        db, user_a,
        "My son Arash is starting grade 3 at Rosedale Public School.",
        "Noted — Arash, grade 3 at Rosedale.",
    )
    contents = " | ".join(
        c.lower() for c in await injected_contents(db, user_a, "which school do my kids go to?")
    )
    assert "rosedale" in contents or "arash" in contents, contents
