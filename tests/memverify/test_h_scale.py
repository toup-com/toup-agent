"""H. Long-horizon and scale.

One user with a thousand facts must still retrieve the right one, and the
injected context must stay bounded.
"""

from __future__ import annotations

import random
import time

import pytest

from .conftest import make_user, record_metric
from .pipeline import active_memories, recall, seed_bulk

N_FACTS = int(__import__("os").environ.get("MEMVERIFY_SCALE_FACTS", "1000"))

NEEDLES = [
    ("The user's storage unit access code is heron-2288.", "what is my storage unit code?"),
    ("The user's daughter Soraya was born on March 14th.", "when is my daughter's birthday?"),
    ("The user is severely allergic to peanuts.", "am I allergic to anything?"),
    ("The user's co-founder is Zbigniew Wojciechowski.", "who is my co-founder?"),
    ("The user drives a 2019 Subaru Outback.", "what car do I drive?"),
]

FILLER_TEMPLATES = [
    "The user read an article about {t} on {d}.",
    "The user attended a meeting about {t} in week {d}.",
    "The user bookmarked a resource on {t} numbered {d}.",
    "The user mentioned an interest in {t} during session {d}.",
    "The user completed a task related to {t}, item {d}.",
]
TOPICS = [
    "database indexing", "kubernetes", "tax filing", "cycling routes",
    "espresso grinders", "typography", "supply chains", "mycology",
    "harbour logistics", "generative art", "ocean freight", "acoustics",
]


async def test_thousand_facts_still_retrieve_the_right_one(db, user_a):
    rng = random.Random(20260801)
    filler = [
        FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)].format(
            t=rng.choice(TOPICS), d=i
        )
        for i in range(N_FACTS - len(NEEDLES))
    ]
    t0 = time.perf_counter()
    await seed_bulk(db, user_a, [n for n, _ in NEEDLES] + filler)
    record_metric("scale_seed_rows", N_FACTS)
    record_metric("scale_seed_seconds", round(time.perf_counter() - t0, 1))

    rows = await active_memories(db, user_a)
    assert len(rows) >= N_FACTS * 0.9, f"only {len(rows)} of {N_FACTS} rows survived seeding"

    misses, latencies = [], []
    for needle, query in NEEDLES:
        t = time.perf_counter()
        hits = await recall(db, user_a, query)
        latencies.append((time.perf_counter() - t) * 1000)
        if not any(needle == h["content"] for h in hits):
            misses.append((query, [h["content"][:60] for h in hits[:3]]))

    record_metric("scale_retrieval_p95_ms", round(sorted(latencies)[-1], 1))
    assert not misses, f"needles lost in a {len(rows)}-row store: {misses}"


async def test_injected_context_stays_bounded_at_scale(db, user_a):
    """The prompt block is bounded by ROW COUNT, not tokens (GAP-8). Measure the
    worst realistic case so the bound is a number, not an assumption."""
    from app.config import settings

    rng = random.Random(7)
    await seed_bulk(
        db, user_a,
        [
            "The user is severely allergic to peanuts and must avoid all "
            "peanut-derived ingredients including peanut oil and satay sauces."
        ]
        + [
            FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)].format(t=rng.choice(TOPICS), d=i)
            for i in range(300)
        ],
    )

    hits = await recall(db, user_a, "suggest somewhere to eat dinner tonight")
    assert len(hits) <= settings.memory_retrieval_limit, len(hits)

    # The rendered block, exactly as agent_runner.py:3532-3553 builds it.
    core = [h for h in hits if h.get("strength", 0) >= 0.7][:5]
    rest = [h for h in hits if h not in core][:10]
    block = "\n".join(
        [f"- {m['content']}" for m in core]
        + [f"{i}. [{m.get('category','')}] {m['content']}" for i, m in enumerate(rest, 1)]
    )
    chars = len(block)
    approx_tokens = chars // 4
    record_metric("injected_block_chars", chars)
    record_metric("injected_block_approx_tokens", approx_tokens)
    assert approx_tokens <= 1500, (
        f"memory block is ~{approx_tokens} tokens — it would eat the turn budget"
    )


async def test_hundred_users_no_crosstalk_under_load(db):
    """100 concurrent synthetic users writing at once."""
    import asyncio

    from app.db.database import async_session_maker
    from .pipeline import store_direct

    n = int(__import__("os").environ.get("MEMVERIFY_LOAD_USERS", "100"))
    async with async_session_maker() as s:
        users = [
            (
                await make_user(s, name=f"Load {i:03d} Nobakht",
                                email=f"load{i}-{int(time.time()*1000)}@memverify.local"),
                f"loadtoken-{i:03d}-ibex",
            )
            for i in range(n)
        ]

    sem = asyncio.Semaphore(16)

    async def _write(uid, token):
        async with sem:
            async with async_session_maker() as s:
                await store_direct(s, uid, f"The user's access token is {token}.")

    t0 = time.perf_counter()
    await asyncio.gather(*(_write(u, t) for u, t in users))
    record_metric("load_users", n)
    record_metric("load_write_seconds", round(time.perf_counter() - t0, 1))

    all_tokens = {t for _, t in users}
    leaks, missing = [], []
    async with async_session_maker() as s:
        for uid, own in users:
            contents = " ".join(
                m.content for m in await active_memories(s, uid)
            )
            if own not in contents:
                missing.append(uid)
            for other in all_tokens - {own}:
                if other in contents:
                    leaks.append((uid, other))
    assert not missing, f"{len(missing)} users lost their write under load"
    assert not leaks, f"cross-user contamination under load: {leaks[:5]}"
