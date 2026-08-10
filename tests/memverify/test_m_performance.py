"""M. Performance budgets.

Budgets enforced here, and why:

  retrieval_db_p95 <= 150 ms   Retrieval with the embedding call stubbed out —
                               every real DB leg, no network. Enforced, because
                               this half is ours.
  retrieval_total_p95 <= 1.5 s An OUTAGE CEILING, not a budget: end-to-end
                               retrieval includes an OpenAI embedding round
                               trip whose p95 moved 267 -> 397 ms between two
                               runs on identical code and data.

  capture blocking = 0 ms   Capture runs in turn post-processing, AFTER the
                            reply is streamed (agent_runner.py:4637 is called
                            from the post-processing block, not before the
                            model call). It is therefore structurally
                            non-blocking; the test asserts the structure, not a
                            timing coincidence.

Every measurement is taken against a 500-row store — an empty store would
flatter the numbers and is not the case that matters.
"""

from __future__ import annotations

import inspect
import os
import random
import statistics
import time

import pytest

from .conftest import record_metric
from .pipeline import recall, seed_bulk, store_direct

# The DB-side budget is the one that is ours to keep: indexes, query plans, RRF
# fusion, row counts. Measured baseline at 500 rows is well under 100 ms.
RETRIEVAL_DB_P95_BUDGET_MS = float(os.environ.get("MEMVERIFY_DB_P95_BUDGET_MS", "150"))
# End-to-end includes an OpenAI embedding round trip. This ceiling is not a
# performance budget — it is an outage detector, set far above observed network
# variance (267-397 ms p95 across runs) so it fires on a real problem only.
RETRIEVAL_TOTAL_P95_CEILING_MS = float(
    os.environ.get("MEMVERIFY_TOTAL_P95_CEILING_MS", "1500")
)
STORE_SIZE = int(os.environ.get("MEMVERIFY_PERF_ROWS", "500"))

QUERIES = [
    "what is my dog's name?",
    "am I allergic to anything?",
    "where do I work?",
    "suggest somewhere to eat dinner tonight",
    "what car do I drive?",
    "who is my manager?",
    "convert 400 kelvin to celsius",
    "what languages do I speak?",
]


@pytest.fixture
async def populated(db, user_a):
    rng = random.Random(4242)
    topics = ["indexing", "kubernetes", "cycling", "espresso", "typography", "freight"]
    facts = [
        "The user's dog is named Kesh, a border collie.",
        "The user is severely allergic to peanuts.",
        "The user is the founder of Toup.",
        "The user drives a 2019 Subaru Outback.",
        "The user's manager is Priya.",
        "The user speaks Farsi, English and some Italian.",
    ]
    filler = [
        f"The user read an article about {rng.choice(topics)} numbered {i}."
        for i in range(STORE_SIZE - len(facts))
    ]
    await seed_bulk(db, user_a, facts + filler)
    return user_a


def _pct(sorted_samples, q):
    return sorted_samples[max(0, int(len(sorted_samples) * q) - 1)]


async def test_retrieval_latency_within_budget(db, populated, monkeypatch):
    """Budget what we control; measure what we do not.

    `hybrid_search` embeds the query before it touches the database, which is an
    OpenAI HTTPS round trip. A single "retrieval p95" therefore budgets someone
    else's network: measured 267 ms in one run and 397 ms with a 1540 ms outlier
    in the next, on identical code and identical data.

    Subtracting the two percentiles was the first attempt and it is not valid —
    they are independently sampled, and across six runs it produced DB estimates
    ranging from 6.9 ms to 384 ms for work that actually takes tens of ms.

    So the DB leg is measured DIRECTLY, by stubbing the embedding call with a
    pre-computed vector. That leaves every real database leg in place — pgvector
    cosine, tsvector keyword, the entity graph, RRF fusion — with the network
    removed. Then the stub is dropped and the real end-to-end path is timed for
    the outage ceiling.
    """
    from app.config import settings
    from app.services import embedding_service as es
    from app.services.embedding_service import get_embedding_service

    # The re-ranker stays IN this measurement, and the LLM fallback stays
    # OUT of the product — which is the honest version of the same idea.
    #
    # The first attempt at G-18 pinned `enable_reranker=False` here to make
    # the ceiling deterministic. That removed the only watchdog on the hop
    # that was actually blowing the budget: production was running the
    # gpt-4o-mini fallback at 2754-4054ms, and quieting the test is exactly
    # how that stayed invisible. Now the fallback is gated off at the
    # source (`reranker_llm_fallback_enabled`), so the remaining backend is
    # Cohere — which fits the budget — and this test measures the real
    # retrieval path again, ceiling included.
    monkeypatch.setattr(settings, "reranker_llm_fallback_enabled", False)

    embedder = get_embedding_service()
    await recall(db, populated, "warmup")  # warm pool, plans, HTTPS session

    # One real vector per query, so the stub returns something the index can
    # genuinely search on rather than a synthetic constant.
    vectors = {q: await embedder.embed_async(q) for q in QUERIES}

    real_embed = es.EmbeddingService.embed_async
    current = {"q": QUERIES[0]}

    async def _stub(self, text, *a, **k):
        return vectors[current["q"]]

    monkeypatch.setattr(es.EmbeddingService, "embed_async", _stub)
    db_samples = []
    for _ in range(5):
        for q in QUERIES:
            current["q"] = q
            t = time.perf_counter()
            await recall(db, populated, q)
            db_samples.append((time.perf_counter() - t) * 1000)
    monkeypatch.setattr(es.EmbeddingService, "embed_async", real_embed)

    total_samples = []
    for _ in range(2):
        for q in QUERIES:
            t = time.perf_counter()
            await recall(db, populated, q)
            total_samples.append((time.perf_counter() - t) * 1000)

    db_samples.sort()
    total_samples.sort()
    db_p50, db_p95 = statistics.median(db_samples), _pct(db_samples, 0.95)
    total_p50, total_p95 = statistics.median(total_samples), _pct(total_samples, 0.95)

    record_metric("retrieval_rows_in_store", STORE_SIZE)
    record_metric("retrieval_db_samples", len(db_samples))
    record_metric("retrieval_db_p50_ms", round(db_p50, 1))
    record_metric("retrieval_db_p95_ms", round(db_p95, 1))
    record_metric("retrieval_db_max_ms", round(db_samples[-1], 1))
    record_metric("retrieval_total_p50_ms", round(total_p50, 1))
    record_metric("retrieval_total_p95_ms", round(total_p95, 1))
    record_metric("retrieval_db_p95_budget_ms", RETRIEVAL_DB_P95_BUDGET_MS)
    record_metric("retrieval_total_p95_ceiling_ms", RETRIEVAL_TOTAL_P95_CEILING_MS)

    assert db_p95 <= RETRIEVAL_DB_P95_BUDGET_MS, (
        f"DB-side retrieval p95 {db_p95:.0f} ms exceeds the "
        f"{RETRIEVAL_DB_P95_BUDGET_MS:.0f} ms budget over {STORE_SIZE} rows "
        f"(p50 {db_p50:.0f} ms, max {db_samples[-1]:.0f} ms) — this measurement "
        "excludes the embedding call, so it is ours"
    )
    assert total_p95 <= RETRIEVAL_TOTAL_P95_CEILING_MS, (
        f"end-to-end retrieval p95 {total_p95:.0f} ms exceeds the "
        f"{RETRIEVAL_TOTAL_P95_CEILING_MS:.0f} ms outage ceiling"
    )


async def test_write_latency_recorded(db, user_a):
    samples = []
    for i in range(8):
        t = time.perf_counter()
        await store_direct(db, user_a, f"The user's project {i} is codenamed vireo-{i}.")
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    record_metric("write_p50_ms", round(statistics.median(samples), 1))
    record_metric("write_p95_ms", round(samples[-1], 1))
    # No hard budget: writes are off the user-visible path (see below).
    assert samples[-1] < 15000, f"a single memory write took {samples[-1]:.0f} ms"


def test_capture_is_structurally_off_the_reply_path():
    """Capture must add ~0 blocking time to the user-visible reply.

    Timing cannot prove this — a fast LLM call would pass a timing assertion
    while still being on the critical path. What proves it is the call ORDER in
    agent_runner.run(): `_extract_memories` must be invoked after the reply has
    been produced and persisted, in post-processing.
    """
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner.run)
    assert "_extract_memories" in src, (
        "capture is no longer called from run(); this test's premise is stale"
    )

    call_idx = src.index("_extract_memories")
    # Anchors that mark the reply having already been produced/persisted.
    anchors = ["_persist_turn", "assistant_response", "post_process", "_post_process"]
    seen = [a for a in anchors if a in src[:call_idx]]
    record_metric("capture_post_reply_anchors", seen)
    assert seen, (
        "no evidence that the reply is produced before _extract_memories is "
        "called — capture may be blocking the user-visible response"
    )
