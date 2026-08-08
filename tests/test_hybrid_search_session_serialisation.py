"""Regression: ``hybrid_search`` must never run two statements on one session at once.

THE BUG
-------
``hybrid_search`` fanned its retrieval strategies out with::

    results = await asyncio.gather(*task_coros, return_exceptions=True)

Every strategy executes on the same ``self.db``.  A SQLAlchemy ``AsyncSession``
is explicitly *not* safe for concurrent use, so the gathered coroutines raced to
provision the session's connection and all but the first died with::

    InvalidRequestError: This session is provisioning a new connection;
    concurrent operations are not permitted

Three of the four strategies caught their own exception and returned ``[]``, so
nothing failed — the search just came back short, with no log line.  Only
``_graph_search`` (no catch-all) ever surfaced, which is why production showed
"Strategy 'graph' failed" and nothing else.

WHO IT HIT
----------
The race only bites when the search is the *first* statement on the session.
Auto-recall (``agent_runner``) reuses the turn's warm session and was fine.
``tool_executor._tool_memory_search`` opens a fresh session per call, so the
agent's explicit memory lookup was degraded on **every** invocation.

Measured on a live tenant, production settings (limit=10, min_similarity=0.35),
12 realistic queries: cold session retrieved 11 memories, warm session 33 —
**66.7% of retrievable memories lost**.  "what is my name?" returned 0 cold, 5
warm.

WHY THESE TESTS DON'T NEED THE RACE
-----------------------------------
Reproducing the *provisioning* race needs a real async driver and a cold pool,
which is not portable across the sqlite and Postgres CI jobs.  So these tests
assert the invariant that actually matters and holds on every backend: while
one statement is in flight on the session, no second statement starts.  The
detector forces a suspension point inside ``execute`` so any concurrency is
observable, whether or not the driver itself would have yielded.

Run against the pre-fix code, ``test_strategies_never_overlap_on_the_session``
fails with overlaps > 0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


pytestmark = pytest.mark.asyncio


# ── Harness ───────────────────────────────────────────────────────────

async def _engine():
    """Engine with just the tables the retrieval strategies touch.

    `memories` is AGENT_ONLY, so conftest's platform-profile init_db() does not
    create it — same reason test_hybrid_search_honours_limit.py builds its own.
    Honours an ambient Postgres DATABASE_URL so this file runs on both CI jobs.
    """
    from app.db.models.base import Base
    from app.db.models.memory import Memory as _M, MemoryEvent as _ME
    from app.db.models.user import User as _U

    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgresql"):
        engine = create_async_engine(url)
    else:
        engine = create_async_engine(
            "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
        )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[_U.__table__, _M.__table__, _ME.__table__],
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def session_maker():
    engine, maker = await _engine()
    try:
        yield maker
    finally:
        await engine.dispose()


async def _seed(maker, n: int = 6) -> str:
    from app.db.models.memory import Memory
    from app.db.models.user import User

    user_id = str(uuid.uuid4())
    async with maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"serial-{user_id}@example.invalid",
                hashed_password="not-a-real-hash",
                name="Serialisation Probe",
            )
        )
        for i in range(n):
            db.add(
                Memory(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    brain_type="user",
                    content=f"The user owns widget number {i}",
                    summary=f"widget {i}",
                    category="preference",
                    memory_type="semantic",
                    importance=0.3 + i / 100.0,
                    strength=1.0,
                )
            )
        await db.commit()
    return user_id


class _OverlapDetector:
    """Wraps an AsyncSession and records overlapping ``execute`` calls.

    ``execute`` yields to the event loop *before* delegating, so a second call
    entering during that window is caught deterministically.  That is the whole
    point: it makes "these coroutines share a session" observable without
    depending on the driver's own suspension behaviour.
    """

    def __init__(self, session) -> None:
        self._session = session
        self._in_flight = 0
        self.overlaps = 0
        self.max_in_flight = 0
        self.calls = 0

    async def execute(self, *args, **kwargs):
        self._in_flight += 1
        self.calls += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        if self._in_flight > 1:
            self.overlaps += 1
        try:
            await asyncio.sleep(0)  # a real suspension point inside the statement
            return await self._session.execute(*args, **kwargs)
        finally:
            self._in_flight -= 1

    def __getattr__(self, name):
        return getattr(self._session, name)


class _NoEmbedding:
    """Stand-in for EmbeddingService that always fails to embed.

    A separate object rather than a patched method: EmbeddingService is a
    process-wide singleton (``__new__`` caches ``_instance``), so assigning to
    ``svc.embedding_service.embed_async`` poisons it for every later test in the
    same process.
    """

    async def embed_async(self, *_a, **_kw):
        raise RuntimeError("embedding disabled in this test")

    def embed(self, *_a, **_kw):
        raise RuntimeError("embedding disabled in this test")


def _service(db):
    """MemoryService with the embedder disabled (no network, no pgvector).

    Raising is the documented degrade path: hybrid_search logs, sets
    query_embedding=None and skips the vector strategy.  The keyword, graph and
    temporal legs still run, which is all these tests need.
    """
    from app.services.memory_service import MemoryService

    svc = MemoryService(db)
    svc.embedding_service = _NoEmbedding()
    return svc


# ── 1. The regression ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "strategies",
    [
        ["keyword", "graph"],
        ["keyword", "temporal"],
        ["keyword", "graph", "temporal"],
    ],
)
async def test_strategies_never_overlap_on_the_session(session_maker, strategies):
    """Two statements must never be in flight on one AsyncSession at once.

    Pre-fix this failed with overlaps >= 1 and max_in_flight == 2: asyncio.gather
    started every strategy before the first one's statement had returned.
    """
    user_id = await _seed(session_maker)

    async with session_maker() as db:
        detector = _OverlapDetector(db)
        await _service(detector).hybrid_search(
            user_id=user_id,
            query="widget yesterday",  # 'yesterday' also arms the temporal leg
            limit=5,
            min_similarity=0.1,
            strategies=strategies,
        )

    assert detector.calls > 1, (
        f"only {detector.calls} statement(s) ran for strategies={strategies} — "
        "the test did not exercise the fan-out it claims to"
    )
    assert detector.overlaps == 0, (
        f"hybrid_search ran {detector.overlaps} overlapping statement(s) on one "
        f"AsyncSession (max {detector.max_in_flight} in flight) for "
        f"strategies={strategies}. An AsyncSession is not safe for concurrent "
        "use: on a cold session all but the first leg dies with "
        "'This session is provisioning a new connection'."
    )


async def test_a_single_strategy_still_returns_results(session_maker):
    """Sanity: serialising the legs did not break ordinary retrieval."""
    user_id = await _seed(session_maker, 6)

    async with session_maker() as db:
        rows = await _service(db).hybrid_search(
            user_id=user_id, query="widget", limit=5, strategies=["keyword"]
        )

    assert rows, "keyword retrieval returned nothing after serialisation"
    assert len(rows) <= 5


# ── 2. The silent-failure half ────────────────────────────────────────

async def test_a_failing_strategy_is_logged_not_swallowed(session_maker, caplog):
    """A dead strategy must be visible in the log.

    An empty result is indistinguishable from "the user has no matching
    memories", so the log is the only place a degraded search is ever
    observable.  Before the fix, three of four strategies swallowed their
    exception into `return []` with no log line at all — which is precisely how
    a 66.7% recall loss ran in production unnoticed.
    """
    from app.services import memory_service as ms

    user_id = await _seed(session_maker, 4)

    async def _boom(*_a, **_kw):
        raise RuntimeError("simulated strategy failure")

    with caplog.at_level(logging.ERROR, logger=ms.__name__):
        async with session_maker() as db:
            svc = _service(db)
            svc._keyword_search = _boom
            await svc.hybrid_search(
                user_id=user_id, query="widget", limit=5, strategies=["keyword"]
            )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "keyword" in text, f"a dead strategy left no trace in the log:\n{text}"
    assert "DEGRADED" in text, (
        "hybrid_search did not report that the search it returned was "
        f"incomplete:\n{text}"
    )
