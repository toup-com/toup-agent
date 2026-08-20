"""Regression: ``MemoryService.hybrid_search`` must honour the caller's ``limit``.

hybrid_search deliberately over-fetches so the re-ranker has candidates to
choose from — ``fetch_limit = limit * 4`` per strategy, then RRF keeps
``limit * 3``.  The trim back down to ``limit`` used to live ONLY inside the
re-rank branch and its ``except`` handler:

    if _settings.enable_reranker and len(scored_memories) > limit:
        scored_memories = await reranker.rerank(..., top_k=limit)
    except Exception:
        scored_memories = scored_memories[:limit]
    ...
    return scored_memories        # <- never trimmed when the `if` was False

``settings.enable_reranker`` defaults True but lives in
``app/agent/config_reload.py::RELOADABLE_FIELDS``, so a tenant can turn it off
at runtime with no deploy.  When they did, neither branch ran and hybrid_search
returned up to 3x the requested rows — which ``agent_runner`` injects straight
into the turn prompt.

These tests drive the real hybrid_search against a real database (sqlite by
default, Postgres when DATABASE_URL points at one — see ``_engine`` below).
They use ``strategies=["keyword"]`` and a stubbed embedder so no network and no
pgvector operator is required, which keeps the file portable across both CI
database jobs.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


pytestmark = pytest.mark.asyncio


# ── Fixtures ──────────────────────────────────────────────────────────

async def _engine():
    """Engine + sessionmaker with just the tables hybrid_search touches.

    `memories` is an AGENT_ONLY table, so conftest's platform-profile
    `init_db()` does not create it — same reason test_memory_taxonomy_and_ttl.py
    builds its own engine.  Schema comes from the live ORM so it cannot drift.

    Honours an ambient Postgres DATABASE_URL so the same file exercises real
    PG semantics in the pytest-postgres job and plain sqlite in the sweep.
    Note the Postgres tsvector trigger is NOT created here, so `search_vector`
    is NULL and `_keyword_search` takes its ILIKE fallback on both backends —
    which is exactly the leg these tests want to drive.
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


async def _seed(maker, n: int) -> str:
    """Create a user (real FK on Postgres) and `n` keyword-matching memories."""
    from app.db.models.memory import Memory
    from app.db.models.user import User

    user_id = str(uuid.uuid4())
    async with maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"limit-{user_id}@example.invalid",
                hashed_password="not-a-real-hash",
                name="Limit Probe",
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
                    # Distinct scores so the ordering assertions are meaningful.
                    importance=0.2 + (i / (n * 10.0)),
                    strength=1.0,
                )
            )
        await db.commit()
    return user_id


class _NoEmbedding:
    """Stand-in for EmbeddingService that always fails to embed.

    A separate object rather than a patched method: EmbeddingService is a
    process-wide singleton, so patching its methods leaks into every later test.
    """

    async def embed_async(self, *_a, **_kw):
        raise RuntimeError("embedding disabled in this test")

    def embed(self, *_a, **_kw):
        raise RuntimeError("embedding disabled in this test")


def _service(db):
    """MemoryService with the embedder stubbed out (no network, no pgvector).

    Returning None from the shared embed call is the documented degrade path:
    hybrid_search logs and skips the vector strategy.  We ask for keyword only
    anyway, but this also short-circuits the similarity_map query, which would
    otherwise need the pgvector cosine operator.
    """
    from app.services.memory_service import MemoryService

    svc = MemoryService(db)
    # REBIND the attribute; do not mutate what it points at. EmbeddingService
    # is a real singleton (`__new__` caches `_instance`), so
    # `svc.embedding_service.embed_async = ...` poisoned the shared object for
    # every test that ran afterwards in the same process — this file used to
    # break test_memory_taxonomy_and_ttl.py two files later, and only when the
    # two happened to be collected together.
    svc.embedding_service = _NoEmbedding()
    return svc


class _FakeReranker:
    """Stands in for RerankerService: records the call, truncates to top_k."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.received_ids: list[str] = []

    async def rerank(self, query, candidates, top_k=15):
        self.calls.append((query, len(candidates), top_k))
        self.received_ids = [c["id"] for c in candidates]
        # Reverse so a test can tell "the re-ranker's order survived" apart
        # from "we just re-used the score sort".
        ordered = list(reversed(candidates))
        return ordered[:top_k]


# ── 1. The regression ─────────────────────────────────────────────────

async def test_limit_is_honoured_when_reranker_is_disabled(session_maker, monkeypatch):
    """enable_reranker=False must not leak the over-fetch to the caller.

    Before the fix this returned 9 (`fused_ids` is capped at `limit * 3`) for
    limit=3 — 3x the requested prompt budget, every turn.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "enable_reranker", False, raising=False)
    user_id = await _seed(session_maker, 12)

    async with session_maker() as db:
        results = await _service(db).hybrid_search(
            user_id=user_id,
            query="widget",
            limit=3,
            strategies=["keyword"],
        )

    assert len(results) == 3, (
        f"hybrid_search returned {len(results)} rows for limit=3 — the "
        "over-fetch leaked to the caller"
    )


async def test_limit_is_honoured_when_reranker_is_disabled_at_default_limit(
    session_maker, monkeypatch
):
    """Same defect at the production limit (settings.memory_retrieval_limit=10).

    Guards against a fix that only holds for small limits, and pins that the
    top-5 auto-reinforce slice still sees a full five rows.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "enable_reranker", False, raising=False)
    user_id = await _seed(session_maker, 40)

    async with session_maker() as db:
        results = await _service(db).hybrid_search(
            user_id=user_id,
            query="widget",
            limit=10,
            strategies=["keyword"],
        )

    assert len(results) == 10


# ── 2. ANTI-VACUITY control ───────────────────────────────────────────

async def test_fewer_candidates_than_limit_are_all_returned(
    session_maker, monkeypatch
):
    """CONTROL: the trim must not become a fixed-size window.

    A "fix" that pads, truncates to a constant, or drops the tail would fail
    here.  This must stay GREEN under the mutation check.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "enable_reranker", False, raising=False)
    user_id = await _seed(session_maker, 2)

    async with session_maker() as db:
        results = await _service(db).hybrid_search(
            user_id=user_id,
            query="widget",
            limit=25,
            strategies=["keyword"],
        )

    assert len(results) == 2
    assert {r["content"] for r in results} == {
        "The user owns widget number 0",
        "The user owns widget number 1",
    }


async def test_top_of_the_ranking_is_kept_not_the_tail(session_maker, monkeypatch):
    """CONTROL: truncation keeps the highest-scoring rows, in order.

    `scored_memories` is sorted by `final_score` desc immediately above the
    trim; slicing the wrong end would still return `limit` rows and pass the
    length assertions.

    v3: the auto-reinforce loop this test used to neutralise is GONE with
    `decay_service`, so the two searches no longer interfere and there is
    nothing left to patch. The truncation itself is unchanged and still
    matters — `hybrid_search` remains the document/media leg's ranking.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "enable_reranker", False, raising=False)
    user_id = await _seed(session_maker, 12)

    async with session_maker() as db:
        svc = _service(db)
        wide = await svc.hybrid_search(
            user_id=user_id, query="widget", limit=36, strategies=["keyword"]
        )
    async with session_maker() as db:
        svc = _service(db)
        narrow = await svc.hybrid_search(
            user_id=user_id, query="widget", limit=6, strategies=["keyword"]
        )

    assert len(wide) == 12  # every seeded row — nothing to trim
    assert len(narrow) == 6
    assert [r["id"] for r in narrow] == [r["id"] for r in wide[:6]]


# ── 3. The re-ranker paths are unchanged ──────────────────────────────

async def test_reranker_enabled_still_reranks_and_returns_limit(
    session_maker, monkeypatch
):
    """enable_reranker=True: the re-ranker still runs and still decides order."""
    from app.config import settings
    from app.services import reranker_service as _rs

    fake = _FakeReranker()
    monkeypatch.setattr(settings, "enable_reranker", True, raising=False)
    monkeypatch.setattr(_rs, "get_reranker_service", lambda *a, **kw: fake)
    user_id = await _seed(session_maker, 12)

    async with session_maker() as db:
        results = await _service(db).hybrid_search(
            user_id=user_id,
            query="widget",
            limit=4,
            strategies=["keyword"],
        )

    assert fake.calls, "the re-ranker was not invoked"
    query, n_candidates, top_k = fake.calls[0]
    assert query == "widget"
    assert top_k == 4
    assert n_candidates > 4, "the re-ranker must still receive the over-fetch"
    assert len(results) == 4
    # The fake reverses its input; if the unconditional trim had re-sorted,
    # replaced, or sliced the re-ranked list from the wrong end, this exact
    # order would not survive.
    assert [r["id"] for r in results] == list(reversed(fake.received_ids))[:4]


async def test_reranker_failure_still_returns_limit(session_maker, monkeypatch):
    """The `except` branch no longer trims — the unconditional trim must cover it."""
    from app.config import settings
    from app.services import reranker_service as _rs

    class _Boom:
        async def rerank(self, *a, **kw):
            raise RuntimeError("cohere is on fire")

    monkeypatch.setattr(settings, "enable_reranker", True, raising=False)
    monkeypatch.setattr(_rs, "get_reranker_service", lambda *a, **kw: _Boom())
    user_id = await _seed(session_maker, 12)

    async with session_maker() as db:
        results = await _service(db).hybrid_search(
            user_id=user_id,
            query="widget",
            limit=3,
            strategies=["keyword"],
        )

    assert len(results) == 3
