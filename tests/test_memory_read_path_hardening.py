"""The rest of the read path, after the two big ones in #502.

#502 fixed the shared-session race and the temporal window — the two defects
that were deleting most of the user's recall. This is the remainder of the same
audit: smaller, but every one of them is the same shape, a failure or a filter
that changes the answer without saying so.

1. CANONICALISATION ON A FILTER IS NOT THE SAME FUNCTION AS ON A WRITE.
   `normalize_category` must always return something storable, so an unknown
   value falls back to `other` (user brain) or `domain_knowledge` (agent). On a
   filter that is a silent lie: `domain_knowledge` and `process` are REAL
   categories in the agent and work vocabularies, and the write-path function
   maps both to `other`. Filtering through it searches for a category the
   caller never asked for.

2. A pgvector FAILURE RETURNED AN EMPTY 200. `search_memories` caught the
   exception, set `scored_memories = []`, and fell through to a Python-side
   fallback guarded by `not use_pgvector` — which is False on every tenant. The
   fallback was unreachable and nothing was logged.

3. THE MERGE BRANCH IGNORED THE TTL LEASE. Reinforce and merge are the two
   outcomes of one dedup decision; only reinforce reconciled `expires_at`, so a
   transient memory restated as a durable fact kept its lease and was archived.

4. `memory_level` WAS ACCEPTED AND DISCARDED. The schema field is `memory_levels`
   (plural); the endpoint passed the singular name, so pydantic dropped it and
   the caller silently got everything.

5. A FAILED ALIAS LOOKUP WAS CACHED. `_user_aliases` cached `[]` from a failure
   for the life of the service instance, so no later call could recover.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


pytestmark = pytest.mark.asyncio


# ── 1. Filter-side canonicalisation ───────────────────────────────────

def test_a_filter_never_falls_back_to_other():
    """The write-path fallback would turn a real category into `other`."""
    from app.memory_taxonomy import canonical_category_for_filter, normalize_category

    # Real categories in the agent / work vocabularies.
    for value in ("domain_knowledge", "process", "tool_usage", "skills_learned"):
        assert normalize_category(value) == "other", (
            f"precondition changed: normalize_category({value!r}) no longer "
            "falls back, so this test is no longer testing anything"
        )
        assert canonical_category_for_filter(value) == value, (
            f"filtering on {value!r} was rewritten to "
            f"{canonical_category_for_filter(value)!r} — the query no longer "
            "means what the caller asked"
        )


def test_a_retired_alias_is_still_mapped_on_a_filter():
    """Not widening must not mean 'do nothing' — real aliases still map."""
    from app.memory_taxonomy import canonical_category_for_filter

    assert canonical_category_for_filter("places") == "locations"
    assert canonical_category_for_filter("family") == "people"
    assert canonical_category_for_filter("agent_tools", brain_type="agent") == "tool_usage"


def test_an_unknown_filter_value_is_left_alone():
    """An empty result is the honest answer; `other` rows are not."""
    from app.memory_taxonomy import canonical_category_for_filter

    assert canonical_category_for_filter("nonsense_xyz") == "nonsense_xyz"
    assert canonical_category_for_filter(None) is None


def test_every_category_filter_uses_the_filter_side_function():
    """Guard rail: a future filter must not reach for the write-path helper."""
    import ast
    import pathlib

    src = pathlib.Path("app/services/memory_service.py").read_text()
    tree = ast.parse(src)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        body = ast.unparse(node)
        if "Memory.category" not in body:
            continue
        # get_core_facts / _update_brain_stats filter on fixed literals.
        if node.name in ("get_core_facts", "_update_brain_stats"):
            continue
        if "normalize_category" in body and "canonical_category_for_filter" not in body:
            offenders.append(node.name)

    assert not offenders, (
        f"{offenders} filter on Memory.category through the WRITE-path "
        "normalizer. It falls back to `other`, so the query silently stops "
        "meaning what the caller asked for."
    )


# ── 2/3/5. Behavioural harness ────────────────────────────────────────

async def _engine():
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


class _Embedder:
    async def embed_async(self, *_a, **_kw):
        return [0.1] * 1536

    def embed(self, *_a, **_kw):
        return [0.1] * 1536

    def cosine_similarity(self, _a, _b):
        return 0.9


def _service(db):
    """EmbeddingService is a process-wide singleton — rebind, never patch it."""
    from app.services.memory_service import MemoryService

    svc = MemoryService(db)
    svc.embedding_service = _Embedder()
    return svc


async def _user(maker) -> str:
    from app.db.models.user import User

    uid = str(uuid.uuid4())
    async with maker() as db:
        db.add(
            User(
                id=uid,
                email=f"harden-{uid[:8]}@example.invalid",
                hashed_password="not-a-real-hash",
                name="Nariman Probe",
            )
        )
        await db.commit()
    return uid


# ── 2. A retrieval failure must be loud ───────────────────────────────

async def test_a_pgvector_failure_is_logged_and_does_not_return_an_empty_200(
    session_maker, caplog
):
    """Pre-fix: empty list, 200, no log — indistinguishable from 'no memories'."""
    from app.schemas import MemorySearchRequest
    from app.services import memory_service as ms

    uid = await _user(session_maker)

    with caplog.at_level(logging.ERROR, logger=ms.__name__):
        async with session_maker() as db:
            svc = _service(db)
            real_execute = db.execute
            calls = {"n": 0}

            async def _boom(stmt, *a, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("simulated pgvector failure")
                return await real_execute(stmt, *a, **kw)

            svc.db = type("SpyDB", (), {
                "execute": staticmethod(_boom),
                "__getattr__": staticmethod(lambda n: getattr(db, n)),
            })()
            await svc.search_memories(uid, MemorySearchRequest(query="anything"))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "search_memories" in text and "pgvector" in text, (
        f"a retrieval failure left no trace in the log:\n{text}"
    )


# ── 3. The merge branch must honour the lease ─────────────────────────

def test_merge_and_reinforce_both_reconcile_the_expiry_lease():
    """They are the two outcomes of one decision; only one honoured the TTL."""
    import ast
    import pathlib

    src = pathlib.Path("app/services/memory_dedup_service.py").read_text()
    tree = ast.parse(src)
    fns = {
        n.name: ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
    }

    for fn in ("_reinforce_existing_memory", "_merge_memories"):
        assert "expires_at" in fns[fn], (
            f"{fn} is a dedup outcome that never touches expires_at — a "
            "transient memory restated as durable keeps its lease and is "
            "archived on schedule"
        )


# ── 4. A filter the API accepts must actually bind ────────────────────

def test_memory_level_reaches_the_search_request():
    """The schema field is plural; the singular name was silently dropped."""
    from app.schemas import MemorySearchRequest, MemoryLevel

    # The bug, pinned: the singular name is not a field at all.
    r = MemorySearchRequest(query="x", memory_level="semantic")
    assert not hasattr(r, "memory_level")
    assert r.memory_levels is None

    # What the endpoint must build instead.
    r2 = MemorySearchRequest(query="x", memory_levels=[MemoryLevel("semantic")])
    assert r2.memory_levels == [MemoryLevel.SEMANTIC]


def test_the_search_endpoint_maps_memory_level_and_rejects_unknown_values():
    import ast
    import pathlib

    src = pathlib.Path("app/api/memories.py").read_text()
    tree = ast.parse(src)
    fn = next(
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        and "memory_level" in ast.unparse(n)
        and "MemorySearchRequest(" in ast.unparse(n)
    )
    assert "memory_levels=" in fn, (
        "the endpoint still passes the singular `memory_level=`, which pydantic "
        "drops — the caller's filter never binds"
    )
    assert "HTTP_422" in fn, (
        "an unknown memory_level should 422, not silently widen the search"
    )


# ── 5. A failed lookup must not be cached ─────────────────────────────

async def test_a_failed_alias_lookup_is_not_cached(session_maker, caplog):
    """Caching [] from a failure pinned the gate stricter for the whole turn."""
    from app.services import memory_service as ms

    uid = await _user(session_maker)

    async with session_maker() as db:
        svc = _service(db)
        real_execute = db.execute
        state = {"fail": True}

        async def _flaky(stmt, *a, **kw):
            if state["fail"]:
                raise RuntimeError("simulated lookup failure")
            return await real_execute(stmt, *a, **kw)

        svc.db = type("SpyDB", (), {
            "execute": staticmethod(_flaky),
            "__getattr__": staticmethod(lambda n: getattr(db, n)),
        })()

        with caplog.at_level(logging.WARNING, logger=ms.__name__):
            first = await svc._user_aliases(uid)
        assert first == []
        assert any("alias lookup failed" in r.getMessage() for r in caplog.records), (
            "the failure was invisible above DEBUG"
        )

        # The lookup now works. A cached failure would make this unrecoverable.
        state["fail"] = False
        second = await svc._user_aliases(uid)

    assert second, (
        "a recovered alias lookup still returned nothing — the empty result "
        "from the earlier failure was cached, so the gate stays stricter for "
        "the life of the service instance with no way back"
    )
    assert "Nariman" in second
