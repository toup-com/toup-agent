"""Four filters that were quietly deleting the user's memories on the read path.

Every one of these is the same shape: a predicate ANDed onto retrieval that
looked reasonable in isolation and, in production, returned a fraction of what
the user actually had.  Measured on a live tenant on 2026-08-07 at production
settings (limit=10, min_similarity=0.35):

1. TEMPORAL WINDOW — `extract_temporal_filters` fires on any month name or
   "today"/"this week"/"yesterday", and hybrid_search ANDed the resulting
   `created_at` window onto the shared `conditions` list that EVERY strategy
   inherits.  `created_at` is when the row was WRITTEN; it is not what the
   memory is about.

       what am I doing today?         1 of 10 memories reachable
       what happened yesterday?       0 of 10
       what did we decide in March?   0 of 10
       my birthday is in June         1 of 10
       -> 59 of 60 hidden (98.3%)

2. CATEGORY FILTER — `search_memories` did `[c.value for c in
   request.categories]`, but `MemorySearchRequest.categories` is
   `Optional[List[str]]`, so every category-filtered search raised
   AttributeError.  The `brain_type` branch six lines above already had the
   `hasattr` guard.

3. CORE FACTS — active_task rows are written at importance 0.9, above the 0.7
   floor of the every-turn "Core facts about this user" block, and carried
   `memory_type="semantic"` — which is a MemoryLevel, not a MemoryType — so the
   `notin_(["event","conversation","task","file"])` exclusion never matched
   them.  Every live reminder was asserted as a permanent fact about the user.

4. TTL — `get_core_facts` filtered on `expires_at`; `hybrid_search` did not.
   The sweep that would archive expired rows is behind a flag that is not
   enabled in production, so an expired memory stayed retrievable indefinitely.
   A TTL that only one read path enforces is not a TTL.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


pytestmark = pytest.mark.asyncio


# ── Harness ───────────────────────────────────────────────────────────

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


class _NoEmbedding:
    """EmbeddingService is a process-wide singleton — rebind, never patch it."""

    async def embed_async(self, *_a, **_kw):
        raise RuntimeError("embedding disabled in this test")

    def embed(self, *_a, **_kw):
        raise RuntimeError("embedding disabled in this test")


class _FixedEmbedding:
    """Deterministic vector, so search_memories' pgvector branch is exercised
    without a network call.  On sqlite there is no vector column, so the
    service falls back to its embedding_json path."""

    async def embed_async(self, *_a, **_kw):
        return [0.1] * 1536

    def embed(self, *_a, **_kw):
        return [0.1] * 1536

    def cosine_similarity(self, _a, _b):
        return 0.9


def _service(db, embedder=None):
    from app.services.memory_service import MemoryService

    svc = MemoryService(db)
    svc.embedding_service = embedder or _NoEmbedding()
    return svc


async def _user(maker) -> str:
    from app.db.models.user import User

    user_id = str(uuid.uuid4())
    async with maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"filters-{user_id[:8]}@example.invalid",
                hashed_password="not-a-real-hash",
                name="Filter Probe",
            )
        )
        await db.commit()
    return user_id


async def _add(maker, user_id, **kw):
    from app.db.models.memory import Memory

    defaults = dict(
        id=str(uuid.uuid4()),
        user_id=user_id,
        brain_type="user",
        category="preference",
        memory_type="fact",
        importance=0.5,
        strength=1.0,
    )
    defaults.update(kw)
    async with maker() as db:
        db.add(Memory(**defaults))
        await db.commit()
    return defaults["id"]


# ── 1. A time word must steer retrieval, not amputate it ──────────────

# The seeded content deliberately contains every time word the queries below
# use. Both keyword backends AND their terms together (Postgres via
# plainto_tsquery, sqlite via the ILIKE fallback's per-keyword conjunction), so
# a term absent from the content would fail the match for reasons that have
# nothing to do with this bug. With the words present, the ONLY thing that can
# hide the row is the created_at window — which is exactly the variable under
# test.
_TIME_WORDY = (
    "The user owns a widget — noted today, this week, in March, in June, "
    "and yesterday"
)


@pytest.mark.parametrize(
    "query",
    [
        "widget today",
        "widget this week",
        "widget in March",
        "widget in June",
        "widget yesterday",
    ],
)
async def test_a_time_word_does_not_hide_older_memories(session_maker, query):
    """A memory written 90 days ago must still be reachable.

    Pre-fix every one of these returned 0 rows: the auto-detected window was
    ANDed onto `conditions`, which every strategy inherits, and the only row in
    the database was written long before the window opened.
    """
    user_id = await _user(session_maker)
    old = datetime.utcnow() - timedelta(days=90)
    await _add(
        session_maker,
        user_id,
        content=_TIME_WORDY,
        summary="widget",
        created_at=old,
        updated_at=old,
    )

    async with session_maker() as db:
        rows = await _service(db).hybrid_search(
            user_id=user_id,
            query=query,
            limit=10,
            min_similarity=0.1,
            strategies=["keyword"],
        )

    assert rows, (
        f"{query!r} returned nothing. A memory written 90 days ago is invisible "
        "because the time word in the question was turned into a created_at "
        "filter — but created_at is when the row was WRITTEN, not what the "
        "memory is about."
    )


async def test_an_explicit_created_after_is_still_a_hard_filter(session_maker):
    """Caller-supplied date bounds are an API contract and must still bind.

    Only the *inferred* window was demoted to ranking; `created_after=` from the
    caller means "memories I recorded since then" and stays a real filter.
    """
    user_id = await _user(session_maker)
    old = datetime.utcnow() - timedelta(days=90)
    await _add(
        session_maker,
        user_id,
        content="The user owns a widget",
        summary="widget",
        created_at=old,
        updated_at=old,
    )

    async with session_maker() as db:
        rows = await _service(db).hybrid_search(
            user_id=user_id,
            query="widget",
            limit=10,
            min_similarity=0.1,
            strategies=["keyword"],
            created_after=datetime.utcnow() - timedelta(days=7),
        )

    assert rows == [], (
        "an explicit created_after was ignored — that bound is a caller "
        "contract, not an inference from prose"
    )


# ── 2. Category-filtered search must not raise ────────────────────────

async def test_a_category_filter_does_not_raise(session_maker):
    """`categories` arrives as plain strings; `c.value` used to blow up.

    Pre-fix: AttributeError: 'str' object has no attribute 'value' on every
    category-filtered search — REST /api/memories/search, /api/agent, and the
    MCP memory tools all reach this line.
    """
    from app.schemas import MemorySearchRequest

    user_id = await _user(session_maker)
    await _add(
        session_maker,
        user_id,
        content="The user prefers dark roast coffee",
        summary="coffee",
        category="preferences",
        embedding_json="[0.1]",
    )

    async with session_maker() as db:
        results, total, _ms = await _service(db, _FixedEmbedding()).search_memories(
            user_id,
            MemorySearchRequest(query="coffee", categories=["preferences"]),
        )

    assert isinstance(total, int)
    assert isinstance(results, list)


async def test_a_retired_category_alias_is_normalised_into_the_filter(session_maker):
    """`places` was renamed `locations`. The filter ANDs, so an un-normalised
    alias returns zero rows rather than degrading.

    Asserted on the compiled predicate rather than on returned rows:
    ``search_memories`` runs the pgvector branch, which cannot execute on
    sqlite, so an end-to-end row assertion would only ever be exercised in the
    Postgres CI job and would silently pass as vacuous in the sqlite sweep.
    """
    from app.schemas import MemorySearchRequest

    user_id = await _user(session_maker)
    seen: list[str] = []

    async with session_maker() as db:
        svc = _service(db, _FixedEmbedding())
        real_execute = db.execute

        async def _spy(stmt, *a, **kw):
            # An IN (...) stays a POSTCOMPILE placeholder in the SQL text, so
            # the values live in the compiled params, not the string.
            try:
                compiled = stmt.compile()
                seen.append(f"{compiled}\n-- params: {compiled.params}")
            except Exception:
                seen.append(str(stmt))
            return await real_execute(stmt, *a, **kw)

        svc.db = type("SpyDB", (), {
            "execute": staticmethod(_spy),
            "__getattr__": staticmethod(lambda n: getattr(db, n)),
        })()
        await svc.search_memories(
            user_id, MemorySearchRequest(query="Toronto", categories=["places"])
        )

    sql = "\n".join(seen)
    assert "locations" in sql, (
        "filtering by the retired alias 'places' did not normalise to "
        f"'locations' — a row stored under the canonical value can never "
        f"match.\nSQL seen:\n{sql}"
    )


# ── 3. Core facts is the standing portrait, not the working set ───────

async def test_an_active_task_is_not_a_core_fact(session_maker):
    """A live reminder must not be asserted as a permanent fact every turn.

    active_task rows are written at importance 0.9 — above this channel's 0.7
    floor — and rows already on the fleet carry the off-vocabulary
    memory_type="semantic", so excluding by memory_type alone is not enough.
    """
    user_id = await _user(session_maker)
    await _add(
        session_maker,
        user_id,
        content="The user asked to be reminded to move the car",
        summary="move the car",
        category="active_task",
        memory_type="semantic",  # what the fleet already has on disk
        importance=0.9,
        memory_level="working",
    )
    keeper = await _add(
        session_maker,
        user_id,
        content="The user is severely allergic to peanuts",
        summary="peanut allergy",
        category="health",
        importance=0.95,
    )

    async with session_maker() as db:
        facts = await _service(db).get_core_facts(user_id)

    ids = {f["id"] for f in facts}
    assert keeper in ids, "a genuine standing fact was dropped from core facts"
    assert not any(f["content"].startswith("The user asked to be reminded") for f in facts), (
        "an active_task row reached 'Core facts about this user' — that block is "
        "the standing portrait, injected on every turn, not the working set"
    )


# `test_active_task_writes_use_a_real_memory_type` RETIRED with
# `active_task_service` (v3 §1.1). It pinned that the active-task producer
# did not write the MemoryLevel value "semantic" into the MemoryType column.
# There is no active-task producer, no `working` file and no per-row type in
# v3; the confusion it guarded against cannot recur because neither column
# is on the write path.


# ── 4. An expired memory must not come back ───────────────────────────

async def test_an_expired_memory_is_not_retrievable(session_maker):
    """hybrid_search must honour the lease the write side granted.

    The maintenance sweep that archives expired rows is behind a flag that is
    not enabled in production, so this predicate is the only thing standing
    between an expired transient memory and the turn prompt.
    """
    user_id = await _user(session_maker)
    await _add(
        session_maker,
        user_id,
        content="The user is waiting on the widget delivery",
        summary="widget delivery",
        expires_at=datetime.utcnow() - timedelta(days=1),
    )
    live = await _add(
        session_maker,
        user_id,
        content="The user collects widget memorabilia",
        summary="widget memorabilia",
        expires_at=datetime.utcnow() + timedelta(days=1),
    )

    async with session_maker() as db:
        rows = await _service(db).hybrid_search(
            user_id=user_id, query="widget", limit=10,
            min_similarity=0.1, strategies=["keyword"],
        )

    ids = {r["id"] for r in rows}
    assert live in ids, "a memory whose lease has NOT expired was dropped"
    assert not any("delivery" in r["content"] for r in rows), (
        "an expired memory was returned by hybrid_search — get_core_facts has "
        "always filtered expires_at and this path did not"
    )


# ── 5. The ingest surface must apply the FULL gate ────────────────────

def test_the_ingest_surface_extracts_nothing_at_all():
    """STRONGER than the gate-ordering rule it replaces.

    This used to assert that BOTH `/ingest` extraction sites ran
    `memory_gate_reason` before `create_memory`, and it checked the ORDER —
    the right thing to check while those sites existed. They were the sixth
    memory writer: the rule-based extractor plus a direct `create_memory`,
    with `extract_memories` defaulting to True on a router mounted by both
    entrypoints, still minting `MemoryCategory.ACTIVE_TASK` rows after that
    category's whole surface was deleted.

    v3 severed them (docs/memory/rebuild-2026-08-v3.md §2.1), so "the
    extraction sites are gated" is no longer the invariant — "there are no
    extraction sites" is, and it cannot be satisfied by a gate someone later
    reorders or removes. The MESSAGE STORAGE half survives and is a live
    control, pinned in tests/test_ingestion_routes_to_tenant.py.

    Asserted on the AST, because the module now carries a long comment
    explaining what was severed and a grep would match its own explanation.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("app/api/ingest.py").read_text())

    forbidden = {"extract_memories", "create_memory", "smart_create_memory"}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name in forbidden:
            offenders.append(f"line {node.lineno}: {name}()")
        # Constructing the row model directly is the other way in.
        if isinstance(func, ast.Name) and func.id == "Memory":
            offenders.append(f"line {node.lineno}: Memory(...)")
    assert not offenders, (
        "app/api/ingest.py writes memory again — it was the SIXTH writer and "
        "was severed in v3; the curator is the one writer: " + "; ".join(offenders)
    )

    # Anti-vacuity: the file must still BE the ingest route. Deleting the
    # module would satisfy every assertion above.
    handlers = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    assert {"ingest_message", "ingest_conversation"} <= handlers, (
        "the ingest handlers are gone — this test now proves nothing"
    )


# ── 6. Every write surface honours the never-store tier ───────────────

def test_every_memory_write_surface_screens_never_store_values():
    """create / update / merge must all refuse a card number.

    create_memory has always had the storage backstop. update_memory reached
    the same column with nothing, and POST /{id}/merge took its content as a
    QUERY PARAMETER and called MemoryDedupService._merge_memories directly —
    which UPDATES an existing row, so it passed through neither the full gate
    nor create_memory's backstop. A value refused at create was accepted by
    editing or merging instead.
    """
    import ast
    import pathlib

    svc = pathlib.Path("app/services/memory_service.py").read_text()
    api = pathlib.Path("app/api/memories.py").read_text()

    for src, name in ((svc, "memory_service.py"), (api, "memories.py")):
        tree = ast.parse(src)
        fns = {
            n.name: ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        if name == "memory_service.py":
            for fn in ("create_memory", "update_memory"):
                assert "sensitive_content_reason" in fns[fn], (
                    f"MemoryService.{fn} writes content with no never-store screen"
                )
        else:
            # v3 §4: `POST /memories/{id}/merge` — which wrote an unscreened
            # QUERY PARAMETER into memory content — is gone with the row
            # surface. Every v3 write is an ops batch, and the never-store
            # screen moved to the one place a bullet can be born:
            # `memory_file_ops.validate_ops` → `_secret_problem`. Asserted
            # there rather than here, because "the route that had the hole
            # no longer exists" is only half the guarantee.
            assert "merge_into_memory" not in fns
            ops = pathlib.Path("app/services/memory_file_ops.py").read_text()
            assert "sensitive_content_reason" in ops, (
                "the v3 ops engine writes bullets with no never-store screen"
            )
