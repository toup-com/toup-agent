"""Every retrieval leg must honour the caller's filters, not just ownership.

THE DEFECT

`hybrid_search` builds one `conditions` list — ownership and liveness, plus
`brain_types`, `categories` and `min_strength` — and runs four legs
concurrently. Two of them took that list; two did not:

    _vector_search(user_id, emb, limit, conditions, ...)   <- filtered
    _keyword_search(user_id, query, limit, conditions)     <- filtered
    _graph_search(user_id, query, limit)                   <- NOT filtered
    _temporal_search(user_id, query, limit, after, before) <- NOT filtered

The graph and temporal legs re-derived their own predicate list and hardcoded
`user_id / is_deleted / is_active`. Everything else was silently dropped.

WHY IT MATTERS, concretely

`agent_runner` runs a dedicated agent-brain query with `brain_types=["agent"]`
and renders whatever comes back under:

    "## How this user wants you to work
     (learned from their corrections — follow these)"

A USER-brain row reaching that section is a plain fact about the user being
presented to the model as a behavioural rule it should obey. "The user's
storage locker is unit 14" becomes an instruction.

The `categories` bypass is the same shape: `query_classifier` ANDs a category
filter on precisely so an unrelated class cannot answer a query, and a leg
that ignores it defeats that with no signal.

Run:
    cd backend && RUN_MODE=agent PYTHONPATH=. \
        pytest tests/test_retrieval_legs_honor_filters.py
"""
from __future__ import annotations

import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio

from app.db.models.memory import Memory
from app.services.memory_service import MemoryService


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db_session(requires_agent_tables) -> AsyncIterator:
    """A session against the AGENT_ONLY `memories` table.

    `requires_agent_tables` skips the DB-backed cases when the suite is run
    against a database without the agent schema, which is what the sqlite CI
    job does. The two wiring tests below take no session and therefore run
    everywhere — they are the ones that pin the actual regression.
    """
    from app.db.database import async_session_maker

    async with async_session_maker() as session:
        yield session


async def _user(db) -> str:
    """Create a User row and return its id.

    `memories.user_id` carries a real FOREIGN KEY to `users`. SQLite does not
    enforce it by default and the DB-backed cases here SKIP on SQLite, so an
    unparented user_id looks fine in CI and fails only on Postgres with
    `ForeignKeyViolationError`. Seed the parent row.
    """
    from app.db import User

    user_id = str(uuid.uuid4())
    db.add(User(
        id=user_id,
        email=f"legs-{user_id[:8]}@example.com",
        hashed_password="x",
        name="Retrieval Legs Test",
    ))
    await db.commit()
    return user_id


def _mem(user_id: str, **kw) -> Memory:
    """A minimally-populated memory row."""
    base = dict(
        id=str(uuid.uuid4()),
        user_id=user_id,
        content="reference content",
        category="knowledge",
        memory_type="fact",
        brain_type="user",
        importance=0.5,
        strength=1.0,
        emotional_salience=0.0,
        is_active=True,
        is_deleted=False,
    )
    base.update(kw)
    return Memory(**{k: v for k, v in base.items()})


async def _seed(db, user_id: str) -> dict:
    """One agent-brain row and one user-brain row, otherwise identical."""
    agent_row = _mem(user_id, brain_type="agent", content="Prefer terse answers.")
    user_row = _mem(user_id, brain_type="user", content="Storage locker is unit 14.")
    db.add_all([agent_row, user_row])
    await db.commit()
    return {"agent": agent_row.id, "user": user_row.id}


async def test_temporal_leg_honors_brain_type(db_session):
    """MUTATION: drop `conditions` from _temporal_search's signature or stop
    threading it at the call site and this goes red — the user-brain row
    comes back for an agent-brain query."""
    user_id = await _user(db_session)
    ids = await _seed(db_session, user_id)
    svc = MemoryService(db_session)

    scoped = [
        Memory.user_id == user_id,
        Memory.is_deleted == False,  # noqa: E712
        Memory.is_active == True,  # noqa: E712
        Memory.brain_type == "agent",
    ]
    got = {mid for mid, _ in await svc._temporal_search(
        user_id, "what happened recently", 20, None, None, scoped,
    )}

    assert ids["agent"] in got, "the agent-brain row should still be reachable"
    assert ids["user"] not in got, (
        "a USER-brain row was returned for a brain_types=['agent'] query — "
        "agent_runner renders these as behavioural directives the model obeys"
    )


async def test_temporal_leg_honors_min_strength(db_session):
    """`min_strength` is the decay floor. A leg that ignores it resurrects
    memories the rest of the system considers faded."""
    user_id = await _user(db_session)
    strong = _mem(user_id, strength=0.9, content="strong")
    faded = _mem(user_id, strength=0.05, content="faded")
    db_session.add_all([strong, faded])
    await db_session.commit()
    svc = MemoryService(db_session)

    scoped = [
        Memory.user_id == user_id,
        Memory.is_deleted == False,  # noqa: E712
        Memory.is_active == True,  # noqa: E712
        Memory.strength >= 0.1,
    ]
    got = {mid for mid, _ in await svc._temporal_search(
        user_id, "recently", 20, None, None, scoped,
    )}

    assert strong.id in got
    assert faded.id not in got, "a below-floor memory came back from the temporal leg"


async def test_temporal_leg_still_scopes_to_the_user_without_conditions(db_session):
    """Back-compat + the security floor.

    Callers that pass no `conditions` must keep the old ownership/liveness
    behaviour — this leg must never become cross-tenant just because a caller
    omitted the argument.
    """
    mine = await _user(db_session)
    theirs = await _user(db_session)
    a = _mem(mine, content="mine")
    b = _mem(theirs, content="theirs")
    db_session.add_all([a, b])
    await db_session.commit()
    svc = MemoryService(db_session)

    got = {mid for mid, _ in await svc._temporal_search(mine, "recently", 20)}
    assert a.id in got
    assert b.id not in got, "temporal leg leaked another user's memory"


async def test_inactive_and_deleted_stay_excluded_on_the_temporal_leg(db_session):
    """Anti-vacuity control.

    If the filter list were being dropped entirely rather than replaced, the
    assertions above could pass for the wrong reason. Liveness is the one
    predicate that was always there — pin that it survives the change.
    """
    user_id = await _user(db_session)
    live = _mem(user_id, content="live")
    archived = _mem(user_id, content="archived", is_active=False)
    removed = _mem(user_id, content="removed", is_deleted=True)
    db_session.add_all([live, archived, removed])
    await db_session.commit()
    svc = MemoryService(db_session)

    got = {mid for mid, _ in await svc._temporal_search(user_id, "recently", 20)}
    assert live.id in got
    assert archived.id not in got
    assert removed.id not in got


async def test_graph_leg_accepts_and_applies_conditions(db_session):
    """The graph leg's own scope filter must use the caller's predicates.

    The leg only reaches its filter when the query names an entity that
    resolves to seeded rows, which needs the entity/link tables populated.
    Rather than build that fixture, assert the contract that made the bug
    possible: the parameter exists and the predicates reach the query.
    """
    import inspect

    sig = inspect.signature(MemoryService._graph_search)
    assert "conditions" in sig.parameters, (
        "_graph_search takes no `conditions` — hybrid_search's brain_type / "
        "category / min_strength filters are dropped on this leg"
    )

    src = inspect.getsource(MemoryService._graph_search)
    assert "*_scope" in src or "*conditions" in src, (
        "_graph_search accepts `conditions` but does not splat them into its "
        "scope query — the parameter would be inert"
    )


async def test_hybrid_search_threads_conditions_into_every_leg():
    """The wiring. All four legs must receive the same predicate list.

    This is the assertion that would have caught the original defect: the two
    filtered legs were correct in isolation, and the bug was entirely in which
    arguments the call site passed.
    """
    import inspect

    src = inspect.getsource(MemoryService.hybrid_search)
    for leg in ("_vector_search", "_keyword_search", "_graph_search", "_temporal_search"):
        idx = src.find(f"self.{leg}(")
        assert idx != -1, f"{leg} is no longer called from hybrid_search"
        call = src[idx: idx + 260]
        assert "conditions" in call, (
            f"{leg} is called without `conditions` — it will silently ignore "
            "brain_types, categories and min_strength"
        )
