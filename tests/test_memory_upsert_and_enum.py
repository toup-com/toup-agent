"""Tickets 2 + 2.1 regression tests.

Locks three properties:

  1. The DB partial unique index `ix_memories_ref_unique` rejects a
     second active memory with the same (user_id, ref_kind, ref_id).
     This is the floor — even if a caller skips the MCP tool path, the
     constraint prevents the duplicate-routine-memory bug from
     reappearing.

  2. The MCP `memory_create` tool validates `memory_type` and
     `memory_level` against their respective enums BEFORE the
     round-trip to Pydantic. The error message explicitly distinguishes
     the two concepts so future LLM prompts can't conflate them.

  3. `memory_create` defaults are valid values in their respective
     Pydantic enums (cross-checked test). Invariant #3 from the bug
     sweep — tool schemas must agree with backend Pydantic models.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError


@pytest_asyncio.fixture
async def memory_user():
    """Create a User row. Returns the id."""
    from app.db import User, async_session_maker
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"mem-{user_id[:8]}@example.com",
            hashed_password="x",
            name="Memory Test",
        ))
        await db.commit()
    return user_id


@pytest.mark.asyncio
async def test_partial_unique_index_blocks_duplicate_ref(memory_user):
    """Floor invariant: two active memories with the same (user_id,
    ref_kind, ref_id) cannot coexist. This is what makes the Ticket 2
    upsert behavior correct — even if an MCP tool regression slips in,
    the DB stops the duplication."""
    from app.db import async_session_maker
    from app.db.models import Memory

    async with async_session_maker() as db:
        db.add(Memory(
            id=str(uuid.uuid4()),
            user_id=memory_user,
            content="routine fires at 10:43",
            category="schedule",
            memory_type="note",
            ref_kind="routine",
            ref_id="r-1",
        ))
        await db.commit()

    async with async_session_maker() as db:
        db.add(Memory(
            id=str(uuid.uuid4()),
            user_id=memory_user,
            content="routine fires at 10:58",
            category="schedule",
            memory_type="note",
            ref_kind="routine",
            ref_id="r-1",
        ))
        with pytest.raises(IntegrityError):
            await db.commit()


@pytest.mark.asyncio
async def test_partial_unique_allows_different_ref_id(memory_user):
    """The partial unique index is scoped to (user, ref_kind, ref_id).
    Two routines with different ref_ids must each get their own row."""
    from app.db import async_session_maker
    from app.db.models import Memory

    async with async_session_maker() as db:
        for rid in ("r-1", "r-2"):
            db.add(Memory(
                id=str(uuid.uuid4()),
                user_id=memory_user,
                content=f"memory for {rid}",
                category="schedule",
                memory_type="note",
                ref_kind="routine",
                ref_id=rid,
            ))
        await db.commit()

    async with async_session_maker() as db:
        count = (await db.execute(
            select(Memory).where(
                Memory.user_id == memory_user,
                Memory.ref_kind == "routine",
            )
        )).scalars().all()
        assert len(count) == 2


@pytest.mark.asyncio
async def test_partial_unique_allows_null_ref(memory_user):
    """Legacy memories (no ref_kind/ref_id) must coexist freely. The
    partial predicate `WHERE ref_id IS NOT NULL` excludes them."""
    from app.db import async_session_maker
    from app.db.models import Memory

    async with async_session_maker() as db:
        for i in range(3):
            db.add(Memory(
                id=str(uuid.uuid4()),
                user_id=memory_user,
                content=f"freeform memory {i}",
                category="context",
                memory_type="note",
            ))
        await db.commit()  # No conflict — null ref_id is unconstrained.


def test_mcp_memory_create_defaults_are_valid_enum_values():
    """Invariant #3 — the MCP tool's default values must be valid in
    the Pydantic enums they map to. The original bug: memory_type
    default was 'episodic', which is a MemoryLevel value, not a
    MemoryType value. The agent inherited that mis-default and
    propagated it.

    This test pins the agreement: introspect the tool signature, pull
    the defaults, and validate each against its enum.
    """
    import inspect
    from app import mcp_server
    from app.schemas import MemoryType, MemoryLevel

    sig = inspect.signature(mcp_server.memory_create.fn)
    params = sig.parameters

    assert params["memory_type"].default in [t.value for t in MemoryType], (
        f"memory_create default memory_type "
        f"({params['memory_type'].default!r}) is not a valid MemoryType "
        f"enum value. This is exactly the Ticket 2.1 drift."
    )
    assert params["memory_level"].default in [l.value for l in MemoryLevel], (
        f"memory_create default memory_level "
        f"({params['memory_level'].default!r}) is not a valid MemoryLevel "
        f"enum value."
    )


@pytest.mark.asyncio
async def test_mcp_memory_create_rejects_layer_in_type_slot():
    """When the agent passes a layer name (e.g. 'semantic') as
    `memory_type`, the tool must reject FAST with a useful error
    message distinguishing the two concepts. This is what prevents
    repeat of the 3.1s validation-error round-trip."""
    from app import mcp_server

    # Direct call to the underlying function bypasses MCP transport.
    result = await mcp_server.memory_create.fn(
        content="test",
        category="context",
        memory_type="semantic",  # WRONG SLOT — semantic is a level
    )
    assert "error" in result, result
    assert "memory_type" in result["error"]
    # Error must guide the LLM toward `memory_level`.
    assert "memory_level" in result["error"]
