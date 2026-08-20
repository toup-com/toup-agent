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


def _tenant_ddl_for(marker: str) -> str:
    """Pull a DDL statement verbatim out of ``database.py``'s ALTER list.

    ``memories`` is AGENT_ONLY and its indexes are created by the ALTER
    mirror in ``app/db/database.py`` at boot — NOT by ``create_all()``,
    which only builds what the ORM models declare, and a PARTIAL unique
    index is not expressible there. So a test database gets the table
    without the constraint, and this test asserted an invariant its own
    database could not hold.

    Reading the statement out of the shipped source rather than copying
    it means the two cannot drift: edit the index in database.py and this
    test exercises the edited version, or fails loudly because the marker
    stopped matching.
    """
    import inspect

    from app.db import database as _db

    lines = inspect.getsource(_db).splitlines()
    idx = next(
        (i for i, ln in enumerate(lines) if marker in ln and "CREATE" in ln),
        None,
    )
    assert idx is not None, f"marker {marker!r} not found in database.py DDL"
    # The list literal wraps one statement across adjacent string
    # fragments; re-join the run that starts at the marker.
    parts, j = [], idx
    while j < len(lines):
        raw = lines[j].strip()
        if not raw.startswith('"'):
            break
        parts.append(raw.rstrip(",").strip('"'))
        if lines[j].rstrip().endswith(","):
            break
        j += 1
    assert parts, f"no DDL found for marker {marker!r} in database.py"
    return "".join(parts)


@pytest_asyncio.fixture
async def memories_ref_unique_index(requires_agent_tables):
    """Apply the real partial unique index to the test database.

    Without this the test SKIPPED in CI (``requires_agent_tables``
    short-circuits when the AGENT_ONLY tables are absent, which is every
    platform-lane run) and FAILED anywhere the tables did exist — so the
    "floor invariant" in this module's docstring had never once been
    verified. Production does carry the index; probed on the canary
    tenant 2026-08-10: ``ix_memories_ref_unique | UNIQUE | partial``.
    """
    from sqlalchemy import text

    from app.db.database import engine

    ddl = _tenant_ddl_for("ix_memories_ref_unique")
    async with engine.begin() as conn:
        try:
            await conn.execute(text(ddl))
        except Exception as e:  # pragma: no cover - dialect-specific
            pytest.skip(f"cannot create partial unique index here: {e}")
    yield


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
async def test_partial_unique_index_blocks_duplicate_ref(
    memory_user, memories_ref_unique_index
):
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
async def test_partial_unique_allows_different_ref_id(memory_user, requires_agent_tables):
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
async def test_partial_unique_allows_null_ref(memory_user, requires_agent_tables):
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


# ── RETIRED: the MCP `memory_create` enum guards (v3 §2.1.6) ──────────
#
# `test_mcp_memory_create_defaults_are_valid_enum_values` and
# `test_mcp_memory_create_rejects_layer_in_type_slot` pinned that the MCP
# tool validated `memory_type` / `memory_level` against their enums before
# the Pydantic round trip, with an error message that distinguished the two
# concepts so a model could not put "semantic" (a LEVEL) in the TYPE slot.
#
# The tool is deleted. v3's MCP write is `memory_remember(instruction)` — a
# sentence in plain language, routed to the curator, which chooses the file.
# There is no type, no level and no category for a model to conflate, which
# is a stronger guarantee than validating them was. Its replacement is
# pinned in tests/test_curator_producers.py::
# test_no_mcp_tool_proxies_a_deleted_route.
#
# The three DB-level tests ABOVE stay: `ix_memories_ref_unique` is still on
# the legacy table, which v3 leaves on disk untouched as the rollback.
