"""Sub-agent dispatcher unit tests.

Exercises the Phase 2 dispatcher in isolation against the test
database (autouse ``init_db`` fixture from conftest.py creates the
``build_jobs`` table from the ORM). The dispatcher itself is not
invoked from production code until Phase 4.

What we pin
-----------
- Rejection codes are a stable closed set; the strings the tool
  handler will quote are pinned by ``REJECTION_CODES``.
- Idempotency key derivation is content-hash-based and stable
  across calls.
- ``walk_parent_chain`` returns the chain including the start row,
  handles missing rows defensively, and is cycle-bounded.
- Each cap-count query counts the right rows under realistic
  fixtures.
- ``pre_spawn_checks`` runs the gates in the documented priority
  order; the kill switch beats everything else.
- ``cascade_cancel`` flips active descendants only, preserves
  terminal-state stamps, and recurses arbitrarily deep (bounded).
- ``transition_job_status`` is a chokepoint that:
    - preserves an existing terminal status (race-safe)
    - cascades on failure-shape transitions when propagate_cancel=True
    - does not cascade on successful transitions
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio


# ──────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────


def _new_id() -> str:
    return str(uuid.uuid4())


async def _seed_user(db, user_id: str) -> None:
    """Insert a minimal User row so the FK on build_jobs.user_id
    doesn't fail on dialects that enforce it (Postgres). Idempotent."""
    from app.db.models import User
    from sqlalchemy import select
    existing = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if existing is None:
        db.add(User(
            id=user_id,
            email=f"{user_id[:8]}@test.local",
            hashed_password="x" * 60,
            name="Test",
        ))
        await db.flush()


async def _make_job(
    db,
    *,
    user_id: str,
    job_type: str = "agent_task",
    status: str = "queued",
    parent_job_id: str | None = None,
    created_at: datetime | None = None,
) -> str:
    """Insert a BuildJob with the columns the dispatcher reads.
    Returns the job id."""
    from app.db.models import BuildJob

    job_id = _new_id()
    job = BuildJob(
        id=job_id,
        user_id=user_id,
        title="Test job",
        prompt="p",
        job_type=job_type,
        status=status,
        parent_job_id=parent_job_id,
    )
    if created_at is not None:
        job.created_at = created_at
    db.add(job)
    await db.flush()
    return job_id


@pytest_asyncio.fixture
async def db():
    """Yield a single AsyncSession scoped to the test. Commits are
    implicit at scope exit so each test sees a clean transaction."""
    from app.db.database import async_session_maker
    async with async_session_maker() as s:
        yield s


@pytest.fixture
def enable_spawning(monkeypatch):
    """Flip the kill switch ON for the duration of a test, without
    rebooting Settings. Other tests get the off default."""
    from app.config import settings
    monkeypatch.setattr(settings, "subagent_spawning_enabled", True)
    return settings


# ──────────────────────────────────────────────────────────────────────
# REJECTION_CODES enum is stable
# ──────────────────────────────────────────────────────────────────────


def test_rejection_codes_are_a_stable_closed_set():
    """Phase 4's tool handler quotes these strings; tests grep for
    them; logs filter on them. If a code is renamed, this lights up
    so the renamer fixes every site."""
    from app.agent.subagent_dispatcher import REJECTION_CODES

    assert set(REJECTION_CODES.all()) == {
        "SUBAGENT_DISABLED",
        "SUBAGENT_DEPTH_EXCEEDED",
        "SUBAGENT_PARENT_CAP",
        "SUBAGENT_USER_CONCURRENT_CAP",
        "SUBAGENT_USER_24H_CAP",
        "SUBAGENT_EMPTY_TASK",
    }


# ──────────────────────────────────────────────────────────────────────
# Idempotency key
# ──────────────────────────────────────────────────────────────────────


def test_idempotency_key_stable_across_calls():
    from app.agent.subagent_dispatcher import derive_idempotency_key

    k1 = derive_idempotency_key("parent-1", "research X")
    k2 = derive_idempotency_key("parent-1", "research X")
    assert k1 == k2
    # Differs when parent or task differ
    assert k1 != derive_idempotency_key("parent-2", "research X")
    assert k1 != derive_idempotency_key("parent-1", "research Y")


def test_idempotency_key_normalizes_whitespace_and_case():
    """A stray space or case change shouldn't defeat the dedup —
    otherwise a retry whose Claude rephrased the task by a single
    space slips through and double-spawns."""
    from app.agent.subagent_dispatcher import derive_idempotency_key

    assert derive_idempotency_key("p", "Research X  ") == derive_idempotency_key("p", "research x")
    assert derive_idempotency_key("p", "  RESEARCH X") == derive_idempotency_key("p", "research x")


def test_idempotency_key_handles_null_parent():
    """A top-level (depth-1) sub-agent has no parent_job_id — the
    key must still be deterministic, namespaced as ``sa:top:``."""
    from app.agent.subagent_dispatcher import derive_idempotency_key

    key = derive_idempotency_key(None, "research X")
    assert key.startswith("sa:top:")


def test_idempotency_key_fits_in_column():
    """The build_jobs.idempotency_key column is VARCHAR(120). Our
    key must fit comfortably."""
    from app.agent.subagent_dispatcher import derive_idempotency_key

    long_task = "x" * 10_000
    long_parent = "a" * 36
    key = derive_idempotency_key(long_parent, long_task)
    assert len(key) <= 120


# ──────────────────────────────────────────────────────────────────────
# walk_parent_chain
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_walk_parent_chain_single_row_no_parent(db):
    from app.agent.subagent_dispatcher import walk_parent_chain

    uid = _new_id()
    await _seed_user(db, uid)
    j = await _make_job(db, user_id=uid)

    chain = await walk_parent_chain(db, j)
    assert chain == [j]


@pytest.mark.asyncio
async def test_walk_parent_chain_multi_level(db):
    from app.agent.subagent_dispatcher import walk_parent_chain

    uid = _new_id()
    await _seed_user(db, uid)
    grandparent = await _make_job(db, user_id=uid)
    parent = await _make_job(db, user_id=uid, parent_job_id=grandparent)
    child = await _make_job(db, user_id=uid, parent_job_id=parent)

    chain = await walk_parent_chain(db, child)
    assert chain == [child, parent, grandparent]


@pytest.mark.asyncio
async def test_walk_parent_chain_missing_row_returns_empty(db):
    """Defensive: a parent_job_id passed into pre_spawn_checks that
    references no existing row must not crash; it returns []."""
    from app.agent.subagent_dispatcher import walk_parent_chain

    chain = await walk_parent_chain(db, "nonexistent-id")
    assert chain == []


@pytest.mark.asyncio
async def test_walk_parent_chain_cycle_short_circuits(db):
    """Belt-and-suspenders: if a manual UPDATE introduces a cycle,
    the walk shouldn't loop. Force a cycle and verify."""
    from sqlalchemy import update

    from app.agent.subagent_dispatcher import walk_parent_chain
    from app.db.models import BuildJob

    uid = _new_id()
    await _seed_user(db, uid)
    a = await _make_job(db, user_id=uid)
    b = await _make_job(db, user_id=uid, parent_job_id=a)
    # Force a → b (creates a cycle a → b → a)
    await db.execute(update(BuildJob).where(BuildJob.id == a).values(parent_job_id=b))
    await db.flush()

    chain = await walk_parent_chain(db, a)
    assert len(chain) <= 2  # short-circuited at the repeat
    assert a in chain and b in chain


# ──────────────────────────────────────────────────────────────────────
# Cap-count queries
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_count_running_children_counts_active_only(db):
    from app.agent.subagent_dispatcher import count_running_children

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid)
    await _make_job(db, user_id=uid, parent_job_id=parent, status="running")
    await _make_job(db, user_id=uid, parent_job_id=parent, status="queued")
    await _make_job(db, user_id=uid, parent_job_id=parent, status="completed")
    await _make_job(db, user_id=uid, parent_job_id=parent, status="failed")
    await _make_job(db, user_id=uid, parent_job_id=parent, status="cancelled")
    # Sibling under a different parent — must not be counted
    other_parent = await _make_job(db, user_id=uid)
    await _make_job(db, user_id=uid, parent_job_id=other_parent, status="running")

    assert await count_running_children(db, parent) == 2


@pytest.mark.asyncio
async def test_count_running_subagents_for_user_only_counts_subagent_type(db):
    """Pinned: only ``job_type='subagent'`` rows in active states
    count. The user-isolation half (``WHERE user_id = ?``) is covered
    by SQL inspection rather than a parallel-user fixture — the
    ``users.is_canary`` partial-unique index is Postgres-only and on
    SQLite gets compiled as a plain UNIQUE, blocking multi-user
    inserts in the test schema. The user filter is one line of SQL
    in the function; tests don't need to retest the engine."""
    from app.agent.subagent_dispatcher import count_running_subagents_for_user

    uid = _new_id()
    await _seed_user(db, uid)
    # Correct shape: 2 active subagent rows
    await _make_job(db, user_id=uid, job_type="subagent", status="running")
    await _make_job(db, user_id=uid, job_type="subagent", status="queued")
    # Wrong status (terminal) — not counted
    await _make_job(db, user_id=uid, job_type="subagent", status="completed")
    await _make_job(db, user_id=uid, job_type="subagent", status="failed")
    await _make_job(db, user_id=uid, job_type="subagent", status="cancelled")
    # Wrong type — not counted
    await _make_job(db, user_id=uid, job_type="agent_task", status="running")
    await _make_job(db, user_id=uid, job_type="trigger_run", status="running")
    await _make_job(db, user_id=uid, job_type="routine_run", status="running")

    assert await count_running_subagents_for_user(db, uid) == 2

    # User-isolation sanity: a bogus user_id (no matching rows)
    # returns 0, confirming the WHERE user_id clause is wired.
    assert await count_running_subagents_for_user(db, "nonexistent-uid") == 0


@pytest.mark.asyncio
async def test_count_subagent_spawns_24h_window(db):
    from app.agent.subagent_dispatcher import count_subagent_spawns_24h

    uid = _new_id()
    await _seed_user(db, uid)
    now = datetime.utcnow()
    # Fresh — counted
    await _make_job(db, user_id=uid, job_type="subagent", created_at=now)
    await _make_job(db, user_id=uid, job_type="subagent", created_at=now - timedelta(hours=12))
    # On the edge (just past 24h) — NOT counted
    await _make_job(db, user_id=uid, job_type="subagent", created_at=now - timedelta(hours=25))
    # Inside 24h but wrong type — not counted
    await _make_job(db, user_id=uid, job_type="agent_task", created_at=now)
    # Includes completed sub-agents (the cap is on spawns, not on outcomes)
    await _make_job(db, user_id=uid, job_type="subagent", status="completed", created_at=now - timedelta(hours=1))

    assert await count_subagent_spawns_24h(db, uid) == 3


# ──────────────────────────────────────────────────────────────────────
# pre_spawn_checks priority
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pre_spawn_kill_switch_rejects_everything(db, monkeypatch):
    from app.config import settings
    from app.agent.subagent_dispatcher import (
        REJECTION_CODES, pre_spawn_checks,
    )

    monkeypatch.setattr(settings, "subagent_spawning_enabled", False)
    uid = _new_id()
    await _seed_user(db, uid)
    rej = await pre_spawn_checks(db, user_id=uid, parent_job_id=None, task="anything")
    assert rej is not None
    assert rej.error_code == REJECTION_CODES.DISABLED


@pytest.mark.asyncio
async def test_pre_spawn_empty_task_rejected(db, enable_spawning):
    from app.agent.subagent_dispatcher import REJECTION_CODES, pre_spawn_checks

    uid = _new_id()
    await _seed_user(db, uid)
    rej = await pre_spawn_checks(db, user_id=uid, parent_job_id=None, task="   ")
    assert rej is not None
    assert rej.error_code == REJECTION_CODES.EMPTY_TASK


@pytest.mark.asyncio
async def test_pre_spawn_depth_cap_rejects_grandchild_in_v1(
    db, enable_spawning, monkeypatch,
):
    """v1 contract: ``subagent_max_depth=1`` blocks grandchildren.
    The user's main-turn job is the depth-0 root; a sub-agent
    spawned from it is depth 1 (allowed); a sub-agent spawned from
    THAT sub-agent is depth 2 (rejected).
    """
    from app.config import settings
    from app.agent.subagent_dispatcher import REJECTION_CODES, pre_spawn_checks

    monkeypatch.setattr(settings, "subagent_max_depth", 1)

    uid = _new_id()
    await _seed_user(db, uid)
    # User's main turn (depth 0 root)
    main_turn = await _make_job(db, user_id=uid, job_type="agent_task")
    # First-level sub-agent (depth 1) spawned from main turn
    first_subagent = await _make_job(
        db, user_id=uid, job_type="subagent", parent_job_id=main_turn,
    )

    # Now try to spawn a child off the first sub-agent — that's a
    # grandchild at depth 2 → rejected.
    rej = await pre_spawn_checks(
        db, user_id=uid, parent_job_id=first_subagent, task="X",
    )
    assert rej is not None
    assert rej.error_code == REJECTION_CODES.DEPTH_EXCEEDED


@pytest.mark.asyncio
async def test_pre_spawn_depth_one_allowed_from_main_turn(
    db, enable_spawning, monkeypatch,
):
    """A first-level sub-agent spawned from the user's main turn
    is depth 1 — the v1 happy path. Must be allowed at
    ``max_depth=1``."""
    from app.config import settings
    from app.agent.subagent_dispatcher import pre_spawn_checks

    monkeypatch.setattr(settings, "subagent_max_depth", 1)

    uid = _new_id()
    await _seed_user(db, uid)
    # User's main turn (non-subagent — depth 0 root)
    main_turn = await _make_job(db, user_id=uid, job_type="agent_task")

    rej = await pre_spawn_checks(
        db, user_id=uid, parent_job_id=main_turn, task="X",
    )
    assert rej is None


@pytest.mark.asyncio
async def test_subagent_depth_of_counts_only_subagent_ancestors(db):
    """The depth counter walks parent_job_id while ancestors are
    themselves subagent rows. The first non-subagent ancestor
    terminates the walk — it's the user's main turn root."""
    from app.agent.subagent_dispatcher import subagent_depth_of

    uid = _new_id()
    await _seed_user(db, uid)
    # Main turn (non-subagent) → depth 0
    main_turn = await _make_job(db, user_id=uid, job_type="agent_task")
    assert await subagent_depth_of(db, main_turn) == 0

    # First-level subagent (parent is non-subagent) → depth 1
    sa1 = await _make_job(
        db, user_id=uid, job_type="subagent", parent_job_id=main_turn,
    )
    assert await subagent_depth_of(db, sa1) == 1

    # Second-level subagent → depth 2
    sa2 = await _make_job(
        db, user_id=uid, job_type="subagent", parent_job_id=sa1,
    )
    assert await subagent_depth_of(db, sa2) == 2

    # Top-level subagent (no parent) → depth 1 (it's still a sub-agent;
    # the absence of a parent doesn't change that it's nested one level)
    orphan_sa = await _make_job(
        db, user_id=uid, job_type="subagent",
    )
    assert await subagent_depth_of(db, orphan_sa) == 1


@pytest.mark.asyncio
async def test_pre_spawn_per_parent_cap(db, enable_spawning, monkeypatch):
    from app.config import settings
    from app.agent.subagent_dispatcher import REJECTION_CODES, pre_spawn_checks

    monkeypatch.setattr(settings, "subagent_max_children_per_parent", 2)

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid)
    await _make_job(db, user_id=uid, parent_job_id=parent, status="running")
    await _make_job(db, user_id=uid, parent_job_id=parent, status="running")

    rej = await pre_spawn_checks(db, user_id=uid, parent_job_id=parent, task="X")
    assert rej is not None
    assert rej.error_code == REJECTION_CODES.PARENT_CAP


@pytest.mark.asyncio
async def test_pre_spawn_user_concurrent_cap(db, enable_spawning, monkeypatch):
    from app.config import settings
    from app.agent.subagent_dispatcher import REJECTION_CODES, pre_spawn_checks

    monkeypatch.setattr(settings, "subagent_max_per_user_concurrent", 1)
    monkeypatch.setattr(settings, "subagent_max_children_per_parent", 99)

    uid = _new_id()
    await _seed_user(db, uid)
    # The user already has one active sub-agent under a different
    # parent — the per-user concurrent cap kicks in regardless of
    # which parent the new spawn targets.
    other_parent = await _make_job(db, user_id=uid)
    await _make_job(
        db, user_id=uid, parent_job_id=other_parent,
        job_type="subagent", status="running",
    )

    new_parent = await _make_job(db, user_id=uid)
    rej = await pre_spawn_checks(db, user_id=uid, parent_job_id=new_parent, task="X")
    assert rej is not None
    assert rej.error_code == REJECTION_CODES.USER_CONCURRENT_CAP


@pytest.mark.asyncio
async def test_pre_spawn_user_24h_cap(db, enable_spawning, monkeypatch):
    from app.config import settings
    from app.agent.subagent_dispatcher import REJECTION_CODES, pre_spawn_checks

    # Set the 24h cap below the existing fixture spawn count to trip
    # the rejection without piling on dozens of inserts.
    monkeypatch.setattr(settings, "subagent_max_per_user_24h", 2)
    monkeypatch.setattr(settings, "subagent_max_children_per_parent", 99)
    monkeypatch.setattr(settings, "subagent_max_per_user_concurrent", 99)

    uid = _new_id()
    await _seed_user(db, uid)
    # Mix completed + running — both count toward 24h spawns.
    await _make_job(db, user_id=uid, job_type="subagent", status="completed")
    await _make_job(db, user_id=uid, job_type="subagent", status="running")
    await _make_job(db, user_id=uid, job_type="subagent", status="failed")

    rej = await pre_spawn_checks(db, user_id=uid, parent_job_id=None, task="X")
    assert rej is not None
    assert rej.error_code == REJECTION_CODES.USER_24H_CAP


@pytest.mark.asyncio
async def test_pre_spawn_top_level_no_parent_allowed(db, enable_spawning):
    """A top-level spawn (no parent_job_id) skips the depth + per-parent
    cap and lands on the per-user caps only."""
    from app.agent.subagent_dispatcher import pre_spawn_checks

    uid = _new_id()
    await _seed_user(db, uid)
    rej = await pre_spawn_checks(db, user_id=uid, parent_job_id=None, task="hello")
    assert rej is None


# ──────────────────────────────────────────────────────────────────────
# cascade_cancel
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cascade_cancel_flips_active_descendants(db):
    from app.agent.subagent_dispatcher import cascade_cancel
    from sqlalchemy import select
    from app.db.models import BuildJob

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid)
    c1 = await _make_job(db, user_id=uid, parent_job_id=parent, status="running")
    c2 = await _make_job(db, user_id=uid, parent_job_id=parent, status="queued")

    cancelled = await cascade_cancel(db, parent, reason="parent failed")
    assert set(cancelled) == {c1, c2}

    rows = (await db.execute(select(BuildJob).where(BuildJob.parent_job_id == parent))).scalars().all()
    for r in rows:
        assert r.status == "cancelled"
        assert r.error_message == "parent failed"
        assert r.completed_at is not None


@pytest.mark.asyncio
async def test_cascade_cancel_preserves_terminal_children(db):
    """A child that already finished (success or failure) must not
    be clobbered to 'cancelled' — its outcome stands."""
    from app.agent.subagent_dispatcher import cascade_cancel
    from sqlalchemy import select
    from app.db.models import BuildJob

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid)
    completed_child = await _make_job(
        db, user_id=uid, parent_job_id=parent, status="completed",
    )
    failed_child = await _make_job(
        db, user_id=uid, parent_job_id=parent, status="failed",
    )
    running_child = await _make_job(
        db, user_id=uid, parent_job_id=parent, status="running",
    )

    cancelled = await cascade_cancel(db, parent, reason="x")
    assert cancelled == [running_child]

    # Terminal children unchanged
    rows = {r.id: r for r in (await db.execute(
        select(BuildJob).where(BuildJob.parent_job_id == parent)
    )).scalars().all()}
    assert rows[completed_child].status == "completed"
    assert rows[failed_child].status == "failed"
    assert rows[running_child].status == "cancelled"


@pytest.mark.asyncio
async def test_cascade_cancel_recurses_through_grandchildren(db):
    """Even though v1 caps depth at 1, the helper must be correct
    for arbitrary depth in case the cap is raised later."""
    from app.agent.subagent_dispatcher import cascade_cancel

    uid = _new_id()
    await _seed_user(db, uid)
    grandparent = await _make_job(db, user_id=uid)
    parent = await _make_job(db, user_id=uid, parent_job_id=grandparent, status="running")
    child = await _make_job(db, user_id=uid, parent_job_id=parent, status="running")

    cancelled = await cascade_cancel(db, grandparent, reason="root failed")
    assert set(cancelled) == {parent, child}


@pytest.mark.asyncio
async def test_cascade_cancel_returns_empty_for_no_children(db):
    from app.agent.subagent_dispatcher import cascade_cancel

    uid = _new_id()
    await _seed_user(db, uid)
    solo = await _make_job(db, user_id=uid)
    assert await cascade_cancel(db, solo, reason="x") == []


# ──────────────────────────────────────────────────────────────────────
# transition_job_status (the chokepoint)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transition_job_status_failure_propagates_to_children(db):
    from app.agent.subagent_dispatcher import transition_job_status
    from sqlalchemy import select
    from app.db.models import BuildJob

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid, status="running")
    child = await _make_job(db, user_id=uid, parent_job_id=parent, status="running")

    transitioned, cascaded = await transition_job_status(
        db, parent, new_status="failed", error_message="boom",
    )
    assert transitioned is True
    assert cascaded == [child]

    parent_row = (await db.execute(select(BuildJob).where(BuildJob.id == parent))).scalar_one()
    assert parent_row.status == "failed"
    assert parent_row.error_message == "boom"

    child_row = (await db.execute(select(BuildJob).where(BuildJob.id == child))).scalar_one()
    assert child_row.status == "cancelled"
    assert child_row.error_message == "boom"


@pytest.mark.asyncio
async def test_transition_job_status_completed_does_not_propagate(db):
    """A successfully completed parent must NOT cancel its children —
    they run on their own lifecycle."""
    from app.agent.subagent_dispatcher import transition_job_status
    from sqlalchemy import select
    from app.db.models import BuildJob

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid, status="running")
    child = await _make_job(db, user_id=uid, parent_job_id=parent, status="running")

    transitioned, cascaded = await transition_job_status(
        db, parent, new_status="completed",
    )
    assert transitioned is True
    assert cascaded == []

    child_row = (await db.execute(select(BuildJob).where(BuildJob.id == child))).scalar_one()
    assert child_row.status == "running"  # untouched


@pytest.mark.asyncio
async def test_transition_job_status_preserves_existing_terminal(db):
    """If the row was already finalized by a handler, the transition
    must be a no-op (race-safety)."""
    from app.agent.subagent_dispatcher import transition_job_status
    from sqlalchemy import select
    from app.db.models import BuildJob

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid, status="completed")

    transitioned, cascaded = await transition_job_status(
        db, parent, new_status="cancelled", error_message="late cancel",
    )
    assert transitioned is False
    assert cascaded == []

    parent_row = (await db.execute(select(BuildJob).where(BuildJob.id == parent))).scalar_one()
    assert parent_row.status == "completed"  # preserved


@pytest.mark.asyncio
async def test_transition_job_status_propagate_off_skips_cascade(db):
    """Caller can opt out of cascade for cases where they manage
    children explicitly (e.g. an orphan-sweep pass that flips many
    parents but already cancelled their children in a separate
    pass)."""
    from app.agent.subagent_dispatcher import transition_job_status
    from sqlalchemy import select
    from app.db.models import BuildJob

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid, status="running")
    child = await _make_job(db, user_id=uid, parent_job_id=parent, status="running")

    transitioned, cascaded = await transition_job_status(
        db, parent, new_status="failed", error_message="x",
        propagate_cancel=False,
    )
    assert transitioned is True
    assert cascaded == []

    child_row = (await db.execute(select(BuildJob).where(BuildJob.id == child))).scalar_one()
    assert child_row.status == "running"


@pytest.mark.asyncio
async def test_transition_job_status_rejects_invalid_status(db):
    from app.agent.subagent_dispatcher import transition_job_status

    uid = _new_id()
    await _seed_user(db, uid)
    parent = await _make_job(db, user_id=uid)

    with pytest.raises(ValueError):
        await transition_job_status(db, parent, new_status="not-a-real-status")
