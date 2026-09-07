"""The stranded backstop re-provisions accounts that are being deleted.

MEASURED ON PRODUCTION, 2026-09-07. A throwaway account (user c7905c52) was
deleted at 15:07:44 UTC. The bridge journal:

    15:07:49  sudo revoke_tenant_db feed0076
    15:07:53  POST /v1/pool/release            200      ← the only one
    15:07:57  GET  /v1/pool/whois?user_id=c7905c52-…    ← the platform, again
    15:07:57  caddy route removal failed (404) prefix=c7905c52
    15:07:57  [pool] workspace save failed prefix=c7905c52: No such container

That whois is the 15 s stranded fast pass. `destroy_container` had just set
the managed_containers row to status='deleted' while the row still names
`toup-agent-pool-76` and the User row is still present and `is_active` — which
is exactly `_stranded_user_ids`' "managed user with a non-running pool-bound
row, regardless of signup age", and `_recently_stranded_user_ids`' predicate
too (the account was ten minutes old). Deletion takes seconds more after that
point (the OpenAI archive alone logged at 15:07:55, the table wipe and the user
row after it), and it can stop there permanently: a `DeletionAbortedError` at
the container, cascade or user-row step leaves the User row and the 'deleted'
container row in place forever.

Nothing bound a fresh slot on 2026-09-07 only because the fast pass adopts and
the released member was already DRAINING, so `_is_adoptable` said no. The 180 s
`reclaim_stranded_users` does not adopt — it calls `claim_for_user`, which
CLAIMS. A tick landing in that window binds a brand-new pool container, with a
new database and a new Caddy route, to an account the platform is in the middle
of erasing; deletion has already passed its container step, so nothing will
ever release it. That is the same leak `PoolRelease`'s docstring records for
three slots on 2026-08-31.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import uuid
from datetime import datetime

os.environ.setdefault("ENVIRONMENT", "test")

import pytest

_port = itertools.count(9800)


@pytest.fixture(autouse=True)
def _no_leaked_discovery_loops():
    yield
    from app.services import pool_service as ps
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    for t in list(asyncio.all_tasks()) if running else []:
        if not t.done() and (t.get_name() or "").startswith(("discover:", "adopt-once:")):
            t.cancel()
    ps._DISCOVERY_INFLIGHT.clear()
    ps._invalidate_pool_list_cache()


async def _seed(db, *, container=("toup-agent-pool-76", "deleted"),
                deletion: tuple | None = None):
    """A managed user whose pool container row is in the state
    `destroy_container` leaves it, optionally with a deletion audit row.

    `deletion` is (status, failure_step)."""
    from app.db.models import (
        User, AgentConfig, ManagedContainer, DeletionAuditEvent,
    )
    uid = str(uuid.uuid4())
    db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                name="T", is_active=True))
    await db.flush()
    db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                       bundle_status="active", llm_mode="bundle"))
    if container is not None:
        name, status = container
        db.add(ManagedContainer(
            id=str(uuid.uuid4()), user_id=uid, container_name=name,
            host_port=next(_port), db_name=f"db_{uid[:8]}", status=status,
        ))
    if deletion is not None:
        status, step = deletion
        db.add(DeletionAuditEvent(
            id=str(uuid.uuid4()), user_id=uid, user_email=f"{uid[:8]}@t.local",
            actor="self", actor_user_id=uid, initiated_at=datetime.utcnow(),
            status=status, failure_step=step,
        ))
    await db.commit()
    return uid


# ═══════════════════════════════════════════════════════════════════════
# (a) the state the 2026-09-07 deletion was actually in
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_user_mid_deletion_is_not_a_stranded_candidate(monkeypatch):
    """FALSIFIER: on the unchanged tree this user appears in BOTH predicates,
    which is what put `GET /v1/pool/whois?user_id=c7905c52-…` on the bridge
    four seconds after that account's slot had been released."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db, deletion=("in_progress", None))

    async with async_session_maker() as db:
        slow = await ps._stranded_user_ids(db, limit=60)
        fast = await ps._recently_stranded_user_ids(db)

    assert uid not in slow, (
        "the 180 s reclaim would call claim_for_user for an account being "
        "erased — binding a fresh pool slot, database and Caddy route that "
        "deletion has already passed the point of releasing"
    )
    assert uid not in fast, (
        "the 15 s fast pass would ask the bridge about an account being "
        "erased — the production whois at 15:07:57"
    )


@pytest.mark.asyncio
async def test_the_fast_pass_asks_the_bridge_nothing_about_a_deleted_user(monkeypatch):
    """The end-to-end shape of the production observation: one bridge call, or
    none."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    async with async_session_maker() as db:
        await _seed(db, deletion=("in_progress", None))

    asked: list = []

    async def _lookup(uid, timeout_s=None):
        asked.append(uid)
        return None
    monkeypatch.setattr(ps, "bridge_lookup_user_slot", _lookup)

    summary = await ps.reclaim_stranded_fast()
    assert asked == [], (
        f"the fast pass asked the bridge about {len(asked)} user(s) mid-"
        f"deletion; on 2026-09-07 that call is in the bridge journal at "
        f"15:07:57, four seconds after the slot was released"
    )
    assert summary.get("candidates") == 0


# ═══════════════════════════════════════════════════════════════════════
# (b) guards against over-fixing
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_an_ordinary_stranded_user_is_still_reclaimed(monkeypatch):
    """The backstop's whole job. No deletion row → unchanged."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db, container=("toup-agent-pool-76", "error"))

    async with async_session_maker() as db:
        slow = await ps._stranded_user_ids(db, limit=60)
        fast = await ps._recently_stranded_user_ids(db)
    assert uid in slow and uid in fast


@pytest.mark.asyncio
async def test_a_deletion_that_failed_before_the_container_step_still_heals(monkeypatch):
    """`delete_user_completely` runs Stripe FIRST, and aborts there without
    touching the container: "we have NOT yet started destroying local data, so
    the user is still functional". Such a user must still be healed — they are
    an ordinary customer whose delete request failed."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db, container=("toup-agent-pool-76", "error"),
                          deletion=("failed", "stripe"))

    async with async_session_maker() as db:
        slow = await ps._stranded_user_ids(db, limit=60)
    assert uid in slow, (
        "a deletion that aborted at Stripe never touched the container; "
        "refusing to heal that user leaves a paying customer's agent dead"
    )


@pytest.mark.asyncio
async def test_a_deletion_that_failed_at_teardown_is_left_to_an_operator(monkeypatch):
    """`DeletionAbortedError(CONTAINER)` means teardown was owed and did not
    land — `ContainerTeardownIncomplete`. Claiming a NEW slot on top of that is
    how a released-but-not-reaped slot becomes a permanently leaked one."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db, deletion=("failed", "container"))
        uid2 = await _seed(db, deletion=("failed", "cascade"))

    async with async_session_maker() as db:
        slow = await ps._stranded_user_ids(db, limit=60)
    assert uid not in slow and uid2 not in slow


@pytest.mark.asyncio
async def test_the_guard_reads_the_row_the_deletion_writes_first(monkeypatch):
    """The marker has to exist BEFORE anything is torn down, or the window it
    closes is the window that matters. `delete_user_completely` inserts the
    audit row and commits it as its first transaction — before Stripe, before
    the container. This asserts that ordering at the source, because a later
    refactor that moved the insert after the teardown would leave every test
    above green and the defect restored."""
    import inspect
    from app.services import user_deletion as ud

    src = inspect.getsource(ud.delete_user_completely)
    i_audit = src.index("db.add(audit)")
    i_commit = src.index("await db.commit()", i_audit)
    i_container = src.index("destroy_container")
    i_stripe = src.index("stripe_service")
    assert i_commit < i_stripe < i_container, (
        "the deletion audit row must be committed before Stripe and before "
        "container teardown; otherwise the reclaim guard cannot see a deletion "
        "that is already past the point of no return"
    )
