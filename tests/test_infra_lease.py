"""One fleet-wide runner for the infra-mutating loops (2026-09-12, L3-1).

The incident, restated as the property under test: `reclaim_stranded_users`
ran on BOTH Railway replicas with no election, so 46 reconciliation ticks
produced 92 container restarts — exactly one POST from each of the two
Railway egress IPs, every tick, for 2 h 28 m. Nothing in the loop could
notice, because "am I the only one running this?" was never a question the
code asked.

What is asserted here is the CAS itself (two holders contend, exactly one
wins; expiry hands over; a renewal does not hand over) and that the loops
actually consult it before doing work. The second half matters more than it
looks: a correct lease helper that nothing calls is the shape this incident
already took once — `container_monitor` has printed "started" on every boot
for four months while returning on line 2.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q tests/test_infra_lease.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

os.environ.setdefault("ENVIRONMENT", "test")


def _src(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text()


# ── The CAS ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_holders_contend_exactly_one_wins():
    from app.services.infra_lease import acquire_lease

    a = await acquire_lease("t_contend", holder="replica-A", ttl_s=600)
    b = await acquire_lease("t_contend", holder="replica-B", ttl_s=600)

    assert a is True
    assert b is False, "a live lease must not be stealable"


@pytest.mark.asyncio
async def test_concurrent_acquire_still_yields_one_winner():
    """The read-then-write version of this passes the test above and fails
    here. The upsert is one statement precisely so there is no window."""
    from app.services.infra_lease import acquire_lease

    results = await asyncio.gather(*[
        acquire_lease("t_race", holder=f"replica-{i}", ttl_s=600)
        for i in range(6)
    ])
    assert sum(1 for r in results if r) == 1, results


@pytest.mark.asyncio
async def test_holder_renews_without_losing_the_lease():
    from app.services.infra_lease import acquire_lease, lease_holder

    assert await acquire_lease("t_renew", holder="replica-A", ttl_s=600)
    assert await acquire_lease("t_renew", holder="replica-A", ttl_s=600)
    assert await acquire_lease("t_renew", holder="replica-A", ttl_s=600)
    assert await lease_holder("t_renew") == "replica-A"
    # …and the other replica is still locked out after all that renewing.
    assert await acquire_lease("t_renew", holder="replica-B", ttl_s=600) is False


@pytest.mark.asyncio
async def test_tick_seq_counts_the_holder_s_ticks():
    """The operator-facing half: 'which replica, for how many ticks'."""
    from app.db import async_session_maker
    from app.db.models import InfraLease
    from sqlalchemy import select
    from app.services.infra_lease import acquire_lease

    for _ in range(4):
        await acquire_lease("t_seq", holder="replica-A", ttl_s=600)
    async with async_session_maker() as db:
        seq = (await db.execute(
            select(InfraLease.tick_seq).where(InfraLease.name == "t_seq")
        )).scalar_one()
    assert seq == 4


@pytest.mark.asyncio
async def test_expiry_hands_over_to_the_other_replica():
    """A holder that dies mid-tick releases by TTL, with no operator."""
    from app.db import async_session_maker
    from app.db.models import InfraLease
    from sqlalchemy import update
    from app.services.infra_lease import acquire_lease, lease_holder

    assert await acquire_lease("t_expire", holder="replica-A", ttl_s=600)
    assert await acquire_lease("t_expire", holder="replica-B", ttl_s=600) is False

    # replica-A's process is gone; only the clock can free the row.
    async with async_session_maker() as db:
        await db.execute(
            update(InfraLease).where(InfraLease.name == "t_expire").values(
                expires_at=datetime.utcnow() - timedelta(seconds=1),
            )
        )
        await db.commit()

    assert await acquire_lease("t_expire", holder="replica-B", ttl_s=600) is True
    assert await lease_holder("t_expire") == "replica-B"
    # And now A is the one locked out — handover, not a free-for-all.
    assert await acquire_lease("t_expire", holder="replica-A", ttl_s=600) is False


@pytest.mark.asyncio
async def test_leases_are_independent_per_name():
    from app.services.infra_lease import acquire_lease

    assert await acquire_lease("t_one", holder="replica-A", ttl_s=600)
    assert await acquire_lease("t_two", holder="replica-B", ttl_s=600)


@pytest.mark.asyncio
async def test_release_hands_over_immediately():
    from app.services.infra_lease import acquire_lease, release_lease

    assert await acquire_lease("t_rel", holder="replica-A", ttl_s=600)
    assert await release_lease("t_rel", holder="replica-A") is True
    assert await acquire_lease("t_rel", holder="replica-B", ttl_s=600) is True


@pytest.mark.asyncio
async def test_release_by_a_non_holder_is_a_no_op():
    from app.services.infra_lease import acquire_lease, release_lease, lease_holder

    assert await acquire_lease("t_rel2", holder="replica-A", ttl_s=600)
    assert await release_lease("t_rel2", holder="replica-B") is False
    assert await lease_holder("t_rel2") == "replica-A"


@pytest.mark.asyncio
async def test_db_failure_answers_false_not_true(monkeypatch):
    """Fail CLOSED. Answering True on a DB blip re-creates the two-runner
    bug on exactly the day the database is unhappy."""
    from app.services import infra_lease as il

    def _boom(*a, **k):
        raise RuntimeError("pool exhausted")

    monkeypatch.setattr(il, "select", _boom, raising=False)
    monkeypatch.setattr(
        "app.db.database.async_session_maker", _boom, raising=False,
    )
    assert await il.acquire_lease("t_boom", holder="replica-A", ttl_s=600) is False


@pytest.mark.asyncio
async def test_acquire_is_silent_and_stores_naive_utc():
    """The first deployed version called `datetime.utcnow()` on every acquire
    and Python 3.12 answered with a DeprecationWarning each time — ~14 lines a
    minute across two replicas into the Railway logs the incident forensics
    read. The stored value must stay NAIVE UTC: the columns carry no timezone
    and the upsert compares `expires_at < :now` on both dialects."""
    import warnings

    from sqlalchemy import select

    from app.db import async_session_maker
    from app.db.models import InfraLease
    from app.services.infra_lease import acquire_lease

    before = datetime.now(timezone.utc).replace(tzinfo=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert await acquire_lease("t_quiet", holder="replica-A", ttl_s=600) is True
    ours = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning) and "infra_lease" in str(w.filename)
    ]
    assert not ours, [str(w.message) for w in ours]

    async with async_session_maker() as db:
        row = (await db.execute(
            select(InfraLease).where(InfraLease.name == "t_quiet")
        )).scalar_one()
    assert row.expires_at.tzinfo is None and row.acquired_at.tzinfo is None
    assert before - timedelta(seconds=5) <= row.acquired_at <= before + timedelta(seconds=60)
    assert timedelta(seconds=595) <= row.expires_at - row.acquired_at <= timedelta(seconds=605)


def test_holder_id_prefers_the_railway_replica_id(monkeypatch):
    from app.services import infra_lease as il

    il.reset_for_tests()
    monkeypatch.setenv("RAILWAY_REPLICA_ID", "rep-abc123")
    assert il.holder_id() == "rep-abc123"
    # Cached: a renewal cannot silently change identity mid-tick.
    monkeypatch.setenv("RAILWAY_REPLICA_ID", "rep-CHANGED")
    assert il.holder_id() == "rep-abc123"
    il.reset_for_tests()


def test_holder_id_falls_back_to_a_per_process_uuid(monkeypatch):
    from app.services import infra_lease as il

    il.reset_for_tests()
    monkeypatch.delenv("RAILWAY_REPLICA_ID", raising=False)
    monkeypatch.delenv("RAILWAY_INSTANCE_ID", raising=False)
    h = il.holder_id()
    assert h.startswith("proc-") and len(h) > 5
    il.reset_for_tests()


def test_ttl_is_three_times_the_interval():
    from app.services.infra_lease import lease_ttl_for

    assert lease_ttl_for(180) == 540.0
    assert lease_ttl_for(3600) == 10800.0
    # …with a floor, so a 1 s loop cannot make the lease unrenewable.
    assert lease_ttl_for(1) == 30.0


# ── The loops actually consult it ─────────────────────────────────────
#
# Source probes, deliberately: a behavioural test of `container_reconciler_loop`
# would have to run a forever-loop with real sleeps. What must never regress is
# that the gate is PRESENT and sits ABOVE the work — a guard whose subject runs
# before it is invisible to every other check in this repo.

def test_container_reconciler_gates_both_the_tick_and_the_fast_subtick():
    src = _src("app/services/docker_host_service.py")
    body = src[src.index("async def container_reconciler_loop"):
               src.index("async def stop_container")]
    assert body.count("acquire_lease(_lease, ttl_s=_ttl)") == 2, (
        "the 180 s tick and the 15 s stranded-fast sub-tick must BOTH be gated"
    )
    # The gate must precede the work in the full tick.
    gate = body.index("if not await acquire_lease(_lease, ttl_s=_ttl):\n"
                      "            # Another replica")
    assert gate < body.index("backfill_sentinel_image_containers(db)")
    assert gate < body.index("reclaim_stranded_users()")
    assert gate < body.index("reconcile_managed_rows(db)")
    # One lease name for both — they are the same work at two grains.
    assert '_lease = "container_reconciler"' in body


@pytest.mark.parametrize("rel,name,marker", [
    ("app/services/container_monitor.py", "container_monitor",
     "check_all_containers()"),
    ("app/services/credit_health_monitor.py", "credit_health_monitor",
     "check_credit_health()"),
    ("app/services/search_quota_monitor.py", "search_quota_monitor",
     "check_search_quota()"),
    ("app/services/connector_health_probe.py", "connector_health_probe",
     "self.run_once()"),
    ("app/services/bridge_supervisor.py", "bridge_supervisor",
     "_supervisor_tick()"),
    # The Apple reconciler moves real money: two replicas racing the same
    # finding write two plan_change ledger rows for one event, and on
    # apply_plan_change's PRORATED arm they apply the delta TWICE. The FOR
    # UPDATE inside _lock_balance does not help — the second replica is a
    # different process reading its own snapshot.
    ("app/services/apple_reconciler.py", "apple-reconciler",
     "run_reconcile_pass()"),
    # The rollout reconciler drives blue-green upgrades, CREATES sweep
    # rollouts and pages the operator. It was exempted below on the claim
    # that it "already owns a DB lock"; it owns no lock of any kind (see
    # that test's docstring), and on 2026-09-16 the bridge access log caught
    # both replicas polling /v1/pool/health every 30 s, 11 s apart, with the
    # duplicate fleet pages to match.
    ("app/services/rollout_service.py", "rollout_reconciler",
     "_fleet_watch_once()"),
])
def test_every_gated_loop_acquires_before_it_works(rel, name, marker):
    src = _src(rel)
    assert f'acquire_lease("{name}"' in src, f"{rel} never claims its lease"
    assert src.index(f'acquire_lease("{name}"') < src.rindex(marker), (
        f"{rel}: the lease gate must sit ABOVE the work it gates"
    )


def test_loops_that_must_not_be_leader_gated_are_not():
    """`db_watchdog` heals THIS process's engine — electing a leader for it
    would be a bug, not a fix. `notification_dispatch` is per-row CAS, which
    is strictly better than a leader for throughput.

    `rollout_service` used to sit in this list on the justification that
    "rollout_reconciler already owns a DB lock". It owns none: there is no
    `FOR UPDATE`, no advisory lock and no unique constraint anywhere on that
    path, and its three exclusions (`_resume_inflight`,
    `_rollout_creation_lock` — whose own comment asserts a single
    platform-api instance — and `_FLEET_STATE`) are all process-local, so
    none of them could see the second Railway replica. It is gated now and
    asserted in the table above."""
    for rel in ("app/services/db_watchdog.py",
                "app/services/notification_dispatcher.py"):
        assert "infra_lease" not in _src(rel), rel


def test_the_rollout_gate_is_on_the_loop_not_on_the_shared_functions():
    """One acquire, in the reconciler loop's own tick body, above the work.

    (The body lives in `_rollout_reconciler_ticks`, between the entry point
    and the convergence-sweep section, so the slice below covers both. The
    per-tick renewal itself is pinned behaviourally in
    tests/test_rollout_fleet_alerting.py::TestLeaderGate — a source probe
    cannot tell one acquire inside the loop from one hoisted above it.)

    `start_rollout` runs `_reconcile_once(db)` as its lock self-heal pass and
    the admin API calls into this module directly. Those serve a request an
    operator is waiting on, not a periodic tick: a lease check there would
    make every CI push 409 for as long as the OTHER replica held the lease,
    and would silently stop orphaning the stuck rollout that is causing the
    409 in the first place."""
    src = _src("app/services/rollout_service.py")
    assert src.count("acquire_lease(") == 1, (
        "exactly one gate — a second one is a shared function being gated"
    )
    loop = src[src.index("async def rollout_reconciler_loop"):
               src.index("# ─── Convergence sweep")]
    assert 'acquire_lease("rollout_reconciler"' in loop, "the gate is the LOOP's"
    assert (loop.index('acquire_lease("rollout_reconciler"')
            < loop.index("await _reconcile_once()")), "gate above the work"


def test_no_loop_reaches_for_a_session_scoped_advisory_lock():
    """L3 §11.1: `pg_advisory_lock` is bound to a connection, and this
    platform runs behind a transaction pooler — a lock you cannot reliably
    release. The per-user `pg_advisory_xact_lock` sites stay."""
    for rel in ("app/services/docker_host_service.py",
                "app/services/container_monitor.py"):
        assert "pg_advisory_lock" not in _src(rel), rel
    # infra_lease NAMES it in its own docstring (that is the record of why it
    # was rejected); it must never CALL it. Strip the module docstring and the
    # comments, then look again.
    import ast
    body = _src("app/services/infra_lease.py")
    tree = ast.parse(body)
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.Expr)
                         and isinstance(n.value, ast.Constant))]
    assert "pg_advisory_lock" not in ast.unparse(tree)
