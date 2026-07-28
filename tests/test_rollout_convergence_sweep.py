"""
Rollout convergence-sweep suite — honest terminal status + divergence
healing + orphaned-attempt cleanup.

Background (2026-07-28 incident, platform DB): rollout 01a945e2's driver
was killed by its own merge's Railway redeploy — the reconciler orphaned
it, leaving FOUR attempt rows stuck in status='upgrading' forever. The
re-drive d79584ea then hit the bridge mid-restart (502) on tenant
2739b5c6, stamped rollouts.status='complete' anyway, and a real beta
user sat silently on the OLD image until an operator noticed ~40 min
later. Three fixes, pinned here:

  D1 — a rollout with any per-tenant failure ends
       'complete_with_failures', not 'complete' (both terminal).
  D2 — the reconciler's convergence sweep: when no rollout is active,
       re-drive tenants whose managed_containers.image_tag diverges from
       the newest complete/complete_with_failures rollout's tag, via a
       trigger='sweep' rollout through the ordinary _upgrade_one path.
       Kill switch settings.rollout_convergence_sweep (default ON);
       bounded at 3 sweeps per tag per 24h, then alert + skip.
  D3 — orphaning a rollout also closes its in-flight 'upgrading'
       attempts as 'orphaned'; a backfill pass heals 'upgrading' rows
       whose parent rollout is already terminal.

Mirrors the real-DB integration style of test_rollout_service.py
(seed through app.db.async_session_maker, expunge, mock the bridge with
patch.object on rollout_service, record Telegram with a fake).
"""

import uuid as _uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

OLD_TAG = "ghcr.io/toup-com/toup-agent:oldtag00"
NEW_TAG = "ghcr.io/toup-com/toup-agent:newtag00"

# Monotonic host ports — hash-derived ports can collide when one test
# seeds several containers.
_next_port = iter(range(9500, 9999))


# ─── Seed helpers (plain async helpers, not fixtures — same pattern as
#     _seed_orphan_quarantine_fixtures in test_rollout_service.py) ─────


async def _seed_tenant(
    *,
    image_tag: str = OLD_TAG,
    container_name: str | None = "__default__",
    pin_image_tag: str | None = None,
    status: str = "running",
    is_canary: bool = False,
):
    """Create a User + ManagedContainer; return the container detached."""
    from app.db import async_session_maker
    from app.db.models import ManagedContainer, User
    from app.services.auth_service import get_password_hash

    user_id = str(_uuid.uuid4())
    if container_name == "__default__":
        container_name = f"toup-agent-{user_id[:8]}"
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"sweep-{user_id[:8]}@example.com",
            hashed_password=get_password_hash("test1234abcd"),
            name=f"Sweep {user_id[:6]}",
            is_canary=is_canary,
        ))
        await db.flush()
        mc = ManagedContainer(
            id=str(_uuid.uuid4()),
            user_id=user_id,
            container_id=f"docker-{user_id[:8]}",
            container_name=container_name,
            host_port=next(_next_port),
            db_name=f"toup_agent_{user_id[:8]}",
            status=status,
            image_tag=image_tag,
            pin_image_tag=pin_image_tag,
        )
        db.add(mc)
        await db.commit()
        await db.refresh(mc)
        db.expunge(mc)
    return mc


async def _seed_rollout(**overrides):
    """Create a Rollout row; return it detached."""
    from app.db import async_session_maker
    from app.db.models import Rollout

    kwargs = dict(
        image_tag=NEW_TAG,
        status="complete",
        trigger="ci",
        canary_wait_minutes=0,
        started_at=datetime.utcnow() - timedelta(minutes=10),
        completed_at=datetime.utcnow() - timedelta(minutes=5),
    )
    kwargs.update(overrides)
    async with async_session_maker() as db:
        rollout = Rollout(**kwargs)
        db.add(rollout)
        await db.commit()
        await db.refresh(rollout)
        db.expunge(rollout)
    return rollout


async def _seed_attempt(rollout_id: str, container_id: str, **overrides):
    """Create a RolloutAttempt row; return its id."""
    from app.db import async_session_maker
    from app.db.models import RolloutAttempt

    kwargs = dict(
        rollout_id=rollout_id,
        container_id=container_id,
        prior_tag=OLD_TAG,
        new_tag=NEW_TAG,
        status="upgrading",
        started_at=datetime.utcnow() - timedelta(minutes=8),
    )
    kwargs.update(overrides)
    async with async_session_maker() as db:
        attempt = RolloutAttempt(**kwargs)
        db.add(attempt)
        await db.commit()
        await db.refresh(attempt)
        attempt_id = attempt.id
    return attempt_id


async def _sweep_rollouts():
    """All trigger='sweep' rollouts currently in the DB."""
    from app.db import async_session_maker
    from app.db.models import Rollout

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Rollout).where(Rollout.trigger == "sweep")
        )).scalars().all()
        for r in rows:
            db.expunge(r)
    return rows


def _clear_sweep_alert_dedup():
    from app.services import rollout_service
    rollout_service._sweep_exhausted_alerted.clear()


# ─── D1: honest terminal status ────────────────────────────────────


@pytest.mark.asyncio
async def test_drive_rollout_stamps_complete_with_failures_on_partial_failure():
    """total_fail > 0 must end 'complete_with_failures', not 'complete'.

    The incident's re-drive d79584ea failed tenant 2739b5c6 with a bridge
    502 yet reported 'complete' — nothing downstream could tell the fleet
    had NOT converged.
    """
    from app.db import async_session_maker
    from app.db.models import Rollout
    from app.services import rollout_service

    canary = await _seed_tenant(image_tag=OLD_TAG, is_canary=True)
    victim = await _seed_tenant(image_tag=OLD_TAG)
    rollout = await _seed_rollout(
        status="running", completed_at=None,
        started_at=datetime.utcnow() - timedelta(minutes=1),
    )

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        if user_id == canary.user_id:
            return {"health_checks_passed": 3, "duration_ms": 40}
        # Victim: upgrade AND rollback both blow up (bridge restarting —
        # the 502 class from the incident) → attempt 'rollback_failed'.
        raise RuntimeError("Server error '502 Bad Gateway'")

    async def fake_telegram(level, message):
        pass

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram), \
         patch("app.services.pool_service.notify_pool_image_refresh", AsyncMock()):
        async with async_session_maker() as db:
            attached = (await db.execute(
                select(Rollout).where(Rollout.id == rollout.id)
            )).scalar_one()
            await rollout_service._drive_rollout(db, attached)

    async with async_session_maker() as db:
        refreshed = (await db.execute(
            select(Rollout).where(Rollout.id == rollout.id)
        )).scalar_one()
        assert refreshed.status == "complete_with_failures", (
            "a rollout that failed a tenant must NOT report 'complete' — "
            f"got {refreshed.status!r}"
        )
        assert "1 ok, 1 failed/rolled-back of 2 total" in (refreshed.notes or "")
        assert refreshed.completed_at is not None
        # Still terminal: must not hold the rollout lock.
        assert await rollout_service.active_rollout(db) is None
    # `victim` seeded the failing side; silence the linters' unused warning.
    assert victim.user_id != canary.user_id


# ─── D2: convergence sweep ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_upgrades_only_divergent_eligible_tenants():
    """The sweep re-drives ONLY divergent tenants that pass the
    _running_tenants eligibility filter (running, unpinned, not
    pool-bound) — under a fresh trigger='sweep' rollout with ordinary
    attempt accounting."""
    from app.db import async_session_maker
    from app.db.models import RolloutAttempt
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    converged = await _seed_tenant(image_tag=NEW_TAG)
    divergent = await _seed_tenant(image_tag=OLD_TAG)
    pool_bound = await _seed_tenant(image_tag=OLD_TAG, container_name="toup-agent-pool-12")
    pinned = await _seed_tenant(image_tag=OLD_TAG, pin_image_tag=OLD_TAG)

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {"health_checks_passed": 3, "duration_ms": 12}

    async def fake_telegram(level, message):
        pass

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        await rollout_service._convergence_sweep_once()

    assert calls == [(divergent.user_id, NEW_TAG)], (
        f"sweep must upgrade exactly the divergent eligible tenant; got {calls!r} "
        f"(converged={converged.user_id[:8]}, pool={pool_bound.user_id[:8]}, "
        f"pinned={pinned.user_id[:8]})"
    )

    sweeps = await _sweep_rollouts()
    assert len(sweeps) == 1
    assert sweeps[0].image_tag == NEW_TAG
    assert sweeps[0].status == "complete"
    assert sweeps[0].completed_at is not None
    assert "convergence sweep completed: 1 ok, 0 failed" in (sweeps[0].notes or "")

    async with async_session_maker() as db:
        attempts = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.rollout_id == sweeps[0].id)
        )).scalars().all()
        assert len(attempts) == 1
        assert attempts[0].status == "ok"
        assert attempts[0].container_id == divergent.id


@pytest.mark.asyncio
async def test_sweep_failure_marks_sweep_rollout_complete_with_failures():
    """A sweep whose tenant fails again ends 'complete_with_failures' —
    it stays visible AND counts toward the 24h cap (its row has
    trigger='sweep' + the target tag)."""
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    await _seed_tenant(image_tag=OLD_TAG)

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        raise RuntimeError("Server error '502 Bad Gateway'")

    async def fake_telegram(level, message):
        pass

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        await rollout_service._convergence_sweep_once()

    sweeps = await _sweep_rollouts()
    assert len(sweeps) == 1
    assert sweeps[0].status == "complete_with_failures"
    assert "1 failed/rolled-back of 1 divergent" in (sweeps[0].notes or "")


@pytest.mark.asyncio
async def test_sweep_noops_when_rollout_active():
    """An active (pending/running) rollout owns the fleet — the sweep
    must not race it."""
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    await _seed_tenant(image_tag=OLD_TAG)
    await _seed_rollout(
        status="running", completed_at=None,
        started_at=datetime.utcnow(), last_progress_at=datetime.utcnow(),
    )

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {}

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._convergence_sweep_once()

    assert calls == [], "sweep must not touch tenants while a rollout is active"
    assert await _sweep_rollouts() == []


@pytest.mark.asyncio
async def test_sweep_noops_when_no_divergence():
    """Converged fleet → no sweep rollout, no bridge calls, no alerts."""
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    await _seed_tenant(image_tag=NEW_TAG)

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {}

    sent: list[tuple[str, str]] = []

    async def fake_telegram(level, message):
        sent.append((level, message))

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        await rollout_service._convergence_sweep_once()

    assert calls == []
    assert sent == []
    assert await _sweep_rollouts() == []


def test_sweep_flag_defaults_on():
    """The kill switch ships ON — the sweep is the safety net, disabling
    it is the exceptional operator action (ROLLOUT_CONVERGENCE_SWEEP=false)."""
    from app.config import Settings

    assert Settings.model_fields["rollout_convergence_sweep"].default is True


@pytest.mark.asyncio
async def test_sweep_noops_when_flag_off():
    """Kill switch: settings.rollout_convergence_sweep=False disables the
    sweep entirely — checked at the top, before any DB read."""
    from app.config import settings as _settings
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    await _seed_tenant(image_tag=OLD_TAG)

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {}

    _orig = _settings.rollout_convergence_sweep
    _settings.rollout_convergence_sweep = False
    try:
        with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
             patch.object(rollout_service, "_send_telegram", AsyncMock()):
            await rollout_service._convergence_sweep_once()
    finally:
        _settings.rollout_convergence_sweep = _orig

    assert calls == []
    assert await _sweep_rollouts() == []


@pytest.mark.asyncio
async def test_sweep_cap_exhausted_alerts_once_and_skips():
    """3 prior sweeps for the same tag inside 24h → no fourth sweep;
    one 'convergence sweep exhausted' alert naming the divergent
    prefix(es), deduped on subsequent ticks (the reconciler ticks every
    30s — re-alerting each tick would flood Telegram)."""
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    divergent = await _seed_tenant(image_tag=OLD_TAG)
    for _ in range(rollout_service._SWEEP_MAX_PER_TAG_24H):
        await _seed_rollout(
            status="complete_with_failures", trigger="sweep", image_tag=NEW_TAG,
            started_at=datetime.utcnow() - timedelta(hours=1),
            completed_at=datetime.utcnow() - timedelta(hours=1),
        )

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {}

    sent: list[tuple[str, str]] = []

    async def fake_telegram(level, message):
        sent.append((level, message))

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", fake_telegram):
        await rollout_service._convergence_sweep_once()
        # Second tick: still capped — must not re-alert.
        await rollout_service._convergence_sweep_once()

    assert calls == [], "capped tag must not start another sweep"
    assert len(await _sweep_rollouts()) == rollout_service._SWEEP_MAX_PER_TAG_24H
    exhausted = [(lvl, m) for lvl, m in sent if "convergence sweep exhausted" in m]
    assert len(exhausted) == 1, (
        f"expected exactly ONE exhausted alert across two ticks, got: {sent!r}"
    )
    level, message = exhausted[0]
    assert level == "critical"
    assert NEW_TAG in message
    assert divergent.user_id[:8] in message


@pytest.mark.asyncio
async def test_sweep_cap_ignores_sweeps_older_than_24h():
    """The cap is per-24h, not forever: stale sweep history must not
    block a fresh convergence attempt."""
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    divergent = await _seed_tenant(image_tag=OLD_TAG)
    for _ in range(rollout_service._SWEEP_MAX_PER_TAG_24H):
        await _seed_rollout(
            status="complete_with_failures", trigger="sweep", image_tag=NEW_TAG,
            started_at=datetime.utcnow() - timedelta(hours=25),
            completed_at=datetime.utcnow() - timedelta(hours=25),
        )

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {"health_checks_passed": 3, "duration_ms": 12}

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._convergence_sweep_once()

    assert calls == [(divergent.user_id, NEW_TAG)], (
        ">24h-old sweeps must not count toward the cap"
    )


# ─── Sweep stand-down guards (adversarial-review round) ────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "newest_status", ["aborted_orphan", "cancelled", "aborted_canary_failed"]
)
async def test_sweep_stands_down_when_newest_rollout_is_a_newer_tag(newest_status):
    """Incident replay guard: [complete(OLD), aborted_orphan(NEW)] + a
    tenant already upgraded to NEW must NOT be downgraded back to OLD.

    Without the guard, the reconciler orphans the NEW-tag rollout and the
    sweep fires in the SAME loop iteration: target = the previous
    complete tag, the freshly-upgraded tenants read as 'divergent', and
    the sweep blue-green DOWNGRADES them while holding the lock the CI
    gate's re-fire needs. Same shape for 'cancelled' (cancel_rollout
    documents it does NOT revert already-upgraded tenants) and
    'aborted_canary_failed'."""
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(
        status="complete", image_tag=OLD_TAG,
        started_at=datetime.utcnow() - timedelta(minutes=40),
        completed_at=datetime.utcnow() - timedelta(minutes=35),
    )
    await _seed_rollout(
        status=newest_status, image_tag=NEW_TAG,
        started_at=datetime.utcnow() - timedelta(minutes=5),
        completed_at=datetime.utcnow() - timedelta(minutes=1),
    )
    upgraded = await _seed_tenant(image_tag=NEW_TAG)  # canary et al., already swapped

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {"health_checks_passed": 3, "duration_ms": 12}

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._convergence_sweep_once()

    assert calls == [], (
        f"sweep must stand down when the newest rollout ({newest_status}, NEW tag) "
        f"is not the target — downgrading {upgraded.user_id[:8]} back to OLD "
        f"re-creates the incident"
    )
    assert await _sweep_rollouts() == []


@pytest.mark.asyncio
async def test_sweep_proceeds_when_newest_nontarget_rollout_has_same_tag():
    """The stand-down guard keys on a DIFFERENT tag: an orphaned re-drive
    of the SAME tag leaves stragglers the sweep can still safely heal —
    converging toward the same tag is definitionally not a downgrade."""
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(
        status="complete", image_tag=NEW_TAG,
        started_at=datetime.utcnow() - timedelta(minutes=40),
        completed_at=datetime.utcnow() - timedelta(minutes=35),
    )
    await _seed_rollout(
        status="aborted_orphan", image_tag=NEW_TAG,
        started_at=datetime.utcnow() - timedelta(minutes=5),
        completed_at=datetime.utcnow() - timedelta(minutes=1),
    )
    divergent = await _seed_tenant(image_tag=OLD_TAG)

    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {"health_checks_passed": 3, "duration_ms": 12}

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._convergence_sweep_once()

    assert calls == [(divergent.user_id, NEW_TAG)]


@pytest.mark.asyncio
async def test_sweep_waits_out_min_gap_between_sweeps():
    """No back-to-back sweeps: a failed sweep completes in seconds while
    the bridge outage it hit spans minutes — without spacing, the whole
    3-per-24h budget burned in ~90s of consecutive 30s ticks and the tag
    was then capped for 24h. The tick right after a failed sweep must
    NOT start sweep 2; once the gap has elapsed it must."""
    from app.db import async_session_maker
    from app.db.models import Rollout
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    await _seed_rollout(status="complete", image_tag=NEW_TAG)
    divergent = await _seed_tenant(image_tag=OLD_TAG)

    async def failing_upgrade(_db, user_id, image_tag, rollout_id=None):
        raise RuntimeError("Server error '502 Bad Gateway'")

    with patch.object(rollout_service, "upgrade_tenant_image", failing_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        # Sweep 1 fires and fails fast (bridge restarting).
        await rollout_service._convergence_sweep_once()
        assert len(await _sweep_rollouts()) == 1
        # Next tick, seconds later: must stand down, not burn slot 2.
        await rollout_service._convergence_sweep_once()
        await rollout_service._convergence_sweep_once()
        assert len(await _sweep_rollouts()) == 1, (
            "consecutive ticks must not start another sweep inside the min gap"
        )

    # Age sweep 1 past the gap → the next tick may retry.
    async with async_session_maker() as db:
        row = (await db.execute(
            select(Rollout).where(Rollout.trigger == "sweep")
        )).scalar_one()
        row.started_at = datetime.utcnow() - timedelta(
            minutes=rollout_service._SWEEP_MIN_GAP_MIN + 1
        )
        await db.commit()

    calls: list[tuple[str, str]] = []

    async def ok_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {"health_checks_passed": 3, "duration_ms": 12}

    with patch.object(rollout_service, "upgrade_tenant_image", ok_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._convergence_sweep_once()

    assert calls == [(divergent.user_id, NEW_TAG)], (
        "once the gap has elapsed the sweep must retry"
    )
    assert len(await _sweep_rollouts()) == 2


@pytest.mark.asyncio
async def test_run_rollout_task_crash_stamps_aborted_orphan_not_complete():
    """The orchestrator crash catch-all must NOT stamp 'complete': the
    sweep targets the newest complete/complete_with_failures row, so a
    crash-stamped 'complete' (crash can predate canary observation) would
    have the sweep batch-drive the fleet onto a tag whose canary never
    passed. 'aborted_orphan' is equally terminal and the CI gate already
    re-fires on it."""
    from app.db import async_session_maker
    from app.db.models import Rollout
    from app.services import rollout_service

    _clear_sweep_alert_dedup()
    # Settled history on OLD; the fleet is converged to it.
    await _seed_rollout(
        status="complete", image_tag=OLD_TAG,
        started_at=datetime.utcnow() - timedelta(hours=1),
        completed_at=datetime.utcnow() - timedelta(hours=1),
    )
    await _seed_tenant(image_tag=OLD_TAG)
    crashed = await _seed_rollout(
        status="pending", image_tag=NEW_TAG, completed_at=None,
        started_at=datetime.utcnow(),
    )

    with patch.object(
        rollout_service, "_drive_rollout",
        AsyncMock(side_effect=RuntimeError("pgbouncer hiccup before canary")),
    ), patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._run_rollout_task(crashed.id)

    async with async_session_maker() as db:
        refreshed = (await db.execute(
            select(Rollout).where(Rollout.id == crashed.id)
        )).scalar_one()
        assert refreshed.status == "aborted_orphan", (
            "a crashed orchestrator must not report 'complete' — "
            f"got {refreshed.status!r}"
        )
        assert "ORCHESTRATOR CRASH" in (refreshed.notes or "")
        assert refreshed.completed_at is not None

    # And the crashed tag must never become the sweep target: the fleet
    # is converged to OLD, so a correct sweep does nothing.
    calls: list[tuple[str, str]] = []

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        calls.append((user_id, image_tag))
        return {}

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._convergence_sweep_once()

    assert calls == [], "crash-stamped rollout must not become the sweep target"
    assert await _sweep_rollouts() == []


# ─── D1 (resume path): honest terminal status ──────────────────────


@pytest.mark.asyncio
async def test_resume_rollout_task_stamps_complete_with_failures_on_partial_failure():
    """The resume path — what actually runs after a Railway redeploy kills
    the driver mid-observation, i.e. the incident's own trigger — must
    apply the same honest-status rule as _drive_rollout. A regression
    that reverts only this stamp to unconditional 'complete' would pass
    the rest of the suite."""
    from app.db import async_session_maker
    from app.db.models import Rollout
    from app.services import rollout_service

    canary = await _seed_tenant(image_tag=NEW_TAG, is_canary=True)
    victim = await _seed_tenant(image_tag=OLD_TAG)
    rollout = await _seed_rollout(
        status="running", phase="canary_observing", completed_at=None,
        canary_prefix=canary.user_id[:8], canary_wait_minutes=5,
        resume_after=datetime.utcnow() - timedelta(minutes=1),
        started_at=datetime.utcnow() - timedelta(minutes=10),
        last_progress_at=datetime.utcnow() - timedelta(minutes=4),
    )
    await _seed_attempt(
        rollout.id, canary.id, status="ok", prior_tag=OLD_TAG, new_tag=NEW_TAG,
        completed_at=datetime.utcnow() - timedelta(minutes=9),
    )

    async def fake_upgrade(_db, user_id, image_tag, rollout_id=None):
        if user_id == canary.user_id:
            return {"health_checks_passed": 3, "duration_ms": 40}
        # The victim's upgrade AND rollback both 502 (bridge restarting).
        raise RuntimeError("Server error '502 Bad Gateway'")

    with patch.object(rollout_service, "upgrade_tenant_image", fake_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()), \
         patch("app.services.pool_service.notify_pool_image_refresh", AsyncMock()):
        await rollout_service._resume_rollout_task(rollout.id)

    async with async_session_maker() as db:
        refreshed = (await db.execute(
            select(Rollout).where(Rollout.id == rollout.id)
        )).scalar_one()
        assert refreshed.status == "complete_with_failures", (
            "a RESUMED rollout that failed a tenant must not report 'complete' — "
            f"got {refreshed.status!r}"
        )
        assert refreshed.completed_at is not None
        assert await rollout_service.active_rollout(db) is None
    assert victim.user_id != canary.user_id


# ─── D3: orphaned-attempt cleanup ──────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_auto_orphan_also_orphans_upgrading_attempts():
    """When the reconciler auto-orphans a stale-heartbeat rollout, its
    in-flight 'upgrading' attempts must be closed as 'orphaned' in the
    same pass — 01a945e2 left four such rows dangling forever. Attempts
    already terminal ('ok') stay untouched."""
    from app.db import async_session_maker
    from app.db.models import Rollout, RolloutAttempt
    from app.services import rollout_service

    tenant = await _seed_tenant(image_tag=OLD_TAG)
    stale = await _seed_rollout(
        status="running", phase="batching", completed_at=None,
        started_at=datetime.utcnow() - timedelta(minutes=10),
        last_progress_at=datetime.utcnow() - timedelta(minutes=10),
    )
    stuck_id = await _seed_attempt(stale.id, tenant.id, status="upgrading")
    ok_id = await _seed_attempt(
        stale.id, tenant.id, status="ok",
        completed_at=datetime.utcnow() - timedelta(minutes=9),
    )

    with patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._reconcile_once()

    async with async_session_maker() as db:
        refreshed = (await db.execute(
            select(Rollout).where(Rollout.id == stale.id)
        )).scalar_one()
        assert refreshed.status == "aborted_orphan"

        stuck = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == stuck_id)
        )).scalar_one()
        assert stuck.status == "orphaned", (
            "dead driver's 'upgrading' attempt must be closed with the rollout"
        )
        assert "driver died" in (stuck.error or "")
        assert "bridge may have completed the swap" in (stuck.error or "")
        assert stuck.completed_at is not None

        ok_row = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == ok_id)
        )).scalar_one()
        assert ok_row.status == "ok", "terminal attempts must not be rewritten"


@pytest.mark.asyncio
async def test_reconcile_heals_upgrading_attempts_of_already_terminal_rollout():
    """Backfill path: 'upgrading' attempts whose parent rollout is ALREADY
    terminal (the four existing 01a945e2 rows) get closed on the next
    tick — while a live rollout's in-flight attempt stays untouched."""
    from app.db import async_session_maker
    from app.db.models import Rollout, RolloutAttempt
    from app.services import rollout_service

    tenant = await _seed_tenant(image_tag=OLD_TAG)
    # The incident shape: rollout long since orphaned, attempts dangling.
    terminal = await _seed_rollout(
        status="aborted_orphan",
        started_at=datetime.utcnow() - timedelta(hours=2),
        completed_at=datetime.utcnow() - timedelta(hours=2),
    )
    dangling_id = await _seed_attempt(terminal.id, tenant.id, status="upgrading")

    # Control: a healthy running rollout mid-bridge-call — its 'upgrading'
    # attempt is live work, not debris.
    live = await _seed_rollout(
        status="running", phase="batching", completed_at=None,
        started_at=datetime.utcnow() - timedelta(minutes=1),
        last_progress_at=datetime.utcnow(),
    )
    live_attempt_id = await _seed_attempt(live.id, tenant.id, status="upgrading")

    with patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._reconcile_once()

    async with async_session_maker() as db:
        dangling = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == dangling_id)
        )).scalar_one()
        assert dangling.status == "orphaned", (
            "terminal-parent 'upgrading' rows must be healed by the backfill"
        )
        assert "driver died" in (dangling.error or "")
        assert dangling.completed_at is not None

        live_attempt = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == live_attempt_id)
        )).scalar_one()
        assert live_attempt.status == "upgrading", (
            "a live rollout's in-flight attempt must NOT be touched"
        )
        live_rollout = (await db.execute(
            select(Rollout).where(Rollout.id == live.id)
        )).scalar_one()
        assert live_rollout.status == "running"


@pytest.mark.asyncio
async def test_backfill_spares_cancelled_rollouts_live_batch():
    """cancel_rollout documents that the in-flight batch COMPLETES (up to
    ~210s of live 'upgrading' rows under a status='cancelled' parent).
    The backfill must not stamp those live attempts 'orphaned' on the
    next 30s tick — only attempts older than the driver's hard timeout
    are debris."""
    from app.db import async_session_maker
    from app.db.models import RolloutAttempt
    from app.services import rollout_service

    tenant = await _seed_tenant(image_tag=OLD_TAG)
    cancelled = await _seed_rollout(
        status="cancelled",
        started_at=datetime.utcnow() - timedelta(minutes=2),
        completed_at=datetime.utcnow() - timedelta(seconds=20),
    )
    # Live: the driver is still finishing its batch (attempt is 60s old,
    # hard timeout is 210s).
    live_id = await _seed_attempt(
        cancelled.id, tenant.id, status="upgrading",
        started_at=datetime.utcnow() - timedelta(seconds=60),
    )
    # Debris: past the hard timeout, no driver can still own it.
    debris_id = await _seed_attempt(
        cancelled.id, tenant.id, status="upgrading",
        started_at=datetime.utcnow() - timedelta(minutes=8),
    )

    with patch.object(rollout_service, "_send_telegram", AsyncMock()):
        await rollout_service._reconcile_once()

    async with async_session_maker() as db:
        live = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == live_id)
        )).scalar_one()
        assert live.status == "upgrading", (
            "a cancelled rollout's live in-flight attempt must not be "
            "stamped 'orphaned' while its driver can still be running"
        )
        debris = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == debris_id)
        )).scalar_one()
        assert debris.status == "orphaned", (
            "attempts past the hard timeout are debris and must be closed"
        )


@pytest.mark.asyncio
async def test_upgrade_one_ok_write_clears_stale_orphan_stamp():
    """Reconciler-vs-driver interleaving: a heartbeat-stale false positive
    stamps a LIVE attempt 'orphaned' + 'driver died...' while the driver
    awaits the bridge. The driver's terminal write wins on status — it
    must win on error too, or the permanent record reads status='ok'
    error='driver died...'."""
    from app.db import async_session_maker
    from app.db.models import RolloutAttempt
    from app.services import rollout_service

    tenant = await _seed_tenant(image_tag=OLD_TAG)
    rollout = await _seed_rollout(
        status="running", phase="batching", completed_at=None,
        started_at=datetime.utcnow() - timedelta(minutes=1),
    )

    async def racing_upgrade(_db, user_id, image_tag, rollout_id=None):
        # While the driver awaits the bridge, the reconciler (falsely)
        # orphans its attempt row — same write _orphan_stuck_attempts does.
        async with async_session_maker() as db:
            attempt = (await db.execute(
                select(RolloutAttempt).where(
                    RolloutAttempt.rollout_id == rollout_id,
                    RolloutAttempt.status == "upgrading",
                )
            )).scalar_one()
            attempt.status = "orphaned"
            attempt.error = rollout_service._ORPHANED_ATTEMPT_ERROR
            attempt.completed_at = datetime.utcnow()
            await db.commit()
        return {"health_checks_passed": 3, "duration_ms": 25}

    with patch.object(rollout_service, "upgrade_tenant_image", racing_upgrade), \
         patch.object(rollout_service, "_send_telegram", AsyncMock()):
        result = await rollout_service._upgrade_one(None, rollout, tenant, NEW_TAG)

    assert result.status == "ok"
    assert result.error is None, (
        f"the ok write must clear the stale orphan stamp — got error={result.error!r}"
    )

    async with async_session_maker() as db:
        persisted = (await db.execute(
            select(RolloutAttempt).where(RolloutAttempt.id == result.id)
        )).scalar_one()
        assert persisted.status == "ok"
        assert persisted.error is None
