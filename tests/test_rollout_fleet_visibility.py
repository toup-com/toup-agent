"""A fleet split across images must be visible without asking the bridge by hand.

WHY THIS EXISTS — measured 2026-09-07 15:xx UTC, twenty-five hours after the
rollout was paused:

    GET /v1/pool/health  ->  current_image_tag 7edaed3ab644
                             image_lag_seconds 83405
                             members {total 84, generic 10, assigned 74, dead 0}
    GET /v1/pool/list    ->  44 ASSIGNED on 7edaed3ab644
                             30 ASSIGNED on 11022e00cedf   <-- two images behind
                             10 GENERIC  on 7edaed3ab644

Nothing in the platform could see that. `image_lag_seconds` is `now -
current_image_tag_ts`, i.e. how long ago an operator SET the tag — it is 83405
whether every slot is converged or none of them are. `assigned_stale` is written
only inside the bridge's auto-upgrade branch, and that branch is skipped while
`BRIDGE_POOL_AUTO_UPGRADE_ASSIGNED=0`, so `pool_churn_snapshot` reads stale=0
and `pool_is_busy` reports a quiet pool — the paused fleet and the converged
fleet are the same reading. 41 users sat on the hot image for nine hours with no
alert, and 30 still sit two images behind.

Run:
  cd backend && PYTHONPATH=. pytest tests/test_rollout_fleet_visibility.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.services.rollout_service as RS  # noqa: E402


CURRENT = "ghcr.io/toup-com/toup-agent:7edaed3ab644"
OLD = "ghcr.io/toup-com/toup-agent:11022e00cedf"

# The live shape, as the bridge contract renders it.
SPLIT = {
    "images": {CURRENT: 54, OLD: 30},
    "current_image_tag": CURRENT,
    "assigned_on_current": 44,
    "assigned_on_other": 30,
    "generic_on_other": 0,
    "auto_upgrade_assigned": False,
    "last_rollout_tick": 1788710000.0,
}
CONVERGED = {
    "images": {CURRENT: 84},
    "current_image_tag": CURRENT,
    "assigned_on_current": 74,
    "assigned_on_other": 0,
    "generic_on_other": 0,
    "auto_upgrade_assigned": True,
    "last_rollout_tick": 1788790000.0,
}


@pytest.fixture(autouse=True)
def _clean_state():
    RS._reset_fleet_watch_state()
    yield
    RS._reset_fleet_watch_state()


def _health(fleet):
    body = {
        "ok": True,
        "current_image_tag": CURRENT,
        "image_lag_seconds": 83405,
        "members": {"total": 84, "generic": 10, "assigned": 74, "dead": 0},
    }
    if fleet is not None:
        body["fleet"] = fleet
    return body


# ─── the pure warning rules ───────────────────────────────────────


class TestWarnings:
    def test_a_fresh_split_is_not_yet_an_alert(self):
        """A rollout in flight IS a split. The alert is for a split that has
        stopped moving — two hours is past every measured convergence window
        (49 of 50 pool members inside 30 minutes, 2026-08-01)."""
        w = RS._fleet_warnings(dict(SPLIT, auto_upgrade_assigned=True), split_for_s=600.0)
        assert w == []

    def test_a_split_older_than_two_hours_alerts(self):
        w = RS._fleet_warnings(dict(SPLIT, auto_upgrade_assigned=True), split_for_s=3 * 3600)
        assert w, "a fleet split for 3h must warn"
        assert any("30" in s for s in w), w

    def test_auto_upgrade_off_with_slots_behind_alerts_immediately(self):
        """The pause is the thing nobody can see: it was set on the bridge's
        root-only env at 14:35 and nothing reported it for nine hours."""
        w = RS._fleet_warnings(SPLIT, split_for_s=60.0)
        assert any("auto-upgrade" in s.lower() for s in w), w

    def test_auto_upgrade_off_with_a_converged_fleet_is_fine(self):
        w = RS._fleet_warnings(dict(CONVERGED, auto_upgrade_assigned=False), split_for_s=0.0)
        assert w == []

    def test_a_converged_fleet_never_warns(self):
        w = RS._fleet_warnings(CONVERGED, split_for_s=10 * 3600)
        assert w == []


# ─── the snapshot + the split clock ───────────────────────────────


class TestFleetStatus:
    @pytest.mark.asyncio
    async def test_it_reads_the_fleet_block(self):
        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(_health(SPLIT), "ok"))):
            st = await RS.fleet_status(now=1000.0)
        assert st["available"] is True
        assert st["fleet"]["assigned_on_other"] == 30
        assert st["warnings"], "auto-upgrade is off with 30 behind"

    @pytest.mark.asyncio
    async def test_a_bridge_without_the_block_is_unavailable_not_converged(self):
        """An older bridge must never read as 'fleet converged' — that is the
        exact misreading `image_lag_seconds` already invites."""
        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(_health(None), "ok"))):
            st = await RS.fleet_status(now=1000.0)
        assert st["available"] is False
        assert st["fleet"] is None
        assert st["warnings"] == []
        assert "fleet" in st["reason"].lower()

    @pytest.mark.asyncio
    async def test_an_unreachable_bridge_is_unavailable(self):
        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(None, "ReadTimeout"))):
            st = await RS.fleet_status(now=1000.0)
        assert st["available"] is False
        assert "ReadTimeout" in st["reason"]

    @pytest.mark.asyncio
    async def test_the_split_clock_starts_at_the_first_sighting_and_resets(self):
        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(_health(SPLIT), "ok"))):
            first = await RS.fleet_status(now=1_000.0)
            later = await RS.fleet_status(now=1_000.0 + 7200)
        assert first["split_for_seconds"] == 0
        assert later["split_for_seconds"] == 7200
        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(_health(CONVERGED), "ok"))):
            done = await RS.fleet_status(now=1_000.0 + 7300)
        assert done["split_for_seconds"] == 0
        assert done["warnings"] == []


# ─── the periodic watch ───────────────────────────────────────────


class TestFleetWatch:
    @pytest.mark.asyncio
    async def test_a_long_split_pages_once_not_every_tick(self):
        alerts = []

        # The fake answers True: `_send_telegram` now reports whether Telegram
        # CONFIRMED the send, and the suppression window is only consumed on a
        # confirmed one. A fake returning None would take the not-confirmed
        # path instead (retry if configured, consume if not), which is a
        # different test — see test_rollout_fleet_alerting.py.
        async def _alert(level, msg):
            alerts.append((level, msg))
            return True

        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(_health(SPLIT), "ok"))), \
             patch.object(RS, "_send_telegram", _alert):
            await RS._fleet_watch_once(now=1_000.0)
            for i in range(10):
                await RS._fleet_watch_once(now=1_000.0 + 30 * (i + 1))
        assert len(alerts) == 1, f"reconciler ticks every 30s; got {len(alerts)} alerts"
        assert alerts[0][0] == "warning"
        assert "30" in alerts[0][1]

    @pytest.mark.asyncio
    async def test_a_converged_fleet_never_pages(self):
        alerts = []

        async def _alert(level, msg):
            alerts.append((level, msg))
            return True

        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(_health(CONVERGED), "ok"))), \
             patch.object(RS, "_send_telegram", _alert):
            for i in range(5):
                await RS._fleet_watch_once(now=1_000.0 + 30 * i)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_a_bridge_failure_never_raises_into_the_reconciler(self):
        with patch.object(RS, "_bridge_get", AsyncMock(side_effect=RuntimeError("boom"))), \
             patch.object(RS, "_send_telegram", AsyncMock()):
            await RS._fleet_watch_once(now=1.0)   # must not raise


# ─── the reconciler actually runs it ──────────────────────────────


class TestReconcilerWiring:
    def test_the_reconciler_loop_calls_the_watch(self):
        """A watch nobody ticks reports nothing. Source-order probe: the call
        must be inside the reconciler's tick body, before the tick sleep, and
        under its own try/except — like the convergence sweep beside it, so a
        bridge blip cannot kill the loop that also resumes rollouts.

        The body moved into `_rollout_reconciler_ticks` when the loop gained
        its shutdown lease release (the handler has to wrap every await in a
        tick, not just the sleep), so the probe reads the ticks function AND
        asserts the entry point still awaits it — a tick body nobody calls is
        the same silent nothing as a watch nobody ticks."""
        import inspect
        entry = inspect.getsource(RS.rollout_reconciler_loop)
        assert "await _rollout_reconciler_ticks()" in entry, \
            "the loop the app starts must still drive the tick body"
        lines = inspect.getsource(RS._rollout_reconciler_ticks).splitlines()
        call = [i for i, l in enumerate(lines) if "_fleet_watch_once" in l]
        assert call, "the reconciler must tick the fleet watch"
        i = call[0]
        assert any(l.strip() == "try:" for l in lines[max(0, i - 3):i]), \
            "the fleet watch call must sit under its own try:"
        assert any("except" in l for l in lines[i + 1:i + 5]), \
            "the fleet watch call must have its own except"
        # The literal 30 became `_RECONCILER_TICK_S` when the loop was
        # leader-gated: the tick and the lease TTL are one number.
        sleep = [j for j, l in enumerate(lines) if "await asyncio.sleep(" in l]
        assert sleep and i < sleep[-1], "the watch must run before the tick sleeps"


# ─── the admin API exposes it ─────────────────────────────────────


class TestFleetEndpoint:
    def test_the_route_exists_and_is_not_shadowed(self):
        """`/{rollout_id}` is declared in the same router and FastAPI matches
        in declaration order, so a `/fleet` added below it would answer
        "rollout fleet not found" forever."""
        import app.api.admin.rollouts as R
        paths = [r.path for r in R.router.routes]
        assert "/admin/rollout/fleet" in paths
        assert paths.index("/admin/rollout/fleet") < paths.index("/admin/rollout/{rollout_id}")

    @pytest.mark.asyncio
    async def test_it_answers_with_the_fleet_block(self):
        import app.api.admin.rollouts as R
        fn = next(r.endpoint for r in R.router.routes if r.path == "/admin/rollout/fleet")
        with patch.object(RS, "_bridge_get", AsyncMock(return_value=(_health(SPLIT), "ok"))):
            body = await fn(_=None)
        assert body["available"] is True
        assert body["fleet"]["assigned_on_other"] == 30
