"""A rollout must not start while the bridge is still recycling the pool.

WHY THIS EXISTS — the production timeline, 2026-08-01

A rollout that COMPLETES calls notify_pool_image_refresh, and the bridge then
recycles every pool member. Measured from the bridge's own registry, members
whose state changed per 10-minute bucket after the 16:54 completion:

    16:50  ############## (14)
    17:00  #################### (20)
    17:10  ############### (15)
    17:20  # (1)

49 of 50 members inside 30 minutes. Every rollout started in that window died,
and each died differently because contention bites wherever it can:

    17:15:54  e9beb950  ConnectError after 8.9s      aborted_canary_failed
    17:19:24  e9beb950  heartbeat stale (3.4min)     aborted_orphan
    17:27:07  9413682b  0 health checks in 259s      aborted_canary_failed

The diffs were a config int, a prompt string and one db.commit() — nothing
that can fail a container boot. At 18:31, with the pool settled, the SAME
image upgraded the SAME canary in 97449ms with health_checks_passed=3.

Merges arrive in bursts (5 PRs in ~45 min that day), so every rollout after
the first was landing inside its predecessor's churn. That is the bug.

Run:
  cd backend && pytest tests/test_rollout_pool_quiescence.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.services.pool_service as PS  # noqa: E402


BUSY = {"upgrading": ["26"], "stale": 1, "spawning": 0, "draining": 0}
QUIET = {"upgrading": [], "stale": 0, "spawning": 0, "draining": 0}


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """The poll interval is 15s; tests must not actually wait."""
    monkeypatch.setattr(PS.asyncio, "sleep", AsyncMock())


class TestBusyPredicate:
    def test_an_upgrade_in_flight_is_busy(self):
        assert PS.pool_is_busy(BUSY) is True

    def test_a_settled_pool_is_not_busy(self):
        assert PS.pool_is_busy(QUIET) is False

    @pytest.mark.parametrize("field", ["spawning", "draining"])
    def test_spawning_or_draining_also_counts(self, field):
        """A generic member draining onto the new image spawns a container too
        — it competes for the same host I/O as the canary's green slot."""
        snap = dict(QUIET, **{field: 2})
        assert PS.pool_is_busy(snap) is True

    def test_stale_alone_counts(self):
        """assigned_stale>0 means the reconciler is ABOUT to start a batch on
        its next 30s tick. Starting a canary now just loses the race later."""
        assert PS.pool_is_busy(dict(QUIET, stale=3)) is True


class TestWait:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_quiescent(self):
        with patch.object(PS, "pool_churn_snapshot", AsyncMock(return_value=QUIET)):
            quiet, why = await PS.wait_for_pool_quiescence(900)
        assert quiet is True and "already quiescent" in why

    @pytest.mark.asyncio
    async def test_waits_then_proceeds_once_settled(self):
        snaps = [BUSY, BUSY, QUIET]
        with patch.object(PS, "pool_churn_snapshot", AsyncMock(side_effect=snaps)):
            quiet, why = await PS.wait_for_pool_quiescence(900)
        assert quiet is True and "quiescent after" in why

    @pytest.mark.asyncio
    async def test_timeout_PROCEEDS_rather_than_blocking(self, monkeypatch):
        """A late rollout beats a wedged one. The canary gate is still there to
        catch a genuinely bad image, so waiting forever buys nothing."""
        t = {"now": 0.0}
        monkeypatch.setattr(PS.time, "time", lambda: t["now"])

        async def _always_busy():
            t["now"] += 20.0          # advance past the poll interval
            return BUSY

        with patch.object(PS, "pool_churn_snapshot", _always_busy):
            quiet, why = await PS.wait_for_pool_quiescence(60)
        assert quiet is False, "must report it, not raise or hang"
        assert "still busy" in why

    @pytest.mark.asyncio
    async def test_unreachable_bridge_is_UNKNOWN_not_busy(self):
        """Losing telemetry must never wedge deploys. None means proceed."""
        with patch.object(PS, "pool_churn_snapshot", AsyncMock(return_value=None)):
            quiet, why = await PS.wait_for_pool_quiescence(900)
        assert quiet is True and "unreachable" in why

    @pytest.mark.asyncio
    async def test_bridge_lost_mid_wait_also_proceeds(self):
        with patch.object(PS, "pool_churn_snapshot", AsyncMock(side_effect=[BUSY, None])):
            quiet, why = await PS.wait_for_pool_quiescence(900)
        assert quiet is True and "unreachable mid-wait" in why

    @pytest.mark.asyncio
    async def test_zero_timeout_disables_the_wait(self):
        probe = AsyncMock(return_value=BUSY)
        with patch.object(PS, "pool_churn_snapshot", probe):
            quiet, why = await PS.wait_for_pool_quiescence(0)
        assert quiet is True and "disabled" in why
        probe.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_fires_every_poll(self):
        """The reconciler orphans a rollout after 3 minutes without progress —
        which is exactly how 17:19:24 died. A wait that does not heartbeat
        would replace one failure mode with the other."""
        beats = {"n": 0}

        async def _beat():
            beats["n"] += 1

        with patch.object(PS, "pool_churn_snapshot", AsyncMock(side_effect=[BUSY, BUSY, QUIET])):
            await PS.wait_for_pool_quiescence(900, heartbeat=_beat)
        assert beats["n"] == 2, f"expected a beat per poll, got {beats['n']}"

    @pytest.mark.asyncio
    async def test_a_raising_heartbeat_does_not_abort_the_wait(self):
        async def _bad_beat():
            raise RuntimeError("db went away")

        with patch.object(PS, "pool_churn_snapshot", AsyncMock(side_effect=[BUSY, QUIET])):
            quiet, _ = await PS.wait_for_pool_quiescence(900, heartbeat=_bad_beat)
        assert quiet is True


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_reads_the_reconciler_summary_shape_the_bridge_sends(self):
        """Pinned against a REAL /v1/pool/health body captured from production
        so a shape change is caught here rather than by a failed rollout."""
        real = {
            "ok": True, "target": 10, "min_k": 10, "max_k": 30,
            "current_image_tag": "ghcr.io/toup-com/toup-agent:746bc552f79f",
            "image_lag_seconds": 5261,
            "members": {"total": 50, "generic": 10, "assigned": 40,
                        "spawning": 0, "draining": 0, "dead": 0},
            "last_reconciler_summary": {
                "ts": 1785608491, "target": 10, "generic": 10, "assigned": 40,
                "dead": 0, "spawned": 0, "drained": 0, "errors": [],
                "assigned_upgrading": ["26"], "assigned_stale": 1,
            },
        }

        class _Resp:
            status_code = 200
            def json(self): return real

        class _Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, path): return _Resp()

        with patch("app.services.docker_host_service._bridge_client", lambda *a, **k: _Client()):
            snap = await PS.pool_churn_snapshot()

        assert snap == {"upgrading": ["26"], "stale": 1, "spawning": 0, "draining": 0}
        assert PS.pool_is_busy(snap) is True

    @pytest.mark.asyncio
    async def test_a_non_200_is_unknown(self):
        class _Resp:
            status_code = 503
            def json(self): return {}

        class _Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, path): return _Resp()

        with patch("app.services.docker_host_service._bridge_client", lambda *a, **k: _Client()):
            assert await PS.pool_churn_snapshot() is None


class TestRolloutWiring:
    @pytest.fixture(scope="class")
    def src(self):
        import app.services.rollout_service as RS
        return Path(RS.__file__).read_text()

    def test_the_wait_runs_BEFORE_the_canary_upgrade(self, src):
        """Ordering is the entire fix."""
        wait = src.index("wait_for_pool_quiescence(")
        canary = src.index('_heartbeating(rollout.id, "canary-upgrade")')
        assert wait < canary

    def test_the_wait_passes_a_heartbeat(self, src):
        i = src.index("wait_for_pool_quiescence(")
        assert "heartbeat=" in src[i:i + 300]

    def test_a_non_quiescent_start_is_recorded_on_the_rollout(self, src):
        """If it ever proceeds anyway, the next person reading the rollout row
        must be able to see that it did."""
        assert "pool NOT quiescent at canary start" in src
