"""The canary must be able to fail for the reason the 2026-09-06 image failed.

WHY THIS EXISTS — the production record

Rollout 48191bdd promoted `ghcr.io/toup-com/toup-agent:a962b7340717` at
2026-09-06 13:31–13:39 with `complete: 7 ok, 0 failed/rolled-back of 7 total`
and `health_checks_passed=3` on every attempt. The bridge then blue-greened 41
real users' pool slots onto it. That image's voice-task supervisor opened three
NullPool connections and three SELECTs per second on every container, bound or
not. Measured on the host nine hours later:

    image a962b7340717 (41 assigned)   18.0 % CPU each   6.99 xact/s per DB
    image 11022e00cedf (30 assigned)    3.25 %           1.46 xact/s
    image 7edaed3ab644 (13 slots)       3.42 %           1.40 xact/s

Host load 42–55 on 16 cores; tenant Postgres at 292–302 of max_connections=300;
pgbouncer died twice. Every gate the rollout owns was green throughout, because
every gate asks the same question: does /agent/health answer 200, and does one
turn complete. An idle container that burns 3.6× its neighbours answers yes to
both.

The gate added here asks the other question — what does this image COST at rest
— by sampling the bridge's `GET /v1/tenants/{prefix}/stats` for the canary and
for the peers that are still on the prior tag, during the canary stability hold.

Run:
  cd backend && PYTHONPATH=. pytest tests/test_rollout_resource_gate.py -q
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.services.rollout_service as RS  # noqa: E402


# The production numbers above, as the bridge contract renders them.
def _stats(prefix: str, cpu: float, xact: float, tag: str = "old") -> dict:
    return {
        "prefix": prefix,
        "container_name": f"toup-agent-{prefix}",
        "image_tag": f"ghcr.io/toup-com/toup-agent:{tag}",
        "cpu_pct_60s": cpu,
        "mem_mb": 812.0,
        "pg_backends": 3,
        "xact_per_s": xact,
        "sampled_at": 1788800000.0,
    }


HOT = _stats("533354ce", 18.0, 6.99, tag="a962b7340717")     # the incident image
COOL = _stats("533354ce", 3.42, 1.40, tag="7edaed3ab644")    # the fixed image
PEERS = [
    _stats("871bac24", 3.25, 1.46),
    _stats("2739b5c6", 3.10, 1.52),
    _stats("c47c5b4b", 3.60, 1.38),
]


def _fetch(table: dict, missing_status: str = "HTTP 404 (route not implemented)"):
    """Build a stats_fetch fake over a {prefix: body-or-None} table."""

    async def _f(prefix: str):
        body = table.get(prefix)
        if body is None:
            return None, missing_status
        parsed = RS.TenantStats.from_body(body)
        return parsed, "ok"

    return _f


# ─── the pure verdict ─────────────────────────────────────────────


class TestVerdict:
    def test_the_incident_image_is_rejected(self):
        canary = RS.TenantStats.from_body(HOT)
        peers = [RS.TenantStats.from_body(p) for p in PEERS]
        ok, why = RS._resource_gate_verdict(
            canary, peers, cpu_pct_max=12.0, cpu_ratio_max=2.5, xact_ratio_max=3.0
        )
        assert ok is False, f"a962b734 must not pass the resource gate; got {why}"
        assert "cpu" in why.lower()

    def test_the_fixed_image_passes(self):
        canary = RS.TenantStats.from_body(COOL)
        peers = [RS.TenantStats.from_body(p) for p in PEERS]
        ok, why = RS._resource_gate_verdict(
            canary, peers, cpu_pct_max=12.0, cpu_ratio_max=2.5, xact_ratio_max=3.0
        )
        assert ok is True, why

    def test_the_absolute_ceiling_fires_without_any_peers(self):
        """A rollout with no comparable peer still must not promote a container
        burning 18 % at rest — the ratio rule has nothing to divide by."""
        canary = RS.TenantStats.from_body(HOT)
        ok, why = RS._resource_gate_verdict(
            canary, [], cpu_pct_max=12.0, cpu_ratio_max=2.5, xact_ratio_max=3.0
        )
        assert ok is False
        assert "12" in why

    def test_transaction_rate_alone_can_reject(self):
        """CPU can be hidden by a bigger host; the DB rate cannot. 7 tx/s
        against a 1.5 tx/s fleet is the same defect seen from Postgres."""
        canary = RS.TenantStats.from_body(_stats("533354ce", 6.0, 6.99))
        peers = [RS.TenantStats.from_body(p) for p in PEERS]
        ok, why = RS._resource_gate_verdict(
            canary, peers, cpu_pct_max=12.0, cpu_ratio_max=2.5, xact_ratio_max=3.0
        )
        assert ok is False
        assert "xact" in why.lower()

    def test_a_near_zero_median_does_not_make_every_ratio_infinite(self):
        """Peers reading ~0 (a quiet sample window) must not turn the ratio
        rule into 'fail on any measurable CPU' — that would make the gate a
        coin toss on the sampling window rather than a measurement."""
        canary = RS.TenantStats.from_body(_stats("533354ce", 1.2, 0.3))
        peers = [RS.TenantStats.from_body(_stats(f"p{i}", 0.0, 0.0)) for i in range(3)]
        ok, why = RS._resource_gate_verdict(
            canary, peers, cpu_pct_max=12.0, cpu_ratio_max=2.5, xact_ratio_max=3.0
        )
        assert ok is True, why


class TestParsing:
    def test_a_body_missing_a_measurement_is_not_a_zero(self):
        """The bridge samples a cgroup and a pg_stat row; either can fail while
        the route still answers 200. Reading an absent number as 0.0 makes an
        unmeasured canary look free — the same failure as a gate that is not
        run, but with a passing reason string attached."""
        for missing in ("cpu_pct_60s", "xact_per_s"):
            body = dict(HOT)
            body.pop(missing)
            assert RS.TenantStats.from_body(body) is None, missing
            body = dict(HOT)
            body[missing] = None
            assert RS.TenantStats.from_body(body) is None, missing

    def test_the_bridge_field_names_are_the_contract(self):
        """The bridge ships `cpu_pct` over `cpu_window_s`, not `cpu_pct_60s`
        (integrator finding, 2026-09-07): a gate that only knew the proposed
        name would have called every real body malformed and refused every
        rollout. Both spellings parse; the value is the same number."""
        body = dict(HOT); body["cpu_pct"] = body.pop("cpu_pct_60s"); body["cpu_window_s"] = 5.0
        st = RS.TenantStats.from_body(body)
        assert st is not None and st.cpu_pct_60s == 18.0 and st.xact_per_s == 6.99

    def test_an_unmeasured_xact_with_a_reason_keeps_the_cpu_rules(self):
        """The bridge answers `xact_per_s: null` + `pg_reason` when psql cannot
        reach the tenant DB. That is a reading, not a zero: the CPU ceiling
        still refuses the incident image, and the verdict says what was skipped."""
        body = dict(HOT); body["xact_per_s"] = None; body["pg_reason"] = "tenant database unreachable from the bridge"
        st = RS.TenantStats.from_body(body)
        assert st is not None and st.xact_per_s is None
        ok, why = RS._resource_gate_verdict(
            st, [], cpu_pct_max=12.0, cpu_ratio_max=2.5, xact_ratio_max=3.0)
        assert ok is False and "12%" in why and "skipped" in why
        cool = dict(COOL); cool["xact_per_s"] = None; cool["pg_reason"] = "x"
        ok, why = RS._resource_gate_verdict(
            RS.TenantStats.from_body(cool), [], cpu_pct_max=12.0, cpu_ratio_max=2.5, xact_ratio_max=3.0)
        assert ok is True and "skipped" in why

    @pytest.mark.asyncio
    async def test_a_partial_body_makes_the_gate_unavailable_not_passing(self):
        body = dict(HOT)
        body.pop("cpu_pct_60s")
        ok, why = await RS._resource_gate(
            "533354ce", [], stats_fetch=_fetch({"533354ce": body})
        )
        assert ok is False
        assert "unavailable" in why.lower()


# ─── availability: the route does not exist yet ───────────────────


class TestGateAvailability:
    @pytest.mark.asyncio
    async def test_a_missing_route_fails_the_rollout_by_default(self):
        """Measured 2026-09-07: GET /v1/tenants/533354ce/stats answers 404 on
        the deployed bridge. An unmeasurable canary is not a passed canary."""
        ok, why = await RS._resource_gate(
            "533354ce", ["871bac24"], stats_fetch=_fetch({})
        )
        assert ok is False, why
        assert "unavailable" in why.lower()

    @pytest.mark.asyncio
    async def test_the_operator_can_opt_out_explicitly(self):
        with patch.object(RS.settings, "rollout_allow_no_resource_gate", True):
            ok, why = await RS._resource_gate(
                "533354ce", ["871bac24"], stats_fetch=_fetch({})
            )
        assert ok is True
        assert "unavailable" in why.lower()
        assert "ROLLOUT_ALLOW_NO_RESOURCE_GATE" in why

    @pytest.mark.asyncio
    async def test_unreachable_peers_do_not_fail_a_healthy_canary(self):
        """Peers are a baseline, not a quorum: losing them costs the ratio
        rule, not the rollout. The absolute ceiling still applies."""
        ok, why = await RS._resource_gate(
            "533354ce", ["871bac24", "2739b5c6"],
            stats_fetch=_fetch({"533354ce": COOL}),
        )
        assert ok is True, why
        assert "0 peer" in why or "no peer" in why.lower()

    @pytest.mark.asyncio
    async def test_the_gate_can_be_switched_off_wholesale(self):
        with patch.object(RS.settings, "rollout_resource_gate_enabled", False):
            ok, why = await RS._resource_gate("533354ce", [], stats_fetch=_fetch({}))
        assert ok is True
        assert "disabled" in why.lower()


# ─── wired into the canary observation ────────────────────────────


def _health_client(body: bytes = b'{"ok":true,"turn_ready":true}'):
    import httpx

    ok = httpx.Response(200, content=body)

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url):
            return ok

        async def post(self, url, **kw):
            return httpx.Response(200, content=b'{"text":"OK"}')

    return _Client


class TestObserveIntegration:
    @pytest.mark.asyncio
    async def test_a_healthy_but_expensive_canary_is_refused(self):
        """Health 200 + turn_ready + a completed turn probe — exactly what
        a962b734 delivered — must still fail once the stats say 18 %."""
        with patch("httpx.AsyncClient", _health_client()), \
             patch("asyncio.sleep", AsyncMock()):
            passed, reason = await RS._observe_canary_signal(
                "https://test.local",
                cap_seconds=5.0, boot_gate_s=1.0, boot_interval_s=0.0,
                required_ok=3, stability_hold_s=0.5, stability_interval_s=0.0,
                canary_prefix="533354ce",
                peer_prefixes=["871bac24", "2739b5c6", "c47c5b4b"],
                stats_fetch=_fetch({"533354ce": HOT, **{p["prefix"]: p for p in PEERS}}),
            )
        assert passed is False, reason
        assert "resource gate" in reason.lower()

    @pytest.mark.asyncio
    async def test_a_healthy_cheap_canary_still_passes(self):
        with patch("httpx.AsyncClient", _health_client()), \
             patch("asyncio.sleep", AsyncMock()):
            passed, reason = await RS._observe_canary_signal(
                "https://test.local",
                cap_seconds=5.0, boot_gate_s=1.0, boot_interval_s=0.0,
                required_ok=3, stability_hold_s=0.5, stability_interval_s=0.0,
                canary_prefix="533354ce",
                peer_prefixes=["871bac24"],
                stats_fetch=_fetch({"533354ce": COOL, "871bac24": PEERS[0]}),
            )
        assert passed is True, reason


# ─── the wiring itself (a gate nobody calls is not a gate) ────────


def _fast_observe(calls: list):
    """The REAL `_observe_canary_signal` with its wall-clock phases shrunk.
    `asyncio.sleep` is mocked in these tests, so a 60 s stability hold is a
    60 s busy loop; the 0.5 s cap that used to bound it made the test depend
    on how fast the runner got from `Rollout(...)` to the loop's first
    `utcnow()`. On a 2-core hosted runner (CI run 34156…) that took longer
    than 0.5 s, the cap read as already passed — and the OLD loop answered a
    passed cap with `return True`, never calling this at all."""
    real = RS._observe_canary_signal

    async def _observe(*args, **kw):
        calls.append(dict(kw))
        kw.setdefault("boot_gate_s", 5.0)
        kw.setdefault("stability_hold_s", 0.2)
        return await real(*args, **kw)

    return _observe


async def _drive_wired_loop(resume_after: datetime, calls: list):
    """Drive the real `_canary_observe_loop` with the HOT canary and one cool
    peer — everything below the loop real except transport and time."""
    from app.db.models import Rollout, ManagedContainer

    rollout = Rollout(
        id="r-1", image_tag="ghcr.io/toup-com/toup-agent:a962b7340717",
        status="running", phase="canary_observing", trigger="ci",
        canary_wait_minutes=5,
        started_at=datetime.utcnow(),
        resume_after=resume_after,
    )
    canary = ManagedContainer(
        id="c-1", user_id="533354ce-0000-0000-0000-000000000000",
        container_name="toup-agent-533354ce", status="running",
        image_tag="ghcr.io/toup-com/toup-agent:11022e00cedf",
    )
    peer = ManagedContainer(
        id="c-2", user_id="871bac24-0000-0000-0000-000000000000",
        container_name="toup-agent-871bac24", status="running",
        image_tag="ghcr.io/toup-com/toup-agent:11022e00cedf",
    )

    db = MagicMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(
        scalar_one_or_none=MagicMock(return_value="agent-key"),
    ))

    rolled_back: list = []

    async def _fake_upgrade(_db, _rollout, container, tag):
        rolled_back.append(tag)
        return MagicMock(status="ok")

    table = {"533354ce": HOT, "871bac24": PEERS[0]}

    with patch("httpx.AsyncClient", _health_client()), \
         patch("asyncio.sleep", AsyncMock()), \
         patch.object(RS, "_observe_canary_signal", _fast_observe(calls)), \
         patch.object(RS, "_running_tenants", AsyncMock(return_value=[canary, peer])), \
         patch.object(RS, "_fetch_tenant_stats", _fetch(table)), \
         patch.object(RS, "_upgrade_one", _fake_upgrade), \
         patch.object(RS, "_send_telegram", AsyncMock()):
        proceed = await RS._canary_observe_loop(
            db, rollout, canary,
            "ghcr.io/toup-com/toup-agent:11022e00cedf",
            "https://agent-533354ce.agents.toup.ai",
        )
    return proceed, rollout, rolled_back


class TestCanaryLoopWiring:
    @pytest.mark.asyncio
    async def test_the_production_call_chain_reaches_the_gate(self):
        """`_canary_observe_loop` is the only production caller. If it stops
        passing the canary's prefix / peers / fetcher, every test above stays
        green and the fleet loses the gate — so drive the real loop and assert
        the rollout is aborted and the canary rolled back."""
        calls: list = []
        proceed, rollout, rolled_back = await _drive_wired_loop(
            datetime.utcnow() + timedelta(seconds=30), calls,
        )
        assert len(calls) == 1 and 0 < calls[0]["cap_seconds"] <= 30
        assert proceed is False, "the fleet must not be batched onto this image"
        assert rollout.status == "aborted_canary_failed"
        assert "resource gate" in (rollout.notes or "").lower()
        assert rolled_back == ["ghcr.io/toup-com/toup-agent:11022e00cedf"], \
            "the canary must be rolled back to its prior tag"

    @pytest.mark.asyncio
    async def test_a_passed_deadline_is_observed_not_waved_through(self):
        """FALSIFIER for the resume path. The reconciler invokes
        `_resume_rollout_task` precisely when `resume_after` has PASSED (the
        driver died mid-observation), and until this change the loop answered
        that shape with `return True` — "deadline already passed — proceeding
        to batch" — before `_observe_canary_signal` was called at all: no boot
        gate, no stability hold, no resource gate, no turn probe. The fleet was
        batched onto an image nobody had looked at, and #718's gate had nothing
        to say because it lives inside the call that was skipped. A passed
        deadline now re-arms the operator's window from now and observes."""
        calls: list = []
        proceed, rollout, rolled_back = await _drive_wired_loop(
            datetime.utcnow() - timedelta(seconds=10), calls,
        )
        assert calls, "a passed deadline skipped the observation entirely"
        assert calls[0]["cap_seconds"] == pytest.approx(5 * 60), \
            "the re-armed cap must be the operator's canary_wait_minutes"
        assert proceed is False, "the fleet must not be batched onto this image"
        assert rollout.status == "aborted_canary_failed"
        assert "resource gate" in (rollout.notes or "").lower()
        assert rolled_back == ["ghcr.io/toup-com/toup-agent:11022e00cedf"]


class TestObservationIsHeartbeated:
    def test_the_observation_beats_while_it_runs(self):
        """The reconciler orphans a rollout that has not stamped
        `last_progress_at` for 3 minutes. The observation is one await from its
        point of view, and it now costs boot (≤75 s) + stability (60 s) +
        resource gate (two bounded bridge reads) + turn probe (≤120 s) — past
        the threshold on the happy path. Without the beat a HEALTHY canary is
        orphaned and re-driven. Source-order probe: the loop call must be inside
        `_heartbeating`."""
        import inspect
        src = inspect.getsource(RS._drive_rollout)
        i = src.index("_canary_observe_loop(")
        window = src[max(0, i - 400):i]
        assert "_heartbeating(" in window, \
            "_canary_observe_loop must run inside a _heartbeating block"

    def test_the_resumed_observation_beats_too(self):
        """The resume path never needed a beat: with `resume_after` passed the
        loop returned at once. It now observes for real — boot (≤75 s) +
        stability (60 s) + gate + turn probe (≤120 s) — on the one path that
        runs after a redeploy killed the driver, and without a beat the
        reconciler's 180 s threshold would orphan the healthy canary it is
        re-checking."""
        import inspect
        src = inspect.getsource(RS._resume_rollout_task)
        i = src.index("_canary_observe_loop(")
        window = src[max(0, i - 400):i]
        assert "_heartbeating(" in window, \
            "_resume_rollout_task's observation must run inside a _heartbeating block"
