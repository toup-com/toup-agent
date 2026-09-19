"""The sweep's mass-failure guard covers the DANGEROUS mode, and a deploy
does not blind the reconciler for ten minutes.

Two defects, both measured on the 2026-09-16 ToupInfraAlertsBot feed.

1. The guard was inverted in practice. `reclaim_stranded_users`' authenticated
   sweep counted only `status is None` — a probe that never got an answer —
   and exempted strikes only for that mode. A tenant-Postgres / pgbouncer
   death answers 500: at 05:17Z, 172 probes returned 5xx inside one minute,
   every one of them walked straight past the guard, so there was no
   sweep-quorum page and all 88 tenants took a strike toward
   PROBE_STRIKES_BEFORE_RESTART (2) — with RESTARTS_PER_TICK restarts per
   tick, during an outage that no container restart could fix. The guard fired
   on the harmless mode and stood down on the dangerous one.

   The per-agent paths must stay exactly as they were: a keyless 401/403 and a
   4xx routing miss are real single-agent faults with their own repair paths,
   and a MINORITY of 5xx is a genuinely sick container.

   The exemption is therefore per CLASS and by SHARE of the event, not by
   dominant mode: a pgbouncer death answers 500 on the agents that reach their
   DB and times out the ones that hang on the connect, so it is one event in
   two classes and never an exact tie. Scoped to the dominant mode, the
   smaller half took a strike each. See §2b.

2. A deploy blinded the reconciler for 733 s and 833 s (03:19:14Z and
   07:30:09Z), against a 178-200 s cadence with no other gap above 200 s in
   1,952 ticks. Two compounding causes: the lease was never released on
   shutdown, so the incoming process waited out the dead holder's TTL (3x the
   interval = 540 s), and the loop sleeps a full interval before its first
   tick. Both are covered below — including the part that must NOT change: the
   catch-up pass never runs the authenticated probe sweep, because
   PROBE_STRIKES_BEFORE_RESTART means "N consecutive ticks at the full
   cadence" and a compressed pass would spend both strikes in one interval.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. python -m pytest -q \
        tests/test_sweep_quorum_guard.py
"""

from __future__ import annotations

import asyncio
import itertools
import os
import uuid

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

_port = itertools.count(9900)


# ── helpers ───────────────────────────────────────────────────────────

async def _seed_running_user(container_name: str) -> str:
    """A managed user the sweep will probe: running row + a stored agent key."""
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig, ManagedContainer

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                    name="T", is_active=True))
        await db.flush()
        db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                           agent_url=f"https://agent-{uid[:8]}.test",
                           agent_api_key="platform-key"))
        db.add(ManagedContainer(
            id=str(uuid.uuid4()), user_id=uid, container_name=container_name,
            host_port=next(_port), db_name=f"toup_agent_{uid[:8]}",
            status="running",
        ))
        await db.commit()
    return uid


def _install_probe(monkeypatch, status_for) -> None:
    """Fake httpx.AsyncClient whose GET status is `status_for(url)`.

    `status_for` may return None to mean "raise", i.e. the transport mode.
    """
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        async def get(self, url, *a, **k):
            code = status_for(url)
            if code is None:
                raise RuntimeError("connect failed")
            return _Resp(code)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


async def _seed_fleet(n: int, prefix: str = "toup-agent-pool-") -> list:
    """N running managed users AND each one's probe URL, so one tick can hand
    different agents different answers — which is the only way to test that
    the strike exemption is scoped to the class that dominated the tick."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from sqlalchemy import select

    out: list = []
    for i in range(n):
        uid = await _seed_running_user(f"{prefix}{i}")
        async with async_session_maker() as db:
            url = (await db.execute(
                select(AgentConfig.agent_url).where(AgentConfig.user_id == uid)
            )).scalar_one()
        out.append((uid, url))
    return out


def _answers(fleet: list, statuses: list, unseeded: list):
    """Positional plan: `statuses[i]` is what `fleet[i]` answers (None raises).

    An unplanned URL is RECORDED in `unseeded` and answered 200 — never
    raised. A raise here would be invisible for the same reason the catch-up
    tripwire records instead of raising: `pool_service._probe` wraps its GET in
    `except Exception`, so the AssertionError would come back as
    `(None, "AssertionError")`, i.e. an ordinary transport error that feeds the
    quorum as data, and a test that probed an agent it never planned for would
    read as a pass. Callers assert `unseeded == []` FIRST.
    """
    assert len(fleet) == len(statuses)
    plan = [(url, st) for (_, url), st in zip(fleet, statuses)]

    def _status_for(probe_url: str):
        for url, st in plan:
            if probe_url.startswith(url):
                return st
        unseeded.append(probe_url)
        return 200

    return _status_for


def _capture_alerts(monkeypatch) -> list:
    alerts: list = []

    async def _alert(category, level, message, **kw):
        alerts.append((category, level, message, kw.get("subject")))
        return True

    monkeypatch.setattr("app.services.alerting.send_infra_alert", _alert)
    return alerts


async def _probe_rows() -> list:
    from app.db import async_session_maker
    from app.db.models import AgentProbeState
    from sqlalchemy import select
    async with async_session_maker() as db:
        return list((await db.execute(select(AgentProbeState))).scalars().all())


def _neutralise_heals(monkeypatch):
    """Stub the outbound repair calls so a sweep test never touches a bridge."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)
    monkeypatch.setattr(ps, "bridge_lookup_user_slot", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "claim_for_user", AsyncMock(return_value=None))
    return restart


# ── 1. the quorum rule, pure ──────────────────────────────────────────

def _r(status):
    return (str(uuid.uuid4()), "toup-agent-pool-1", status, None)


def test_a_5xx_majority_is_a_mass_failure_and_names_the_server_mode():
    """The 05:17Z shape: the agents ANSWER and fail. The old rule counted only
    `status is None`, so this whole class had a correlated count of zero."""
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure([_r(500)] * 170 + [_r(None)] * 2 + [_r(200)] * 2)
    assert info["mass_failure"] is True
    assert info["mode"] == "server"
    assert (info["server"], info["transport"], info["correlated"]) == (170, 2, 172)
    assert info["total"] == 174
    assert "pgbouncer" in info["hint"], info["hint"]


def test_transport_majority_still_reads_as_transport():
    """The mode the guard already covered must keep its own name and hint —
    "nothing answered" and "everything answered 500" have different first
    moves for an operator."""
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure([_r(None)] * 5 + [_r(500)] * 1)
    assert info["mass_failure"] is True
    assert info["mode"] == "transport"
    assert "egress" in info["hint"]


def test_an_even_split_is_mixed_not_silently_one_of_them():
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure([_r(None)] * 3 + [_r(500)] * 3)
    assert info["mass_failure"] is True
    assert info["mode"] == "mixed"


def test_a_minority_of_failures_is_not_a_mass_failure():
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure([_r(500)] * 1 + [_r(200)] * 3)
    assert info["mass_failure"] is False
    assert info["correlated"] == 1


def test_401_and_4xx_are_never_counted_toward_the_quorum():
    """Load-bearing: a keyless agent and a missing Caddy route are per-agent
    faults with their own repair paths. Folding them in would let one bad
    reload suppress the strikes that heal them — and 401 is the single most
    common non-200 class this sweep sees."""
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure([_r(401)] * 9 + [_r(404)] * 9)
    assert info["mass_failure"] is False
    assert (info["correlated"], info["mode"]) == (0, None)


def test_an_empty_sweep_is_never_a_mass_failure():
    from app.services.pool_service import classify_sweep_failure

    assert classify_sweep_failure([])["mass_failure"] is False


# ── 1b. the exemption set: every class that is a SHARE of the event ───
#
# Counts here are the real fleet's — ~88 probes a tick — so the minority bar
# is max(SWEEP_MINORITY_EXEMPT_MIN, ceil(0.05 x 88)) = 5. `exempt` is returned
# by classify_sweep_failure so the sweep cannot re-derive it differently.

def test_the_pgbouncer_shape_exempts_both_classes_not_only_the_dominant_one():
    """THE scenario the guard exists for, and the one a dominant-mode rule got
    wrong. pgbouncer dies; agents that reach their DB answer 500, agents that
    hang on the connect blow past the probe's 8 s timeout. 53 + 27 is never an
    exact tie, so "mixed" never fires and the 27 timeouts took a strike each —
    two ticks from restarting containers in a database outage."""
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure(
        [_r(500)] * 53 + [_r(None)] * 27 + [_r(200)] * 8
    )
    assert (info["total"], info["server"], info["transport"]) == (88, 53, 27)
    assert info["mass_failure"] is True
    assert info["mode"] == "server", "the headline stays the dominant class"
    assert info["minority_bar"] == 5
    assert info["exempt"] == ("transport", "5xx")


def test_a_lone_5xx_inside_a_transport_outage_still_takes_its_strike():
    """What the narrower rule was right about, and must survive: one agent
    answering 500 while nothing else is reachable is one sick agent."""
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure([_r(None)] * 60 + [_r(500)] + [_r(200)] * 27)
    assert info["mass_failure"] is True
    assert info["mode"] == "transport"
    assert info["exempt"] == ("transport",)


def test_two_timeouts_inside_a_5xx_outage_are_not_a_class():
    """The mirror. Two unreachable agents inside a shared-database outage are
    two agents, not a second failure mode."""
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure(
        [_r(500)] * 60 + [_r(None)] * 2 + [_r(200)] * 26
    )
    assert info["mass_failure"] is True
    assert info["mode"] == "server"
    assert info["exempt"] == ("5xx",)


def test_the_minority_share_bar_is_pinned_on_both_sides():
    """At 88 probes the bar is 5: four co-failing agents are four agents, five
    are part of the event. Pinned on both sides, because the whole judgement
    of this round lives in that one number."""
    from app.services.pool_service import classify_sweep_failure

    below = classify_sweep_failure(
        [_r(500)] * 60 + [_r(None)] * 4 + [_r(200)] * 24
    )
    at = classify_sweep_failure(
        [_r(500)] * 60 + [_r(None)] * 5 + [_r(200)] * 23
    )
    assert (below["total"], at["total"]) == (88, 88)
    assert below["minority_bar"] == at["minority_bar"] == 5
    assert below["exempt"] == ("5xx",)
    assert at["exempt"] == ("transport", "5xx")


def test_on_a_tiny_fleet_a_single_agent_is_never_a_class():
    """The floor. On six probes ceil(0.05 x 6) is 1, so without the floor one
    unreachable agent would be a "correlated class" and lose the strike that
    is the only thing which heals it."""
    from app.services.pool_service import (
        SWEEP_MINORITY_EXEMPT_MIN, classify_sweep_failure,
    )

    info = classify_sweep_failure([_r(500)] * 4 + [_r(None)] + [_r(200)])
    assert (info["total"], info["mass_failure"]) == (6, True)
    assert info["minority_bar"] == SWEEP_MINORITY_EXEMPT_MIN == 3
    assert info["exempt"] == ("5xx",)


def test_below_the_quorum_the_share_rule_exempts_nothing():
    """No mass failure, no exemption — 3 + 3 failures out of 46 is six sick
    agents, and both counts clear the bar of 3 on their own. The share rule
    may only ever narrow a mass tick, never create one."""
    from app.services.pool_service import classify_sweep_failure

    info = classify_sweep_failure(
        [_r(500)] * 3 + [_r(None)] * 3 + [_r(200)] * 40
    )
    assert info["mass_failure"] is False
    assert info["exempt"] == ()
    assert info["minority_bar"] == 0


# ── 2. the sweep, end to end ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_fleetwide_5xx_records_no_strikes_and_pages_with_the_mode(monkeypatch):
    """THE regression. Every tenant answering 500 is one shared dependency
    dying, not N sick containers: no strikes, no restarts, and a page that
    says which mode it is and how many of each."""
    from app.services import pool_service as ps

    for n in range(4):
        await _seed_running_user(f"toup-agent-pool-{n}")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_probe(monkeypatch, lambda url: 500)
    restart = _neutralise_heals(monkeypatch)
    alerts = _capture_alerts(monkeypatch)

    summary = await ps.reclaim_stranded_users()

    assert await _probe_rows() == [], (
        "a fleet-wide 5xx tick must leave NO strike behind — two of these "
        "reach PROBE_STRIKES_BEFORE_RESTART and start restarting the fleet"
    )
    assert summary.get("sick") == 0
    restart.assert_not_awaited()

    quorum = [a for a in alerts if a[0] == "sweep-quorum"]
    assert len(quorum) == 1, alerts
    assert quorum[0][1] == "critical"
    body = quorum[0][2]
    for token in ("4/4", "transport=0", "5xx=4", "mode=server"):
        assert token in body, f"{token} missing from the page: {body!r}"


@pytest.mark.asyncio
async def test_two_fleetwide_5xx_ticks_never_reach_a_restart(monkeypatch):
    """PROBE_STRIKES_BEFORE_RESTART is 2, so the pre-change code restarted on
    the SECOND tick of an outage — RESTARTS_PER_TICK containers per tick, for
    as long as the shared database stayed down."""
    from app.services import pool_service as ps

    for n in range(4):
        await _seed_running_user(f"toup-agent-pool-{n}")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_probe(monkeypatch, lambda url: 503)
    restart = _neutralise_heals(monkeypatch)
    _capture_alerts(monkeypatch)

    await ps.reclaim_stranded_users()
    second = await ps.reclaim_stranded_users()

    restart.assert_not_awaited()
    assert second.get("sick") == 0
    assert second.get("sweep_mass_failure") == "server"


@pytest.mark.asyncio
async def test_a_single_sick_agent_still_strikes_and_restarts(monkeypatch):
    """The other half of the property. One container answering 500 while the
    fleet is fine is a genuine per-agent fault, and two consecutive ticks must
    still restart it — this passes both before and after the change."""
    from app.services import pool_service as ps

    sick_uid = await _seed_running_user("toup-agent-pool-9")
    for n in range(3):
        await _seed_running_user(f"toup-agent-pool-{n}")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)

    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from sqlalchemy import select
    async with async_session_maker() as db:
        sick_url = (await db.execute(
            select(AgentConfig.agent_url).where(AgentConfig.user_id == sick_uid)
        )).scalar_one()

    _install_probe(monkeypatch, lambda url: 500 if url.startswith(sick_url) else 200)
    restart = _neutralise_heals(monkeypatch)
    _capture_alerts(monkeypatch)

    first = await ps.reclaim_stranded_users()
    assert first.get("sick") == 0            # one strike is not two
    restart.assert_not_awaited()

    second = await ps.reclaim_stranded_users()
    assert second.get("sick") == 1
    restart.assert_awaited_once()
    assert restart.await_args[0][0] == sick_uid


@pytest.mark.asyncio
async def test_a_fleet_of_401s_does_not_arm_the_quorum_guard(monkeypatch):
    """A Caddy reload that de-routes everyone answers 401/403, and the repair
    for that is the keyless path — which needs its strikes."""
    from app.services import pool_service as ps

    for n in range(4):
        await _seed_running_user(f"toup-agent-{n}a1b2c3d")   # named, not pool
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_probe(monkeypatch, lambda url: 401)
    _neutralise_heals(monkeypatch)
    alerts = _capture_alerts(monkeypatch)

    summary = await ps.reclaim_stranded_users()

    assert [a for a in alerts if a[0] == "sweep-quorum"] == []
    assert summary.get("keyless") == 4
    assert all(r.last_class == "401" for r in await _probe_rows())


# ── 2b. the exemption is per CLASS, by share of the event ─────────────
#
# Two wrong rules preceded this one, in opposite directions.
#
# The first exempted `status is None or status >= 500` off a single
# `mass_failure` flag, which widened the rule by accident: in a
# transport-dominant blip a genuinely sick 5xx container ALSO lost its strike,
# which it had kept for as long as the guard existed.
#
# The second scoped the exemption to the DOMINANT mode, which is too narrow
# for the one fault this guard exists for. A pgbouncer death answers 500 on
# the agents that reach their DB and times out the ones that hang on the
# connect — one event, two classes, never an exact tie — so a tick of 53x500 +
# 27xtimeout was "server" mode and the 27 timeouts each took a strike. Two
# such ticks and the restarter walks the fleet through a database outage.
#
# The rule that holds both ends: a class is exempt when it dominates OR when
# it is a real share of the same event (>= max(SWEEP_MINORITY_EXEMPT_MIN,
# ceil(FRACTION x total))). One or two stragglers are individual agents and
# keep their strikes, which is what the tests below pin on both sides.

@pytest.mark.asyncio
async def test_a_transport_tick_exempts_transport_and_not_the_sick_5xx(monkeypatch):
    """4 unreachable + 1 answering 500 + 1 healthy. "We could not reach
    anyone" is no evidence at all about the one container that answered."""
    from app.services import pool_service as ps

    fleet = await _seed_fleet(6)
    unseeded: list = []
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_probe(monkeypatch, _answers(fleet, [None] * 4 + [500, 200], unseeded))
    restart = _neutralise_heals(monkeypatch)
    alerts = _capture_alerts(monkeypatch)

    summary = await ps.reclaim_stranded_users()

    assert unseeded == [], f"the tick probed an agent this test never planned: {unseeded}"
    assert summary.get("sweep_mass_failure") == "transport"
    rows = await _probe_rows()
    assert [(r.user_id, r.last_class, r.consecutive_failures) for r in rows] == [
        (fleet[4][0], "5xx", 1)
    ], "the 5xx agent inside a transport blip must keep its strike"
    assert summary.get("sick") == 0            # one strike is not two
    restart.assert_not_awaited()
    body = [a for a in alerts if a[0] == "sweep-quorum"][0][2]
    assert "mode=transport" in body
    assert "No strikes for transport=4 (one event)." in body, (
        "the page must NAME the exempted class and its count — an operator "
        f"reading a bare 'strikes skipped' cannot act on it: {body!r}"
    )
    assert "1 5xx agent kept a strike." in body, (
        "one class was NOT exempt and did fail: the page has to say so, or "
        f"the per-agent healing that DID happen is invisible: {body!r}"
    )


@pytest.mark.asyncio
async def test_a_server_tick_exempts_5xx_and_not_the_unreachable_agent(monkeypatch):
    """The mirror: 4 answering 500 + 1 unreachable + 1 healthy. A shared
    database dying says nothing about the agent nobody could reach."""
    from app.services import pool_service as ps

    fleet = await _seed_fleet(6)
    unseeded: list = []
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_probe(monkeypatch, _answers(fleet, [500] * 4 + [None, 200], unseeded))
    restart = _neutralise_heals(monkeypatch)
    alerts = _capture_alerts(monkeypatch)

    summary = await ps.reclaim_stranded_users()

    assert unseeded == [], f"the tick probed an agent this test never planned: {unseeded}"
    assert summary.get("sweep_mass_failure") == "server"
    rows = await _probe_rows()
    assert [(r.user_id, r.last_class, r.consecutive_failures) for r in rows] == [
        (fleet[4][0], "transport", 1)
    ], "the unreachable agent inside a 5xx outage must keep its strike"
    assert summary.get("sick") == 0
    restart.assert_not_awaited()
    body = [a for a in alerts if a[0] == "sweep-quorum"][0][2]
    assert "No strikes for 5xx=4 (one event)." in body, body
    assert "1 transport agent kept a strike." in body, (
        "one class exempted — the page must say the minority kept its strike: "
        f"{body!r}"
    )


@pytest.mark.asyncio
async def test_a_mixed_tick_exempts_both_classes(monkeypatch):
    """3 unreachable + 3 answering 500. Neither class is a minority of the
    other, the host and its database are going down together, and there is no
    per-agent reading of either half."""
    from app.services import pool_service as ps

    fleet = await _seed_fleet(6)
    unseeded: list = []
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_probe(monkeypatch, _answers(fleet, [None] * 3 + [500] * 3, unseeded))
    restart = _neutralise_heals(monkeypatch)
    alerts = _capture_alerts(monkeypatch)

    summary = await ps.reclaim_stranded_users()

    assert unseeded == [], f"the tick probed an agent this test never planned: {unseeded}"
    assert summary.get("sweep_mass_failure") == "mixed"
    assert (summary.get("sweep_transport_errors"),
            summary.get("sweep_server_errors")) == (3, 3)
    assert await _probe_rows() == [], "a mixed tick exempts both classes"
    restart.assert_not_awaited()
    # The page has to match the rows above: both classes exempt means there is
    # no kept-strike clause to write, and any fixed sentence claiming one
    # ("a minority failure in the other class still took its strike") would
    # send an operator looking for per-agent healing that did not happen.
    body = [a for a in alerts if a[0] == "sweep-quorum"][0][2]
    assert "No strikes for transport=3 + 5xx=3 (one event)." in body, body
    assert "kept a strike" not in body, (
        "both classes exempt — no agent kept a strike in either class, so the "
        f"page must not claim one did: {body!r}"
    )


@pytest.mark.asyncio
async def test_a_minority_class_of_one_shared_outage_records_no_strikes(monkeypatch):
    """The pgbouncer shape on the real sweep, at a fleet a test can seed: 11
    agents answer 500 and 4 time out in the same tick. The mode is "server",
    but 4 clears the bar (max(3, ceil(0.05 x 20)) = 3) so both classes are
    part of one event. Under the dominant-only rule those 4 took a strike
    each, and the tick after that restarts containers during a database
    outage no restart can fix."""
    from app.services import pool_service as ps

    fleet = await _seed_fleet(20)
    unseeded: list = []
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_probe(monkeypatch, _answers(
        fleet, [500] * 11 + [None] * 4 + [200] * 5, unseeded))
    restart = _neutralise_heals(monkeypatch)
    alerts = _capture_alerts(monkeypatch)

    summary = await ps.reclaim_stranded_users()

    assert unseeded == [], f"the tick probed an agent this test never planned: {unseeded}"
    assert summary.get("sweep_mass_failure") == "server"
    assert (summary.get("sweep_transport_errors"),
            summary.get("sweep_server_errors")) == (4, 11)
    assert await _probe_rows() == [], (
        "the timeouts are the same event as the 5xx — a strike on each is the "
        "first half of a restart storm through a shared outage"
    )
    assert summary.get("sick") == 0
    restart.assert_not_awaited()
    body = [a for a in alerts if a[0] == "sweep-quorum"][0][2]
    assert "No strikes for transport=4 + 5xx=11 (one event)." in body, body
    assert "kept a strike" not in body, (
        f"both classes were exempt — no agent kept a strike: {body!r}"
    )


# ── 3. the post-boot catch-up, and the lease handover ─────────────────

@pytest.mark.asyncio
async def test_catchup_pass_skips_the_probe_sweep_entirely(monkeypatch):
    """`probe_sweep=False` must not probe AT ALL — not probe-and-discard.

    The tripwire RECORDS rather than raises, because a raise here is
    invisible: `_probe` wraps its GET in `except Exception`, so an
    AssertionError from the fake client comes back as `(None, 'AssertionError')`
    — a transport error — and a single-user fleet with one transport error
    trips the (deliberately floor-free) transport arm, so the strike is
    skipped and every other assertion below still passes. The only question
    that cannot be answered by accident is "was a URL fetched".
    """
    from app.services import pool_service as ps

    await _seed_running_user("toup-agent-pool-1")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)

    probed: list = []
    _install_probe(monkeypatch, lambda url: probed.append(url) or 200)
    restart = _neutralise_heals(monkeypatch)
    _capture_alerts(monkeypatch)

    summary = await ps.reclaim_stranded_users(probe_sweep=False)

    assert probed == [], f"the catch-up pass probed the fleet: {probed}"
    assert summary.get("probe_sweep") == "skipped"
    assert await _probe_rows() == []
    restart.assert_not_awaited()


def _stub_reconciler_legs(monkeypatch, interval: int, fast: int, catchup: int):
    """Drive container_reconciler_loop with every leg stubbed, fast clocks.

    `catchup` goes on `settings`, not on a module constant: the pass is
    default-ON and force-claims pool slots, so it has to be switchable off
    during an incident without a deploy.
    """
    from unittest.mock import AsyncMock
    from app.services import docker_host_service as dhs
    from app.services import pool_service as ps

    monkeypatch.setattr(dhs.settings, "container_reconciler_interval_s",
                        interval, raising=False)
    monkeypatch.setattr(dhs.settings, "stranded_fast_scan_interval_s",
                        fast, raising=False)
    monkeypatch.setattr(dhs.settings, "bridge_url", "https://bridge.test",
                        raising=False)
    monkeypatch.setattr(dhs.settings, "post_boot_catchup_s", catchup,
                        raising=False)

    calls = {
        "backfill": AsyncMock(return_value={}),
        "rowsync": AsyncMock(return_value={}),
        "reclaim": AsyncMock(return_value={}),
        "fast": AsyncMock(return_value={}),
        "acquire": AsyncMock(return_value=True),
        "release": AsyncMock(return_value=True),
    }
    monkeypatch.setattr(dhs, "backfill_sentinel_image_containers", calls["backfill"])
    monkeypatch.setattr(dhs, "reconcile_managed_rows", calls["rowsync"])
    monkeypatch.setattr(ps, "reclaim_stranded_users", calls["reclaim"])
    monkeypatch.setattr(ps, "reclaim_stranded_fast", calls["fast"])
    monkeypatch.setattr("app.services.infra_lease.acquire_lease", calls["acquire"])
    monkeypatch.setattr("app.services.infra_lease.release_lease", calls["release"])
    return calls


def _dont_actually_sleep(monkeypatch) -> None:
    """Make the LOOP's sleeps free, so a test can span many intervals.

    Patched on `docker_host_service`'s own module global, never on the real
    `asyncio`: the test itself awaits a real sleep to hand the loop time, and
    the event loop reads that module too. The shim still yields once per nap,
    or `task.cancel()` could never land and the test would hang.
    """
    import asyncio as _real

    class _NoWait:
        async def sleep(self, _d):
            await _real.sleep(0)

        def __getattr__(self, name):
            return getattr(_real, name)

    from app.services import docker_host_service as dhs
    monkeypatch.setattr(dhs, "asyncio", _NoWait())


@pytest.mark.asyncio
async def test_the_loop_hands_its_lease_back_on_a_graceful_shutdown(monkeypatch):
    """The dominant half of the 733 s gap. ttl = 3x interval, so without a
    release the incoming replica waits out the dead holder's row before it may
    sweep at all."""
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=30, fast=0, catchup=0)

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    calls["release"].assert_awaited_once()
    assert calls["release"].await_args[0][0] == "container_reconciler"


@pytest.mark.asyncio
async def test_catchup_runs_the_uncovered_legs_long_before_the_first_tick(monkeypatch):
    """Boot already ran the backfill IN THIS PROCESS, so the catch-up must not
    repeat it — that is the "redundant churn" the sleep-first comment is
    about. Nothing at boot runs the stranded backstop or the row-sync, which
    is what the catch-up is for. It rides the fast sub-tick, which is the
    production path (stranded_fast_scan_interval_s = 15)."""
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=30, fast=1, catchup=1)

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(2.6)          # past the catch-up, far short of a tick
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls["fast"].await_count >= 1
    calls["reclaim"].assert_awaited_once()
    assert calls["reclaim"].await_args.kwargs.get("probe_sweep") is False, (
        "the catch-up may not advance restart strikes"
    )
    calls["rowsync"].assert_awaited_once()
    calls["backfill"].assert_not_awaited()


@pytest.mark.asyncio
async def test_no_fast_subtick_means_no_catchup(monkeypatch):
    """The coupling, stated so it cannot rot into a silent surprise: the
    catch-up hooks the fast sub-tick rather than opening a second lease gate
    (test_infra_lease pins one gate per grain), so turning the sub-tick off
    turns the catch-up off too."""
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=30, fast=0, catchup=1)

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(1.6)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    calls["reclaim"].assert_not_awaited()
    calls["release"].assert_awaited_once()      # the lease is still handed back


@pytest.mark.asyncio
async def test_the_catchups_own_wall_time_is_charged_against_the_interval(
    monkeypatch,
):
    """`slept` is a SLEEP tally, not a clock.

    The catch-up does real work — up to `max_per_tick` bridge-backed claims —
    and it is awaited from inside the accounting, so without charging its
    elapsed time the first full tick lands that much AFTER the interval. A
    pass whose whole purpose is to close a post-deploy gap must not widen the
    next one. The clock is faked (only this module's `time`, never the real
    one — asyncio's event loop reads that) so the assertion is arithmetic
    rather than a race: a catch-up "taking" 29 s of a 30 s interval leaves
    the inner loop with nothing left to sleep.
    """
    import time as _real_time
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=30, fast=1, catchup=1)

    class _Clock:
        """monotonic() answers 0 then 29; everything else is the real module."""
        def __init__(self):
            self._v = [0.0, 29.0]

        def monotonic(self):
            return self._v.pop(0) if self._v else 29.0

        def __getattr__(self, name):
            return getattr(_real_time, name)

    monkeypatch.setattr(dhs, "time", _Clock())

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(1.6)          # one sub-tick, nowhere near 30 s
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    sweeps = [c for c in calls["reclaim"].await_args_list
              if c.kwargs.get("probe_sweep") is False]
    assert len(sweeps) == 1, "the catch-up itself must have run"
    assert calls["backfill"].await_count == 1, (
        "the 29 s the catch-up spent must count toward the interval — "
        "uncharged, the first full tick is 29 s late on every boot"
    )


@pytest.mark.asyncio
async def test_post_boot_catchup_s_0_turns_the_catchup_off(monkeypatch):
    """The off switch, and the reason the delay is a setting rather than a
    module constant: this pass force-claims pool slots ~20 s into every boot,
    and an operator holding an incident has to be able to stop it from the
    Railway variables page instead of by shipping a build.

    The window is measured in INTERVALS, not seconds. `catchup_at` is clamped
    to what a sub-tick can answer (`interval - fast`) and the inner loop breaks
    at `slept >= interval` BEFORE the catch-up check, so a bug that silently read
    0 as the 20 s default could never fire inside a 2.6 s test against a 30 s
    interval — the test would pass on exactly the code it exists to reject.
    With the loop's sleeps freed, several intervals go by, so a 20 s delay
    would fire many times over. The sub-tick it rides is left on, which is
    what makes this the knob and not `fast = 0`."""
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=30, fast=1, catchup=0)
    _dont_actually_sleep(monkeypatch)

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls["backfill"].await_count >= 2, (
        "the window must span whole intervals or it cannot see a 20 s delay"
    )
    assert calls["fast"].await_count >= 29, "the sub-tick itself must still run"
    sweeps = [c for c in calls["reclaim"].await_args_list
              if c.kwargs.get("probe_sweep") is False]
    assert sweeps == [], f"the catch-up ran {len(sweeps)} times with the knob off"


@pytest.mark.parametrize("catchup", [30, 45])
@pytest.mark.asyncio
async def test_a_delay_no_subtick_can_reach_is_clamped_not_silently_dropped(
    monkeypatch, catchup,
):
    """An operator raising the delay during an incident must not be turning
    the pass OFF by accident.

    The catch-up check sits BELOW the inner loop's `break` at
    `slept >= interval`, so the largest value any sub-tick can ever answer is
    `interval - fast`: unclamped, `post_boot_catchup_s` at or above the
    interval fired never, with nothing logged and config.py documenting only
    "0 disables it". 0 is the off switch (the test above); everything else
    lands on the first sub-tick that can carry it, and the clamp says so once.
    """
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=30, fast=1,
                                  catchup=catchup)
    _dont_actually_sleep(monkeypatch)

    warned: list = []
    monkeypatch.setattr(dhs.logger, "warning",
                        lambda m, *a, **k: warned.append(m % a if a else m))

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls["backfill"].await_count >= 2, (
        "the window must span whole intervals, or 'fired never' and 'fired "
        "late' look the same"
    )
    sweeps = [c for c in calls["reclaim"].await_args_list
              if c.kwargs.get("probe_sweep") is False]
    assert len(sweeps) == 1, (
        f"post_boot_catchup_s={catchup} against a 30 s interval ran the "
        f"catch-up {len(sweeps)} times — clamped, it runs exactly once"
    )
    clamp = [w for w in warned if "clamped" in w]
    assert len(clamp) == 1, (
        f"the clamp must be stated ONCE at loop start, not silently: {warned}"
    )
    assert "clamped to 29s" in clamp[0], clamp[0]


@pytest.mark.asyncio
async def test_the_catchup_is_once_per_process_not_once_per_interval(monkeypatch):
    """It exists to cover a deploy, not to double the cadence."""
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=2, fast=1, catchup=1)

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(5.2)          # >= two full intervals
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls["backfill"].await_count >= 2, "the full tick still runs"
    assert calls["reclaim"].await_count >= 3
    sweeps = [c for c in calls["reclaim"].await_args_list
              if c.kwargs.get("probe_sweep") is False]
    assert len(sweeps) == 1, f"catch-up ran {len(sweeps)} times"


@pytest.mark.asyncio
async def test_a_lost_lease_race_skips_the_catchup_rather_than_running_it(monkeypatch):
    """The catch-up is leader-gated like every other leg: two replicas booting
    together must produce ONE pass, not two."""
    from app.services import docker_host_service as dhs

    calls = _stub_reconciler_legs(monkeypatch, interval=30, fast=1, catchup=1)
    calls["acquire"].return_value = False

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(2.6)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    calls["reclaim"].assert_not_awaited()
    calls["rowsync"].assert_not_awaited()


def test_a_tiny_fleet_gets_no_5xx_quorum_verdict():
    """The floor, and it is not cosmetic. At a fleet of one, "most probes
    answered 500" and "this one agent is sick" are the same sentence — reading
    it as the first disables the per-agent restart that is the only thing
    which heals the second, forever. Three existing sweep tests (one sick
    agent, probed alone) caught exactly that."""
    from app.services.pool_service import (
        SWEEP_QUORUM_MIN_PROBES, classify_sweep_failure,
    )

    for n in range(1, SWEEP_QUORUM_MIN_PROBES):
        info = classify_sweep_failure([_r(500)] * n)
        assert info["mass_failure"] is False, f"{n} probes must not be a quorum"
        # …the counts are still reported; only the VERDICT stands down.
        assert info["correlated"] == n
    assert classify_sweep_failure(
        [_r(500)] * SWEEP_QUORUM_MIN_PROBES
    )["mass_failure"] is True


def test_the_floor_does_not_apply_to_the_transport_arm():
    """The asymmetry, which is load-bearing and pre-existing: a probe that
    never got an answer is not evidence the AGENT is broken, so the restarter
    stands down on it at any fleet size. `test_reclaim_stranded_users.
    test_sweep_mass_transport_failure_records_no_strikes` is that behaviour,
    and it probes a fleet of one."""
    from app.services.pool_service import classify_sweep_failure

    solo = classify_sweep_failure([_r(None)])
    assert solo["mass_failure"] is True
    assert solo["mode"] == "transport"


def test_the_released_lease_is_the_one_the_loop_gates_on():
    """Two spellings of one name, because test_infra_lease pins the literal
    `_lease = "container_reconciler"` inside the loop body. Pin that they
    agree: a rename that misses one leaves the shutdown releasing a lease
    nobody holds, which fails SILENTLY and costs a full TTL on every deploy —
    exactly the 733 s gap this arm exists to remove."""
    import inspect
    import re
    from app.services import docker_host_service as dhs

    gated = re.search(
        r'_lease = "([^"]+)"', inspect.getsource(dhs._container_reconciler_ticks),
    ).group(1)
    released = re.search(
        r'release_lease\("([^"]+)"\)',
        inspect.getsource(dhs.container_reconciler_loop),
    ).group(1)
    assert gated == released == "container_reconciler"
