"""Late-success discovery — the 2026-09-06 "bridge unreachable: " incident.

WHAT HAPPENED (Railway `platform-api-window.jsonl`, bridge registry,
managed_containers row timestamps):

    18:17:09  alidstm registers
    18:17:39  BOTH replicas start a prewarm for them, 19 ms apart
    18:18:09  both raise httpx.ReadTimeout at the flat 30 s
              `bridge_request_timeout_s`; the trail reads, twice,
              "[PREWARM] user=aec1977b task failed: bridge unreachable: "
              — with nothing after the colon, because
              `str(httpx.ReadTimeout())` is the EMPTY STRING
    18:18:11  the bridge finishes the bind: toup-agent-pool-73, ASSIGNED
    18:18:45  the user's app (build 109) stops retrying
    18:19:12  the 180 s container reconciler notices, 61 s late

The bind SUCCEEDED. The platform had simply stopped listening two seconds
early and had no way of asking again. Everything here tests the "ask again"
path, plus the logging that made the failure unreadable.

Falsifier note: on the pre-fix tree `pool_service.discover_and_adopt_bind`,
`ensure_discovery` and `try_adopt_stranded` do not exist, so these tests
ERROR at import rather than merely failing. The pre-fix behaviour is
additionally demonstrated INLINE in
`test_timed_out_claim_is_discovered_and_adopted`: it asserts that right
after the timeout there is no container row (which is where the old tree
stopped) before running discovery.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import httpx
import pytest

_port = itertools.count(9300)


@pytest.fixture(autouse=True)
def _no_leaked_discovery_loops():
    """`ensure_discovery` deliberately outlives its caller. In a test that
    would leave a task polling a fixture bridge for `provision_discovery_max_s`
    seconds after the test that made it has gone."""
    yield
    from app.services import pool_service as ps
    for t in list(asyncio.all_tasks()) if _loop_running() else []:
        if (t.get_name() or "").startswith(("discover:", "adopt-once:")):
            t.cancel()
    ps._DISCOVERY_INFLIGHT.clear()
    ps._invalidate_pool_list_cache()
    # Process-wide latch: one test's 404 would otherwise disable the whois
    # route for every test that runs after it.
    ps._WHOIS_UNAVAILABLE = False


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


# ── A fake bridge ────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=self  # type: ignore[arg-type]
            )


class FakeBridge:
    """Models the ONE property that makes this incident possible: the bind
    outlives the platform's timeout. `claim_hangs` makes POST /v1/pool/claim
    raise ReadTimeout while still recording the bind in the registry, exactly
    as the real bridge did at 18:18:11."""

    def __init__(self, *, user_id: str, slot: str = "toup-agent-pool-73"):
        self.user_id = str(user_id)
        self.slot = slot
        self.claim_hangs = True
        self.bound = False           # has the bridge finished the bind?
        self.claim_calls = 0
        self.list_calls = 0
        self.whois_calls = 0

    # POST
    async def post(self, path, json=None, **kw):
        if path == "/v1/pool/claim":
            self.claim_calls += 1
            # The bind happens regardless — that is the whole point.
            self.bound = True
            if self.claim_hangs:
                raise httpx.ReadTimeout("")   # str() == "" — see module docstring
            return _Resp(200, {
                "ok": True,
                "container_name": self.slot,
                "container_id": "deadbeef",
                "host_port": 18073,
                "db_pool_slot": "toup_agent_feed0073",
                "agent_url": f"https://agent-{self.user_id[:8]}.agents.toup.ai",
                "idempotent": True,
            })
        raise AssertionError(f"unexpected POST {path}")

    # GET
    async def get(self, path, params=None, timeout=None, **kw):
        if path.endswith("/whois"):
            # Lane B's route: a CLAIM-shaped body, not a registry member.
            self.whois_calls += 1
            if not self.bound or str((params or {}).get("user_id")) != self.user_id:
                return _Resp(200, {"found": False, "slot": None,
                                   "state": None, "bound": False})
            return _Resp(200, {
                "found": True, "slot": "73", "state": "ASSIGNED", "bound": True,
                "prefix": self.user_id[:8], "user_id": self.user_id,
                "container_name": self.slot, "container_id": "deadbeef",
                "host_port": 18073, "db_pool_slot": "toup_agent_feed0073",
                "agent_url": f"https://agent-{self.user_id[:8]}.agents.toup.ai",
                "state_changed_at": 1788718691,
            })
        if path == "/v1/pool/list":
            self.list_calls += 1
            m = {
                "slot": 73, "port": 18073, "container_name": self.slot,
                "db_name": "toup_agent_feed0073", "image_tag": "toup-agent:abc",
                "docker_id": "deadbeef",
                "state": "ASSIGNED" if self.bound else "GENERIC",
            }
            if self.bound:
                m["assigned_user_id"] = self.user_id
                m["assigned_prefix"] = self.user_id[:8]
            return _Resp(200, {"target": 10, "members": [m]})
        raise AssertionError(f"unexpected GET {path}")


class _Lease:
    def __init__(self, bridge):
        self._b = bridge

    async def __aenter__(self):
        return self._b

    async def __aexit__(self, *a):
        return None


def install_fake_bridge(monkeypatch, bridge: FakeBridge):
    from app.services import docker_host_service as dhs
    from app.services import pool_service as ps

    monkeypatch.setattr(dhs, "_bridge_client", lambda *a, **k: _Lease(bridge))
    async def _get_client():
        return bridge
    monkeypatch.setattr(dhs, "get_bridge_client", _get_client)
    ps._invalidate_pool_list_cache()


# ── Seeding ──────────────────────────────────────────────────────────────

async def _seed_user(db, *, with_container=None):
    from app.db.models import User, AgentConfig, ManagedContainer
    uid = str(uuid.uuid4())
    db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                name="T", is_active=True))
    await db.flush()
    db.add(AgentConfig(
        user_id=uid, hosting_mode="managed",
        # 'active' skips claim_for_user's free-tier activation block, which
        # would otherwise reach for the OpenAI admin API in a unit test.
        bundle_status="active", llm_mode="bundle",
    ))
    if with_container is not None:
        name, status = with_container
        db.add(ManagedContainer(
            id=str(uuid.uuid4()), user_id=uid, container_name=name,
            host_port=next(_port), db_name=f"db_{uid[:8]}", status=status,
        ))
    await db.commit()
    return uid


async def _rows_for(uid):
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import ManagedContainer, AgentConfig
    async with async_session_maker() as db:
        mcs = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == uid)
        )).scalars().all()
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == uid)
        )).scalar_one_or_none()
        return list(mcs), cfg


# ═══════════════════════════════════════════════════════════════════════
# (a) the headline case
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_timed_out_claim_is_discovered_and_adopted(monkeypatch):
    """The claim raises ReadTimeout, the bridge binds anyway, discovery finds
    it and adopts it — one container row, agent_url set, and a second pass is
    a no-op."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_interval_s", 0.05, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_max_s", 6, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 0.0, raising=False)

    async with async_session_maker() as db:
        uid = await _seed_user(db)

    bridge = FakeBridge(user_id=uid)
    install_fake_bridge(monkeypatch, bridge)

    # 1. The claim that "failed".
    async with async_session_maker() as db:
        got = await ps.claim_for_user(db, uid)
    assert got is None, "a ReadTimeout must not be reported as a claim"
    assert bridge.bound is True, "fixture invariant: the bridge DID bind"

    # 2. WHERE THE OLD TREE STOPPED. This is the state the user was left in
    #    at 18:18:09 and would have stayed in until the 180 s reconciler.
    mcs, cfg = await _rows_for(uid)
    assert mcs == [], "no container row yet — this is the incident state"
    assert cfg.agent_url in (None, ""), "no agent_url yet — the user cannot chat"

    # 3. Discovery goes and asks.
    bridge.claim_hangs = False        # the bridge is responsive again
    url = await asyncio.wait_for(
        ps.discover_and_adopt_bind(uid, reason="test"), timeout=10,
    )
    assert url, "discovery did not adopt a bind the bridge had already made"
    assert url.endswith(f"agent-{uid[:8]}.agents.toup.ai")

    mcs, cfg = await _rows_for(uid)
    assert len(mcs) == 1, f"exactly one assignment per user; got {len(mcs)}"
    assert mcs[0].container_name == "toup-agent-pool-73"
    assert mcs[0].status == "running"
    assert cfg.agent_url == url
    assert cfg.agent_api_key, (
        "the adopt must go through claim_for_user so the bridge RE-PUSHES the "
        "bind — a DB-only adopt leaves the platform key NULL while the agent "
        "holds the key the timed-out claim minted, and every chat 401s"
    )

    # 4. A racing second adopter (the reconciler, or the other replica) must
    #    be a no-op, not a second bind.
    claims_before = bridge.claim_calls
    again = await ps._adopt_discovered_bind(uid, {
        "container_name": "toup-agent-pool-73", "state": "ASSIGNED",
    })
    assert again == url
    assert bridge.claim_calls == claims_before, (
        "an already-converged user must not be re-claimed on the bridge"
    )
    mcs, _ = await _rows_for(uid)
    assert len(mcs) == 1


@pytest.mark.asyncio
async def test_reclaim_afterwards_is_a_no_op(monkeypatch):
    """Once discovery has adopted, the 180 s reconciler must not see the user
    as stranded — otherwise the two systems fight over the same row."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db, with_container=("toup-agent-pool-73", "running"))
        stranded = await ps._stranded_user_ids(db)
    assert uid not in stranded, (
        "a user adopted onto a running pool slot is not stranded"
    )
    async with async_session_maker() as db:
        fast = await ps._recently_stranded_user_ids(db)
    assert uid not in fast, "nor is it a candidate for the 15 s fast pass"


@pytest.mark.asyncio
async def test_fast_pass_finds_the_late_bind(monkeypatch):
    """The 15 s sub-tick: a recent signup with no running row, whose bind the
    bridge already holds, is adopted without waiting for the 180 s scan."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 0.0, raising=False)

    async with async_session_maker() as db:
        uid = await _seed_user(db)
    bridge = FakeBridge(user_id=uid)
    bridge.bound = True               # the bridge finished while we were away
    bridge.claim_hangs = False
    install_fake_bridge(monkeypatch, bridge)

    async with async_session_maker() as db:
        assert uid in await ps._recently_stranded_user_ids(db)

    summary = await ps.reclaim_stranded_fast()
    assert summary.get("adopted") == 1, summary
    mcs, cfg = await _rows_for(uid)
    assert len(mcs) == 1 and mcs[0].status == "running"
    assert cfg.agent_url


@pytest.mark.asyncio
async def test_fast_pass_costs_nothing_when_nobody_is_stranded(monkeypatch):
    """It runs every 15 s, so the steady state must be one indexed query and
    ZERO bridge calls."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db, with_container=("toup-agent-pool-9", "running"))
    bridge = FakeBridge(user_id=uid)
    install_fake_bridge(monkeypatch, bridge)
    summary = await ps.reclaim_stranded_fast()
    assert summary["candidates"] == 0
    assert bridge.list_calls == 0, "no candidates must mean no bridge traffic"


# ═══════════════════════════════════════════════════════════════════════
# The adapter + its safety rules
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_lookup_only_adopts_ASSIGNED_not_ASSIGNING(monkeypatch):
    """`assigned_user_id` is stamped by the bridge at ASSIGNING — BEFORE the
    agent has been bound (bridge/pool_addon.py `_claim_one`). Adopting on that
    state publishes an agent_url for a container that cannot answer yet."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 0.0, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "provision_adopt_budget_s", 2.0, raising=False)
    # This one is about the LIST shape's ASSIGNING state specifically.
    monkeypatch.setattr(settings, "bridge_pool_whois_route", "", raising=False)

    async with async_session_maker() as db:
        uid = await _seed_user(db)
    bridge = FakeBridge(user_id=uid)
    # The claim itself would SUCCEED here — so if anything adopted on
    # ASSIGNING it would really write a row, and the assertion below has
    # something to catch. (With a hanging claim the mutation survives: the
    # adopt fails for the wrong reason.)
    bridge.claim_hangs = False
    install_fake_bridge(monkeypatch, bridge)

    # Mid-bind: the user id is stamped, the state is not ASSIGNED yet.
    bridge.bound = True

    async def _get(path, params=None, timeout=None, **kw):
        return _Resp(200, {"members": [{
            "slot": 73, "port": 18073, "container_name": "toup-agent-pool-73",
            "db_name": "d", "docker_id": "x", "state": "ASSIGNING",
            "assigned_user_id": uid, "assigned_prefix": uid[:8],
        }]})
    monkeypatch.setattr(bridge, "get", _get)

    slot = await ps.bridge_lookup_user_slot(uid)
    assert slot is not None and slot["state"] == "ASSIGNING"

    async with async_session_maker() as db:
        out = await ps.try_adopt_stranded(db, uid)
    assert out is None, "ASSIGNING is not a bind; nothing may be adopted"
    mcs, _ = await _rows_for(uid)
    assert mcs == []


@pytest.mark.asyncio
async def test_lookup_is_shared_between_concurrent_pollers(monkeypatch):
    """One bridge read must answer for every waiting caller. The bridge's
    event loop is blocked by docker work under load (27 synchronous
    subprocess.run calls in pool_addon.py) — a stampede is the last thing it
    needs."""
    from app.config import settings
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 5.0, raising=False)
    # The shared read is a property of /v1/pool/list, which answers for every
    # user at once. /v1/pool/whois is per-user by construction and is not
    # cached — concurrent pollers for the SAME user are already deduped one
    # layer up by `_DISCOVERY_INFLIGHT`.
    monkeypatch.setattr(settings, "bridge_pool_whois_route", "", raising=False)
    bridge = FakeBridge(user_id=str(uuid.uuid4()))
    install_fake_bridge(monkeypatch, bridge)
    await asyncio.gather(*[
        ps.bridge_lookup_user_slot(str(uuid.uuid4())) for _ in range(20)
    ])
    assert bridge.list_calls == 1, (
        f"20 pollers must cost the bridge one call, not {bridge.list_calls}"
    )


@pytest.mark.asyncio
async def test_discovery_never_rehomes_a_named_tenant(monkeypatch):
    """A named tenant's database is `toup_agent_<prefix>`; a pool slot's is the
    slot's own. Adopting a pool slot over a named row on the strength of a
    registry read is the same data loss `provision_container`'s
    PoolMemberSwapRefused exists to prevent, in the other direction."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db, with_container=("toup-agent-abcd1234", "error"))
    bridge = FakeBridge(user_id=uid)
    bridge.bound = True
    bridge.claim_hangs = False
    install_fake_bridge(monkeypatch, bridge)

    out = await ps._adopt_discovered_bind(uid, {
        "container_name": "toup-agent-pool-73", "state": "ASSIGNED",
    })
    assert out is None, "a named tenant must never be adopted onto a pool slot"
    assert bridge.claim_calls == 0
    mcs, _ = await _rows_for(uid)
    assert mcs[0].container_name == "toup-agent-abcd1234"


@pytest.mark.asyncio
async def test_discovery_unsticks_a_wedged_provisioning_row(monkeypatch):
    """schedule_prewarm stamps status='provisioning' BEFORE its bridge call.
    claim_for_user's existing-row check then short-circuits every later claim,
    and reclaim only unsticks it after 15 minutes. A bridge that says ASSIGNED
    is proof that provisioning is over — the unstick is evidence-gated."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db, with_container=("toup-agent-pool-73", "provisioning"))
    bridge = FakeBridge(user_id=uid)
    bridge.bound = True
    bridge.claim_hangs = False
    install_fake_bridge(monkeypatch, bridge)

    out = await ps._adopt_discovered_bind(uid, {
        "container_name": "toup-agent-pool-73", "state": "ASSIGNED",
    })
    assert out, "a wedged 'provisioning' row must not block a proven bind"
    mcs, _ = await _rows_for(uid)
    assert len(mcs) == 1 and mcs[0].status == "running"


@pytest.mark.asyncio
async def test_try_adopt_stranded_is_bounded_and_never_cancels_a_write(monkeypatch):
    """Lane D's hook. It must answer inside its budget against a bridge that
    does not answer at all, and it must NOT cancel the adopt it started —
    cancelling between the bridge's bind and the row write manufactures the
    very lost-response state this change removes."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps
    import time as _t

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "provision_adopt_budget_s", 0.3, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 0.0, raising=False)

    async with async_session_maker() as db:
        uid = await _seed_user(db)
    bridge = FakeBridge(user_id=uid)
    install_fake_bridge(monkeypatch, bridge)

    finished = {"v": False}

    async def _slow_get(path, params=None, timeout=None, **kw):
        await asyncio.sleep(1.0)
        finished["v"] = True
        return _Resp(200, {"members": []})
    monkeypatch.setattr(bridge, "get", _slow_get)

    t0 = _t.monotonic()
    async with async_session_maker() as db:
        out = await ps.try_adopt_stranded(db, uid)
    dt = _t.monotonic() - t0
    assert out is None
    assert dt < 1.0, f"the hook must not become what the client waits on ({dt:.2f}s)"
    assert finished["v"] is False, "fixture invariant: the work was still running"
    # And it was NOT cancelled — it completes on its own.
    await asyncio.sleep(1.2)
    assert finished["v"] is True, (
        "the overrunning adopt must be left running, never cancelled"
    )


@pytest.mark.asyncio
async def test_try_adopt_stranded_short_circuits_on_converged_rows(monkeypatch):
    """The common case for the WS proxy: the rows are already consistent, so
    the answer costs one DB read and zero bridge traffic."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from sqlalchemy import select
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db, with_container=("toup-agent-pool-5", "running"))
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == uid)
        )).scalar_one()
        cfg.agent_url = f"https://agent-{uid[:8]}.agents.toup.ai"
        cfg.agent_api_key = "k"
        await db.commit()

    bridge = FakeBridge(user_id=uid)
    install_fake_bridge(monkeypatch, bridge)
    async with async_session_maker() as db:
        out = await ps.try_adopt_stranded(db, uid)
    assert out == f"https://agent-{uid[:8]}.agents.toup.ai"
    assert bridge.list_calls == 0


@pytest.mark.asyncio
async def test_try_adopt_stranded_never_raises(monkeypatch):
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "provision_adopt_budget_s", 0.5, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 0.0, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db)
    bridge = FakeBridge(user_id=uid)

    async def _boom(*a, **k):
        raise httpx.ConnectError("no route to host")
    monkeypatch.setattr(bridge, "get", _boom)
    install_fake_bridge(monkeypatch, bridge)
    async with async_session_maker() as db:
        assert await ps.try_adopt_stranded(db, uid) is None


@pytest.mark.asyncio
async def test_disabled_flag_turns_the_whole_thing_off(monkeypatch):
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "provision_discovery_enabled", False, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db)
        assert await ps.try_adopt_stranded(db, uid) is None
    ps.ensure_discovery(uid, reason="test")
    assert uid not in ps._DISCOVERY_INFLIGHT
    assert (await ps.reclaim_stranded_fast()).get("skipped") == "discovery_disabled"


# ═══════════════════════════════════════════════════════════════════════
# (d) the empty-reason regression
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_pool_claim_timeout_names_the_failure(monkeypatch, caplog):
    """`str(httpx.ReadTimeout())` is ''. The trail read "bridge unreachable: "
    and named nothing — not the class, not the budget. MUTATION: change the
    `%r` back to `%s` and this goes red."""
    import logging
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", False, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db)
    bridge = FakeBridge(user_id=uid)
    install_fake_bridge(monkeypatch, bridge)

    with caplog.at_level(logging.WARNING, logger="app.services.pool_service"):
        async with async_session_maker() as db:
            assert await ps.claim_for_user(db, uid) is None

    lines = [r.getMessage() for r in caplog.records if "bridge unreachable" in r.getMessage()]
    assert lines, "the timeout must be logged at all"
    assert any("ReadTimeout" in ln for ln in lines), (
        f"the log must name the failure; got {lines!r} — this is the literal "
        f"2026-09-06 line 'bridge unreachable: ' with nothing after the colon"
    )


@pytest.mark.asyncio
async def test_provision_timeout_names_the_failure(monkeypatch):
    """Same defect in `provision_container`, which is the one the two
    duplicate prewarms actually hit ("task failed: bridge unreachable: ")."""
    from app.db import async_session_maker
    from app.services import docker_host_service as dhs

    async with async_session_maker() as db:
        uid = await _seed_user(db)

    class _Hang:
        async def post(self, *a, **k):
            raise httpx.ReadTimeout("")
    monkeypatch.setattr(dhs, "_bridge_client", lambda *a, **k: _Lease(_Hang()))

    async def _tag(_db):
        return "toup-agent:abc123"
    monkeypatch.setattr(dhs, "_latest_known_good_image_tag", _tag)

    from sqlalchemy import select
    from app.db.models import AgentConfig
    with pytest.raises(RuntimeError) as ei:
        async with async_session_maker() as db:
            cfg = (await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == uid)
            )).scalar_one()
            await dhs.provision_container(db, uid, cfg)
    assert "ReadTimeout" in str(ei.value), (
        f"the raised error must name the failure; got {str(ei.value)!r}"
    )


def test_agent_setup_no_longer_logs_a_bare_percent_s(monkeypatch):
    """The third site: "[AGENT-SETUP] Managed container provision/sync failed: "
    with nothing after the colon, twice, at 18:18:09."""
    import inspect
    from app.api import agent_setup
    src = inspect.getsource(agent_setup)
    assert 'provision/sync failed: %s", e)' not in src, (
        "the empty-reason format is back at the agent-setup site"
    )
    assert "provision/sync failed for %s: %r" in src


@pytest.mark.asyncio
async def test_a_timed_out_claim_starts_discovery_by_itself(monkeypatch):
    """The claim path's own follow-up. Nothing else on 2026-09-06 asked again
    until the 180 s reconciler, and the user had stopped retrying 27 s before
    that. MUTATION: delete the `ensure_discovery` call on claim_for_user's
    httpx.HTTPError branch → red."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed_user(db)
    bridge = FakeBridge(user_id=uid)
    install_fake_bridge(monkeypatch, bridge)

    started: list = []
    monkeypatch.setattr(
        ps, "ensure_discovery", lambda u, **k: started.append((u, k.get("reason"))),
    )
    async with async_session_maker() as db:
        assert await ps.claim_for_user(db, uid) is None
    assert started == [(uid, "claim_timeout")], (
        f"a claim that lost its response must ask again; got {started!r}"
    )


@pytest.mark.asyncio
async def test_signup_trace_is_one_cheap_line_per_hop(monkeypatch, caplog):
    """The trace has to be readable from Railway logs with a grep for the
    user's 8-hex prefix, and must never carry a secret or any message content.
    MUTATION: drop the `hop=` or `elapsed_ms=` token → red."""
    import logging
    from app.services import pool_service as ps

    uid = "aec1977b-1fe0-4565-956a-ae960d06719c"
    with caplog.at_level(logging.INFO, logger="app.services.pool_service"):
        ps.seed_signup_trace(uid)
        ps.signup_trace(uid, "registered")
        ps.signup_trace(uid, "claim_timeout", "ReadTimeout")
    msgs = [r.getMessage() for r in caplog.records if "[signup-trace]" in r.getMessage()]
    assert len(msgs) == 2, msgs
    for m in msgs:
        assert "user=aec1977b " in m, m
        assert "hop=" in m and "elapsed_ms=" in m and "detail=" in m, m
        assert uid not in m, "only the 8-hex prefix may appear, never the full id"
    assert "hop=claim_timeout" in msgs[1] and "detail=ReadTimeout" in msgs[1]


@pytest.mark.asyncio
async def test_reconciler_runs_the_fast_pass_between_full_ticks(monkeypatch):
    """The 180 s full scan is UNCHANGED; the sleep is subdivided so the cheap
    stranded pass runs in between. On 2026-09-06 the platform waited 61 s to
    learn something the bridge had known since 18:18:11.

    MUTATION: set `stranded_fast_scan_interval_s = 0` (or restore the single
    `await asyncio.sleep(interval)`) → fast_calls drops to 0 → red.
    """
    from app.config import settings
    from app.services import docker_host_service as dhs
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "bridge_url", "https://bridge.test", raising=False)
    # Both are read through int(), so they must be whole seconds — which is
    # also what production uses (180 / 15).
    monkeypatch.setattr(settings, "container_reconciler_interval_s", 3, raising=False)
    monkeypatch.setattr(settings, "stranded_fast_scan_interval_s", 1, raising=False)

    calls = {"fast": 0, "full": 0}

    async def _fast():
        calls["fast"] += 1
        return {"candidates": 0}
    monkeypatch.setattr(ps, "reclaim_stranded_fast", _fast)

    async def _full(db, **kw):
        calls["full"] += 1
        return {}
    monkeypatch.setattr(dhs, "backfill_sentinel_image_containers", _full)

    async def _reclaim(*a, **k):
        return {}
    monkeypatch.setattr(ps, "reclaim_stranded_users", _reclaim)

    async def _rows(db):
        return {}
    monkeypatch.setattr(dhs, "reconcile_managed_rows", _rows)

    task = asyncio.ensure_future(dhs.container_reconciler_loop())
    await asyncio.sleep(3.4)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert calls["fast"] >= 2, (
        f"the fast pass must run several times per full tick (180/15 = 12 in "
        f"production); ran {calls['fast']} time(s)"
    )
    assert calls["full"] >= 1, "the 180 s full scan must still run"
    assert calls["fast"] > calls["full"], (
        "the point is a shorter reaction for the stranded set, not a shorter "
        "full scan"
    )


# ═══════════════════════════════════════════════════════════════════════
# The adapter against the route Lane B actually built
# ═══════════════════════════════════════════════════════════════════════

# Copied verbatim from Lane B's bridge. `GET /v1/pool/whois?user_id=<uuid>`
# answers a CLAIM-SHAPED body, not a registry member: no `assigned_user_id`,
# no `assigned_prefix`, `host_port` not `port`, `db_pool_slot` not `db_name`,
# `container_id` not `docker_id` — and `slot` is a STRING, which is what made
# the first version of this adapter raise AttributeError on
# `body.get("member") or body.get("slot") or body`.
WHOIS_FOUND = {
    "found": True, "slot": "73", "state": "ASSIGNED", "bound": True,
    "prefix": "aec1977b", "user_id": "aec1977b-1fe0-4565-956a-ae960d06719c",
    "container_name": "toup-agent-pool-73", "container_id": "c0ffee1234",
    "host_port": 9560, "db_pool_slot": "toup_agent_feed0073",
    "agent_url": "https://agent-aec1977b.agents.toup.ai",
    "state_changed_at": 1788718691,
}
WHOIS_NOT_FOUND = {"found": False, "slot": None, "state": None, "bound": False}


def _whois_bridge(monkeypatch, body, *, status=200, on_get=None):
    bridge = FakeBridge(user_id=WHOIS_FOUND["user_id"])

    async def _get(path, params=None, timeout=None, **kw):
        if on_get is not None:
            on_get(path, params)
        if path.endswith("/whois"):
            return _Resp(status, body)
        return await FakeBridge.get(bridge, path, params=params, timeout=timeout, **kw)
    monkeypatch.setattr(bridge, "get", _get)
    install_fake_bridge(monkeypatch, bridge)
    return bridge


@pytest.mark.asyncio
async def test_whois_found_body_normalises(monkeypatch):
    """Fails on the pre-review adapter with
    `AttributeError: 'str' object has no attribute 'get'` — `body["slot"]` is
    "73", and `x or y or z` picks the first truthy value."""
    from app.config import settings
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "bridge_pool_whois_route", "/v1/pool/whois", raising=False)
    _whois_bridge(monkeypatch, WHOIS_FOUND)

    slot = await ps.bridge_lookup_user_slot(WHOIS_FOUND["user_id"])
    assert slot == {
        "slot": "73",
        "container_name": "toup-agent-pool-73",
        "container_id": "c0ffee1234",
        "host_port": 9560,
        "db_name": "toup_agent_feed0073",
        "state": "ASSIGNED",
        "prefix": "aec1977b",
        "bound": True,
    }, slot
    assert ps._is_adoptable(slot) is True


@pytest.mark.asyncio
async def test_whois_not_found_body_is_None(monkeypatch):
    """`{"found": false, ...}` is the bridge ANSWERING "no bind", which is a
    different thing from the bridge not answering — it must be None, not a
    raise and not a half-filled slot dict."""
    from app.config import settings
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "bridge_pool_whois_route", "/v1/pool/whois", raising=False)
    _whois_bridge(monkeypatch, WHOIS_NOT_FOUND)
    assert await ps.bridge_lookup_user_slot(WHOIS_FOUND["user_id"]) is None


@pytest.mark.asyncio
async def test_whois_bound_false_is_not_adoptable(monkeypatch):
    """The bridge stamps ASSIGNED before /admin/bind has necessarily landed,
    so `bound` is the stronger signal of the two. Adopting an unbound slot
    publishes an agent_url for a container that cannot answer."""
    from app.config import settings
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "bridge_pool_whois_route", "/v1/pool/whois", raising=False)
    _whois_bridge(monkeypatch, {**WHOIS_FOUND, "bound": False})
    slot = await ps.bridge_lookup_user_slot(WHOIS_FOUND["user_id"])
    assert slot is not None and slot["state"] == "ASSIGNED"
    assert ps._is_adoptable(slot) is False


@pytest.mark.asyncio
async def test_whois_404_falls_back_to_the_list_and_never_asks_again(monkeypatch):
    """A bridge that has not deployed the route yet must degrade ONCE. Asking
    every 5 s forever would double the traffic of the very poll that exists to
    be cheap — and the fallback answer is correct, so nothing is lost."""
    from app.config import settings
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "bridge_pool_whois_route", "/v1/pool/whois", raising=False)
    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 0.0, raising=False)
    uid = WHOIS_FOUND["user_id"]
    seen: list = []
    bridge = _whois_bridge(monkeypatch, {}, status=404,
                           on_get=lambda p, q: seen.append(p))
    bridge.bound = True                      # the registry knows the answer

    slot = await ps.bridge_lookup_user_slot(uid)
    assert slot is not None and slot["container_name"] == "toup-agent-pool-73"
    assert seen == ["/v1/pool/whois", "/v1/pool/list"]

    # A later 200 on whois must NOT be retried in this process.
    seen.clear()
    bridge2 = _whois_bridge(monkeypatch, WHOIS_FOUND,
                            on_get=lambda p, q: seen.append(p))
    bridge2.bound = True
    slot = await ps.bridge_lookup_user_slot(uid)
    assert slot is not None
    assert seen == ["/v1/pool/list"], (
        f"the whois route was re-probed after a 404; got {seen!r}"
    )
