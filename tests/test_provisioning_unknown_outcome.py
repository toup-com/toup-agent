"""A claim whose outcome is UNKNOWN must not start a second provisioning path.

WHAT THE TREE DOES TODAY (origin/main e698ea78)

    pool_service.claim_for_user   POST /v1/pool/claim raises httpx.ReadTimeout
                                  → ensure_discovery(...) → `return None`
    pool_service.claim_or_prewarm `None` means "nobody has a slot for this
                                  user" → schedule_prewarm
    prewarm_service._run_prewarm  takes `provision_drive:<uid>` — a DIFFERENT
                                  advisory key from claim_for_user's
                                  `pool_claim:<uid>`, so nothing excludes the
                                  two — → docker_host_service.provision_container
    docker_host_service           POST /v1/tenants  ← the cold NAMED path

So the platform asks the bridge for a second container for a user the bridge
is at that moment binding a pool slot for.

THIS IS NOT HYPOTHETICAL. On 2026-09-06 the named create really ran for the
incident user: docker network `tnt_aec1977b` on the VPS is stamped
`created=2026-09-06 18:18:00.155932865 +0000 UTC` with `containers=0`, and
`/data/agents/aec1977b` dates from 18:17 — eleven seconds BEFORE the bridge
finished binding that same user to `toup-agent-pool-73` at 18:18:11. It only
failed to produce a second container because Postgres was at 300/300 and
`_create_tenant_db` died. Two live fossils of the completed version are still
on the host: prefixes `667cf3de` and `51d4ed2f` each have a named container
serving them (Caddy → :9077 / :9083) AND a pool slot the bridge still reports
ASSIGNED+bound (40 and 17) that serves nobody.

With #712 the bridge no longer freezes its event loop and finishes work after
the caller hangs up, and with #716 discovery adopts the pool slot beside the
named container — so on a healthy host this composition now COMPLETES.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import httpx
import pytest

_port = itertools.count(9700)


@pytest.fixture(autouse=True)
def _no_leaked_background_work(monkeypatch):
    """Discovery loops and the post-start soul sync both outlive their caller
    ON PURPOSE. In a test that means a task still holding a sqlite session
    while conftest drops the schema — which surfaces as
    `database table is locked` in TEARDOWN, i.e. a green test reported as an
    error. Stub the one, cancel the other."""
    from app.services import docker_host_service as dhs

    async def _no_soul_sync(*a, **k):
        return None
    monkeypatch.setattr(dhs, "_sync_soul_after_start", _no_soul_sync, raising=False)
    yield
    from app.services import pool_service as ps
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    for t in list(asyncio.all_tasks()) if running else []:
        if t.done():
            continue
        name = t.get_name() or ""
        coro = getattr(t, "get_coro", lambda: None)()
        cname = getattr(coro, "__qualname__", "") or ""
        if name.startswith(("discover:", "adopt-once:")) or "_sync_soul_after_start" in cname:
            t.cancel()
    ps._DISCOVERY_INFLIGHT.clear()
    ps._invalidate_pool_list_cache()
    ps._WHOIS_UNAVAILABLE = False


# ── A bridge that records which door was used ────────────────────────────

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


class TwoDoorBridge:
    """POST /v1/pool/claim  — the pool door (what a signup should use)
       POST /v1/tenants     — the cold NAMED door (what must not open)

    `claim_mode` chooses the answer the pool door gives:
      'timeout'  the bind happens, the response is lost (the incident)
      '5xx'      the bind may have happened, the bridge errored after it
      '503'      a DEFINITE answer: the pool is empty, nothing was bound
    """

    def __init__(self, *, user_id: str, claim_mode: str = "timeout",
                 slot: str = "toup-agent-pool-73"):
        self.user_id = str(user_id)
        self.slot = slot
        self.claim_mode = claim_mode
        self.bound = False
        self.claim_calls = 0
        self.named_calls = 0
        self.whois_calls = 0
        self.pool_key: str | None = None

    async def post(self, path, json=None, **kw):
        if path == "/v1/pool/claim":
            self.claim_calls += 1
            if self.claim_mode == "503":
                return _Resp(503, {"ok": False, "reason": "pool exhausted"})
            # Every other mode BINDS — that is the whole point.
            self.bound = True
            self.pool_key = (json or {}).get("agent_api_key")
            if self.claim_mode == "timeout":
                raise httpx.ReadTimeout("")
            if self.claim_mode == "5xx":
                return _Resp(502, {}, text="bad gateway")
            return _Resp(200, {
                "ok": True,
                "container_name": self.slot,
                "container_id": "deadbeef",
                "host_port": 18073,
                "db_pool_slot": "toup_agent_feed0073",
                "idempotent": True,
            })
        if path == "/v1/tenants":
            self.named_calls += 1
            return _Resp(200, {
                "container_name": f"toup-agent-{self.user_id[:8]}",
                "container_id": "namedcafe",
                "host_port": next(_port),
                "image_tag": "ghcr.io/toup-com/toup-agent:abc123def456",
                "agent_url": f"https://agent-{self.user_id[:8]}.agents.toup.ai",
                "agent_api_key": "K_NAMED",
            })
        raise AssertionError(f"unexpected POST {path}")

    async def get(self, path, params=None, timeout=None, **kw):
        if path.endswith("/whois"):
            self.whois_calls += 1
            if not self.bound or str((params or {}).get("user_id")) != self.user_id:
                return _Resp(200, {"found": False, "slot": None,
                                   "state": None, "bound": False})
            return _Resp(200, {
                "found": True, "slot": "73", "state": "ASSIGNED", "bound": True,
                "prefix": self.user_id[:8], "user_id": self.user_id,
                "container_name": self.slot, "container_id": "deadbeef",
                "host_port": 18073, "db_pool_slot": "toup_agent_feed0073",
                "state_changed_at": 1788718691,
            })
        if path == "/v1/pool/list":
            m = {"slot": 73, "port": 18073, "container_name": self.slot,
                 "db_name": "toup_agent_feed0073", "image_tag": "toup-agent:abc",
                 "docker_id": "deadbeef",
                 "state": "ASSIGNED" if self.bound else "GENERIC"}
            if self.bound:
                m["assigned_user_id"] = self.user_id
                m["assigned_prefix"] = self.user_id[:8]
            return _Resp(200, {"target": 10, "members": [m]})
        raise AssertionError(f"unexpected GET {path}")


class _Lease:
    def __init__(self, b):
        self._b = b

    async def __aenter__(self):
        return self._b

    async def __aexit__(self, *a):
        return None


def install(monkeypatch, bridge: TwoDoorBridge):
    from app.services import docker_host_service as dhs
    from app.services import pool_service as ps
    monkeypatch.setattr(dhs, "_bridge_client", lambda *a, **k: _Lease(bridge))

    async def _get_client():
        return bridge
    monkeypatch.setattr(dhs, "get_bridge_client", _get_client)
    # A real rollout SHA so provision_container does not refuse the sentinel.
    async def _tag(_db):
        return "ghcr.io/toup-com/toup-agent:abc123def456"
    monkeypatch.setattr(dhs, "_latest_known_good_image_tag", _tag)
    monkeypatch.setattr(ps, "_latest_known_good_image_tag", _tag, raising=False)
    ps._invalidate_pool_list_cache()


async def _seed(db):
    from app.db.models import User, AgentConfig
    uid = str(uuid.uuid4())
    db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                name="T", is_active=True))
    await db.flush()
    db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                       bundle_status="active", llm_mode="bundle"))
    await db.commit()
    return uid


async def _settle():
    """schedule_prewarm hands _run_prewarm to a background task."""
    for _ in range(40):
        await asyncio.sleep(0.02)


def _quiet_prewarm(monkeypatch):
    from app.services import prewarm_service as pw

    async def _no_boot(*a, **k):
        return None
    monkeypatch.setattr(pw, "_await_boot_ready", _no_boot)


# ═══════════════════════════════════════════════════════════════════════
# (a) the headline: an unknown outcome opens exactly one door
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "5xx"])
async def test_an_unknown_claim_outcome_never_opens_the_named_door(monkeypatch, mode):
    """FALSIFIER. On origin/main this fails with named_calls == 1: the timeout
    returns None, claim_or_prewarm reads None as "no slot", and the prewarm
    drives POST /v1/tenants for a user the bridge has already bound."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", False, raising=False)
    _quiet_prewarm(monkeypatch)

    async with async_session_maker() as db:
        uid = await _seed(db)
    bridge = TwoDoorBridge(user_id=uid, claim_mode=mode)
    install(monkeypatch, bridge)

    async with async_session_maker() as db:
        ok = await ps.claim_or_prewarm(db, uid)
    await _settle()

    assert bridge.claim_calls >= 1, "fixture invariant: the pool door was tried"
    assert bridge.bound is True, "fixture invariant: the bridge DID bind"
    assert bridge.named_calls == 0, (
        f"POST /v1/tenants was called {bridge.named_calls} time(s) for a user "
        f"the bridge was binding a pool slot for — this is the 2026-09-06 "
        f"duplicate (docker network tnt_aec1977b, created 18:18:00, 0 containers)"
    )
    assert ok is True, (
        "claim_or_prewarm must report that provisioning is being driven; "
        "returning False would make auth.register log a failure for a signup "
        "that is about to succeed"
    )


@pytest.mark.asyncio
async def test_a_definite_empty_pool_still_takes_the_named_path(monkeypatch):
    """The guard against over-fixing. 503 is the bridge ANSWERING "there is no
    generic slot" — nothing was bound, and the named path is the only way this
    user gets an agent. It must still run."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", False, raising=False)
    _quiet_prewarm(monkeypatch)

    async with async_session_maker() as db:
        uid = await _seed(db)
    bridge = TwoDoorBridge(user_id=uid, claim_mode="503")
    install(monkeypatch, bridge)

    async with async_session_maker() as db:
        ok = await ps.claim_or_prewarm(db, uid)
    await _settle()

    assert ok is True
    assert bridge.bound is False, "fixture invariant: a 503 binds nothing"
    assert bridge.named_calls == 1, (
        "a DEFINITE 'no generic slot' must still fall back to POST /v1/tenants "
        "— otherwise a signup during the post-rollout GENERIC==0 window gets "
        "no agent at all"
    )

    from sqlalchemy import select
    from app.db.models import ManagedContainer
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == uid)
        )).scalars().all()
    assert len(rows) == 1 and not rows[0].container_name.startswith("toup-agent-pool-")


# ═══════════════════════════════════════════════════════════════════════
# (b) the ordering that ends in a permanent 401
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_the_permanent_401_ordering_is_unreachable(monkeypatch):
    """The audit's Outcome B: the pool bind is adopted (row → pool slot, key
    K_pool, Caddy → pool port), then the late named create lands and Caddy is
    re-pointed at the named container, which holds K_named. The platform never
    learns the new key, so every chat 401s "Authentication required" — the
    2026-06-30 class.

    The ordering needs BOTH doors. Assert the second one never opens, and that
    the key the platform stores is the one the bound container was given."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import AgentConfig, ManagedContainer
    from sqlalchemy import select
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_interval_s", 0.05, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_max_s", 6, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_cache_ttl_s", 0.0, raising=False)
    _quiet_prewarm(monkeypatch)

    async with async_session_maker() as db:
        uid = await _seed(db)
    bridge = TwoDoorBridge(user_id=uid, claim_mode="timeout")
    install(monkeypatch, bridge)

    async with async_session_maker() as db:
        await ps.claim_or_prewarm(db, uid)
    await _settle()

    assert bridge.named_calls == 0, (
        "the named door opened — with it, discovery adopts the pool slot and "
        "the late named create then owns the Caddy route with a key the "
        "platform never stored: a permanent 401"
    )

    # Discovery converges on the slot the bridge actually bound.
    bridge.claim_mode = "ok"
    url = await asyncio.wait_for(
        ps.discover_and_adopt_bind(uid, reason="test"), timeout=10,
    )
    assert url

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == uid)
        )).scalars().all()
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == uid)
        )).scalar_one_or_none()
    assert len(rows) == 1, f"one container per user; got {len(rows)}"
    assert rows[0].container_name == "toup-agent-pool-73"
    assert cfg.agent_api_key and cfg.agent_api_key == bridge.pool_key, (
        "the platform must hold the key the bound container was given; "
        "holding K_named (or nothing) while Caddy dials the pool slot is the "
        "401 the ordering produces"
    )
    assert bridge.named_calls == 0


# ═══════════════════════════════════════════════════════════════════════
# (c) the second door: a soul-save prewarm racing an in-flight claim
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_prewarm_asks_whether_a_claim_is_already_in_flight(monkeypatch):
    """PUT /api/soul with PREWARM_ON_SOUL_SAVE=true calls schedule_prewarm
    directly (soul.py:366). `_run_prewarm` locks `provision_drive:<uid>` while
    `claim_for_user` locks `pool_claim:<uid>` — two different keys, so on
    origin/main a soul save lands POST /v1/tenants on top of a bind that is
    still in flight.

    FALSIFIER: on the unchanged tree `try_take_claim_drive` does not exist and
    nothing consults it, so `gate['consulted']` stays 0."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import prewarm_service as pw
    from app.services import pool_service as ps
    from app.services import docker_host_service as dhs

    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db)

    gate = {"consulted": 0}

    async def _claim_in_flight(_db, _uid):
        gate["consulted"] += 1
        return False        # somebody is mid-claim for this user
    monkeypatch.setattr(ps, "try_take_claim_drive", _claim_in_flight, raising=False)

    async def _granted(_db, _uid):
        return True
    monkeypatch.setattr(ps, "try_take_provision_drive", _granted)

    started: list = []
    monkeypatch.setattr(
        ps, "ensure_discovery", lambda u, **k: started.append((u, k.get("reason"))),
    )

    async def _must_not_run(*a, **k):
        raise AssertionError(
            "a prewarm must not drive POST /v1/tenants while a pool claim for "
            "the same user is in flight"
        )
    monkeypatch.setattr(dhs, "provision_container", _must_not_run)
    _quiet_prewarm(monkeypatch)

    await pw._run_prewarm(uid)

    assert gate["consulted"] == 1, (
        f"_run_prewarm consulted the claim drive {gate['consulted']} time(s); "
        f"0 is the unchanged tree, where nothing excludes claim-vs-prewarm"
    )
    assert started and started[0][0] == uid, (
        "an observer must start discovery, not go quiet"
    )


@pytest.mark.asyncio
async def test_a_prewarm_with_no_claim_in_flight_still_provisions(monkeypatch):
    """The other half: the gate must not turn every soul-save prewarm off."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import prewarm_service as pw
    from app.services import pool_service as ps
    from app.services import docker_host_service as dhs

    monkeypatch.setattr(settings, "provision_discovery_enabled", False, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db)

    async def _granted(_db, _uid):
        return True
    monkeypatch.setattr(ps, "try_take_provision_drive", _granted)
    monkeypatch.setattr(ps, "try_take_claim_drive", _granted, raising=False)

    ran = {"n": 0}

    async def _provision(db, user_id, agent_config=None, **kw):
        ran["n"] += 1
        from app.db.models import ManagedContainer
        db.add(ManagedContainer(
            id=str(uuid.uuid4()), user_id=user_id,
            container_name=f"toup-agent-{user_id[:8]}", host_port=next(_port),
            db_name="d", status="running",
        ))
        agent_config.agent_url = f"https://agent-{user_id[:8]}.agents.toup.ai"
        agent_config.agent_api_key = "k"
        await db.commit()
        return None
    monkeypatch.setattr(dhs, "provision_container", _provision)
    _quiet_prewarm(monkeypatch)

    await pw._run_prewarm(uid)
    assert ran["n"] == 1, "an uncontended prewarm must still provision"


# ═══════════════════════════════════════════════════════════════════════
# (d) the lock hand-off the gate above depends on
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_the_claim_lock_is_handed_back_before_the_named_fallback(monkeypatch):
    """`pool_claim:<uid>` is an xact-scoped lock: it lives until the session's
    next commit or rollback, and every early return in claim_for_user does
    neither. So on the 503 path the lock was still held while the caller went
    on to schedule the prewarm — and the prewarm now refuses to drive while
    somebody holds it. Without the hand-off the fix above would deadlock the
    legitimate pool-exhausted fallback.

    Not visible on sqlite (advisory locks are Postgres-only and the helpers
    answer True there), so this asserts the WIRING; the primitive is proved
    against real Postgres below."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", False, raising=False)
    _quiet_prewarm(monkeypatch)

    released = {"n": 0}
    real = ps._release_claim_drive

    async def _counting(db):
        released["n"] += 1
        await real(db)
    monkeypatch.setattr(ps, "_release_claim_drive", _counting)

    async with async_session_maker() as db:
        uid = await _seed(db)
    bridge = TwoDoorBridge(user_id=uid, claim_mode="503")
    install(monkeypatch, bridge)

    async with async_session_maker() as db:
        await ps.claim_or_prewarm(db, uid)
    await _settle()

    assert released["n"] == 1, (
        f"the claim drive was handed back {released['n']} time(s) on the "
        f"pool-exhausted path; holding it would starve the named fallback the "
        f"previous test requires to still work"
    )
    assert bridge.named_calls == 1


@pytest.mark.asyncio
async def test_the_claim_drive_key_really_excludes_on_real_postgres():
    """The primitive, on a real Postgres: the key `try_take_claim_drive`
    probes is byte-identical to the one `claim_for_user` holds, a second
    session is refused while it is held, and a rollback releases it.

    MUTATION: change either side's key string (drop the `pool_claim:` prefix,
    or key on the 8-hex prefix instead of the full user id) and the try-lock
    below is GRANTED — the gate becomes a no-op that reads as working."""
    asyncpg = pytest.importorskip("asyncpg")
    dsn = os.environ.get("LANE_C_PG_DSN", "postgresql://localhost/postgres")
    try:
        a = await asyncpg.connect(dsn)
    except Exception as e:                                    # pragma: no cover
        pytest.skip(f"no local Postgres: {e}")
    b = await asyncpg.connect(dsn)
    try:
        from app.services.pool_service import _claim_drive_key
        uid = str(uuid.uuid4())
        key = _claim_drive_key(uid)
        assert key == f"pool_claim:{uid}", (
            "claim_for_user's lock and try_take_claim_drive's probe must be "
            "the same string; this is the whole gate"
        )
        ta = a.transaction(); await ta.start()
        tb = b.transaction(); await tb.start()
        held = await a.fetchval(
            "SELECT pg_try_advisory_xact_lock(hashtext($1)::bigint)", key)
        assert held is True
        denied = await b.fetchval(
            "SELECT pg_try_advisory_xact_lock(hashtext($1)::bigint)", key)
        assert denied is False, (
            "a second driver was granted the claim drive while the first held "
            "it — the prewarm would go straight to POST /v1/tenants"
        )
        await ta.rollback()
        granted = await b.fetchval(
            "SELECT pg_try_advisory_xact_lock(hashtext($1)::bigint)", key)
        assert granted is True, (
            "the lock survived its transaction — the pool-exhausted fallback "
            "would never get the drive back"
        )
        await tb.rollback()
    finally:
        await a.close()
        await b.close()


@pytest.mark.asyncio
async def test_a_named_provision_that_lost_its_response_is_still_recoverable(monkeypatch):
    """The residue this fix does NOT remove, pinned so the next change cannot
    quietly make it worse.

    On the definite-empty-pool path the named create can itself time out, and
    `provision_container` writes NO managed_containers row when there was none
    to begin with (docker_host_service.py: the `existing` guard on both error
    branches). The audit asked for a row-on-timeout; `managed_containers` has
    `Index("ix_managed_containers_host_port", unique=True)` on a NOT NULL
    column, so there is no port to write and no sentinel that two concurrent
    provisions could share — it needs a migration, which is another lane's
    file. What must hold in the meantime is that recovery still EXISTS:
    a managed user with no row is a `_stranded_user_ids` candidate regardless
    of signup age, so the 180 s reclaim replays the finalize.

    On 2026-09-06 this really happened — /data/agents/aec1977b and docker
    network tnt_aec1977b (0 containers) are the residue, and the reclaim is
    what put that user on pool-73 at 18:19:12."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services import pool_service as ps
    from app.services import docker_host_service as dhs
    from sqlalchemy import select
    from app.db.models import ManagedContainer

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db)

    class _Hang:
        async def post(self, *a, **k):
            raise httpx.ReadTimeout("")
    monkeypatch.setattr(dhs, "_bridge_client", lambda *a, **k: _Lease(_Hang()))

    async def _tag(_db):
        return "ghcr.io/toup-com/toup-agent:abc123def456"
    monkeypatch.setattr(dhs, "_latest_known_good_image_tag", _tag)

    async with async_session_maker() as db:
        with pytest.raises(RuntimeError):
            await dhs.provision_container(db, uid)

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == uid)
        )).scalars().all()
        stranded = await ps._stranded_user_ids(db, limit=60)
    assert rows == [], (
        "documenting the gap: the named path writes no row on an unknown "
        "outcome, so the attempt is invisible in the platform DB"
    )
    assert uid in stranded, (
        "…but the user MUST still be a reclaim candidate, or the residue is "
        "permanent and the account never gets an agent"
    )
