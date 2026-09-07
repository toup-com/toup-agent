"""`[signup-trace]` is the log the runbook points at, and it answers the wrong
question in both directions.

COMPLETION-REPORT.md section 5 tells on-call to run
`railway logs --filter "signup-trace"` on the next signup. What comes back,
measured on production since the 2026-09-06 deploy:

    2026-09-07T00:48:04 [signup-trace] user=f02202a0 hop=claim_ok elapsed_ms=0 detail=slot=toup-agent-pool-34
    2026-09-07T00:51:21 [signup-trace] user=a8176b1d hop=claim_ok elapsed_ms=0 detail=slot=toup-agent-pool-42
    2026-09-07T15:46:09 [signup-trace] user=876d73fa hop=claim_ok elapsed_ms=0 detail=slot=toup-agent-pool-77

Neither of the first two is a signup: each row had just been flipped
`running -> stopped` by the P0 walk's row-sync, and `reclaim_stranded_users`
re-claimed it. `claim_ok` is emitted unconditionally by `claim_for_user`, so a
reclaim, a keyless force re-bind and a discovery adopt are all indistinguishable
from a fresh registration.

And `elapsed_ms=0` is not a 0 ms hop. `seed_signup_trace` has no caller
(`grep -rn seed_signup_trace backend/app` finds only the definition), so
`_trace_origin` stamps t0 the first time THIS process happens to see the user —
which on the claim path is the claim itself. The 2026-09-07 e2e signup took
4 s from register to bound and printed `elapsed_ms=0`; the one number the
incident was about cannot be read from the trace at all.

Worse than useless with two replicas: the hop that fires on replica B has no t0
from replica A's registration, so a 0 there means "different process", not
"instant" — and the two are printed identically.
"""
from __future__ import annotations

import logging
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_discovery_loops():
    """`ensure_discovery` outlives its caller on purpose; in a test that is a
    task still holding a sqlite session when conftest drops the schema."""
    import asyncio
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


def _trace_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[signup-trace]" in r.getMessage()]


def _field(line: str, key: str) -> str:
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok[len(key) + 1:]
    return ""


# ═══════════════════════════════════════════════════════════════════════
# (a) is test_signup_trace_register_hop.py — it registers for real, and
#     registration seeds `identities` (AGENT_ONLY); this file needs
#     `managed_containers` (PLATFORM_ONLY) for (c). Different lanes.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_elapsed_ms_is_measured_from_registration(monkeypatch, caplog):
    """With a monotonic clock stub: seed at t0, emit 4 s later, read 4000.

    MUTATION: delete the `seed_signup_trace` call and `_trace_origin` stamps
    t0 at the first hop instead — elapsed_ms reads 0 for a 4 s signup, which
    is exactly what production printed for the 2026-09-07 e2e run."""
    from app.services import pool_service as ps

    clock = {"t": 1000.0}
    monkeypatch.setattr(ps.time, "monotonic", lambda: clock["t"])
    uid = str(uuid.uuid4())

    with caplog.at_level(logging.INFO, logger="app.services.pool_service"):
        ps.seed_signup_trace(uid)
        clock["t"] += 4.0
        ps.signup_trace(uid, "claim_ok", "slot=toup-agent-pool-76", origin="signup")

    lines = _trace_lines(caplog)
    assert len(lines) == 1, lines
    assert _field(lines[0], "elapsed_ms") == "4000", (
        f"a 4 s register->bound hop must read 4000; got {lines[0]!r}"
    )


# ═══════════════════════════════════════════════════════════════════════
# (b) a hop with no seed says so, instead of lying with a 0
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_an_unseeded_hop_prints_null_not_zero(caplog):
    """Two replicas, one Postgres: the claim can finish on the replica that
    did NOT serve the registration. That process has no t0.

    FALSIFIER: `_trace_origin` used to CREATE a t0 on first sighting, so this
    printed `elapsed_ms=0` — indistinguishable from a genuinely instant hop,
    which is how three production lines came to read 0 for hops of 4 s and
    more."""
    from app.services import pool_service as ps

    uid = str(uuid.uuid4())
    ps._TRACE_T0.pop(uid[:8], None)
    with caplog.at_level(logging.INFO, logger="app.services.pool_service"):
        ps.signup_trace(uid, "claim_ok", "slot=toup-agent-pool-76", origin="signup")

    lines = _trace_lines(caplog)
    assert len(lines) == 1, lines
    assert _field(lines[0], "elapsed_ms") == "null", (
        f"an unseeded hop must say it does not know; got {lines[0]!r}"
    )


@pytest.mark.asyncio
async def test_a_created_at_already_in_hand_seeds_the_trace(monkeypatch, caplog):
    """The allowed exception: when the caller ALREADY holds the User row it
    may hand `created_at` over. It must never cost a DB round trip inside a
    logging helper — so the helper takes a value, never a user id to look up.
    """
    from datetime import datetime, timedelta
    from app.services import pool_service as ps

    uid = str(uuid.uuid4())
    ps._TRACE_T0.pop(uid[:8], None)
    created = datetime.utcnow() - timedelta(seconds=7)

    with caplog.at_level(logging.INFO, logger="app.services.pool_service"):
        ps.signup_trace(uid, "claim_ok", "slot=x", origin="signup", created_at=created)

    lines = _trace_lines(caplog)
    ms = _field(lines[0], "elapsed_ms")
    assert ms != "null" and 6000 <= int(ms) <= 9000, (
        f"a created_at in hand must produce a real elapsed; got {lines[0]!r}"
    )


# ═══════════════════════════════════════════════════════════════════════
# (c) a reclaim must not read as a signup
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_reclaim_is_tagged_reclaim_not_signup(monkeypatch, caplog):
    """FALSIFIER: `claim_ok` is emitted unconditionally at the bottom of
    `claim_for_user`, so on the unchanged tree the only two production lines
    since the deploy — both keyless force re-binds of months-old users during
    the P0 walk — were indistinguishable from two instantaneous signups."""
    import httpx
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig
    from app.services import pool_service as ps
    from app.services import docker_host_service as dhs

    monkeypatch.setattr(settings, "use_container_pool", True, raising=False)

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                    name="T", is_active=True))
        await db.flush()
        db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                           bundle_status="active", llm_mode="bundle"))
        await db.commit()

    class _Resp:
        status_code = 200

        def json(self):
            return {"ok": True, "container_name": "toup-agent-pool-34",
                    "container_id": "abc", "host_port": 19034,
                    "db_pool_slot": "toup_agent_feed0034", "idempotent": True}

        def raise_for_status(self):
            return None

    class _Bridge:
        async def post(self, path, json=None, **kw):
            assert path == "/v1/pool/claim"
            return _Resp()

    class _Lease:
        async def __aenter__(self):
            return _Bridge()

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr(dhs, "_bridge_client", lambda *a, **k: _Lease())

    async def _tag(_db):
        return "ghcr.io/toup-com/toup-agent:abc123def456"
    monkeypatch.setattr(dhs, "_latest_known_good_image_tag", _tag)

    with caplog.at_level(logging.INFO, logger="app.services.pool_service"):
        async with async_session_maker() as db:
            c = await ps.claim_for_user(db, uid, force=True, origin="reclaim")
    assert c is not None

    lines = [ln for ln in _trace_lines(caplog) if "hop=claim_ok" in ln]
    assert lines, "the successful claim must still be traced"
    assert _field(lines[0], "origin") == "reclaim", (
        f"a reclaim tagged {_field(lines[0], 'origin')!r} — on-call greps "
        f"hop=claim_ok to find signups, and this is not one: {lines[0]!r}"
    )


@pytest.mark.asyncio
async def test_every_claim_caller_states_its_origin():
    """Source probe. The tag is only worth anything if no call site forgets
    it, and a default of 'signup' would silently mislabel every backstop.

    MUTATION: drop `origin=` from the reclaim call at pool_service.py and this
    goes red."""
    import inspect
    import re
    from app.services import pool_service as ps

    src = inspect.getsource(ps)
    # `await claim_for_user(...)` only — never the definition, which carries
    # the default and would otherwise satisfy its own probe.
    invocations = re.findall(r"await claim_for_user\([^)]*\)", src)
    assert len(invocations) >= 4, (
        f"expected every claim_for_user call site; found {invocations!r}"
    )
    missing = [c for c in invocations if "origin=" not in c]
    assert not missing, (
        f"claim_for_user call sites with no origin tag: {missing!r}"
    )
