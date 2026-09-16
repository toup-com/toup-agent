"""`/agent-setup/ready` asks the AGENT, and the tunnel is not a green light.

Two readiness lies from the 2026-09-12 onboarding incident.

D-8: `GET /api/agent-setup/ready` resolved `ready` from
`managed_containers.status` alone. That row said `running` throughout — while
the container was mid `docker restart`, while its Caddy route had been deleted,
and while the named replacement was crash-looping — and the endpoint emitted the
`onboarding.agent_ready` funnel event off the same lie each time. The SPA polls
this every second to gate "Open chat".

L10 §5b: `POST /agent-setup/test-connection` falls through, on ANY public-
hostname failure, to `is_agent_connected(user)` and used to answer
`reachable: true, boot_progress: 100, boot_ready: true` UNCONDITIONALLY. A
tenant whose public route is broken — which is exactly what breaks chat,
because `ws_chat_proxy` dials the public hostname — reported fully ready. From
19:14:02 onward in that incident it would have. The mobile client (build 123)
reads `boot_ready` from this endpoint.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q tests/test_agent_ready_probe.py
"""

from __future__ import annotations

import itertools
import os
import uuid

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

_port = itertools.count(9500)


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def _install_health(monkeypatch, resp_or_exc):
    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k):
            if isinstance(resp_or_exc, Exception):
                raise resp_or_exc
            return resp_or_exc

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)


async def _seed(status="running", container_name=None, agent_url="https://a.test"):
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig, ManagedContainer

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                    name="T", is_active=True))
        await db.flush()
        db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                           agent_url=agent_url, agent_api_key="k"))
        if status is not None:
            db.add(ManagedContainer(
                id=str(uuid.uuid4()), user_id=uid,
                container_name=container_name or f"toup-agent-pool-{next(_port) % 90}",
                host_port=next(_port), db_name="d", status=status,
            ))
        await db.commit()
    return uid


async def _call_ready(uid):
    """Call the route function directly — the test app does not mount auth for
    a hand-made user, and what is under test is the handler's decision."""
    from app.api.agent_setup import get_agent_ready
    from app.db import async_session_maker

    class _U:
        id = uid

    async with async_session_maker() as db:
        return await get_agent_ready(current_user=_U(), db=db)


# ── /ready ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_running_row_alone_is_no_longer_ready(monkeypatch):
    """THE lie. The row said `running` for the whole incident."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, ConnectionError("no route"))

    out = await _call_ready(uid)

    assert out["stage"] == "running", "the row's own stage is unchanged"
    assert out["ready"] is False, (
        "a row is a precondition, not an answer — the agent has to say so"
    )
    assert out["probe"]["reachable"] is False


@pytest.mark.asyncio
async def test_a_404_from_the_public_hostname_is_not_ready(monkeypatch):
    """Caddy's `respond 404 "Unknown tenant"` — the 12 Sep shape exactly."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(404, {}, "Unknown tenant"))
    out = await _call_ready(uid)
    assert out["ready"] is False
    assert out["probe"]["reachable"] is False


@pytest.mark.asyncio
async def test_an_agent_that_answers_not_ready_is_not_ready(monkeypatch):
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": False, "phase": "loading_skills", "percent": 60},
    }))
    out = await _call_ready(uid)
    assert out["ready"] is False
    # The probe block is ADDITIVE (round 46 gained `serving` and
    # `not_ready_because`), so this is a subset test rather than an equality
    # one — an equality test here turns every future additive field into a
    # false failure, which is how a contract designed to grow ends up frozen.
    assert out["probe"].items() >= {
        "reachable": True, "boot_phase": "loading_skills", "boot_progress": 60,
    }.items()
    # …and the new fields say what is missing without inventing readiness: this
    # image reports no `serving`, so it stays None rather than becoming False.
    assert out["probe"]["serving"] is None
    assert out["probe"]["not_ready_because"] == "boot_loading_skills"


@pytest.mark.asyncio
async def test_a_ready_agent_behind_a_running_row_is_ready(monkeypatch):
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
    }))
    out = await _call_ready(uid)
    assert out["ready"] is True
    assert out["stage"] == "running"
    assert out["probe"]["reachable"] is True
    # Backward compatible: every pre-existing field survives.
    for k in ("ready", "stage", "duration_ms", "cold_start"):
        assert k in out


@pytest.mark.asyncio
async def test_an_old_agent_image_without_boot_progress_is_trusted(monkeypatch):
    """Refusing a 200 from an image that predates `boot_progress` would wedge
    every not-yet-rolled-out tenant at "not ready" forever."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {"status": "ok"}))
    out = await _call_ready(uid)
    assert out["ready"] is True


@pytest.mark.asyncio
async def test_a_generic_unbound_pool_container_is_not_ready_for_this_user(monkeypatch):
    """A GENERIC pool member reports boot.ready=true and is not ready for THIS
    user — its first message would 401 on a stale token."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": False,
    }))
    out = await _call_ready(uid)
    assert out["ready"] is False
    assert out["probe"]["boot_phase"] in ("ready", "not_bound")


@pytest.mark.asyncio
async def test_a_pool_container_bound_to_someone_else_is_not_ready(monkeypatch):
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True, "bound_user_id": str(uuid.uuid4()),
    }))
    out = await _call_ready(uid)
    assert out["ready"] is False


@pytest.mark.asyncio
async def test_a_non_running_stage_does_not_spend_a_probe(monkeypatch):
    """The SPA polls every second; probing a row that is provisioning buys
    nothing and costs the agent a request per poller per second."""
    uid = await _seed(status="provisioning")
    calls = {"n": 0}

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k):
            calls["n"] += 1
            return _Resp(200, {"boot_progress": {"ready": True}})

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    out = await _call_ready(uid)
    assert out["stage"] == "provisioning"
    assert out["ready"] is False
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_the_funnel_event_follows_the_probe_not_the_row(monkeypatch):
    """`onboarding.agent_ready` is emitted off `ready`, so a lying `ready`
    means a lying funnel. It fired for this user on 12 Sep."""
    from app.api import agent_setup as mod

    emitted: list = []
    monkeypatch.setattr(
        "app.services.onboarding_events.emit_agent_ready",
        lambda **kw: emitted.append(kw),
    )
    mod._AGENT_READY_EMITTED.clear()

    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(502, {}, "Bad Gateway"))
    await _call_ready(uid)
    assert emitted == [], "a 502 from the agent must not record a ready"

    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
    }))
    await _call_ready(uid)
    assert len(emitted) == 1, emitted


# ── test-connection's tunnel fallback ─────────────────────────────────

async def _call_test_connection(uid):
    from app.api.agent_setup import test_agent_connection
    from app.db import async_session_maker

    class _U:
        id = uid

    async with async_session_maker() as db:
        return await test_agent_connection(current_user=_U(), db=db)


@pytest.mark.asyncio
async def test_tunnel_only_is_reachable_but_NOT_boot_ready(monkeypatch):
    """The unconditional green. A tunnel proves the PROCESS is alive; it
    proves nothing about the public hostname, and the public hostname is what
    chat dials."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, ConnectionError("no route"))
    monkeypatch.setattr("app.api.ws_agent_tunnel.is_agent_connected",
                        lambda u: True)

    out = await _call_test_connection(uid)

    assert out["reachable"] is True
    assert out["via"] == "tunnel"
    assert out["public_ok"] is False
    assert out["boot_ready"] is False, (
        "build 123 reads boot_ready; a broken public route must not read ready"
    )
    assert out["error"]


@pytest.mark.asyncio
async def test_a_healthy_public_probe_reports_public_ok(monkeypatch):
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
    }))
    out = await _call_test_connection(uid)
    assert out["reachable"] is True
    assert out["via"] == "public"
    assert out["public_ok"] is True
    assert out["boot_ready"] is True


@pytest.mark.asyncio
async def test_neither_path_is_offline(monkeypatch):
    uid = await _seed(status="running")
    _install_health(monkeypatch, ConnectionError("nope"))
    monkeypatch.setattr("app.api.ws_agent_tunnel.is_agent_connected",
                        lambda u: False)
    out = await _call_test_connection(uid)
    assert out["reachable"] is False
    assert out["public_ok"] is False
    assert out["boot_ready"] is False


@pytest.mark.asyncio
async def test_ready_does_not_reuse_the_tunnel_fallback():
    """The honest probe must not inherit the false green (L10 §5b)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "app" / "api" / "agent_setup.py").read_text()
    body = src[src.index("async def _probe_agent_ready"):
               src.index("@router.get(\"/ready\")")]
    assert "is_agent_connected" not in body
    assert "/agent/health" in body


# ── The missing log line (L10 §5h) ────────────────────────────────────

@pytest.mark.asyncio
async def test_a_readiness_miss_records_the_status_and_the_body(monkeypatch, caplog):
    """The entire cost of the 12 Sep investigation traces to this absence:
    `_is_agent_actually_healthy` returned False on any non-200 with no log and
    no status, so the only surviving evidence that the route was 404ing was an
    unrelated soul-sync line."""
    import logging
    from app.services import prewarm_service as ps

    uid = await _seed(status="running", agent_url="https://agent-f261b564.test")
    ps._PROBE_MISS_LOGGED.clear()
    _install_health(monkeypatch, _Resp(404, {}, "Unknown tenant"))

    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        assert await ps._is_agent_actually_healthy(uid) is False

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "[PREWARM-HEALTH]" in line, line
    assert "status=404" in line, line
    assert "Unknown tenant" in line, line
    assert "agent-f261b564.test" in line, line


@pytest.mark.asyncio
async def test_a_transport_failure_records_its_exception_class(monkeypatch, caplog):
    import logging
    from app.services import prewarm_service as ps

    uid = await _seed(status="running")
    ps._PROBE_MISS_LOGGED.clear()
    _install_health(monkeypatch, ConnectionError("boom"))

    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        assert await ps._is_agent_actually_healthy(uid) is False

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "[PREWARM-HEALTH]" in line and "ConnectionError" in line, line


@pytest.mark.asyncio
async def test_the_miss_log_is_rate_limited_per_user(monkeypatch, caplog):
    """The readiness poll runs every ~2 s for 30 s. One line per user per
    window keeps the signal without 15 lines per heal attempt."""
    import logging
    from app.services import prewarm_service as ps

    uid = await _seed(status="running")
    ps._PROBE_MISS_LOGGED.clear()
    _install_health(monkeypatch, _Resp(502, {}, "bad gateway"))

    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        for _ in range(5):
            await ps._is_agent_actually_healthy(uid)

    hits = [r for r in caplog.records if "[PREWARM-HEALTH]" in r.getMessage()]
    assert len(hits) == 1, [h.getMessage() for h in hits]


def test_the_miss_log_never_carries_a_credential():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "prewarm_service.py").read_text()
    body = src[src.index("def _log_probe_miss"):src.index("async def _is_agent_actually_healthy")]
    assert "agent_api_key" not in body
    assert "X-Agent-Key" not in body
    # It logs the HOST, not the full URL (which can carry a query string).
    assert "netloc" in body


# ── test-connection EXECUTES the shared predicate ─────────────────────
#
# The incident being fixed is "two probes with two copies of the rule". Every
# case above sends a body whose verdict is identical under the old inline
# `boot.get("ready", True)` and under the shared predicate, so reintroducing
# the inline rule would leave the whole round green. These two disagree.


@pytest.mark.asyncio
async def test_test_connection_refuses_a_container_bound_to_someone_else(monkeypatch):
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True,
        "bound_user_id": "00000000-0000-4000-8000-0000000000ff",
    }))
    out = await _call_test_connection(uid)
    assert out["turn_ready"] is False, out
    assert out["boot_ready"] is False, out
    assert out["not_ready_because"] == "not_bound", out


@pytest.mark.asyncio
async def test_no_runner_refuses_the_TURN_but_not_the_BOOT(monkeypatch):
    """The split that matters for compatibility. `turn_ready` is the strict
    predicate the round-46 onboarding gates on. `boot_ready` keeps the meaning
    the DEPLOYED web client gives it (frontend/src/App.tsx keys the tunnel
    indicator and the "waking up" banner on it): under the strict rule a
    healthy, answering agent reads as disconnected on every 5 s `deps.db`
    blip.

    `runner` is the term used here rather than `llm_wire_warm`: D1 demoted the
    warm (and `channels_settled`) to diagnostics, because each has a single
    producer that can fail silently and would then refuse forever."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True,
        "turn_ready_detail": {
            "runner": False, "db": True, "bound_user": True,
            "llm_wire_warm": True, "channels_settled": True,
        },
    }))
    out = await _call_test_connection(uid)
    assert out["turn_ready"] is False, out
    assert out["not_ready_because"] == "runner", out
    assert out["boot_ready"] is True, out


@pytest.mark.asyncio
async def test_a_missing_boot_ready_field_is_no_longer_read_as_ready(monkeypatch):
    """The old inline rule was `boot.get("ready", True)` — a container whose
    boot_progress exists but says nothing reported ready."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"phase": "embeddings", "percent": 25},
    }))
    out = await _call_test_connection(uid)
    assert out["boot_ready"] is False, out
    assert out["turn_ready"] is False, out


# ── the two probes execute ONE predicate ──────────────────────────────
#
# The incident being fixed is "two probes with two copies of the rule". Every
# case above sends a body whose verdict is identical under the old inline
# `boot.get("ready", True)` and under `agent_turn_readiness`, so reintroducing
# the old rule would leave them all green. These two disagree.


@pytest.mark.asyncio
async def test_a_container_bound_to_SOMEONE_ELSE_is_not_ready(monkeypatch):
    """A generic pool member is booted, healthy — and not this user's. The old
    rule read `boot_progress` alone and answered ready."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True,
        "bound_user_id": "00000000-0000-4000-8000-0000000000ff",
    }))
    out = await _call_test_connection(uid)
    assert out["reachable"] is True
    assert out["turn_ready"] is False
    assert out["not_ready_because"] == "not_bound"
    assert out["boot_ready"] is False, (
        "not_bound is the one strict reason that must also fail boot_ready — "
        "build 123 and the web client key their waking UI on it"
    )


@pytest.mark.asyncio
async def test_a_booted_container_whose_DB_is_down_cannot_serve_a_turn(monkeypatch):
    """The other direction, and the one that proves `boot_ready` kept its own
    meaning: the container finished booting (so `boot_ready` is true and the
    deployed web client does not flip to "waking up"), and it still cannot
    answer, so the round-46 clients read `turn_ready`."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True,
        "bound_user_id": None,
        "serving": False,
        "turn_ready_detail": {
            "runner": True, "db": False, "bound_user": True,
            "llm_wire_warm": True, "channels_settled": True,
        },
    }))
    out = await _call_test_connection(uid)
    assert out["boot_ready"] is True, "boot_ready must keep its BOOT meaning"
    assert out["turn_ready"] is False
    assert out["not_ready_because"] == "db"
    assert out["serving"] is False


@pytest.mark.asyncio
async def test_a_cold_wire_or_a_restarting_sidecar_never_refuses(monkeypatch):
    """D1. Both have a single producer that can fail silently; neither may
    hold onboarding at "warming up" for a container that answers turns."""
    uid = await _seed(status="running")
    _install_health(monkeypatch, _Resp(200, {
        "boot_progress": {"ready": True, "phase": "ready", "percent": 100},
        "is_bound": True,
        "bound_user_id": None,
        "turn_ready_detail": {
            "runner": True, "db": True, "bound_user": True,
            "llm_wire_warm": False, "channels_settled": False,
        },
    }))
    out = await _call_test_connection(uid)
    assert out["turn_ready"] is True, out
    assert out["not_ready_because"] is None
    assert out["boot_ready"] is True
