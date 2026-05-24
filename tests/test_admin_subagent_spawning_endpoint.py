"""Admin endpoint that flips ``agent_configs.subagent_spawning_enabled``
for a tenant and triggers an immediate container restart so the flag
takes effect now (not at the next rollout).

Three production behaviours pinned:

  1. Happy path — flag flips, restart_container is called with the
     same user_id, response carries ``container_restart="ok"``.
  2. Unknown user → 404 (defense-in-depth; bare UPDATE silently
     succeeds with 0 rows otherwise).
  3. Restart bridge unreachable → DB change still committed, response
     carries ``container_restart="failed:..."`` so operators can
     spot the gap. The flag IS the source of truth; restart is an
     eager-apply optimization.
"""
from __future__ import annotations

import uuid

import pytest

from app.api.admin.users import (
    SubagentSpawningRequest,
    set_subagent_spawning,
)


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeDB:
    """Stand-in AsyncSession. Returns user/agent_config from a dict
    keyed on the SQL fragment so tests can stub each branch
    independently."""

    def __init__(self, *, user=None, agent_config=None):
        self._user = user
        self._ac = agent_config
        self.committed = False

    async def execute(self, stmt):
        # Heuristic: the User select is checked first in the
        # endpoint, then the AgentConfig select. Return them in order.
        # Simple stateful counter — flips after first .execute call.
        if not getattr(self, "_user_returned", False):
            self._user_returned = True
            return _FakeResult(self._user)
        return _FakeResult(self._ac)

    async def commit(self):
        self.committed = True


class _FakeAgentConfig:
    def __init__(self, enabled=False):
        self.subagent_spawning_enabled = enabled
        self.updated_at = None


class _FakeUser:
    def __init__(self, uid="u-1"):
        self.id = uid


# ─── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_flips_flag_and_restarts(monkeypatch):
    """Admin POSTs enabled=true → AgentConfig flips, restart_container
    is called with the same user_id, response reports ok."""
    uid = str(uuid.uuid4())
    ac = _FakeAgentConfig(enabled=False)
    db = _FakeDB(user=_FakeUser(uid), agent_config=ac)

    restart_calls: list[str] = []

    async def _fake_restart(db_, user_id):
        restart_calls.append(user_id)

        class _C:
            status = "running"

        return _C()

    import app.services.docker_host_service as dhs
    monkeypatch.setattr(dhs, "restart_container", _fake_restart)

    resp = await set_subagent_spawning(
        user_id=uid,
        body=SubagentSpawningRequest(enabled=True),
        admin=_FakeUser("admin-1"),
        db=db,
    )

    assert ac.subagent_spawning_enabled is True
    assert db.committed is True
    assert restart_calls == [uid]
    assert resp.enabled is True
    assert resp.container_restart == "ok"


@pytest.mark.asyncio
async def test_unknown_user_returns_404(monkeypatch):
    """Defense-in-depth — a bare UPDATE on a missing user_id silently
    affects 0 rows. The endpoint MUST 404 instead so the admin UI
    surfaces a clear error."""
    from fastapi import HTTPException

    db = _FakeDB(user=None, agent_config=None)
    with pytest.raises(HTTPException) as ei:
        await set_subagent_spawning(
            user_id="00000000-0000-0000-0000-000000000000",
            body=SubagentSpawningRequest(enabled=True),
            admin=_FakeUser(),
            db=db,
        )
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_missing_agent_config_returns_404(monkeypatch):
    """User exists but has no AgentConfig row (never provisioned) →
    404 with a specific message so the operator knows it's not a
    user-id typo but a provisioning state."""
    from fastapi import HTTPException

    db = _FakeDB(user=_FakeUser(), agent_config=None)
    with pytest.raises(HTTPException) as ei:
        await set_subagent_spawning(
            user_id="aabbccdd-1111-2222-3333-444444444444",
            body=SubagentSpawningRequest(enabled=True),
            admin=_FakeUser(),
            db=db,
        )
    assert ei.value.status_code == 404
    assert "AgentConfig" in str(ei.value.detail)


@pytest.mark.asyncio
async def test_restart_failure_still_commits_db(monkeypatch):
    """Bridge unreachable during restart_container — DB change MUST
    still be committed (the column is the source of truth; restart
    is an eager-apply optimization). Response surfaces the failure
    so the operator can manually restart later."""
    uid = str(uuid.uuid4())
    ac = _FakeAgentConfig(enabled=False)
    db = _FakeDB(user=_FakeUser(uid), agent_config=ac)

    async def _exploding_restart(db_, user_id):
        raise RuntimeError("bridge unreachable")

    import app.services.docker_host_service as dhs
    monkeypatch.setattr(dhs, "restart_container", _exploding_restart)

    resp = await set_subagent_spawning(
        user_id=uid,
        body=SubagentSpawningRequest(enabled=True),
        admin=_FakeUser("admin-1"),
        db=db,
    )

    # DB STILL committed despite restart blowing up.
    assert ac.subagent_spawning_enabled is True
    assert db.committed is True
    assert resp.enabled is True
    # Operator-visible failure marker.
    assert resp.container_restart.startswith("failed:")
    assert "RuntimeError" in resp.container_restart


@pytest.mark.asyncio
async def test_no_container_returns_skipped(monkeypatch):
    """User has AgentConfig but no managed_containers row (self-
    hosted, or container destroyed). DB updates; response reports
    ``skipped_no_container`` so the operator knows the flip happened
    but had no live tenant to apply it to."""
    uid = str(uuid.uuid4())
    ac = _FakeAgentConfig(enabled=True)
    db = _FakeDB(user=_FakeUser(uid), agent_config=ac)

    async def _fake_restart(db_, user_id):
        return None  # docker_host_service returns None when no container

    import app.services.docker_host_service as dhs
    monkeypatch.setattr(dhs, "restart_container", _fake_restart)

    resp = await set_subagent_spawning(
        user_id=uid,
        body=SubagentSpawningRequest(enabled=False),
        admin=_FakeUser(),
        db=db,
    )
    assert ac.subagent_spawning_enabled is False
    assert resp.container_restart == "skipped_no_container"


@pytest.mark.asyncio
async def test_can_disable_a_previously_enabled_tenant(monkeypatch):
    """Symmetry check — the endpoint flips both directions."""
    uid = str(uuid.uuid4())
    ac = _FakeAgentConfig(enabled=True)  # was enabled
    db = _FakeDB(user=_FakeUser(uid), agent_config=ac)

    async def _fake_restart(db_, user_id):
        class _C:
            status = "running"
        return _C()

    import app.services.docker_host_service as dhs
    monkeypatch.setattr(dhs, "restart_container", _fake_restart)

    resp = await set_subagent_spawning(
        user_id=uid,
        body=SubagentSpawningRequest(enabled=False),
        admin=_FakeUser(),
        db=db,
    )
    assert ac.subagent_spawning_enabled is False
    assert resp.enabled is False
    assert resp.container_restart == "ok"
