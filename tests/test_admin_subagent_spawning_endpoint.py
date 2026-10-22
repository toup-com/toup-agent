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


# ─── Supervisor / executor settings (voice round 2026-10-05, D-P6) ────
#
# PUT/GET /api/admin/users/{id}/subagent-settings store the tenant's
# overrides in platform_settings `subagent.settings.<user_id>` — NO container
# restart (the tenant reads them through runtime-flags with a 60 s cache), and
# runtime-flags surfaces them as additive fields.


async def _real_user() -> str:
    from app.db import User, async_session_maker
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="x" * 60,
                    name="T"))
        await db.commit()
    return uid


@pytest.mark.asyncio
async def test_subagent_settings_round_trip_without_a_restart(monkeypatch):
    from app.api.admin.users import (
        SubagentSettingsBody, get_subagent_settings, put_subagent_settings,
    )
    from app.db import async_session_maker
    import app.services.docker_host_service as dhs

    async def _no_restart(*a, **k):
        raise AssertionError("subagent-settings must never restart the tenant")

    monkeypatch.setattr(dhs, "restart_container", _no_restart)
    uid = await _real_user()
    async with async_session_maker() as db:
        put = await put_subagent_settings(
            user_id=uid,
            body=SubagentSettingsBody(enabled=True, executor_model="gpt-5.6-luna",
                                      max_workers=2),
            admin=_FakeUser("admin-1"), db=db,
        )
    assert (put.enabled, put.executor_model, put.max_workers) == (True, "gpt-5.6-luna", 2)
    async with async_session_maker() as db:
        got = await get_subagent_settings(user_id=uid, admin=_FakeUser(), db=db)
    assert got.model_dump(exclude_none=True) == {
        "user_id": uid, "enabled": True, "executor_model": "gpt-5.6-luna",
        "max_workers": 2,
    }
    # PUT replaces: an omitted field returns to the image default.
    async with async_session_maker() as db:
        await put_subagent_settings(user_id=uid, body=SubagentSettingsBody(enabled=False),
                                    admin=_FakeUser("admin-1"), db=db)
        got = await get_subagent_settings(user_id=uid, admin=_FakeUser(), db=db)
    assert got.model_dump(exclude_none=True) == {"user_id": uid, "enabled": False}


@pytest.mark.asyncio
async def test_subagent_settings_unknown_user_is_404():
    from fastapi import HTTPException
    from app.api.admin.users import SubagentSettingsBody, put_subagent_settings
    from app.db import async_session_maker
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as ei:
            await put_subagent_settings(user_id=str(uuid.uuid4()),
                                        body=SubagentSettingsBody(enabled=True),
                                        admin=_FakeUser(), db=db)
    assert ei.value.status_code == 404


def test_subagent_settings_normalize_keeps_only_valid_known_keys():
    from app.services.subagent_settings import normalize
    assert normalize({"enabled": "yes", "max_workers": 9, "executor_model": "  ",
                      "worker_timeout_seconds": 120, "bogus": 1,
                      "max_per_user_concurrent": True}) == {"worker_timeout_seconds": 120}
    assert normalize("not a dict") == {}


@pytest.mark.asyncio
async def test_runtime_flags_surface_the_overrides_as_additive_fields():
    import json as _json
    from app.api.agent import get_runtime_flags

    class _Cfg:
        subagent_spawning_enabled = True

    class _Row:
        value = _json.dumps({"enabled": True, "supervisor_model": "gpt-6-sol",
                             "max_children_per_parent": 4})

    class _DB:
        async def execute(self, _stmt):
            class _Res:
                def scalar_one_or_none(self): return _Cfg()
            return _Res()

        async def get(self, _model, key):
            assert key == "subagent.settings.44444444-4444-4444-4444-444444444444"
            return _Row()

    resp = await get_runtime_flags(
        x_agent_key="k", x_agent_user_id="44444444-4444-4444-4444-444444444444",
        db=_DB(),
    )
    assert resp.subagent_spawning_enabled is True
    assert resp.supervisor_enabled is True
    assert resp.supervisor_model == "gpt-6-sol"
    assert resp.max_children_per_parent == 4
    assert resp.executor_model is None and resp.supervisor_max_workers is None
