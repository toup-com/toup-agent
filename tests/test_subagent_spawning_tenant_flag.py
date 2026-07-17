"""Per-tenant sub-agent spawning kill-switch (mig 057).

Pins three surfaces:

  1. The ``agent_configs.subagent_spawning_enabled`` column exists on
     the ORM with the right type, nullability, and default — operators
     who SELECT or UPDATE this column from psql rely on the schema.
  2. ``_agent_config_to_bridge_body`` forwards the field to the bridge
     dict ONLY when True (omitted when False so existing tenants don't
     get a redundant ``SUBAGENT_SPAWNING_ENABLED=false`` line in their
     .env). The bridge silently drops unknown fields until the VPS
     contract is updated, but this forward-compat wiring keeps the
     contract single-source-of-truth.
  3. ``_read_subagent_flag_for_user`` (the runtime path that actually
     unlocks per-tenant spawning today) returns the column value and
     fails closed on any error.
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


# ──────────────────────────────────────────────────────────────────────
# 1. ORM column declaration
# ──────────────────────────────────────────────────────────────────────


def test_orm_declares_subagent_spawning_enabled_column():
    """The column must be on AgentConfig with the right shape so
    operators can SELECT/UPDATE it from psql without surprise."""
    from app.db.models.agent import AgentConfig
    col = AgentConfig.__table__.columns.get("subagent_spawning_enabled")
    assert col is not None, (
        "AgentConfig missing subagent_spawning_enabled column — "
        "migration 057 expects the ORM to declare it"
    )
    # NOT NULL with default false — matches the migration's server_default.
    assert col.nullable is False, (
        "subagent_spawning_enabled must be NOT NULL so a NULL row can't "
        "silently disable the kill-switch for tenants who skipped the "
        "backfill"
    )
    # Pin the boolean type — a stray Integer would land as a different
    # column shape on Postgres and surprise the runtime read.
    assert isinstance(col.type, sa.Boolean)


# ──────────────────────────────────────────────────────────────────────
# 2. Bridge whitelist forwarding behaviour
# ──────────────────────────────────────────────────────────────────────


def test_bridge_body_omits_flag_when_false():
    """When the column is False (the default), the bridge body must NOT
    carry the field. We don't want every tenant's .env to grow a
    redundant ``SUBAGENT_SPAWNING_ENABLED=false`` line just because
    the column was added."""
    from app.db.models.agent import AgentConfig
    from app.services.docker_host_service import _agent_config_to_bridge_body

    cfg = AgentConfig(
        user_id="11111111-1111-1111-1111-111111111111",
        subagent_spawning_enabled=False,
    )
    body = _agent_config_to_bridge_body(cfg)
    assert "subagent_spawning_enabled" not in body, (
        "subagent_spawning_enabled=False must be omitted from bridge body"
    )


def test_bridge_body_includes_flag_when_true():
    """When ops flips the column to True (the rollout step), the bridge
    body must carry ``subagent_spawning_enabled: True`` so the .env
    write path sees it."""
    from app.db.models.agent import AgentConfig
    from app.services.docker_host_service import _agent_config_to_bridge_body

    cfg = AgentConfig(
        user_id="22222222-2222-2222-2222-222222222222",
        subagent_spawning_enabled=True,
    )
    body = _agent_config_to_bridge_body(cfg)
    assert body.get("subagent_spawning_enabled") is True, (
        f"subagent_spawning_enabled=True must be forwarded; got {body!r}"
    )


def test_bridge_body_handles_missing_attribute_gracefully():
    """An older AgentConfig instance (or a mock that doesn't have the
    field) MUST NOT raise — getattr defaults to False and the field
    is omitted. Protects against rolling updates where the platform
    code is one revision behind the DB."""
    from app.services.docker_host_service import _agent_config_to_bridge_body

    class _Stub:
        # Only the absolute minimum the function needs; deliberately
        # missing subagent_spawning_enabled.
        user_id = "33333333-3333-3333-3333-333333333333"

    body = _agent_config_to_bridge_body(_Stub())
    assert "subagent_spawning_enabled" not in body


# ──────────────────────────────────────────────────────────────────────
# 3. AgentEnvContract field declaration
# ──────────────────────────────────────────────────────────────────────


def test_agent_env_contract_declares_subagent_spawning_enabled():
    """Forward-compat: the field is in the contract so when the bridge
    VPS code is updated to know it, the .env path lights up. Until
    then the contract field default (False) makes the absent .env
    line a no-op."""
    from app.types.agent_env import AgentEnvContract
    fields = AgentEnvContract.model_fields
    assert "subagent_spawning_enabled" in fields
    assert fields["subagent_spawning_enabled"].default is False


def test_agent_env_contract_emits_uppercase_env_key_when_true():
    """``to_env_lines`` must emit ``SUBAGENT_SPAWNING_ENABLED=true``
    (lowercase boolean rendering matches pydantic-settings parsing)
    when the field is True. Pins the serialization shape so a future
    refactor that drops the field silently is caught."""
    from app.types.agent_env import AgentEnvContract

    contract = AgentEnvContract(
        user_id="44444444-4444-4444-4444-444444444444",
        agent_api_key="x" * 40,
        database_url=(
            "postgresql+asyncpg://toup_agent_44444444:pw@"
            "host.docker.internal:6432/toup_agent_44444444"
        ),
        platform_api_url="https://toup.ai/api",
        subagent_spawning_enabled=True,
    )
    lines = contract.to_env_lines()
    assert "SUBAGENT_SPAWNING_ENABLED=true" in lines


# ──────────────────────────────────────────────────────────────────────
# 4. Runtime HTTP-callback path (the actual unlock on partitioned agents)
# ──────────────────────────────────────────────────────────────────────
#
# The tenant agent process has a partitioned DB — agent_configs lives
# ONLY on the platform DB. So the helper queries the platform via the
# /agent/runtime-flags HTTP endpoint, auth'd with X-Agent-Key +
# X-Agent-User-Id (the same multi-tenant contract used by
# streaming.py, credits.py, etc).


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Override conftest autouse — this test doesn't need an engine."""
    yield


@pytest.fixture
def fake_settings(monkeypatch):
    """Wire ``settings.platform_api_url`` + ``settings.agent_api_key``
    so the helper has a callback target to reach."""
    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "https://toup.ai/api", raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "toup_ak_testkey_xxxxxxxx", raising=False)
    return settings


def _patch_httpx(monkeypatch, response_status: int, response_body: dict | None = None,
                 raise_exc: Exception | None = None,
                 capture: dict | None = None):
    """Patch ``httpx.AsyncClient`` so the helper hits a fake response.

    capture: optional dict that records the outgoing (url, headers)
    so tests can assert the auth surface is correct."""
    import httpx

    class _FakeResp:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body or {}
            # The helper rejects non-JSON responses (SPA index.html
            # fallback guard, 2026-05-24) — fake the honest header.
            self.headers = {"content-type": "application/json"}
        def json(self):
            return self._body

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None, **_):
            if capture is not None:
                capture["url"] = url
                capture["headers"] = dict(headers or {})
            if raise_exc is not None:
                raise raise_exc
            return _FakeResp(response_status, response_body)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_true_when_platform_says_true(
    fake_settings, monkeypatch,
):
    """The unlock path: platform returns ``subagent_spawning_enabled=true``
    → helper returns True. Pin the JSON-decode shape so a server-side
    rename surfaces here."""
    from app.agent.tool_executor import _read_subagent_flag_for_user
    capture: dict = {}
    _patch_httpx(monkeypatch, 200, {"subagent_spawning_enabled": True}, capture=capture)

    uid = str(uuid.uuid4())
    assert await _read_subagent_flag_for_user(uid) is True
    # Verify the auth surface — wrong URL or missing header is a
    # silent production-grade bug otherwise.
    assert capture["url"] == "https://toup.ai/api/agent/runtime-flags"
    assert capture["headers"].get("X-Agent-Key") == "toup_ak_testkey_xxxxxxxx"
    assert capture["headers"].get("X-Agent-User-Id") == uid


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_when_platform_says_false(
    fake_settings, monkeypatch,
):
    """Default-False tenants stay on the legacy path."""
    from app.agent.tool_executor import _read_subagent_flag_for_user
    _patch_httpx(monkeypatch, 200, {"subagent_spawning_enabled": False})
    assert await _read_subagent_flag_for_user(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_on_403(fake_settings, monkeypatch):
    """403 (key mismatch / unknown user) MUST fail closed — bug-class
    where a key gets rotated and the agent process still has the old
    one. We don't want a rotation to silently break the kill-switch's
    safety semantics in either direction."""
    from app.agent.tool_executor import _read_subagent_flag_for_user
    _patch_httpx(monkeypatch, 403)
    assert await _read_subagent_flag_for_user(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_on_500(fake_settings, monkeypatch):
    """Platform-side 5xx → fail closed."""
    from app.agent.tool_executor import _read_subagent_flag_for_user
    _patch_httpx(monkeypatch, 500)
    assert await _read_subagent_flag_for_user(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_on_network_error(
    fake_settings, monkeypatch,
):
    """Platform unreachable (timeout, ConnectError) → fail closed."""
    import httpx
    from app.agent.tool_executor import _read_subagent_flag_for_user
    _patch_httpx(monkeypatch, 0, raise_exc=httpx.ConnectError("boom"))
    assert await _read_subagent_flag_for_user(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_for_empty_user_id(fake_settings):
    """Empty/None user_id short-circuits without an HTTP hit. Belt
    and suspenders — _tool_spawn rejects empty user_id earlier, but
    the helper should still be safe to call from anywhere."""
    from app.agent.tool_executor import _read_subagent_flag_for_user
    assert await _read_subagent_flag_for_user("") is False
    assert await _read_subagent_flag_for_user(None) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_when_no_platform_url(monkeypatch):
    """Self-hosted tenant with no platform_api_url → no callback target
    → fail closed (env-var path is the only enable lever)."""
    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "", raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "toup_ak_x", raising=False)
    from app.agent.tool_executor import _read_subagent_flag_for_user
    assert await _read_subagent_flag_for_user(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_when_no_agent_key(monkeypatch):
    """Misconfigured tenant with no agent_api_key → fail closed."""
    from app.config import settings
    monkeypatch.setattr(settings, "platform_api_url", "https://toup.ai/api", raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "", raising=False)
    from app.agent.tool_executor import _read_subagent_flag_for_user
    assert await _read_subagent_flag_for_user(str(uuid.uuid4())) is False


# ──────────────────────────────────────────────────────────────────────
# 5. /api/agent/runtime-flags endpoint behaviour
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_endpoint_requires_x_agent_key_and_user_id_headers():
    """Missing either header → 401. Pin the contract so a future
    refactor that loosens auth lights up here."""
    from fastapi import HTTPException
    from app.api.agent import get_runtime_flags

    # Missing both
    with pytest.raises(HTTPException) as ei:
        await get_runtime_flags(x_agent_key=None, x_agent_user_id=None, db=None)
    assert ei.value.status_code == 401

    # Missing user_id only
    with pytest.raises(HTTPException) as ei:
        await get_runtime_flags(x_agent_key="x", x_agent_user_id=None, db=None)
    assert ei.value.status_code == 401

    # Missing key only
    with pytest.raises(HTTPException) as ei:
        await get_runtime_flags(x_agent_key=None, x_agent_user_id="x", db=None)
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_endpoint_returns_403_when_key_does_not_match():
    """Wrong (user, key) pair → 403, NOT 404 (so a leaked key can't
    enumerate user_ids via probe-with-known-user, probe-with-unknown-
    user). The platform DB lookup uses ``AND user_id=? AND
    agent_api_key=?``, so any mismatch lands here."""
    from fastapi import HTTPException
    from app.api.agent import get_runtime_flags

    class _FakeDB:
        async def execute(self, _stmt):
            class _Res:
                def scalar_one_or_none(self): return None
            return _Res()

    with pytest.raises(HTTPException) as ei:
        await get_runtime_flags(
            x_agent_key="wrong", x_agent_user_id="11111111-1111-1111-1111-111111111111",
            db=_FakeDB(),
        )
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_endpoint_returns_flag_when_key_matches():
    """Happy path: matching (user, key) pair returns the row's flag
    value. ``subagent_spawning_enabled`` defaults False but operators
    flip it to True for tenants that should use the new path."""
    from app.api.agent import get_runtime_flags

    class _FakeCfg:
        subagent_spawning_enabled = True

    class _FakeDB:
        async def execute(self, _stmt):
            class _Res:
                def scalar_one_or_none(self): return _FakeCfg()
            return _Res()

    resp = await get_runtime_flags(
        x_agent_key="right-key",
        x_agent_user_id="22222222-2222-2222-2222-222222222222",
        db=_FakeDB(),
    )
    assert resp.subagent_spawning_enabled is True


@pytest.mark.asyncio
async def test_endpoint_returns_false_when_column_value_is_false():
    """Default-False rows return False — the legacy path stays
    selected for tenants not yet rolled out."""
    from app.api.agent import get_runtime_flags

    class _FakeCfg:
        subagent_spawning_enabled = False

    class _FakeDB:
        async def execute(self, _stmt):
            class _Res:
                def scalar_one_or_none(self): return _FakeCfg()
            return _Res()

    resp = await get_runtime_flags(
        x_agent_key="right-key",
        x_agent_user_id="33333333-3333-3333-3333-333333333333",
        db=_FakeDB(),
    )
    assert resp.subagent_spawning_enabled is False


# ──────────────────────────────────────────────────────────────────────
# 5. Disabled-path honesty (2026-07-16 founder repro)
# ──────────────────────────────────────────────────────────────────────
#
# With spawning disabled and no live Telegram chat, the old code
# returned SUBAGENT_LEGACY_TELEGRAM_ONLY — which reads as a channel
# problem when the actual blocker is the kill switch. The honest
# error is SUBAGENT_DISABLED with the operator hint.


@pytest.mark.asyncio
async def test_spawn_disabled_no_chat_reports_disabled(monkeypatch, tmp_path):
    import app.agent.tool_executor as te_mod
    from app.agent.tool_executor import ToolExecutor
    from app.config import settings

    monkeypatch.setattr(settings, "subagent_spawning_enabled", False)

    async def _flag_false(user_id):
        return False

    monkeypatch.setattr(te_mod, "_read_subagent_flag_for_user", _flag_false)

    te = ToolExecutor(workspace=str(tmp_path), subagent_manager=object())
    te.set_user_id("11111111-1111-1111-1111-111111111111")
    # No Telegram chat in context (ContextVar default) — the exact
    # shape of a web/mobile chat turn or an autopilot tick.
    out = await te._tool_spawn({"task": "research something"})
    assert '"SUBAGENT_DISABLED"' in out
    assert "SUBAGENT_LEGACY_TELEGRAM_ONLY" not in out
